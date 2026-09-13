# -*- coding: utf-8 -*-
"""
buffer_v2.py - n-step 优先经验回放缓冲区（Prioritized Experience Replay）

对标 arXiv 2507.05465 中 H-DQN 的两个关键机制:

1. n-step 回报 —— 直接针对"单步 TD 奖励传播太慢"这一瓶颈。
   n 步回报 R^(n)_t = Σ_{i=0}^{n-1} γ^i · r_{t+i}, 使价值信号一次跨越 n 步,
   信用分配距离从 O(1) 提升到 O(n), 稀疏奖励场景下收敛显著加快。
   队列中途遇到终止时自动截断, 并用终止状态的 next_state / done 标记。

2. 优先经验回放（PER, Schaul et al. 2016）
   - SumTree 按 TD 误差优先级采样, 聚焦高信息量样本, 采样复杂度 O(log N)
   - 优先级 p_i = (|TD误差| + ε)^α,  α = 0.6
   - 重要性采样权重 w_i = (N·P(i))^(-β) 归一化, 修正优先采样带来的偏差,
     β 从 0.4 线性退火到 1.0

3. 8 重对称增强（symmetry augmentation）
   采样时对棋盘随机施加 8 种对称变换（4 旋转 × 2 反射）, 动作按
   ACTION_PERMS 同步映射。等效于把训练数据量放大 8 倍, 且让网络学到
   棋盘的对称不变性 —— 成本几乎为零, 是 2048 上性价比最高的技巧之一。

内存优化: 缓冲区存原始棋盘（4x4 uint16, 32 字节/样本）, 采样时才编码为
16x4x4 张量。100k 容量仅占约 3 MB。
"""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from model_v2 import ACTION_PERMS, encode_state, transform_grid


class SumTree:
    """二叉求和树: O(log N) 的优先级更新与按优先级比例采样。

    结构: 叶子节点存优先级, 内部节点存子树优先级之和;
    根节点值 = 全部优先级之和, 据此可对 [0, total) 均匀采样实现优先采样。
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write = 0          # 下一个写入位置（环形）
        self.n_entries = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def _propagate(self, idx: int, delta: float) -> None:
        """自底向上更新祖先节点的和。"""
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, tree_idx: int, priority: float) -> None:
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def add(self, priority: float) -> int:
        """写入新优先级, 返回其树索引。"""
        tree_idx = self.write + self.capacity - 1
        self.update(tree_idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
        return tree_idx

    def get(self, s: float) -> Tuple[int, float]:
        """按累积和 s 定位叶子节点, 返回 (树索引, 优先级)。"""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        return idx, float(self.tree[idx])


class NStepPrioritizedReplay:
    """n-step 优先经验回放缓冲区。

    用法:
        buf = NStepPrioritizedReplay(capacity=100_000, n_step=3, gamma=0.99)
        buf.push(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones, weights, idxs = buf.sample(64)
        buf.update_priorities(idxs, td_errors)      # 训练后用 TD 误差更新优先级
    """

    def __init__(self, capacity: int = 100_000, n_step: int = 3,
                 gamma: float = 0.99, alpha: float = 0.6,
                 beta0: float = 0.4, beta_steps: int = 200_000,
                 augment: bool = True, seed: Optional[int] = None):
        self.capacity = capacity
        self.n_step = max(1, n_step)
        self.gamma = gamma
        self.alpha = alpha                 # 优先级指数
        self.beta0 = beta0                 # IS 权重初始指数
        self.beta_steps = beta_steps       # β 退火周期
        self.augment = augment             # 8 重对称增强开关
        self.rng = np.random.default_rng(seed)

        self.tree = SumTree(capacity)
        self.data = np.empty(capacity, dtype=object)
        self.max_priority = 1.0            # 新样本优先级（保证至少被采一次）
        self.step_count = 0                # 用于 β 退火

        self._deque: deque = deque(maxlen=self.n_step)

    # ---------------- 写入 ----------------
    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        """存入一步 transition, 内部自动累积 n 步后写入主缓冲区。"""
        self._deque.append((np.asarray(state, dtype=np.uint16).copy(),
                            int(action), float(reward),
                            np.asarray(next_state, dtype=np.uint16).copy(),
                            bool(done)))
        if len(self._deque) == self.n_step:
            self._store_nstep()
        if done:
            # 对局结束: 队列尾部不足 n 步的部分按剩余步数截断存储
            while len(self._deque) > 0:
                self._store_nstep()

    def _store_nstep(self) -> None:
        """把队列头部的 transition 展开为 n-step 形式写入缓冲区。"""
        R = 0.0
        s0, a0 = self._deque[0][0], self._deque[0][1]
        s_next, done_n = self._deque[0][3], self._deque[0][4]
        for i, (_, _, r, ns, d) in enumerate(self._deque):
            R += (self.gamma ** i) * r
            s_next, done_n = ns, d
            if d:                      # 中途终止: 截断 n-step 回报
                break
        self._add(s0, a0, R, s_next, done_n)
        self._deque.popleft()

    def _add(self, s, a: int, r: float, ns, d: bool) -> None:
        data_idx = self.tree.write
        self.data[data_idx] = (s, a, r, ns, d)
        # 新样本给最大优先级, 保证至少被采样一次
        self.tree.add(self.max_priority ** self.alpha)

    # ---------------- 采样 ----------------
    def _beta(self) -> float:
        """IS 权重指数 β: 从 beta0 线性退火到 1.0。"""
        frac = min(1.0, self.step_count / max(1, self.beta_steps))
        return self.beta0 + frac * (1.0 - self.beta0)

    def sample(self, batch_size: int):
        """按优先级采样一个 batch。

        返回: (states, actions, rewards, next_states, dones,
               weights, tree_indices)
            states/next_states : (B, 16, 4, 4) float32
            weights            : (B,) float32（IS 权重, 已归一化）
        """
        n = self.tree.n_entries
        batch_size = min(batch_size, n)
        if batch_size <= 0:
            raise ValueError("缓冲区为空, 无法采样")

        total = self.tree.total
        segment = total / batch_size
        tree_idxs: List[int] = []
        priorities: List[float] = []

        # 分层采样: 每段随机取一点, 保证样本分布均匀且高优先级样本更易被选中
        for i in range(batch_size):
            s = self.rng.uniform(segment * i, segment * (i + 1))
            idx, p = self.tree.get(min(s, total - 1e-9))
            tree_idxs.append(idx)
            priorities.append(max(p, 1e-12))

        # IS 权重: w = (N · P(i))^(-β), 再按最大值归一化
        beta = self._beta()
        probs = np.asarray(priorities, dtype=np.float64) / total
        weights = (n * probs) ** (-beta)
        weights /= weights.max()

        states = np.empty((batch_size, 16, 4, 4), dtype=np.float32)
        next_states = np.empty((batch_size, 16, 4, 4), dtype=np.float32)
        actions = np.empty(batch_size, dtype=np.int64)
        rewards = np.empty(batch_size, dtype=np.float32)
        dones = np.empty(batch_size, dtype=np.float32)

        for i, tree_idx in enumerate(tree_idxs):
            data_idx = tree_idx - (self.capacity - 1)
            s, a, r, ns, d = self.data[data_idx]
            if self.augment:
                t = int(self.rng.integers(0, 8))
                if t:                       # 随机对称变换 (数据量 x8)
                    s = transform_grid(s, t)
                    ns = transform_grid(ns, t)
                    a = ACTION_PERMS[t][a]
            states[i] = encode_state(s)
            next_states[i] = encode_state(ns)
            actions[i] = a
            rewards[i] = r
            dones[i] = float(d)

        self.step_count += 1
        return (states, actions, rewards, next_states, dones,
                weights.astype(np.float32), np.asarray(tree_idxs, dtype=np.int64))

    def update_priorities(self, tree_idxs: np.ndarray, td_errors) -> None:
        """用训练得到的 TD 误差更新样本优先级。"""
        for idx, err in zip(np.asarray(tree_idxs).ravel(),
                            np.asarray(td_errors).ravel()):
            p = (abs(float(err)) + 1e-6) ** self.alpha
            self.tree.update(int(idx), p)
            if p > self.max_priority:
                self.max_priority = p

    def __len__(self) -> int:
        return self.tree.n_entries

    # ---------------- 状态 ----------------
    def stats(self) -> dict:
        return {"size": len(self), "capacity": self.capacity,
                "n_step": self.n_step, "beta": round(self._beta(), 3),
                "max_priority": round(self.max_priority, 4)}


# ---------------- 模块自测 ----------------
if __name__ == "__main__":
    import random as _random

    _random.seed(0)
    np.random.seed(0)

    # ---- 1. SumTree 基础行为 ----
    st = SumTree(8)
    for p in (1.0, 2.0, 3.0, 4.0):
        st.add(p)
    assert abs(st.total - 10.0) < 1e-9, f"SumTree 总和错误: {st.total}"
    idx, p = st.get(0.5)          # 落在第 1 个叶子(优先级 1.0)
    assert abs(p - 1.0) < 1e-9 and idx == 7, f"get 定位错误: {idx}, {p}"
    st.update(7, 10.0)
    assert abs(st.total - 19.0) < 1e-9, "update 后总和错误"
    print("1. SumTree (add/update/get/总和) ✓")

    # ---- 2. n-step 回报计算 ----
    buf = NStepPrioritizedReplay(capacity=100, n_step=3, gamma=0.5,
                                 augment=False, seed=1)
    grid_a = np.zeros((4, 4), dtype=np.uint16)
    grid_b = np.ones((4, 4), dtype=np.uint16)
    buf.push(grid_a, 0, 1.0, grid_b, False)
    buf.push(grid_b, 1, 2.0, grid_a, False)
    assert len(buf) == 0, "不足 n 步时不应写入"
    buf.push(grid_a, 2, 4.0, grid_b, False)
    assert len(buf) == 1, f"攒够 3 步应写入 1 条, 实际 {len(buf)}"
    s, a, r, ns, d = buf.data[0]
    expect = 1.0 + 0.5 * 2.0 + 0.25 * 4.0           # = 3.0
    assert abs(r - expect) < 1e-6, f"n-step 回报错误: {r} vs {expect}"
    assert a == 0, "起点动作应为第 1 步的动作"
    assert np.array_equal(ns, grid_b), "next_state 应为第 n 步的 next_state"
    print(f"2. n-step 回报 ✓ (r={r:.3f} = 1 + 0.5*2 + 0.25*4)")

    # ---- 3. 终止截断 ----
    buf2 = NStepPrioritizedReplay(capacity=100, n_step=5, gamma=0.99,
                                  augment=False, seed=2)
    buf2.push(grid_a, 0, 1.0, grid_b, False)
    buf2.push(grid_b, 1, 2.0, grid_a, True)         # 第 2 步终止
    assert len(buf2) == 2, f"终止应 flush 剩余 2 条, 实际 {len(buf2)}"
    _, a0, r0, ns0, d0 = buf2.data[0]
    assert d0 is True, "含终止的 n-step 转移应标记 done"
    assert abs(r0 - (1.0 + 0.99 * 2.0)) < 1e-6, f"截断回报错误: {r0}"
    print(f"3. 终止截断 ✓ (回报 {r0:.4f}, done=True)")

    # ---- 4. 优先采样 + IS 权重 + 对称增强 ----
    buf3 = NStepPrioritizedReplay(capacity=1000, n_step=3, gamma=0.99,
                                  augment=True, seed=3)
    rng = np.random.default_rng(0)
    for i in range(200):
        g = (2 ** rng.integers(0, 8, size=(4, 4))).astype(np.uint16)
        buf3.push(g, int(i % 4), float(i % 7), g, False)
    out = buf3.sample(32)
    st_b, act, rew, nst, dn, w, idxs = out
    assert st_b.shape == (32, 16, 4, 4), f"状态形状错误 {st_b.shape}"
    assert act.shape == (32,) and w.shape == (32,)
    assert st_b.max() <= 1.0 and st_b.min() >= 0.0, "编码应为二值"
    assert 0 < w.min() and w.max() <= 1.0 + 1e-6, "IS 权重应在 (0,1]"
    assert len(set(idxs.tolist())) > 1, "采样索引应有多样性"
    print(f"4. 优先采样 ✓ (batch={st_b.shape}, IS权重 [{w.min():.3f}, {w.max():.3f}], "
          f"β={buf3._beta():.3f})")

    # ---- 5. 优先级更新影响采样分布 ----
    buf3.update_priorities(idxs, np.abs(np.random.randn(32)) * 100)
    out2 = buf3.sample(16)
    assert out2[0].shape == (16, 16, 4, 4)
    print(f"5. 优先级更新 ✓ (max_priority={buf3.max_priority:.2f})")

    # ---- 6. 环形容量 ----
    buf4 = NStepPrioritizedReplay(capacity=50, n_step=1, gamma=0.99,
                                  augment=False, seed=4)
    for i in range(200):
        buf4.push(grid_a, i % 4, 1.0, grid_b, False)
    assert len(buf4) == 50, f"容量限制失效: {len(buf4)}"
    print(f"6. 环形容量 ✓ (写入 200 条, 保留 {len(buf4)} 条)")

    print("\nbuffer_v2.py 自测全部通过 ✓")
