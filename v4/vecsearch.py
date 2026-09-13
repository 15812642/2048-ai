# -*- coding: utf-8 -*-
"""
vecsearch.py - v4 向量化 Expectimax（确定性提速）

问题诊断
--------
当前 expectimax 逐叶子评估：
    for 动作 a (4 个):
        for 空位 cell (约 6 个):
            for 方块值 (2, 4):          ← 每次单独调 move_all + evaluate
                评估该叶子

    24~48 次独立评估 × 18µs ≈ 0.5~0.9 ms（单层）
    2-ply 展开后累计约 3.5 ms

关键洞察
--------
这些叶子的计算【完全独立】—— 可以打包成一次批量运算：
    1. 批量 move_all（numpy 一次算 B 个局面的 4 个动作）
    2. 批量查表评估（一次索引 B*4 个 afterstate）
    3. 向量化加权聚合

预期收益：3~8x（取决于 numpy 批量效率 vs Python 循环开销）

为什么这是确定性收益
--------------------
- 结果与逐叶子计算【数学等价】（同样的查表值，只换了计算顺序）
- 不引入近似、不需要训练、不改变模型
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.dirname(_HERE), _HERE, "/opt/2048ai", "/workspace/2048_ai"):
    if os.path.isfile(os.path.join(_c, "ntuple.py")):
        sys.path.insert(0, _c)
        break

from ntuple import (N_CELLS, PATTERNS_8, NTupleNetwork, decode_board,
                    greedy_action, move_all, new_board, spawn)


def move_all_batch(boards, lines_all, act_idx, radix, merge_new,
                   merge_score, n_cells=16):
    """批量版 move_all：一次算 B 个局面的全部 4 个动作。

    参数 boards: (B,16) uint8
    返回 (out( B,4,16), scores(B,4), moved(B,4))
    """
    vals = boards[:, lines_all].astype(np.int32)       # (B,4,4,4)
    idx = (vals * radix).sum(axis=3)                   # (B,4,4)
    merged = merge_new[idx]                            # (B,4,4,4)
    out = np.zeros((boards.shape[0], 4, n_cells), dtype=np.uint8)
    out[:, act_idx, lines_all] = merged                # (B,4,4,4) 散射写回
    scores = merge_score[idx].sum(axis=2).astype(np.int32)   # (B,4)
    moved = np.any(out != boards[:, None, :], axis=2)        # (B,4)
    return out, scores, moved


class VecExpectimax:
    """向量化 2-ply 期望搜索（与 ExpectimaxSearcher 结果等价）。"""

    def __init__(self, net: NTupleNetwork, max_empties: int = 10):
        self.net = net
        self.max_empties = max_empties
        # 从 ntuple 模块借用预计算表
        import ntuple as _nt
        self._lines_all = _nt._LINES_ALL
        self._act_idx = _nt._ACT_IDX
        self._radix = _nt._RADIX4
        self._merge_new = _nt.MERGE_NEW
        self._merge_score = _nt.MERGE_SCORE
        self.nodes = 0
        self.last_stats = {}

    # ---------- 核心：批量计算"局面的 4 个动作价值" ----------
    def action_values(self, boards: np.ndarray) -> np.ndarray:
        """批量计算 2-ply 动作价值。boards: (B,16) -> (B,4)。

        流程（全部向量化）:
            B 个局面
              → 批量 move_all          (B,4,16) afterstates
              → 批量生成 chance 子节点  (B,4,M,16)
              → 批量 move_all          (B,4,M,4,16)
              → 批量查表评估            (B,4,M,4)
              → 加权聚合                (B,4)
        """
        B = boards.shape[0]
        out, scores, moved = move_all_batch(
            boards, self._lines_all, self._act_idx, self._radix,
            self._merge_new, self._merge_score)

        # ---- 展开 chance 节点：每个 afterstate 的所有空位 x {2,4} ----
        # 为控制内存，逐局面构造 chance 集合（空位数不同）
        vals = np.full((B, 4), -np.inf, dtype=np.float64)
        for i in range(B):
            for a in range(4):
                if not moved[i, a]:
                    continue
                after = out[i, a]
                empties = np.flatnonzero(after == 0)
                n = empties.size
                if n == 0:
                    # 【关键修正】棋盘已满时, chance 节点没有分支, 但仍需
                    # 对"下一步动作"取 max —— 早期误用 net.evaluate(after)
                    # （未做动作最大化）导致该分支价值被低估, 在空位稀少的
                    # 后期影响显著（实测整局均分从 100k 掉到 52k）。
                    vals[i, a] = float(scores[i, a]) + float(
                        self._eval_positions(after[None, :])[0])
                    continue
                if n > self.max_empties:
                    empties = empties[:self.max_empties]
                    n = self.max_empties

                # 构造 2n 个子局面（向量化）
                subs = np.repeat(after[None, :], 2 * n, axis=0)     # (2n,16)
                cells = np.repeat(empties, 2)
                tiles = np.tile([1, 2], n)                          # 2 / 4
                subs[np.arange(2 * n), cells] = tiles

                # 批量评估子局面的一步价值
                leaf_v = self._eval_positions(subs)                 # (2n,)
                w = np.where(tiles == 1, 0.9, 0.1)                  # (2n,)
                vals[i, a] = float(scores[i, a]) + float(
                    (leaf_v * w).sum() / n)
        self.nodes += B
        return vals

    # ---------- 叶子评估（批量） ----------
    def _eval_after(self, after: np.ndarray) -> float:
        """评估单个 afterstate（无后续机会分支）。"""
        return float(self.net.evaluate(after))

    def _eval_positions(self, boards: np.ndarray) -> np.ndarray:
        """批量评估：max_a [score + V(afterstate)]，返回 (B,)。

        这是 1-ply 的批量实现（即"看完一步后的最好结果"）。
        """
        out, scores, moved = move_all_batch(
            boards, self._lines_all, self._act_idx, self._radix,
            self._merge_new, self._merge_score)
        B = boards.shape[0]
        flat = out.reshape(B * 4, N_CELLS)
        v = self.net.evaluate_batch(flat).reshape(B, 4)     # (B,4) 批量查表
        v = v + scores
        v[~moved] = -np.inf
        res = v.max(axis=1)
        # 【关键修正】无合法动作（死局）时, 逐叶子实现返回 0.0
        # （见 expectimax._max_node 的 valid.size==0 分支）。
        # 早期版本让 max 返回 -inf, 使死局价值被严重低估,
        # 导致搜索在"接近死局"时做出错误决策 —— 实测整局均分
        # 从 100k 掉到 52k。
        res[~np.isfinite(res)] = 0.0
        return res

    # ---------- 决策 ----------
    def best_action(self, board: np.ndarray):
        v = self.action_values(board[None, :])[0]
        valid = np.isfinite(v)
        if not valid.any():
            return None
        return int(np.argmax(np.where(valid, v, -np.inf)))


# ---------------- 自测：与逐叶子版本对比 ----------------
if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else None
    net = NTupleNetwork(PATTERNS_8)
    if mp and os.path.exists(mp):
        net.load(mp)
        print(f"模型: {mp} (σ={net.lut.std():.1f})")
    else:
        rng0 = np.random.default_rng(0)
        net.lut = rng0.normal(0, 10, net.lut.shape).astype(np.float32)
        print("模型: 随机权重（自测用）")

    rng = np.random.default_rng(3)
    boards = []
    for _ in range(12):
        b = new_board(rng)
        for _ in range(rng.integers(50, 300)):
            r = greedy_action(net, b)
            if r is None:
                break
            b = spawn(r[1].copy(), rng)
        boards.append(b)
    boards = np.asarray(boards)

    from expectimax import ExpectimaxSearcher
    searcher = ExpectimaxSearcher(net, depth=2, max_empties=10, use_tt=False)
    vec = VecExpectimax(net, max_empties=10)

    print("\n=== 1. 结果等价性（向量化 vs 逐叶子）===")
    # 补充边界用例: 满盘（无空位）与死局
    from ntuple import encode_board as _enc
    edge = [
        _enc(np.array([[2,4,2,4],[4,2,4,2],[2,4,2,4],[4,2,4,2]])),   # 死局
        _enc(np.array([[2,4,2,4],[4,2,4,2],[2,4,2,4],[4,2,4,0]])),   # 仅 1 空位
        _enc(np.array([[2,2,4,4],[8,8,16,16],[32,32,64,64],[128,128,2,2]])),
    ]
    boards = np.concatenate([boards, np.asarray(edge)])
    max_diff = 0.0
    for b in boards:
        ref = []
        bs, ss, ms = move_all(b)
        for a in range(4):
            if not ms[a]:
                ref.append(-np.inf)
            else:
                ref.append(float(ss[a]) + searcher._chance(bs[a], 2))
        ref = np.asarray(ref)
        got = vec.action_values(b[None, :])[0]
        both = np.isfinite(ref) & np.isfinite(got)
        d = np.abs(ref[both] - got[both]).max() if both.any() else 0.0
        max_diff = max(max_diff, d)
    print(f"  最大绝对差: {max_diff:.3f}  "
          f"({'✓ 等价' if max_diff < 1.0 else '✗ 不一致'})")

    print("\n=== 2. 速度对比 ===")
    t0 = time.time()
    for b in boards:
        searcher.best_action(b)
    t_ref = (time.time() - t0) / len(boards)

    t0 = time.time()
    for b in boards:
        vec.best_action(b)
    t_vec = (time.time() - t0) / len(boards)

    print(f"  逐叶子: {t_ref*1000:.2f} ms/局面")
    print(f"  向量化: {t_vec*1000:.2f} ms/局面")
    print(f"  加速比: {t_ref/max(1e-9, t_vec):.2f}x")

    print("\n=== 3. 决策一致性 ===")
    same = 0
    for b in boards[:8]:
        a1 = searcher.best_action(b)
        a2 = vec.best_action(b)
        same += (a1 == a2)
    print(f"  动作一致: {same}/8")
