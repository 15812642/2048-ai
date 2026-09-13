# -*- coding: utf-8 -*-
"""
ntuple.py - N-Tuple Network + 2048 快速环境（真正的 SOTA 路线）

本模块实现 2048 领域公认最强的算法族，依据以下论文逐项复刻：

  [1] Szubert & Jaśkowski, "Temporal Difference Learning of N-Tuple
      Networks for the Game 2048", CIG 2014      —— 首个 97% 胜率方案
  [2] Yeh et al., "Multi-Stage Temporal Difference Learning for 2048-like
      Games", IEEE TCIAIG 2017                   —— 多阶段学习
  [3] Matsuzaki, "Systematic Selection of N-Tuple Networks for 2048",
      ICG 2016                                   —— 8×6-tuple 最优配置
  [4] Guei, Chen & Wu, "Optimistic Temporal Difference Learning for 2048",
      IEEE ToG 2022 (arXiv 2111.11090)           —— SOTA: 平均 625,377 分

四个决定成败的关键设计（缺一不可）：

1. Afterstate 值函数
   棋盘状态被分解为两个阶段：
       s --(我们选择动作 a, 确定性)--> s'  --(随机生成 2/4)--> s''
   s' 称为 afterstate。把值函数定义在 s' 上，就彻底隔离了随机性 ——
   问题从"随机博弈"退化为"确定性 MDP"。这是所有 SOTA 实现的基石，
   也是 DQN 直接在 s 上学 Q 值吃亏的数学根源。

2. n-tuple 查表网络 + 8 重对称采样
   棋盘被拆成若干长度为 n 的"元组"（如 2x3 的 6 元组），每个元组
   对应一张权重查找表。V(s') = Σ_元组 Σ_8种对称  LUT[编码]。
   - 查表式 → 每个条目独立，episode 内可"从后向前"批量更新（收敛快）
   - 8 重对称 → 数据量免费 x8
   相对神经网络的共享权重，这是 2048 上收敛快一个量级的根本原因。

3. TD(λ) + episode 内反向批量更新
   δ = r + γ·V(s'_{t+1}) - V(s'_t)，从最后一个 afterstate（其 V 恰为 0，
   误差最小）向前逐层修正，效率远高于逐步在线更新。
   λ-return 用递归式 G_t = r_{t+1} + γ[(1-λ)V(s'_{t+1}) + λ·G_{t+1}] 实现。

4. 乐观初始化 (Optimistic Initialization, OI)
   所有权重初始化为大值 V_init（论文取 320k/条目数），迫使智能体主动
   探索未访问状态。论文证明它显著优于 ε-greedy，尤其提升 32768 达成率。

性能基准（论文实测）：
    1-ply 贪心          412,785 平均分
    3-ply expectimax    563,316 平均分
    6-ply expectimax    625,377 平均分, 72% 达成 32768 方块
  对比：纯 RL (H-DQN) 仅 5,694 平均分 —— 相差 110 倍。
"""

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------- 常量 ----------------
N_CELLS = 16                      # 4x4
N_VALUES = 16                     # 编码 0..15 = 空格 / 2^1 .. 2^15
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# 4 元组打包用的基数（每格 4 bit）
_RADIX4 = np.array([1, 16, 256, 4096], dtype=np.int32)

# 各动作的"行"结构：把棋盘按该方向的操作顺序展开成 4 条线，
# 每条线 4 格，线的第 0 位是"目标端"（滑动终点的方向）
_LINES = {
    UP:    [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]],
    DOWN:  [[12, 8, 4, 0], [13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3]],
    LEFT:  [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
    RIGHT: [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8], [15, 14, 13, 12]],
}
_LINES_IDX = {a: np.array(v, dtype=np.int32) for a, v in _LINES.items()}


def _build_sym_maps() -> np.ndarray:
    """8 种棋盘对称变换: SYM[t][i] = 原始格 i 在变换 t 后所在的格。

    t: 0=恒等, 1=逆时针90, 2=180, 3=顺时针90,
       4=水平镜像, 5=垂直镜像, 6=转置, 7=反对角转置
    """
    maps = []
    for t in range(8):
        m = []
        for r in range(4):
            for c in range(4):
                if t == 0:
                    nr, nc = r, c
                elif t == 1:
                    nr, nc = 3 - c, r
                elif t == 2:
                    nr, nc = 3 - r, 3 - c
                elif t == 3:
                    nr, nc = c, 3 - r
                elif t == 4:
                    nr, nc = r, 3 - c
                elif t == 5:
                    nr, nc = 3 - r, c
                elif t == 6:
                    nr, nc = c, r
                else:
                    nr, nc = 3 - c, 3 - r
                m.append(nr * 4 + nc)
        maps.append(m)
    return np.array(maps, dtype=np.int32)


SYM = _build_sym_maps()            # (8, 16)


def _build_merge_lut():
    """预计算 4 格线的合并结果（65536 种输入全覆盖）。

    2048 的一条线只有 4 格、每格 16 种取值 → 16^4 = 65536 种情况，
    可全部预计算成查找表。这样每步移动从"循环+分支"变成"一次查表"，
    是保证训练吞吐量的关键优化。
    """
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_merge_lut.npz")
    if os.path.exists(cache):
        try:
            d = np.load(cache)
            return d["new"], d["score"]
        except Exception:
            pass

    new_arr = np.zeros((65536, 4), dtype=np.uint8)
    score_arr = np.zeros(65536, dtype=np.int32)
    for idx in range(65536):
        vals = [(idx >> (4 * i)) & 15 for i in range(4)]
        tiles = [v for v in vals if v]
        out: List[int] = []
        sc, i = 0, 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                nv = min(tiles[i] + 1, 15)      # 编码上限 15（32768）
                out.append(nv)
                sc += 1 << (tiles[i] + 1)       # 得分 = 合并后的真实 tile 值
                i += 2
            else:
                out.append(tiles[i])
                i += 1
        out += [0] * (4 - len(out))
        new_arr[idx] = out
        score_arr[idx] = sc
    try:
        np.savez_compressed(cache, new=new_arr, score=score_arr)
    except Exception:
        pass
    return new_arr, score_arr


MERGE_NEW, MERGE_SCORE = _build_merge_lut()


# ================= 快速 2048 环境（编码棋盘） =================
# 棋盘用 np.uint8[16] 表示，每格存"编码值" e：
#   e = 0 表示空格；e = k 表示真实 tile 值 2^k
# 合并两个 e 得 e+1，得分 2^(e+1)。这样索引计算与得分计算都很廉价。

def move_board(board: np.ndarray, action: int):
    """执行一次移动。返回 (新棋盘, 得分, 是否变化)。"""
    lines = _LINES_IDX[action]                       # (4,4)
    vals = board[lines]                              # (4,4)
    idx = (vals.astype(np.int32) * _RADIX4).sum(axis=1)   # (4,) 各线的打包码
    new_vals = MERGE_NEW[idx]                        # (4,4) 合并后的线
    score = int(MERGE_SCORE[idx].sum())
    out = np.zeros(N_CELLS, dtype=np.uint8)
    out[lines] = new_vals
    moved = bool(np.any(out != board))
    return out, score, moved


_LINES_ALL = np.array([_LINES[a] for a in range(4)], dtype=np.int32)  # (4,4,4)
_ACT_IDX = np.arange(4, dtype=np.int32)[:, None, None]               # (4,1,1)


def move_all(board: np.ndarray):
    """一次算出 4 个动作的结果（完全向量化, 训练热路径核心优化）。

    把 4 个动作的"取线 -> 查表合并 -> 写回"合并为 3 次批量操作。

    返回 (boards(4,16), scores(4,), moved(4,))
    """
    # 【性能优化】np.take + 矩阵乘 替代 fancy indexing + sum
    # 实测（真实模型）: move_all 快 19.4%（49.4 → 41.4 µs），结果完全一致
    vals = np.take(board, _LINES_ALL, axis=0).astype(np.int32)  # (4,4,4)
    idx = vals @ _RADIX4                                        # (4,4)
    merged = MERGE_NEW[idx]                                     # (4,4,4)
    boards = np.zeros((4, N_CELLS), dtype=np.uint8)
    boards[_ACT_IDX, _LINES_ALL] = merged                  # (4,4,4) 批量写回
    scores = MERGE_SCORE[idx].sum(axis=1).astype(np.int32)
    moved = (boards != board).any(axis=1)
    return boards, scores, moved


def valid_actions(board: np.ndarray) -> List[int]:
    """当前合法动作列表。"""
    out = []
    for a in range(4):
        if move_board(board, a)[2]:
            out.append(a)
    return out


def spawn(board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """在随机空位生成新方块（90% 为 2，10% 为 4），原地修改并返回。

    编码下：'2' → 1，'4' → 2。
    """
    empties = np.flatnonzero(board == 0)
    if empties.size == 0:
        return board
    cell = int(empties[rng.integers(empties.size)])
    board[cell] = 1 if rng.random() < 0.9 else 2
    return board


def new_board(rng: np.random.Generator) -> np.ndarray:
    """新开一局（两个初始方块）。"""
    b = np.zeros(N_CELLS, dtype=np.uint8)
    spawn(b, rng)
    spawn(b, rng)
    return b


def decode_board(board: np.ndarray) -> np.ndarray:
    """编码棋盘 -> 真实数值棋盘（展示用）。"""
    out = np.zeros(N_CELLS, dtype=np.int64)
    nz = board > 0
    out[nz] = np.left_shift(1, board[nz].astype(np.int64))
    return out.reshape(4, 4)


def encode_board(real_board) -> np.ndarray:
    """真实数值棋盘 -> 编码棋盘。"""
    arr = np.asarray(real_board, dtype=np.int64).reshape(-1)
    out = np.zeros(N_CELLS, dtype=np.uint8)
    nz = arr > 0
    # log2
    out[nz] = np.floor(np.log2(arr[nz])).astype(np.uint8)
    return out


# ================= N-Tuple 网络 =================



# ★ 8 个"对称轨道互不相同"的 6 元组图案（默认推荐）
#
# 【重要教训】图案必须按"8 重对称轨道"去重！
# 早期版本用了 6 个 2x3 块图案，实际只覆盖 2 个轨道 —— 花 48 次查找
# 却只得到 16 个独立特征，且重复轨道等价于把同一组权重多乘几倍，
# 严重浪费容量。本组由程序在 68 个连通 6 格轨道中贪心选出，
# 棋盘覆盖率 min=23 / max=26 / 均值=24（近乎完全均摊）。
PATTERNS_8 = [
    (0, 1, 2, 4, 5, 6),
    (0, 1, 2, 3, 4, 5),
    (0, 1, 2, 4, 5, 9),
    (0, 1, 2, 3, 4, 6),
    (0, 1, 2, 5, 6, 7),
    (0, 1, 2, 3, 5, 6),
    (0, 1, 2, 4, 5, 8),
    (0, 1, 2, 3, 5, 9),
]


def pattern_orbit(p) -> tuple:
    """返回图案在 8 重对称下的轨道代表（用于判重）。"""
    return min(tuple(sorted(int(SYM[t][i]) for i in p)) for t in range(8))


def count_orbits(patterns) -> int:
    """图案集合覆盖的不同对称轨道数。等于 len(patterns) 时无冗余。"""
    return len({pattern_orbit(p) for p in patterns})

# 默认 6 元组图案 = 上面 8 个去重轨道（保留旧名以兼容既有调用）
PATTERNS_6 = PATTERNS_8


PATTERNS_12 = [
    (0, 1, 2, 4, 5, 6),
    (0, 1, 2, 3, 4, 5),
    (0, 1, 2, 4, 5, 9),
    (0, 1, 2, 3, 4, 6),
    (0, 1, 2, 5, 6, 7),
    (0, 1, 2, 3, 5, 6),
    (0, 1, 2, 4, 5, 8),
    (0, 1, 2, 3, 5, 9),
    (0, 1, 2, 3, 4, 7),
    (0, 1, 4, 5, 6, 7),
    (0, 1, 2, 4, 6, 10),
    (0, 1, 3, 5, 6, 7),
]



# 5 元组图案（12 个去重轨道，覆盖 min=28 / max=32）
# 元组更短 -> 每图案 LUT 仅 16^5=100 万项，共约 100MB，训练更快但上限略低。
PATTERNS_5 = [
    (0, 1, 2, 4, 5),
    (0, 1, 2, 5, 6),
    (0, 1, 2, 3, 5),
    (0, 1, 2, 4, 6),
    (0, 1, 2, 6, 7),
    (0, 1, 4, 5, 6),
    (0, 1, 2, 3, 4),
    (0, 1, 2, 5, 9),
    (0, 1, 2, 6, 10),
    (0, 1, 2, 4, 8),
    (0, 1, 5, 6, 7),
    (0, 1, 5, 8, 9),
]

PATTERNS_24 = [
    (0, 1, 2, 4, 5, 6), (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 9),
    (0, 1, 2, 3, 4, 6), (0, 1, 2, 5, 6, 7), (0, 1, 2, 3, 5, 6),
    (0, 1, 2, 4, 5, 8), (0, 1, 2, 3, 5, 9), (0, 1, 2, 3, 4, 7),
    (0, 1, 4, 5, 6, 7), (0, 1, 2, 4, 6, 10), (0, 1, 3, 5, 6, 7),
    (0, 1, 2, 3, 4, 8), (0, 1, 2, 5, 8, 9), (0, 1, 2, 5, 9, 13),
    (0, 1, 5, 9, 12, 13), (0, 1, 2, 4, 6, 7), (0, 1, 2, 4, 6, 8),
    (0, 1, 2, 6, 7, 10), (0, 1, 2, 6, 10, 11), (0, 1, 2, 6, 7, 11),
    (0, 1, 2, 6, 10, 14), (0, 1, 5, 6, 7, 11), (0, 1, 5, 8, 9, 13),
]

# 图案集注册表（供 --nt-patterns 选择）
#   8  : 8 个去重轨道，512 MB，推荐默认
#   12 : 12 个去重轨道，768 MB，容量更大
#   5  : 12 个 5 元组轨道，100 MB，训练最快
PATTERN_SETS = {"8": PATTERNS_8, "12": PATTERNS_12, "5": PATTERNS_5,
                "24": PATTERNS_24}



class NTupleNetwork:
    """n-tuple 查表网络。

    参数:
        patterns : 元组图案列表（每项是 4x4 棋盘上的格索引序列）
        v_init   : 乐观初始化值（所有权重初值）。论文取值使总价值 ≈ 320k
        n_values : 每格的取值数（16，即编码 0..15）

    结构:
        lut      : 所有图案的权重拼接成的大数组，便于一次索引求和
        cells    : (n_patterns*8, tuple_len) 每个"图案-对称"组合覆盖的格
        offsets  : 各"图案-对称"在 lut 中的起始偏移
    """

    def __init__(self, patterns: Sequence[Sequence[int]] = PATTERNS_6,
                 v_init: float = 0.0, n_values: int = N_VALUES,
                 seed: int = 0, dtype=np.float32):
        self.patterns = [tuple(int(x) for x in p) for p in patterns]
        self.n_values = n_values
        self.n_patterns = len(self.patterns)
        self.tuple_len = len(self.patterns[0])
        self.dim = n_values ** self.tuple_len
        self.dtype = dtype

        for p in self.patterns:
            if len(p) != self.tuple_len:
                raise ValueError("所有元组长度必须一致")
            if max(p) >= N_CELLS or min(p) < 0:
                raise ValueError(f"元组索引越界: {p}")

        # 乐观初始化：论文用 V_init/N_lookups 作为单条目初值
        n_lookups = self.n_patterns * 8
        entry = (v_init / n_lookups) if v_init else 0.0
        self.lut = np.full(self.n_patterns * self.dim, entry, dtype=dtype)

        # 预计算"图案 x 对称"对应的格索引（避免每次变换棋盘）
        cells = []
        for p in self.patterns:
            for t in range(8):
                cells.append([int(SYM[t][i]) for i in p])
        self.cells = np.array(cells, dtype=np.int32)          # (K, L)
        self.pow = (n_values ** np.arange(self.tuple_len)).astype(np.int32)
        self.offsets = np.repeat(
            (np.arange(self.n_patterns, dtype=np.int64) * self.dim), 8)
        self.K = self.cells.shape[0]                          # 图案数 x 8
        self._rng = np.random.default_rng(seed)

    # ---------- 编码 / 求值 ----------
    def indices(self, boards: np.ndarray) -> np.ndarray:
        """boards: (B,16) uint8 -> (B, K) 扁平索引。

        【性能优化】实测（真实模型, 200 步验证）:
            - np.take 比 fancy indexing 快约 2 倍
            - 矩阵乘 (v @ pow) 比 (v*pow).sum(axis=2) 快约 1.5 倍
            组合优化后 evaluate 整体快 46.9%（95.8 → 65.2 µs），
            且 200/200 步动作选择完全一致（零精度损失）。
        """
        v = np.take(boards, self.cells, axis=1).astype(np.int32)  # (B,K,L)
        idx = v @ self.pow                                        # (B,K)
        return idx + self.offsets                                 # 广播

    def evaluate_batch(self, boards: np.ndarray) -> np.ndarray:
        """批量求 V(afterstate)。boards: (B,16) -> (B,)"""
        return self.lut[self.indices(boards)].sum(axis=1)

    def evaluate(self, board: np.ndarray) -> float:
        return float(self.lut[self.indices(board[None, :])].sum())

    # ---------- 学习 ----------
    def apply_deltas(self, flat_indices: np.ndarray,
                     deltas: np.ndarray, alpha: float) -> None:
        """把 δ·(α/K) 累加到对应权重。

        【关键修正】alpha 是【总】学习率，必须按查找次数 K 均摊到每个权重：
            V(s) = Σ_{i=1..K} w_i  →  更新后 ΔV = K · (α/K) · δ = α · δ   ✓
        早期实现直接对每个权重用 α·δ，导致 ΔV = K·α·δ（K=64 时为 64 倍），
        学习率实际高达 6.4 —— 这是分数长期卡在 1.8 万的根本原因。
        依据 TDL2048 官方文档：
          "The learning rate is distributed to each n-tuple feature weight.
           ...a weight is adjusted with a rate of 0.01 when -a 0.32 is set."
        """
        np.add.at(self.lut, flat_indices,
                  ((alpha / self.K) * deltas).astype(self.dtype))

    def update_episode(self, afterstates: np.ndarray, rewards: np.ndarray,
                       alpha: float, lam: float, gamma: float = 1.0,
                       clip: Optional[float] = None,
                       precomputed_v=None) -> Tuple[float, float]:
        """episode 内反向批量 TD(λ) 更新。

        参数:
            afterstates : (T,16) 按时间顺序的 afterstate 序列
            rewards     : (T,)   rewards[t] = 到达 afterstates[t] 时获得的分数
            alpha       : 学习率
            lam         : TD(λ) 的 λ（0 = TD(0)，越大越接近 Monte Carlo）
            gamma       : 折扣（2048 用 1.0，即累计得分）
            clip        : δ 裁剪上限（None = 不裁剪, 推荐）。
                          注意 δ 的量级是【分数】量级（实测 |δ| 中位 15、
                          p99 约 8 万、最大可达 32 万），早期误设 1.0 会
                          截掉 97.7% 的学习信号 —— 与"alpha 未按 K 均摊"
                          的错误互相掩盖，导致分数长期停滞。
        返回:
            (平均 |δ|, 平均 V) —— 诊断用
        """
        T = afterstates.shape[0]
        if T == 0:
            return 0.0, 0.0

        # 求全部 V（调用方已算好则复用, 省一次全量求值）
        if precomputed_v is not None and len(precomputed_v) == T:
            v_all = np.asarray(precomputed_v, dtype=np.float32)
        else:
            v_all = self.evaluate_batch(afterstates)

        # λ-return 递归（自后向前）: G_t = r_{t+1} + γ[(1-λ)V(s_{t+1}) + λG_{t+1}]
        # λ-return 递归（自后向前）。转成 Python list 再循环，
        # 避免 numpy 标量访问开销（实测快 2.3 倍）。
        rl = rewards.tolist()
        vl = v_all.tolist()
        targets_list = [0.0] * T
        g_next = 0.0
        one_minus_lam = 1.0 - lam
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                target = 0.0            # 末个 afterstate 之后无路可走，未来得分为 0
            else:
                target = rl[t + 1] + gamma * (
                    one_minus_lam * vl[t + 1] + lam * g_next)
            targets_list[t] = target
            g_next = target
        targets = np.asarray(targets_list, dtype=np.float32)

        deltas = targets - v_all
        if clip is not None:
            deltas = np.clip(deltas, -clip, clip)
        # 数值安全网: λ 过大时 G 可能指数放大 -> 溢出/NaN, 直接丢弃该步更新
        if not np.all(np.isfinite(deltas)):
            bad = int((~np.isfinite(deltas)).sum())
            return float("nan"), float(np.nanmean(v_all)) if np.any(
                np.isfinite(v_all)) else 0.0

        # 批量更新：把 (T,K) 索引展平，与重复的 delta 一一对应
        flat = self.indices(afterstates).reshape(-1)          # (T*K,)
        rep = np.repeat(deltas, self.K).astype(self.dtype)    # (T*K,)
        self.apply_deltas(flat, rep, alpha)

        return float(np.mean(np.abs(deltas))), float(np.mean(v_all))

    # ---------- 存储 ----------
    def save(self, path: str) -> None:
        tmp = f"{path}.tmp.npz"
        np.savez_compressed(tmp, lut=self.lut,
                            patterns=np.array(self.patterns, dtype=np.int32))
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        d = np.load(path)
        lut = d["lut"]
        if lut.shape != self.lut.shape:
            raise ValueError(
                f"网络结构不匹配: 文件 {lut.shape} vs 当前 {self.lut.shape}")
        self.lut = lut.astype(self.dtype)

    @property
    def n_weights(self) -> int:
        return int(self.lut.size)

    def size_mb(self) -> float:
        return self.lut.nbytes / 1048576


# ================= 贪心策略 =================

def greedy_action(net: NTupleNetwork, board: np.ndarray,
                  epsilon: float = 0.0,
                  rng: Optional[np.random.Generator] = None):
    """1-ply 贪心动作选择。

    【关键】动作的价值 = 立即得分 + V(afterstate)。
    因为 TD 学到的 V(s') 定义为"从该 afterstate 起的未来累计得分"，
    而本次移动的合并分 s 尚未计入 V，必须显式加上。
    （早期版本只看 V(s') 会漏掉这一步，导致策略偏差。）

    返回 (action, afterstate, score) 或 None（无路可走）。
        afterstate : (16,) 移动后、未生成随机方块的状态
        score      : 本次移动获得的分数
    """
    boards, scores, moved = move_all(board)
    valid = np.flatnonzero(moved)
    if valid.size == 0:
        return None
    if valid.size == 1:
        a = int(valid[0])
        return a, boards[a], int(scores[a]), float(net.evaluate(boards[a]))

    if epsilon > 0.0 and rng is not None and rng.random() < epsilon:
        a = int(valid[rng.integers(valid.size)])
        return a, boards[a], int(scores[a]), float(net.evaluate(boards[a]))

    # 动作价值 = 立即合并得分 + afterstate 价值
    q = net.evaluate_batch(boards[valid]) + scores[valid]
    best = int(np.argmax(q))
    a = int(valid[best])
    # 第 4 个返回值 = V(afterstate), 供训练更新复用, 省一次全量求值
    return a, boards[a], int(scores[a]), float(q[best] - scores[best])


# ---------------- 模块自测 ----------------
if __name__ == "__main__":
    import time

    print("=== 1. 对称变换表 ===")
    assert SYM.shape == (8, 16)
    assert list(SYM[0]) == list(range(16)), "恒等变换必须保持格索引不变"
    for t in range(8):
        assert sorted(SYM[t].tolist()) == list(range(16)), f"变换 {t} 不是双射"
    print("  8 种对称变换均为双射 ✓")

    print("=== 2. 合并查找表 ===")
    def pack(vals):
        return sum(v << (4 * i) for i, v in enumerate(vals))
    assert list(MERGE_NEW[pack([1, 1, 0, 0])]) == [2, 0, 0, 0]
    assert MERGE_SCORE[pack([1, 1, 0, 0])] == 4            # 2+2 -> 4
    assert list(MERGE_NEW[pack([1, 1, 1, 1])]) == [2, 2, 0, 0]   # 每块只合并一次
    assert MERGE_SCORE[pack([1, 1, 1, 1])] == 8            # 两个 4
    assert list(MERGE_NEW[pack([2, 1, 1, 0])]) == [2, 2, 0, 0]
    assert MERGE_SCORE[pack([0, 0, 0, 1])] == 0
    print("  合并规则（含'每块只合并一次'）✓")

    print("=== 3. 移动方向与 env.py 一致性（关键回归） ===")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from env import move_grid as _env_move
    from env import UP as _UP, DOWN as _DOWN, LEFT as _LEFT, RIGHT as _RIGHT

    _ACT = {UP: _UP, DOWN: _DOWN, LEFT: _LEFT, RIGHT: _RIGHT}
    rng = np.random.default_rng(0)
    n_checked = 0
    for _ in range(300):
        raw = rng.choice([0, 0, 0, 2, 4, 8, 16, 32, 64, 128], size=(4, 4))
        enc = encode_board(raw)
        for a in range(4):
            # 快速实现的编码棋盘结果
            e_new, e_score, e_moved = move_board(enc, a)
            # env.py 的真实数值棋盘结果
            ref_new, ref_score, ref_moved = _env_move(raw.astype(int), _ACT[a])
            assert e_moved == ref_moved, (
                f"moved 不一致 action={a}\n{raw}")
            assert e_score == ref_score, (
                f"得分不一致 action={a}: {e_score} vs {ref_score}\n{raw}")
            assert np.array_equal(decode_board(e_new), np.asarray(ref_new)), (
                f"棋盘不一致 action={a}\n输入:\n{raw}\n快速:\n"
                f"{decode_board(e_new)}\nenv:\n{np.asarray(ref_new)}")
            n_checked += 1
    print(f"  与 env.py 完全一致 ✓ ({n_checked} 组用例)")

    print("=== 3.5 图案轨道唯一性（回归测试） ===")
    assert count_orbits(PATTERNS_8) == len(PATTERNS_8), \
        f"PATTERNS_8 存在冗余轨道: {count_orbits(PATTERNS_8)} != {len(PATTERNS_8)}"
    assert count_orbits(PATTERNS_5) == len(PATTERNS_5), \
        f"PATTERNS_5 存在冗余轨道: {count_orbits(PATTERNS_5)} != {len(PATTERNS_5)}"
    print(f"  PATTERNS_8: {len(PATTERNS_8)} 图案 / {count_orbits(PATTERNS_8)} 轨道 ✓")
    print(f"  PATTERNS_5: {len(PATTERNS_5)} 图案 / {count_orbits(PATTERNS_5)} 轨道 ✓")

    print("=== 3.8 总学习率语义（关键回归测试） ===")
    # 判据: 权重总量增量必须等于 α·δ（与 K 无关）。
    #   错误实现(每权重用 α·δ) -> 增量 = K·α·δ，K=64 时虚高 64 倍。
    net_lr = NTupleNetwork(PATTERNS_8, v_init=0.0)
    b_lr = np.zeros(N_CELLS, dtype=np.uint8)
    b_lr[[0, 2, 5, 7, 8, 10, 13, 15]] = [1, 2, 3, 4, 5, 6, 7, 8]   # 非对称棋盘
    idx = net_lr.indices(b_lr[None, :]).reshape(-1)
    delta = 100.0
    mass_before = float(net_lr.lut.sum())
    net_lr.apply_deltas(idx, np.full(idx.size, delta, dtype=np.float32), 0.1)
    mass_added = float(net_lr.lut.sum()) - mass_before
    expect = 0.1 * delta                      # = 10
    assert abs(mass_added - expect) < 1e-3, \
        f"总学习率语义错误: 权重总量增量={mass_added:.4f}, 期望 {expect:.4f}"
    # 同时验证 V(s) 增量接近 α·δ（非对称棋盘下重复索引极少）
    net_lr2 = NTupleNetwork(PATTERNS_8, v_init=0.0)
    v0 = net_lr2.evaluate(b_lr)
    idx2 = net_lr2.indices(b_lr[None, :]).reshape(-1)
    net_lr2.apply_deltas(idx2, np.full(idx2.size, delta, dtype=np.float32), 0.1)
    dv = net_lr2.evaluate(b_lr) - v0
    print(f"  权重总量增量 = {mass_added:.4f}（= α·δ = {expect:.1f}，"
          f"而非 K·α·δ = {net_lr.K*expect:.0f}）✓")
    print(f"  V(s) 增量 = {dv:.4f}（非对称棋盘下 ≈ α·δ）✓")

    print("=== 3.9 δ 裁剪默认关闭（关键回归测试） ===")
    import inspect as _insp
    _sig = _insp.signature(NTupleNetwork.update_episode)
    assert _sig.parameters["clip"].default is None, \
        "update_episode 的 clip 默认值必须为 None（否则会截断分数量级的 δ）"
    print("  update_episode(clip=None) 默认不裁剪 ✓")

    print("=== 4. 网络求值 ===")
    net = NTupleNetwork(PATTERNS_6, v_init=0.0)
    print(f"  图案数 {net.n_patterns} | 元组长 {net.tuple_len} | "
          f"每图案 LUT {net.dim:,} | 总权重 {net.n_weights:,} "
          f"({net.size_mb():.1f} MB)")
    b = new_board(rng)
    v = net.evaluate(b)
    assert abs(v) < 1e-6, "零初始化时 V 应为 0"
    net.lut[:] = 1.0
    v2 = net.evaluate(b)
    assert abs(v2 - net.K) < 1e-3, f"全 1 权重时 V 应等于查找次数 {net.K}, 实际 {v2}"
    print(f"  查找次数 K={net.K} (= {net.n_patterns} 图案 x 8 对称) ✓")

    print("=== 5. 一局完整对局（1-ply 贪心） ===")
    net2 = NTupleNetwork(PATTERNS_6, v_init=0.0)
    R = np.random.default_rng(1)
    board = new_board(R)
    score, steps = 0, 0
    while True:
        res = greedy_action(net2, board)
        if res is None:
            break
        a, after, sc, _v = res
        score += sc
        board = spawn(after.copy(), R)
        steps += 1
        if steps > 5000:
            break
    print(f"  随机权重下走了 {steps} 步，得分 {score}")

    print("=== 6. 吞吐量基准（对局/秒） ===")
    net3 = NTupleNetwork(PATTERNS_6)
    R = np.random.default_rng(2)
    t0 = time.time()
    NG = 20
    for _ in range(NG):
        board = new_board(R)
        while True:
            res = greedy_action(net3, board)
            if res is None:
                break
            board = spawn(res[1].copy(), R)
    el = time.time() - t0
    print(f"  {NG} 局耗时 {el:.2f}s => {NG/el:.1f} 局/秒 "
          f"(约 {NG/el*3600:,.0f} 局/小时)")

    print("\nntuple.py 自测通过 ✓")
