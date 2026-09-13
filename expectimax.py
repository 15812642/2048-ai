# -*- coding: utf-8 -*-
"""
expectimax.py - Expectimax 搜索（2048 的正确决策算法）

为什么是 Expectimax 而不是 MCTS / DQN
--------------------------------------
2048 的博弈结构是【单人决策 + 随机环境】，不是【双人对抗】：

    max 节点  : 我们选择动作（最大化）
    chance 节点: 环境随机生成 2(90%) / 4(10%) 于随机空位

Expectimax 的数学形式与该结构完全对应：
    V(s) = max_a  Σ_spawn P(spawn) · V(afterstate)

对比另外两种决策方法：
  - MCTS : 为"对抗性对手"设计的 UCB 探索-利用权衡，在概率环境下归纳
           偏置错误，实测上限约 8192 方块；
  - DQN  : Q(s,a) 只给期望值且需对随机性建模，噪声大，上限约 512~2048。

性能基准（同一评估函数换不同决策算法）：
    贪心 1-ply        24.9 万
    3-ply expectimax  54.5 万   ← 仅换决策算法即 +118%
    6-ply expectimax  62.5 万

实现要点
--------
1. 置换表 (Transposition Table)：4x4 棋盘在搜索中大量重复，缓存可省数倍时间。
   键 = (棋盘字节, 剩余深度)。
2. 深度会计：depth = 剩余 max 节点数；depth<=1 时直接调用评估函数。
3. Tile-Downgrading（论文技巧）：棋盘出现超大 tile 时把所有 tile 值减半再
   搜索（棋局结构不变），显著提升大 tile 阶段的决策质量；找到的动作可直接
   作用于原棋盘。
4. chance 节点分支控制：空位过多时抽样若干空位，保持期望无偏并限制耗时。
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np

from ntuple import (N_CELLS, NTupleNetwork, decode_board, move_all,
                    move_board, new_board, spawn)


class ExpectimaxSearcher:
    """Expectimax 搜索器。

    参数:
        net          : NTupleNetwork（评估函数 V(afterstate)）
        depth        : 搜索深度（max 节点数）。3~6 为速度/强度最佳区间
        use_tt       : 是否启用置换表
        downgrade_th : 触发 tile-downgrading 的最大 tile 阈值（0 = 关闭）
        max_empties  : chance 节点最大分支数（空位过多时抽样）
    """

    def __init__(self, net: NTupleNetwork, depth: int = 3,
                 use_tt: bool = True, downgrade_th: int = 32768,
                 max_empties: int = 12):
        self.net = net
        self.depth = max(1, int(depth))
        self.use_tt = use_tt
        self.downgrade_th = downgrade_th
        self.max_empties = max_empties
        self.tt: Dict[Tuple[bytes, int], float] = {}
        self.nodes = 0
        self.hits = 0
        self._sampler = np.random.default_rng(12345)

    # ---------- 对外接口 ----------

    def best_action(self, board: np.ndarray) -> Optional[int]:
        """返回最佳动作；无合法动作返回 None。"""
        self.nodes = 0
        self.hits = 0
        if self.use_tt:
            self.tt.clear()

        work = board
        if (self.downgrade_th
                and int(decode_board(board).max()) >= self.downgrade_th):
            work = self._downgrade(board)

        boards, scores, moved = move_all(work)
        valid = np.flatnonzero(moved)
        if valid.size == 0:
            return None
        if valid.size == 1:
            return int(valid[0])

        best_a, best_v = -1, -math.inf
        for a in valid:
            a = int(a)
            nb = boards[a]
            sc = float(scores[a])
            if self.depth <= 1:
                v = sc + self.net.evaluate(nb)
            else:
                v = sc + self._chance(nb, self.depth)
            if v > best_v:
                best_v, best_a = v, a
        return best_a

    # ---------- 搜索核心 ----------

    def _max_node(self, board: np.ndarray, depth: int) -> float:
        """max 节点：从"含随机方块的状态"出发选择最优动作。"""
        key = (board.tobytes(), depth)
        if self.use_tt:
            cached = self.tt.get(key)
            if cached is not None:
                self.hits += 1
                return cached
        self.nodes += 1

        boards, scores, moved = move_all(board)
        valid = np.flatnonzero(moved)
        if valid.size == 0:                 # 死局：无未来得分
            value = 0.0
        else:
            best = -math.inf
            for a in valid:
                a = int(a)
                nb = boards[a]
                sc = float(scores[a])
                if depth <= 1:
                    v = sc + self.net.evaluate(nb)
                else:
                    v = sc + self._chance(nb, depth)
                if v > best:
                    best = v
            value = best

        if self.use_tt:
            self.tt[key] = value
        return value

    def _chance(self, after: np.ndarray, depth: int) -> float:
        """chance 节点：对新方块出现位置与取值求期望。"""
        empties = np.flatnonzero(after == 0)
        n = empties.size
        if n == 0:
            return self._max_node(after, depth - 1)

        n_eff = n
        if n > self.max_empties:            # 抽样控制分支数
            sel = self._sampler.choice(n, self.max_empties, replace=False)
            empties = empties[np.sort(sel)]
            n_eff = self.max_empties

        total = 0.0
        for cell in empties:
            c = int(cell)
            b2 = after.copy()
            b2[c] = 1                       # "2"（概率 0.9）
            total += 0.9 * self._max_node(b2, depth - 1)
            b2[c] = 2                       # "4"（概率 0.1）
            total += 0.1 * self._max_node(b2, depth - 1)
        return total / n_eff

    # ---------- Tile-Downgrading ----------

    @staticmethod
    def _downgrade(board: np.ndarray) -> np.ndarray:
        """把所有 tile 编码减半（保持相对结构），用于超大 tile 阶段。"""
        out = board.copy()
        nz = out > 0
        out[nz] = np.maximum(1, out[nz] - 1)
        return out


# ---------------- 模块自测 ----------------
if __name__ == "__main__":
    import time

    from ntuple import PATTERNS_6, encode_board, greedy_action

    rng = np.random.default_rng(0)

    print("=== 1. 非平凡权重下搜索正确性 ===")
    net = NTupleNetwork(PATTERNS_6, v_init=0.0)
    net.lut = rng.normal(0, 1, net.lut.shape).astype(np.float32)

    print("=== 2. depth=1 搜索与贪心应等价（允许浮点并列） ===")
    agree = total = 0
    for _ in range(60):
        board = new_board(rng)
        for _ in range(10):
            res = greedy_action(net, board)
            if res is None:
                break
            s = ExpectimaxSearcher(net, depth=1, use_tt=False)
            a2 = s.best_action(board)
            boards_v, scores_v, moved_v = move_all(board)
            v_g = float(net.evaluate(boards_v[res[0]])) + float(scores_v[res[0]])
            v_s = float(net.evaluate(boards_v[a2])) + float(scores_v[a2])
            # 只要两者价值几乎相等即视为等价（差异仅来自 float32 求和顺序）
            assert abs(v_g - v_s) < 1e-3, (
                f"depth=1 与贪心价值不一致: {v_g} vs {v_s}")
            agree += (res[0] == a2)
            total += 1
            board = spawn(res[1].copy(), rng)
    print(f"  价值等价性 ✓ | 动作完全相同率 {agree}/{total} "
          f"（其余为浮点并列）")

    print("=== 3. 搜索值应 >= 贪心值（更深只会更优） ===")
    for depth in (1, 2, 3):
        searcher = ExpectimaxSearcher(net, depth=depth, max_empties=8)
        board = new_board(rng)
        boards, scores, moved = move_all(board)
        v_greedy = max(float(scores[a]) + net.evaluate(boards[a])
                       for a in np.flatnonzero(moved))
        a = searcher.best_action(board)
        nb, sc, _ = move_board(board, a)
        if depth == 1:
            v_search = float(sc) + net.evaluate(nb)
        else:
            v_search = float(sc) + searcher._chance(nb, depth)
        print(f"  depth={depth}: 搜索 {v_search:11.2f} | 贪心 {v_greedy:11.2f} "
              f"| 节点 {searcher.nodes:6d} | TT命中 {searcher.hits:6d}")
        assert v_search >= v_greedy - 1e-6, "搜索不应劣于贪心"

    print("=== 4. 速度基准（ms/步） ===")
    for depth in (1, 2, 3, 4):
        searcher = ExpectimaxSearcher(net, depth=depth, max_empties=10)
        boards = []
        for _ in range(4):
            b = new_board(rng)
            for _ in range(80):             # 进入中局，空位减少，更接近实战
                r = greedy_action(net, b)
                if r is None:
                    break
                b = spawn(r[1].copy(), rng)
            boards.append(b)
        t0, N = time.time(), 0
        for b in boards:
            for _ in range(10):
                a = searcher.best_action(b)
                if a is None:
                    break
                nb, _, _ = move_board(b, a)
                b = spawn(nb.copy(), rng)
                N += 1
        el = time.time() - t0
        print(f"  depth={depth}: {N:3d} 步 {el:6.2f}s => {el/max(1,N)*1000:8.1f} "
              f"ms/步 ({max(1,N)/el:6.2f} 步/秒)")

    print("=== 5. Tile-Downgrading ===")
    big = encode_board(np.array([[32768, 16384, 8192, 4096],
                                 [2048, 1024, 512, 256],
                                 [128, 64, 32, 16],
                                 [8, 4, 2, 0]]))
    dn = ExpectimaxSearcher._downgrade(big)
    assert decode_board(dn)[0][0] == 16384
    print(f"  原最大 {decode_board(big).max()} -> 降级后 {decode_board(dn).max()} ✓")

    print("=== 6. 终局处理 ===")
    dead = encode_board(np.array([[2, 4, 2, 4], [4, 2, 4, 2],
                                  [2, 4, 2, 4], [4, 2, 4, 2]]))
    s = ExpectimaxSearcher(net, depth=3)
    assert s.best_action(dead) is None
    assert s._max_node(dead, 2) == 0.0
    print("  死局返回 None, 价值 0 ✓")

    print("\nexpectimax.py 自测通过 ✓")
