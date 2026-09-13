# -*- coding: utf-8 -*-
"""
vecsearch3.py - 向量化 3-ply 期望搜索（真正的两层展开）

结构对比
--------
2-ply（vecsearch.py）:
    max(4动作) → chance → max_a2[score + V]          ← 约 48 次查表

3-ply（本文件）:
    max(4动作) → chance → max(4动作) → chance → V    ← 约 2304 次查表

    第一层 max   : 4 个 afterstate
    第一层 chance: 4 × 2n ≈ 48 个 state1
    第二层 max   : 48 × 4 = 192 个 afterstate2
    第二层 chance: 192 × 2n ≈ 2304 个 state2
    评估         : V(state2) —— **一次性批量查表**

为什么需要向量化
----------------
逐叶子递归要 2304 次独立的 move + evaluate（每次约 85µs）→ 约 197 ms/步。
向量化后：批量 move + 批量查表 → 预计 5-15 ms/步。

正确性要点
----------
1. chance 采样方式与逐叶子实现保持一致（取前 M 个空位；M >= 空位数时全采样）
2. 死局（无合法动作）价值 = 0.0（与 expectimax._max_node 的约定一致）
3. 满盘（无空位）时 chance 分支退化为单点（权重 1.0）
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

from ntuple import (N_CELLS, PATTERNS_8, NTupleNetwork, greedy_action,
                    move_all, new_board, spawn)


class VecExpectimax3:
    """向量化 3-ply 期望搜索。

    参数:
        max_empties : 每个 afterstate 采样的最大空位数
        beam        : >0 时第二层 max 只保留 top-k 动作（束搜索）
    """

    def __init__(self, net: NTupleNetwork, max_empties: int = 6,
                 beam: int = 0):
        self.net = net
        self.max_empties = max_empties
        self.beam = beam
        self.nodes = 0
        import ntuple as _nt
        self._lines_all = _nt._LINES_ALL
        self._act_idx = _nt._ACT_IDX
        self._radix = _nt._RADIX4
        self._merge_new = _nt.MERGE_NEW
        self._merge_score = _nt.MERGE_SCORE

    # ---------- 批量原语 ----------
    def _move_batch(self, boards):
        """批量 move_all: (B,16) -> (B,4,16), (B,4), (B,4)"""
        vals = boards[:, self._lines_all].astype(np.int32)
        idx = (vals * self._radix).sum(axis=3)
        merged = self._merge_new[idx]
        out = np.zeros((boards.shape[0], 4, N_CELLS), dtype=np.uint8)
        out[:, self._act_idx, self._lines_all] = merged
        scores = self._merge_score[idx].sum(axis=2).astype(np.int32)
        moved = np.any(out != boards[:, None, :], axis=2)
        return out, scores, moved

    def _chance_cells(self, after: np.ndarray):
        """返回 chance 采样用的空位（确定性：取前 M 个）。"""
        empties = np.flatnonzero(after == 0)
        n = empties.size
        if n == 0:
            return empties, 0
        if n > self.max_empties:
            empties = empties[:self.max_empties]
            n = self.max_empties
        return empties, n

    # ---------- 核心 ----------
    def action_values(self, boards: np.ndarray) -> np.ndarray:
        """3-ply 动作价值。boards: (B,16) -> (B,4)"""
        B = boards.shape[0]
        result = np.full((B, 4), -np.inf, dtype=np.float64)

        # ===== 第一层 max =====
        out1, sc1, mv1 = self._move_batch(boards)

        for b in range(B):
            cand = [a for a in range(4) if mv1[b, a]]
            if not cand:
                continue

            # ===== 第一层 chance: 收集 state1 =====
            # s1_meta[k] = (动作 a1, 权重 w, a1 的即时得分 sc)
            s1_list, s1_meta = [], []
            for a1 in cand:
                after = out1[b, a1]
                cells, n = self._chance_cells(after)
                if n == 0:
                    # 满盘：无 chance 分支，该 action 退化为"直接再走一步"
                    s1_list.append(after)          # 占位（后面单独处理）
                    s1_meta.append((a1, 1.0, float(sc1[b, a1]), True))
                    continue
                w_each = 1.0 / n
                for cell in cells:
                    c = int(cell)
                    for tile, pw in ((1, 0.9), (2, 0.1)):
                        nb = after.copy()
                        nb[c] = tile
                        s1_list.append(nb)
                        s1_meta.append((a1, w_each * pw, float(sc1[b, a1]), False))

            if not s1_list:
                continue

            arr1 = np.asarray(s1_list, dtype=np.uint8)       # (K1,16)
            K1 = arr1.shape[0]

            # ===== 第二层 max: 批量展开动作 =====
            out2, sc2, mv2 = self._move_batch(arr1)          # (K1,4,16)

            if 0 < self.beam < 4:
                # 束搜索：用 1-ply 粗筛保留 top-k
                flat = out2.reshape(K1 * 4, N_CELLS)
                rough = (self.net.evaluate_batch(flat).reshape(K1, 4)
                         + sc2.astype(np.float64))
                rough[~mv2] = -np.inf
                keep = np.zeros_like(mv2, dtype=bool)
                order = np.argsort(-rough, axis=1)[:, :self.beam]
                np.put_along_axis(keep, order, True, axis=1)
                mv2 = mv2 & keep

            # ===== 第二层 chance: 收集 state2（同时记录归属）=====
            # 每条记录: (k1 索引, a2 动作, 权重, sc2 得分)
            s2_list, s2_rec = [], []
            k1_full = {}          # k1 -> (a2 -> 满盘价值)，处理无 chance 分支
            for k in range(K1):
                for a2 in range(4):
                    if not mv2[k, a2]:
                        continue
                    after2 = out2[k, a2]
                    cells, n = self._chance_cells(after2)
                    if n == 0:
                        # 满盘：直接 1-ply 评估（无 chance2）
                        k1_full.setdefault(k, {})[a2] = (
                            float(sc2[k, a2]) + self._eval_1ply_single(after2))
                        continue
                    w_each = 1.0 / n
                    for cell in cells:
                        c = int(cell)
                        for tile, pw in ((1, 0.9), (2, 0.1)):
                            nb = after2.copy()
                            nb[c] = tile
                            s2_list.append(nb)
                            s2_rec.append((k, a2, w_each * pw,
                                           float(sc2[k, a2])))

            # ===== 批量评估: max_a3 [score + V(after3)] =====
            # 【关键】不能直接 V(state2)！expectimax 的 depth=3 语义是
            #   max → chance → max → chance → max → V
            # 即最后还要再选一次动作。早期误用 evaluate_batch 少了一层 max，
            # 导致价值系统性偏低（实测最大差异 7642、决策一致率仅 57%）。
            if s2_list:
                v2 = self._eval_1ply_batch(
                    np.asarray(s2_list, dtype=np.uint8))
            else:
                v2 = np.empty(0, dtype=np.float64)

            # ===== 聚合第二层：max over a2 =====
            # k1 -> {a2: 加权期望价值}
            per_a2 = [dict() for _ in range(K1)]
            for (k, a2, w, scv), val in zip(s2_rec, v2):
                per_a2[k][a2] = per_a2[k].get(a2, 0.0) + w * (scv + float(val))
            for k, d in k1_full.items():
                for a2, val in d.items():
                    per_a2[k][a2] = val           # 满盘分支直接覆盖

            k1_vals = np.zeros(K1, dtype=np.float64)
            for k in range(K1):
                if per_a2[k]:
                    k1_vals[k] = max(per_a2[k].values())

            # ===== 聚合第一层 chance + max =====
            acc1 = {a: 0.0 for a in cand}
            for k, (a1, w, scv, is_full) in enumerate(s1_meta):
                if is_full:
                    # 满盘 afterstate1：直接取其第二层 max 价值
                    acc1[a1] += k1_vals[k]
                else:
                    acc1[a1] += w * k1_vals[k]

            for a in cand:
                result[b, a] = acc1[a]

        self.nodes += B
        return result

    def _eval_1ply_batch(self, boards: np.ndarray) -> np.ndarray:
        """批量 1-ply: max_a [score + V(after)]；死局返回 0。"""
        if boards.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        out, scores, moved = self._move_batch(boards)
        B = boards.shape[0]
        flat = out.reshape(B * 4, N_CELLS)
        v = self.net.evaluate_batch(flat).reshape(B, 4) + scores
        v[~moved] = -np.inf
        res = v.max(axis=1)
        res[~np.isfinite(res)] = 0.0
        return res.astype(np.float64)

    def _eval_1ply_single(self, board: np.ndarray) -> float:
        """单棋盘 1-ply 评估: max_a [score + V(after)]；死局返回 0。"""
        out, scores, moved = move_all(board)
        best = -np.inf
        for a in range(4):
            if not moved[a]:
                continue
            v = float(scores[a]) + float(self.net.evaluate(out[a]))
            if v > best:
                best = v
        return 0.0 if best == -np.inf else best

    def best_action(self, board: np.ndarray):
        v = self.action_values(board[None, :])[0]
        valid = np.isfinite(v)
        if not valid.any():
            return None
        return int(np.argmax(np.where(valid, v, -np.inf)))


# ---------------- 自测 ----------------
if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else None
    net = NTupleNetwork(PATTERNS_8)
    if mp and os.path.exists(mp):
        net.load(mp)
        print(f"模型: {mp} (σ={net.lut.std():.1f})")
    else:
        rng0 = np.random.default_rng(0)
        net.lut = rng0.normal(0, 10, net.lut.shape).astype(np.float32)
        print("模型: 随机权重（自测）")

    rng = np.random.default_rng(5)
    grids = []
    for _ in range(8):
        b = new_board(rng)
        for _ in range(rng.integers(80, 200)):
            r = greedy_action(net, b)
            if r is None:
                break
            b = spawn(r[1].copy(), rng)
        grids.append(b)

    from expectimax import ExpectimaxSearcher
    ref = ExpectimaxSearcher(net, depth=3, max_empties=6, use_tt=False)
    vec = VecExpectimax3(net, max_empties=6)

    print("\n=== 1. 速度 ===")
    t0 = time.time()
    for b in grids[:3]:
        ref.best_action(b)
    t_ref = (time.time() - t0) / 3
    t0 = time.time()
    for b in grids:
        vec.best_action(b)
    t_vec = (time.time() - t0) / len(grids)
    print(f"  逐叶子: {t_ref*1000:8.2f} ms/步")
    print(f"  向量化: {t_vec*1000:8.2f} ms/步")
    print(f"  加速比: {t_ref/max(1e-9,t_vec):.0f}x")

    print("\n=== 2. 决策一致性 ===")
    same = 0
    for b in grids[:6]:
        same += (ref.best_action(b) == vec.best_action(b))
    print(f"  {same}/6")
