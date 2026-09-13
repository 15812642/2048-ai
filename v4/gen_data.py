# -*- coding: utf-8 -*-
"""
gen_data.py - v4 第一阶段：推理数据集生成

思路（AlphaZero 式知识蒸馏）
---------------------------
    n-tuple + expectimax（慢但强）
            │  搜索 = "推理"
            ▼
    生成 (局面 → 各动作搜索价值) 数据集
            │
            ▼
    训练快速学生网络（学"搜索的直觉"）
            │
            ▼
    推理时用学生网络替代搜索（快 100 倍）

为什么有价值
------------
实测发现：2-ply 搜索比 1-ply 贪心强 36%（40k → 54k 分），但慢 26 倍
（0.13ms → 3.5ms）。若学生网络能学到 2-ply 的决策质量，就能在
接近 1-ply 的速度下获得 2-ply 的水平。

数据格式（每条记录）
--------------------
    board   : uint8[16]      编码局面
    values  : float32[4]     各动作的 2-ply 搜索价值（非法动作为 -inf）
    best_a  : int8           搜索选出的最优动作
    stage   : int16          对局步数（用于阶段分桶分析）
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

from ntuple import (PATTERNS_8, NTupleNetwork, decode_board, greedy_action,
                    move_all, new_board, spawn)
from expectimax import ExpectimaxSearcher


def gen_data(model_path, out_path, n_games=200, depth=2, max_empties=10,
             sample_every=1, seed=42, verbose=True):
    """生成推理数据集。

    参数:
        model_path  : 教师模型（n-tuple checkpoint）
        out_path    : 输出 .npz 路径
        n_games     : 走多少局
        depth       : 搜索深度（2 = 2-ply）
        sample_every: 每 N 步采样一次（控制数据量）
        seed        : 随机种子
    """
    net = NTupleNetwork(PATTERNS_8)
    net.load(model_path)
    searcher = ExpectimaxSearcher(net, depth=depth, max_empties=max_empties,
                                  use_tt=True)

    rng = np.random.default_rng(seed)
    boards, values, best_actions, stages, scores_after = [], [], [], [], []
    game_scores = []
    t0 = time.time()
    n_samples = 0

    for gi in range(n_games):
        b = new_board(rng)
        step = 0
        while True:
            res = greedy_action(net, b)
            if res is None:
                break

            # ---- 采样：用 2-ply 搜索标注该局面 ----
            if step % sample_every == 0:
                bs_a, sc_a, moved_a = move_all(b)
                vals = np.full(4, -np.inf, dtype=np.float32)
                for a in range(4):
                    if moved_a[a]:
                        nb = bs_a[a]
                        vals[a] = float(sc_a[a]) + searcher._chance(nb, depth)
                valid = np.isfinite(vals)
                if valid.sum() >= 2:
                    best = int(np.argmax(np.where(valid, vals, -np.inf)))
                    boards.append(b.copy())
                    values.append(vals)
                    best_actions.append(best)
                    stages.append(step)
                    n_samples += 1

            # ---- 用教师走一步（保持数据分布接近实战）----
            _, nb, _, _ = res
            b = spawn(nb.copy(), rng)
            step += 1

        # 该局最终得分（作为额外标签）
        sc = 0
        for arr, mv in zip(boards[-step:] if step else [], []):
            pass
        game_scores.append(step)

        if verbose and (gi + 1) % max(1, n_games // 10) == 0:
            el = time.time() - t0
            print(f"  局 {gi+1}/{n_games} | 样本 {n_samples:,} | "
                  f"{el:.0f}s ({n_samples/max(1,el):.1f} 样本/秒)", flush=True)

    el = time.time() - t0
    print(f"\n生成完成: {n_samples:,} 样本, 用时 {el/60:.1f} 分钟 "
          f"({n_samples/max(1,el):.1f} 样本/秒)")

    # 保存
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        boards=np.asarray(boards, dtype=np.uint8),
        values=np.asarray(values, dtype=np.float32),
        best_actions=np.asarray(best_actions, dtype=np.int8),
        stages=np.asarray(stages, dtype=np.int16),
        meta=np.array([depth, n_games, n_samples], dtype=np.int64),
    )
    os.replace(tmp, out_path)
    print(f"数据已保存: {out_path} ({os.path.getsize(out_path)/1048576:.1f} MB)")
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="v4 推理数据集生成")
    p.add_argument("--model", default="/opt/2048ai/exp_hp/models/best.npz")
    p.add_argument("--out", default="/opt/2048ai/v4_data/train_2ply.npz")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    gen_data(args.model, args.out, args.games, args.depth,
             sample_every=args.sample_every, seed=args.seed)
