# -*- coding: utf-8 -*-
"""
probe_pruning.py - 剪枝可行性探测（v4 前置实验）

目标：回答"用策略网络只搜 top-k 动作，会损失多少准确性？"

方法：
    1. 用当前最强 n-tuple 模型 + 2-ply expectimax 走若干局面
    2. 每个局面计算【全部合法动作】的 2-ply 搜索价值
    3. 统计：
       - 最优动作与次优动作的价值差（margin）分布
       - 若只保留 top-2 再全搜，最终选择与全搜是否一致
       - 剪枝掉的动作里"本可能是最优"的比例（剪枝错误率）

判据：
    - 若 margin 大（最优明显领先）→ 剪枝安全，可做
    - 若 margin 小（常并列）→ 剪枝风险高，需要更精细的策略
"""

import sys
import time
from collections import Counter

import numpy as np

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.dirname(_HERE), _HERE, "/opt/2048ai", "/workspace/2048_ai"):
    if os.path.isfile(os.path.join(_cand, "ntuple.py")):
        sys.path.insert(0, _cand)
        break

from ntuple import (PATTERNS_8, NTupleNetwork, decode_board, greedy_action,
                    move_all, new_board, spawn)
from expectimax import ExpectimaxSearcher


def action_values(net, searcher, board):
    """返回 [v(action) for action in 0..3]，非法动作给 -inf。"""
    boards, scores, moved = move_all(board)
    vals = []
    for a in range(4):
        if not moved[a]:
            vals.append(float("-inf"))
            continue
        nb = boards[a]
        sc = float(scores[a])
        if searcher is None:
            vals.append(sc + net.evaluate(nb))
        else:
            vals.append(sc + searcher._chance(nb, searcher.depth))
    return vals


def main(model_path, n_boards=300, depth=2):
    net = NTupleNetwork(PATTERNS_8)
    net.load(model_path)
    print(f"模型: {model_path}")
    print(f"  σ={net.lut.std():.2f} | K={net.K}")

    searcher = ExpectimaxSearcher(net, depth=depth, max_empties=10)
    rng = np.random.default_rng(7)

    margins = []            # 最优 - 次优
    rel_margins = []        # (最优-次优)/|最优|
    top1_win = 0            # 最优动作明显领先（> 阈值）的样本数
    ties = 0                # 并列（差距 < 1% 的样本）
    n_valid2 = 0            # 至少 2 个合法动作的局面数
    branch_counts = Counter()

    t0 = time.time()
    collected = 0
    while collected < n_boards:
        b = new_board(rng)
        # 快速推进到中局（空位少, 更能反映实战决策压力）
        for _ in range(rng.integers(60, 160)):
            r = greedy_action(net, b)
            if r is None:
                break
            b = spawn(r[1].copy(), rng)
        if b.sum() == 0:
            continue

        vals = action_values(net, searcher, b)
        valid = [(a, v) for a, v in enumerate(vals) if v > float("-inf")]
        branch_counts[len(valid)] += 1
        if len(valid) < 2:
            continue

        svals = sorted((v for _, v in valid), reverse=True)
        m = svals[0] - svals[1]
        margins.append(m)
        rel_margins.append(m / max(1e-9, abs(svals[0])))
        if rel_margins[-1] > 0.01:
            top1_win += 1
        else:
            ties += 1
        n_valid2 += 1
        collected += 1

    el = time.time() - t0
    margins = np.asarray(margins)
    rel = np.asarray(rel_margins)

    print(f"\n=== 剪枝可行性统计（{depth}-ply 搜索, {n_valid2} 个局面, {el:.0f}s）===")
    print(f"  平均耗时: {el / max(1, n_valid2) * 1000:.1f} ms/局面")
    print()
    print("  合法动作数分布:")
    for k, v in sorted(branch_counts.items()):
        print(f"    {k} 个: {v} ({v / sum(branch_counts.values()) * 100:.1f}%)")
    print()
    print("  最优 vs 次优 价值差（margin）:")
    print(f"    平均 {margins.mean():,.0f} | 中位 {np.median(margins):,.0f} | "
          f"p10 {np.percentile(margins, 10):,.0f} | p90 {np.percentile(margins, 90):,.0f}")
    print()
    print("  相对差（margin / |最优值|）:")
    print(f"    平均 {rel.mean() * 100:.2f}% | 中位 {np.median(rel) * 100:.2f}%")
    print(f"    最优明显领先(>1%): {top1_win} ({top1_win / n_valid2 * 100:.1f}%)")
    print(f"    接近并列(<=1%)   : {ties} ({ties / n_valid2 * 100:.1f}%)")
    print()
    print("  【剪枝判据】")
    if np.median(rel) > 0.02:
        print(f"    ✓ 中位相对差 {np.median(rel) * 100:.1f}% > 2% —— "
              f"最优动作通常明显领先，top-2 剪枝安全")
    else:
        print(f"    ⚠ 中位相对差 {np.median(rel) * 100:.1f}% <= 2% —— "
              f"动作价值接近，剪枝风险较高，需谨慎")


if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else "/opt/2048ai/exp_hp/models/best.npz"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    d = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    main(mp, n, d)
