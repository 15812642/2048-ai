# -*- coding: utf-8 -*-
"""
probe2.py - 修正版瓶颈分析

修正前一版的两个问题：
  1. 测试局面应覆盖【全阶段】（开局/中局/终盘）, 而非只有终盘
  2. 分别统计各阶段的速度与动作价值分布

同时定位"转置表命中率 0"的原因（2-ply 结构下是否正常）。
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

from ntuple import (PATTERNS_8, NTupleNetwork, greedy_action, new_board, spawn)
from expectimax import ExpectimaxSearcher


def make_stage(net, rng, target_steps):
    """生成指定阶段的对局局面。"""
    b = new_board(rng)
    for _ in range(target_steps):
        r = greedy_action(net, b)
        if r is None:
            return None
        b = spawn(r[1].copy(), rng)
    return b


def main(model_path):
    net = NTupleNetwork(PATTERNS_8)
    net.load(model_path)
    print(f"模型: {model_path} (σ={net.lut.std():.1f})\n")

    rng = np.random.default_rng(11)
    stages = {
        "开局 (~100步)": [make_stage(net, rng, 100) for _ in range(15)],
        "中局 (~300步)": [make_stage(net, rng, 300) for _ in range(15)],
        "后期 (~600步)": [make_stage(net, rng, 600) for _ in range(15)],
        "终盘 (~900步)": [make_stage(net, rng, 900) for _ in range(15)],
    }

    print("=== 各阶段特征 ===")
    print("  %-16s | %8s | %10s | %s" % ("阶段", "空位数", "最大方块", "说明"))
    print("  " + "-" * 60)
    for name, boards in stages.items():
        bs = [b for b in boards if b is not None]
        if not bs:
            continue
        e = np.mean([(b == 0).sum() for b in bs])
        mt = np.mean([int(np.max(np.where(b > 0, np.left_shift(1, b), 0))) for b in bs])
        note = "chance分支≈%d" % (2 * e)
        print("  %-16s | %8.1f | %10.0f | %s" % (name, e, mt, note))

    print("\n=== 搜索速度（depth=2, max_empties=10）===")
    searcher = ExpectimaxSearcher(net, depth=2, use_tt=True)
    print("  %-16s | %10s | %8s | %s" % ("阶段", "ms/局面", "节点数", "TT命中"))
    print("  " + "-" * 60)
    for name, boards in stages.items():
        bs = [b for b in boards if b is not None][:10]
        t0 = time.time()
        for b in bs:
            searcher.best_action(b)
        el = (time.time() - t0) / len(bs)
        print("  %-16s | %10.2f | %8.0f | %8.0f" % (
            name, el * 1000, searcher.nodes, searcher.hits))

    print("\n=== 动作价值分布（各阶段, 2-ply）===")
    print("  %-16s | %12s | %12s | %s" % ("阶段", "最优-次优(中位)", "相对差(中位)", "4动作占比"))
    print("  " + "-" * 66)
    for name, boards in stages.items():
        bs = [b for b in boards if b is not None]
        margins, rels, n4 = [], [], 0
        for b in bs[:8]:
            from ntuple import move_all
            boards_a, scores_a, moved_a = move_all(b)
            vals = []
            for a in range(4):
                if not moved_a[a]:
                    vals.append(float("-inf"))
                else:
                    nb = boards_a[a]
                    vals.append(float(scores_a[a]) + searcher._chance(nb, 2))
            valid = sorted((v for v in vals if v > float("-inf")), reverse=True)
            if len(valid) >= 2:
                m = valid[0] - valid[1]
                margins.append(m)
                rels.append(m / max(1e-9, abs(valid[0])))
            if len(valid) == 4:
                n4 += 1
        if margins:
            print("  %-16s | %12.0f | %11.2f%% | %d/%d" % (
                name, np.median(margins), np.median(rels) * 100, n4, len(bs[:8])))

    print("\n=== 关键结论 ===")
    print("  1) TT 在 2-ply 下命中率为 0 是正常的：")
    print("     2-ply 只有一个 max 层, 所有 chance 子状态互不相同, 无重复可复用")
    print("     -> TT 只在 3-ply 及以上才产生价值")
    print("  2) 空位越多, chance 分支越大, 搜索越慢（开局最慢）")
    print("  3) 若某阶段'最优-次优'差距小, 说明该阶段决策本身容错高（剪枝安全）")


if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else "/opt/2048ai/exp_hp/models/best.npz"
    main(mp)
