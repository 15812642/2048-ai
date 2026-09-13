# -*- coding: utf-8 -*-
"""
probe_bottleneck.py - 搜索瓶颈分析（v4 设计依据）

分析 2-ply expectimax 的耗时构成，找出真正的优化空间：
    - max 节点（动作分支）: 4 个
    - chance 节点（空位 x 2/4）: 空位数 x 2 个
    哪个是瓶颈？减少哪个最有效？

同时测试三个优化手段的实际收益：
    A. chance 采样数 max_empties（10 -> 4/6）
    B. 转置表命中率
    C. max 节点剪枝（top-k）
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


def make_midgame(net, rng, lo=80, hi=200):
    b = new_board(rng)
    for _ in range(rng.integers(lo, hi)):
        r = greedy_action(net, b)
        if r is None:
            break
        b = spawn(r[1].copy(), rng)
    return b


def bench(net, depth, max_empties, boards, tag):
    """测速度 + 统计节点数"""
    searcher = ExpectimaxSearcher(net, depth=depth, max_empties=max_empties,
                                  use_tt=True)
    nodes_list, hits_list = [], []
    t0 = time.time()
    for b in boards:
        a = searcher.best_action(b)
        nodes_list.append(searcher.nodes)
        hits_list.append(searcher.hits)
    el = time.time() - t0
    n = len(boards)
    print(f"  {tag:34s} | {el/n*1000:8.2f} ms/局面 | "
          f"节点 {np.mean(nodes_list):7.0f} | TT命中 {np.mean(hits_list):7.0f} "
          f"({np.mean(hits_list)/max(1,np.mean(nodes_list)+np.mean(hits_list))*100:4.0f}%)")
    return el / n


def main(model_path):
    net = NTupleNetwork(PATTERNS_8)
    net.load(model_path)
    print(f"模型: {model_path} (σ={net.lut.std():.1f})\n")

    rng = np.random.default_rng(11)
    boards = [make_midgame(net, rng) for _ in range(24)]

    # 空位分布（决定 chance 分支数）
    empties = [int((b == 0).sum()) for b in boards]
    print(f"=== 测试局面空位分布 ===")
    print(f"  平均 {np.mean(empties):.1f} 个 | 范围 {min(empties)}-{max(empties)}")
    print(f"  → chance 节点分支 = 空位数 x 2 = 平均 {np.mean(empties)*2:.0f} 个\n")

    print("=== 优化 A: chance 采样数（max_empties）===")
    base = bench(net, 2, 10, boards, "max_empties=10 (默认, 全采样)")
    bench(net, 2, 8, boards, "max_empties=8")
    bench(net, 2, 6, boards, "max_empties=6")
    bench(net, 2, 4, boards, "max_empties=4")

    print("\n=== 优化 C: 深度对比（固定 max_empties=8）===")
    for d in (1, 2):
        bench(net, d, 8, boards, f"depth={d}")

    print("\n=== 参考文献：节点数理论值 ===")
    e = np.mean(empties)
    print(f"  2-ply 展开 ≈ 4动作 x (2e 个 chance 子) x 4动作 x (2e 个 chance 子)")
    print(f"              ≈ 4 x {2*e:.0f} x 4 x {2*e:.0f} = {4*2*e*4*2*e:,.0f} 个叶子（未剪枝）")


if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else "/opt/2048ai/exp_hp/models/best.npz"
    main(mp)
