# -*- coding: utf-8 -*-
"""
bench_phone.py - 手机端性能基准（对比服务器）

用途
----
在手机上直接运行，测出真实训练吞吐，与服务器对比：
    - 服务器（AMD EPYC 7K62, 4 核切片）: 约 10-15 局/秒（3 workers）
    - 若手机（骁龙 8 Gen 2, 8 核）明显更快，值得迁移

测试项
------
1. 单核纯计算（numpy 查表吞吐）
2. 多核扩展性（1/2/4 worker）
3. 实际训练吞吐（10 秒小样本）

用法
----
    cp bench_phone.py ~ && cd ~
    python bench_phone.py            # 自动探测最佳 worker 数
    python bench_phone.py 4          # 指定 worker 数
"""

import os
import sys
import time

import numpy as np

# 自动探测 ntuple.py 位置
for _c in (os.path.dirname(os.path.abspath(__file__)), ".", "/opt/2048ai",
           os.path.expanduser("~/2048_ai"), os.path.expanduser("~")):
    if os.path.isfile(os.path.join(_c, "ntuple.py")):
        sys.path.insert(0, _c)
        break
else:
    print("✗ 找不到 ntuple.py，请先把项目拷到手机")
    sys.exit(1)


def cpu_info():
    """打印 CPU 信息（手机上是 ARM 核心）。"""
    try:
        with open("/proc/cpuinfo") as f:
            lines = [l for l in f if l.startswith(("processor", "model name",
                                                   "CPU part", "Hardware"))]
        print("  CPU 信息:")
        seen = set()
        for l in lines[:24]:
            l = l.strip()
            if l not in seen:
                print("   ", l)
                seen.add(l)
    except Exception:
        pass
    import multiprocessing
    print(f"  可用核心: {multiprocessing.cpu_count()}")


def bench_single_core(net, seconds=5):
    """单核吞吐：纯查表评估。"""
    from ntuple import new_board, spawn, greedy_action
    rng = np.random.default_rng(0)
    b = new_board(rng)
    for _ in range(50):
        r = greedy_action(net, b)
        if r is None:
            break
        b = spawn(r[1].copy(), rng)

    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        net.evaluate(b)
        n += 1
    el = time.time() - t0
    return n / el


def bench_game_throughput(net, seconds=8):
    """单核实际对局吞吐（跑完整局）。"""
    from ntuple import new_board, spawn, greedy_action
    rng = np.random.default_rng(1)
    t0 = time.time()
    games = 0
    steps = 0
    while time.time() - t0 < seconds:
        b = new_board(rng)
        while True:
            r = greedy_action(net, b)
            if r is None:
                break
            b = spawn(r[1].copy(), rng)
            steps += 1
        games += 1
    el = time.time() - t0
    return games / el, steps / el


def main():
    print("=" * 62)
    print("2048 训练性能基准（手机端）")
    print("=" * 62)
    cpu_info()
    print()

    from ntuple import PATTERN_SETS, NTupleNetwork

    # 用 5 图案集（48MB，手机内存友好）
    net = NTupleNetwork(PATTERN_SETS["5"], v_init=0.0)
    print(f"  测试网络: 5 图案集 | {net.size_mb():.0f} MB | {net.K} 次查表/评估")
    print()

    print("【1】单核评估吞吐")
    eval_rate = bench_single_core(net, 5)
    print(f"  {eval_rate:,.0f} 次评估/秒  ({1e6/eval_rate:.1f} µs/次)")
    print(f"  参考: 服务器约 30,000-55,000 次/秒")
    print()

    print("【2】单核对局吞吐（实际训练场景）")
    g_rate, s_rate = bench_game_throughput(net, 8)
    print(f"  {g_rate:.1f} 局/秒 | {s_rate:,.0f} 步/秒")
    print(f"  参考: 服务器单 worker 约 30 局/秒")
    print()

    # 多核测试
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if nw <= 0:
        import multiprocessing
        nw = max(1, min(4, multiprocessing.cpu_count() - 1))
    print(f"【3】多 worker 并行（{nw} 进程，10 秒样本）")
    try:
        from train_nt_parallel import ParallelNTupleTrainer
        from train_nt import NTupleConfig
        import logging
        logging.getLogger("2048").handlers = []
        logging.getLogger("2048").addHandler(logging.NullHandler())

        cfg = NTupleConfig(episodes=10 ** 9, out_dir="/tmp/bench_out",
                           pattern_set="5", eval_every=0, ckpt_every=0,
                           v_init=0.0, alpha=0.32)
        t = ParallelNTupleTrainer(cfg, n_workers=nw)
        t._start_time = time.time()
        t.lut = t.shm.create()
        t._start_workers()

        time.sleep(10)
        t._drain_stats()
        rate = t._rate()
        games = t.episode
        t._stop_workers()
        t.shm.close(unlink=True)
        print(f"  {rate:.1f} 局/秒（{nw} workers, {games} 局样本）")
        print(f"  参考: 服务器 3 workers 约 10-15 局/秒")
        print()
        print("=" * 62)
        if rate > 15:
            print(f"✓ 手机吞吐 {rate:.1f} 局/秒 —— 优于服务器！建议手机跑训练")
        elif rate > 8:
            print(f"○ 手机吞吐 {rate:.1f} 局/秒 —— 与服务器相当，可作补充算力")
        else:
            print(f"△ 手机吞吐 {rate:.1f} 局/秒 —— 低于服务器，建议只作演示/评测")
        print("=" * 62)
    except Exception as e:
        print(f"  并行测试失败: {type(e).__name__}: {e}")
        print("  （若为 /dev/shm 缺失，属正常 —— 代码会自动回退 memmap）")


if __name__ == "__main__":
    main()
