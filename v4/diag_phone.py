#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手机训练性能诊断"""
import os, sys, time
import numpy as np

def cpu_freqs():
    """读当前 CPU 频率（检测降频）"""
    out = []
    for i in range(8):
        mx = cur = None
        for f, key in ((f"/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq", "max"),
                       (f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq", "cur")):
            try:
                with open(f) as fh:
                    v = int(fh.read().strip()) // 1000
                if key == "max": mx = v
                else: cur = v
            except Exception:
                pass
        if mx: out.append((i, mx, cur))
    return out

def cpu_load():
    """读 CPU 使用率（1 秒采样）"""
    def snap():
        with open("/proc/stat") as f:
            parts = [l for l in f if l.startswith("cpu")]
        res = []
        for p in parts[1:9]:
            v = [int(x) for x in p.split()[1:]]
            idle = v[3] + v[4]
            total = sum(v)
            res.append((idle, total))
        return res
    a = snap(); time.sleep(1); b = snap()
    out = []
    for i, ((i1, t1), (i2, t2)) in enumerate(zip(a, b)):
        dt = t2 - t1
        usage = 100 * (1 - (i2 - i1) / dt) if dt > 0 else 0
        out.append(round(usage))
    return out

def main():
    print("=" * 66)
    print("  手机训练性能诊断")
    print("=" * 66)
    
    print("\n【1】CPU 频率（检测降频）")
    freqs = cpu_freqs()
    if freqs:
        for i, mx, cur in freqs:
            tag = "大核" if mx > 3000 else ("中核" if mx > 2500 else "小核")
            ratio = f"{100*cur/mx:.0f}%" if cur else "–"
            print(f"    cpu{i} ({tag}): 最大 {mx}MHz | 当前 {cur or '?'}MHz ({ratio})")
    else:
        print("    （读不到频率信息）")
    
    print("\n【2】CPU 使用率（1秒采样）")
    loads = cpu_load()
    for i, l in enumerate(loads):
        bar = "█" * (l // 5)
        print(f"    cpu{i}: {l:3d}% {bar}")
    avg = sum(loads) / len(loads)
    print(f"    平均: {avg:.0f}%")
    busy = [i for i, l in enumerate(loads) if l > 70]
    print(f"    高负载核心: {busy if busy else '无'}")
    
    print("\n【3】单核基准（10秒）")
    # 探测 ntuple 位置
    for c in (".", os.path.expanduser("~/2048"), os.path.expanduser("~/2048_phone"),
              "/workspace/2048_ai"):
        if os.path.isfile(os.path.join(c, "ntuple.py")):
            sys.path.insert(0, c); break
    else:
        print("    ✗ 找不到 ntuple.py")
        return
    
    from ntuple import PATTERN_SETS, NTupleNetwork, new_board, spawn, greedy_action
    net = NTupleNetwork(PATTERN_SETS["8"], v_init=0.0)
    rng = np.random.default_rng(0)
    b = new_board(rng)
    for _ in range(60):
        r = greedy_action(net, b)
        if r is None: break
        b = spawn(r[1].copy(), rng)
    t0 = time.time(); N = 0
    while time.time() - t0 < 10:
        net.evaluate(b); N += 1
    el = time.time() - t0
    rate = N / el
    print(f"    评估速度: {rate:,.0f} 次/秒")
    print(f"    参考: 工作区实测 63,579 次/秒")
    if rate < 30000:
        print(f"    ⚠️ 明显偏慢，可能降频或有其他进程抢占")
    
    print("\n【4】并行测试（不绑定 CPU，15秒）")
    import logging
    logging.getLogger("2048").handlers = []
    logging.getLogger("2048").addHandler(logging.NullHandler())
    from train_nt import NTupleConfig
    from train_nt_parallel import ParallelNTupleTrainer
    for nw in (3, 5):
        cfg = NTupleConfig(episodes=10**9, out_dir=f"/tmp/diag_{nw}",
                           pattern_set="8", eval_every=0, ckpt_every=0,
                           v_init=0.0, alpha=0.16)
        t = ParallelNTupleTrainer(cfg, n_workers=nw)
        t._start_time = time.time()
        t.lut = t.shm.create()
        t._start_workers()
        time.sleep(15)
        t._drain_stats()
        print(f"    {nw} workers: {t._rate():.1f} 局/秒")
        t._stop_workers(); t.shm.close(unlink=True)
        time.sleep(2)
    
    print("\n【结论】")
    print("    若单核基准正常但并行慢 → CPU 绑定问题，去掉绑定")
    print("    若单核基准也慢 → 降频或有其他 App 抢占")
    print("=" * 66)

if __name__ == "__main__":
    main()
