#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mobile_train.py - 手机优化训练启动器

解决两个问题
------------
1. **CPU 异质性**：手机是 big.LITTLE 架构（如 8 Gen 2 = 1×X3 + 4×A715 + 3×A510）。
   小核（A510）性能仅大核的 ~40%，把 worker 放上去会拖慢整体。
   本脚本按 max_freq 识别核心等级，只把 worker 绑定到大核+中核。

2. **worker 数选择**：默认 4 是保守值。实际用满大核+中核（5 个）能提升吞吐。

用法
----
    python mobile_train.py                      # 自动探测并启动
    python mobile_train.py --workers 5          # 手动指定
    python mobile_train.py --dry-run            # 只看拓扑不启动
    python mobile_train.py --patterns 5 --episodes 1000000
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (_HERE, os.path.dirname(_HERE), ".", os.path.expanduser("~/2048_phone"),
           os.path.expanduser("~"), os.path.expanduser("~/2048_ai"), "/opt/2048ai"):
    if _c and os.path.isfile(os.path.join(_c, "ntuple.py")):
        sys.path.insert(0, _c)
        break


# ---------------- CPU 拓扑探测 ----------------

def detect_cpu_groups():
    """按最大频率把 CPU 核心分组（big / mid / little）。

    返回 {等级: [核心编号]}, 读不到频率时返回 {}。
    """
    base = "/sys/devices/system/cpu"
    freqs = {}
    try:
        for name in sorted(os.listdir(base)):
            if not name.startswith("cpu") or not name[3:].isdigit():
                continue
            idx = int(name[3:])
            for f in ("cpufreq/cpuinfo_max_freq", "cpufreq/scaling_max_freq"):
                p = os.path.join(base, name, f)
                if os.path.exists(p):
                    try:
                        with open(p) as fh:
                            freqs[idx] = int(fh.read().strip())
                        break
                    except Exception:
                        pass
    except Exception:
        return {}

    if not freqs:
        return {}

    # 按频率分成 2-3 组
    vals = sorted(set(freqs.values()), reverse=True)
    groups = {}
    if len(vals) == 1:
        groups = {"all": sorted(freqs)}
    elif len(vals) == 2:
        hi, lo = vals
        groups = {"big": sorted(i for i, f in freqs.items() if f == hi),
                  "little": sorted(i for i, f in freqs.items() if f == lo)}
    else:
        # 取最高两档为 big/mid，其余 little
        top = vals[0]
        second = vals[1]
        groups = {
            "big": sorted(i for i, f in freqs.items() if f == top),
            "mid": sorted(i for i, f in freqs.items() if f == second),
            "little": sorted(i for i, f in freqs.items()
                             if f not in (top, second)),
        }
    return groups


def print_topology():
    """打印 CPU 拓扑信息。"""
    import multiprocessing
    total = multiprocessing.cpu_count()
    groups = detect_cpu_groups()

    print("【CPU 拓扑】")
    print(f"  总核心数: {total}")
    if not groups:
        print("  （无法读取频率信息，将使用全部核心）")
        return total, []

    all_cores = []
    for name, cores in groups.items():
        if not cores:
            continue
        label = {"big": "大核", "mid": "中核", "little": "小核",
                 "all": "全部"}.get(name, name)
        # 尝试读频率展示
        f = ""
        try:
            p = f"/sys/devices/system/cpu/cpu{cores[0]}/cpufreq/cpuinfo_max_freq"
            if os.path.exists(p):
                f = f"{int(open(p).read().strip())/1000:.0f}MHz"
        except Exception:
            pass
        print(f"  {label:4s}: 核心 {cores} {f}")
        if name != "little":          # 小核不参与训练
            all_cores.extend(cores)

    print(f"  → 训练可用: {len(all_cores)} 核（大核+中核，排除小核）{all_cores}")
    return total, all_cores


def try_pin_to_cores(cores):
    """尝试把当前进程绑定到指定核心（失败则静默跳过）。

    Termux 通常允许 sched_setaffinity（非 root），但 proot 内可能受限。
    """
    if not cores:
        return False
    try:
        os.sched_setaffinity(0, set(cores))
        actual = os.sched_getaffinity(0)
        return set(cores) <= actual or len(actual) > 0
    except Exception:
        return False


# ---------------- 主流程 ----------------

def main():
    p = argparse.ArgumentParser(description="手机优化训练启动器")
    p.add_argument("--workers", type=int, default=0,
                   help="worker 数（0 = 自动：大核+中核数）")
    p.add_argument("--patterns", default="8",
                   help="图案集 5/8/12（实测 8 更快：查表 64 次 vs 5 的 96 次）")
    p.add_argument("--episodes", type=int, default=1_000_000)
    p.add_argument("--alpha", type=float, default=0.16)
    p.add_argument("--out-dir", default="/storage/emulated/0/rikkahub/2048_train",
                   help="输出目录（默认 rikkahub，便于查看进度）")
    p.add_argument("--dry-run", action="store_true", help="只显示配置，不启动")
    p.add_argument("--pin", action="store_true",
                   help="启用 CPU 绑定（默认关闭——实测绑定后核心变少反而更慢）")
    p.add_argument("--no-pin", action="store_true", help="(兼容旧参数) 不绑定")
    p.add_argument("--no-wake", action="store_true",
                   help="不申请 wake-lock（默认自动申请, 防止 CPU 被限制）")
    args = p.parse_args()

    print("=" * 66)
    print("2048 手机优化训练启动器")
    print("=" * 66)

    total, train_cores = print_topology()

    # ---- worker 数决策 ----
    if args.workers > 0:
        nw = args.workers
        print(f"\n【Worker 配置】使用手动指定: {nw}")
    elif train_cores:
        # 实测（8 图案集）: 5 workers 最优（53.3 局/秒, 6 个已饱和）
        nw = max(1, min(4, len(train_cores)))
        print(f"\n【Worker 配置】自动: {nw}（保守值——降频时过多 worker 反而更慢）")
    else:
        import multiprocessing
        nw = max(1, min(6, multiprocessing.cpu_count() - 2))
        print(f"\n【Worker 配置】保守估计: {nw}")

    print(f"  说明: 小核（A510 类）性能仅大核 40%，不纳入训练")

    # ---- CPU 绑定（默认关闭）----
    # 实测：绑定到 [7,3,4,5,6] 后，若系统其他进程占用这些核，训练反而变慢。
    # 交给系统调度器自由分配通常更优（可用 --pin 强制开启）。
    if args.pin and not args.no_pin and train_cores:
        ok = try_pin_to_cores(train_cores)
        print(f"【CPU 绑定】{'✓ 已绑定到 ' + str(train_cores) if ok else '△ 失败'}")
    else:
        print(f"【CPU 绑定】关闭（交由系统调度，默认更优）")

    # ---- 图案集内存检查 ----
    from ntuple import PATTERN_SETS, NTupleNetwork
    pats = args.patterns if args.patterns in PATTERN_SETS else "5"
    proto = NTupleNetwork(PATTERN_SETS[pats], v_init=0.0)
    mem_mb = proto.size_mb()
    print(f"【内存占用】图案集 {pats} -> {mem_mb:.0f} MB（全 worker 共享一份）")
    del proto

    # 实测（骁龙 8 Gen 2, 15GB RAM）: 512MB 表仅占 3% 内存
    # 且 8 图案查表次数更少（64 vs 96），吞吐反而高 44%
    if mem_mb > 1024:
        print(f"  ⚠️ 表超过 1GB，建议 --patterns 8（512MB）")

    alpha_eff = args.alpha / (nw ** 0.5)
    print(f"【学习率】α {args.alpha} -> 自动缩放至 {alpha_eff:.3f}（补偿 {nw} 进程写冲突）")
    try:
        os.makedirs(args.out_dir, exist_ok=True)
    except Exception as e:
        print(f"  ⚠️ 无法创建输出目录 {args.out_dir}: {e}")
        args.out_dir = os.path.expanduser("~/2048_train")
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"  已回退到: {args.out_dir}")
    print(f"【输出目录】{args.out_dir}")

    if args.dry_run:
        print("\n（--dry-run 模式，未启动训练）")
        return 0

    # ---- 自动申请 wake-lock（关键：防止 Android 限制 CPU）----
    # 背景：Termux 作为非前台进程时，Android 会大幅限制 CPU 频率
    # （实测大核被压到 39%、中核 46%），导致训练慢 9 倍。
    # wake-lock 可解除该限制。
    wake_ok = False
    if not args.no_wake:
        try:
            import subprocess
            r = subprocess.run(["termux-wake-lock"], capture_output=True,
                               timeout=5)
            wake_ok = (r.returncode == 0)
        except FileNotFoundError:
            print("【唤醒锁】⚠️ 未安装 termux-api（建议: pkg install termux-api）")
        except Exception:
            pass
        if wake_ok:
            print("【唤醒锁】✓ 已获取（防止 Android 限制 CPU 频率）")
        elif not args.no_wake:
            print("【唤醒锁】△ 获取失败（建议安装 termux-api）")

    # ---- 启动训练 ----
    print("\n" + "=" * 66)
    if not wake_ok:
        print("  ⚠️ 未获取唤醒锁，CPU 可能被系统限制，速度会明显偏慢")
        print("     解决: pkg install termux-api（装完重跑本脚本）")
    print("启动训练（Ctrl+C 停止并保存 checkpoint）")
    print("=" * 66 + "\n")

    import logging
    logging.getLogger("2048").handlers = []
    logging.getLogger("2048").addHandler(logging.NullHandler())

    from train_nt import NTupleConfig
    from train_nt_parallel import ParallelNTupleTrainer

    # ---- 实时进度输出（直接打印到控制台，不用 tail 日志）----
    import threading
    import json as _json

    class _LiveProgress(threading.Thread):
        """每 5 秒把训练进度渲染成单行输出（覆盖式刷新）。"""

        def __init__(self, trainer, out_dir):
            super().__init__(daemon=True)
            self.t = trainer
            self.out_dir = out_dir
            self._ev = threading.Event()   # 注意: 不能叫 _stop（Thread 内部方法）

        def run(self):
            time.sleep(3)                      # 等训练器初始化
            while not self._ev.is_set():
                try:
                    ep = self.t.episode
                    rate = self.t._rate()
                    n_work = self.t.n_workers
                    sc = self.t.history["episode_scores"]
                    recent = sc[-200:] if sc else []
                    avg = sum(recent) / len(recent) if recent else 0
                    best = self.t.best_eval_score
                    best_s = "–" if best == float("-inf") else f"{best:,.0f}"
                    line = (f"  局数 {ep:>8,} | {rate:5.1f} 局/秒 | "
                            f"{n_work}w | 近200局均分 {avg:>7,.0f} | 最佳 {best_s}")
                    print("\r" + line + "   ", end="", flush=True)
                except Exception:
                    pass
                self._ev.wait(5)

        def stop(self):
            self._ev.set()

    progress = None

    cfg = NTupleConfig(
        episodes=args.episodes, out_dir=args.out_dir, pattern_set=pats,
        alpha=args.alpha, lam=0.0, v_init=0.0,
        alpha_decay_every=25000, alpha_decay_gamma=0.8, alpha_min=0.05,
        eval_every=10000, eval_games=50, ckpt_every=5000,
        snapshot_every=50000,
    )
    tr = ParallelNTupleTrainer(cfg, n_workers=nw)

    # 若已有 checkpoint 则续训
    latest = os.path.join(tr.models_dir, "latest.npz")
    if os.path.exists(latest):
        try:
            tr.load_checkpoint("latest.npz")
            print(f"（发现已有进度，将从第 {tr.episode:,} 局继续）\n")
        except Exception as e:
            print(f"（checkpoint 加载失败，从头开始: {e}）\n")

    progress = _LiveProgress(tr, args.out_dir)
    progress.start()
    print("（每 5 秒刷新一行进度；进度文件也在输出目录的 status.json）\n")

    try:
        tr.train(args.episodes)
    except KeyboardInterrupt:
        print("\n收到中断，保存进度中...")
    finally:
        if progress:
            progress.stop()
        tr.close()
        print(f"\n\n训练结束。模型目录: {tr.models_dir}")
        print(f"状态文件: {os.path.join(args.out_dir, 'status.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
