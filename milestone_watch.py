#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
milestone_watch.py - 里程碑监控：exp_hp 的 4096 达成率 >= 90% 时自动跑 3-ply 评测

由 systemd timer 每 20 分钟触发一次：
  1. 读 exp_hp/status.json 的训练评估结果
  2. 若最新 4096 达成率 >= 88%（略低阈值, 因 100 局评估有 ±4% 噪声）
     -> 先跑 200 局 1-ply 复核（降低噪声）
     -> 若复核确认 >= 90%, 跑 10 局 3-ply 期望搜索评测（终极测试）
     -> 结果写入 milestone_report.json, 并创建 DONE 标记防止重复触发
  3. 每次运行都追加一行进度日志（便于查看趋势）
"""

import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
STATUS = os.path.join(BASE, "exp_hp", "status.json")
REPORT = os.path.join(BASE, "milestone_report.json")
DONE_FLAG = os.path.join(BASE, "milestone_4096_DONE")
LOG = os.path.join(BASE, "milestone_watch.log")

TRIGGER_TH = 0.88        # 触发复核的阈值（含噪声余量）
CONFIRM_TH = 0.90        # 复核确认阈值
CONFIRM_GAMES = 200      # 复核用的局数（降低统计噪声）
FINAL_GAMES = 10         # 3-ply 终测局数
FINAL_DEPTH = 3          # 3-ply


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_status():
    try:
        with open(STATUS, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def run_eval(games, depth, tag):
    """调用评测脚本，返回统计字典（失败返回 None）。"""
    code = f'''
import sys, json
sys.path.insert(0, "{BASE}")
import numpy as np
from ntuple import PATTERNS_8, NTupleNetwork, decode_board, new_board, spawn, move_all, greedy_action
from expectimax import ExpectimaxSearcher

net = NTupleNetwork(PATTERNS_8)
net.load("{BASE}/exp_hp/models/best.npz")
rng = np.random.default_rng(20240912)
searcher = ExpectimaxSearcher(net, depth={depth}) if {depth} > 1 else None

scores, tiles, steps_l = [], [], []
for _ in range({games}):
    b = new_board(rng); sc = 0; st = 0
    while st < 20000:
        if searcher is not None:
            a = searcher.best_action(b)
            if a is None: break
            bs, ss, _ = move_all(b)
            nb, s = bs[a], int(ss[a])
        else:
            r = greedy_action(net, b)
            if r is None: break
            _, nb, s, _ = r
        sc += s; b = spawn(nb.copy(), rng); st += 1
    scores.append(sc); tiles.append(int(decode_board(b).max())); steps_l.append(st)

a = np.asarray(scores); t = np.asarray(tiles)
reach = lambda v: float((t >= v).mean())
out = {{
    "games": {games}, "depth": {depth}, "tag": "{tag}",
    "avg": float(a.mean()), "max": int(a.max()), "median": float(np.median(a)),
    "avg_steps": float(np.mean(steps_l)),
    "ms512": reach(512), "ms1024": reach(1024),
    "ms2048": reach(2048), "ms4096": reach(4096),
}}
print("RESULT " + json.dumps(out))
'''
    try:
        r = subprocess.run([sys.executable, "-u", "-c", code],
                           capture_output=True, text=True, timeout=7200,
                           cwd=BASE)
    except subprocess.TimeoutExpired:
        log(f"  [{tag}] 评测超时")
        return None
    for line in (r.stdout or "").splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    log(f"  [{tag}] 评测失败: {(r.stderr or '')[-300:]}")
    return None


def main():
    if os.path.exists(DONE_FLAG):
        return 0                       # 已完成过, 不再重复

    st = load_status()
    if not st:
        log("状态文件缺失, 跳过")
        return 0

    ev = st.get("evals") or []
    ep = st.get("episode") or 0
    if not ev:
        log(f"@{ep:,} 局: 暂无评估数据")
        return 0

    last = ev[-1]
    r4096 = last.get("ms4096", 0.0)
    r2048 = last.get("ms2048", 0.0)

    log(f"@{ep:,} 局 | 4096 {r4096*100:.0f}% | 2048 {r2048*100:.0f}% "
        f"| 均分 {last['avg']:,.0f} | best {st.get('best_eval_score')}")

    if r4096 < TRIGGER_TH:
        return 0                       # 未达标, 等待下次

    # ---- 达到触发阈值: 先做 200 局复核 ----
    log(f"★ 4096 达 {r4096*100:.1f}% (>= {TRIGGER_TH*100:.0f}%), 启动 {CONFIRM_GAMES} 局复核...")
    confirm = run_eval(CONFIRM_GAMES, 1, "confirm-1ply")
    if confirm is None:
        return 1
    log(f"  复核结果: 4096 {confirm['ms4096']*100:.1f}% | "
        f"2048 {confirm['ms2048']*100:.1f}% | 均分 {confirm['avg']:,.0f}")

    if confirm["ms4096"] < CONFIRM_TH:
        log(f"  复核未达 {CONFIRM_TH*100:.0f}%（噪声导致误报）, 继续训练")
        return 0

    # ---- 确认达标: 跑 3-ply 终测 ----
    log(f"★★ 确认 4096 >= {CONFIRM_TH*100:.0f}%！启动 {FINAL_GAMES} 局 {FINAL_DEPTH}-ply 终测（较慢）...")
    final = run_eval(FINAL_GAMES, FINAL_DEPTH, f"final-{FINAL_DEPTH}ply")

    report = {
        "triggered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episode": ep,
        "train_eval_4096": r4096,
        "confirm_200g_1ply": confirm,
        "final_eval": final,
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(DONE_FLAG, "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

    if final:
        log(f"★★★ 终测完成: 均分 {final['avg']:,.0f} | 最高 {final['max']:,} | "
            f"2048 {final['ms2048']*100:.0f}% | 4096 {final['ms4096']*100:.0f}%")
    log(f"报告已写入 {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
