# -*- coding: utf-8 -*-
"""
verify.py - v4 验证框架（推理层强化学习效果对比）

统一对比多种"决策配置"的速度与水平：

    A. n-tuple 1-ply            基线（训练用）
    B. n-tuple 2-ply            教师（搜索增强）
    C. n-tuple 2-ply 向量化      等价但快 4x
    D. student 网络             蒸馏学生（若已训练）
    E. student + n-tuple        ensemble（若已训练）

输出：
    - 每个配置的：平均分 / 最高分 / 中位数 / 方块达成率 / 步数 / 速度
    - 汇总对比表 + 相对基线的提升百分比

用法:
    python verify.py --games 20
    python verify.py --games 30 --configs A,B,C
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (_HERE, os.path.dirname(_HERE), "/opt/2048ai", "/workspace/2048_ai"):
    if os.path.isfile(os.path.join(_c, "ntuple.py")):
        sys.path.insert(0, _c)
        break

from ntuple import (PATTERNS_8, NTupleNetwork, decode_board, greedy_action,
                    move_all, new_board, spawn)


# ---------------- 各类决策器 ----------------

class GreedyPolicy:
    """A: 1-ply 贪心（n-tuple）。"""
    name = "A. n-tuple 1-ply"

    def __init__(self, net, **kw):
        self.net = net

    def act(self, board):
        r = greedy_action(self.net, board)
        return None if r is None else r[0]


class SearchPolicy:
    """B/C: expectimax（可选向量化）。"""
    def __init__(self, net, depth=2, vectorized=False, max_empties=10):
        self.depth = depth
        self.vectorized = vectorized
        if vectorized and depth == 2:
            from vecsearch import VecExpectimax
            self.impl = VecExpectimax(net, max_empties=max_empties)
        else:
            from expectimax import ExpectimaxSearcher
            self.impl = ExpectimaxSearcher(net, depth=depth,
                                           max_empties=max_empties)
        self.net = net
        self.name = (f"{'C' if vectorized else 'B'}. n-tuple {depth}-ply"
                     + ("（向量化）" if vectorized else ""))

    def act(self, board):
        return self.impl.best_action(board)


class StudentPolicy:
    """D: 蒸馏学生网络（仅 1-ply 前向）。"""
    name = "D. student 网络"

    def __init__(self, model_path, **kw):
        from student_net import StudentAgent
        self.agent = StudentAgent(model_path)

    def act(self, board):
        vals = self.agent.predict(board)
        from ntuple import move_all
        _, _, moved = move_all(board)
        v = np.where(moved, vals, -np.inf)
        if not np.isfinite(v).any():
            return None
        return int(np.argmax(v))


class EnsemblePolicy:
    """E: 学生网络 + n-tuple 集成（两者归一化后相加）。

    归一化：各自减去各自的有效动作均值（消除量纲差异）。
    """
    name = "E. student + n-tuple"

    def __init__(self, second_net, model_path, **kw):
        from student_net import StudentAgent
        self.agent = StudentAgent(model_path)
        self.net = second_net
        self.w_student = float(kw.get("w_student", 1.0))
        self.w_ntuple = float(kw.get("w_ntuple", 1.0))

    def act(self, board):
        sv = self.agent.predict(board)
        # n-tuple 1-ply 动作价值
        bs, ss, moved = move_all(board)
        nv = np.full(4, -np.inf, dtype=np.float32)
        for a in range(4):
            if moved[a]:
                nv[a] = float(ss[a]) + self.net.evaluate(bs[a])

        def norm(x):
            m = np.isfinite(x)
            if not m.any():
                return x
            mu = x[m].mean()
            sd = x[m].std() + 1e-9
            y = np.where(m, (x - mu) / sd, -np.inf)
            return y

        v = self.w_student * norm(sv.astype(np.float64)) + \
            self.w_ntuple * norm(nv.astype(np.float64))
        if not np.isfinite(v).any():
            return None
        return int(np.argmax(v))


# ---------------- 对局执行 ----------------

def play_games(policy, n_games, seed=2024, max_steps=20000):
    """跑 n 局，返回统计。"""
    rng = np.random.default_rng(seed)
    scores, steps_l, tiles = [], [], []
    t0 = time.time()
    total_steps = 0
    for _ in range(n_games):
        b = new_board(rng)
        sc, st = 0, 0
        while st < max_steps:
            a = policy.act(b)
            if a is None:
                break
            bs, ss, _ = move_all(b)
            if not np.isfinite(ss[a]):
                break
            sc += int(ss[a])
            b = spawn(bs[a].copy(), rng)
            st += 1
        scores.append(sc)
        steps_l.append(st)
        tiles.append(int(decode_board(b).max()))
        total_steps += st
    el = time.time() - t0
    s = np.asarray(scores)
    t = np.asarray(tiles)

    def reach(v):
        return float((t >= v).mean())

    return {
        "avg": float(s.mean()), "max": int(s.max()),
        "median": float(np.median(s)),
        "avg_steps": float(np.mean(steps_l)),
        "ms512": reach(512), "ms1024": reach(1024),
        "ms2048": reach(2048), "ms4096": reach(4096),
        "games": n_games, "seconds": el,
        "steps_per_sec": total_steps / max(1e-9, el),
        "ms_per_step": el / max(1, total_steps) * 1000,
    }


def build_policies(cfg_str, net, student_path):
    """按配置字符串构建策略列表。"""
    configs = [c.strip().upper() for c in cfg_str.split(",")] if cfg_str else \
        ["A", "B", "C", "D", "E"]
    out = []
    for c in configs:
        if c == "A":
            out.append(GreedyPolicy(net))
        elif c == "B":
            out.append(SearchPolicy(net, depth=2, vectorized=False))
        elif c == "C":
            out.append(SearchPolicy(net, depth=2, vectorized=True))
        elif c == "D":
            if student_path and os.path.exists(student_path):
                try:
                    out.append(StudentPolicy(student_path))
                except Exception as e:
                    print(f"  ⚠ 学生网络加载失败: {e}")
            else:
                print("  ⚠ 跳过 D（未找到学生模型）")
        elif c == "E":
            if student_path and os.path.exists(student_path):
                try:
                    out.append(EnsemblePolicy(net, student_path))
                except Exception as e:
                    print(f"  ⚠ ensemble 加载失败: {e}")
            else:
                print("  ⚠ 跳过 E（未找到学生模型）")
    return out


def main():
    p = argparse.ArgumentParser(description="v4 验证框架")
    p.add_argument("--model", default="/opt/2048ai/exp_hp/models/best.npz")
    p.add_argument("--student", default="/opt/2048ai/v4_data/student.pt")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--configs", default="A,B,C,D,E")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", default="/opt/2048ai/v4_data/verify_report.json")
    args = p.parse_args()

    net = NTupleNetwork(PATTERNS_8)
    net.load(args.model)
    print(f"教师模型: {args.model} (σ={net.lut.std():.1f})")
    print(f"对局数: {args.games} | 随机种子: {args.seed}\n")

    policies = build_policies(args.configs, net, args.student)
    if not policies:
        print("没有可运行的配置")
        return 1

    results = []
    for pol in policies:
        print(f"运行 {pol.name} ...", flush=True)
        st = play_games(pol, args.games, seed=args.seed)
        st["name"] = pol.name
        results.append(st)
        print(f"  均分 {st['avg']:>10,.0f} | 最高 {st['max']:>9,} | "
              f"2048 {st['ms2048']*100:>3.0f}% | 4096 {st['ms4096']*100:>3.0f}% | "
              f"{st['ms_per_step']:>7.2f} ms/步 | {st['seconds']:.0f}s", flush=True)

    # ---- 汇总 ----
    print("\n" + "=" * 92)
    print("%-26s %11s %10s %7s %7s %10s %9s" % (
        "配置", "平均分", "最高分", "2048", "4096", "ms/步", "步/秒"))
    print("-" * 92)
    base = results[0]["avg"] if results else 0
    for r in results:
        rel = ""
        if base > 0 and r is not results[0]:
            rel = f" ({r['avg']/base*100:.0f}%)"
        print("%-26s %11s %10s %6.0f%% %6.0f%% %10.2f %9.0f" % (
            r["name"], f"{r['avg']:,.0f}{rel}", f"{r['max']:,}",
            r["ms2048"] * 100, r["ms4096"] * 100,
            r["ms_per_step"], r["steps_per_sec"]))
    print("=" * 92)

    # 保存报告
    import json
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "games": args.games,
                   "seed": args.seed, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"报告: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
