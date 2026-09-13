# -*- coding: utf-8 -*-
"""
visualize.py - 训练过程可视化模块

功能
----
使用 matplotlib（Agg 后端, 无需显示设备）绘制 2x2 训练监控面板:
    1. 每局分数曲线 + 滑动平均 —— 反映整体水平与趋势
    2. TD loss 曲线（滑动平均） —— 反映收敛情况
    3. ε 探索率衰减曲线         —— 反映探索 -> 利用的过渡
    4. 里程碑达成率（512/1024/2048/4096）随评估局数变化

每次调用 plot() 都将最新图表覆盖保存为 PNG（默认每 100 局调用一次）。
"""

import os
from typing import List

import matplotlib

matplotlib.use("Agg")  # 无显示环境下的后端
import matplotlib.pyplot as plt
import numpy as np

# 字体回退: 优先中文字体, 缺失时自动回退, 标签使用英文保证可读性
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def moving_average(values: List[float], window: int = 50) -> np.ndarray:
    """滑动平均。数据不足一个窗口时自动缩小窗口; 前段用首个均值填充对齐。"""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x
    window = max(1, min(window, x.size))
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    pad = np.full(window - 1, ma[0]) if ma.size else np.array([])
    return np.concatenate([pad, ma])


class TrainerVisualizer:
    """训练可视化器: 保存 2x2 监控面板 PNG。"""

    def __init__(self, out_dir: str = "outputs"):
        self.plot_dir = os.path.join(out_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.plot_path = os.path.join(self.plot_dir, "training_progress.png")

    def plot(self, history: dict) -> str:
        """根据训练历史绘制并保存监控面板。

        history 需要包含:
            episode_scores : List[float]  每局分数
            losses         : List[float]  每次训练步的 TD loss
            epsilons       : List[float]  每局结束时的 ε
            evals          : List[dict]   评估记录, 含 episode 与 milestones
        """
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle("2048 AI Training Monitor", fontsize=14, fontweight="bold")

        # ---- 1. 分数曲线 ----
        ax = axes[0][0]
        scores = history.get("episode_scores", [])
        if scores:
            ax.plot(scores, alpha=0.25, color="#4C9BE8", linewidth=0.8,
                    label="score (raw)")
            ma = moving_average(scores, window=50)
            ax.plot(ma, color="#1D6FBF", linewidth=1.8, label="MA-50")
            ax.axhline(float(np.median(scores)), color="gray", linestyle="--",
                       linewidth=0.8, alpha=0.6, label="median")
        ax.set_title("Episode Score")
        ax.set_xlabel("episode")
        ax.set_ylabel("score")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        # ---- 2. Loss 曲线 ----
        ax = axes[0][1]
        losses = history.get("losses", [])
        if losses:
            ax.plot(moving_average(losses, window=500), color="#E8744C",
                    linewidth=1.2, label="TD loss (MA-500)")
            ax.set_yscale("log")
        ax.set_title("TD Loss (log scale)")
        ax.set_xlabel("train step")
        ax.set_ylabel("MSE loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        # ---- 3. ε 衰减曲线 ----
        ax = axes[1][0]
        eps = history.get("epsilons", [])
        if eps:
            ax.plot(eps, color="#5CA65C", linewidth=1.5)
            ax.set_ylim(-0.02, 1.02)
        ax.set_title("Epsilon Decay")
        ax.set_xlabel("episode")
        ax.set_ylabel("epsilon")
        ax.grid(alpha=0.25)

        # ---- 4. 里程碑达成率 ----
        ax = axes[1][1]
        evals = history.get("evals", [])
        if evals:
            episodes = [e["episode"] for e in evals]
            for key, label, color in (("ms512", "512", "#9E86C8"),
                                      ("ms1024", "1024", "#5CA6C8"),
                                      ("ms2048", "2048", "#C89E5C"),
                                      ("ms4096", "4096", "#C85C5C")):
                ratio = [e.get(key, 0.0) for e in evals]
                if max(ratio) > 0:
                    ax.plot(episodes, ratio, marker="o", markersize=3,
                            linewidth=1.2, label=f"reach {label}", color=color)
            ax.set_ylim(-0.03, 1.05)
        ax.set_title("Milestone Reach Rate (per eval)")
        ax.set_xlabel("episode")
        ax.set_ylabel("ratio")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(self.plot_path, dpi=110)
        plt.close(fig)
        return self.plot_path


# ---------------- 模块自测 ----------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    history = {
        "episode_scores": list(np.cumsum(rng.normal(5, 20, 300))),
        "losses": list(np.abs(rng.normal(1, 0.5, 3000)) / (1 + np.arange(3000) / 500)),
        "epsilons": list(np.clip(1.0 - np.arange(300) / 200, 0.05, 1.0)),
        "evals": [{"episode": e * 10, "ms512": min(1.0, e / 15),
                   "ms1024": min(1.0, e / 25), "ms2048": min(1.0, e / 40),
                   "ms4096": 0.0} for e in range(1, 31)],
    }
    path = TrainerVisualizer("outputs").plot(history)
    print("visualize.py 自测通过 ✓  图表已保存:", path)
