# -*- coding: utf-8 -*-
"""
train_v2.py - Rainbow-Lite (Horizon-DQN) 训练引擎

对标 arXiv 2507.05465《2048: Reinforcement Learning in a Delayed Reward
Environment》中的 H-DQN, 逐项落实其相对 vanilla DQN 的全部核心改进:

    DQN 基线                        本 v2 (Rainbow-Lite)
    -----------------------------   ------------------------------------
    log2 标量向量 (16,)             16x4x4 二值张量（tile 独热编码）
    MLP 256-256                     两层卷积 encoder（局部空间模式归纳偏置）
    单流输出 4 个 Q                  Dueling 双流 + C51 分布式 (4x51 分位数)
    单步 TD                          n-step(3) 多步回报
    均匀回放                         PER 优先回放 (α=0.6, β: 0.4→1.0)
    ε-greedy 手工衰减                NoisyNet 因子化高斯噪声（学习式探索）
    无数据增强                       8 重对称增强（样本量 x8）

另外两个来自实测的关键工程处理:
    * 奖励缩放 (reward_scale): 合并奖励可达 2048, 远超 C51 支撑范围
      [-30, 200], 因此统一缩放 0.1 后再入网（等价于对价值函数做线性变换,
      不改变最优策略）。
    * 单局步数上限: 塑形奖励中"每步都给"的项存在"无限存活"退化解,
      必须设上限防止训练被超长对局阻塞。
"""

import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from buffer_v2 import NStepPrioritizedReplay
from env import Game2048
from model import atomic_torch_save, load_torch
from model_v2 import (NUM_ATOMS, V_MAX, V_MIN, RainbowAgent, c51_loss,
                      project_distribution)
from train import setup_logging      # 复用日志设置
from visualize import TrainerVisualizer

logger = logging.getLogger("2048")
logging.getLogger("2048").setLevel(logging.INFO)


@dataclass
class RainbowConfig:
    """v2 (Rainbow-Lite) 超参数。"""
    episodes: int = 5000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-4
    buffer_size: int = 100_000
    n_step: int = 3                    # n 步回报（论文关键机制之一）
    per_alpha: float = 0.6             # PER 优先级指数
    per_beta0: float = 0.4             # IS 权重初始 β（退火到 1.0）
    augment: bool = True               # 8 重对称增强
    grad_clip: float = 1.0
    train_start: int = 1000            # 缓冲区达到该规模后开始训练
    train_every: int = 4               # 每 N 个环境步执行一次梯度更新
                                       #   （Conv+C51 单次反向约 30ms, 逐步训练
                                       #   会让训练时间 8 倍于环境交互; 取 4 与
                                       #   Atari 的 frame-skip 训练等价, 样本吞吐
                                       #   不变而墙钟时间降 4 倍）
    target_sync_every: int = 500       # target 网络同步间隔（环境步）
    torch_threads: int = 1             # 小网络单线程最快
    eps_start: float = 0.2             # NoisyNet 已自带探索, ε 仅作辅助
    eps_end: float = 0.0
    eps_decay_steps: int = 50_000
    reward_scale: float = 0.1          # 奖励缩放（适配 C51 支撑范围）
    eval_every: int = 100
    eval_games: int = 20
    ckpt_every: int = 500
    max_episode_steps: int = 3000      # 单局步数上限
    seed: int = 42
    device: str = "auto"
    out_dir: str = "outputs_v2"


class RainbowTrainer:
    """Rainbow-Lite 训练器（接口与 v1 DQNTrainer 对齐, 便于 Web 控制台复用）。"""

    def __init__(self, cfg: RainbowConfig):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.models_dir = os.path.join(cfg.out_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        setup_logging(cfg.out_dir)

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.torch_threads > 0:
            torch.set_num_threads(cfg.torch_threads)

        self.env = Game2048(seed=cfg.seed, max_steps=cfg.max_episode_steps)
        self.agent = RainbowAgent(lr=cfg.lr, gamma=cfg.gamma,
                                  device=cfg.device, n_atoms=NUM_ATOMS,
                                  v_min=V_MIN, v_max=V_MAX, noisy=True)
        self.buffer = NStepPrioritizedReplay(
            capacity=cfg.buffer_size, n_step=cfg.n_step, gamma=cfg.gamma,
            alpha=cfg.per_alpha, beta0=cfg.per_beta0, augment=cfg.augment,
            seed=cfg.seed)
        self.viz = TrainerVisualizer(cfg.out_dir)

        self.gamma_n = cfg.gamma ** cfg.n_step     # n 步折扣 γ^n
        self.episode = 0
        self.total_steps = 0
        self.best_eval_score = float("-inf")
        self._recent_scores: List[float] = []
        self._start_time = time.time()
        self.status_path = os.path.join(cfg.out_dir, "status.json")

        self.history: Dict[str, list] = {
            "episode_scores": [], "episode_steps": [], "episode_max": [],
            "losses": [], "epsilons": [], "evals": [], "selfplay": [],
        }

        logger.info("=" * 66)
        logger.info("Rainbow-Lite (H-DQN 复刻) 初始化")
        logger.info("  网络     : Conv(2层) + Dueling + C51(%d atoms) + NoisyNet",
                    NUM_ATOMS)
        logger.info("  状态表示 : 16x4x4 二值张量 (tile 独热)")
        logger.info("  回报     : %d-step (γ^n=%.4f) | PER α=%.2f β0=%.2f",
                    cfg.n_step, self.gamma_n, cfg.per_alpha, cfg.per_beta0)
        logger.info("  增强     : 8 重对称变换 = %s | 奖励缩放 %.3f | 训练频率 1/%d 步",
                    "开" if cfg.augment else "关", cfg.reward_scale,
                    cfg.train_every)
        logger.info("  设备     : %s | 线程 %d", self.agent.device, cfg.torch_threads)
        logger.info("=" * 66)

    # ================= 状态文件（Web 控制台） =================

    def _write_status(self, running: bool = True) -> None:
        """原子写入 status.json 供 Web 实时展示。"""
        try:
            scores = self.history["episode_scores"]
            losses = self.history["losses"]
            status = {
                "running": running,
                "algo": "rainbow_v2",
                "episode": self.episode,
                "total_steps": self.total_steps,
                "epsilon": round(self.epsilon(), 4),
                "buffer_size": len(self.buffer),
                "recent_scores": [int(s) for s in scores[-300:]],
                "recent_scores_start": max(0, len(scores) - 300),
                "epsilons_tail": [round(float(e), 4)
                                  for e in self.history["epsilons"][-300:]],
                "losses_tail": [round(float(x), 5) for x in losses[-2000:]],
                "evals": self.history["evals"][-60:],
                "selfplay": [],
                "best_eval_score": (None if self.best_eval_score == float("-inf")
                                    else round(self.best_eval_score, 1)),
                "selfplay_version": 0,
                "mcts_backend": None,
                "use_mcts": False,
                "per_beta": self.buffer._beta(),
                "buffer_stats": self.buffer.stats(),
                "elapsed_sec": round(time.time() - self._start_time, 1),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(self.agent.device),
                "config": asdict(self.cfg),
            }
            tmp = f"{self.status_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False)
            os.replace(tmp, self.status_path)
        except Exception as exc:
            logger.debug("status.json 写入失败: %s", exc)

    # ================= 训练核心 =================

    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def train_step(self) -> Optional[float]:
        """一次 Rainbow 更新: Double 动作选择 + C51 分布投影 + PER 加权损失。"""
        cfg = self.cfg
        if len(self.buffer) < max(cfg.batch_size, cfg.n_step):
            return None
        (states, actions, rewards, next_states, dones,
         weights, idxs) = self.buffer.sample(cfg.batch_size)

        dev = self.agent.device
        s = torch.as_tensor(states, device=dev)
        a = torch.as_tensor(actions, device=dev).view(-1, 1, 1).expand(
            -1, 1, NUM_ATOMS)
        r = torch.as_tensor(rewards, device=dev)
        ns = torch.as_tensor(next_states, device=dev)
        d = torch.as_tensor(dones, device=dev)
        w = torch.as_tensor(weights, device=dev)

        # 当前动作的回报分布 Z(s,a)  ->  (B, n_atoms)
        dist_all = self.agent.policy_net(s)             # (B, 4, 51)
        dist_a = dist_all.gather(1, a).squeeze(1)

        with torch.no_grad():
            # Double-Q 思想: 用 policy 网络选动作（分布均值最大者）,
            # 再用 target 网络给出该动作的分布 —— 抑制过估计
            next_q = self.agent.policy_net(ns).mean(dim=2)         # (B, 4)
            best_a = next_q.argmax(dim=1).view(-1, 1, 1).expand(-1, 1, NUM_ATOMS)
            next_dist = self.agent.target_net(ns).gather(1, best_a).squeeze(1)

            # n-step 目标分布投影回固定支撑
            target_dist = project_distribution(
                next_dist, r, d, gamma_n=self.gamma_n,
                v_min=V_MIN, v_max=V_MAX, n_atoms=NUM_ATOMS)

            # TD 误差（分布均值的差）用于更新 PER 优先级
            support = torch.linspace(V_MIN, V_MAX, NUM_ATOMS, device=dev)
            td = ((target_dist * support).sum(dim=1)
                  - dist_a.mean(dim=1)).abs().cpu().numpy()

        loss = c51_loss(dist_a, target_dist, w)
        self.agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(),
                                       cfg.grad_clip)
        self.agent.optimizer.step()
        self.agent.reset_noise()          # NoisyNet: 每步刷新噪声采样
        self.buffer.update_priorities(idxs, td)
        return float(loss.item())

    def run_episode(self) -> Dict:
        """跑一局并在线学习。state 直接使用原始棋盘（编码在采样时进行）。"""
        cfg = self.cfg
        self.env.reset()
        grid = [list(r) for r in self.env.grid]
        done = False
        ep_reward, ep_losses = 0.0, []
        eps = self.epsilon()

        while not done:
            valid = self.env.get_valid_actions()
            if not valid:
                break
            action = self.agent.select_action(grid, eps, valid)
            _, reward, done, info = self.env.step(action)
            next_grid = [list(r) for r in self.env.grid]

            # 奖励缩放后入缓冲（适配 C51 支撑范围）
            self.buffer.push(grid, action, reward * cfg.reward_scale,
                             next_grid, done)
            grid = next_grid
            ep_reward += reward
            self.total_steps += 1

            if (len(self.buffer) >= cfg.train_start
                    and self.total_steps % cfg.train_every == 0):
                loss = self.train_step()
                if loss is not None:
                    ep_losses.append(loss)
                    self.history["losses"].append(loss)
                    if len(self.history["losses"]) > 200_000:
                        del self.history["losses"][:100_000]

            if self.total_steps % cfg.target_sync_every == 0:
                self.agent.sync_target()

        return {"score": info["score"], "steps": self.env.steps,
                "max_tile": self.env.max_tile, "reward": ep_reward,
                "avg_loss": float(np.mean(ep_losses)) if ep_losses else None,
                "epsilon": eps}

    # ================= 评估 =================

    @torch.no_grad()
    def evaluate(self, n_games: Optional[int] = None) -> Dict:
        """纯推理评估（评估模式下 NoisyNet 不加噪声）。"""
        n = n_games or self.cfg.eval_games
        scores, steps_l, tiles = [], [], []
        for _ in range(n):
            try:
                env = Game2048(max_steps=self.cfg.max_episode_steps)
                env.reset()
                grid = [list(r) for r in env.grid]
                done = False
                while not done:
                    valid = env.get_valid_actions()
                    if not valid:
                        break
                    action = self.agent.select_action(grid, 0.0, valid)
                    _, _, done, _ = env.step(action)
                    grid = [list(r) for r in env.grid]
                scores.append(env.score)
                steps_l.append(env.steps)
                tiles.append(env.max_tile)
            except Exception as exc:
                logger.warning("评估局异常, 跳过: %s", exc)

        if not scores:
            return {"avg": 0.0, "max": 0, "median": 0.0, "avg_steps": 0.0,
                    "ms512": 0.0, "ms1024": 0.0, "ms2048": 0.0, "ms4096": 0.0,
                    "games": 0}

        def reach(t: int) -> float:
            return sum(1 for x in tiles if x >= t) / len(tiles)

        return {"avg": float(np.mean(scores)), "max": int(np.max(scores)),
                "median": float(np.median(scores)),
                "avg_steps": float(np.mean(steps_l)),
                "ms512": reach(512), "ms1024": reach(1024),
                "ms2048": reach(2048), "ms4096": reach(4096),
                "games": len(scores)}

    # ================= checkpoint =================

    def save_checkpoint(self, name: str = "latest.pt") -> str:
        path = os.path.join(self.models_dir, name)
        sd = self.agent.optimizer.state_dict()
        opt_state = {
            "state": {k: {kk: (vv.cpu() if torch.is_tensor(vv) else vv)
                          for kk, vv in v.items()} for k, v in sd["state"].items()},
            "param_groups": sd["param_groups"],
        }
        atomic_torch_save({
            "algo": "rainbow_v2",
            "episode": self.episode,
            "total_steps": self.total_steps,
            "best_eval_score": self.best_eval_score,
            "policy_sd": self.agent.state_dict_cpu(),
            "target_sd": self.agent.target_state_dict_cpu(),
            "optimizer_full": opt_state,
            "history": self.history,
            "config": asdict(self.cfg),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        ckpt = load_torch(path, map_location=self.agent.device)
        self.episode = int(ckpt.get("episode", 0))
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.best_eval_score = float(ckpt.get("best_eval_score", float("-inf")))
        self.agent.load_both_state_dict(ckpt["policy_sd"])
        opt = ckpt.get("optimizer_full")
        if opt:
            try:
                self.agent.optimizer.load_state_dict(opt)
            except Exception as exc:
                logger.warning("优化器状态恢复失败（不影响继续训练）: %s", exc)
        for key, val in (ckpt.get("history") or {}).items():
            if key in self.history and isinstance(val, list):
                self.history[key] = val
        logger.info("已恢复 v2 checkpoint: 第 %d 局, 总步数 %d",
                    self.episode, self.total_steps)

    # ================= 主循环 =================

    def train(self, episodes: Optional[int] = None) -> None:
        total = episodes or self.cfg.episodes
        cfg = self.cfg
        logger.info("开始 v2 训练: %d 局", total)
        start = time.time()
        for _ in range(total):
            self.episode += 1
            try:
                stats = self.run_episode()
            except KeyboardInterrupt:
                logger.info("收到中断信号, 保存进度后退出...")
                self.save_checkpoint("latest.pt")
                self._write_status(running=False)
                raise
            except Exception as exc:
                logger.exception("第 %d 局异常, 跳过: %s", self.episode, exc)
                self.episode -= 1
                continue

            self.history["episode_scores"].append(stats["score"])
            self.history["episode_steps"].append(stats["steps"])
            self.history["episode_max"].append(stats["max_tile"])
            self.history["epsilons"].append(stats["epsilon"])
            self._recent_scores.append(stats["score"])
            if len(self._recent_scores) > 50:
                self._recent_scores.pop(0)

            self._write_status()          # 每局刷新 Web 状态

            if self.episode % 10 == 0:
                loss_str = (f"{stats['avg_loss']:.4f}"
                            if stats["avg_loss"] is not None else "  --  ")
                logger.info("局 %5d | 分数 %6d | 近50局均值 %7.1f | 最大块 %5d | "
                            "ε %.3f | β %.2f | loss %s | 步数 %4d",
                            self.episode, stats["score"],
                            float(np.mean(self._recent_scores)),
                            stats["max_tile"], stats["epsilon"],
                            self.buffer._beta(), loss_str, stats["steps"])

            if self.episode % cfg.eval_every == 0:
                self._eval_cycle()
            if self.episode % cfg.ckpt_every == 0:
                self.save_checkpoint(f"checkpoint_ep{self.episode}.pt")
                self.save_checkpoint("latest.pt")
                logger.info(">> checkpoint 已保存 (局 %d)", self.episode)

        self.save_checkpoint("latest.pt")
        self._write_status(running=False)
        logger.info("v2 训练完成: %d 局, 用时 %.1f 分钟",
                    total, (time.time() - start) / 60)

    def _eval_cycle(self) -> None:
        stats = self.evaluate()
        stats["episode"] = self.episode
        self.history["evals"].append(stats)
        self.viz.plot(self.history)

        logger.info("-" * 66)
        logger.info("评估 @ 局 %d (%d 局): 平均分 %.0f | 最高 %d | 中位 %.0f | "
                    "平均存活 %.0f 步", self.episode, stats["games"], stats["avg"],
                    stats["max"], stats["median"], stats["avg_steps"])
        logger.info("里程碑: 512 %.0f%% | 1024 %.0f%% | 2048 %.0f%% | 4096 %.0f%%",
                    stats["ms512"] * 100, stats["ms1024"] * 100,
                    stats["ms2048"] * 100, stats["ms4096"] * 100)
        if stats["avg"] > self.best_eval_score:
            self.best_eval_score = stats["avg"]
            atomic_torch_save({
                "algo": "rainbow_v2",
                "policy_sd": self.agent.state_dict_cpu(),
                "episode": self.episode,
                "eval_avg": stats["avg"],
                "eval_stats": stats,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, os.path.join(self.models_dir, "best.pt"))
            logger.info("★ 新最佳评估均分 %.0f, 已保存 best.pt", stats["avg"])
        logger.info("-" * 66)

    def close(self) -> None:
        pass


# ---------------- 命令行入口 ----------------

def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="2048 Rainbow-Lite (H-DQN) 训练")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-dir", default="outputs_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--train-every", type=int, default=None,
                   help="每 N 个环境步训练一次（默认 4）")
    p.add_argument("--eval-games", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()

    cfg = RainbowConfig(episodes=args.episodes, out_dir=args.out_dir,
                        seed=args.seed, device=args.device,
                        torch_threads=args.torch_threads, n_step=args.n_step,
                        augment=not args.no_augment)
    if args.eval_games is not None:
        cfg.eval_games = args.eval_games
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    if args.lr is not None:
        cfg.lr = args.lr
    if args.train_every is not None:
        cfg.train_every = args.train_every

    trainer = RainbowTrainer(cfg)
    try:
        if args.resume:
            latest = os.path.join(trainer.models_dir, "latest.pt")
            if os.path.exists(latest):
                trainer.load_checkpoint(latest)
        trainer.train(args.episodes)
    except KeyboardInterrupt:
        logger.info("用户中断, 进度已保存")
    finally:
        trainer.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
