# -*- coding: utf-8 -*-
"""
train.py - DQN 训练引擎

训练流程
--------
每一局游戏:
    1. reset 环境, 用 ε-greedy（默认）或 MCTS（每 mcts_interval 局插入一局）
       选择动作;
    2. 执行动作, 收集 transition 存入 Replay Buffer;
    3. Buffer 内数据 >= train_start 后, 每步随机采样 batch(64) 做一次
       DQN 更新:
           Q(s,a) ← r + γ · max_a' Q_target(s', a') · (1 - done)
           loss = MSE(Q(s,a), TD目标), Adam(lr=1e-4) + 梯度裁剪(max_norm=1.0)
    4. 每 500 环境步将 policy_net 参数硬同步到 target_net;
    5. ε 从 1.0 线性衰减到 0.05, 衰减周期 10,000 步。

周期任务
--------
- 每 100 局: 纯推理评估（平均分/最高分/中位数/里程碑达成率/平均存活步数）,
  并刷新训练监控图表; 平均分创历史新高则保存 best.pt。
- 每 500 局: 保存完整 checkpoint（模型权重 + 优化器状态 + 训练进度, 原子写入）。
- 每 selfplay_every 局: 发起自我对弈挑战（selfplay.py）。

健壮性
------
- 单局训练/评估被 try-except 包裹, 单局崩溃只记录日志不中断训练;
- checkpoint 使用原子写入; 日志同时输出到控制台与文件。
"""

import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from env import Game2048
from model import DQNAgent, ReplayBuffer, atomic_torch_save, load_torch
from mcts import MCTSAgent
from selfplay import SelfPlayManager
from visualize import TrainerVisualizer

logger = logging.getLogger("2048")


def setup_logging(out_dir: str, level: int = logging.INFO) -> None:
    """日志同时输出到控制台与文件（重复调用安全）。"""
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    lg = logging.getLogger("2048")
    lg.setLevel(level)
    if lg.handlers:                       # 防止重复添加 handler
        return
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"),
                             encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"))
    lg.addHandler(fh)


@dataclass
class TrainConfig:
    """训练超参数配置（与用户规格一一对应）。"""
    episodes: int = 5000            # 训练总局数
    batch_size: int = 64            # 训练 batch 大小
    gamma: float = 0.99             # 折扣因子
    lr: float = 1e-4                # Adam 学习率
    buffer_size: int = 100_000      # 回放缓冲区容量
    grad_clip: float = 1.0          # 梯度裁剪 max_norm
    train_start: int = 1000         # Buffer 数据达到该值后开始训练
    target_sync_every: int = 500    # 每 N 环境步同步 target_net
    eps_start: float = 1.0          # ε 初始值
    eps_end: float = 0.05           # ε 终值
    eps_decay_steps: int = 10_000   # ε 线性衰减周期（环境步）

    eval_every: int = 100           # 每 N 局评估一次
    eval_games: int = 20            # 每次评估局数
    ckpt_every: int = 500           # 每 N 局保存 checkpoint

    use_mcts: bool = True           # 是否启用 MCTS 搜索增强
    mcts_interval: int = 20         # 每 N 局插入 1 局 MCTS 决策局
    mcts_simulations: int = 100     # 每次决策模拟次数
    mcts_max_depth: int = 200       # 模拟最大深度（防树无限加深）
    parallel: int = 0               # MCTS 并行进程数, 0 = CPU 核数 - 1
    max_episode_steps: int = 3000   # 单局步数上限（防"无限存活"退化解阻塞训练）

    selfplay_every: int = 200       # 每 N 局发起一次自我对弈挑战
    selfplay_games: int = 50        # 挑战场双方各玩局数
    improve_margin: float = 0.05    # 晋级门槛: 平均分领先 5%

    seed: int = 42
    device: str = "auto"            # auto / cpu / cuda
    torch_threads: int = 1          # PyTorch 计算线程数（小网络单线程最快,
                                    #  且能给 MCTS worker 让出 CPU 核心）
    out_dir: str = "outputs"        # 输出根目录


class DQNTrainer:
    """DQN 训练器: 整合环境 / 网络 / 回放 / MCTS / 自我对弈 / 可视化。"""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.models_dir = os.path.join(cfg.out_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        setup_logging(cfg.out_dir)

        # 复现性
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # 限制 PyTorch 计算线程数: 16→256→256→4 的小网络用多线程反而因
        # 线程同步而变慢, 且会与 MCTS worker 进程争抢 CPU 核心（实测可使
        # MCTS 决策慢 60 倍）。单线程为最优默认值。
        if cfg.torch_threads > 0:
            torch.set_num_threads(cfg.torch_threads)

        self.env = Game2048(seed=cfg.seed, max_steps=cfg.max_episode_steps)
        self.agent = DQNAgent(lr=cfg.lr, gamma=cfg.gamma, device=cfg.device)
        self.buffer = ReplayBuffer(capacity=cfg.buffer_size)

        self.mcts: Optional[MCTSAgent] = None
        if cfg.use_mcts:
            n_proc = cfg.parallel if cfg.parallel > 0 else None
            self.mcts = MCTSAgent(n_simulations=cfg.mcts_simulations,
                                  n_processes=n_proc, gamma=cfg.gamma,
                                  max_depth=cfg.mcts_max_depth)
            logger.info("MCTS 已启用: %d 次模拟/决策, %d 并行进程, 最大深度 %d",
                        cfg.mcts_simulations, self.mcts.n_processes,
                        cfg.mcts_max_depth)

        self.selfplay = SelfPlayManager(self.models_dir,
                                        improve_margin=cfg.improve_margin,
                                        n_games=cfg.selfplay_games,
                                        device="cpu")
        self.viz = TrainerVisualizer(cfg.out_dir)

        # 训练进度
        self.episode = 0
        self.total_steps = 0
        self.best_eval_score = float("-inf")
        self._recent_scores: List[float] = []   # 最近 50 局滑动日志用

        # 训练历史（用于可视化与 checkpoint 恢复）
        self.history: Dict[str, list] = {
            "episode_scores": [],   # 每局分数
            "episode_steps": [],    # 每局步数
            "episode_max": [],      # 每局最大方块
            "losses": [],           # 每训练步 TD loss
            "epsilons": [],         # 每局结束时的 ε
            "evals": [],            # 评估记录
            "selfplay": [],         # 自我对弈挑战记录
        }

        # Web 控制台实时状态文件（每 10 局 / 评估后 / 结束时刷新）
        self.status_path = os.path.join(cfg.out_dir, "status.json")
        self._start_time = time.time()

    # ================= 实时状态导出（供 Web 控制台读取） =================

    def _write_status(self, running: bool = True, extra: Optional[dict] = None) -> None:
        """原子写入 status.json: 供 Web 控制台实时展示训练进度与曲线。

        只写尾部窗口数据（分数/损失），避免文件随训练无限膨胀。
        """
        try:
            best = None if self.best_eval_score == float("-inf") else round(self.best_eval_score, 1)
            scores = self.history["episode_scores"]
            status = {
                "running": running,
                "episode": self.episode,
                "total_steps": self.total_steps,
                "epsilon": round(self.epsilon(), 4),
                "buffer_size": len(self.buffer),
                "recent_scores": [int(s) for s in scores[-300:]],
                "recent_scores_start": max(0, len(scores) - 300),
                "epsilons_tail": [round(float(e), 4) for e in self.history["epsilons"][-300:]],
                "losses_tail": [round(float(x), 5) for x in self.history["losses"][-2000:]],
                "evals": self.history["evals"][-60:],
                "selfplay": self.history["selfplay"][-20:],
                "best_eval_score": best,
                "selfplay_version": self.selfplay.version,
                "mcts_backend": (self.mcts._backend_name if self.mcts else None),
                "use_mcts": self.cfg.use_mcts,
                "elapsed_sec": round(time.time() - self._start_time, 1),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(self.agent.device),
                "config": asdict(self.cfg),
            }
            if extra:
                status.update(extra)
            tmp = f"{self.status_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False)
            os.replace(tmp, self.status_path)   # 原子替换
        except Exception as exc:                # 状态文件失败绝不能影响训练
            logger.debug("status.json 写入失败: %s", exc)

    # ================= 核心组件 =================

    def epsilon(self) -> float:
        """ε 线性衰减: eps_start -> eps_end, 周期 eps_decay_steps。"""
        cfg = self.cfg
        frac = min(1.0, self.total_steps / max(1, cfg.eps_decay_steps))
        return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

    def train_step(self) -> Optional[float]:
        """从 Buffer 采样一个 batch 执行一次 DQN 更新, 返回 loss（或 None）。

        TD 目标: y = r + γ · max_a' Q_target(s', a') · (1 - done)
        """
        if len(self.buffer) < self.cfg.batch_size:
            return None
        cfg = self.cfg
        states, actions, rewards, next_states, dones = self.buffer.sample(cfg.batch_size)

        states_t = torch.as_tensor(states, device=self.agent.device)
        actions_t = torch.as_tensor(actions, device=self.agent.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.agent.device)
        next_states_t = torch.as_tensor(next_states, device=self.agent.device)
        dones_t = torch.as_tensor(dones, device=self.agent.device)

        # 当前 Q: Q(s,a)（gather 取出实际执行动作的 Q 值）
        q_values = self.agent.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # 目标 Q: r + γ · max Q_target(s', ·) · (1 - done) —— no_grad 稳定目标
        with torch.no_grad():
            next_q = self.agent.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + cfg.gamma * next_q * (1.0 - dones_t)

        loss = F.mse_loss(q_values, target)
        self.agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(),
                                       cfg.grad_clip)   # 梯度裁剪防爆炸
        self.agent.optimizer.step()
        return float(loss.item())

    def run_episode(self, use_mcts: bool = False, explore: bool = True) -> Dict:
        """运行一局训练局（探索 + 在线学习）。返回当局统计。"""
        state = self.env.reset()
        done = False
        ep_reward, ep_losses = 0.0, []
        eps = self.epsilon() if explore else 0.0

        while not done:
            valid = self.env.get_valid_actions()
            if not valid:
                break
            if use_mcts and self.mcts is not None:
                action = self.mcts.search(self.env.grid, self.agent)
                if action is None:                       # MCTS 失败回退贪心
                    action = self.agent.select_action(state, 0.0, valid)
            else:
                action = self.agent.select_action(state, eps, valid)

            next_state, reward, done, info = self.env.step(action)
            self.buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward
            self.total_steps += 1

            # 在线学习: Buffer 攒够 train_start 条后每步训练一次
            if len(self.buffer) >= self.cfg.train_start:
                loss = self.train_step()
                if loss is not None:
                    ep_losses.append(loss)
                    self.history["losses"].append(loss)
                    if len(self.history["losses"]) > 200_000:   # 防内存膨胀
                        del self.history["losses"][:100_000]

            # 周期同步目标网络
            if self.total_steps % self.cfg.target_sync_every == 0:
                self.agent.sync_target()

        return {"score": info["score"] if info else 0,
                "steps": self.env.steps,
                "max_tile": self.env.max_tile,
                "reward": ep_reward,
                "avg_loss": float(np.mean(ep_losses)) if ep_losses else None,
                "epsilon": eps}

    # ================= 评估 =================

    @torch.no_grad()
    def evaluate(self, n_games: Optional[int] = None) -> Dict:
        """纯推理评估（不探索、不学习、不写 Buffer）。

        返回: {avg, max, median, avg_steps, ms512, ms1024, ms2048, ms4096, games}
        """
        n = n_games or self.cfg.eval_games
        scores, steps_l, tiles = [], [], []
        for _ in range(n):
            try:
                env = Game2048(max_steps=self.cfg.max_episode_steps)  # 独立环境
                state = env.reset()
                done = False
                while not done:
                    valid = env.get_valid_actions()
                    if not valid:
                        break
                    action = self.agent.select_action(state, 0.0, valid)
                    state, _, done, _ = env.step(action)
                scores.append(env.score)
                steps_l.append(env.steps)
                tiles.append(env.max_tile)
            except Exception as exc:      # 单局评估崩溃不影响整体
                logger.warning("评估局异常, 跳过: %s", exc)

        if not scores:
            return {"avg": 0.0, "max": 0, "median": 0.0, "avg_steps": 0.0,
                    "ms512": 0.0, "ms1024": 0.0, "ms2048": 0.0, "ms4096": 0.0,
                    "games": 0}

        def reach(tile_value: int) -> float:
            return sum(1 for t in tiles if t >= tile_value) / len(tiles)

        return {"avg": float(np.mean(scores)),
                "max": int(np.max(scores)),
                "median": float(np.median(scores)),
                "avg_steps": float(np.mean(steps_l)),
                "ms512": reach(512), "ms1024": reach(1024),
                "ms2048": reach(2048), "ms4096": reach(4096),
                "games": len(scores)}

    # ================= checkpoint =================

    def save_checkpoint(self, name: str = "latest.pt") -> str:
        """原子保存完整训练状态（模型 + 优化器 + 进度 + 历史）。"""
        path = os.path.join(self.models_dir, name)
        atomic_torch_save({
            "episode": self.episode,
            "total_steps": self.total_steps,
            "best_eval_score": self.best_eval_score,
            "policy_sd": self.agent.state_dict_cpu(),
            "target_sd": self.agent.target_state_dict_cpu(),
            "optimizer_full": self._optimizer_state_cpu(),
            "history": self.history,
            "config": asdict(self.cfg),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, path)
        return path

    def _optimizer_state_cpu(self) -> dict:
        """导出优化器完整状态（移到 CPU 便于跨设备恢复）。"""
        sd = self.agent.optimizer.state_dict()
        state = {}
        for k, v in sd["state"].items():
            state[k] = {kk: (vv.cpu() if torch.is_tensor(vv) else vv)
                        for kk, vv in v.items()}
        return {"state": state, "param_groups": sd["param_groups"]}

    def load_checkpoint(self, path: str) -> None:
        """从 checkpoint 恢复训练进度。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        ckpt = load_torch(path, map_location=self.agent.device)
        self.episode = int(ckpt.get("episode", 0))
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.best_eval_score = float(ckpt.get("best_eval_score", float("-inf")))
        self.agent.load_both_state_dict(ckpt["policy_sd"])
        opt_state = ckpt.get("optimizer_full")
        if opt_state:
            try:
                self.agent.optimizer.load_state_dict(opt_state)
            except Exception as exc:
                logger.warning("优化器状态恢复失败（不影响继续训练）: %s", exc)
        saved_hist = ckpt.get("history") or {}
        for key, val in saved_hist.items():
            if key in self.history and isinstance(val, list):
                self.history[key] = val
        logger.info("已恢复 checkpoint: 第 %d 局, 总步数 %d", self.episode, self.total_steps)

    # ================= 训练主循环 =================

    def train(self, episodes: Optional[int] = None) -> None:
        """主训练循环（带完整的周期任务与异常防护）。"""
        total = episodes or self.cfg.episodes
        cfg = self.cfg
        logger.info("=" * 62)
        logger.info("开始训练: %d 局 | batch=%d γ=%.2f lr=%.0e buffer=%d",
                    total, cfg.batch_size, cfg.gamma, cfg.lr, cfg.buffer_size)
        logger.info("ε: %.2f -> %.2f (衰减 %d 步) | 设备: %s",
                    cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps,
                    self.agent.device)
        logger.info("=" * 62)

        start_time = time.time()
        for _ in range(total):
            self.episode += 1

            # ---- 单局训练（异常防护: 单局崩溃不中断训练）----
            try:
                use_mcts = (self.mcts is not None and cfg.mcts_interval > 0
                            and self.episode % cfg.mcts_interval == 0)
                stats = self.run_episode(use_mcts=use_mcts)
            except KeyboardInterrupt:
                logger.info("收到中断信号, 保存进度后退出...")
                self.save_checkpoint("latest.pt")
                raise
            except Exception as exc:
                logger.exception("第 %d 局训练异常, 跳过该局: %s", self.episode, exc)
                self.episode -= 1        # 异常局不计入进度
                continue

            # ---- 记录历史 ----
            self.history["episode_scores"].append(stats["score"])
            self.history["episode_steps"].append(stats["steps"])
            self.history["episode_max"].append(stats["max_tile"])
            self.history["epsilons"].append(stats["epsilon"])
            self._recent_scores.append(stats["score"])
            if len(self._recent_scores) > 50:
                self._recent_scores.pop(0)

            # 每局刷新 Web 控制台状态文件（开销 ~1ms, 保证监控实时性）
            self._write_status()

            # ---- 周期日志 ----
            if self.episode % 10 == 0:
                recent_avg = float(np.mean(self._recent_scores))
                loss_str = (f"{stats['avg_loss']:.4f}"
                            if stats["avg_loss"] is not None else "  --  ")
                tag = "[MCTS]" if use_mcts else "      "
                logger.info("局 %5d | %s 分数 %6d | 近50局均值 %7.1f | "
                            "最大块 %5d | ε %.3f | loss %s | 步数 %4d",
                            self.episode, tag, stats["score"], recent_avg,
                            stats["max_tile"], stats["epsilon"], loss_str,
                            stats["steps"])

            # ---- 每 100 局评估 ----
            if self.episode % cfg.eval_every == 0:
                self._run_eval_cycle()

            # ---- 每 500 局保存 checkpoint ----
            if self.episode % cfg.ckpt_every == 0:
                path = self.save_checkpoint(f"checkpoint_ep{self.episode}.pt")
                self.save_checkpoint("latest.pt")
                logger.info(">> checkpoint 已保存: %s", path)

            # ---- 自我对弈挑战 ----
            if (cfg.selfplay_every > 0
                    and self.episode % cfg.selfplay_every == 0):
                self._run_selfplay_cycle()

        # ---- 训练结束收尾 ----
        self.save_checkpoint("latest.pt")
        self._write_status(running=False)
        elapsed = time.time() - start_time
        logger.info("训练完成: %d 局, 用时 %.1f 分钟, 最终 checkpoint 已保存",
                    total, elapsed / 60)

    def _run_eval_cycle(self) -> None:
        """评估周期: 纯推理评估 + 更新图表 + 保存最佳模型。"""
        stats = self.evaluate()
        stats["episode"] = self.episode
        self.history["evals"].append(stats)
        self.viz.plot(self.history)   # 每 100 局刷新监控图表

        logger.info("-" * 62)
        logger.info("评估 @ 局 %d (%d 局): 平均分 %.0f | 最高 %d | 中位 %.0f | "
                    "平均存活 %.0f 步",
                    self.episode, stats["games"], stats["avg"], stats["max"],
                    stats["median"], stats["avg_steps"])
        logger.info("里程碑达成率: 512 %.0f%% | 1024 %.0f%% | 2048 %.0f%% | "
                    "4096 %.0f%%",
                    stats["ms512"] * 100, stats["ms1024"] * 100,
                    stats["ms2048"] * 100, stats["ms4096"] * 100)

        if stats["avg"] > self.best_eval_score:
            self.best_eval_score = stats["avg"]
            atomic_torch_save({
                "policy_sd": self.agent.state_dict_cpu(),
                "episode": self.episode,
                "eval_avg": stats["avg"],
                "eval_stats": stats,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, os.path.join(self.models_dir, "best.pt"))
            logger.info("★ 新最佳平均分 %.0f, 已保存 best.pt", stats["avg"])
        logger.info("-" * 62)
        self._write_status()          # 评估后刷新状态

    def _run_selfplay_cycle(self) -> None:
        """自我对弈周期: 当前 policy_net 挑战磁盘上的冠军模型。"""
        try:
            result = self.selfplay.challenge(self.agent.state_dict_cpu())
            result["episode"] = self.episode
            self.history["selfplay"].append({
                "episode": self.episode,
                "passed": result["passed"],
                "challenger_avg": result["challenger_avg"],
                "champion_avg": result.get("champion_avg"),
                "version": result["version"],
            })
        except Exception as exc:
            logger.exception("自我对弈挑战异常: %s", exc)

    def close(self) -> None:
        """释放资源（MCTS 进程池等）。"""
        if self.mcts is not None:
            self.mcts.close()


# ---------------- 模块自测 ----------------
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cfg = TrainConfig(episodes=3, train_start=64, eval_every=2,
                          eval_games=2, ckpt_every=2, selfplay_every=3,
                          selfplay_games=2, use_mcts=False, out_dir=td)
        trainer = DQNTrainer(cfg)
        trainer.train(3)
        ckpt = trainer.save_checkpoint("latest.pt")
        # 验证 checkpoint 可恢复
        trainer2 = DQNTrainer(cfg)
        trainer2.load_checkpoint(ckpt)
        assert trainer2.episode == 3
        assert len(trainer2.history["episode_scores"]) == 3
        trainer.close()
        trainer2.close()
    print("train.py 自测通过 ✓")
