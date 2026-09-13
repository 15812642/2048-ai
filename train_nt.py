# -*- coding: utf-8 -*-
"""
train_nt.py - N-Tuple + TD(λ) 训练引擎（SOTA 路线）

训练流程（对应 Szubert 2014 / Guei 2022 的标准做法）：
    1. 用当前 V 以 1-ply 贪心走一局，记录 afterstate 序列与动作得分；
    2. 局终后【自后向前】批量更新：
         G_t = r_{t+1} + γ[(1-λ)·V(s'_{t+1}) + λ·G_{t+1}]
         w   ← w + α·δ·(权重增量裁剪)
       从末尾开始更新的原因：最后一个 afterstate 的真实价值恰为 0，
       误差最小，向前逐层修正的效率远高于逐步在线更新。
    3. 周期评估（1-ply 贪心 / 可选 expectimax），保存 checkpoint。

为什么这套方法能碾压 RL：
    - afterstate 隔离随机性 → 确定性 MDP；
    - 查表式权重互相独立 → 一次 UPDATE 不影响无关状态，可批量反向更新；
      神经网络共享权重，只能逐步更新，收敛慢一个数量级；
    - 8 重对称采样 → 样本量免费 ×8。
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from ntuple import (PATTERNS_5, PATTERNS_6, NTupleNetwork, decode_board,
                    greedy_action, move_all, new_board, spawn)
try:
    from train import setup_logging     # 服务器上复用 v1 的日志设置
except ImportError:
    def setup_logging(out_dir: str, level: int = logging.INFO) -> None:
        """内置日志设置（手机端无 train.py 时的回退实现）。

        功能与原版一致: 同时输出到控制台与 <out_dir>/logs/train.log。
        之所以内联: train.py 会拉入 v1 的完整依赖链（torch/mcts/selfplay），
        手机端只需 n-tuple 相关模块，不应被迫安装这些重依赖。
        """
        log_dir = os.path.join(out_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        lg = logging.getLogger("2048")
        lg.setLevel(level)
        if lg.handlers:
            return
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        lg.addHandler(sh)
        try:
            fh = logging.FileHandler(os.path.join(log_dir, "train.log"),
                                     encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"))
            lg.addHandler(fh)
        except Exception:
            pass

logger = logging.getLogger("2048")


@dataclass
class NTupleConfig:
    """N-Tuple / TD(λ) 训练配置。"""
    episodes: int = 100_000             # 训练局数
    alpha: float = 0.32                 # 总学习率（会按查找次数 K 均摊到每个权重）
    alpha_decay_every: int = 0          # 每 N 局把 α 乘以衰减系数（0=不衰减）
    alpha_decay_gamma: float = 0.8      # α 衰减系数
    alpha_min: float = 0.02             # α 下限
    lam: float = 0.0                    # TD(λ) 的 λ。0=TD(0)（实测最优）；
                                        # λ≥0.9 会数值发散，不建议
    gamma: float = 1.0                  # 折扣（2048 用 1.0 = 累计得分）
    v_init: float = 0.0                 # 乐观初始化值。
                                        # 【实测】320000 反而严重拖慢收敛：
                                        # 初始 V≈V' 使 δ≈r 很小，网络只学相对差
                                        # 异、无法把整体水平降下来（V/实际 = 29 倍）；
                                        # 改为 0 后同样局数得分提升 2.2 倍且标定良好。
    weight_clip: Optional[float] = None  # δ 裁剪上限（None=不裁剪）。
                                         #  δ 为分数量级（中位 15 / 最大 32 万），
                                         #  曾误设 1.0 导致 97.7% 信号被丢弃。
    epsilon_start: float = 0.0          # 探索率（OI 已提供探索，默认 0）
    epsilon_end: float = 0.0
    epsilon_decay_episodes: int = 5_000

    tuple_len: int = 6                  # 元组长度（6 = 论文最优 8x6 配置）
    pattern_set: str = "8"              # 图案集: "8"/"12"/"5"（见 ntuple.PATTERN_SETS）
    max_tile_exp: int = 15              # 编码上限（15 = 32768）

    eval_every: int = 2_000             # 每 N 局评估
    eval_games: int = 200               # 评估局数
    ckpt_every: int = 10_000            # 每 N 局保存 checkpoint（latest.npz）
    snapshot_every: int = 50_000        # 每 N 局保存里程碑快照（可回看各阶段水平）
    max_snapshots: int = 6              # 里程碑快照最多保留个数（自动清理最旧）
    log_every: int = 500                # 每 N 局打日志

    search_depth: int = 0               # 评估时的 expectimax 深度（0 = 1-ply 贪心）
    seed: int = 42
    out_dir: str = "outputs_v3"


def _make_searcher(net, depth: int):
    """按深度选择搜索实现（优先向量化）。

    实测加速比（vs 逐叶子）:
        2-ply: 3.13 ms/步 -> 0.92 ms/步  (3.4x)
        3-ply: 210 ms/步  -> 19 ms/步    (11x, 决策一致率 97-100%)
    向量化版本与逐叶子实现数学等价，已在边界局面（满盘/死局）验证。
    """
    if depth == 2:
        for mod in ("v4.vecsearch", "vecsearch"):
            try:
                m = __import__(mod, fromlist=["VecExpectimax"])
                return m.VecExpectimax(net, max_empties=10)
            except Exception:
                continue
    elif depth == 3:
        for mod in ("v4.vecsearch3", "vecsearch3"):
            try:
                m = __import__(mod, fromlist=["VecExpectimax3"])
                return m.VecExpectimax3(net, max_empties=8)
            except Exception:
                continue
    from expectimax import ExpectimaxSearcher
    return ExpectimaxSearcher(net, depth=depth)


class NTupleTrainer:
    """N-Tuple + TD(λ) 训练器。"""

    def __init__(self, cfg: NTupleConfig):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.models_dir = os.path.join(cfg.out_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        setup_logging(cfg.out_dir)

        self.rng = np.random.default_rng(cfg.seed)
        from ntuple import PATTERN_SETS
        if cfg.pattern_set in PATTERN_SETS:
            patterns = PATTERN_SETS[cfg.pattern_set]
        else:                            # 回退: 按元组长度取默认集
            patterns = PATTERNS_6 if cfg.tuple_len == 6 else PATTERNS_5
        self.net = NTupleNetwork(patterns, v_init=cfg.v_init, seed=cfg.seed)

        self.episode = 0
        self._start_episode = 0      # 本次进程启动时的起始局数（续训时 >0）
        self._start_time = time.time()
        # status.json 按【时间】节流写入（与训练速度解耦）。
        # 早期版本挂在 log_every（每 500 局）上，导致前端每 2s 轮询却
        # 要等约 38 秒才见一次变化，看起来像"卡住"。
        self._last_status_t = 0.0
        self.status_interval = 1.5      # 秒：最小写入间隔
        self._recent: List[int] = []
        self.best_eval_score = float("-inf")
        self.status_path = os.path.join(cfg.out_dir, "status.json")
        self.history: Dict[str, list] = {
            "episode_scores": [], "evals": [], "losses": [],
            "epsilons": [], "episode_steps": [], "episode_max": [],
        }

        logger.info("=" * 70)
        logger.info("N-Tuple + TD(λ) 训练器初始化（SOTA 路线）")
        logger.info("  网络    : %d 个 %d-元组(图案集 %s) x 8 重对称 = %d 次查找",
                    self.net.n_patterns, self.net.tuple_len,
                    cfg.pattern_set, self.net.K)
        logger.info("  权重    : %s 个 (%.1f MB)",
                    f"{self.net.n_weights:,}", self.net.size_mb())
        logger.info("  学习    : TD(λ) α=%.3f λ=%.2f γ=%.1f | 增量裁剪 %.1f",
                    cfg.alpha, cfg.lam, cfg.gamma, cfg.weight_clip)
        logger.info("  乐观初始化: V_init = %s（单条目 %.1f）",
                    f"{cfg.v_init:,.0f}", cfg.v_init / self.net.K)
        logger.info("=" * 70)

    # ================= 状态文件（Web 控制台） =================

    def _write_status(self, running: bool = True) -> None:
        try:
            scores = self.history["episode_scores"]
            status = {
                "running": running,
                "algo": "ntuple_v3",
                "episode": self.episode,
                "total_steps": sum(self.history["episode_steps"][-1000:]),
                "epsilon": round(self.epsilon(), 4),
                "buffer_size": 0,
                "recent_scores": [int(s) for s in scores[-300:]],
                "recent_scores_start": max(0, len(scores) - 300),
                "epsilons_tail": [round(float(e), 4)
                                  for e in self.history["epsilons"][-300:]],
                "losses_tail": [round(float(x), 3)
                                for x in self.history["losses"][-2000:]],
                "evals": self.history["evals"][-60:],
                "selfplay": [],
                "best_eval_score": (None if self.best_eval_score == float("-inf")
                                    else round(self.best_eval_score, 1)),
                "selfplay_version": 0,
                "mcts_backend": None,
                "use_mcts": False,
                "net_weights": self.net.n_weights,
                "snapshots": len(__import__("glob").glob(
                    os.path.join(self.models_dir, "snap_ep*.npz"))),
                "net_size_mb": round(self.net.size_mb(), 1),
                "n_patterns": self.net.n_patterns,
                "tuple_len": self.net.tuple_len,
                "search_depth": self.cfg.search_depth,
                "elapsed_sec": round(time.time() - self._start_time, 1),
                "episodes_per_sec": round(self._rate(), 2),
                "processed_this_run": self.episode - self._start_episode,
                "eta_sec": self._eta_seconds(),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": "cpu",
                "config": asdict(self.cfg),
            }
            tmp = f"{self.status_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False)
            os.replace(tmp, self.status_path)
        except Exception as exc:
            logger.debug("status.json 写入失败: %s", exc)

    def _rate(self) -> float:
        """本次进程的平均训练速度（局/秒），已扣除续训前的基础局数。"""
        el = time.time() - self._start_time
        done = self.episode - self._start_episode
        return (done / el) if (el > 0 and done > 0) else 0.0

    def _eta_seconds(self) -> Optional[float]:
        """基于当前速度预估剩余时间（秒）；目标未知或已完成返回 None。"""
        target = self.cfg.episodes or 0
        if not target or self.episode >= target:
            return None
        rate = self._rate()
        if rate <= 0:
            return None
        return round((target - self.episode) / rate, 0)

    def _maybe_write_status(self, force: bool = False) -> None:
        """按时间节流写 status.json，保证前端 2s 轮询能看到实时局数。

        与训练速度解耦：快时（20 局/秒）每 ~1.5s 写一次，
        慢时（1 局/秒）每局都写，前端始终近似实时。
        """
        now = time.time()
        if force or (now - self._last_status_t) >= self.status_interval:
            self._last_status_t = now
            self._write_status()

    # ================= 训练 =================

    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.episode / max(1, c.epsilon_decay_episodes))
        return c.epsilon_start + frac * (c.epsilon_end - c.epsilon_start)

    def run_episode(self) -> Dict:
        """走一局并做一次反向 TD(λ) 更新。"""
        cfg = self.cfg
        net = self.net
        rng = self.rng
        eps = self.epsilon()

        board = new_board(rng)
        afterstates: List[np.ndarray] = []
        rewards: List[float] = []
        values: List[float] = []          # 动作选择时顺带得到的 V(afterstate)

        while True:
            res = greedy_action(net, board, epsilon=eps, rng=rng)
            if res is None:
                break
            _, after, sc, v = res
            afterstates.append(after)
            rewards.append(float(sc))
            values.append(v)
            board = spawn(after.copy(), rng)

        T = len(afterstates)
        if T == 0:
            return {"score": 0, "steps": 0, "max_tile": 0, "delta": 0.0}

        # 局终后反向批量更新（核心步骤）
        mean_delta, mean_v = net.update_episode(
            np.asarray(afterstates, dtype=np.uint8),
            np.asarray(rewards, dtype=np.float32),
            alpha=cfg.alpha, lam=cfg.lam, gamma=cfg.gamma,
            clip=cfg.weight_clip, precomputed_v=values)

        return {"score": int(sum(rewards)), "steps": T,
                "max_tile": int(decode_board(afterstates[-1]).max()),
                "delta": mean_delta, "mean_v": mean_v}

    def train(self, episodes: Optional[int] = None) -> None:
        cfg = self.cfg
        total = episodes or cfg.episodes
        start = time.time()
        logger.info("开始 v3 (N-Tuple) 训练: %s 局", f"{total:,}")

        for _ in range(total):
            self.episode += 1
            try:
                stats = self.run_episode()
            except KeyboardInterrupt:
                logger.info("收到中断, 保存进度...")
                self.save_checkpoint("latest.npz")
                self._write_status(running=False)
                raise
            except Exception as exc:
                logger.exception("第 %d 局异常, 跳过: %s", self.episode, exc)
                self.episode -= 1
                continue

            self.history["episode_scores"].append(stats["score"])
            self.history["episode_steps"].append(stats["steps"])
            self.history["episode_max"].append(stats["max_tile"])
            self.history["epsilons"].append(self.epsilon())
            self.history["losses"].append(stats.get("delta", 0.0))
            self._recent.append(stats["score"])
            if len(self._recent) > 200:
                self._recent.pop(0)

            # 每局都尝试刷新状态文件（内部按时间节流 -> 前端近似实时）
            self._maybe_write_status()

            if self.episode % cfg.log_every == 0:
                el = time.time() - start
                rate = self._rate()
                recent = np.asarray(self._recent[-200:])
                eta = self._eta_seconds()
                eta_str = ("%.1f 小时" % (eta / 3600)) if eta else "–"
                logger.info(
                    "局 %8s | 近200局均分 %9.0f | 近200局最高 %8d | "
                    "|δ| %8.2f | 速度 %6.1f 局/秒 | 已用 %.1f 分钟 | 剩余 %s",
                    f"{self.episode:,}", recent.mean(), recent.max(),
                    np.mean(self.history["losses"][-500:]),
                    rate, el / 60, eta_str)
                self._maybe_write_status(force=True)

            if cfg.eval_every > 0 and self.episode % cfg.eval_every == 0:
                self._eval_cycle()

            if cfg.ckpt_every > 0 and self.episode % cfg.ckpt_every == 0:
                self.save_checkpoint("latest.npz")
                logger.info(">> checkpoint 已保存 (局 %s)", f"{self.episode:,}")

            # 里程碑快照: 保留各训练阶段的模型, 便于对比与回退
            if (cfg.snapshot_every > 0
                    and self.episode % cfg.snapshot_every == 0):
                name = f"snap_ep{self.episode}.npz"
                self.save_checkpoint(name)
                self._prune_snapshots()
                logger.info(">> 里程碑快照已保存: %s", name)

        self.save_checkpoint("latest.npz")
        self._write_status(running=False)
        logger.info("v3 训练完成: %s 局, 用时 %.1f 分钟",
                    f"{total:,}", (time.time() - start) / 60)

    # ================= 评估 =================

    def evaluate(self, games: Optional[int] = None,
                 search_depth: Optional[int] = None,
                 max_moves: int = 20000) -> Dict:
        """评估：默认 1-ply 贪心；search_depth>0 时用 expectimax。"""
        n = games or self.cfg.eval_games
        depth = self.cfg.search_depth if search_depth is None else search_depth
        net, rng = self.net, self.rng

        searcher = None
        if depth and depth > 0:
            searcher = _make_searcher(net, depth)

        scores, steps_l, maxes, tiles_count = [], [], [], {}
        for _ in range(n):
            try:
                board = new_board(rng)
                score, steps = 0, 0
                while steps < max_moves:
                    if searcher is not None:
                        a = searcher.best_action(board)
                        if a is None:
                            break
                        _bs, _ss, _ms = move_all(board)
                        nb, sc = _bs[a], int(_ss[a])
                    else:
                        res = greedy_action(net, board)
                        if res is None:
                            break
                        a, nb, sc, _v = res
                    score += int(sc)
                    board = spawn(nb.copy(), rng)
                    steps += 1
                mt = int(decode_board(board).max())
                scores.append(score)
                steps_l.append(steps)
                maxes.append(mt)
                tiles_count[mt] = tiles_count.get(mt, 0) + 1
            except Exception as exc:
                logger.warning("评估局异常: %s", exc)

        if not scores:
            return {"avg": 0.0, "max": 0, "median": 0.0, "avg_steps": 0.0,
                    "ms512": 0.0, "ms1024": 0.0, "ms2048": 0.0, "ms4096": 0.0,
                    "games": 0, "tiles": {}}

        def reach(t: int) -> float:
            return sum(1 for x in maxes if x >= t) / len(maxes)

        return {"avg": float(np.mean(scores)), "max": int(np.max(scores)),
                "median": float(np.median(scores)),
                "avg_steps": float(np.mean(steps_l)),
                "ms512": reach(512), "ms1024": reach(1024),
                "ms2048": reach(2048), "ms4096": reach(4096),
                "games": len(scores), "tiles": tiles_count}

    def _eval_cycle(self) -> None:
        stats = self.evaluate()
        stats["episode"] = self.episode
        self.history["evals"].append(stats)

        logger.info("-" * 70)
        logger.info("★ 评估 @ 局 %s (%d 局): 平均分 %s | 最高 %s | 中位 %s | "
                    "平均 %s 步",
                    f"{self.episode:,}", stats["games"],
                    f"{stats['avg']:,.0f}", f"{stats['max']:,}",
                    f"{stats['median']:,.0f}", f"{stats['avg_steps']:.0f}")
        logger.info("  达成率: 512 %.0f%% | 1024 %.0f%% | 2048 %.0f%% | 4096 %.0f%%",
                    stats["ms512"] * 100, stats["ms1024"] * 100,
                    stats["ms2048"] * 100, stats["ms4096"] * 100)
        top = sorted(stats["tiles"].items(), reverse=True)[:6]
        logger.info("  最大方块分布: %s",
                    " | ".join(f"{k}:{v}" for k, v in top))
        logger.info("-" * 70)

        if stats["avg"] > self.best_eval_score:
            self.best_eval_score = stats["avg"]
            self.save_checkpoint("best.npz")
            logger.info("★ 新最佳平均分 %s, 已保存 best.npz", f"{stats['avg']:,.0f}")

    # ================= 存储 =================

    def save_checkpoint(self, name: str = "latest.npz") -> str:
        path = os.path.join(self.models_dir, name)
        self.net.save(path)
        meta = {
            "episode": self.episode,
            "best_eval_score": self.best_eval_score,
            "config": asdict(self.cfg),
            "history": {k: (v[-2000:] if isinstance(v, list) else v)
                        for k, v in self.history.items()},
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = os.path.join(self.models_dir, name + ".meta.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.models_dir, name + ".meta.json"))
        return path

    def _prune_snapshots(self) -> None:
        """只保留最新的 max_snapshots 个里程碑快照，避免磁盘无限增长。"""
        import glob as _glob
        snaps = sorted(_glob.glob(os.path.join(self.models_dir, "snap_ep*.npz")),
                       key=lambda p: os.path.getmtime(p))
        for path in snaps[:max(0, len(snaps) - self.cfg.max_snapshots)]:
            try:
                os.remove(path)
                meta = path + ".meta.json"
                if os.path.exists(meta):
                    os.remove(meta)
            except OSError:
                pass

    def load_checkpoint(self, name: str = "latest.npz") -> bool:
        path = os.path.join(self.models_dir, name)
        if not os.path.exists(path):
            return False
        self.net.load(path)
        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.episode = int(meta.get("episode", 0))
                self.best_eval_score = float(
                    meta.get("best_eval_score", float("-inf")))
                for k, v in (meta.get("history") or {}).items():
                    if k in self.history and isinstance(v, list):
                        self.history[k] = v
            except Exception as exc:
                logger.warning("meta 恢复失败: %s", exc)
        # 速率/ETA 必须基于"本次进程新增的局数"，否则续训后会算成天文数字
        self._start_episode = self.episode
        logger.info("已恢复 v3 checkpoint: 第 %s 局", f"{self.episode:,}")
        return True

    def close(self) -> None:
        pass


# ---------------- 命令行入口 ----------------
def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="2048 N-Tuple + TD(λ) 训练")
    p.add_argument("--episodes", type=int, default=100_000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-dir", default="outputs_v3")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--v-init", type=float, default=None)
    p.add_argument("--tuple-len", type=int, default=None, choices=[5, 6])
    p.add_argument("--patterns", type=str, default=None,
                   help="图案集: 8(默认,512MB) / 12(768MB) / 5(48MB,最快)")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-games", type=int, default=None)
    p.add_argument("--search-depth", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = NTupleConfig(episodes=args.episodes, out_dir=args.out_dir,
                       seed=args.seed)
    if args.patterns is not None:
        cfg.pattern_set = args.patterns
    for attr in ("alpha", "lam", "tuple_len", "eval_every", "eval_games",
                 "search_depth"):
        v = getattr(args, attr.replace("-", "_"))
        if v is not None:
            setattr(cfg, attr, v)
    if args.v_init is not None:
        cfg.v_init = args.v_init

    tr = NTupleTrainer(cfg)
    try:
        if args.resume:
            tr.load_checkpoint("latest.npz")
        tr.train(args.episodes)
    except KeyboardInterrupt:
        logger.info("用户中断, 进度已保存")
    finally:
        tr.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
