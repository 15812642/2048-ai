# -*- coding: utf-8 -*-
"""
main.py - 2048 自我对弈 AI 训练系统 · 主入口

用法
----
    python main.py --train                          # 默认配置开始训练
    python main.py --train --resume                 # 从最近 checkpoint 恢复训练
    python main.py --train --episodes 2000          # 指定训练局数
    python main.py --train --parallel 4             # 指定 MCTS 并行进程数
    python main.py --train --no-mcts                # 纯 ε-greedy 训练（最快）
    python main.py --eval                           # 评估最佳模型（默认 20 局）
    python main.py --eval --eval-games 50           # 指定评估局数
    python main.py --play                           # 可视化观看 AI 玩一局
    python main.py --play --play-delay 0.05         # 加快播放速度

输出目录结构（--out-dir, 默认 outputs/）
-----------------------------------------
    outputs/
    ├── models/                 # best.pt / latest.pt / checkpoint_epN.pt / versions.json
    ├── logs/train.log          # 完整训练日志
    └── plots/                  # training_progress.png 监控面板
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch

from env import ACTION_NAMES, Game2048
from model import DQNAgent, load_torch
from selfplay import play_greedy_games
from train import DQNTrainer, TrainConfig

logger = logging.getLogger("2048")


# ---------------- 命令行解析 ----------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2048 自我对弈 AI 训练系统 (DQN + MCTS + Self-Play)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="开始训练（默认模式）")
    mode.add_argument("--eval", dest="eval_mode", action="store_true",
                      help="加载模型进行评估")
    mode.add_argument("--play", action="store_true", help="可视化观看 AI 玩一局")

    parser.add_argument("--resume", action="store_true",
                        help="从 outputs/models/latest.pt 恢复训练")
    parser.add_argument("--algo", choices=["v1", "v2", "v3"], default="v1",
                        help="算法版本: v1=经典 DQN; v2=Rainbow-Lite"
                             "(H-DQN 复刻); v3=N-Tuple+TD(λ)+Expectimax"
                             "（SOTA 路线, 平均分可达数十万）")
    parser.add_argument("--n-step", type=int, default=None,
                        help="v2 的 n 步回报步数（默认 3）")
    parser.add_argument("--train-every", type=int, default=None,
                        help="v2 每 N 个环境步训练一次（默认 4）")
    # ---- v3 (N-Tuple) 专属 ----
    parser.add_argument("--nt-len", type=int, default=None, choices=[5, 6],
                        help="v3 元组长度（6=论文最优, 5=更省内存更快）")
    parser.add_argument("--nt-patterns", type=str, default=None,
                        help="v3 图案集: 8(默认,512MB) / 12(768MB) / 5(48MB,最快)")
    parser.add_argument("--nt-alpha", type=float, default=None,
                        help="v3 学习率（默认 0.1）")
    parser.add_argument("--nt-lambda", type=float, default=None,
                        help="v3 TD(λ) 的 λ（默认 0.5）")
    parser.add_argument("--nt-vinit", type=float, default=None,
                        help="v3 乐观初始化值（默认 320000, 0=关闭）")
    parser.add_argument("--search-depth", type=int, default=None,
                        help="v3 评估时的 expectimax 深度（0=1-ply 贪心）")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="v3 每 N 局评估一次（默认 2000）")
    parser.add_argument("--snapshot-every", type=int, default=None,
                        help="v3 每 N 局保存里程碑快照（默认 50000, 0=关闭）")
    parser.add_argument("--ckpt-every", type=int, default=None,
                        help="v3 每 N 局保存 checkpoint（默认 10000；"
                             "调小可减少进程重启时的进度损失）")
    parser.add_argument("--episodes", type=int, default=None,
                        help="训练局数（默认 5000）")
    parser.add_argument("--parallel", type=int, default=0,
                        help="MCTS 并行进程数, 0 = CPU 核数 - 1")
    parser.add_argument("--no-mcts", action="store_true",
                        help="禁用 MCTS（纯 ε-greedy, 训练最快）")
    parser.add_argument("--mcts-interval", type=int, default=None,
                        help="每 N 局插入 1 局 MCTS 决策局（默认 20）")
    parser.add_argument("--mcts-sims", type=int, default=None,
                        help="MCTS 每次决策的模拟次数（默认 100）")
    parser.add_argument("--eval-games", type=int, default=None,
                        help="评估局数（默认 20）")
    parser.add_argument("--eps-decay", type=int, default=None,
                        help="ε 线性衰减周期（环境步, 默认 10000; "
                             "长时间训练建议 200000+ 保证充分探索）")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="PyTorch 计算线程数（默认 1; 小网络单线程最快"
                             "且不与 MCTS 争抢 CPU）")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="单局最大步数上限（默认 3000, 防止无限存活"
                             "退化解阻塞训练）")
    parser.add_argument("--mcts-games", type=int, default=None,
                        help="每次自我对弈挑战双方各玩局数（默认 50）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"], help="计算设备")
    parser.add_argument("--out-dir", default="outputs", help="输出根目录")
    parser.add_argument("--play-delay", type=float, default=0.25,
                        help="--play 模式每步播放间隔秒数")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """命令行参数 -> TrainConfig。"""
    cfg = TrainConfig(seed=args.seed, device=args.device, out_dir=args.out_dir)
    if args.episodes is not None:
        cfg.episodes = args.episodes
    if args.parallel is not None:
        cfg.parallel = args.parallel
    if args.no_mcts:
        cfg.use_mcts = False
    if args.mcts_interval is not None:
        cfg.mcts_interval = args.mcts_interval
    if args.mcts_sims is not None:
        cfg.mcts_simulations = args.mcts_sims
    if args.eps_decay is not None:
        cfg.eps_decay_steps = args.eps_decay
    if args.mcts_games is not None:
        cfg.selfplay_games = args.mcts_games
    if args.torch_threads is not None:
        cfg.torch_threads = args.torch_threads
    if args.max_steps is not None:
        cfg.max_episode_steps = args.max_steps
    return cfg


# ---------------- 模式实现 ----------------

def cmd_train(args: argparse.Namespace) -> None:
    """训练模式（--algo 选择 v1 经典 DQN 或 v2 Rainbow-Lite）。"""
    if args.algo == "v2":
        _cmd_train_v2(args)
        return
    if args.algo == "v3":
        _cmd_train_v3(args)
        return

    cfg = build_config(args)
    trainer = DQNTrainer(cfg)
    try:
        if args.resume:
            latest = os.path.join(trainer.models_dir, "latest.pt")
            trainer.load_checkpoint(latest)
        trainer.train(args.episodes)   # resume 时 None -> 使用配置局数（增量）
    except KeyboardInterrupt:
        logger.info("用户中断, 进度已保存")
    finally:
        trainer.close()


def _cmd_train_v2(args: argparse.Namespace) -> None:
    """v2 (Rainbow-Lite / H-DQN 复刻) 训练路径。

    输出目录固定为 <out_dir>/v2, 与 v1 数据隔离, 便于 A/B 对比。
    """
    from train_v2 import RainbowConfig, RainbowTrainer

    cfg = RainbowConfig(episodes=args.episodes or 5000,
                        out_dir=os.path.join(args.out_dir, "v2"),
                        seed=args.seed, device=args.device)
    if args.torch_threads is not None:
        cfg.torch_threads = args.torch_threads
    if args.n_step is not None:
        cfg.n_step = args.n_step
    if args.train_every is not None:
        cfg.train_every = args.train_every
    if args.eval_games is not None:
        cfg.eval_games = args.eval_games

    trainer = RainbowTrainer(cfg)
    try:
        if args.resume:
            latest = os.path.join(trainer.models_dir, "latest.pt")
            if os.path.exists(latest):
                trainer.load_checkpoint(latest)
            else:
                logger.warning("未找到 v2 checkpoint, 从头开始训练")
        trainer.train(args.episodes)
    except KeyboardInterrupt:
        logger.info("用户中断, 进度已保存")
    finally:
        trainer.close()


def _cmd_eval_v2(args: argparse.Namespace) -> None:
    """v2 模型评估。"""
    from train_v2 import RainbowConfig, RainbowTrainer

    cfg = RainbowConfig(out_dir=os.path.join(args.out_dir, "v2"),
                        device=args.device)
    trainer = RainbowTrainer(cfg)
    try:
        path = next((p for p in (os.path.join(trainer.models_dir, n)
                                 for n in ("best.pt", "latest.pt"))
                     if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(
                f"未找到 v2 模型（{trainer.models_dir}）, 请先训练")
        ckpt = load_torch(path, map_location="cpu")
        trainer.agent.load_both_state_dict(ckpt["policy_sd"])
        n = args.eval_games or 20
        logger.info("评估 v2 模型: %s | %d 局纯推理", path, n)
        stats = trainer.evaluate(n)
        logger.info("=" * 50)
        logger.info("平均分   : %8.1f", stats["avg"])
        logger.info("最高分   : %8d", stats["max"])
        logger.info("中位数   : %8.1f", stats["median"])
        logger.info("平均存活 : %8.0f 步", stats["avg_steps"])
        for m, key in ((512, "ms512"), (1024, "ms1024"),
                       (2048, "ms2048"), (4096, "ms4096")):
            logger.info("达到 %5d : %7.1f%%", m, stats[key] * 100)
        logger.info("=" * 50)
    finally:
        trainer.close()


def _build_nt_config(args, out_dir: str):
    """命令行参数 -> NTupleConfig。"""
    from train_nt import NTupleConfig
    cfg = NTupleConfig(episodes=args.episodes or 200_000, out_dir=out_dir,
                       seed=args.seed)
    if args.nt_len is not None:
        cfg.tuple_len = args.nt_len
    if args.nt_patterns is not None:
        cfg.pattern_set = args.nt_patterns
    if args.nt_alpha is not None:
        cfg.alpha = args.nt_alpha
    if args.nt_lambda is not None:
        cfg.lam = args.nt_lambda
    if args.nt_vinit is not None:
        cfg.v_init = args.nt_vinit
    if args.search_depth is not None:
        cfg.search_depth = args.search_depth
    if getattr(args, "eval_every", None) is not None:
        cfg.eval_every = args.eval_every
    if getattr(args, "ckpt_every", None) is not None:
        cfg.ckpt_every = args.ckpt_every
    if getattr(args, "snapshot_every", None) is not None:
        cfg.snapshot_every = args.snapshot_every
    if args.eval_games is not None:
        cfg.eval_games = args.eval_games
    return cfg


def _cmd_train_v3(args: argparse.Namespace) -> None:
    """v3 (N-Tuple + TD(λ) + Expectimax) 训练路径。

    输出目录固定为 <out_dir>/v3, 与 v1/v2 数据完全隔离。
    """
    from train_nt import NTupleTrainer

    trainer = NTupleTrainer(_build_nt_config(args, os.path.join(args.out_dir, "v3")))
    try:
        if args.resume:
            if not trainer.load_checkpoint("latest.npz"):
                logger.warning("未找到 v3 checkpoint, 从头开始训练")
        trainer.train(args.episodes)
    except KeyboardInterrupt:
        logger.info("用户中断, 进度已保存")
    finally:
        trainer.close()


def _cmd_eval_v3(args: argparse.Namespace) -> None:
    """v3 模型评估（可选 expectimax 搜索）。"""
    from train_nt import NTupleTrainer

    trainer = NTupleTrainer(
        _build_nt_config(args, os.path.join(args.out_dir, "v3")))
    try:
        if not trainer.load_checkpoint("best.npz"):
            if not trainer.load_checkpoint("latest.npz"):
                raise FileNotFoundError(
                    f"未找到 v3 模型（{trainer.models_dir}）, 请先训练")
        n = args.eval_games or 20
        depth = trainer.cfg.search_depth
        logger.info("评估 v3 模型: %d 局, expectimax 深度 %d", n, depth)
        stats = trainer.evaluate(n)
        logger.info("=" * 56)
        logger.info("平均分   : %12.1f", stats["avg"])
        logger.info("最高分   : %12d", stats["max"])
        logger.info("中位数   : %12.1f", stats["median"])
        logger.info("平均步数 : %12.0f", stats["avg_steps"])
        for m, key in ((512, "ms512"), (1024, "ms1024"),
                       (2048, "ms2048"), (4096, "ms4096")):
            logger.info("达到 %5d : %11.1f%%", m, stats[key] * 100)
        logger.info("最大方块分布: %s", stats.get("tiles", {}))
        logger.info("=" * 56)
    finally:
        trainer.close()


def _cmd_play_v3(args: argparse.Namespace) -> None:
    """v3 观赏模式: 渲染 N-Tuple 智能体的对局。"""
    from ntuple import decode_board, greedy_action, new_board, spawn
    from train_nt import NTupleTrainer
    import numpy as _np

    trainer = NTupleTrainer(
        _build_nt_config(args, os.path.join(args.out_dir, "v3")))
    try:
        if not (trainer.load_checkpoint("best.npz")
                or trainer.load_checkpoint("latest.npz")):
            raise FileNotFoundError("未找到 v3 模型, 请先训练")
        net = trainer.net
        rng = _np.random.default_rng()
        board = new_board(rng)
        score, step = 0, 0
        print("v3 (N-Tuple + TD(λ)) 对局演示")

        def render(b, sc, st):
            real = decode_board(b)
            print("+" + "-" * 33 + "+")
            for row in real:
                print("|" + "|".join(f"{v:^7}" if v else "   .   " for v in row) + "|")
            print("+" + "-" * 33 + "+")
            print(f"分数: {sc}   步数: {st}")

        render(board, score, step)
        while True:
            time.sleep(args.play_delay)
            res = greedy_action(net, board)
            if res is None:
                break
            a, after, sc, _v = res
            score += sc
            board = spawn(after.copy(), rng)
            step += 1
            print(f"\n第 {step} 步 -> {ACTION_NAMES[a]}")
            render(board, score, step)
        print(f"\n终局: 分数 {score:,} | 最大方块 {decode_board(board).max():,} "
              f"| 存活 {step} 步")
    except KeyboardInterrupt:
        print("\n观看中断")
    finally:
        trainer.close()


def _pick_model_path(models_dir: str) -> str:
    """按优先级选择模型文件: best.pt -> latest.pt。"""
    for name in ("best.pt", "latest.pt"):
        path = os.path.join(models_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"未找到模型文件（{models_dir}/best.pt 或 latest.pt）, 请先运行 --train")


def cmd_eval(args: argparse.Namespace) -> None:
    """评估模式: 加载最佳模型纯推理对战若干局并输出统计。"""
    if args.algo == "v2":
        _cmd_eval_v2(args)
        return
    if args.algo == "v3":
        _cmd_eval_v3(args)
        return
    from train import setup_logging
    setup_logging(args.out_dir)

    model_path = _pick_model_path(os.path.join(args.out_dir, "models"))
    ckpt = load_torch(model_path, map_location="cpu")
    n_games = args.eval_games or 20
    logger.info("评估模型: %s | %d 局纯贪心推理", model_path, n_games)

    scores, steps_l, tiles = play_greedy_games(ckpt["policy_sd"],
                                               n_games=n_games, device="cpu")
    scores_arr = np.asarray(scores)
    logger.info("=" * 50)
    logger.info("平均分   : %8.1f", scores_arr.mean())
    logger.info("最高分   : %8d", scores_arr.max())
    logger.info("中位数   : %8.1f", np.median(scores_arr))
    logger.info("平均存活 : %8.0f 步", np.mean(steps_l))
    for m in (512, 1024, 2048, 4096):
        ratio = sum(1 for t in tiles if t >= m) / len(tiles)
        logger.info("达到 %5d : %7.1f%%", m, ratio * 100)
    logger.info("各局分数 : %s", sorted(scores))
    logger.info("=" * 50)


def _cmd_play_v2(args: argparse.Namespace) -> None:
    """v2 观赏模式: 用 RainbowAgent 逐步渲染对局。"""
    from train_v2 import RainbowConfig, RainbowTrainer

    cfg = RainbowConfig(out_dir=os.path.join(args.out_dir, "v2"),
                        device=args.device)
    trainer = RainbowTrainer(cfg)
    try:
        path = next((p for p in (os.path.join(trainer.models_dir, n)
                                 for n in ("best.pt", "latest.pt"))
                     if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError("未找到 v2 模型, 请先训练")
        ckpt = load_torch(path, map_location="cpu")
        trainer.agent.load_both_state_dict(ckpt["policy_sd"])
        agent = trainer.agent
        agent.policy_net.eval()

        env = Game2048()
        env.reset()
        env.render()
        grid = [list(r) for r in env.grid]
        done, step = False, 0
        while not done:
            time.sleep(args.play_delay)
            valid = env.get_valid_actions()
            if not valid:
                break
            action = agent.select_action(grid, 0.0, valid)
            _, _, done, info = env.step(action)
            grid = [list(r) for r in env.grid]
            step += 1
            print(f"\n第 {step} 步 -> {ACTION_NAMES[action]}   "
                  f"Q 值: {[f'{q:.2f}' for q in agent.q_values(grid)]}")
            env.render()
        print(f"\n终局: 分数 {info['score']} | 最大方块 {info['max_tile']} | "
              f"存活 {step} 步")
    except KeyboardInterrupt:
        print("\n观看中断")
    finally:
        trainer.close()


def cmd_play(args: argparse.Namespace) -> None:
    """观赏模式: 逐步渲染 AI 玩一局。"""
    if args.algo == "v2":
        _cmd_play_v2(args)
        return
    if args.algo == "v3":
        _cmd_play_v3(args)
        return
    model_path = _pick_model_path(os.path.join(args.out_dir, "models"))
    ckpt = load_torch(model_path, map_location="cpu")
    version = ckpt.get("version")
    print(f"加载模型: {model_path}" +
          (f" (self-play v{version})" if version else ""))

    agent = DQNAgent()
    agent.load_both_state_dict(ckpt["policy_sd"])
    agent.policy_net.eval()

    env = Game2048()
    state = env.reset()
    env.render()
    done, step = False, 0
    try:
        while not done:
            time.sleep(args.play_delay)
            valid = env.get_valid_actions()
            if not valid:
                break
            action = agent.select_action(state, epsilon=0.0, valid_actions=valid)
            state, _, done, info = env.step(action)
            step += 1
            print(f"\n第 {step} 步 -> {ACTION_NAMES[action]}   "
                  f"Q 值: {[f'{q:.2f}' for q in agent.q_values(state)]}")
            env.render()
    except KeyboardInterrupt:
        print("\n观看中断")
    print(f"\n终局: 分数 {info['score']} | 最大方块 {info['max_tile']} | "
          f"存活 {step} 步")


# ---------------- 入口 ----------------

def main() -> int:
    args = parse_args()
    try:
        if args.eval_mode:
            cmd_eval(args)
        elif args.play:
            cmd_play(args)
        else:
            cmd_train(args)          # 默认: 训练模式
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    except Exception:
        logging.getLogger("2048").exception("运行异常")
        raise
    return 0


if __name__ == "__main__":
    # multiprocessing 安全护栏: Windows/macOS spawn 模式下必须保留
    sys.exit(main())
