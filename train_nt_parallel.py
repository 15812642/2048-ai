# -*- coding: utf-8 -*-
"""
train_nt_parallel.py - N-Tuple 并行训练（Hogwild 无锁异步更新）

为什么可以并行
--------------
N-Tuple 训练的两个阶段特性完全不同：
    1. 对局生成（greedy_action）：纯读权重表，天然可并行；
    2. 权重更新（update_episode）：稀疏写入（每局只触及极小比例的条目）。

由于写入稀疏，多个 worker 同时更新同一张表的冲突概率极低；且 TD 学习
本身是随机近似算法，偶发的写覆盖（lost update）等价于给梯度加了一点噪声，
不影响收敛 —— 这就是 **Hogwild!**（Recht et al. 2011）的思路。

架构
----
    主进程
      ├─ 创建共享内存（承载整个 LUT，512MB 只有一份物理内存）
      ├─ 启动 N 个 worker，每个直接映射该内存（零拷贝）
      ├─ 每个 worker 独立跑局 → 直接在共享表上做 TD 更新（无锁）
      └─ 汇总统计、写 status.json、保存 checkpoint

关键工程细节
------------
1. 权重表零拷贝共享：shared_memory + np.ndarray(buffer=...)；
   环境不支持时退回 memmap（文件映射）。
2. RNG 独立播种：fork 会继承父进程 RNG，必须重新播种，否则各 worker 跑出
   完全相同的对局序列。
3. 统计经 Pipe 批量回传（每 10 局一次）：避免 IPC 阻塞热路径。
4. 优雅退出：主进程经 Pipe 下发 stop → worker 退出 → 统一保存 checkpoint
   （避免多进程并发写文件）。
"""

import json
import logging
import os
import random
import tempfile
import time

# ============ 【必须在 import numpy 之前执行】 ============
# BLAS/OpenMP 线程池的多线程状态会被 fork 继承，子进程首次调用矩阵运算时
# 可能永久死锁（实测卡死在 greedy_action 的 einsum）。
# 在 numpy 加载前把线程数锁为 1，可从根本上避免 —— 且对这类小矩阵运算
# 单线程本身就是最优的。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "OPENBLAS_MAIN_FREE", "GOTOBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
# =========================================================
from dataclasses import asdict
import multiprocessing as _mp
from multiprocessing import Process, Pipe


def _proc_context():
    """选择 worker 启动方式。

    优先 fork：启动快（不重新 import 模块）、内存共享彻底。
    fork 的唯一风险是继承 BLAS 线程池导致死锁 —— 已在模块顶层把
    OMP/OPENBLAS 等线程数锁为 1 解决（必须在 import numpy 之前）。
    """
    try:
        if "fork" in _mp.get_all_start_methods():
            return _mp.get_context("fork")
    except Exception:
        pass
    return _mp.get_context()
from typing import Dict, List, Optional

# 说明: 不用 multiprocessing.Queue —— 它依赖 POSIX 信号量（/dev/shm），
# 在 Termux / proot / 精简容器中会直接抛 FileNotFoundError。
# Pipe 底层是 socketpair，无此依赖，兼容性更好。

import numpy as np

from ntuple import (PATTERN_SETS, PATTERNS_6, NTupleNetwork, decode_board,
                    greedy_action, move_all, new_board, spawn)
from train_nt import NTupleConfig, setup_logging

logger = logging.getLogger("2048")


# ================= 共享内存工具 =================

def _shm_available() -> bool:
    """检测 multiprocessing.shared_memory 是否可用（依赖 /dev/shm）。"""
    try:
        from multiprocessing import shared_memory
    except ImportError:
        return False
    try:
        shm = shared_memory.SharedMemory(create=True, size=4096)
        shm.close()
        shm.unlink()
        return True
    except Exception:
        return False


class SharedLUT:
    """跨进程共享的权重表（优先 shared_memory，退回 memmap）。"""

    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.mode = None
        self._shm = None
        self._path = None
        self.array = None
        self.name = None

    def create(self):
        """创建共享区域（主进程调用），返回 ndarray。"""
        if _shm_available():
            from multiprocessing import shared_memory
            self._shm = shared_memory.SharedMemory(create=True, size=self.nbytes)
            self.array = np.ndarray(self.shape, dtype=self.dtype,
                                    buffer=self._shm.buf)
            self.array[:] = 0.0
            self.name = self._shm.name
            self.mode = "shm"
        else:
            fd, self._path = tempfile.mkstemp(prefix="ntlut_", suffix=".dat")
            os.close(fd)
            self.array = np.memmap(self._path, dtype=self.dtype, mode="w+",
                                   shape=self.shape)
            self.array[:] = 0.0
            self.array.flush()
            self.name = self._path
            self.mode = "memmap"
        return self.array

    @staticmethod
    def attach(name: str, shape, dtype=np.float32):
        """worker 侧挂载，返回 (ndarray, keepalive)。

        【关键】必须把 SharedMemory 对象一起返回并长期持有！
        若写成:
            shm = SharedMemory(name)
            return np.ndarray(buffer=shm.buf)      # ← 危险
        函数返回后 shm 被 GC，底层映射被关闭，ndarray 变成悬垂指针，
        后续访问会直接段错误（core dumped）—— 本机实测过该崩溃。
        """
        if name.endswith(".dat"):
            arr = np.memmap(name, dtype=dtype, mode="r+", shape=shape)
            return arr, arr
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return arr, shm

    def close(self, unlink: bool = True) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
                if unlink:
                    self._shm.unlink()
            except Exception:
                pass
            self._shm = None
        if self._path and unlink and os.path.exists(self._path):
            try:
                os.remove(self._path)
            except OSError:
                pass


# ================= Worker 进程 =================

def _make_net_on_shared(patterns, lut):
    """构造一个直接使用外部数组的 NTupleNetwork（不分配自己的 LUT）。

    原理：绕过 __init__，手工设置全部必要属性。这样 worker 进程的常驻内存
    只包含网络元数据（几十 KB），512MB 的权重表由共享内存承载（全系统一份）。
    """
    net = NTupleNetwork.__new__(NTupleNetwork)
    net.patterns = [tuple(int(x) for x in p) for p in patterns]
    net.n_patterns = len(net.patterns)
    net.tuple_len = len(net.patterns[0])
    net.n_values = 16
    net.dim = 16 ** net.tuple_len
    net.dtype = lut.dtype
    net.lut = lut                            # 直接指向共享内存
    # 预计算索引结构（与 __init__ 中一致）
    from ntuple import SYM
    cells = []
    for p in net.patterns:
        for t in range(8):
            cells.append([int(SYM[t][i]) for i in p])
    net.cells = np.array(cells, dtype=np.int32)
    net.pow = (16 ** np.arange(net.tuple_len)).astype(np.int32)
    net.offsets = np.repeat(
        (np.arange(net.n_patterns, dtype=np.int64) * net.dim), 8)
    net.K = net.cells.shape[0]
    net._rng = None
    return net


def _load_searcher(net, max_empties=10):
    """尝试加载向量化 2-ply 搜索器；失败返回 None（自动回退纯贪心）。

    导入路径处理：v4 包可能位于项目根目录，需要确保 sys.path 包含它。
    """
    import sys as _sys
    cands = ["/opt/2048ai", os.path.dirname(os.path.abspath(__file__)),
             os.getcwd()]
    for c in cands:
        if c and c not in _sys.path:
            _sys.path.insert(0, c)
    try:
        from v4.vecsearch import VecExpectimax
        return VecExpectimax(net, max_empties=max_empties)
    except Exception:
        return None


def _worker_main(shm_name, shape, dtype_str, patterns, cfg_dict,
                 seed, conn, worker_id):
    """Worker 主循环：独立跑局 + 直接在共享表上做 TD 更新（无锁）。

    参数:
        shm_name : 共享内存名（或 memmap 文件路径）
        patterns : 元组图案（决定 LUT 布局，须与主进程一致）
        cfg_dict : 训练超参（只读）
        conn     : 与主进程的双向 Pipe 连接（发送统计 / 接收停止信号）
    """
    import logging as _lg
    _lg.getLogger("2048").handlers = []
    _lg.getLogger("2048").addHandler(_lg.NullHandler())

    # 【关键】限制 BLAS/OpenMP 线程数为 1。
    # fork 出的子进程会继承父进程已初始化的 BLAS 线程池状态，若父进程曾用
    # 多线程 BLAS，子进程首次调用矩阵运算时可能死锁（本机实测卡死在
    # greedy_action 的第一次 einsum）。单线程既避开死锁，又避免多进程
    # 各自开多线程互相争抢 CPU。
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"

    _keepalive = None                    # 共享内存句柄（防悬垂, 见 attach 注释）
    buf = []
    try:
        dtype = np.dtype(dtype_str)
        lut, _keepalive = SharedLUT.attach(shm_name, shape, dtype)   # 保活
        # 【内存关键】不能用 NTupleNetwork(...) 构造 —— 它会先分配一份
        # 完整 LUT（8 图案时 512MB），再被共享表替换，瞬间双倍占用导致
        # 内存不足。这里用 __new__ 跳过 __init__，只设置必要字段，
        # 直接挂载共享表，worker 的常驻内存从 ~800MB 降到 ~50MB。
        net = _make_net_on_shared(patterns, lut)

        # 独立播种（fork 会继承父进程 RNG，不重新播种则对局完全相同）
        seed = (int(seed) + worker_id * 104729) % (2 ** 31)
        rng = np.random.default_rng(seed)
        random.seed(seed)

        alpha0 = cfg_dict["alpha"]
        lam = cfg_dict["lam"]
        gamma = cfg_dict["gamma"]
        clip = cfg_dict.get("weight_clip")
        # α 衰减: 前期大步长快速学习, 后期小步长精细收敛。
        # 不衰减时权重会在最优解附近震荡而卡住（实测 15k 局后停滞）。
        decay_every = int(cfg_dict.get("alpha_decay_every") or 0)
        decay_gamma = float(cfg_dict.get("alpha_decay_gamma") or 1.0)
        alpha_min = float(cfg_dict.get("alpha_min") or 0.0)
        # ---- 搜索自举（v5 核心）: 用搜索选择动作, 让 V 从"更强的策略"中学习 ----
        # 原理: V 是"给定策略下的期望得分"。用贪心生成对局时, V 评估的是贪心策略;
        # 用搜索生成对局时, V 评估的是搜索策略（强 85%）。训练后 1-ply 贪心
        # 就能接近原搜索的水平 —— 即"内化了搜索的智慧"。
        search_prob = float(cfg_dict.get("search_prob") or 0.0)
        searcher = None
        if search_prob > 0:
            searcher = _load_searcher(net, max_empties=10)
            if searcher is None:
                search_prob = 0.0
        n_search_steps = 0
        n_all_steps = 0

        alpha = alpha0

        local_ep = 0
        while True:
            # 每 20 局检查一次停止信号（热路径不宜频繁检查）
            if local_ep % 20 == 0:
                try:
                    if conn.poll():          # 有数据 = 停止信号
                        conn.recv()
                        break
                except (EOFError, OSError):
                    break

            board = new_board(rng)
            afts: List[np.ndarray] = []
            rws: List[float] = []
            vs: List[float] = []
            while True:
                use_search_now = (searcher is not None
                                  and rng.random() < search_prob)
                if use_search_now:
                    a = searcher.best_action(board)
                    if a is None:
                        break
                    _bs, _ss, _ms = move_all(board)
                    after = _bs[a]
                    sc = int(_ss[a])
                    v = float(net.evaluate(after))
                    n_search_steps += 1
                else:
                    res = greedy_action(net, board)
                    if res is None:
                        break
                    _, after, sc, v = res
                n_all_steps += 1
                afts.append(after)
                rws.append(float(sc))
                vs.append(v)
                board = spawn(after.copy(), rng)

            if not afts:
                continue
            net.update_episode(
                np.asarray(afts, dtype=np.uint8),
                np.asarray(rws, dtype=np.float32),
                alpha=alpha, lam=lam, gamma=gamma, clip=clip,
                precomputed_v=vs)
            local_ep += 1

            # α 衰减（按本地局数, 各 worker 独立计算）
            if decay_every > 0 and local_ep % decay_every == 0:
                alpha = max(alpha_min, alpha * decay_gamma)

            # 批量上报统计（每 10 局或缓冲区满时发一次）
            buf.append((int(sum(rws)), len(afts),
                        int(decode_board(board).max()),
                        n_search_steps, n_all_steps))
            if len(buf) >= 10:
                try:
                    conn.send(buf)
                except (BrokenPipeError, OSError, EOFError):
                    break
                buf = []
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        # spawn 模式下 worker 的 stderr 不易捕获, 落盘以便排查
        try:
            import traceback as _tb
            with open(f"/tmp/nt_worker_{worker_id}.err", "w") as f:
                f.write(f"{type(exc).__name__}: {exc}\n\n")
                f.write(_tb.format_exc())
        except Exception:
            pass
    finally:
        if buf:
            try:
                conn.send(buf)
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass
        try:
            if hasattr(_keepalive, "close"):
                _keepalive.close()          # 先释放映射, 再退出
        except Exception:
            pass


# ================= 并行训练器 =================

class ParallelNTupleTrainer:
    """N-Tuple 并行训练器（Hogwild 无锁异步更新）。

    用法:
        cfg = NTupleConfig(episodes=1_000_000)
        trainer = ParallelNTupleTrainer(cfg, n_workers=3)
        if resume: trainer.load_checkpoint("latest.npz")
        trainer.train()
        trainer.close()
    """

    def __init__(self, cfg: NTupleConfig, n_workers: int = 0):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.models_dir = os.path.join(cfg.out_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        setup_logging(cfg.out_dir)

        self.n_workers = (n_workers if n_workers and n_workers > 0
                          else max(1, (os.cpu_count() or 2) - 1))

        # ---- 并行 α 自动缩放（实测补偿 Hogwild 写冲突）----
        # 写冲突等价于放大有效学习率。实测（5图案, 4000局, 同起点同种子）:
        #   1 worker  α=0.32 → 24,951 分（基准）
        #   3 workers α=0.32 → 20,654 分（-17%）
        #   3 workers α=0.16 → 23,621 分（-5%）  ← 缩放后接近单线程
        # 采用 1/sqrt(n) 缩放：比 1/n 保守，避免学习过慢。
        self._alpha_raw = cfg.alpha
        if self.n_workers > 1 and getattr(cfg, "auto_scale_alpha", True):
            cfg.alpha = round(cfg.alpha / (self.n_workers ** 0.5), 4)

        patterns = PATTERN_SETS.get(cfg.pattern_set, PATTERNS_6)
        self.patterns = [tuple(int(x) for x in p) for p in patterns]

        proto = NTupleNetwork(self.patterns, v_init=0.0)
        self.shape = proto.lut.shape
        self.dtype = proto.lut.dtype
        self.K = proto.K
        self.n_weights = proto.n_weights

        self.shm = SharedLUT(self.shape, self.dtype)
        self.lut = None
        self._workers: List[Process] = []
        self._conns = []                 # 与各 worker 的 Pipe 连接

        self.episode = 0
        self._start_episode = 0
        self._start_time = time.time()
        self._last_status_t = 0.0
        self.status_interval = 1.5
        self.best_eval_score = float("-inf")
        self._recent: List[int] = []
        self._last_eval_ep = 0
        self._last_ckpt_ep = 0
        self._last_snap_ep = 0
        self.snapshot_every = 0          # train() 中按需覆盖
        self.max_snapshots = 6
        # 搜索自举（v5）：用搜索选动作的比例（0 = 纯贪心）
        self.search_prob = 0.0
        self._search_steps = 0
        self._all_steps = 0
        self.status_path = os.path.join(cfg.out_dir, "status.json")
        self.history: Dict[str, list] = {
            "episode_scores": [], "evals": [], "losses": [],
            "epsilons": [], "episode_steps": [], "episode_max": [],
        }

        logger.info("=" * 74)
        logger.info("N-Tuple 并行训练器（Hogwild 无锁异步更新）")
        logger.info("  Worker   : %d 个并行进程", self.n_workers)
        logger.info("  网络     : %d 个 %d-元组(图案集 %s) x 8 对称 = %d 次查找",
                    len(self.patterns), len(self.patterns[0]),
                    cfg.pattern_set, self.K)
        logger.info("  权重表   : %s 项 (%.1f MB) —— 全 worker 零拷贝共享",
                    f"{self.n_weights:,}", self.shm.nbytes / 1048576)
        logger.info("  共享方式 : %s", "shared_memory (RAM)"
                    if _shm_available() else "memmap (文件映射)")
        try:
            _m = _mp.get_start_method(allow_none=True) or "?"
        except Exception:
            _m = "?"
        logger.info("  进程模型 : fork（顶层已锁 BLAS 单线程, 无死锁风险）")
        if self.search_prob > 0:
            logger.info("  ★ 搜索自举 : %.0f%% 的步骤用 2-ply 搜索选动作"
                        "（让 V 学习更强的策略）", self.search_prob * 100)
        _de = getattr(cfg, "alpha_decay_every", 0)
        _scaled = (self.n_workers > 1 and cfg.alpha != self._alpha_raw)
        _alpha_txt = ("α=%.3f（已从 %.3f 自动缩放, 补偿 %d 进程写冲突）"
                      % (cfg.alpha, self._alpha_raw, self.n_workers)
                      if _scaled else "α=%.3f" % cfg.alpha)
        if _de:
            logger.info("  超参     : %s λ=%.2f γ=%.1f | α 每 %s 局 ×%.2f（下限 %.3f）",
                        _alpha_txt, cfg.lam, cfg.gamma, f"{_de:,}",
                        getattr(cfg, "alpha_decay_gamma", 1.0),
                        getattr(cfg, "alpha_min", 0.0))
        else:
            logger.info("  超参     : %s λ=%.2f γ=%.1f v_init=0 clip=%s",
                        _alpha_txt, cfg.lam, cfg.gamma, cfg.weight_clip)
        logger.info("=" * 74)

    # ---------- 辅助 ----------
    def _current_net(self) -> NTupleNetwork:
        """指向共享表的网络视图（用于评估/保存）。"""
        net = NTupleNetwork(self.patterns, v_init=0.0)
        net.lut = self.lut
        return net

    def _rate(self) -> float:
        el = time.time() - self._start_time
        done = self.episode - self._start_episode
        return (done / el) if (el > 0 and done > 0) else 0.0

    def _eta_seconds(self) -> Optional[float]:
        target = self.cfg.episodes or 0
        if not target or self.episode >= target:
            return None
        r = self._rate()
        return round((target - self.episode) / r, 0) if r > 0 else None

    def _write_status(self, running: bool = True) -> None:
        try:
            scores = self.history["episode_scores"]
            status = {
                "running": running,
                "algo": "ntuple_v3_parallel",
                "episode": self.episode,
                "workers": self.n_workers,
                "total_steps": sum(self.history["episode_steps"][-2000:]),
                "epsilon": 0.0,
                "buffer_size": 0,
                "recent_scores": [int(s) for s in scores[-300:]],
                "recent_scores_start": max(0, len(scores) - 300),
                "epsilons_tail": [0.0] * min(300, len(scores)),
                "losses_tail": [round(float(x), 3)
                                for x in self.history["losses"][-2000:]],
                "evals": self.history["evals"][-60:],
                "selfplay": [],
                "best_eval_score": (None if self.best_eval_score == float("-inf")
                                    else round(self.best_eval_score, 1)),
                "selfplay_version": 0,
                "mcts_backend": None,
                "use_mcts": False,
                "net_weights": self.n_weights,
                "net_size_mb": round(self.shm.nbytes / 1048576, 1),
                "n_patterns": len(self.patterns),
                "tuple_len": len(self.patterns[0]),
                "search_depth": 0,
                "episodes_per_sec": round(self._rate(), 2),
                "search_prob": self.search_prob,
                "search_step_ratio": round(
                    self._search_steps / max(1, self._all_steps), 4),
                "processed_this_run": self.episode - self._start_episode,
                "eta_sec": self._eta_seconds(),
                "elapsed_sec": round(time.time() - self._start_time, 1),
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

    def _maybe_write_status(self, force: bool = False) -> None:
        now = time.time()
        if force or (now - self._last_status_t) >= self.status_interval:
            self._last_status_t = now
            self._write_status()

    # ---------- worker 生命周期 ----------
    def _start_workers(self) -> None:
        # 先设置线程环境变量（fork 时被子进程继承），再限制主进程线程
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[v] = "1"

        cfg_dict = {"alpha": self.cfg.alpha, "lam": self.cfg.lam,
                    "gamma": self.cfg.gamma,
                    "weight_clip": self.cfg.weight_clip,
                    "search_prob": self.search_prob,
                    "alpha_decay_every": getattr(self.cfg, "alpha_decay_every", 0),
                    "alpha_decay_gamma": getattr(self.cfg, "alpha_decay_gamma", 1.0),
                    "alpha_min": getattr(self.cfg, "alpha_min", 0.0)}
        self._workers = []
        self._conns = []
        ctx = _proc_context()
        self._ctx = ctx
        for i in range(self.n_workers):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_worker_main,
                            args=(self.shm.name, self.shape, str(self.dtype),
                                  self.patterns, cfg_dict, self.cfg.seed,
                                  child_conn, i),
                            daemon=True)
            p.start()
            child_conn.close()           # 父进程关闭子端
            self._workers.append(p)
            self._conns.append(parent_conn)
        logger.info("已启动 %d 个 worker 进程", len(self._workers))

    def _stop_workers(self) -> None:
        if not self._workers:
            return
        # 先收尾已有统计, 再下发停止信号
        self._drain_stats()
        for conn in self._conns:
            try:
                conn.send("stop")
            except Exception:
                pass
        for p in self._workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        self._drain_stats()              # 收最后一批
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass
        self._workers = []
        self._conns = []

    def _drain_stats(self) -> int:
        """从各 worker 的 Pipe 收集统计（批量）, 返回本轮处理局数。"""
        n = 0
        for conn in self._conns:
            while True:
                try:
                    if not conn.poll():      # 无待收数据
                        break
                    batch = conn.recv()
                except (EOFError, OSError):
                    break
                except Exception:
                    break
                if not isinstance(batch, list):
                    continue
                for item in batch:
                    score, steps, max_tile = item[0], item[1], item[2]
                    if len(item) >= 5:
                        self._search_steps += int(item[3])
                        self._all_steps += int(item[4])
                    if score is None or score < 0:
                        continue
                    self.episode += 1
                    self.history["episode_scores"].append(score)
                    self.history["episode_steps"].append(steps)
                    self.history["episode_max"].append(max_tile)
                    self._recent.append(score)
                    if len(self._recent) > 3000:
                        del self._recent[:1500]
                    n += 1
        return n

    # ---------- 训练主循环 ----------
    def train(self, episodes: Optional[int] = None) -> None:
        cfg = self.cfg
        total = episodes or cfg.episodes
        self._start_time = time.time()
        self._start_episode = self.episode

        if self.lut is None:                 # 未载入 checkpoint -> 从零开始
            self.lut = self.shm.create()
        self._start_workers()

        logger.info("开始并行训练: 目标 %s 局（已完成 %s 局, %d worker）",
                    f"{total:,}", f"{self.episode:,}", self.n_workers)
        last_log = time.time()
        try:
            while self.episode < total:
                self._drain_stats()
                self._maybe_write_status()

                now = time.time()
                if now - last_log >= 20:
                    last_log = now
                    if self._recent:
                        recent = np.asarray(self._recent[-1000:])
                        eta = self._eta_seconds()
                        logger.info(
                            "局 %8s | 近1000局均分 %9.0f | 最高 %9d | "
                            "速度 %6.1f 局/秒 | 剩余 %s",
                            f"{self.episode:,}", recent.mean(), recent.max(),
                            self._rate(),
                            f"{eta/3600:.1f} 小时" if eta else "–")
                    else:
                        logger.info("局 %8s | %6.1f 局/秒（数据收集中）",
                                    f"{self.episode:,}", self._rate())

                # 周期性评估 / 存档（以完成的局数计）
                if (cfg.eval_every > 0 and self.episode > 0
                        and self.episode - self._last_eval_ep >= cfg.eval_every):
                    self._last_eval_ep = self.episode
                    self._eval_cycle()
                if (cfg.ckpt_every > 0 and self.episode > 0
                        and self.episode - self._last_ckpt_ep >= cfg.ckpt_every):
                    self._last_ckpt_ep = self.episode
                    self.save_checkpoint("latest.npz")
                    logger.info(">> checkpoint 已保存 (局 %s)",
                                f"{self.episode:,}")

                # 里程碑快照（保留各阶段模型, 便于对比与回退）
                if (self.snapshot_every > 0 and self.episode > 0
                        and self.episode - self._last_snap_ep >= self.snapshot_every):
                    self._last_snap_ep = self.episode
                    name = f"snap_ep{self.episode}.npz"
                    self.save_checkpoint(name)
                    self._prune_snapshots()
                    logger.info(">> 里程碑快照: %s", name)

                if not any(p.is_alive() for p in self._workers):
                    logger.error("所有 worker 已退出, 训练中止")
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("收到中断, 停止 worker 并保存...")
        finally:
            self._stop_workers()
            self._drain_stats()
            self.save_checkpoint("latest.npz")
            self._write_status(running=False)

        logger.info("并行训练结束: 共 %s 局, 用时 %.1f 分钟",
                    f"{self.episode:,}", (time.time() - self._start_time) / 60)
        logger.info("平均吞吐: %.1f 局/秒 (%d worker)",
                    self._rate(), self.n_workers)

    # ---------- 评估 ----------
    def evaluate(self, games: Optional[int] = None,
                 search_depth: int = 0) -> Dict:
        n = games or self.cfg.eval_games
        net = self._current_net()
        rng = np.random.default_rng(12345)
        searcher = None
        if search_depth and search_depth > 0:
            from expectimax import ExpectimaxSearcher
            searcher = ExpectimaxSearcher(net, depth=search_depth)

        scores, steps_l, tiles = [], [], []
        for _ in range(n):
            try:
                board = new_board(rng)
                score, steps = 0, 0
                while steps < 20000:
                    if searcher is not None:
                        a = searcher.best_action(board)
                        if a is None:
                            break
                        from ntuple import move_all
                        bs, ss, _ = move_all(board)
                        nb, sc = bs[a], int(ss[a])
                    else:
                        res = greedy_action(net, board)
                        if res is None:
                            break
                        _, nb, sc, _ = res
                    score += int(sc)
                    board = spawn(nb.copy(), rng)
                    steps += 1
                scores.append(score)
                steps_l.append(steps)
                tiles.append(int(decode_board(board).max()))
            except Exception as exc:
                logger.warning("评估局异常: %s", exc)

        if not scores:
            return {"avg": 0.0, "max": 0, "median": 0.0, "avg_steps": 0.0,
                    "ms512": 0.0, "ms1024": 0.0, "ms2048": 0.0, "ms4096": 0.0,
                    "games": 0, "tiles": {}}

        def reach(t: int) -> float:
            return sum(1 for x in tiles if x >= t) / len(tiles)

        return {"avg": float(np.mean(scores)), "max": int(np.max(scores)),
                "median": float(np.median(scores)),
                "avg_steps": float(np.mean(steps_l)),
                "ms512": reach(512), "ms1024": reach(1024),
                "ms2048": reach(2048), "ms4096": reach(4096),
                "games": len(scores),
                "tiles": {int(k): int(v) for k, v in
                          zip(*np.unique(tiles, return_counts=True))}}

    def _eval_cycle(self) -> None:
        stats = self.evaluate()
        stats["episode"] = self.episode
        self.history["evals"].append(stats)
        logger.info("-" * 74)
        logger.info("★ 评估 @ 局 %s (%d 局): 均分 %s | 最高 %s | 中位 %s",
                    f"{self.episode:,}", stats["games"],
                    f"{stats['avg']:,.0f}", f"{stats['max']:,}",
                    f"{stats['median']:,.0f}")
        logger.info("  512 %.0f%% | 1024 %.0f%% | 2048 %.0f%% | 4096 %.0f%%",
                    stats["ms512"] * 100, stats["ms1024"] * 100,
                    stats["ms2048"] * 100, stats["ms4096"] * 100)
        logger.info("-" * 74)
        if stats["avg"] > self.best_eval_score:
            self.best_eval_score = stats["avg"]
            self.save_checkpoint("best.npz")
            logger.info("★ 新最佳均分 %s -> best.npz", f"{stats['avg']:,.0f}")

    # ---------- Checkpoint ----------
    def _prune_snapshots(self) -> None:
        """只保留最新的 max_snapshots 个快照, 避免磁盘膨胀。"""
        import glob as _glob
        snaps = sorted(_glob.glob(os.path.join(self.models_dir, "snap_ep*.npz")),
                       key=lambda p: os.path.getmtime(p))
        for path in snaps[:max(0, len(snaps) - self.max_snapshots)]:
            try:
                os.remove(path)
                meta = path + ".meta.json"
                if os.path.exists(meta):
                    os.remove(meta)
            except OSError:
                pass

    def save_checkpoint(self, name: str = "latest.npz") -> str:
        path = os.path.join(self.models_dir, name)
        tmp = f"{path}.tmp.npz"
        np.savez_compressed(tmp, lut=np.asarray(self.lut),
                            patterns=np.array(self.patterns, dtype=np.int32))
        os.replace(tmp, path)
        meta = {"episode": self.episode,
                "best_eval_score": self.best_eval_score,
                "workers": self.n_workers,
                "config": asdict(self.cfg),
                "history": {k: (v[-2000:] if isinstance(v, list) else v)
                            for k, v in self.history.items()},
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        mt = path + ".meta.json.tmp"
        with open(mt, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        os.replace(mt, path + ".meta.json")
        return path

    def load_checkpoint(self, name: str = "latest.npz") -> bool:
        """把 checkpoint 载入共享表（须在 train() 之前调用）。"""
        path = os.path.join(self.models_dir, name)
        if not os.path.exists(path):
            return False
        d = np.load(path)
        lut = d["lut"]
        if lut.shape != self.shape:
            logger.warning("checkpoint 结构不匹配 (%s vs %s), 忽略",
                           lut.shape, self.shape)
            return False
        if self.lut is None:
            self.lut = self.shm.create()
        np.copyto(self.lut, lut.astype(self.dtype))
        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                self.episode = int(meta.get("episode", 0))
                self.best_eval_score = float(
                    meta.get("best_eval_score", float("-inf")))
                self._last_eval_ep = self.episode
                self._last_ckpt_ep = self.episode
                for k, v in (meta.get("history") or {}).items():
                    if k in self.history and isinstance(v, list):
                        self.history[k] = v
            except Exception as exc:
                logger.warning("meta 恢复失败: %s", exc)
        logger.info("已恢复 checkpoint: 第 %s 局 (权重 σ=%.3f)",
                    f"{self.episode:,}", float(np.std(self.lut)))
        return True

    def close(self) -> None:
        self._stop_workers()
        self.shm.close(unlink=True)


# ================= 命令行入口 =================
def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="2048 N-Tuple 并行训练 (Hogwild)")
    p.add_argument("--episodes", type=int, default=3_000_000)
    p.add_argument("--workers", type=int, default=0,
                   help="并行 worker 数（0 = CPU 核数 - 1）")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-dir", default="outputs_v3p")
    p.add_argument("--patterns", default="8")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-games", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="每 N 局保存里程碑快照（0=关闭）")
    p.add_argument("--max-snapshots", type=int, default=6,
                   help="里程碑快照最多保留个数")
    p.add_argument("--alpha-decay-every", type=int, default=None,
                   help="每 N 局衰减一次 α（解决训练饱和）")
    p.add_argument("--alpha-decay-gamma", type=float, default=None,
                   help="α 衰减系数（默认 0.8）")
    p.add_argument("--alpha-min", type=float, default=None,
                   help="α 衰减下限（默认 0.02）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-scale-alpha", action="store_true",
                   help="关闭并行 α 自动缩放（默认开启, 用于补偿 Hogwild 写冲突）")
    p.add_argument("--search-play", type=float, default=0.0,
                   help="用 2-ply 搜索选择动作的比例（0=纯贪心, 1=全搜索）。"
                        "让价值函数从更强的策略中学习, 内化搜索的智慧")
    args = p.parse_args()

    cfg = NTupleConfig(episodes=args.episodes, out_dir=args.out_dir,
                       seed=args.seed, pattern_set=args.patterns)
    for attr, val in (("alpha", args.alpha), ("lam", args.lam),
                      ("eval_every", args.eval_every),
                      ("eval_games", args.eval_games),
                      ("ckpt_every", args.ckpt_every),
                      ("alpha_decay_every", args.alpha_decay_every),
                      ("alpha_decay_gamma", args.alpha_decay_gamma),
                      ("alpha_min", args.alpha_min)):
        if val is not None:
            setattr(cfg, attr, val)

    tr = ParallelNTupleTrainer(cfg, n_workers=args.workers)
    tr.snapshot_every = args.snapshot_every
    tr.max_snapshots = args.max_snapshots
    tr.search_prob = max(0.0, min(1.0, args.search_play))
    if args.no_scale_alpha:
        tr.cfg.alpha = tr._alpha_raw
    try:
        if args.resume:
            tr.load_checkpoint("latest.npz")
        tr.train(args.episodes)
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        tr.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
