# -*- coding: utf-8 -*-
"""
mcts.py - MCTS 搜索增强模块（多进程并行推演）

架构（Root Parallelization 根节点并行方案）
-------------------------------------------
- 主进程把当前棋盘状态 + 最新网络权重发给各 worker;
- 每个 worker 独立维护一棵搜索树, 各自执行分配到的模拟次数;
  单次模拟流程:
      1. 选择 Selection : PUCT 公式选择子节点
         score = Q(child) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
         其中先验 P(a) 由 policy_net 对节点状态的 Q 值做 softmax 得到
      2. 扩展 Expansion  : 展开一个未尝试动作, 生成新方块（chance node
         按 90% 概率 2 / 10% 概率 4 随机采样）
      3. 评估 Evaluation : 用 target_net 对叶子状态评估
         V(s) = max_{a ∈ valid} Q_target(s, a); 终局节点 V = 0
      4. 回传 Backprop   : 折扣回报 G = r + gamma * G 沿路径累加
- 主进程合并所有 worker 根节点的访问次数, 选择访问次数最多的动作。

工程要点（四个真实环境坑的解决方案, 均为生产环境实测踩坑）
----------------------------------------------------------
1. worker 内【不用 PyTorch】, 改为 numpy 手写 3 层 MLP 前向推理;
   并且本模块顶层【不 import torch】（主进程侧用 duck-typing 转换权重）。
   原因: 父进程初始化 torch 后 fork, 子进程首次调用 torch 会永久死锁
   （fork 只复制调用线程, OpenMP/线程池状态残留）。不 import torch 还让
   spawn 的 worker 启动快 2~3 秒。
2. worker 进程必须用【spawn】而非 fork 启动。
   原因: 训练主进程运行着 PyTorch 多线程 + 日志等多把锁, fork 出的 worker
   会继承这些处于"已锁定"状态的锁（持锁线程在子进程中不存在, 锁永不释放）,
   worker 一旦触碰即永久卡死 —— 表现为 MCTS 决策永不返回、后端反复重建、
   CPU 空转、训练停滞。spawn 启动全新解释器, 不继承任何锁状态。
3. 权重视图经进程间传输时【转 numpy 再传】。
   原因: torch.Tensor 经 multiprocessing 会走 /dev/shm 共享内存
   （_share_fd_cpu_）, 在 Termux / proot / 精简容器中 /dev/shm 缺失会抛
   "unable to open shared memory object"。
4. 并行后端【多级自动降级】: spawn-Pool -> spawn-Process+Pipe
   -> fork-Process+Pipe -> 串行, 并带【连续失败熔断】。
   原因: Pool 依赖 POSIX 信号量（需要 /dev/shm）, 缺失时直接失败;
   Pipe 底层是 socketpair, 不依赖共享内存, 兼容性最好; 熔断用于防止
   后端反复重建导致 CPU 空转与训练停滞。

为什么用进程而不是线程
----------------------
Python GIL 使多线程无法加速 CPU 密集的树搜索与 numpy 计算, 因此使用
multiprocessing 进程（默认进程数 = CPU 核心数 - 1）。worker 之间不共享
搜索树（各建一棵）, 主进程对根节点统计求和后决策, 这是 MCTS 常用的
Root Parallelization, 通信开销最小。
"""

import logging
import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
# 注意: 本模块【不 import torch】—— worker 全程只用 numpy 推理,
# 这样 spawn 出的子进程无需加载 torch（启动快 2~3 秒）,
# 主进程侧的权重转换改用 duck-typing（见 sd_to_numpy）。

from env import (GRID_SIZE, NORM_FACTOR, REWARD_WEIGHTS, add_random_tile,
                 can_move, compute_reward, move_grid, state_from_grid)

logger = logging.getLogger("2048")

_MAIN_PID = os.getpid()      # 主进程 PID（用于判断当前是否在子进程中）

# 连续失败达到该次数后熔断为串行（防止后端反复重建导致训练停滞）
MAX_BACKEND_FAILURES = 3


def _worker_context():
    """选择 worker 进程的启动方式: 优先 spawn, 退回平台默认。

    【关键】训练主进程里跑着 PyTorch 多线程与各种锁, 用 fork 启动 worker
    会继承"已锁定但永不会释放"的锁状态, worker 一触碰就永久死锁。
    spawn 会启用全新的解释器进程, 不继承锁 —— 这是本项目的核心修复。
    """
    import multiprocessing as mp
    try:
        if "spawn" in mp.get_all_start_methods():
            return mp.get_context("spawn")
    except Exception:
        pass
    return mp.get_context()

# ---------------- worker 进程全局状态（numpy 网络副本） ----------------
_POLICY_NET: Optional["_NumpyNet"] = None   # 先验网络（policy_net 的 numpy 副本）
_TARGET_NET: Optional["_NumpyNet"] = None   # 叶子评估网络（target_net 副本）
_W_SEEDED = False


class _NumpyNet:
    """纯 numpy 的 3 层 MLP 前向推理（16 -> 256 -> 256 -> 4）。

    结构与 model.QNetwork 完全一致, 只是用 numpy 矩阵乘法实现:
        h = ReLU(W1·x + b1); h = ReLU(W2·h + b2); out = W3·h + b3
    """

    def __init__(self):
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

    def load_numpy_sd(self, sd: dict) -> None:
        """从 numpy 版 state_dict 载入权重。

        键名形如 net.0.weight / net.0.bias / net.2.weight ...,
        字典序排序即网络层顺序。
        """
        weight_keys = sorted(k for k in sd if k.endswith(".weight"))
        self.weights = [np.asarray(sd[k], dtype=np.float32) for k in weight_keys]
        self.biases = [np.asarray(sd[k[: -len(".weight")] + ".bias"],
                                 dtype=np.float32) for k in weight_keys]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向推理。x: (N, 16) float32 -> (N, 4) float32"""
        h = np.asarray(x, dtype=np.float32)
        n_layers = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w.T + b
            if i < n_layers - 1:        # 最后一层不接激活函数
                h = np.maximum(h, 0.0)  # ReLU
        return h


def sd_to_numpy(sd: dict) -> dict:
    """torch state_dict -> numpy 版（进程间传输前必须转换, 见模块 docstring）。

    使用 duck-typing 而非 torch.is_tensor, 使本模块无需 import torch。
    """
    out = {}
    for k, v in sd.items():
        if hasattr(v, "detach"):           # torch.Tensor
            out[k] = v.detach().cpu().numpy()
        elif hasattr(v, "numpy"):          # numpy.ndarray
            out[k] = np.asarray(v)
        else:                              # 标量等
            out[k] = v
    return out


def _ensure_worker() -> None:
    """惰性初始化当前进程的 numpy 网络, 并在子进程中重新播种 RNG。

    fork 出的子进程会继承父进程的 RNG 状态, 若不重新播种, 各 worker 将
    产生完全相同的随机方块序列 —— 这是并行 MCTS 的经典陷阱。
    """
    global _POLICY_NET, _TARGET_NET, _W_SEEDED
    if _POLICY_NET is None:
        _POLICY_NET = _NumpyNet()
        _TARGET_NET = _NumpyNet()
    if not _W_SEEDED and os.getpid() != _MAIN_PID:   # 仅子进程重播种
        seed = int.from_bytes(os.urandom(8), "little")
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        _W_SEEDED = True


def _set_weights(policy_sd: dict, target_sd: dict) -> None:
    """同步最新权重到当前进程的 numpy 网络副本。"""
    _POLICY_NET.load_numpy_sd(policy_sd)
    _TARGET_NET.load_numpy_sd(target_sd)


# ---------------- 网络评估（numpy 前向） ----------------

def _softmax(q: Sequence[float], temperature: float = 1.0) -> List[float]:
    """数值稳定的 softmax, 将 Q 值转为先验概率分布。"""
    if not q:
        return []
    m = max(q)
    exps = [math.exp((x - m) / temperature) for x in q]
    s = sum(exps)
    return [e / s for e in exps]


def _priors_from_q(grid: Tuple, valid: Sequence[int]) -> Dict[int, float]:
    """用 policy_net 的 Q 值（softmax 后）作为各动作的先验概率。"""
    if not valid:
        return {}
    x = state_from_grid(grid).reshape(1, -1)
    q = _POLICY_NET.forward(x)[0]
    probs = _softmax([float(q[a]) for a in valid])
    return dict(zip(valid, probs))


def _leaf_value(grid: Tuple, valid: Sequence[int]) -> float:
    """target_net 评估叶子状态: V(s) = max_{a∈valid} Q_target(s, a)。"""
    if not valid:
        return 0.0
    x = state_from_grid(grid).reshape(1, -1)
    q = _TARGET_NET.forward(x)[0]
    return max(float(q[a]) for a in valid)


# ---------------- 搜索树节点 ----------------

class _Node:
    """MCTS 树节点。

    instant_reward : 进入本节点的即时塑形奖励（创建时计算并缓存）
    untried        : 尚未展开的合法动作
    priors         : action -> 先验概率（policy_net Q 值 softmax）
    """

    __slots__ = ("grid", "valid", "terminal", "instant_reward", "depth",
                 "children", "untried", "priors", "visits", "value_sum")

    def __init__(self, grid: Tuple, valid: List[int], terminal: bool,
                 instant_reward: float, priors: Dict[int, float],
                 depth: int = 0):
        self.grid = grid
        self.valid = valid
        self.terminal = terminal
        self.instant_reward = instant_reward
        self.depth = depth               # 树深度（用于限制模拟长度）
        self.children: Dict[int, "_Node"] = {}
        self.untried = list(valid)
        self.priors = priors
        self.visits = 0
        self.value_sum = 0.0

    @property
    def q(self) -> float:
        """节点平均回报 Q(s)。"""
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def _select_puct(node: _Node, c_puct: float) -> int:
    """PUCT 选择公式（AlphaGo Zero 风格, 结合先验引导与访问次数利用）。"""
    sqrt_n = math.sqrt(node.visits + 1)
    best_a, best_score = None, -math.inf
    for a in node.valid:
        child = node.children.get(a)
        q = child.q if child is not None else 0.0
        prior = node.priors.get(a, 1.0 / max(1, len(node.valid)))
        u = c_puct * prior * sqrt_n / (1 + (child.visits if child else 0))
        score = q + u
        if score > best_score:
            best_a, best_score = a, score
    return best_a


def _simulate(root: _Node, c_puct: float, gamma: float, weights: dict,
              max_depth: int = 200) -> None:
    """执行一次完整模拟: 选择 -> 扩展 -> 评估 -> 回传。

    max_depth 限制单次模拟的最大下降深度。
    必要性: 由于奖励塑形允许"无限存活"的退化解存在, 树可能随对局步数
    无限加深, 导致单次模拟耗时线性增长、MCTS 整体越来越慢。截断后
    单次搜索耗时恒定有界。
    """
    node = root
    path: List[Tuple[_Node, int, float]] = []   # (parent, action, 即时奖励)

    while True:
        if node.terminal:                       # 走到终局节点, V = 0
            value = 0.0
            break
        if node.depth >= max_depth:             # 深度截断, 视作叶子
            value = _leaf_value(node.grid, node.valid)
            break
        if node.untried:                        # 扩展一个未尝试动作
            a = node.untried.pop(0)
            ng, merged_sum, _ = move_grid(node.grid, a)
            ng = add_random_tile(ng)            # chance node 随机采样新方块
            ng = tuple(tuple(r) for r in ng)
            done = not can_move(ng)
            reward = compute_reward(ng, merged_sum, done, weights)
            child_valid = [x for x in range(4) if move_grid(ng, x)[2]]
            priors = _priors_from_q(ng, child_valid) if not done else {}
            child = _Node(ng, child_valid, done, reward, priors,
                          depth=node.depth + 1)
            node.children[a] = child
            path.append((node, a, reward))
            value = 0.0 if done else _leaf_value(ng, child_valid)  # target_net 评估
            node = child
            break
        # 选择: 用 PUCT 沿已展开的树下降
        a = _select_puct(node, c_puct)
        child = node.children[a]
        path.append((node, a, child.instant_reward))
        node = child

    # 回传: G = r + gamma * G, 沿路径自底向上累加, 更新访问统计
    g = value
    for parent, a, r in reversed(path):
        child = parent.children[a]
        child.visits += 1
        child.value_sum += g
        g = r + gamma * g
    root.visits += 1


# ---------------- 单次搜索任务（当前进程内执行） ----------------

def _search_once(root_grid, n_sims: int, c_puct: float, gamma: float,
                 weights: dict, max_depth: int = 200):
    """在当前进程内对 root_grid 建树并执行 n_sims 次模拟。

    返回: (visits[4], value_sum[4]) —— 根节点各动作的访问次数与回报累计
    """
    visits = [0, 0, 0, 0]
    value_sums = [0.0, 0.0, 0.0, 0.0]
    valid = [a for a in range(4) if move_grid(root_grid, a)[2]]
    if not valid or n_sims <= 0:
        return visits, value_sums

    priors = _priors_from_q(root_grid, valid)   # 根节点先验
    root = _Node(root_grid, valid, False, 0.0, priors, depth=0)
    for _ in range(n_sims):
        try:
            _simulate(root, c_puct, gamma, weights, max_depth)
        except Exception:
            continue  # 单次模拟失败不影响整体搜索

    for a, child in root.children.items():
        visits[a] = child.visits
        value_sums[a] = child.value_sum
    return visits, value_sums


def _worker_search_task(args):
    """进程池任务入口: 同步权重后执行搜索。

    policy_sd / target_sd 为 None 时表示"沿用本进程已缓存的权重"
    （主进程按 sync_interval 抽样同步, 大幅减少跨进程传输开销）。
    """
    (policy_sd, target_sd, root_grid, n_sims,
     c_puct, gamma, weights, max_depth) = args
    _ensure_worker()
    if policy_sd is not None:
        _set_weights(policy_sd, target_sd)
    return _search_once(root_grid, n_sims, c_puct, gamma, weights, max_depth)


def _worker_main(conn):
    """长驻子进程主循环: 通过 Pipe 接收任务, 计算完成后回传结果。

    Pipe 底层是 socketpair, 不依赖共享内存, 因此在无 /dev/shm 的
    Termux / proot 环境中依然可用。
    """
    _ensure_worker()
    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break
            if msg is None:                  # 关闭信号
                break
            try:
                (policy_sd, target_sd, root_grid, n_sims,
                 c_puct, gamma, weights, max_depth) = msg
                if policy_sd is not None:      # None = 沿用已缓存权重
                    _set_weights(policy_sd, target_sd)
                conn.send(_search_once(root_grid, n_sims, c_puct, gamma,
                                       weights, max_depth))
            except Exception as exc:         # 单任务失败回传异常对象
                try:
                    conn.send(exc)
                except Exception:
                    break
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------- 三种并行后端 ----------------

def _shm_available() -> bool:
    """检测 /dev/shm 是否可用（Pool 的 POSIX 信号量依赖它）。"""
    try:
        return os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)
    except Exception:
        return False


class _SerialBackend:
    """串行后端: 进程内依次执行所有任务（永远可用的兜底）。"""
    name = "serial"

    def run(self, tasks):
        return [_worker_search_task(t) for t in tasks]

    def close(self):
        pass


class _PoolBackend:
    """进程池后端: multiprocessing.Pool（spawn 上下文, 标准环境最快）。

    spawn 会启动全新解释器进程, 避免 fork 继承主进程（跑着 torch 多线程）
    中已锁定的锁而导致 worker 永久死锁。
    """

    name = "multiprocessing.Pool(spawn)"

    def __init__(self, n_workers: int):
        self.pool = _worker_context().Pool(processes=n_workers)

    def run(self, tasks, timeout: float = 120.0):
        # apply_async + get(timeout) 保证 worker 异常时不会永久阻塞
        results = [self.pool.apply_async(_worker_search_task, (t,))
                   for t in tasks]
        return [r.get(timeout=timeout) for r in results]

    def close(self):
        if self.pool is not None:
            try:
                self.pool.terminate()
                self.pool.join()
            except Exception:
                pass
            self.pool = None


class _ProcessBackend:
    """长驻进程后端: Process + Pipe（不依赖共享内存, 兼容 Termux/proot）。

    主进程先向所有 worker 投递任务, 再依次收集结果;
    每个 worker 都有 poll 超时保护, 避免 worker 崩溃导致永久阻塞。
    """

    name = "multiprocessing.Process+Pipe"

    def __init__(self, n_workers: int, use_spawn: bool = True):
        import multiprocessing as mp
        if use_spawn:
            try:
                ctx = _worker_context()
            except Exception:
                ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() \
                    else mp.get_context()
        else:
            methods = mp.get_all_start_methods()
            ctx = mp.get_context("fork") if "fork" in methods else mp.get_context()
        self.pairs = []
        for _ in range(n_workers):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(target=_worker_main, args=(child_conn,),
                               daemon=True)
            proc.start()
            child_conn.close()      # 父进程关闭子端
            self.pairs.append((proc, parent_conn))

    def run(self, tasks, timeout: float = 60.0):
        pairs = self.pairs[:len(tasks)]          # 任务数不超过 worker 数
        for (_, conn), task in zip(pairs, tasks):
            conn.send(task)
        results = []
        for proc, conn in pairs:
            if not conn.poll(timeout):
                raise TimeoutError(
                    f"MCTS worker 超时({timeout}s)未响应, pid={proc.pid}")
            result = conn.recv()
            if isinstance(result, Exception):
                raise result                      # 交给上层降级处理
            results.append(result)
        return results

    def close(self):
        for proc, conn in self.pairs:
            try:
                conn.send(None)
            except Exception:
                pass
        for proc, conn in self.pairs:
            try:
                conn.close()
            except Exception:
                pass
            try:
                proc.join(timeout=2.0)
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
        self.pairs = []


# ---------------- 主进程决策器 ----------------

class MCTSAgent:
    """MCTS 决策器（多进程并行, 后端惰性创建 + 自动降级）。

    用法:
        mcts = MCTSAgent(n_simulations=100, n_processes=0)  # 0 = CPU 核数 - 1
        action = mcts.search(env.grid, dqn_agent)
        mcts.close()
    """

    def __init__(self, n_simulations: int = 100, c_puct: float = 1.5,
                 n_processes: int = 0, gamma: float = 0.99,
                 reward_weights: Optional[dict] = None,
                 sync_interval: int = 10, max_depth: int = 200):
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.reward_weights = dict(reward_weights or REWARD_WEIGHTS)
        # 模拟最大深度: 防止树随对局步数无限加深导致搜索越来越慢
        self.max_depth = max(10, int(max_depth))
        if n_processes is None or n_processes <= 0:
            n_processes = max(1, (os.cpu_count() or 2) - 1)
        self.n_processes = max(1, n_processes)

        # 权重同步间隔（每 N 次搜索向 worker 推送一次最新权重）。
        # 网络权重体积约 300KB, 每步都传会在 MCTS 局造成显著 IPC 开销;
        # 局内相邻几步的网络差异极小, 抽样同步对决策质量无实质影响。
        self.sync_interval = max(1, int(sync_interval))
        self._sync_counter = 0
        self._weights_synced = False     # 当前后端是否已持有权重

        self._backend = None
        self._backend_name: Optional[str] = None
        self._degrade_count = 0          # 降级次数（诊断用）
        self._consecutive_failures = 0   # 连续失败计数（熔断用）
        self._melted = False             # 是否已熔断为串行
        self._last_stats: Optional[dict] = None

    # ---------- 后端创建 / 降级 ----------
    def _create_backend(self):
        """按 spawn-Pool -> spawn-Pipe -> fork-Pipe -> serial 顺序尝试创建后端。

        每一级失败自动尝试下一级; 连续失败达到阈值后熔断为串行,
        避免后端反复重建导致 CPU 空转、训练停滞。
        """
        if self._melted:                      # 已熔断: 直接串行
            return _SerialBackend()

        candidates = []
        if self.n_processes > 1:
            if _shm_available():
                candidates.append(lambda: _PoolBackend(self.n_processes))
            candidates.append(lambda: _ProcessBackend(self.n_processes, use_spawn=True))
            candidates.append(lambda: _ProcessBackend(self.n_processes, use_spawn=False))
        candidates.append(lambda: _SerialBackend())

        last_exc = None
        for factory in candidates:
            try:
                return factory()
            except Exception as exc:
                last_exc = exc
                continue
        return _SerialBackend()

    def _ensure_backend(self):
        if self._backend is None:
            self._backend = self._create_backend()
            self._backend_name = self._backend.name
            if self._backend.name == "serial":
                if self.n_processes > 1:
                    logger.warning("并行后端不可用, 使用串行搜索")
            else:
                logger.info("MCTS 后端: %s (%d 进程)",
                            self._backend.name, self.n_processes)
        return self._backend

    def _drop_backend(self) -> None:
        """丢弃当前后端（运行中出错时调用）, 连续失败过多则熔断为串行。"""
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
        self._backend = None
        self._degrade_count += 1
        self._weights_synced = False     # 新 worker 需要重新接收权重
        self._consecutive_failures += 1
        if self._consecutive_failures >= MAX_BACKEND_FAILURES:
            # 熔断: 不再重建进程后端, 永久切换为串行（保证训练持续推进）
            if not self._melted:
                self._melted = True
                logger.error("MCTS 并行后端连续失败 %d 次, 已熔断为串行搜索"
                             "（训练继续, 速度下降）", self._consecutive_failures)

    def _mark_success(self) -> None:
        """一次成功搜索: 复位连续失败计数。"""
        self._consecutive_failures = 0

    def close(self) -> None:
        """关闭后端, 释放子进程资源。"""
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
            self._backend = None

    # ---------- 核心搜索 ----------
    def search(self, grid, agent, return_stats: bool = False):
        """对当前棋盘执行 MCTS, 返回访问次数最多的合法动作。

        参数:
            grid  : 环境当前棋盘（4x4 list）
            agent : DQNAgent, 提供最新 policy/target 权重
        返回:
            action (int); 搜索失败或不可用时返回 None（调用方应回退贪心）
        """
        root_grid = tuple(tuple(int(v) for v in row) for row in grid)
        valid = [a for a in range(4) if move_grid(root_grid, a)[2]]
        if not valid:
            return None
        if len(valid) == 1:          # 仅一个合法动作, 无需搜索
            return valid[0]

        try:
            # 权重按 sync_interval 抽样同步: 未同步时传 None, worker 沿用缓存
            self._sync_counter += 1
            need_sync = (not self._weights_synced
                         or self._sync_counter % self.sync_interval == 0)
            if need_sync:
                policy_sd = sd_to_numpy(agent.state_dict_cpu())
                target_sd = sd_to_numpy(agent.target_state_dict_cpu())
                self._weights_synced = True
            else:
                policy_sd = target_sd = None
            # 将模拟次数均分给各 worker（每个 worker 独立建树）
            per_worker = max(1, math.ceil(self.n_simulations / self.n_processes))
            tasks = [(policy_sd, target_sd, root_grid, per_worker,
                      self.c_puct, self.gamma, self.reward_weights,
                      self.max_depth)
                     for _ in range(self.n_processes)]
            backend = self._ensure_backend()
            results = backend.run(tasks)
        except Exception as exc:     # 并行失败 -> 降级并回退贪心
            logger.warning("MCTS 搜索失败(%s), 降级重试: %s",
                           self._backend_name, exc)
            self._drop_backend()
            return None

        # 合并所有 worker 的根节点统计
        visits = [0, 0, 0, 0]
        values = [0.0, 0.0, 0.0, 0.0]
        for w_visits, w_values in results:
            for a in range(4):
                visits[a] += w_visits[a]
                values[a] += w_values[a]

        best = max(valid, key=lambda a: visits[a])
        self._mark_success()          # 搜索成功, 复位失败计数
        self._last_stats = {"visits": visits, "values": values,
                            "valid": valid, "backend": self._backend_name}
        return best if not return_stats else (best, self._last_stats)


# ---------------- 模块自测 ----------------
if __name__ == "__main__":
    import time

    import torch                       # 仅自测时使用（模块本身不依赖 torch）

    from env import Game2048
    from model import DQNAgent, QNetwork

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    random.seed(0)

    # numpy 前向与 torch 前向一致性校验
    net = QNetwork()
    nn_net = _NumpyNet()
    nn_net.load_numpy_sd(sd_to_numpy(net.state_dict()))
    x = np.random.rand(3, 16).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.as_tensor(x)).numpy()
    got = nn_net.forward(x)
    assert np.allclose(ref, got, atol=1e-5), "numpy 前向与 torch 不一致"
    print("numpy 前向一致性校验通过 ✓")

    env = Game2048(seed=1)
    agent = DQNAgent()
    mcts = MCTSAgent(n_simulations=16, n_processes=3)

    state = env.reset()
    t0, steps, n_mcts = time.time(), 0, 0
    done = False
    while not done and steps < 40:
        action = mcts.search(env.grid, agent)
        if action is not None:
            n_mcts += 1
        else:
            action = agent.select_action(state, 0.0, env.get_valid_actions())
        state, _, done, info = env.step(action)
        steps += 1
    print(f"MCTS 后端: {mcts._backend_name} | {n_mcts}/{steps} 步由 MCTS 决策")
    print(f"分数={info['score']} 最大方块={info['max_tile']} "
          f"耗时={time.time() - t0:.2f}s")
    assert n_mcts > 0, "MCTS 未能产生任何决策"
    mcts.close()
    print("mcts.py 自测通过 ✓")
