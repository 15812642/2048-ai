# 2048 AI 开发日志

> 2026-09-11 → 2026-09-14 · 从零到 SOTA 的完整记录
> 12 个重大 Bug · 4 个失败实验 · 2 次架构迁移 · 全套性能数据

---

## 时间线总览

| 时间 | 阶段 | 关键事件 | 结果 |
|---|---|---|---|
| 09-11 | v1: DQN | DQN + MCTS 实现 | 均分 ~2,000 |
| 09-11 | v2: Rainbow | Rainbow-Lite 复刻（H-DQN 论文）| 均分 ~2,500，确认神经网络天花板 |
| 09-11 晚 | v3: N-Tuple | **3 个致命 Bug 修复** | 样本效率 **+82x** |
| 09-12 | 并行化 | Hogwild 无锁共享内存 | **2.66x** 加速 |
| 09-12 | v4 实验 | 搜索自举/蒸馏失败，向量化成功 | 3-ply 摸到 **16,384 方块** |
| 09-12 晚 | C++ 迁移 | TDL2048+ 部署 | 速度 **+37x**（vs Python 同条件）|
| 09-13 | Android 移植 | mmap shm + bionic 编译 | 手机 **561~1150 局/秒** |
| 09-13 | Web 集成 | C++ 日志解析器 + 控制面板 | 实时监控 |
| 09-14 | 三节点协同 | 服务器 + 2 手机并行训练 | 目标 33 万+ 分 |

---

## 第一章：v1 — DQN + MCTS（09-11）

### 1.1 环境 Bug：DOWN 方向实现错误 ⚠️

**症状**：`UD_flip` 对称性检查 400/400 全部失败 —— 上下翻转的棋盘执行相同动作后不一致。

**根因**：`env.move_grid` 的 DOWN 逆变换实现错误。"先转置再反转各列"与正确操作不等价，导致**向下移动行为错乱**。

**影响面**：此前所有训练都建立在错误环境上。

**修复后对照**：
```
修复前：随机策略均分 2,168（bug 让游戏"虚高"）
修复后：随机策略均分   420（正确基线）
```

**永久防护**：加入 **300 例 × 3 种变换**的方向对称性回归测试（UD_flip / LR_flip / 主对角线），锁死该类回归。

### 1.2 MCTS 多进程适配

- proot/Termux 环境无 `/dev/shm` → 三级降级：`Pool` → `长驻 Process + Pipe` → 串行
- `fork + torch` 死锁（OpenMP 线程池）→ MCTS worker 改**纯 numpy 手写前向**
- 服务器最终方案：`Pool(spawn)` + 失败熔断

### 1.3 结论

DQN 在 2048 上天花板约 2,000 分（文献同结论：纯 RL 1,443~5,694）。**数学结构不匹配**：2048 是确定决策 + 随机环境，MCTS 的博弈树假设偏了。

---

## 第二章：v2 — Rainbow-Lite（09-11）

### 2.1 实现

对标论文《2048: Reinforcement Learning in a Delayed Reward Environment》(arXiv 2507.05465) 的 H-DQN：

```
Conv encoder (16×4×4 独热输入)
+ Dueling + C51(51 原子) + NoisyNet
+ PER (SumTree, α=0.6, β:0.4→1.0)
+ n-step (n=3)
+ 8 重对称增强
```

### 2.2 性能问题与修复

**问题**：Conv+C51 单次反向 **30ms**，训练速度 0.24 局/秒（逐步训练慢 8 倍）。

**修复**：`train_every=4`（借鉴 Atari frame-skip 思想，每 4 环境步训练一次）→ **1.25 局/秒（提速 5 倍）**。

### 2.3 结论

均分 ~2,500。**神经网络路线在 2048 上打不过查表**（后面 v3 验证）—— 这个结论和文献一致：最好的纯 RL 是 18.2K（H-DQN 扩训），而 N-Tuple 查表轻松上 10 万+。

---

## 第三章：v3 — N-Tuple 突破（09-11 晚）★

### 3.1 为什么换路线

联网核查 2014–2025 全部文献后的结论：

| 算法 | 平均分 | 最大方块 |
|---|---|---|
| 纯 RL（DQN）| 1,443 | 512 |
| 纯 RL（H-DQN，最佳）| 5,694 | 2048 |
| 神经网络 + 3-ply expectimax | 545,000 | 32768 |
| **n-tuple + 6-ply expectimax** | **625,377** | **32768（72%）** |

**决定强度的三件事**：
1. **Expectimax 搜索**（不是 MCTS/DQN 的归纳偏置）
2. **Afterstate 分解**（`s --动作--> s' --随机--> s''`，值函数定义在 s' 上隔离随机性）
3. **查表式特征**（n-tuple 权重独立 → 批量反向更新，收敛快一个量级）

### 3.2 三个致命 Bug（互相掩盖）★★★

这是本项目最重要的技术章节。三个 Bug 同时存在时，模型看起来"能跑但很弱"（训练 29 万局只有 1.6 万分），且互相掩盖难以定位。

#### Bug A：学习率未按查找次数 K 均摊（虚高 64 倍）

```python
# 错误实现：
for lookup in K个查找: w[lookup] += alpha * delta      # 每权重用 α·δ

# 正确实现（TDL2048 官方文档）：
# "learning rate is distributed to each n-tuple feature weight"
for lookup in K个查找: w[lookup] += (alpha / K) * delta
```

**为什么错**：V(s) = Σ(K 个权重)。正确的 ΔV = α·δ 要求每权重只更新 α·δ/K。逐权重用 α·δ 会让 **ΔV = K·α·δ**（K=64 时虚高 64 倍）。

**验证判据**（修复后写进自测）：**权重总量增量必须 = α·δ，与 K 无关**。

#### Bug B：δ 被 clip=1.0 截断 97.7%

```python
# 错误：clip = 1.0
delta = clip(r + gamma * V(s') - V(s), -1.0, 1.0)
```

**为什么错**：δ 的量级是**分数**量级（实测 |δ| 中位 15、p99 约 8 万、最大 32 万），clip=1.0 把 97.7% 的学习信号砍掉。

**与 Bug A 互相掩盖**：A 放大 64 倍 × B 缩小到 ±1 ≈ 勉强能学一点 → 症状不明显。

**物证**：legacy 模型训练 294,791 局后权重 σ 仅 9.6（初始 5000）→ 权重几乎没动过 → 模型实际退化成"贪心选最大立即合并分"的启发式。

**修复**：`clip=None`（默认关闭裁剪）+ NaN/Inf 安全网。

#### Bug C：乐观初始化 v_init=320000 有害（公式用错）

```python
# 错误：每个权重都设 320000 → V(s) = 64 × 320000 = 2,050 万
net = NTupleNetwork(PATTERNS_8, v_init=320000)

# 正确（论文公式 LUT[i] ← V_init/m，m=图案数）：
# V_init=320000 应均摊 → 每权重 320000/8 = 40000（或按 64 次查找均摊 5000）
```

**注意**：乐观初始化（OI）本身是**论文验证的强方法**（8×6-tuple + OI 从 30 万 → 37 万分），但我们最初实现时值放大了 64 倍变成"过度乐观"，导致无法收敛。

**实测对照**（2500 局）：

| v_init | 得分 | V/实际 比值 |
|---|---|---|
| 320,000（错）| 8,177 | 0.05 |
| 0（修复）| 18,113~19,705 | 0.65~0.80 |

### 3.3 修复效果

```
修复后：exp_h 仅 3,601 局达均分 17,450
对比 legacy：294,791 局才达 16,388
→ 样本效率提升 82 倍
```

**修正后的默认超参**：
```python
alpha=0.32, lam=0.0, gamma=1.0, v_init=0.0, clip=None
```

> ⚠️ λ=0.9 会数值发散（NaN/overflow），λ=0 最优；α 最优 0.32（0.1 差 22%、1.0 差 50%）。

---

## 第四章：并行化 — Hogwild（09-12）

### 4.1 架构

主进程创建 shared_memory（512MB 表全系统一份）→ N 个 worker 用 memmap 直接映射 → 各自跑局并**直接在共享表上无锁更新**（Recht et al. 2011 Hogwild 理论：冲突概率低，偶发写覆盖等价于加噪声）。

### 4.2 四个坑（每个都是血泪）

1. **multiprocessing.Queue 依赖 POSIX 信号量**（/dev/shm）→ 无该目录环境抛 FileNotFoundError → 改用 `Pipe`（socketpair，无依赖）
2. **fork + BLAS 线程池死锁**：worker 卡死在第一次 einsum → 必须在 **import numpy 之前** 于模块顶层设 `OMP_NUM_THREADS=1`（import 后再设无效）
3. **worker 内存翻倍**：`NTupleNetwork(...)` 会先分配完整 LUT 再被替换 → 用 `__new__` 跳过 `__init__` 手工挂载共享表（worker 内存 800MB→500MB）
4. **★共享内存悬垂指针段错误**：`SharedMemory.attach` 里 `return np.ndarray(buffer=shm.buf)` —— 函数返回后 shm 被 GC，ndarray 变悬垂指针，访问即 SIGSEGV（症状：worker 秒退、无异常、core dump）→ attach 返回 `(array, keepalive)`，worker 全程持有

### 4.3 实测加速

```
1 worker : 14.5 局/秒
2 workers: 26.1 (1.80x)
3 workers: 29.6 (2.04x)
独占机器:  16.2 局/秒 vs 单进程 6.1 = 2.66x
```

### 4.4 Hogwild 写冲突的量化与补偿

严格对照（同起点/seed/4000 局）：

| 配置 | 评估均分 | 相对 |
|---|---|---|
| 1 worker α=0.32 | 24,951 | 基准 |
| 3 workers α=0.32 | 20,654 | **-17%** |
| 3 workers α=0.16 | 23,621 | -5% ✓ |
| 3 workers α=0.16（8000局）| 24,170 | -3% ✓✓ |

**结论**：写冲突 ≈ 放大有效学习率（异步 SGD 经典效应），补偿方法 = **α 自动缩放 `alpha / sqrt(n_workers)`**（已内置实现）。

---

## 第五章：v4 实验集 — 四个失败 + 两个成功（09-12）

### 5.1 ❌ 失败：搜索自举训练

**假设**：用 2-ply 搜索选动作生成对局，让 V 学"搜索策略下的价值"，1-ply 就能达到 2-ply 水平。

**结果**（同起点、500 局、seed 严格对照）：

| 模式 | 耗时 | 训练均分 | 评估(纯贪心) |
|---|---|---|---|
| greedy | 66s | 11,721 | 9,954 |
| search | 449s | 13,998 | **6,016** |

**慢 6.8 倍且评估分 -40%，全面失败。**

**原因**：γ=1 时 Bellman 算子非收缩；搜索策略的状态分布与贪心策略不同（对局长 2-3 倍、棋盘更拥挤）；文献佐证：TDL2048 官方明确说"Expectimax 用于 TESTING，不是 training"。

### 5.2 ❌ 失败：蒸馏到学生网络

学生网络（AlphaZero 双头 155k 参数）推理 **4000µs** vs n-tuple **18µs** —— **慢 200 倍**，无加速价值。

### 5.3 ❌ 失败：用搜索微调已训练模型

| 阶段 | 评估均分 |
|---|---|
| 起点 | 16,685 |
| +600 局贪心续训（对照）| 19,191 |
| +600 局搜索续训（实验）| **4,214** ✗ |

**搜索训练会"污染"已学好的 V** —— 与 5.1 结论一致，搜索只能用于测试。

### 5.4 ✅ 成功：向量化 Expectimax

- **2-ply 向量化**（`v4/vecsearch.py`）：3.13ms → 0.92ms/步（**3.4x**），数学等价
- **3-ply 向量化**（`v4/vecsearch3.py`）：210ms → 19.35ms/步（**11x**）

**发现的 Bug**：早期 3-ply 实现**少最后一层 max**（max→chance→max→chance→**V** 而非 →**max**→V），价值系统性偏低（最大差 7,642、决策一致率仅 57%）→ 修复后 97-100%。

### 5.5 ✅ 成功：3-ply 实战战绩

同一模型（40 万权重参数）各 8 局对照：

| 配置 | 均分 | 最高分 | 最大方块 |
|---|---|---|---|
| 2-ply | 70,406 | 85,384 | 4,096 |
| **3-ply** | **129,148** | **286,484** | **16,384** ★ |

---

## 第六章：C++ 迁移（09-12 晚 → 09-13）★

### 6.1 动机

Python 训练速度 8.7 局/秒 → 跑 1 亿局需要 133 天。文献同规模网络（8×6-tuple）训练 1 亿局达到 30.9 万分 —— **训练量差距 300 倍**是根本瓶颈。

**方案**：迁移到论文作者的官方 C++ 实现 [TDL2048+](https://github.com/moporgic/TDL2048)（bitboard 优化）。

### 6.2 关键发现 1：`-t LOOP[N]` 语法无效 ⚠️

**现象**：无论 `LOOP[20000]` 还是 `LOOP[999999999]`，显示的循环目标都是 `NNN/250`（不变）。

**排查过程**：对照 4 组实验（不同线程数/unit），最终确认：

```
实际循环目标 = 1000 × unit（每线程 = 1000/nthreads × unit）
LOOP[...] 参数被忽略（help 文档误导）
```

**差点造成的灾难**：初始配置 `unit=50000` → 每循环全局 50M 局（9 天）才存一次盘，且程序**无信号处理机制**（Ctrl+C 直接杀，进度全丢）。

**修复**：`unit=500` → 每循环 500K 局（约 27 分钟）存盘一次。

### 6.3 关键发现 2：实测速度与优化

```
服务器实测（4 核）：
  早期（对局短）: ~1600 局/秒
  中期（avg 42K）: ~310 局/秒
  后期（对局长）: ~60 局/秒（单次换算）

对比 Python 版同条件：8.7 局/秒 → 实际约 37x 提升（同阶段对比）
```

**反向优化发现**：`-march=native -flto` 编译**反而慢 4 倍**（对随机查表负载，指令集激进优化有害）→ 用官方默认 `-O3`。

### 6.4 训练配方（论文 OTD+TC）

```
Phase A  OTD 乐观探索   α=0.1（固定）
          V_init=320K（= 每查表位 5000 × 64 次查找，规避 "norm" 语义歧义）
Phase B  TC 自适应精调  α=1.0（触发 temporal coherence）
```

依据论文 Table VI：P_TC 10-20% 最优（OTD+TC = 370,907 vs 纯 OTD 361,471）。

### 6.5 Runner v2：Checkpoint 设计

每 500K 局自动存盘（513MB 覆盖写），崩溃最多丢 1 小时进度：

```bash
for i in $(seq 1 8);  do run_cycle 0.1 "A$i/8";  done   # Phase A
for i in $(seq 1 32); do run_cycle 1.0 "B$i/32"; done   # Phase B
```

### 6.6 实测训练曲线（阶段记录）

| 时刻 | 累计局数 | 评估均分 | max | 最大方块 | 2048 达成 |
|---|---|---|---|---|---|
| 27 分钟 | 500K | ~68,000 | 123,344 | 8192 | 98% |
| 2 小时 | 1.5M | ~103,000 | 177,020 | 16384 | 99.2% |
| （对标）Python 29 万局 | – | 73,858 | 166,540 | 8192 | 96% |

**C++ 版 27 分钟 = Python 版 29 万局（多天）的水平。**

---

## 第七章：Android 移植（09-13）★

> 让 C++ 训练器在 Termux（无 root/有 root 手机）上跑起来。三个大坑，全部解决。

### 7.1 坑 1：glibc 静态二进制 → SIGSYS（Signal 31）

**现象**：
- Proot 环境（Ubuntu rootfs）里跑得好好的二进制
- 直接在 Termux 执行 → **Signal 31 秒死**，无任何输出

**根因**：`Signal 31 = SIGSYS`（seccomp 拦截杀死）。glibc 2.39 会尝试用 `clone3` 系统调用，**Android 12+ 的 app seccomp 策略对 clone3 直接 KILL**（Go/Rust/glibc 程序在 Android 上的知名兼容性问题）。Proot 里能跑是因为 ptrace 拦截层改变了执行路径。

**物证**：`nm 二进制 | grep clone3` → `__clone3 / clone3_supported.0`。

**解决**：改用 **Android NDK（bionic libc）** 交叉编译 —— bionic 不用 clone3，所有系统调用都在 Android 白名单内。

### 7.2 坑 2：Android 无 SysV 共享内存

**现象**：`shmget` → `Function not implemented`（Android 内核裁剪了 System V IPC）。

**分析**：TDL2048 的多线程模式依赖 SysV shm（`fork` 前分配统计数组 + 512MB 权重表共享）。但它的使用模式有一个完美替代：

```
时序：shm::alloc() 分配 → fork() 子进程继承 → 父子共享读写
```

这正是 `mmap(MAP_SHARED | MAP_ANONYMOUS)` 的语义（fork 继承映射）。

**解决**：给 `moporgic/shm.h` 增加 `SHM_MMAP` 编译分支（见 `cpp/patches/shm-android.patch`），~57 行，替换全部 `shmget/shmat/shmdt/shmctl` 为 `mmap/munmap`。

### 7.3 坑 3：NDK 工具链修复 + clang-20 严格性

1. NDK 工具链符号链接在解压时损坏（5-8 字节的"实体文件"而非链接）→ 脚本重建 14 个链接
2. clang-20 对 C++ 标准更严格：`constexpr` 构造函数不能带非字面类型参数 → 删掉一个 `constexpr`（见 `cpp/patches/clang20-constexpr.patch`）

### 7.4 手机实测性能

**vivo V2172A / MT6785（Helio G95, 2019 中端, 有 root, 8 核）**：

对照测试（各 50 秒）：

| 配置 | 总速率 |
|---|---|
| 大核×2 绑核 4 线程 | 960 局/秒 |
| 小核×6 绑核 6 线程 | 550→1060 局/秒 |
| **全核 8 线程** ⭐ | **1150 局/秒** |
| 全核 6 线程 | 1060 局/秒 |

**单核对比（同阶段）**：大核 A76 538 局/秒 vs 小核 A55 92 局/秒。

**root 的作用**：`governor=performance` 锁频（`scaling_max_freq` 在 vivo 内核只读，但 governor 可写）→ 大核保持 1.99-2.05GHz。

**骁龙 8 Gen 2（无 root）**：修好降频问题（wake-lock + 关闭绑定）后 900+ 局/秒（早期）。

### 7.5 结论

- 手机从"Python 6 局/秒"到"C++ 1000+ 局/秒"= **~170x 提升**
- 一台上古中端机（MT6785）≈ 云服务器 4 核水平
- 三节点（1 服务器 + 2 手机）并行，最后权重 ensemble 合并（TDL2048+ 原生支持：`-i a.w b.w c.w -o merged.w`）

---

## 附录 A：Bug 清单（12 个）

| # | 模块 | Bug | 根因 | 修复 |
|---|---|---|---|---|
| 1 | env.py | DOWN 方向错误 | 转置+反列 ≠ 正确逆变换 | 重写 + 300 例回归测试 |
| 2 | model.py | 输入未展平 | 4×4 张量直传 Linear | forward 自动展平 |
| 3 | mcts.py | proot 无 /dev/shm | Pool 依赖 POSIX 信号量 | 三级降级 |
| 4 | mcts.py | fork+torch 死锁 | OpenMP 线程池 fork 后失效 | 纯 numpy worker |
| 5 | ntuple.py | **学习率虚高 64x** | 未按 K 均摊 | (α/K)·δ |
| 6 | ntuple.py | **δ 被 clip 砍 97.7%** | clip=1.0 vs 分数量级 δ | clip=None |
| 7 | ntuple.py | **乐观初始化放大 64x** | 每权重 320K 而非均摊 | v_init 默认 0 |
| 8 | 并行 | 段错误（悬垂指针）| SharedMemory GC | keepalive 持有 |
| 9 | 并行 | Queue 无 /dev/shm | POSIX 信号量依赖 | Pipe |
| 10 | 并行 | fork+BLAS 死锁 | OMP 线程池 | import 前设环境变量 |
| 11 | vecsearch3 | **少一层 max** | 3-ply 语义理解错误 | 补 max 层，一致率 57%→100% |
| 12 | TDL2048 | **LOOP[N] 语法无效** | help 文档误导 | unit=500 控制循环粒度 |
| 13 | TDL2048 | Android SIGSYS | glibc clone3 被 seccomp | bionic NDK 编译 |
| 14 | TDL2048 | Android 无 SysV shm | 内核裁剪 IPC | mmap 移植 |

## 附录 B：性能数据总表

### Python 版（N-Tuple v3，8 图案 512MB）

| 配置 | 均分 | 最大方块 | 耗时 |
|---|---|---|---|
| 1-ply 贪心 | 54,751 | 4,096 | 0.16 ms/步 |
| 2-ply 原始 | 86,563 | 8,192 | 3.13 ms/步 |
| 2-ply 向量化 | 94,337 | – | 0.92 ms/步 |
| 3-ply 向量化 | 129,148 | **16,384** | 19.35 ms/步 |

### 训练吞吐对比

| 环境 | 版本 | 吞吐（同阶段） |
|---|---|---|
| 服务器 4 核 | Python 3w | 8.7 局/秒 |
| 服务器 4 核 | C++ | ~310 局/秒（中期）|
| 手机 MT6785 | C++ 8w | 561~1150 局/秒 |
| 手机 8Gen2 | C++ 5w | ~900 局/秒 |

### 训练里程碑（C++）

| 局数 | 均分趋势 |
|---|---|
| 500K | ~68,000 |
| 1.5M | ~103,000 |
| 目标 20M | 330,000+（按论文曲线外推）|

## 附录 C：文献参考

1. Szubert & Jaśkowski, "Temporal Difference Learning of N-Tuple Networks for the Game 2048", CIG 2014
2. Wu et al., "Multi-stage Temporal Difference Learning for 2048", TAAI 2014
3. Guei & Wu, "Optimistic Temporal Difference Learning for 2048", IEEE TCIAIG 2021（**本项目 C++ 阶段的方法基础**）
4. Matsuzaki, "Systematic Selection of N-Tuple Networks", TAAI 2016
5. "2048: Reinforcement Learning in a Delayed Reward Environment", arXiv 2507.05465, 2025（v2 的方法基础）
6. [moporgic/TDL2048+](https://github.com/moporgic/TDL2048) — C++ SOTA 框架（MIT）

---

*本日志由开发过程实时记录整理 · 所有数据来自实测*
