# 2048 AI 训练系统（双算法）

> 📖 **[完整开发日志（docs/DEVELOPMENT_LOG.md）](docs/DEVELOPMENT_LOG.md)** — 12 个重大 Bug 的修复全过程、
> 4 个失败实验的教训、C++/Android 迁移细节、全套性能数据
>
> 🔧 **[C++ 训练器部署（cpp/README.md）](cpp/README.md)** — TDL2048+ 服务器 + Android 移植补丁

两套可对比的算法实现 + **Web 训练控制台**（实时曲线 / 一键训练 / AI 演示 / 模型评测）。

| 版本 | 算法 | 水平 | 说明 |
|---|---|---|---|
| v1 | 经典 DQN + MCTS | 均分 ~2,000 | 16 维 log2 状态、MLP、单步 TD、均匀回放、ε-greedy |
| v2 | Rainbow-Lite (H-DQN 复刻) | 均分 ~2,500 | 卷积 encoder + 16×4×4 独热 + Dueling + C51 + n-step + PER + NoisyNet |
| **v3** | **N-Tuple + TD(λ) + Expectimax** | **均分 25,000+** | ★ **真正的 SOTA 路线**，见下方专章 |

### 为什么 v3 是"最强"的

联网核查 2014–2025 全部文献后的结论：

| 算法 | 平均分 | 最大方块 | 说明 |
|---|---|---|---|
| 纯 RL（DQN） | 1,443 | 512 | 数学结构不匹配 2048 |
| 纯 RL（PPO / QR-DQN） | 1,831 / 3,480 | 512 / 1024 | 同上 |
| 纯 RL（H-DQN，论文最佳） | 5,694 | 2048 | RL 的天花板 |
| 神经网络 + 3-ply expectimax | 545,000 | 32768 | 换成搜索决策后跃升 |
| **n-tuple + 6-ply expectimax** | **625,377** | **32768（72%）** | **SOTA** |

决定强度的不是"网络多深"，而是三件事：
1. **Expectimax 搜索**（2048 是单人决策+随机环境，MCTS 与 DQN 的归纳偏置都不对）
2. **Afterstate 分解**（`s --动作--> s' --随机2/4--> s''`，把值函数定义在 s' 上隔离随机性）
3. **查表式特征**（n-tuple 权重互相独立 → episode 内可批量反向更新，收敛快一个量级）

> 论文《2048: Reinforcement Learning in a Delayed Reward Environment》(arXiv 2507.05465, 2025)
> 的结论是 **distributional + multi-step targets 显著提升稀疏奖励域表现**：
> 最高分 DQN 3.99K → PPO 5.76K → QR-DQN 8.66K → **H-DQN 18.2K（5000 局，可达 2048 方块），
> 扩训至 9000 局达 4.18 万分 / 4096 方块**。v2 即针对该结论实现。

> 已部署示例：http://103.117.136.126/ （Web 控制台，右上角可切换 v1/v2）

## ⚠️ 重要修复记录：DOWN 方向实现 Bug

早期 `env.move_grid` 的 DOWN 逆变换把"先转置再反转各列"错写成等价操作，导致
**向下移动行为错乱**（`UD_flip` 对称性检查 400/400 失败）。该 bug 使此前所有
训练都建立在错误环境上，已修复并加入 **300 例 × 3 种变换的方向对称性回归测试**
永久锁死。这也是 `model_v2.py` 中 8 重对称动作映射表得以求解的前提。

## 项目结构

```
2048_ai/
├── env.py          # 2048 游戏环境 + 奖励塑形（合并/空位/单调性/最大方块/终局惩罚）
├── model.py        # Q 网络(16→256→256→4) + 目标网络 + ε-greedy + 经验回放(100k)
├── mcts.py         # MCTS 搜索增强（多进程并行, 多级后端自动降级）
├── model_v2.py     # ★ v2: Conv + Dueling + C51 + NoisyNet 网络 + 8 重对称工具
├── buffer_v2.py    # ★ v2: n-step + PER(SumTree) + 对称增强回放缓冲区
├── train_v2.py     # ★ v2: Rainbow-Lite 训练引擎
├── train.py        # DQN 训练引擎（在线学习/评估/checkpoint/异常防护）
├── selfplay.py     # 自我对弈进化（新旧模型对战, 平均分领先 5% 才换代）
├── visualize.py    # 训练监控面板 PNG（分数/loss/ε 衰减/里程碑达成率）
├── main.py         # 命令行主入口
├── web/            # ★ Web 训练控制台
│   ├── app.py              # Flask 后端（状态/控制/演示/评测/日志 API）
│   ├── templates/index.html
│   └── static/{app.js, style.css, chart.umd.min.js}
├── requirements.txt
└── README.md
```

## Web 控制台

```bash
python web/app.py --host 0.0.0.0 --port 80        # 启动控制台（默认 8080）
# 或指定数据目录: OUT_DIR=/opt/2048ai/outputs python web/app.py --port 80
```

| 功能页 | 说明 |
|---|---|
| **实时监控** | 局数 / 近50局均分 / ε / 最佳评估分 4 张统计卡；分数曲线、TD Loss、ε 衰减、里程碑达成率 4 张实时图表（每 2s 轮询刷新）；一键启动/停止训练（停止走 SIGINT 正常保存 checkpoint） |
| **AI 演示** | 2048 棋盘逐步动画播放 AI 对局，可选模型、播放速度、MCTS 模拟次数；SSE 流式推送每一步 |
| **模型管理** | 模型文件列表（大小/时间）、自我对弈版本历史、在线快速评测（平均分/最高分/中位数/里程碑） |
| **训练日志** | 实时 tail 训练日志，5s 自动刷新 |

后端接口：`/api/status` `/api/logs` `/api/models` `/api/train/start` `/api/train/stop` `/api/eval` `/api/play`(SSE)

### systemd 部署（生产）

```bash
# /etc/systemd/system/2048ai-web.service
[Unit]
Description=2048 AI Training Console
After=network-online.target
[Service]
WorkingDirectory=/opt/2048ai
Environment=OUT_DIR=/opt/2048ai/outputs
ExecStart=/usr/bin/python3 /opt/2048ai/web/app.py --host 0.0.0.0 --port 80
Restart=always
[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload && systemctl enable --now 2048ai-web
```

## 命令行训练（不依赖 Web）

```bash
python main.py --train                    # 默认 5000 局
python main.py --train --resume           # 断点续训
python main.py --eval --eval-games 50     # 评估
python main.py --play                     # 终端观看 AI 对局
```

## 📱 手机端（Termux）使用指南

### 安装

```bash
cd ~
tar xzf /storage/emulated/0/rikkahub/2048ai-final.tar.gz
cd 2048_ai

# 依赖（v3 只需 numpy；v1/v2 才需要 torch）
pkg install python
pip install numpy --break-system-packages
pip install torch matplotlib flask --break-system-packages   # 可选
```

### 包内自带一个训练好的模型

`models/best.npz`（12 图案集 · 训练 15,000 局 · 评估均分 **14,326 分**）

```bash
# 直接用它玩（无需训练）
mkdir -p outputs/v3/models && cp models/best.npz outputs/v3/models/
python main.py --play --algo v3          # 观看对局
python main.py --eval --algo v3 --eval-games 20   # 评估
```

### 训练

```bash
# v3（推荐，纯 numpy 无需 torch）
python main.py --train --algo v3 --episodes 100000 --nt-patterns 5

# 图案集选择: 8(512MB,默认) / 12(768MB) / 5(48MB,手机推荐)
# 手机内存有限建议用 --nt-patterns 5（仅 48MB）

# v1（若装了 torch，速度最快）
python main.py --train --no-mcts --episodes 2000
```

### Web 控制台

```bash
OUT_DIR=~/2048ai-out python web/app.py --port 8080
# 手机浏览器打开 http://127.0.0.1:8080
```

### 自测

```bash
python env.py && python ntuple.py && python expectimax.py
python model_v2.py && python buffer_v2.py
```

### 手机端性能参考

| 项目 | 速度 |
|---|---|
| v3 训练（8 图案） | 约 15-30 局/秒 |
| v3 训练（5 图案，省内存） | 更快，推荐 |
| expectimax 1-ply 评估 | 约 1 万步/秒 |
| expectimax 2-ply | 约 300 步/秒 |

> ⚠️ 手机端注意
> - `--nt-patterns 8` 需 512MB 内存，12 图案需 768MB；内存紧张请用 `5`
> - MCTS（v1）在 Termux 会自动降级为 Process+Pipe 后端（无 /dev/shm）

## 快速开始

```bash
pip install -r requirements.txt

# 1. 一键启动训练（默认配置, 每 20 局自动插入一局 MCTS 决策局）
python main.py --train

# 2. 从 checkpoint 恢复训练
python main.py --train --resume

# 3. 训练结束后评估最佳模型
python main.py --eval --eval-games 50

# 4. 可视化观看 AI 玩一局
python main.py --play

# 5. ★ 训练 v2 (Rainbow-Lite / H-DQN 复刻)
python main.py --train --algo v2 --episodes 5000
python main.py --train --algo v2 --resume          # 断点续训
python main.py --eval  --algo v2 --eval-games 50   # 评估 v2 模型
python main.py --play  --algo v2                   # 观看 v2 对局

# 6. 各 v2 模块自测
python model_v2.py && python buffer_v2.py
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--train` | 开始训练 | 默认模式 |
| `--eval` | 加载模型评估（平均分/最高分/中位数/里程碑比率/平均存活步数） | - |
| `--play` | 可视化观看 AI 玩一局 | - |
| `--resume` | 从 `outputs/models/latest.pt` 恢复训练 | off |
| `--episodes N` | 训练局数 | 5000 |
| `--parallel N` | MCTS 并行进程数（0 = CPU 核数 - 1） | 0 |
| `--no-mcts` | 禁用 MCTS，纯 ε-greedy 训练（最快） | off |
| `--mcts-interval N` | 每 N 局插入 1 局 MCTS 决策局 | 20 |
| `--mcts-sims N` | MCTS 每次决策模拟次数 | 100 |
| `--eval-games N` | 评估局数 | 20 |
| `--seed N` | 随机种子 | 42 |
| `--device` | `auto` / `cpu` / `cuda` | auto |
| `--out-dir` | 输出目录 | outputs |
| `--algo` | 算法版本 `v1` / `v2` | v1 |
| `--n-step` | v2 的 n 步回报步数 | 3 |
| `--train-every` | v2 每 N 环境步训练一次 | 4 |

## 核心设计

### 1. 奖励塑形（env.py）

总奖励 = 各分项加权和（权重可在 `REWARD_WEIGHTS` 调整）：

| 分量 | 公式 | 作用 |
|---|---|---|
| 合并奖励 | 合并产生的新方块值之和 | 直接奖励高价值合并 |
| 空位奖励 | 空格子数 × 0.1 | 鼓励保持棋盘通畅 |
| 单调性奖励 | 单调性得分 × 0.01 | 鼓励棋盘梯度布局（ovolve 启发式） |
| 最大方块奖励 | log2(最大方块) × 0.5 | 鼓励养大最大方块 |
| 终局惩罚 | -10 | 惩罚提前死亡 |

> 若训练中合并奖励量级过大（合并出 1024+ 时）导致不稳定，
> 可将 `REWARD_WEIGHTS["merge"]` 调低至 0.1。

### 2. 状态表示

4×4 网格每格取 log2（空位为 0），除以 11（2048=2^11）归一化到 [0,1]。

### 3. 网络与训练（model.py / train.py）

- Q 网络：`16 → 256(ReLU) → 256(ReLU) → 4`
- 双网络：`policy_net` 在线训练，`target_net` 每 500 步硬同步
- TD 目标：`y = r + γ · max Q_target(s',·) · (1-done)`，γ=0.99
- Adam(lr=1e-4) + MSE loss + 梯度裁剪(max_norm=1.0)
- ε：1.0 → 0.05 线性衰减，周期 10,000 步
- Replay Buffer：容量 100,000，batch=64 随机采样

### 4. MCTS 搜索增强（mcts.py）

- **Root Parallelization 并行架构**：每个子进程独立建树模拟
  `100 / N` 次，主进程合并根节点访问次数，选访问最多的动作
- PUCT 选择 + **policy_net Q 值 softmax 作先验**引导搜索
- 叶子节点用 **target_net** 评估 `V(s) = max Q(s,·)`
- 模拟中对 chance node（新方块 90%/10%）随机采样
- 默认并行数 = CPU 核数 - 1（8 核机器为 7 进程）

> 为什么用进程不用线程：Python GIL 限制多线程无法加速 CPU 密集的
> 树搜索，multiprocessing 才能真正并行。

### 5. 自我对弈进化（selfplay.py）

- 冠军模型保存在 `outputs/models/best.pt`，版本历史在 `versions.json`
- 每 200 局发起挑战：新模型与旧冠军各玩 50 局纯贪心对战
- 新模型平均分超过旧模型 **5%** 才替换，否则保留旧冠军继续训练
- 即使训练后期性能回退也不会污染最佳模型

### 6. 健壮性

- 单局训练/评估全部 try-except 包裹，单局崩溃只记日志不中断训练
- checkpoint / JSON / best.pt 全部**原子写入**（临时文件 + os.replace）
- 日志同时输出到控制台与 `outputs/logs/train.log`
- checkpoint 含模型权重 + 优化器状态 + 训练进度 + 历史曲线，支持断点续训

## 输出说明

```
outputs/
├── models/
│   ├── best.pt              # 自我对弈挑战产出的冠军模型（--eval/--play 默认加载）
│   ├── latest.pt            # 最近训练状态（--resume 恢复点）
│   ├── checkpoint_epN.pt    # 每 500 局的历史 checkpoint
│   └── versions.json        # 冠军版本历史
├── logs/train.log           # 完整训练日志
└── plots/training_progress.png   # 监控面板（每 100 局刷新）
```

监控面板 4 张子图：分数曲线+滑动平均、TD loss（对数轴）、ε 衰减、
512/1024/2048/4096 里程碑达成率。

## 各模块自测

每个模块都可以独立运行自测：

```bash
python env.py        # 游戏规则 + 奖励塑形单测
python model.py      # 网络结构 + 回放缓冲区单测
python mcts.py       # 多进程 MCTS 冒烟测试
python selfplay.py   # 挑战/晋级流程单测
python train.py      # 训练引擎端到端冒烟测试
python visualize.py  # 图表生成测试
```

## 常见问题

**Q: 训练太慢？**
MCTS 决策局每步要做 100 次完整模拟，单局明显变慢。可以
`--no-mcts` 完全关闭，或 `--mcts-interval 50` 降低 MCTS 局频率，
或 `--mcts-sims 50` 减少模拟次数。

**Q: Windows 下报 multiprocessing 错误？**
确保通过 `python main.py` 入口运行（main.py 已含 `if __name__ == "__main__"`
护栏），不要在交互式解释器里直接 import 训练模块后启动训练。

**Q: 想从零重新训练？**
删除（或更换）`--out-dir` 目录即可；`--resume` 只在 `latest.pt` 存在时生效。

**Q: ε 衰减太快，前期分数低正常吗？**
正常。ε 在 10,000 环境步内从 1.0 衰减到 0.05，前几百局基本是随机策略，
评估平均分通常在 1000 局后明显上升。


---

## v2 (Rainbow-Lite) 详解

### 与 v1 的逐项差异

| 组件 | v1 (经典 DQN) | v2 (Rainbow-Lite) | 作用 |
|---|---|---|---|
| 状态表示 | log2 标量 (16,) | **16×4×4 二值张量** | 保留 tile 空间信息（论文关键） |
| Encoder | MLP 256-256 | **2 层卷积** | 局部 tile 组合模式的归纳偏置 |
| 输出头 | 4 个 Q 值 | **Dueling + C51(51 原子)** | V/A 解耦 + 分布式价值 |
| 回报 | 单步 TD | **3-step** | 信用分配距离 O(1)→O(3)，直击稀疏奖励 |
| 回放 | 均匀采样 | **PER (α=0.6, β:0.4→1.0)** | 聚焦高 TD 误差样本 |
| 探索 | ε-greedy 衰减 | **NoisyNet** | 网络自学何时探索 |
| 数据 | — | **8 重对称增强** | 样本量 ×8，成本近零 |

### 关键实现细节

- **Double-Q on distributional**：用 policy 网络选动作（分布均值最大），target 网络给出该动作的分布 —— 同时抑制过估计与保留分布式信号
- **C51 投影**：`T_z = R + γⁿ·z·(1-done)` 裁剪回支撑 `[-30, 200]`，再线性分配到相邻两原子
- **奖励缩放 0.1**：合并奖励可达 2048，远超 C51 支撑范围，缩放不改变最优策略
- **训练频率 1/4 步**：Conv+C51 单次反向约 30ms，逐步训练会让墙钟时间 8 倍于环境交互；取 4 等价于 Atari 的 frame-skip 训练，实测提速 5 倍（0.24 → 1.25 局/秒）

### 性能参考（4 核 CPU 服务器）

| 配置 | 速度 |
|---|---|
| v1 (MLP, 无 MCTS) | ~2.0 局/秒 |
| v1 (含 MCTS 局) | ~0.6 局/秒 |
| v2 (train_every=4, 2 线程) | ~1.25 局/秒 |
| v2 (train_every=8) | ~2.2 局/秒 |

### 训练预期

按论文曲线外推（本机 CPU 速度约为论文环境的数分之一）：

| 局数 | 预计水平 |
|---|---|
| 500 | 512 方块 |
| 2000 | 1024 方块 |
| 5000 | **2048 方块**（论文 H-DQN 在此达到 2048） |
| 9000+ | 4096 方块（论文扩训结果） |

> 参考：真正的 2048 SOTA 是 n-tuple 网络 + expectimax（平均 62.5 万分、72% 达 32768 方块），
> 纯 RL 方法在 2048 上天然弱于"搜索 + 特征工程"路线。本项目 v1 保留 MCTS 正是因为这个原因。


---

## v3: N-Tuple + TD(λ) + Expectimax（SOTA 路线）

### 快速开始

```bash
python main.py --train --algo v3 --episodes 200000   # 训练（约 1.5~2.5 小时）
python main.py --train --algo v3 --resume            # 断点续训
python main.py --eval  --algo v3 --eval-games 100    # 评估
python main.py --play  --algo v3                     # 观看对局

# 带 expectimax 搜索的评估（更强但更慢）
python main.py --eval --algo v3 --eval-games 100 --search-depth 2

# 各模块自测
python ntuple.py && python expectimax.py
```

### 三个模块

| 文件 | 内容 |
|---|---|
| `ntuple.py` | N-Tuple 查表网络 + 快速编码环境（65536 项合并查找表、向量化 4 动作、8 重对称） |
| `expectimax.py` | Expectimax 搜索（置换表 + chance 节点 + tile-downgrading） |
| `train_nt.py` | TD(λ) 训练引擎（乐观初始化、episode 内反向批量更新） |

### 关键技术点

**Afterstate 分解** —— 所有 SOTA 实现的基石：

```
        ┌─ 确定性（我们可控）─┐   ┌── 随机性 ──┐
   s  ──(动作 a)──▶  s'  ──(生成 2/4)──▶  s''
                     ↑
                 value function 定义在这里
              （问题退化为确定性 MDP）
```

**n-tuple 为什么比神经网络收敛快**：查表法的每个权重条目互相独立，一次更新不影响无关状态 → 可以在收集完整条 episode 后**从后向前**批量更新（顺带利用"末个 afterstate 真实价值恰为 0"这一先验，误差最小）。神经网络共享权重，只能逐步更新。

**乐观初始化 (OI)**：权重初值设为 `V_init / 查找次数`（默认 320,000）。迫使其主动探索未访问状态，论文证明优于 ε-greedy。实测对比（600 局）：

| v_init | 评估均分 | 512 达成率 |
|---|---|---|
| 0（关闭） | 2,621 | 8% |
| **320,000** | **3,653** | **25%** |

**Expectimax 而非 MCTS**：`V(s) = max_a Σ_spawn P(spawn) · V(afterstate)`，数学形式与"单人决策 + 随机环境"完全对应。

### 性能参考（4 核 CPU 服务器）

| 项目 | 速度 |
|---|---|
| 1-ply 贪心（训练用） | 43 局/秒 / 8,900 步/秒 |
| 1-ply 评估 | ~11,000 步/秒 |
| expectimax 2-ply | ~300 步/秒 |
| expectimax 3-ply | ~8 步/秒 |

### 参数调优

| 参数 | 默认 | 说明 |
|---|---|---|
| `--nt-len` | 6 | 元组长度。6=论文最优（384MB）；5=更省内存更快（训练更快但上限略低） |
| `--nt-alpha` | 0.1 | 学习率。查表法可用较大值 |
| `--nt-lambda` | 0.5 | TD(λ)。0=TD(0)，越大越接近 Monte Carlo |
| `--nt-vinit` | 320000 | 乐观初始化值，0=关闭 |
| `--search-depth` | 0 | 评估时的 expectimax 深度（0=贪心） |

### 训练预期

| 局数 | 预期水平 |
|---|---|
| 2,000 | 512 方块稳定 |
| 10,000 | 1024 方块 |
| 50,000 | 2048 方块 |
| 200,000 | 4096+ 方块 |
| 配合 3~6-ply expectimax | 8192～32768 方块 |


---

## ⚠️ 关键 Bug 修复：N-Tuple 图案的"对称轨道冗余"

### 问题

早期版本用了 6 个 2×3 块图案：

```python
PATTERNS = [(0,1,2,4,5,6), (1,2,3,5,6,7), (4,5,6,8,9,10),
            (5,6,7,9,10,11), (8,9,10,12,13,14), (9,10,11,13,14,15)]
```

看起来是 6 个不同的图案。但在 8 重对称（4 旋转 × 2 反射）下，
它们**只覆盖 2 个不同的轨道**：

| 轨道代表 | 出现的图案 |
|---|---|
| `(0,1,2,4,5,6)` | 图案 0, 1, 4, 5 —— 4 次重复 |
| `(1,2,5,6,9,10)` | 图案 2, 3 —— 2 次重复 |

**后果**：花 48 次查找，实际只得到 **16 个独立特征**。更糟的是，
重复轨道等价于把同一组权重乘以一个系数，白白消耗容量与算力。

### 修复

编写程序在 **68 个连通 6 格轨道**中贪心筛选，选出 8 个
**两两处于不同对称轨道**的图案，使棋盘覆盖近乎完全均摊：

```python
PATTERNS_8 = [
    (0, 1, 2, 4, 5, 6),  (0, 1, 2, 3, 4, 5),
    (0, 1, 2, 4, 5, 9),  (0, 1, 2, 3, 4, 6),
    (0, 1, 2, 5, 6, 7),  (0, 1, 2, 3, 5, 6),
    (0, 1, 2, 4, 5, 8),  (0, 1, 2, 3, 5, 9),
]
# 棋盘覆盖次数: min=23 / max=26 / 均值=24（16 格几乎完全均摊）
```

同时加入**轨道唯一性回归测试**，防止此类冗余再次出现：

```python
assert count_orbits(PATTERNS_8) == len(PATTERNS_8)   # 8 图案 / 8 轨道
```

### 实测收益（各 4000 局，同等条件）

| 配置 | 评估均分 | 相对提升 |
|---|---|---|
| 旧：6 图案（实为 2 轨道） | 6,409 | — |
| **新：8 图案（8 轨道）** | **10,218** | **+59.4%** |

### 附带性能优化

| 优化项 | 前 | 后 | 提升 |
|---|---|---|---|
| `move_all`（4 动作向量化，一次批量取线/查表/写回） | 75.5 µs | **16.8 µs** | 4.5× |
| `greedy_action` | 105 µs | **50 µs** | 2.1× |
| λ-return 递归（转 list 循环，避开 numpy 标量开销） | 0.41 ms | **0.18 ms** | 2.3× |
| 端到端（含 TD 更新） | — | **75.8 局/秒 / 15,548 步/秒** | — |

## 并行超参实验

`PATTERN_SETS` 提供三套图案集，可用 `--nt-patterns` 选择：

| 集合 | 图案数 / 轨道数 | 查找次数 | 内存 | 用途 |
|---|---|---|---|---|
| `8` | 8 / 8 | 64 | 512 MB | 默认，推荐 |
| `12` | 12 / 12 | 96 | 768 MB | 容量更大 |
| `5` | 12 / 12（5 元组） | 96 | 48 MB | 训练最快 |

### 对比 API

`GET /api/compare` 自动扫描 `exp_*/v3/status.json` 汇总并排对比，
Web 控制台「实时监控」页会渲染成表格。用于长跑实验（如整夜的
多配置并行训练）早上直接挑选最优。

```bash
# 三实验并行（各占一核）
python main.py --train --algo v3 --episodes 3000000 \
  --nt-patterns 8  --nt-lambda 0.5 --out-dir exp_a &
python main.py --train --algo v3 --episodes 3000000 \
  --nt-patterns 12 --nt-lambda 0.5 --out-dir exp_b &
python main.py --train --algo v3 --episodes 3000000 \
  --nt-patterns 8  --nt-lambda 0.9 --out-dir exp_c &
```

> 💡 因 TD 学习对单核是 CPU 密集的，多进程并行跑不同超参配置是
> 最有效的算力利用方式 —— 总吞吐提升 N 倍，同时横向对比选出最优。


---

## ⚠️ 关键修复记录（v5）：三个互相掩盖的训练 Bug

8 小时训练仅到 1.8 万分（论文同类方法 1-ply 贪心为 41 万），排查出**三个互相掩盖的实现错误**。

### Bug A：学习率未按查找次数均摊

`V(s) = Σ_{i=1..K} w_i`，每个权重应分摊 `α/K`：

```
错误：w_i += α·δ            →  ΔV = K·α·δ       （K=64 时放大 64 倍）
正确：w_i += (α/K)·δ        →  ΔV = α·δ         ✓
```

依据 TDL2048 官方文档：
> *"The learning rate **is distributed to each n-tuple feature weight**. For example, the 4x6patt network has 32 feature weights, so a weight is adjusted with a rate of 0.01 when `-a 0.32` is set."*

**回归测试**：权重总量增量必须等于 `α·δ`（与 K 无关）。

### Bug B：δ 被 clip=1.0 截断 97.7%（最致命）

δ 的量级是**分数**量级（实测中位 15、p99 约 8 万、最大 32 万），而代码里 `clip=1.0`：

```
实测 |δ| 分布: 均值 2,890 | 中位 15.1 | p90 79.7 | max 320,016
clip=1.0 会截掉 97.7% 的更新信号
```

**后果**：legacy 模型训练 294,791 局后，权重标准差仅 **9.6**（初始值 5000）——**权重几乎没动过**。此时 V(s) 对所有状态几乎恒为 320,000，动作选择退化成「贪心选最大立即合并分」的启发式，这才是 1.8 万分的真实来源。

> 两个 bug 恰好互相掩盖：A 放大 64 倍 × B 缩小到 ±1 ≈ 勉强能学一点，所以长期未被发现。

**修复**：`clip` 默认改为 `None`（不裁剪），并加 NaN/Inf 安全网。

### Bug C：乐观初始化 v_init=320000 反而有害

OI 使初始 `V(s) ≈ V(s')`，导致 `δ = r + γV(s') - V(s) ≈ r`（小值），网络只能学到**相对差异**，无法把整体水平从 32 万降下来。

| 配置（2500 局） | 评估得分 | V/实际 比值 |
|---|---|---|
| v_init=320000 | 8,177 | 29.30 ✗ 严重高估 |
| **v_init=0** | **18,113** | **0.80 ✓** |
| **v_init=0, λ=0** | **19,705** | **0.65 ✓** |

**修复**：`v_init` 默认改为 `0`。

### 修复效果（服务器实测）

| 实验 | 配置 | 局数 | 均分 |
|---|---|---|---|
| **exp_h** | v_init=0, λ=0, **α=0.32** | **3,601** | **17,450** |
| exp_g | v_init=0, λ=0, α=0.1 | 4,175 | 13,580 |
| exp_i | v_init=0, λ=0, α=1.0 | 5,566 | 8,745 |
| 旧 exp_a | v_init=320k, λ=0.5, α=0.1 | **294,791** | 16,388 |

> 🎯 **样本效率提升 82 倍**：3,601 局的成绩超过旧版 294,791 局。
> 价值函数标定从 `V/实际 = 0.05`（20 倍高估）改善到 `1.93`（健康）。

### 修正后的默认超参

| 参数 | 旧默认 | **新默认** | 说明 |
|---|---|---|---|
| `--nt-alpha` | 0.1 | **0.32** | 0.1 太慢、1.0 过快 |
| `--nt-lambda` | 0.5 | **0.0** | λ=0.9 会数值发散 |
| `--nt-vinit` | 320000 | **0** | 乐观初始化在此实现中有害 |
| `weight_clip` | 1.0 | **None** | 分数级 δ 不可裁剪 |


---

## 并行训练（Hogwild 无锁异步更新）

`train_nt_parallel.py` 实现多进程并行训练。

### 为什么可以无锁并行

N-Tuple 训练的两个阶段特性截然不同：

| 阶段 | 特性 | 并行性 |
|---|---|---|
| 对局生成 `greedy_action` | 纯读权重表 | 完全可并行 |
| 权重更新 `update_episode` | **稀疏写入**（每局只触及极小比例条目）| 冲突概率极低 |

由于写入稀疏，多个 worker 同时更新同一张表的冲突极少；且 TD 学习本身是
随机近似算法，偶发的写覆盖（lost update）等价于给梯度加噪声，不影响收敛
—— 这就是 **Hogwild!**（Recht et al. 2011）的思路。

### 架构

```
    主进程
      ├─ 创建 shared_memory（512MB 权重表，全系统只有一份物理内存）
      ├─ 启动 N 个 worker，各自 memmap 直接映射该内存（零拷贝）
      ├─ 每个 worker 独立跑局 → 直接在共享表上做 TD 更新（无锁）
      └─ Pipe 批量收集统计 → 写 status.json / 保存 checkpoint
```

### 用法

```bash
# 3 worker 并行训练（默认 CPU 核数 - 1）
python train_nt_parallel.py --episodes 3000000 --workers 3 --patterns 8

# 从已有 checkpoint 继续
python train_nt_parallel.py --resume --workers 3 --out-dir outputs_v3p

# 自定义超参
python train_nt_parallel.py --workers 3 --alpha 0.32 --lam 0.0
```

### 实测加速比（4 核 CPU，8 图案 512MB 表）

| Worker 数 | 吞吐 | 加速比 |
|---|---|---|
| 1 | 14.5 局/秒 | 1.00x |
| 2 | 26.1 局/秒 | 1.80x |
| **3** | **29.6 局/秒** | **2.04x** |
| 3（独占机器实测） | **16.2 局/秒** | **2.66x** vs 单进程 6.1 |

### 四个必须知道的工程坑

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | `multiprocessing.Queue` 依赖 POSIX 信号量 | 无 `/dev/shm` 环境抛 `FileNotFoundError` | 改用 `Pipe`（socketpair，无依赖）|
| 2 | fork + BLAS 线程池死锁 | worker 卡死在首次矩阵运算 | **在 import numpy 之前**于模块顶层设 `OMP_NUM_THREADS=1` |
| 3 | worker 内存翻倍 | `NTupleNetwork()` 先分配完整 LUT 再替换 | `_make_net_on_shared()` 用 `__new__` 跳过 `__init__`，直接挂共享表 |
| 4 | **共享内存悬垂指针** | worker 秒退 + **core dump** + 无任何日志 | `attach()` 必须返回并持有 `SharedMemory` 对象（否则被 GC 后映射关闭）|

> ⚠️ 第 4 个坑最隐蔽：`shm = SharedMemory(name); return np.ndarray(buffer=shm.buf)`
> 函数返回后 `shm` 被垃圾回收，底层映射关闭，ndarray 变成悬垂指针，
> 一访问就 SIGSEGV —— 且不产生任何 Python 异常，极难排查。


---

## Web 控制台功能说明

### 三大功能页

| 页面 | 功能 |
|---|---|
| **实时监控** | 主指标卡 + 4 张图表 + 并行实验对比（卡片 + 曲线叠加）+ 训练控制 |
| **AI 演示** | 2048 棋盘逐步动画；支持 v1/v2/v3 三种算法；v3 可用 `sims` 参数控制搜索深度 |
| **模型管理** | 全部模型列表（跨实验聚合）、评测、下载、**用模型启动新实验** |
| **训练日志** | 实时 tail |

### 用模型启动新实验

模型管理页的「用模型启动新实验」面板支持从任意已有模型热启动：

```
模型来源  [exp_hp ▾]      ← 自动列出所有含模型的目录
模型文件  [best.npz ▾]
新实验名  [exp_m      ]   ← 留空自动生成
并行Worker [3         ]   ← 0 = 单进程；>0 = Hogwild 并行
目标局数  [3000000    ]
        [▶ 用此模型启动]
```

**图案集自动校验**：8 图案与 12 图案的网络结构不同（LUT 布局不一样），
系统会读取模型自带的 `patterns` 并与请求比对，不匹配时直接拒绝并提示。

**路径约定**（重要）：
- 并行训练器 `train_nt_parallel.py` → 模型放在 `<exp>/models/`
- 单进程 `main.py --out-dir <exp>` → 模型放在 `<exp>/v3/models/`（多一层 v3）

### 接口一览

| 接口 | 说明 |
|---|---|
| `GET /api/status?algo=v1\|v2\|v3` | 训练状态（v3 主目录为空时自动回退到最优实验）|
| `GET /api/compare` | 并行实验对比（含曲线数据）|
| `GET /api/experiments` | 全部实验目录列表（含已停止的）|
| `GET /api/models?algo=all` | 全部模型聚合 |
| `GET /api/download?src=X&name=Y` | 下载模型（含路径穿越防护）|
| `POST /api/eval` | 评测模型（自动跨目录查找）|
| `POST /api/train/start_from_model` | ★ 从模型启动新实验 |
| `POST /api/train/start` / `stop` | 启动/停止训练 |
| `GET /api/play` | SSE 流式对局演示 |


---

## α 衰减（解决训练饱和）

### 为什么需要

固定学习率下，权重会在最优解附近**震荡而无法精细收敛**。实测 exp_hp 的表现：

```
 5,000 局:  20,303
10,000 局:  25,709  (+26.6%)
15,000 局:  32,546  (+26.6%)  ← 峰值
23,000 局:  30,197  (-7.2%)   ← 开始震荡
28,000 局:  33,210  (+10.0%)
33,000 局:  29,735  (-10.5%)
38,000 局:  33,900
```

分数在 30k~34k 之间来回跳，**并非收敛，而是步长过大导致的抖动**。

### 用法

```bash
python train_nt_parallel.py --workers 3   --alpha 0.32 --alpha-decay-every 25000 --alpha-decay-gamma 0.8 --alpha-min 0.05
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--alpha-decay-every` | 0（关闭） | 每 N 局衰减一次 |
| `--alpha-decay-gamma` | 0.8 | 衰减系数（α ← α × γ）|
| `--alpha-min` | 0.02 | α 下限（防止衰减到 0）|

**衰减序列示例**（α₀=0.32, γ=0.8, min=0.05）：

```
0~25k 局:   α = 0.320
25k~50k :   α = 0.256
50k~75k :   α = 0.205
75k~100k:   α = 0.164
...下限     α = 0.050
```

> 每个 worker 按自己的本地局数独立计算，无需主进程同步（Hogwild 风格）。

## 里程碑自动监控

`milestone_watch.py` + systemd timer，用于**长跑实验的自动验收**。

### 工作流

```
每 20 分钟检查一次 exp_hp/status.json
        ↓
  4096 达成率 >= 88%?  ← 触发阈值（100 局评估有 ±4% 噪声, 故留余量）
        ↓ 是
  自动跑 200 局 1-ply 复核（降低统计噪声）
        ↓
  确认 >= 90%?
        ↓ 是
  自动跑 10 局 3-ply expectimax 终测（约 40-60 分钟）
        ↓
  结果写入 milestone_report.json + 创建 DONE 标记防重复
```

### 部署

```bash
# 1) 脚本
cp milestone_watch.py /opt/2048ai/

# 2) systemd 单元
cat > /etc/systemd/system/2048-milestone.service << 'EOF'
[Unit]
Description=2048 Milestone Watch
[Service]
Type=oneshot
WorkingDirectory=/opt/2048ai
ExecStart=/usr/bin/python3 -u /opt/2048ai/milestone_watch.py
EOF

cat > /etc/systemd/system/2048-milestone.timer << 'EOF'
[Unit]
Description=Run 2048 milestone watch every 20 minutes
[Timer]
OnBootSec=5min
OnUnitActiveSec=20min
[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now 2048-milestone.timer
```

### 可配置项（脚本顶部常量）

| 常量 | 默认 | 说明 |
|---|---|---|
| `TRIGGER_TH` | 0.88 | 触发复核的阈值 |
| `CONFIRM_TH` | 0.90 | 复核确认阈值 |
| `CONFIRM_GAMES` | 200 | 复核局数 |
| `FINAL_GAMES` / `FINAL_DEPTH` | 10 / 3 | 终测局数与搜索深度 |

### 查看进度

```bash
cat /opt/2048ai/milestone_watch.log       # 每 20 分钟一行
cat /opt/2048ai/milestone_report.json     # 触发后的完整报告
```

## 搜索深度对实战水平的影响（同一模型实测）

| 决策方式 | 平均分 | 最高分 | 2048 达成 | 每步耗时 |
|---|---|---|---|---|
| 1-ply 贪心 | 40,213 | 82,056 | 90% | 0.2 ms |
| **2-ply expectimax** | **54,667** | **125,952** | **100%** | 4.3 ms |
| 3-ply expectimax | — | — | — | 149 ms |

> **+36% 的提升仅来自换决策方式** —— 模型权重完全没变。
> 这也是 2048 的核心结论：决策算法（expectimax）比价值表示更重要。

Web 控制台的「模型管理 → 快速评测」已支持选择搜索深度。


---

# v4: 推理增强框架（Reasoning-Enhanced Framework）

针对"**让模型具备推理能力**"这一目标构建的完整闭环：搜索（推理）生成数据 →
蒸馏训练快速模型 → 学生模型引导更深搜索 → 验证对比。

```
        ┌─────────────────────────────────────────────┐
        │  ① 推理层: Expectimax 搜索（慢但强）          │
        │     n-tuple + 2-ply → 均分 101,310          │
        └──────────────┬──────────────────────────────┘
                       │ 生成 (局面 → 各动作搜索价值) 数据
                       ▼
        ┌─────────────────────────────────────────────┐
        │  ② 学习层: 蒸馏学生网络（AlphaZero 双头）      │
        │     Policy 头（动作）+ Value 头（局面）       │
        │     155k 参数 = 0.6MB（vs n-tuple 512MB）   │
        └──────────────┬──────────────────────────────┘
                       │ 学生模型 / 集成
                       ▼
        ┌─────────────────────────────────────────────┐
        │  ③ 验证框架: 多配置横向对比                   │
        │     A/B/C/D/E 五种决策配置 × 速度 × 水平      │
        └─────────────────────────────────────────────┘
```

## 模块清单

| 文件 | 作用 |
|---|---|
| `v4/probe_pruning.py` | 剪枝可行性探测（动作价值差异分析）|
| `v4/probe2.py` | 搜索瓶颈分析（各阶段速度与价值分布）|
| `v4/vecsearch.py` | **向量化 Expectimax**（3.4x 加速, 结果等价）|
| `v4/gen_data.py` | 推理数据集生成器（~295 样本/秒）|
| `v4/student_net.py` | 蒸馏学生网络（AlphaZero 双头）|
| `v4/train_student.py` | 学生网络训练（相对价值回归）|
| `v4/verify.py` | **验证框架**（5 种配置对比）|

## 实测数据（10 局对局, 同一教师模型）

| 配置 | 平均分 | 最高分 | 2048 | 4096 | ms/步 | 相对基线 |
|---|---|---|---|---|---|---|
| A. n-tuple 1-ply | 54,751 | 85,904 | 90% | 60% | 0.16 | 100% |
| B. n-tuple 2-ply | **101,310** | **166,556** | 100% | 90% | 3.13 | **185%** |
| **C. 2-ply 向量化** | 94,337 | 130,520 | 100% | **100%** | **0.92** | **172%** |

**两个关键结论**：
1. **推理（搜索）比模型本身更重要** —— 同样的 n-tuple，加 2-ply 搜索提升 **+85%**
2. **向量化让推理变便宜** —— 3.13ms → 0.92ms（**3.4x**），水平保持

## 向量化原理

逐叶子实现（原版）：
```
for 动作 a (4 个):
    for 空位 cell (约 6 个):
        for 方块值 (2, 4):
            move_all + evaluate      ← 24~48 次独立的标量操作
```

向量化版本：
```
1. 批量 move_all   (B,16) → (B,4,16)      ← numpy 一次算完
2. 展平所有叶子     (B*4*2n, 16)
3. 批量查表         lut[indices]           ← 一次索引
4. 加权聚合         (leaf × P).sum()       ← 向量化
```

**数学等价**（同样的查表值，只换计算顺序），成本却是原来的 1/3.4。

### 开发中修复的两个 bug

| Bug | 症状 | 修复 |
|---|---|---|
| 满盘分支缺少动作最大化 | 整局均分 100k → 52k | `n==0` 时仍需对下一步取 max |
| 死局价值返回 -inf | 接近死局时决策错误 | 死局应为 `0.0`（与逐叶子实现对齐）|

> ⚠️ 这两个 bug 在"单步等价性测试"中都不会暴露 —— 必须做**完整对局对比**才能发现。
> 有效的回归测试必须覆盖边界局面（满盘 / 死局 / 单空位）。

## 数据管线

```bash
# 1. 生成推理数据（n-tuple + 2-ply 标注）
python v4/gen_data.py --model exp_hp/models/best.npz \
    --out v4_data/train_2ply.npz --games 400 --depth 2
#    ~295 样本/秒, 400 局约 1 小时（单核）

# 2. 训练学生网络
python v4/train_student.py --data v4_data/train_2ply.npz \
    --out v4_data/student.pt --epochs 30

# 3. 验证对比（A/B/C/D/E）
python v4/verify.py --games 20 --configs A,B,C,D,E
```

## 数据集格式

```python
{
    "boards":       (N, 16) uint8     # 编码局面（e=k 表示 2^k）
    "values":       (N, 4) float32    # 各动作 2-ply 搜索价值（非法为 -inf）
    "best_actions": (N,)   int8       # 搜索选出的最优动作
    "stages":       (N,)   int16      # 对局步数（阶段分桶用）
}
```

## 训练目标的特殊设计

**用"相对价值"而非绝对价值**：

```
绝对价值:  开局约 20,000, 后期约 45,000   ← 网络会把容量花在拟合"阶段"
相对价值:  各动作价值减去有效动作均值      ← 只保留"哪个动作更好、好多少"
```

依据：实测发现最优与次优动作的价值差仅 **0.36%~1.85%**，决策所需的信息
完全在"相对排序"里，绝对量级是噪声。

## 学生网络的定位

| 维度 | n-tuple | 学生网络 |
|---|---|---|
| 体积 | 512 MB | **0.6 MB**（1/850）|
| 单次推理 | 18 µs | ~4000 µs |
| 当前决策一致率 | — | 45%（数据量相关）|

**目标不是替代 n-tuple 的速度，而是**：
1. **部署友好**：0.6MB 可在手机/嵌入式运行
2. **集成互补**：与 n-tuple 组成 ensemble（验证框架的 E 配置）
3. **泛化研究**：检验"搜索知识能否被紧凑网络吸收"

> 当前一致率 45% 反映了任务本身的难度：动作价值差异极小（<2%），
> 学生网络需要在"几乎等价的动作"中区分细微优劣。
> 但**即使一致率不高，实际水平损失可能有限**（因为选错的都是近乎等价的动作）
> —— 这正是验证框架要测的。


---

# 实验报告：Hogwild 质量损失 & 搜索微调

两个严格 A/B 实验（均为同起点、同随机种子、5 图案集）。

## 实验 1：Hogwild 并行的写冲突损害

### 背景

并行训练中，多个 worker 无锁地更新同一张权重表。由于写入是稀疏的，
Hogwild 理论假设冲突可忽略。但 2048 的 afterstate 分布并不均匀 ——
角落状态（大 tile 固定处）会被所有 worker 频繁访问，导致高频条目冲突。

### 对照结果

| 配置 | 局数 | 墙钟 | 评估均分 | 相对基线 |
|---|---|---|---|---|
| 1 worker (α=0.32) | 4,000 | 571s | **24,951** | 基准 |
| 3 workers (α=0.32) | 4,000 | 293s | 20,654 | **-17%** |
| 3 workers (α=0.32) | 8,000 | 733s | 23,405 | -6% |
| **3 workers (α=0.16)** | 4,000 | 353s | 23,621 | **-5%** |
| **3 workers (α=0.16)** | 8,000 | 556s | **24,170** | **-3%** |

训练曲线（每 500 局均分）：
```
1 worker : 9.2k → 13.7k → 16.3k → 18.1k → 20.7k → 21.2k → 23.2k → 23.7k
3 workers: 6.7k → 10.1k → 12.0k → 13.7k → 14.9k → 16.1k → 17.8k → 18.7k
```

### 结论

1. **写冲突确实损害收敛质量**（-17%），且损害程度超过并行的速度收益
2. **补偿方法：降低学习率**（0.32 → 0.16），损失收敛到 -3~5%
   - 原理：写冲突等价于放大有效学习率（异步 SGD 的经典效应）
3. **长期仍推荐并行**：8000 局时 3-worker 训练均分 25,177 > 1-worker 的 23,494

### 已实现的自动补偿

```python
# train_nt_parallel.py
if n_workers > 1 and auto_scale_alpha:
    cfg.alpha = cfg.alpha / (n_workers ** 0.5)     # 1/sqrt(n) 缩放
```

实测：`3 workers: α 0.32 → 0.185`，日志显示
`α=0.185（已从 0.320 自动缩放, 补偿 3 进程写冲突）`

用 `--no-scale-alpha` 可关闭。

> ⚠️ 关于"同步屏障"：本实现的 worker 共享**同一块物理内存**
> （验证：系统 `shared` 列 = 562MB，而非 4×512MB），不存在副本，
> 因此「影子复制对比」「重启 worker 同步」都不适用。
> 真正的解决方案是调整学习率，而非引入同步。

## 实验 2：搜索微调（结论：有害）

### 背景

一个自然的想法：既然 3-ply 搜索比 1-ply 强 85%，能否用它做**微调**？
即让一个已训练好的模型继续用搜索选动作训练，期待它变得更强。

### 三阶段对照

```
Phase 1: 共享起点（1200 局贪心训练） → 均分 16,685
              ↓ 复制权重，分两臂
Phase 2a: 控制组 —— 继续 600 局贪心 → 均分 19,191 (+15%) ✓
Phase 2b: 实验组 —— 继续 600 局搜索 → 均分  4,214 (-74.7%) ✗
```

### 结论：搜索微调会**污染**价值函数

**为什么失败**：
- `V(s)` 的定义是「**在某个策略下**从 s 出发的期望得分」
- 用贪心生成对局 → V 评估贪心策略 → 配合贪心使用 ✓
- 用搜索生成对局 → V 评估搜索策略 → 单独用贪心时就失准 ✗

搜索策略的对局长 2-3 倍、棋盘更拥挤，状态分布与贪心完全不同。
微调后模型只会在"配合搜索"时表现正常，一旦脱离搜索就崩溃。

**与文献一致**：TDL2048 官方文档明确说明，expectimax 搜索与
tile-downgrading 用于 **testing，不是 training**。

### 正确用法

| 用途 | 方法 | 效果 |
|---|---|---|
| **训练** | 1-ply 贪心 | ✓ 收敛稳定 |
| **测试/对局** | 2-ply / 3-ply 搜索 | ✓ +85% ~ +150% |

搜索是**推理时的增强**，不是训练手段。
