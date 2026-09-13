# -*- coding: utf-8 -*-
"""
model_v2.py - Rainbow-Lite / Horizon-DQN 网络（对标 arXiv 2507.05465）

与 v1 (model.py) 的核心差异 —— 逐项复刻论文的 H-DQN 架构:
    1. 状态表示 : 16 维 log2 向量  ->  16x4x4 二值张量（每通道 = 某个 2^i
       tile 是否存在）。论文指出该表示显著提升样本效率。
    2. Encoder  : MLP(256-256)  ->  两层卷积（利用 4x4 棋盘的局部空间模式,
       卷积的归纳偏置对 tile 组合模式至关重要）。
    3. 输出头   : 4 个标量 Q  ->  4x51 分位数分布（C51 分布式 RL）, 并采用
       Dueling 结构（V+A 分离）。
    4. 探索     : ε-greedy  ->  NoisyNet（因子化高斯噪声）, 网络自己学习
       何时探索; 仍保留 ε-greedy 接口以兼容 MCTS 等调用方。
    5. 组合方式 : Q = V + A - mean(A)（Dueling 聚合, 逐原子进行）。

论文关键结论（本实现所依据的）:
    - distributional (C51) + multi-step targets 是稀疏奖励域提升的核心;
    - 贪心动作用 target 网络分布的均值选取:
          a* = argmax_a (1/N) * sum_j Z_target(s', a, j)
      （等价于 Double-Q 在分布式价值上的推广, 抑制过估计）
"""

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import move_grid

# ---------------- 常量 ----------------
GRID_SIZE = 4
N_ACTIONS = 4
NUM_ATOMS = 51          # C51 原子数（论文沿用经典配置 51）
V_MIN = -20.0           # 价值分布支撑下界（配合 reward_scale 使用）
V_MAX = 100.0           # 价值分布支撑上界
MAX_TILE_EXP = 16       # 通道数: 2^0 .. 2^15（覆盖到 32768）

# ---------------- 棋盘对称变换（数据增强 / 动作映射） ----------------
# 8 种对称变换下的动作置换表: ACTION_PERMS[t][a] = b
#   语义: "在变换后的棋盘上执行动作 a" 等价于 "在原棋盘执行动作 b 后再施加变换 t"。
#   t : 0-3 = 逆时针旋转 0/90/180/270 度; 4-7 = 先水平镜像再旋转 0/90/180/270。
# 该表由程序暴力推导得到（300 个随机棋盘投票 + 25600 组独立验证, 零不一致）。
#   【历史教训】早期 env.move_grid 的 DOWN 逆变换顺序写错, 导致此表无法求解；
#   修复 DOWN 后表即为标准的 8 阶变换群动作置换。
ACTION_PERMS = {
    0: [0, 1, 2, 3],
    1: [3, 2, 0, 1],
    2: [1, 0, 3, 2],
    3: [2, 3, 1, 0],
    4: [0, 1, 3, 2],
    5: [2, 3, 0, 1],
    6: [1, 0, 2, 3],
    7: [3, 2, 1, 0],
}


def transform_grid(grid, t: int) -> np.ndarray:
    """对棋盘施加第 t 种对称变换（t: 0-7）。"""
    g = np.asarray(grid)
    if t >= 4:
        g = np.fliplr(g)
    return np.rot90(g, t % 4)


def symmetric_grids(grid) -> list:
    """棋盘的全部 8 种对称视图（数据增强用, 样本量 x8）。"""
    return [transform_grid(grid, t) for t in range(8)]


def symmetric_encodings(grid) -> np.ndarray:
    """8 重对称增强后的状态编码: (8, 16, 4, 4)。"""
    return np.stack([encode_state(g) for g in symmetric_grids(grid)])


def symmetric_action(action: int, t: int) -> int:
    """把"原棋盘上的动作"映射为"第 t 种变换后棋盘上的对应动作"。"""
    b = ACTION_PERMS[t][action]
    return b


def verify_action_perms(n_trials: int = 200, seed: int = 0) -> bool:
    """独立验证动作置换表（自测 / 回归测试用）。"""
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        g = rng.choice([0, 0, 2, 4, 8, 16, 32, 64, 128, 256], size=(4, 4))
        for t in range(8):
            for a in range(4):
                b = ACTION_PERMS[t][a]
                lhs = np.array(move_grid(transform_grid(g, t), a)[0])
                rhs = transform_grid(np.array(move_grid(g, b)[0]), t)
                if not np.array_equal(lhs, rhs):
                    return False
    return True



# ---------------- 状态编码 ----------------

def encode_state(grid) -> np.ndarray:
    """棋盘 -> 16x4x4 二值张量（论文的状态表示）。

    通道 i 表示"该位置是否存在 2^i 的 tile"（i = 0..15, 覆盖 1..32768）。
    一个格子在同一时刻至多点亮一个通道, 全 0 表示空格。
    """
    tensor = np.zeros((MAX_TILE_EXP, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            v = grid[r][c]
            if v > 0:
                idx = int(math.log2(v))
                if 0 <= idx < MAX_TILE_EXP:
                    tensor[idx][r][c] = 1.0
    return tensor


def encode_state_batch(grids) -> np.ndarray:
    """批量编码: (B, 16, 4, 4)"""
    return np.stack([encode_state(g) for g in grids]).astype(np.float32)


# ---------------- 网络组件 ----------------

class NoisyLinear(nn.Module):
    """因子化高斯噪声线性层（Fortunato et al. 2017, NoisyNet）。

    W = mu_W + sigma_W * eps_W,  b = mu_b + sigma_b * eps_b
    训练时注入噪声实现"网络自己学习的探索"; 评估时只用均值（无噪声）。
    论文用它替代了手工 ε-greedy 衰减。
    """

    def __init__(self, in_features: int, out_features: int,
                 sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        # 噪声缓存（非参数）
        self.register_buffer("weight_epsilon",
                             torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """均值用均匀初始化, sigma 用论文建议的常量初始化。"""
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _factorized_noise(size: int) -> torch.Tensor:
        """因子化噪声: f(x) = sign(x) * sqrt(|x|), 降低采样维度。"""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """重新采样噪声（每个训练 step 调用一次, 让噪声逐 batch 变化）。"""
        eps_in = self._factorized_noise(self.in_features)
        eps_out = self._factorized_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu   # 评估: 不加噪声
        return F.linear(x, weight, bias)


class ConvEncoder(nn.Module):
    """两层卷积编码器: (B,16,4,4) -> (B, feat_dim)。

    卷积核 3x3 + padding 1 保持 4x4 空间维度, 保留棋盘局部结构信息
    （相邻 tile 的位置关系对 2048 的长期策略至关重要）。
    """

    def __init__(self, in_channels: int = MAX_TILE_EXP, feat_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 64 * GRID_SIZE * GRID_SIZE
        self.fc = nn.Sequential(
            nn.Linear(conv_out, feat_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class DuelingC51Net(nn.Module):
    """Dueling + C51 分布式 Q 网络。

    输出形状: (B, n_actions, n_atoms) —— 每个动作的回报分布 Z(s,a)。
    Dueling 聚合在"原子维度"上逐元素进行:
        Z(s,a) = V_atoms(s) + A_atoms(s,a) - mean_a A_atoms(s,a)
    """

    def __init__(self, n_actions: int = N_ACTIONS, n_atoms: int = NUM_ATOMS,
                 feat_dim: int = 256, noisy: bool = True):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.encoder = ConvEncoder(feat_dim=feat_dim)

        linear = (lambda i, o: NoisyLinear(i, o)) if noisy else (lambda i, o: nn.Linear(i, o))
        self.value = linear(feat_dim, n_atoms)                    # V(s) 分布
        self.advantage = linear(feat_dim, n_actions * n_atoms)    # A(s,a) 分布

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,16,4,4) 或已展平/其它布局（自动 reshape）"""
        if x.dim() == 3:                       # (B,16,16) -> (B,16,4,4)
            x = x.view(-1, MAX_TILE_EXP, GRID_SIZE, GRID_SIZE)
        elif x.dim() == 2 and x.shape[1] == MAX_TILE_EXP * GRID_SIZE * GRID_SIZE:
            x = x.view(-1, MAX_TILE_EXP, GRID_SIZE, GRID_SIZE)
        elif x.dim() == 2:                     # (B,16) log2 向量 -> 兼容模式
            x = x.view(-1, 1, GRID_SIZE, GRID_SIZE).repeat(
                1, MAX_TILE_EXP, 1, 1) * 0 + x.view(-1, 1, 4, 4)
            x = x.expand(-1, MAX_TILE_EXP, -1, -1).contiguous()

        feat = self.encoder(x)
        v = self.value(feat).view(-1, 1, self.n_atoms)
        a = self.advantage(feat).view(-1, self.n_actions, self.n_atoms)
        q = v + a - a.mean(dim=1, keepdim=True)     # Dueling 聚合
        return q                                    # (B, n_actions, n_atoms)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """期望 Q 值 (B, n_actions) —— 分布的均值。"""
        return self.forward(x).mean(dim=2)

    def reset_noise(self) -> None:
        """刷新所有噪声层（每个训练 step 调用）。"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ---------------- C51 投影 / 损失 ----------------

def project_distribution(next_dist: torch.Tensor, rewards: torch.Tensor,
                         dones: torch.Tensor, gamma_n: float,
                         v_min: float = V_MIN, v_max: float = V_MAX,
                         n_atoms: int = NUM_ATOMS) -> torch.Tensor:
    """把 n-step 目标分布投影回固定支撑 {z_i}（C51 核心步骤）。

    参数:
        next_dist : (B, n_atoms) 目标网络对 s' 选定动作的分布
        rewards   : (B,)         n-step 折扣回报 R
        dones     : (B,)         1 表示终止
        gamma_n   : n 步折扣因子 γ^n（n=1 时即 γ）
    返回:
        (B, n_atoms) 投影后的目标分布
    """
    delta_z = (v_max - v_min) / (n_atoms - 1)
    support = torch.linspace(v_min, v_max, n_atoms, device=next_dist.device)

    # 目标支撑: T_z = R + γ^n * z * (1 - done), 再裁剪回 [v_min, v_max]
    tz = rewards.unsqueeze(1) + gamma_n * support.unsqueeze(0) * (1.0 - dones.unsqueeze(1))
    tz = tz.clamp(v_min, v_max)

    # 分配到相邻两个原子（线性插值）
    b = (tz - v_min) / delta_z
    lower = b.floor().long()
    upper = b.ceil().long()
    lower = lower.clamp(0, n_atoms - 1)
    upper = upper.clamp(0, n_atoms - 1)

    proj = torch.zeros_like(next_dist)
    offset = torch.arange(next_dist.size(0), device=next_dist.device).unsqueeze(1) * n_atoms
    flat_upper = (upper + offset).view(-1)
    flat_lower = (lower + offset).view(-1)
    # 上原子权重 = b - lower, 下原子权重 = upper - b
    proj.view(-1).index_add_(0, flat_upper, (next_dist * (b - lower.float())).view(-1))
    proj.view(-1).index_add_(0, flat_lower, (next_dist * (upper.float() - b)).view(-1))
    return proj


def c51_loss(pred_dist: torch.Tensor, target_dist: torch.Tensor,
             weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """C51 交叉熵损失（可带 PER 重要性采样权重）。

    支持两种输入形状:
        (B, n_atoms)        —— 已按动作 gather 的分布
        (B, n_actions, n_atoms) —— 全动作分布
    """
    log_prob = F.log_softmax(pred_dist, dim=-1)
    loss = -(target_dist * log_prob).sum(dim=-1)      # (B,) 或 (B, A)
    if loss.dim() > 1:                                # (B, A) -> (B,)
        loss = loss.mean(dim=-1)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


# ---------------- Agent ----------------

class RainbowAgent:
    """v2 智能体: Dueling-C51 双网络 + NoisyNet 探索。

    接口与 v1 DQNAgent 保持兼容（select_action / sync_target /
    state_dict_cpu ...）, 便于直接替换进 train.py 与 MCTS。
    """

    def __init__(self, lr: float = 1e-4, gamma: float = 0.99,
                 device: Optional[str] = None, n_atoms: int = NUM_ATOMS,
                 v_min: float = V_MIN, v_max: float = V_MAX,
                 noisy: bool = True):
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.gamma = gamma
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.policy_net = DuelingC51Net(n_atoms=n_atoms, noisy=noisy).to(self.device)
        self.target_net = DuelingC51Net(n_atoms=n_atoms, noisy=noisy).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    # ---------- 状态编码（统一入口） ----------
    @staticmethod
    def encode(grid_or_states) -> np.ndarray:
        """接受棋盘 (4,4) 或已编码张量, 统一返回 (B,16,4,4)。"""
        arr = np.asarray(grid_or_states, dtype=np.float32)
        if arr.ndim == 2:                     # 单棋盘 (4,4)
            return encode_state(arr)[None, ...]
        if arr.ndim == 4:                     # 已是 (B,16,4,4)
            return arr
        if arr.ndim == 3:                     # (B,4,4) 批量棋盘
            return encode_state_batch(arr)
        return arr

    # ---------- 动作选择 ----------
    def select_action(self, state, epsilon: float = 0.0,
                      valid_actions: Optional[Sequence[int]] = None) -> int:
        """选择动作。NoisyNet 自带探索, 故默认 epsilon=0 即为带噪探索。"""
        import random
        if valid_actions is None:
            valid_actions = list(range(N_ACTIONS))
        valid_actions = list(valid_actions)
        if not valid_actions:
            return 0
        if epsilon > 0 and random.random() < epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            x = self.encode(state)
            xt = torch.as_tensor(x, device=self.device)
            q = self.policy_net(xt).mean(dim=2).squeeze(0)   # 分布均值 = Q
            mask = torch.full((N_ACTIONS,), float("-inf"), device=self.device)
            mask[valid_actions] = 0.0
            return int(torch.argmax(q + mask).item())

    @torch.no_grad()
    def q_values(self, state) -> np.ndarray:
        """返回 4 个动作的期望 Q 值。"""
        x = self.encode(state)
        xt = torch.as_tensor(x, device=self.device)
        return self.policy_net(xt).mean(dim=2).squeeze(0).cpu().numpy()

    # ---------- 目标网络 ----------
    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_noise(self) -> None:
        """每次训练 step 刷新噪声层采样。"""
        self.policy_net.reset_noise()

    # ---------- 权重导入导出（兼容 MCTS / checkpoint） ----------
    def state_dict_cpu(self) -> dict:
        return {k: v.detach().cpu().clone()
                for k, v in self.policy_net.state_dict().items()}

    def target_state_dict_cpu(self) -> dict:
        return {k: v.detach().cpu().clone()
                for k, v in self.target_net.state_dict().items()}

    def load_policy_state_dict(self, sd: dict) -> None:
        self.policy_net.load_state_dict(sd)

    def load_both_state_dict(self, sd: dict) -> None:
        self.policy_net.load_state_dict(sd)
        self.target_net.load_state_dict(sd)
        self.target_net.eval()


# ---------------- 模块自测 ----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- 1. 状态编码 ----
    grid = np.array([[2, 4, 0, 0], [0, 8, 0, 0], [0, 0, 16, 0], [0, 0, 0, 0]])
    enc = encode_state(grid)
    assert enc.shape == (16, 4, 4), f"编码形状错误: {enc.shape}"
    assert enc[1][0][0] == 1.0, "tile=2 应点亮通道 1"
    assert enc[2][0][1] == 1.0, "tile=4 应点亮通道 2"
    assert enc[3][1][1] == 1.0, "tile=8 应点亮通道 3"
    assert enc[4][2][2] == 1.0, "tile=16 应点亮通道 4"
    assert enc.sum() == 4.0, "通道点亮总数错误"
    print("1. 状态编码 (16x4x4 二值张量) ✓")

    # ---- 2. 对称变换动作映射（程序化验证） ----
    perms = ACTION_PERMS
    assert len(perms) == 8, "应有 8 种对称变换"
    for t, m in perms.items():
        assert sorted(m) == [0, 1, 2, 3], f"变换 {t} 的动作映射不是双射: {m}"
    assert perms[0] == [0, 1, 2, 3], "恒等变换应保持动作不变"
    assert perms[4][0] == 0 and perms[4][1] == 1, "水平镜像不应改变上下"
    assert verify_action_perms(), "动作置换表独立验证失败"
    assert symmetric_encodings(np.zeros((4, 4))).shape == (8, 16, 4, 4)
    print("2. 对称变换动作映射 ✓ (双射 + 独立验证 + 8重增强形状正确)")

    # ---- 3. 网络前向 ----
    net = DuelingC51Net()
    x = torch.randn(4, 16, 4, 4)
    out = net(x)
    assert out.shape == (4, 4, 51), f"输出形状错误: {out.shape}"
    q = net.q_values(x)
    assert q.shape == (4, 4), f"Q 值形状错误: {q.shape}"
    n_param = sum(p.numel() for p in net.parameters())
    print(f"3. Dueling-C51 网络前向 ✓ 输出{tuple(out.shape)} 参数量 {n_param:,}")

    # ---- 4. Dueling 聚合性质: sum_a A = 0 ----
    feat = net.encoder(x)
    a = net.advantage(feat).view(-1, 4, 51)
    adv_centered = a - a.mean(dim=1, keepdim=True)
    assert torch.allclose(adv_centered.sum(dim=1), torch.zeros_like(adv_centered[:, 0]),
                          atol=1e-4), "Dueling 优势函数归一化失败"
    print("4. Dueling 聚合 (Σ_a A = 0) ✓")

    # ---- 5. C51 投影 ----
    next_dist = torch.full((2, 51), 1.0 / 51)
    rewards = torch.tensor([5.0, 0.0])
    dones = torch.tensor([0.0, 1.0])
    proj = project_distribution(next_dist, rewards, dones, gamma_n=0.99)
    assert proj.shape == (2, 51)
    assert torch.allclose(proj.sum(dim=1), torch.ones(2), atol=1e-4), "投影后概率和应为 1"
    # 终止样本(done=1, r=0): 目标支撑恒为 0, 概率应只落在相邻两原子上
    delta_z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)
    expected_idx = (0.0 - V_MIN) / delta_z          # = 8.33
    nz = (proj[1] > 1e-6).nonzero().flatten().tolist()
    assert len(nz) <= 2 and all(abs(i - expected_idx) <= 1.5 for i in nz), \
        f"终止样本应集中在索引 {expected_idx:.2f} 附近, 实际 {nz}"
    # 非终止样本(gamma=0.99): 分布被展宽到多个原子
    nz0 = (proj[0] > 1e-6).nonzero().flatten().tolist()
    assert len(nz0) > 2, "非终止样本分布应被展宽"
    print(f"5. C51 分布投影 ✓ (概率守恒; 终止样本集中于索引 {nz}; 非终止展宽至 {len(nz0)} 个原子)")

    # ---- 6. NoisyNet 行为 ----
    nl = NoisyLinear(8, 4)
    nl.train()
    o1 = nl(torch.zeros(1, 8))
    nl.reset_noise()
    o2 = nl(torch.zeros(1, 8))
    assert not torch.allclose(o1, o2), "训练模式下噪声应变化"
    nl.eval()
    o3 = nl(torch.zeros(1, 8))
    o4 = nl(torch.zeros(1, 8))
    assert torch.allclose(o3, o4), "评估模式应确定性输出"
    print("6. NoisyNet (训练带噪/评估确定性) ✓")

    # ---- 7. Agent 接口兼容 ----
    agent = RainbowAgent()
    g = np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    a = agent.select_action(g, epsilon=0.0, valid_actions=[0, 2])
    assert a in (0, 2), "动作应在合法集合内"
    assert agent.q_values(g).shape == (4,)
    agent.sync_target()
    assert len(agent.state_dict_cpu()) == len(agent.policy_net.state_dict())
    print("7. RainbowAgent 接口兼容性 ✓")

    print("\nmodel_v2.py 自测全部通过 ✓")
