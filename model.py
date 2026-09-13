# -*- coding: utf-8 -*-
"""
model.py - DQN 神经网络 / 目标网络 / 动作选择 / 经验回放缓冲区

功能
----
1. QNetwork:
       输入 16 维（4x4 状态展平） -> 256 (ReLU) -> 256 (ReLU) -> 4 (各动作 Q 值)
2. ReplayBuffer:
       固定容量（默认 100,000）的先进先出经验回放缓冲区,
       存储 (state, action, reward, next_state, done) 五元组,
       训练时随机采样 batch 打破数据相关性, 满时自动淘汰最旧数据。
3. DQNAgent:
       同时维护 policy_net（当前训练网络）与 target_net（目标网络,
       定期从 policy_net 复制参数以稳定 TD 目标）,
       提供 select_action(state, epsilon) ε-greedy 动作选择接口。
"""

import random
from collections import deque
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


# ---------------- 工具函数 ----------------

def load_torch(path: str, map_location="cpu"):
    """兼容不同 torch 版本的 checkpoint 加载。

    torch >= 2.6 默认 weights_only=True, 对含非张量字段的自定义
    checkpoint 会报错, 因此先尝试 weights_only=False, 失败再回退默认。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # 旧版 torch 无 weights_only 参数
        return torch.load(path, map_location=map_location)


def atomic_torch_save(obj, path: str) -> None:
    """原子写入: 先写临时文件再 os.replace, 防止写入中途崩溃损坏文件。"""
    import os
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------- Q 网络 ----------------

class QNetwork(nn.Module):
    """Q 网络: 16 -> 256(ReLU) -> 256(ReLU) -> 4"""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 256, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自动展平: 兼容 (N,4,4) 2D 状态与 (N,16) 已展平输入
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.net(x)


# ---------------- 经验回放缓冲区 ----------------

class ReplayBuffer:
    """固定容量经验回放缓冲区（FIFO, 满时淘汰最旧数据）。

    用法:
        buf = ReplayBuffer(capacity=100_000)
        buf.push(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = buf.sample(64)
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """存入一条 transition。state 存副本, 防止外部原地修改。"""
        self.buffer.append((np.asarray(state, dtype=np.float32),
                            int(action),
                            float(reward),
                            np.asarray(next_state, dtype=np.float32),
                            bool(done)))

    def sample(self, batch_size: int):
        """随机采样一个 batch, 转为 numpy 数组（打破时序相关性）。"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states).astype(np.float32),
                np.asarray(actions, dtype=np.int64),
                np.asarray(rewards, dtype=np.float32),
                np.stack(next_states).astype(np.float32),
                np.asarray(dones, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------- DQN 智能体 ----------------

class DQNAgent:
    """DQN 智能体: policy_net + target_net + ε-greedy 动作选择。

    用法:
        agent = DQNAgent(lr=1e-4, gamma=0.99)
        action = agent.select_action(state, epsilon=0.1, valid_actions=[0, 2])
        agent.sync_target()   # 每 500 步将 policy 参数同步到 target
    """

    def __init__(self, lr: float = 1e-4, gamma: float = 0.99,
                 device: Optional[str] = None):
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.gamma = gamma

        # 双网络结构: policy_net 训练, target_net 提供稳定的 TD 目标
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    # ---------- 动作选择 ----------
    def select_action(self, state: np.ndarray, epsilon: float,
                      valid_actions: Optional[Sequence[int]] = None) -> int:
        """ε-greedy 策略选择动作。

        以 epsilon 概率从【合法动作】中随机探索;
        否则选择 Q 值最大的合法动作（非法动作 Q 值被 -inf 屏蔽）。

        参数:
            state         : 归一化状态 (4, 4) 或 (16,)
            epsilon       : 探索概率 [0, 1]
            valid_actions : 合法动作列表; None 表示四个动作全部合法
        """
        if valid_actions is None:
            valid_actions = list(range(4))
        valid_actions = list(valid_actions)
        if not valid_actions:          # 终局兜底
            return 0

        if random.random() < epsilon:  # 探索
            return random.choice(valid_actions)

        with torch.no_grad():          # 利用: 取 Q 值最大的合法动作
            s = torch.as_tensor(np.asarray(state, dtype=np.float32),
                                device=self.device).reshape(1, -1)
            q = self.policy_net(s).squeeze(0)
            mask = torch.full((q.shape[0],), float("-inf"), device=self.device)
            mask[valid_actions] = 0.0
            return int(torch.argmax(q + mask).item())

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        """返回 4 个动作的 Q 值（调试 / 可视化用）。"""
        s = torch.as_tensor(np.asarray(state, dtype=np.float32),
                            device=self.device).reshape(1, -1)
        return self.policy_net(s).squeeze(0).cpu().numpy()

    # ---------- 目标网络同步 ----------
    def sync_target(self) -> None:
        """将 policy_net 参数硬拷贝到 target_net。"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ---------- 权重导出 / 导入 ----------
    def state_dict_cpu(self) -> dict:
        """导出 policy_net 权重的 CPU 副本（供 MCTS 子进程使用）。"""
        return {k: v.detach().cpu().clone()
                for k, v in self.policy_net.state_dict().items()}

    def target_state_dict_cpu(self) -> dict:
        """导出 target_net 权重的 CPU 副本（供 MCTS 叶子评估使用）。"""
        return {k: v.detach().cpu().clone()
                for k, v in self.target_net.state_dict().items()}

    def load_policy_state_dict(self, sd: dict) -> None:
        """从 state_dict 加载策略网络权重。"""
        self.policy_net.load_state_dict(sd)

    def load_both_state_dict(self, sd: dict) -> None:
        """从 state_dict 同时加载 policy_net 与 target_net（评估/恢复时用）。"""
        self.policy_net.load_state_dict(sd)
        self.target_net.load_state_dict(sd)
        self.target_net.eval()


# ---------------- 模块自测 ----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    net = QNetwork()
    x = torch.randn(2, 16)
    out = net(x)
    assert out.shape == (2, 4), "Q 网络输出形状错误"
    assert sum(p.numel() for p in net.parameters()) == (16 * 256 + 256) + (256 * 256 + 256) + (256 * 4 + 4)

    buf = ReplayBuffer(capacity=100)
    for i in range(150):  # 超过容量, 验证 FIFO 淘汰
        s = np.full((4, 4), i % 16, dtype=np.float32)
        buf.push(s, i % 4, float(i), s, False)
    assert len(buf) == 100, "缓冲区容量限制失效"
    states, actions, rewards, next_states, dones = buf.sample(64)
    assert states.shape == (64, 4, 4) and actions.shape == (64,)

    agent = DQNAgent()
    state = np.random.rand(4, 4).astype(np.float32)
    a1 = agent.select_action(state, epsilon=1.0, valid_actions=[0, 2])
    assert a1 in (0, 2), "探索未限制在合法动作内"
    a2 = agent.select_action(state, epsilon=0.0, valid_actions=[1, 3])
    assert a2 in (1, 3), "利用未限制在合法动作内"
    agent.sync_target()

    print("model.py 自测通过 ✓  参数量:", sum(p.numel() for p in net.parameters()))
