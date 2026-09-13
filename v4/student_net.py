# -*- coding: utf-8 -*-
"""
student_net.py - v4 学生网络（AlphaZero 双头架构）

结构
----
    输入 16x4x4 独热（tile 是否存在的二值张量）
        ↓
    Conv(16→32, 3x3) + ReLU
        ↓
    Conv(32→64, 3x3) + ReLU
        ↓ Flatten (64x16=1024)
    FC(1024→128) + ReLU        ← 共享特征
        ↓                ↓
    Policy 头           Value 头
    FC(128→4)          FC(128→1)
    各动作价值           局面价值

为什么用双头
------------
- Policy 头：直接给出"该走哪"（决策用），目标是教师搜索的动作价值
- Value 头：给出局面评分（可用于未来引导搜索 / 判断局面优劣）

目标值处理
----------
- 动作价值范围 20k~45k，训练时除以 VALUE_SCALE(=10000) 归一化
- 无效动作（非法）在 loss 中用 mask 排除
- 回归目标是【相对价值】（减去有效动作均值），数值更稳定
  —— 因为绝对价值受对局阶段影响大，相对排序才是决策关键
"""

import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 4
N_CH = 16          # 通道数：2^0 .. 2^15
N_ACTIONS = 4
VALUE_SCALE = 10000.0


def encode_board(board: np.ndarray) -> np.ndarray:
    """编码棋盘 uint8[16] -> 16x4x4 二值张量。

    通道 i 表示"该位置是否存在 2^i 的 tile"。
    """
    out = np.zeros((N_CH, GRID, GRID), dtype=np.float32)
    b = np.asarray(board, dtype=np.int32).reshape(GRID, GRID)
    for r in range(GRID):
        for c in range(GRID):
            v = int(b[r, c])
            if v > 0:
                out[min(v, N_CH - 1), r, c] = 1.0
    return out


def encode_batch(boards: np.ndarray) -> np.ndarray:
    """批量编码: (N,16) -> (N,16,4,4)"""
    return np.stack([encode_board(b) for b in boards]).astype(np.float32)


class StudentNet(nn.Module):
    """双头卷积网络。"""

    def __init__(self, feat_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_CH, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, hidden, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * GRID * GRID, feat_dim), nn.ReLU(),
        )
        self.policy = nn.Linear(feat_dim, N_ACTIONS)   # 各动作价值
        self.value = nn.Linear(feat_dim, 1)            # 局面价值

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B,16,4,4) -> (policy( B,4), value(B,1))"""
        h = self.fc(self.conv(x).flatten(1))
        return self.policy(h), self.value(h)


class StudentAgent:
    """推理封装（与 n-tuple 接口对齐，便于替换测试）。"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.net = StudentNet().to(self.device)
        if model_path and os.path.exists(model_path):
            sd = torch.load(model_path, map_location=self.device,
                            weights_only=False)
            self.net.load_state_dict(sd["model"])
            self.meta = sd.get("meta", {})
        else:
            self.meta = {}
        self.net.eval()

    @torch.no_grad()
    def predict(self, board: np.ndarray) -> np.ndarray:
        """返回 4 个动作的预测价值（相对值，无量纲）。"""
        x = torch.as_tensor(encode_board(board)[None], device=self.device)
        p, v = self.net(x)
        return p.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def predict_value(self, board: np.ndarray) -> float:
        x = torch.as_tensor(encode_board(board)[None], device=self.device)
        _, v = self.net(x)
        return float(v.item())


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
    """只在合法动作上计算 MSE。"""
    diff = (pred - target) ** 2 * valid
    denom = valid.sum().clamp(min=1.0)
    return diff.sum() / denom


# ---------------- 自测 ----------------
if __name__ == "__main__":
    import os

    torch.manual_seed(0)
    np.random.seed(0)

    print("=== 1. 棋盘编码 ===")
    b = np.zeros(16, dtype=np.uint8)
    b[0], b[5], b[10] = 1, 3, 7        # 2, 8, 128
    enc = encode_board(b)
    assert enc.shape == (16, 4, 4)
    assert enc[1, 0, 0] == 1.0 and enc[3, 1, 1] == 1.0 and enc[7, 2, 2] == 1.0
    assert enc.sum() == 3.0
    print("  ✓ 编码正确")

    print("=== 2. 网络前向 ===")
    net = StudentNet()
    x = torch.randn(8, 16, 4, 4)
    p, v = net(x)
    assert p.shape == (8, 4) and v.shape == (8, 1)
    n_params = sum(q.numel() for q in net.parameters())
    print(f"  ✓ 输出 {tuple(p.shape)} / {tuple(v.shape)} | 参数量 {n_params:,}")

    print("=== 3. masked loss ===")
    pred = torch.zeros(2, 4)
    tgt = torch.ones(2, 4)
    valid = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 0.]])
    loss = masked_mse(pred, tgt, valid)
    # 期望: 所有被计入的项都是 (0-1)^2 = 1 -> loss = 1
    assert abs(float(loss) - 1.0) < 1e-6, float(loss)
    print(f"  ✓ masked MSE = {float(loss):.4f}")

    print("=== 4. 推理速度（单局面）===")
    agent = StudentAgent()
    brd = np.random.randint(0, 12, size=16).astype(np.uint8)
    import time
    agent.predict(brd)                     # 预热
    t0 = time.time()
    N = 1000
    for _ in range(N):
        agent.predict(brd)
    el = (time.time() - t0) / N
    print(f"  ✓ 单次推理 {el*1e6:.0f} µs  (n-tuple 评估约 18 µs, 1-ply 决策 84 µs)")

    print("\nstudent_net.py 自测通过 ✓")
