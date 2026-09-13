# -*- coding: utf-8 -*-
"""
train_student.py - v4 学生网络训练（从搜索数据蒸馏）

训练目标
--------
给定 (局面, 各动作 2-ply 搜索价值)，训练学生网络模仿教师的决策：

    Policy 头: 预测【相对动作价值】(减去有效动作均值)
    Value 头 : 预测【最优动作价值】(减去均值, 即搜索给出的局面评分)

为什么用"相对价值"
------------------
绝对价值随对局阶段剧烈变化（开局约 2 万, 后期约 4 万），直接回归会让
网络把大量容量花在拟合"阶段"而非"决策差异"。减去有效动作均值后，
目标变成"哪个动作更好、好多少"，这才是决策所需的信息。

用法
----
    python train_student.py --data v4_data/train_2ply.npz \
        --out v4_data/student.pt --epochs 30
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (_HERE, os.path.dirname(_HERE), "/opt/2048ai", "/workspace/2048_ai"):
    if os.path.isfile(os.path.join(_c, "ntuple.py")):
        sys.path.insert(0, _c)
        break

from student_net import StudentNet, encode_batch, masked_mse, VALUE_SCALE


def load_dataset(path):
    d = np.load(path)
    boards = d["boards"]                    # (N,16) uint8
    values = d["values"]                    # (N,4) float32（含 -inf）
    print(f"数据: {len(boards):,} 样本 | {path}")
    return boards, values


def build_targets(values, temp=1.0):
    """把搜索价值转成训练目标。

    返回:
        rel_target : (N,4)   相对动作价值（减均值, 缩放到 ±1 量级）
        val_target : (N,)    局面价值（最优减均值, 同上缩放）
        valid      : (N,4)   合法动作 mask
    """
    v = values.astype(np.float64)
    valid = np.isfinite(v)
    vv = np.where(valid, v, 0.0)
    cnt = valid.sum(axis=1, keepdims=True).clip(min=1)
    mean = vv.sum(axis=1, keepdims=True) / cnt      # 有效动作均值
    best = np.where(valid, v, -np.inf).max(axis=1)  # 最优动作价值

    # 相对价值：只看"好多少"，与阶段无关；除以 VALUE_SCALE 缩放到 ±1 量级
    rel = (vv - mean) / VALUE_SCALE
    rel[~valid] = 0.0
    val = (best - mean[:, 0]) / VALUE_SCALE
    return rel.astype(np.float32), val.astype(np.float32), valid


def main():
    import argparse
    p = argparse.ArgumentParser(description="v4 学生网络训练")
    p.add_argument("--data", default="/opt/2048ai/v4_data/train_2ply.npz")
    p.add_argument("--out", default="/opt/2048ai/v4_data/student.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--policy-weight", type=float, default=1.0)
    p.add_argument("--value-weight", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    boards, values = load_dataset(args.data)
    rel, val, valid = build_targets(values)

    # 划分训练/验证
    N = len(boards)
    idx = np.random.permutation(N)
    n_val = int(N * args.val_split)
    tr, va = idx[n_val:], idx[:n_val]

    print(f"划分: 训练 {len(tr):,} | 验证 {len(va):,}")
    print(f"目标示例（相对价值）: {rel[tr[0]]}")

    dev = torch.device("cpu")
    net = StudentNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    # 预转张量（一次性, 避免每 batch 重复转换）
    def to_t(arr, dtype=torch.float32):
        return torch.as_tensor(arr, dtype=dtype, device=dev)

    X = to_t(encode_batch(boards))
    Trel = to_t(rel)
    Tval = to_t(val)
    Tvalid = to_t(valid)

    best_val = float("inf")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = torch.as_tensor(np.random.permutation(tr))
        tot_loss = tot_p = tot_v = 0.0
        nb = 0
        for i in range(0, len(perm), args.batch):
            bidx = perm[i:i + args.batch]
            x = X[bidx]
            p_pred, v_pred = net(x)
            lp = masked_mse(p_pred, Trel[bidx], Tvalid[bidx])
            lv = nn.functional.mse_loss(v_pred.squeeze(1), Tval[bidx])
            loss = args.policy_weight * lp + args.value_weight * lv
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss)
            tot_p += float(lp)
            tot_v += float(lv)
            nb += 1
        sched.step()

        # ---- 验证: 决策一致率 ----
        net.eval()
        with torch.no_grad():
            vx = X[va]
            pv, _ = net(vx)
            # 只比较"至少有 2 个合法动作"的样本，且教师有明确选择
            vmask = Tvalid[va]
            multi = vmask.sum(dim=1) >= 2
            pred_a = pv.masked_fill(vmask <= 0, float("-inf")).argmax(dim=1)
            # 教师最优 = 相对价值最大（等价于绝对值最大）
            tgt_a = Trel[va].masked_fill(vmask <= 0, float("-inf")).argmax(dim=1)
            acc = float((pred_a[multi] == tgt_a[multi]).float().mean())
            vl = float(masked_mse(pv, Trel[va], vmask))

        print(f"  epoch {ep:2d}/{args.epochs} | loss {tot_loss/nb:.5f} "
              f"(p {tot_p/nb:.5f} v {tot_v/nb:.5f}) | "
              f"val_policy {vl:.5f} | 决策一致率 {acc*100:.2f}%", flush=True)

        if vl < best_val:
            best_val = vl
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            tmp = args.out + ".tmp"
            torch.save({
                "model": net.state_dict(),
                "meta": {"epochs": args.epochs, "val_policy_mse": vl,
                         "decision_acc": acc, "n_train": len(tr),
                         "value_scale": VALUE_SCALE,
                         "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            }, tmp)
            os.replace(tmp, args.out)

    print(f"\n训练完成 ({time.time()-t0:.0f}s) | 最优验证 policy MSE {best_val:.5f}")
    print(f"模型: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
