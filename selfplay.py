# -*- coding: utf-8 -*-
"""
selfplay.py - 自我对弈进化模块

机制
----
1. 冠军模型（champion）保存在 <models_dir>/best.pt;
2. 定期让当前训练中的挑战者（challenger, 即 policy_net 当前权重）
   与冠军各玩 50 局（纯贪心推理, 无探索、无 MCTS）;
3. 挑战者平均分超过冠军 5% 才替换冠军, 否则保留冠军继续训练;
4. 维护模型版本历史 versions.json, 记录每次挑战与换代。

这样即使训练中后期出现性能回退, 也不会污染"最佳模型"。
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from env import Game2048
from model import QNetwork, atomic_torch_save, load_torch

logger = logging.getLogger("2048")


def play_greedy_games(policy_sd: Dict[str, torch.Tensor], n_games: int = 50,
                      device: str = "cpu") -> Tuple[List[int], List[int], List[int]]:
    """用给定 policy 权重纯贪心（argmax Q over 合法动作）玩 n_games 局。

    返回: (每局分数列表, 每局步数列表, 每局最大方块列表)
    """
    net = QNetwork().to(device)
    net.load_state_dict(policy_sd)
    net.eval()

    scores: List[int] = []
    steps_list: List[int] = []
    max_tiles: List[int] = []
    with torch.no_grad():
        for _ in range(n_games):
            env = Game2048()          # 每局不同随机种子, 保证覆盖性
            state = env.reset()
            done = False
            while not done:
                valid = env.get_valid_actions()
                if not valid:
                    break
                s = torch.as_tensor(state, dtype=torch.float32,
                                    device=device).reshape(1, -1)
                q = net(s).squeeze(0)
                action = max(valid, key=lambda a: float(q[a]))
                state, _, done, _ = env.step(action)
            scores.append(env.score)
            steps_list.append(env.steps)
            max_tiles.append(env.max_tile)
    return scores, steps_list, max_tiles


def _atomic_json_write(obj, path: str) -> None:
    """JSON 原子写入（临时文件 + os.replace）。"""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class SelfPlayManager:
    """自我对弈挑战管理器。

    用法（训练循环内, 每 N 局调用一次）:
        result = manager.challenge(policy_state_dict)
        if result["passed"]:
            logger.info("新模型晋级: %.0f vs %.0f", ...)
    """

    def __init__(self, models_dir: str, improve_margin: float = 0.05,
                 n_games: int = 50, device: str = "cpu"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.champion_path = os.path.join(models_dir, "best.pt")
        self.history_path = os.path.join(models_dir, "versions.json")
        self.improve_margin = improve_margin   # 晋级门槛: 平均分领先比例
        self.n_games = n_games                 # 双方各玩局数
        self.device = device
        self.version = self._load_version()

    # ---------- 版本历史 ----------
    def _load_version(self) -> int:
        """从 versions.json 恢复当前版本号。"""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                if history:
                    return int(history[-1]["version"])
            except (json.JSONDecodeError, KeyError, IndexError):
                logger.warning("versions.json 损坏, 版本号重置为 0")
        return 0

    def _load_history(self) -> List[dict]:
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _push_history(self, record: dict) -> None:
        history = self._load_history()
        history.append(record)
        _atomic_json_write(history, self.history_path)  # 原子写入防损坏

    # ---------- 冠军模型 ----------
    def has_champion(self) -> bool:
        return os.path.exists(self.champion_path)

    def load_champion_sd(self) -> Optional[Dict[str, torch.Tensor]]:
        """加载冠军模型权重; 不存在返回 None。"""
        if not self.has_champion():
            return None
        try:
            ckpt = load_torch(self.champion_path, map_location="cpu")
            return ckpt["policy_sd"]
        except Exception as exc:
            logger.error("冠军模型加载失败: %s", exc)
            return None

    def _promote(self, policy_sd: Dict[str, torch.Tensor], avg_score: float,
                 prev_avg: Optional[float], reason: str) -> None:
        """挑战者晋级: 原子保存新冠军 + 追加版本历史。"""
        self.version += 1
        atomic_torch_save({
            "policy_sd": {k: v.cpu() for k, v in policy_sd.items()},
            "version": self.version,
            "avg_score": avg_score,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, self.champion_path)
        self._push_history({
            "version": self.version,
            "avg_score": round(avg_score, 1),
            "prev_champion_avg": None if prev_avg is None else round(prev_avg, 1),
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        logger.info("★ 新冠军 v%d 诞生 (avg=%.0f, prev=%s, 原因=%s)",
                    self.version, avg_score,
                    "N/A" if prev_avg is None else f"{prev_avg:.0f}", reason)

    # ---------- 挑战流程 ----------
    def challenge(self, challenger_sd: Dict[str, torch.Tensor]) -> Dict:
        """发起一次挑战: 挑战者 vs 冠军各玩 n_games 局。

        晋级条件: 挑战者平均分 >= 冠军平均分 * (1 + improve_margin)。
        返回 dict: passed / challenger_avg / champion_avg / version 等。
        """
        ch_scores, ch_steps, ch_tiles = play_greedy_games(
            challenger_sd, self.n_games, self.device)
        challenger_avg = float(np.mean(ch_scores))

        champion_sd = self.load_champion_sd()
        if champion_sd is None:
            # 尚无冠军, 首个模型直接成为 v1
            self._promote(challenger_sd, challenger_avg, None, "initial")
            return {"passed": True, "challenger_avg": challenger_avg,
                    "champion_avg": None, "version": self.version,
                    "reason": "initial"}

        cp_scores, cp_steps, cp_tiles = play_greedy_games(
            champion_sd, self.n_games, self.device)
        champion_avg = float(np.mean(cp_scores))

        passed = challenger_avg >= champion_avg * (1 + self.improve_margin)
        if passed:
            self._promote(challenger_sd, challenger_avg, champion_avg, "improved")
        else:
            logger.info("挑战失败: 挑战者 %.0f 未超过冠军 %.0f 的 %.0f%%, 保留旧模型",
                        challenger_avg, champion_avg, self.improve_margin * 100)

        return {
            "passed": passed,
            "challenger_avg": challenger_avg,
            "champion_avg": champion_avg,
            "challenger_max": max(ch_tiles),
            "champion_max": max(cp_tiles),
            "challenger_steps": float(np.mean(ch_steps)),
            "champion_steps": float(np.mean(cp_steps)),
            "version": self.version,
            "reason": "improved" if passed else "retained",
        }


# ---------------- 模块自测 ----------------
if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory() as td:
        mgr = SelfPlayManager(td, improve_margin=0.05, n_games=5)
        sd = QNetwork().state_dict()
        r1 = mgr.challenge(sd)                 # 首次: 直接成为 v1
        assert r1["passed"] and mgr.version == 1
        r2 = mgr.challenge(sd)                 # 同权重再挑战: 大概率失败
        print(f"挑战1: {r1['passed']}  挑战2: {r2['passed']}  "
              f"chall={r2['challenger_avg']:.0f} champ={r2['champion_avg']:.0f}")
        assert os.path.exists(mgr.champion_path)
        assert os.path.exists(mgr.history_path)
        assert len(mgr._load_history()) >= 1
    print("selfplay.py 自测通过 ✓")
