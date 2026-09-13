# -*- coding: utf-8 -*-
"""
env.py - 2048 游戏环境（含奖励塑形 Reward Shaping）

功能
----
1. 完整实现 2048 游戏规则（4x4 网格）:
   - 动作空间: 0=上(UP) 1=下(DOWN) 2=左(LEFT) 3=右(RIGHT)
   - 每次有效移动后随机在空位生成新方块: 90% 概率为 2, 10% 概率为 4
   - 相同数字方块合并, 合并产生的新方块值计入游戏分数
   - 游戏结束判定: 棋盘无空格且任意方向均无法合并
2. 标准强化学习接口:
   - reset()              -> state (np.float32, shape=[4, 4])
   - step(action)         -> (next_state, reward, done, info)
   - render()             -> 打印当前棋盘
   - get_valid_actions()  -> 当前合法动作列表
3. 奖励塑形 (总奖励 = 以下各项加权和, 权重见 REWARD_WEIGHTS):
   - 合并奖励     = 本步所有合并产生的新方块值之和
   - 空位奖励     = 当前空格子数 * 0.1
   - 单调性奖励   = 棋盘单调性得分 * 0.01   (0 表示完美单调, 越负越乱)
   - 最大方块奖励 = 最大方块的 log2 值 * 0.5
   - 终局惩罚     = -10

设计说明
--------
棋盘运算（滑动/合并/生成方块/终局检测/奖励计算）全部实现为模块级
纯函数, 不依赖类实例状态 —— 这样 mcts.py 可以在 multiprocessing 子
进程中直接复用同一套逻辑, 而无需 pickle 整个环境对象。

状态表示
--------
4x4 网格每个方块取 log2, 空位为 0, 再除以 11（2048 = 2^11）归一化到
[0, 1], 返回 np.float32 的 (4, 4) 数组。
"""

import math
import random

import numpy as np

# ---------------- 动作与常量 ----------------
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = {UP: "上", DOWN: "下", LEFT: "左", RIGHT: "右"}
GRID_SIZE = 4
NORM_FACTOR = 11.0          # 2048 = 2**11, 用于状态归一化
GAME_OVER_PENALTY = -10.0   # 终局惩罚

# 各奖励分量的权重（"总奖励 = 加权和"）。若训练中合并奖励量级过大
# 导致不稳定, 可将 'merge' 调低（如 0.1）。
REWARD_WEIGHTS = {
    "merge": 1.0,
    "empty": 1.0,
    "mono": 1.0,
    "max_tile": 1.0,
    "game_over": 1.0,
}


# ================= 模块级纯函数（MCTS 子进程复用同一逻辑） =================

def slide_row_left(row):
    """单行向左滑动 + 合并（2048 核心规则）。

    规则: 非零方块压缩靠左; 相邻相等方块合并一次
    （每方块每步最多合并一次, 如 [2,2,2,2] -> [4,4,0,0]）。

    返回: (new_row, merged_sum, moved)
        new_row    合并后的新行 (list)
        merged_sum 本行合并产生的新方块值之和
        moved      行是否发生变化
    """
    tiles = [v for v in row if v != 0]
    out = []
    merged_sum = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] * 2
            out.append(merged)
            merged_sum += merged
            i += 2                      # 跳过两个已合并方块
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (len(row) - len(out))  # 右侧补零
    return out, merged_sum, out != list(row)


def move_grid(grid, action):
    """对 4x4 棋盘执行一次滑动动作。

    参数:
        grid   : 4x4 嵌套 list/tuple 的整数棋盘
        action : UP / DOWN / LEFT / RIGHT
    返回:
        (new_grid, merged_sum, moved)
        new_grid 为 tuple[tuple[int]] 的不可变棋盘。
    """
    g = [list(r) for r in grid]

    # 通过 转置/翻转 将四个方向统一成"向左滑动"
    if action == LEFT:
        rows = g
    elif action == RIGHT:
        rows = [r[::-1] for r in g]
    elif action == UP:
        rows = [list(col) for col in zip(*g)]            # 转置
    elif action == DOWN:
        rows = [list(col)[::-1] for col in zip(*g)]      # 转置 + 上下翻转
    else:
        raise ValueError(f"非法动作: {action}")

    new_rows, merged_sum = [], 0
    for row in rows:
        nr, s, _ = slide_row_left(row)
        new_rows.append(nr)
        merged_sum += s

    # 逆变换还原方向
    # 【重要】DOWN 的还原必须是"先逐行反转, 再转置"（即 [::-1] 作用在行上）,
    # 而不是"先转置, 再反转各列" —— 两者不等价, 写错会导致 DOWN 完全错乱
    # （曾导致 UD_flip 对称性检查 400/400 失败, 严重损害训练效果）。
    if action == LEFT:
        ng = new_rows
    elif action == RIGHT:
        ng = [r[::-1] for r in new_rows]
    elif action == UP:
        ng = [list(col) for col in zip(*new_rows)]
    else:  # DOWN
        ng = [list(col) for col in zip(*[r[::-1] for r in new_rows])]

    moved = ng != g
    return tuple(tuple(r) for r in ng), merged_sum, moved


def can_move(grid):
    """棋盘是否还能移动（有空格或存在相邻相等方块）。"""
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            v = grid[r][c]
            if v == 0:
                return True
            if c + 1 < len(grid[0]) and grid[r][c + 1] == v:
                return True
            if r + 1 < len(grid) and grid[r + 1][c] == v:
                return True
    return False


def is_terminal(grid):
    """终局判定: 无空格且无可合并方块。"""
    return not can_move(grid)


def add_random_tile(grid, rng=None):
    """随机在空位生成新方块: 90% 概率 2, 10% 概率 4。返回新棋盘 (list[list])。"""
    rng = rng or random
    g = [list(r) for r in grid]
    empties = [(r, c) for r in range(len(g)) for c in range(len(g[0])) if g[r][c] == 0]
    if not empties:
        return g
    r, c = rng.choice(empties)
    g[r][c] = 2 if rng.random() < 0.9 else 4
    return g


def monotonicity_score(grid):
    """计算棋盘单调性得分（ovolve 启发式的简化实现）。

    对每一行/每一列, 计算相邻方块 log2 值的差分, 分别累计
    "递增方向惩罚"与"递减方向惩罚"; 取四个方向中惩罚最小者。

    返回: -max_penalty  =>  0 表示存在完美单调方向, 越负表示越混乱。
    """
    n = len(grid)
    totals = [0.0, 0.0, 0.0, 0.0]  # [上, 下, 左, 右] 四个方向的惩罚累计

    def lg(v):  # 空位按 0 处理
        return math.log2(v) if v > 0 else 0.0

    # 行方向（左/右）
    for r in range(n):
        for c in range(n - 1):
            a, b = lg(grid[r][c]), lg(grid[r][c + 1])
            if a > b:
                totals[2] += b - a
                totals[3] -= b - a
            else:
                totals[3] += a - b
                totals[2] -= a - b

    # 列方向（上/下）
    for c in range(n):
        for r in range(n - 1):
            a, b = lg(grid[r][c]), lg(grid[r + 1][c])
            if a > b:
                totals[0] += b - a
                totals[1] -= b - a
            else:
                totals[1] += a - b
                totals[0] -= a - b

    return -max(totals)


def count_empty(grid):
    """统计空格子数量。"""
    return sum(1 for row in grid for v in row if v == 0)


def compute_reward(grid_after, merged_sum, done, weights=None):
    """奖励塑形: 总奖励 = 各分项加权和（见模块 docstring）。

    参数:
        grid_after : 本步移动 + 生成新方块之后的棋盘
        merged_sum : 本步合并产生的新方块值之和
        done       : 是否终局
        weights    : 奖励权重字典（None 则用默认 REWARD_WEIGHTS）
    """
    w = weights or REWARD_WEIGHTS
    empty_r = count_empty(grid_after) * 0.1
    mono_r = monotonicity_score(grid_after) * 0.01
    max_tile = max(max(row) for row in grid_after)
    max_r = math.log2(max_tile) * 0.5 if max_tile > 0 else 0.0
    over_r = GAME_OVER_PENALTY if done else 0.0
    return (w["merge"] * merged_sum
            + w["empty"] * empty_r
            + w["mono"] * mono_r
            + w["max_tile"] * max_r
            + w["game_over"] * over_r)


def state_from_grid(grid):
    """棋盘 -> 归一化状态张量。

    每格取 log2（空位 0）, 除以 11 归一化到 [0, 1]。
    返回 np.float32, shape=(4, 4)。
    """
    arr = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            v = grid[r][c]
            if v > 0:
                arr[r][c] = math.log2(v) / NORM_FACTOR
    return arr


def grid_to_tuples(grid):
    """将棋盘转为不可变 tuple[tuple[int]]（便于 pickle 与哈希）。"""
    return tuple(tuple(int(v) for v in row) for row in grid)


# ================= 游戏环境类 =================

class Game2048:
    """2048 游戏 Gym 风格环境。

    用法:
        env = Game2048(seed=42)
        state = env.reset()
        next_state, reward, done, info = env.step(LEFT)
        env.render()

    参数 max_steps: 单局最大步数上限（None = 不限制）。
        注意: 本项目的奖励塑形包含"每一步都给"的空位奖励与最大方块奖励
        （见 compute_reward）, 因此理论上存在"保持棋盘不满、无限存活"的
        退化解 —— 步数上限是防止该退化解阻塞训练的必要保护。
    """

    def __init__(self, seed=None, reward_weights=None, max_steps=None):
        self.rng = random.Random(seed)
        self.reward_weights = dict(reward_weights or REWARD_WEIGHTS)
        self.max_steps = max_steps
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.score = 0
        self.steps = 0

    # ---------- 生命周期 ----------
    def reset(self):
        """重置环境, 随机生成 2 个初始方块, 返回初始状态。"""
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.score = 0
        self.steps = 0
        self._spawn_tile()
        self._spawn_tile()
        return self.get_state()

    def _spawn_tile(self):
        """在随机空位生成 2（90%）或 4（10%）。"""
        empties = [(r, c) for r in range(GRID_SIZE)
                   for c in range(GRID_SIZE) if self.grid[r][c] == 0]
        if not empties:
            return
        r, c = self.rng.choice(empties)
        self.grid[r][c] = 2 if self.rng.random() < 0.9 else 4

    # ---------- RL 接口 ----------
    def step(self, action):
        """执行动作。

        返回: (next_state, reward, done, info)
            非法动作: 状态不变, reward=0, done=False, info['invalid']=True
            有效动作: 滑动合并 -> 随机生成新方块 -> 计算塑形奖励
        info: {invalid, score, max_tile, merged_sum, steps}
        """
        if action not in (UP, DOWN, LEFT, RIGHT):
            raise ValueError(f"非法动作: {action}")

        new_grid, merged_sum, moved = move_grid(self.grid, action)
        if not moved:
            # 非法移动: 棋盘不变化, 不生成新方块
            return (self.get_state(), 0.0, False,
                    {"invalid": True, "score": self.score,
                     "max_tile": self.max_tile, "merged_sum": 0,
                     "steps": self.steps})

        self.grid = [list(r) for r in new_grid]
        self.score += merged_sum
        self.steps += 1
        self._spawn_tile()

        terminal = is_terminal(self.grid)
        done = terminal
        truncated = False
        # 步数上限保护: 防止"无限存活"退化解把训练卡死
        if not done and self.max_steps is not None and self.steps >= self.max_steps:
            done = True
            truncated = True
        info = {"invalid": False, "score": self.score,
                "max_tile": self.max_tile, "merged_sum": merged_sum,
                "steps": self.steps, "truncated": truncated}
        reward = compute_reward(self.grid, merged_sum, done, self.reward_weights)
        return self.get_state(), reward, done, info

    def get_state(self):
        """返回归一化状态 (np.float32, [4, 4])。"""
        return state_from_grid(self.grid)

    def get_valid_actions(self):
        """返回当前合法动作列表（执行后棋盘会发生变化的动作）。"""
        return [a for a in (UP, DOWN, LEFT, RIGHT) if move_grid(self.grid, a)[2]]

    # ---------- 展示 ----------
    def render(self):
        """打印当前棋盘（人类可读格式）。"""
        print("+" + "-" * (8 * GRID_SIZE + 1) + "+")
        for row in self.grid:
            cells = []
            for v in row:
                cells.append(f"{v:^7}" if v else ".".center(7))
            print("|" + "|".join(f" {c} " for c in cells) + "|")
        print("+" + "-" * (8 * GRID_SIZE + 1) + "+")
        print(f"分数: {self.score}   步数: {self.steps}   最大方块: {self.max_tile}")

    # ---------- 属性 ----------
    @property
    def max_tile(self):
        """当前最大方块值。"""
        return max(max(row) for row in self.grid)


# ---------------- 模块自测 ----------------
if __name__ == "__main__":
    env = Game2048(seed=0)
    state = env.reset()
    print("初始状态 shape:", state.shape, "dtype:", state.dtype)
    env.render()
    done, total_reward = False, 0.0
    while not done:
        valid = env.get_valid_actions()
        action = env.rng.choice(valid)
        state, reward, done, info = env.step(action)
        total_reward += reward
    env.render()
    print(f"随机策略总塑形奖励: {total_reward:.2f}")

    # 基础规则单测
    assert slide_row_left([2, 2, 2, 2])[0] == [4, 4, 0, 0]   # 每块只合并一次
    assert slide_row_left([4, 2, 2, 0])[0] == [4, 4, 0, 0]   # 只合并相邻的 2
    assert slide_row_left([2, 0, 2, 4])[0] == [4, 4, 0, 0]   # 压缩后合并
    assert slide_row_left([0, 0, 0, 2])[0] == [2, 0, 0, 0]   # 仅滑动
    assert slide_row_left([2, 0, 0, 0])[2] is False          # 无变化
    g4 = [[4, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    ng, ms, mv = move_grid(g4, UP)
    assert ng[0][0] == 8 and ms == 8 and mv, "UP 合并错误"
    assert move_grid(g4, LEFT)[2] is False, "LEFT 不应产生变化"
    assert is_terminal([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert not is_terminal([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 0]])

    # ---- 方向对称性回归测试（锁死历史上出现过的 DOWN 实现 bug）----
    # 原理: 2048 的规则在 上下翻转 / 左右翻转 / 转置 / 旋转 下完全对称,
    #       因此 move 必须满足这些等价关系。历史上 DOWN 的逆变换顺序写错,
    #       导致 UD_flip 全部失败、DOWN 行为错乱, 训练效果严重受损。
    _rng = random.Random(12345)
    for _ in range(300):
        g = [[_rng.choice([0, 0, 0, 2, 4, 8, 16, 32]) for _ in range(4)]
             for _ in range(4)]
        gT = [list(r) for r in zip(*g)]                 # 转置

        def _flipud(m):
            return [list(r) for r in m[::-1]]

        def _fliplr(m):
            return [list(r)[::-1] for r in m]

        # 上下翻转: DOWN 等价于 翻转后 UP 再翻回
        assert move_grid(g, DOWN)[0] == tuple(map(tuple, _flipud(
            move_grid(_flipud(g), UP)[0]))), "DOWN/UP 上下对称性失败"
        # 左右翻转: RIGHT 等价于 翻转后 LEFT 再翻回
        assert move_grid(g, RIGHT)[0] == tuple(map(tuple, _fliplr(
            move_grid(_fliplr(g), LEFT)[0]))), "RIGHT/LEFT 左右对称性失败"
        # 转置: LEFT 与 UP 互换
        assert move_grid(g, LEFT)[0] == tuple(map(tuple, zip(*
            move_grid(gT, UP)[0]))), "LEFT/UP 转置对称性失败"
        assert move_grid(g, UP)[0] == tuple(map(tuple, zip(*
            move_grid(gT, LEFT)[0]))), "UP/LEFT 转置对称性失败"
    print("方向对称性回归测试通过 ✓ (上下/左右/转置 各 300 例)")

    # ---- DOWN 具体语义检查 ----
    g_down = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0]]
    ng_down = move_grid(g_down, DOWN)[0]
    assert ng_down[3][0] == 4 and ng_down[2][0] == 2, \
        f"DOWN 语义错误: {ng_down}（应 2 落在倒数第二行、4 落在最后一行）"
    print("DOWN 语义检查通过 ✓")

    print("env.py 自测通过 ✓")
