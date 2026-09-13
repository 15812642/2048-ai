# -*- coding: utf-8 -*-
"""
web/app.py - 2048 AI 训练系统 · Web 控制台后端（Flask）

功能
----
1. 实时监控 : 读取 outputs/status.json 展示训练进度（局数/ε/buffer/曲线数据）
2. 训练控制 : 网页启动 / 停止训练进程（SIGINT 停止, 触发 checkpoint 正常保存）
3. AI 演示  : SSE 流式推送 AI 玩 2048 的每一步棋盘, 前端动画播放
4. 模型管理 : 列出 checkpoint / best.pt / 自我对弈版本历史
5. 日志查看 : 实时 tail 训练日志

接口一览
--------
GET  /                    控制台页面
GET  /api/status          训练状态（status.json + 进程状态 + 模型信息）
GET  /api/logs?lines=200  训练日志尾部
GET  /api/models          模型文件与版本历史
POST /api/train/start     {episodes, parallel, no_mcts, resume} 启动训练
POST /api/train/stop      停止训练（SIGINT, 保存进度）
POST /api/eval            {model, games} 快速评测模型, 返回统计
GET  /api/play            SSE: AI 演示一局（model/delay/mcts 参数）

启动: python web/app.py --port 8080 --host 0.0.0.0
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Dict, Optional

from flask import (Flask, Response, jsonify, render_template, request,
                   stream_with_context)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJ_DIR)          # 使 env/model/selfplay 可导入

OUT_DIR = os.environ.get("OUT_DIR", os.path.join(PROJ_DIR, "outputs"))
MODELS_DIR = os.path.join(OUT_DIR, "models")
LOGS_DIR = os.path.join(OUT_DIR, "logs")

app = Flask(__name__, static_folder="static", template_folder="templates")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("web")

# 训练子进程句柄: 按算法分槽（v1 / v2 可各自独立运行）
_train_proc: Dict[str, Optional[subprocess.Popen]] = {"v1": None, "v2": None}


# ==================== 工具函数 ====================

def _read_json(path: str, default=None):
    """安全读取 JSON 文件。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _data_dir(algo: str = "v1") -> str:
    """算法对应的数据目录。

    v1 (经典 DQN) 使用 OUT_DIR 根目录; v2 (Rainbow/H-DQN) 隔离在 OUT_DIR/v2,
    两套训练数据、模型、日志完全独立, 便于 A/B 对比。
    """
    if algo in ("v2", "v3"):
        return os.path.join(OUT_DIR, algo)
    if algo.startswith("exp_"):          # 并行实验目录 exp_a / exp_b / exp_c
        return os.path.join(os.path.dirname(OUT_DIR), algo)
    return OUT_DIR


def _best_experiment_dir(by: str = "episode") -> Optional[str]:
    """返回最靠前的实验目录（exp_*/v3 或 exp_*），供 v3 视图回退使用。

    by="episode" : 取训练局数最多者（进度最靠前）
    by="score"   : 取近期均分最高者（水平最好）
    """
    import glob as _glob
    root = os.path.dirname(OUT_DIR)
    best, best_key = None, -1.0
    for pat in (os.path.join(root, "exp_*", "v3"), os.path.join(root, "exp_*")):
        for d in _glob.glob(pat):
            if not os.path.isdir(d):        # 排除 exp_a.log 之类的同名文件
                continue
            if not os.path.exists(os.path.join(d, "status.json")):
                continue
            st = _read_json(os.path.join(d, "status.json"), default=None)
            if not st:
                continue
            if by == "score":
                s = st.get("recent_scores") or []
                key = (sum(s[-200:]) / len(s[-200:])) if s else 0.0
            else:
                key = float(st.get("episode") or 0)
            if key > best_key:
                best, best_key = d, key
    return best


def _train_running(algo: str = "v1") -> bool:
    """指定算法的训练进程是否仍在运行。"""
    proc = _train_proc.get(algo) if isinstance(_train_proc, dict) else None
    return proc is not None and proc.poll() is None


def _resolve_model(algo: str, name: str):
    """统一解析模型文件路径。返回 (绝对路径, 来源标签) 或 (None, None)。

    搜索顺序:
      1. 该算法自己的目录 (<out>/<algo>/models) —— v1/v2/v3
      2. 若未命中, 扫描全部模型目录（含并行实验 exp_*/models）
    这样评测/演示/下载都能找到实际存在的模型, 不再出现
    "模型不存在" 却明明看得见的矛盾。
    """
    if not name or "/" in name or "\\" in name:
        return None, None
    # 1) 本算法主目录（若确实有该文件）
    d = os.path.join(_data_dir(algo), "models")
    p = os.path.join(d, name)
    if os.path.isfile(p):
        return p, algo
    # 2) 扫描全部目录 —— 按【质量】排序, 同名模型优先取最强实验的
    for label, d in _all_model_dirs(by_quality=True):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p, label
    return None, None


def _model_progress(name: str) -> dict:
    """查询某个模型文件在各来源的"质量"信息（局数/最佳分）—— 供前端展示。"""
    out = {}
    for label, d in _all_model_dirs(by_quality=True):
        if not os.path.isfile(os.path.join(d, name)):
            continue
        parent = os.path.dirname(d)
        st = _read_json(os.path.join(parent, "status.json"), default=None)
        if st is None:
            st = _read_json(os.path.join(parent, "v3", "status.json"),
                            default={}) or {}
        if isinstance(st, dict):
            out[label] = {"episode": st.get("episode"),
                          "best_eval": st.get("best_eval_score"),
                          "running": st.get("running")}
    return out


def _all_model_dirs(by_quality: bool = False) -> list:
    """列出所有模型目录: [(来源标签, 绝对路径), ...]

    覆盖 v1/v2/v3 主目录与全部实验目录。

    参数 by_quality:
        True  -> 按【质量】排序（运行中的实验优先, 其次训练局数多的）,
                 用于"同名模型解析"——best.npz 存在多处时优先取最强的那个。
        False -> 固定顺序（前端列表展示用, 保持稳定）。
    """
    import glob as _glob
    root = os.path.dirname(OUT_DIR)
    out = [
        ("v1", os.path.join(OUT_DIR, "models")),
        ("v2", os.path.join(OUT_DIR, "v2", "models")),
        ("v3", os.path.join(OUT_DIR, "v3", "models")),
    ]
    for pat in (os.path.join(root, "exp_*", "v3", "models"),
                os.path.join(root, "exp_*", "models")):
        for d in sorted(_glob.glob(pat)):
            if not os.path.isdir(d):
                continue
            label = os.path.basename(os.path.dirname(d))
            if label == "v3":
                label = os.path.basename(os.path.dirname(os.path.dirname(d)))
            if any(label == lb for lb, _ in out):
                continue
            out.append((label, d))
    dirs = [(lb, d) for lb, d in out if os.path.isdir(d)]

    if by_quality:
        def score(item):
            label, d = item
            st = _read_json(os.path.join(os.path.dirname(d), "status.json"),
                            default=None)
            if st is None:
                # 并行训练器的 status.json 在 <exp>/status.json
                st = _read_json(os.path.join(os.path.dirname(d), "status.json"),
                                default={}) or {}
            ep = float(st.get("episode") or 0) if isinstance(st, dict) else 0
            running = 1 if (isinstance(st, dict) and st.get("running")) else 0
            best = float(st.get("best_eval_score") or 0) if isinstance(st, dict) else 0
            return (running, best, ep)
        try:
            dirs = sorted(dirs, key=score, reverse=True)
        except Exception:
            pass
    return dirs


def _list_models(models_dir: str, source: str) -> list:
    """列出某目录下的模型文件（含大小/时间/来源/可否下载）。"""
    files = []
    if not os.path.isdir(models_dir):
        return files
    for name in sorted(os.listdir(models_dir)):
        if not name.endswith((".pt", ".npz")):
            continue
        path = os.path.join(models_dir, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        files.append({
            "name": name, "source": source,
            "size_mb": round(st.st_size / 1048576, 2),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                   time.localtime(st.st_mtime)),
            "dir": models_dir,
        })
    return files


def _model_info(models_dir: Optional[str] = None) -> dict:
    """汇总模型目录信息（best.pt / latest.pt 的时间与大小）。"""
    models_dir = models_dir or MODELS_DIR
    info = {}
    for name in ("best.pt", "latest.pt", "best.npz", "latest.npz"):
        path = os.path.join(models_dir, name)
        if os.path.exists(path):
            st = os.stat(path)
            info[name] = {"size_mb": round(st.st_size / 1048576, 2),
                          "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                                 time.localtime(st.st_mtime))}
    return info


# ==================== 页面 ====================

@app.route("/")
def index():
    return render_template("index.html")


# ==================== 状态 / 日志 / 模型 ====================

@app.route("/api/status")
def api_status():
    """训练状态: status.json + 进程状态 + 模型信息（?algo=v1|v2|v3）。

    若主目录（如 outputs/v3）尚无 status.json 而 exp_*/ 下存在正在运行的
    实验，则自动回退展示【进度最靠前的实验】，避免页面空白。
    """
    algo = request.args.get("algo", "v1")
    base = _data_dir(algo)
    status = _read_json(os.path.join(base, "status.json"), default={}) or {}

    if not status and algo == "v3":     # 仅 v3 回退到实验（实验均为 v3）
        exp_dir = _best_experiment_dir()
        if exp_dir:
            st = _read_json(os.path.join(exp_dir, "status.json"), default=None)
            if st:
                status = st
                status["_fallback_from_exp"] = True
                # 名字取实验目录名（去掉 v3 中间层）
                p_ = exp_dir
                if os.path.basename(p_) == "v3":
                    p_ = os.path.dirname(p_)
                status["_exp_name"] = os.path.basename(p_)
    versions = _read_json(os.path.join(base, "models", "versions.json"),
                          default=[]) or []
    status["algo"] = algo
    status["proc_running"] = _train_running(algo)
    status["train_pid"] = (_train_proc[algo].pid
                           if _train_running(algo) else None)
    models_dir = os.path.join(base, "models")
    if not os.path.isdir(models_dir) or not os.listdir(models_dir):
        exp_dir = _best_experiment_dir()
        if exp_dir and os.path.isdir(os.path.join(exp_dir, "models")):
            models_dir = os.path.join(exp_dir, "models")
    status["models"] = _model_info(models_dir)
    status["_models_dir"] = models_dir
    status["versions"] = versions[-20:]
    return jsonify(status)


@app.route("/api/logs")
def api_logs():
    """返回训练日志尾部若干行。"""
    lines = int(request.args.get("lines", 200))
    algo = request.args.get("algo", "v1")
    path = os.path.join(_data_dir(algo), "logs", "train.log")
    if algo == "v3" and not os.path.exists(path):
        exp_dir = _best_experiment_dir()
        if exp_dir:
            path = os.path.join(exp_dir, "logs", "train.log")
    if not os.path.exists(path):
        return jsonify({"lines": [], "path": path, "exists": False})
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.readlines()
        return jsonify({"lines": [ln.rstrip("\n") for ln in content[-lines:]],
                        "total": len(content), "exists": True})
    except Exception as exc:
        return jsonify({"lines": [f"[读取失败] {exc}"], "exists": False})


@app.route("/api/compare")
def api_compare():
    """并行实验对比: 扫描 exp_*/status.json 汇总关键指标。

    用于长跑实验的横向对比（例如 overnight 跑多个超参配置，早上取最优）。
    """
    import glob as _glob
    rows = []
    # v3 训练器会在 out-dir 下再建一层 v3/, 故两种层级都要扫描
    pats = [os.path.join(os.path.dirname(OUT_DIR), "exp_*", "status.json"),
            os.path.join(os.path.dirname(OUT_DIR), "exp_*", "v3", "status.json")]
    paths = []
    for pt in pats:
        paths.extend(_glob.glob(pt))
    for path in sorted(paths):
        st = _read_json(path, default={}) or {}
        ev = st.get("evals") or []
        cfg = st.get("config") or {}
        s = st.get("recent_scores") or []
        # v3 训练器多套一层 v3/ 目录, 名字取真正的实验目录名
        parent = os.path.dirname(path)
        if os.path.basename(parent) in ("v3", "v2"):
            parent = os.path.dirname(parent)
        rows.append({
            "name": os.path.basename(parent),
            "algo": st.get("algo"),
            "episode": st.get("episode") or 0,
            # 曲线对比用: 尾部 300 局积分 + 起始局号
            "score_series": [int(x) for x in (st.get("recent_scores") or [])[-300:]],
            "score_start": st.get("recent_scores_start") or 0,
            "delta_series": [round(float(x), 2)
                             for x in (st.get("losses_tail") or [])[-600:]],
            "target_episodes": (cfg.get("episodes") or 0),
            "elapsed_sec": st.get("elapsed_sec") or 0,
            # 速率优先取 status.json 自报值（训练器已扣除续训的基础局数）；
            # 老版本 status 无该字段时按 course 计算会因续训而虚高，故兜底用
            # 「本进程新增局数 / 已运行秒数」。
            "episodes_per_sec": (st.get("episodes_per_sec")
                                 if st.get("episodes_per_sec") is not None
                                 else round((st.get("processed_this_run")
                                             or st.get("episode") or 0)
                                            / max(1, st.get("elapsed_sec") or 1), 2)),
            "recent_avg": round(sum(s[-200:]) / len(s[-200:]), 1) if s else 0,
            "best_eval": st.get("best_eval_score"),
            "last_eval": ev[-1] if ev else None,
            "patterns": cfg.get("pattern_set"),
            "lam": cfg.get("lam"),
            "alpha": cfg.get("alpha"),
            "v_init": cfg.get("v_init"),
            "running": st.get("running"),
        })
    return jsonify({"experiments": rows,
                    "base": os.path.dirname(OUT_DIR)})


@app.route("/api/models")
def api_models():
    """列出模型文件与自我对弈版本历史（?algo=v1|v2）。"""
    algo = request.args.get("algo", "v1")

    # ---- algo=all: 汇总全部目录（含各并行实验）----
    if algo == "all":
        files, dirs = [], []
        for label, d in _all_model_dirs(by_quality=True):
            got = _list_models(d, label)
            if not got:
                continue
            # 附加该来源的"质量"信息（局数 / 最佳评估分 / 是否运行中）
            parent = os.path.dirname(d)
            st = _read_json(os.path.join(parent, "status.json"), default=None)
            if st is None:
                st = _read_json(os.path.join(parent, "v3", "status.json"),
                                default={}) or {}
            quality = {
                "episode": (st or {}).get("episode"),
                "best_eval": (st or {}).get("best_eval_score"),
                "running": bool((st or {}).get("running")),
            }
            for f in got:
                f.update(quality)
            dirs.append({"source": label, "dir": d, "count": len(got),
                         **quality})
            files.extend(got)
        # 默认按"来源质量"排序（运行中 + 局数多 在前）, 便于直接选到最强模型
        files.sort(key=lambda f: (f.get("running") or False,
                                  f.get("episode") or 0), reverse=True)
        return jsonify({"algo": "all", "files": files, "dirs": dirs,
                        "versions": []})

    # 单算法视图只返回该算法自己目录下的模型（精确不混淆）；
    # 「全部模型」(algo=all) 才是聚合视图, 前端默认用它。
    models_dir = os.path.join(_data_dir(algo), "models")
    files = _list_models(models_dir, algo)
    versions = _read_json(os.path.join(models_dir, "versions.json"),
                          default=[]) or []
    return jsonify({"algo": algo, "files": files, "versions": versions,
                    "dir": models_dir,
                    "dirs": [{"source": algo, "dir": models_dir,
                              "count": len(files)}]})


@app.route("/api/experiments")
def api_experiments():
    """列出所有实验目录（含已停止的）, 供前端选择与启动新实验。"""
    import glob as _glob
    root = os.path.dirname(OUT_DIR)
    rows = []
    seen = set()
    for pat in (os.path.join(root, "exp_*"),
                os.path.join(OUT_DIR, "v3"),
                os.path.join(root, "exp_*", "v3")):
        for d in _glob.glob(pat):
            if not os.path.isdir(d) or d in seen:
                continue
            st_path = os.path.join(d, "status.json")
            if not os.path.exists(st_path):
                continue
            seen.add(d)
            st = _read_json(st_path, default={}) or {}
            cfg = st.get("config") or {}
            label = os.path.basename(d)
            if label == "v3":
                label = os.path.basename(os.path.dirname(d))
            rows.append({
                "label": label, "dir": d,
                "episode": st.get("episode") or 0,
                "running": st.get("running"),
                "workers": st.get("workers"),
                "patterns": cfg.get("pattern_set"), "alpha": cfg.get("alpha"),
                "lam": cfg.get("lam"),
                "best_eval": st.get("best_eval_score"),
                "recent_avg": (lambda s: round(sum(s[-200:]) / len(s[-200:]), 1)
                               if s else 0)(st.get("recent_scores") or []),
                "has_model": os.path.isdir(os.path.join(d, "models")) and bool(
                    [f for f in os.listdir(os.path.join(d, "models"))
                     if f.endswith((".npz", ".pt"))]),
            })
    rows.sort(key=lambda r: -(r["episode"] or 0))
    return jsonify({"experiments": rows, "base": root})


@app.route("/api/train/start_from_model", methods=["POST"])
def api_start_from_model():
    """用指定模型启动一个新实验（模型热启动 / 继续训练）。

    body:
        model     模型文件名（如 latest.npz）
        src       模型来源标签（v1/v2/v3/exp_hp...）；留空则全局搜索
        name      新实验名（如 exp_m）；留空自动生成 exp_<序号>
        episodes  目标局数
        workers   并行 worker 数（>0 用并行训练器）
        patterns  图案集（8/12/5）—— 须与模型匹配
        alpha     学习率
        lam       TD(λ)
    说明:
        v3 模型可在【相同图案集】间自由热启动;
        不同图案集（8<->12）结构不同, 会自动拒绝并提示。
    """
    data = request.get_json(silent=True) or {}
    model_name = data.get("model", "")
    src_label = data.get("src", "")
    if not model_name:
        return jsonify({"ok": False, "msg": "缺少 model 参数"}), 400

    # 定位模型
    path = None
    if src_label:
        for lb, d in _all_model_dirs():
            if lb == src_label:
                cand = os.path.join(d, model_name)
                if os.path.isfile(cand):
                    path = cand
                    break
    if path is None:
        path, src_label = _resolve_model("v3", model_name)
    if path is None:
        return jsonify({"ok": False, "msg": f"未找到模型 {model_name}"}), 404

    # 读取模型的图案集, 决定新实验配置
    try:
        import numpy as _np
        if path.endswith(".npz"):
            with _np.load(path) as z:
                pats = z["patterns"]
                model_n_pats = int(pats.shape[0])
                model_tuple_len = int(pats.shape[1])
        else:
            return jsonify({"ok": False,
                            "msg": "仅支持从 v3 (.npz) 模型继续实验"}), 400
    except Exception as exc:
        return jsonify({"ok": False, "msg": f"模型读取失败: {exc}"}), 500

    # 图案集校验（结构必须一致才能热启动）
    pat_map = {8: "8", 12: "12"}
    inferred = pat_map.get(model_n_pats)
    if model_tuple_len == 5:
        inferred = "5"
    req_patterns = str(data.get("patterns") or inferred or "8")
    if req_patterns != inferred:
        return jsonify({"ok": False,
                        "msg": f"图案集不匹配: 模型是 {inferred} 图案"
                               f"({model_n_pats}个{model_tuple_len}元组), "
                               f"请求 {req_patterns}。请选择相同图案集。"}), 400

    # 训练参数（需先于目录规划确定）
    workers = int(data.get("workers") or 0)
    episodes = int(data.get("episodes") or 3_000_000)
    alpha = float(data.get("alpha") or 0.32)
    lam = float(data.get("lam") or 0.0)

    # 实验名
    import glob as _glob
    root = os.path.dirname(OUT_DIR)
    name = (data.get("name") or "").strip()
    if not name:
        i = 0
        while True:
            cand = f"exp_{chr(ord('m') + i)}"
            if not os.path.exists(os.path.join(root, cand)):
                name = cand
                break
            i += 1
    if "/" in name or " " in name:
        return jsonify({"ok": False, "msg": "实验名不能含空格或斜杠"}), 400
    exp_dir = os.path.join(root, name)
    # main.py 会在 out-dir 下再建一层 v3/（保持与既有目录结构一致）,
    # 并行训练器 train_nt_parallel.py 直接用 out-dir。
    sub = "" if workers > 0 else "v3"
    models_dir = os.path.join(exp_dir, sub, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 复制模型作为新实验的起点
    import shutil
    base = os.path.basename(path)
    shutil.copy2(path, os.path.join(models_dir, base))
    meta_src = path + ".meta.json"
    if os.path.exists(meta_src):
        try:
            with open(meta_src, encoding="utf-8") as f:
                meta = json.load(f)
            meta["copied_from"] = path
            meta["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(models_dir, base + ".meta.json"),
                      "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
        except Exception:
            pass

    # 启动训练进程
    if workers > 0:
        cmd = [sys.executable, "-u",
               os.path.join(PROJ_DIR, "train_nt_parallel.py"),
               "--episodes", str(episodes), "--workers", str(workers),
               "--patterns", req_patterns, "--alpha", str(alpha),
               "--lam", str(lam), "--resume", "--out-dir", exp_dir,
               "--eval-every", "5000", "--eval-games", "100",
               "--ckpt-every", "3000"]
    else:
        cmd = [sys.executable, "-u", os.path.join(PROJ_DIR, "main.py"),
               "--train", "--algo", "v3", "--episodes", str(episodes),
               "--nt-patterns", req_patterns, "--nt-alpha", str(alpha),
               "--nt-lambda", str(lam), "--nt-vinit", "0",
               "--resume", "--out-dir", exp_dir,
               "--eval-every", "5000", "--eval-games", "100"]

    log_path = os.path.join(exp_dir, "web_train.log")
    try:
        logf = open(log_path, "ab", buffering=0)
        proc = subprocess.Popen(cmd, cwd=PROJ_DIR, stdout=logf,
                                stderr=subprocess.STDOUT)
    except Exception as exc:
        return jsonify({"ok": False, "msg": f"启动失败: {exc}"}), 500

    logger.info("从模型启动实验 %s: pid=%s model=%s", name, proc.pid, path)
    return jsonify({"ok": True, "experiment": name, "dir": exp_dir,
                    "pid": proc.pid, "from_model": path,
                    "patterns": req_patterns, "workers": workers,
                    "cmd": " ".join(cmd)})


@app.route("/api/download")
def api_download():
    """下载模型文件（仅允许访问已知模型目录内的文件, 防路径穿越）。

    参数: src=<来源标签 v1/v2/v3/exp_a...>  name=<文件名>
    """
    from flask import send_file
    src_label = request.args.get("src", "")
    name = request.args.get("name", "")
    if not name:
        return jsonify({"ok": False, "msg": "缺少 name 参数"}), 400
    if "/" in name or "\\" in name or name.startswith("."):
        return jsonify({"ok": False, "msg": "非法文件名"}), 400
    if not name.endswith((".pt", ".npz")):
        return jsonify({"ok": False, "msg": "仅支持下载 .pt / .npz 模型"}), 400

    path = None
    if src_label:
        target_dir = next((d for lb, d in _all_model_dirs()
                           if lb == src_label), None)
        if target_dir:
            cand = os.path.join(target_dir, name)
            if os.path.isfile(cand):
                path = cand
    if path is None:                     # 未指定来源或未命中 -> 全局搜索
        path, src_label = _resolve_model("v3", name)
    if path is None:
        return jsonify({"ok": False, "msg": f"文件不存在: {name}"}), 404

    logger.info("下载模型: %s/%s", src_label, name)
    return send_file(path, as_attachment=True, download_name=name)


# ==================== 训练控制 ====================

@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    """启动训练子进程（网页一键训练）。body 可含 algo=v1|v2。"""
    data = request.get_json(silent=True) or {}
    algo = data.get("algo", "v1")
    if algo not in ("v1", "v2", "v3"):
        return jsonify({"ok": False, "msg": f"未知算法: {algo}"}), 400
    if _train_running(algo):
        return jsonify({"ok": False, "msg": f"{algo} 训练已在运行中",
                        "pid": _train_proc[algo].pid}), 409

    cmd = [sys.executable, "-u", os.path.join(PROJ_DIR, "main.py"),
           "--train", "--out-dir", OUT_DIR]
    if algo == "v2":
        # v2 的 Conv+C51 反向较重, 2 线程在此类多核服务器上实测最优
        cmd += ["--algo", "v2", "--torch-threads",
                str(int(data.get("torch_threads", 2)))]
        if data.get("n_step"):
            cmd += ["--n-step", str(int(data["n_step"]))]
        if data.get("train_every"):
            cmd += ["--train-every", str(int(data["train_every"]))]
    elif algo == "v3":
        # v3 (N-Tuple) 纯 numpy 查表, 不需要 torch 线程参数
        cmd += ["--algo", "v3"]
        for key, flag in (("nt_alpha", "--nt-alpha"),
                          ("nt_lambda", "--nt-lambda"),
                          ("nt_vinit", "--nt-vinit"),
                          ("nt_len", "--nt-len"),
                          ("nt_patterns", "--nt-patterns"),
                          ("search_depth", "--search-depth")):
            if data.get(key) is not None:
                cmd += [flag, str(data[key])]
    if data.get("episodes"):
        cmd += ["--episodes", str(int(data["episodes"]))]
    if data.get("parallel"):
        cmd += ["--parallel", str(int(data["parallel"]))]
    if data.get("no_mcts"):
        cmd += ["--no-mcts"]
    if data.get("mcts_interval"):
        cmd += ["--mcts-interval", str(int(data["mcts_interval"]))]
    if data.get("eps_decay"):
        cmd += ["--eps-decay", str(int(data["eps_decay"]))]
    if data.get("resume"):
        cmd += ["--resume"]

    log_dir = os.path.join(_data_dir(algo), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "web_train.log")
    try:
        logf = open(log_path, "ab", buffering=0)
        _train_proc[algo] = subprocess.Popen(
            cmd, cwd=PROJ_DIR, stdout=logf, stderr=subprocess.STDOUT)
    except Exception as exc:
        return jsonify({"ok": False, "msg": f"启动失败: {exc}"}), 500

    logger.info("[%s] 训练已启动 pid=%s: %s", algo,
                _train_proc[algo].pid, " ".join(cmd))
    return jsonify({"ok": True, "algo": algo, "pid": _train_proc[algo].pid,
                    "cmd": " ".join(cmd)})


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    """停止训练: 发送 SIGINT, 让训练器走正常保存流程（保存 checkpoint）。"""
    data = request.get_json(silent=True) or {}
    algo = data.get("algo", "v1")
    if not _train_running(algo):
        return jsonify({"ok": False, "msg": f"{algo} 当前没有运行中的训练"}), 409
    proc = _train_proc[algo]
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=60)            # 等待正常收尾（保存 checkpoint）
        logger.info("[%s] 训练已停止, 退出码 %s", algo, proc.returncode)
        return jsonify({"ok": True, "algo": algo, "exit_code": proc.returncode})
    except subprocess.TimeoutExpired:
        proc.kill()                      # 超时强杀
        return jsonify({"ok": True, "algo": algo, "msg": "超时强杀"})
    except Exception as exc:
        return jsonify({"ok": False, "msg": str(exc)}), 500


# ==================== 模型评测 ====================

@app.route("/api/eval", methods=["POST"])
def api_eval():
    """快速评测指定模型（同步执行, 局数建议 <= 20）。"""
    from model import load_torch        # 延迟导入: 需先完成 sys.path 注入

    data = request.get_json(silent=True) or {}
    algo = data.get("algo", "v1")
    name = data.get("model", "best.pt")
    games = max(1, min(int(data.get("games", 10)), 200))
    search_depth = max(0, min(int(data.get("search_depth", 0)), 4))
    path, src_label = _resolve_model(algo, name)
    if path is None:
        # 列出可用模型, 方便前端提示
        avail = []
        for label, d in _all_model_dirs():
            avail.extend(f"{label}/{f}" for f in os.listdir(d)
                         if f.endswith((".npz", ".pt")))
        return jsonify({"ok": False,
                        "msg": f"模型不存在: {name}",
                        "available": avail[:20]}), 404
    # 模型来自实验目录时, 按 v3 算法路径评测
    if src_label not in ("v1", "v2", "v3"):
        algo = "v3"

    # ---- v3 (N-Tuple) 评估路径 ----
    if algo == "v3":
        try:
            import sys as _sys
            _proj = os.path.dirname(PROJ_DIR)
            if _proj not in _sys.path:
                _sys.path.insert(0, _proj)
            from train_nt import NTupleConfig, NTupleTrainer
            # 直接加载已解析出的模型路径（可能在实验目录, 不在 v3 主目录）
            cfg = NTupleConfig(out_dir=os.path.dirname(os.path.dirname(path)),
                               eval_games=games, search_depth=search_depth)
            trainer = NTupleTrainer(cfg)
            try:
                import numpy as _np
                from ntuple import (PATTERN_SETS, NTupleNetwork,
                                    greedy_action, new_board, spawn,
                                    decode_board)
                ck = _np.load(path)
                model_pats = [tuple(int(x) for x in r) for r in ck["patterns"]]
                net = NTupleNetwork(model_pats, v_init=0.0)
                net.lut = ck["lut"].astype(net.lut.dtype)
                trainer.net = net            # 用模型自带图案集, 不依赖目录配置
                st = trainer.evaluate(games,
                                      search_depth=search_depth)
                logger.info("[v3] 评测 %s: avg=%.0f max=%d",
                            name, st["avg"], st["max"])
                return jsonify({"ok": True, "algo": "v3", "model": name,
                                "search_depth": search_depth,
                                "games": st["games"],
                                "avg": round(st["avg"], 1), "max": st["max"],
                                "median": round(st["median"], 1),
                                "avg_steps": round(st["avg_steps"], 1),
                                "max_tile": 0,
                                "ms512": round(st["ms512"] * 100, 1),
                                "ms1024": round(st["ms1024"] * 100, 1),
                                "ms2048": round(st["ms2048"] * 100, 1),
                                "ms4096": round(st["ms4096"] * 100, 1),
                                "scores": []})
            finally:
                trainer.close()
        except Exception as exc:
            logger.exception("v3 评测失败")
            return jsonify({"ok": False, "msg": str(exc)}), 500

    # ---- v2 (Rainbow-Lite) 评估路径 ----
    if algo == "v2":
        try:
            from train_v2 import RainbowConfig, RainbowTrainer
            cfg = RainbowConfig(out_dir=_data_dir("v2"))
            trainer = RainbowTrainer(cfg)
            try:
                ckpt = load_torch(path, map_location="cpu")
                trainer.agent.load_both_state_dict(ckpt["policy_sd"])
                st = trainer.evaluate(games)
                logger.info("[v2] 评测 %s: avg=%.0f max=%d", name, st["avg"], st["max"])
                return jsonify({"ok": True, "algo": "v2", "model": name,
                                "games": st["games"], "avg": round(st["avg"], 1),
                                "max": st["max"], "median": round(st["median"], 1),
                                "avg_steps": round(st["avg_steps"], 1),
                                "max_tile": 0,
                                "ms512": round(st["ms512"] * 100, 1),
                                "ms1024": round(st["ms1024"] * 100, 1),
                                "ms2048": round(st["ms2048"] * 100, 1),
                                "scores": []})
            finally:
                trainer.close()
        except Exception as exc:
            logger.exception("v2 评测失败")
            return jsonify({"ok": False, "msg": str(exc)}), 500

    try:
        from selfplay import play_greedy_games
        ckpt = load_torch(path, map_location="cpu")
        scores, steps, tiles = play_greedy_games(ckpt["policy_sd"], n_games=games)
        result = {
            "ok": True, "model": name, "games": games,
            "avg": round(sum(scores) / len(scores), 1),
            "max": max(scores),
            "median": sorted(scores)[len(scores) // 2],
            "avg_steps": round(sum(steps) / len(steps), 1),
            "max_tile": max(tiles),
            "scores": scores,
        }
        logger.info("评测完成: %s avg=%.1f max=%d", name, result["avg"], result["max"])
        return jsonify(result)
    except Exception as exc:
        logger.exception("评测失败")
        return jsonify({"ok": False, "msg": str(exc)}), 500


# ==================== AI 演示（SSE 流式） ====================

@app.route("/api/play")
def api_play():
    """SSE: AI 玩一局, 每步推送棋盘 JSON。

    参数: algo=v1|v2 | model=best.pt | delay=0.25 | mcts=0|1 | sims=50
    事件: {"type":"start"|"step"|"end"|"error", ...}
    """
    algo = request.args.get("algo", "v1")
    model_name = request.args.get("model", "best.pt")
    delay = max(0.0, min(float(request.args.get("delay", 0.25)), 2.0))
    use_mcts = request.args.get("mcts", "0") in ("1", "true") and algo == "v1"
    sims = max(4, min(int(request.args.get("sims", 50)), 200))
    # v3: sims 参数复用为 expectimax 搜索深度（0/1 = 贪心）
    depth = max(0, min(int(request.args.get("sims", 0)), 6)) if algo == "v3" else 0
    # 支持 src 参数精确指定来源（同名模型跨实验时避免歧义）
    req_src = request.args.get("src", "")
    model_path = None
    if req_src:
        for lb, d in _all_model_dirs():
            if lb == req_src:
                cand = os.path.join(d, model_name)
                if os.path.isfile(cand):
                    model_path = cand
                    break
    if model_path is None:
        model_path, _src = _resolve_model(algo, model_name)
    if model_path is None:
        model_path = os.path.join(_data_dir(algo), "models", model_name)

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        try:
            from env import ACTION_NAMES, Game2048
            from model import load_torch

            if not os.path.exists(model_path):
                yield sse({"type": "error",
                           "msg": f"模型不存在: {model_name}, 请先训练"})
                return

            mcts = None
            searcher = None
            nt_net = None

            # ---- v3: N-Tuple + 可选 expectimax ----
            if algo == "v3":
                from ntuple import PATTERNS_5, PATTERNS_6, NTupleNetwork
                from train_nt import NTupleConfig
                _cfg = NTupleConfig()
                nt_net = NTupleNetwork(
                    PATTERNS_6 if _cfg.tuple_len == 6 else PATTERNS_5)
                nt_net.load(model_path)
                if depth > 1:
                    from expectimax import ExpectimaxSearcher
                    searcher = ExpectimaxSearcher(nt_net, depth=depth)

            agent = None
            if algo != "v3":                      # v3 不需要 torch 智能体
                ckpt = load_torch(model_path, map_location="cpu")
                if algo == "v2":                  # Rainbow-Lite
                    from model_v2 import RainbowAgent
                    agent = RainbowAgent()
                    agent.load_both_state_dict(ckpt["policy_sd"])
                    agent.policy_net.eval()
                else:                             # 经典 DQN
                    from model import DQNAgent
                    agent = DQNAgent()
                    agent.load_both_state_dict(ckpt["policy_sd"])
                    agent.policy_net.eval()
                    if use_mcts:
                        from mcts import MCTSAgent
                        mcts = MCTSAgent(n_simulations=sims)

            # ---- v3 走独立的高性能路径（编码棋盘 + 查表） ----
            if algo == "v3":
                import numpy as _np
                from ntuple import (decode_board as _dec,
                                    greedy_action as _ga, move_all as _ma,
                                    new_board as _nb, spawn as _sp)

                rng = _np.random.default_rng()
                board = _nb(rng)
                score, step = 0, 0
                yield sse({"type": "start", "model": model_name, "algo": algo,
                           "src": req_src or os.path.basename(
                               os.path.dirname(os.path.dirname(model_path))),
                           "board": _dec(board).tolist(), "score": 0,
                           "mcts": bool(searcher),
                           "sims": depth if searcher else 0})
                while True:
                    if searcher is not None:
                        a = searcher.best_action(board)
                        if a is None:
                            break
                        _bs, _ss, _ms = _ma(board)
                        nb, sc = _bs[a], int(_ss[a])
                    else:
                        res = _ga(nt_net, board)
                        if res is None:
                            break
                        a, nb, sc, _v = res
                    score += sc
                    board = _sp(nb.copy(), rng)
                    step += 1
                    yield sse({"type": "step", "board": _dec(board).tolist(),
                               "score": score, "action": int(a),
                               "action_name": ACTION_NAMES[int(a)],
                               "max_tile": int(_dec(board).max()),
                               "step": step, "done": False})
                    if delay > 0:
                        time.sleep(delay)
                yield sse({"type": "end", "score": score,
                           "max_tile": int(_dec(board).max()), "steps": step})
                return

            env = Game2048()
            state = env.reset()
            yield sse({"type": "start", "model": model_name, "algo": algo,
                       "src": req_src, "board": env.grid, "score": 0,
                       "mcts": bool(mcts), "sims": sims if mcts else 0})

            def agent_input():
                """v2 吃原始棋盘, v1 吃归一化状态。"""
                return env.grid if algo == "v2" else state

            done, step = False, 0
            while not done:
                valid = env.get_valid_actions()
                if not valid:
                    break
                action = None
                if mcts is not None:
                    action = mcts.search(env.grid, agent)
                if action is None:
                    action = agent.select_action(agent_input(), 0.0, valid)
                state, _, done, info = env.step(action)
                step += 1
                yield sse({"type": "step", "board": env.grid,
                           "score": info["score"], "action": action,
                           "action_name": ACTION_NAMES[action],
                           "max_tile": info["max_tile"], "step": step,
                           "done": done})
                if delay > 0:
                    time.sleep(delay)

            if mcts is not None:
                mcts.close()
            yield sse({"type": "end", "score": info["score"],
                       "max_tile": info["max_tile"], "steps": step})
        except GeneratorExit:
            return
        except Exception as exc:
            logger.exception("演示异常")
            yield sse({"type": "error", "msg": str(exc)})

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


# ==================== 入口 ====================

def main() -> None:
    parser = argparse.ArgumentParser(description="2048 AI Web 控制台")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info("2048 AI 控制台启动: http://%s:%d  (out_dir=%s)",
                args.host, args.port, OUT_DIR)
    # threaded=True 保证训练/演示/轮询并发不互相阻塞
    app.run(host=args.host, port=args.port, debug=args.debug,
            threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
