"""
xT 引擎 — Karun Singh (2018) Expected Threat 实现
==================================================

参考：
  - Singh, K. (2018). "Introducing Expected Threat (xT)."
  - 参考报告 2.2.2：xT 模型 9，将球场切 16×12 网格，
    每格基于"最终进球概率"赋值，传球/盘带的 xT 增量 =
    目标格 xT - 起始格 xT。

数学模型：
  对每格 (x,y) 定义两个量：
    s(x,y) = 在该格射门的概率
    g(x,y) = 在该格射门的进球率（射门后 xG 期望）
    m_t(x,y) = 在该格运球转移到 (x',y') 的概率
    m_p(x,y, x',y') = 在该格传球到 (x',y') 的概率
  
  迭代求解：
    xT(x,y) = s·g + (1 - s)·Σ m_t(x',y')·xT(x',y')
                  + (1 - s)·Σ m_p(x',y')·xT(x',y')
  
  收敛后即得每格 xT 值。每次成功传球/盘带的 xT 增量 = ΔxT。

球队 xt_per_90 = (∑ 该队所有有效推进的 xT 增量) / 90 分钟

输出：
  data/outputs/xt_grid.json     ← 16×12 价值矩阵
  data/raw/teams_xt.json         ← 各国家队 xt_per_90
  自动写回 data/raw/teams.json["teams"][team]["xt_per_90"]

CLI：
  python3 xt_engine.py train      # 用 StatsBomb 数据学习 xT 矩阵
  python3 xt_engine.py compute    # 算各队 xt_per_90
  python3 xt_engine.py viz        # 文本可视化矩阵
"""
from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_RAW = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_OUTPUTS = Path(__file__).parent.parent.parent / "data" / "outputs"
DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)


# ============ 球场参数（StatsBomb 坐标）============
PITCH_LENGTH = 120.0   # x
PITCH_WIDTH = 80.0     # y
N_X = 16
N_Y = 12
DX = PITCH_LENGTH / N_X   # 每格 x 跨度
DY = PITCH_WIDTH / N_Y    # 每格 y 跨度


def cell_of(x: float, y: float) -> Tuple[int, int]:
    """坐标 → 网格 (i, j)；返回 (-1,-1) 表示越界"""
    if x is None or y is None:
        return (-1, -1)
    i = min(int(x / DX), N_X - 1)
    j = min(int(y / DY), N_Y - 1)
    if i < 0 or j < 0:
        return (-1, -1)
    return (i, j)


# ============ 1. 学习状态分布 ============
def _load_all_events():
    """加载所有 StatsBomb 事件"""
    import pandas as pd
    parts = []
    for p in DATA_RAW.glob("statsbomb_events_*.parquet"):
        df = pd.read_parquet(p)
        df["_source"] = p.name
        parts.append(df)
    if not parts:
        raise RuntimeError("没有 StatsBomb 数据，先跑 statsbomb_loader.py fetch")
    return pd.concat(parts, ignore_index=True)


def _parse_loc(loc):
    """StatsBomb location 是 [x, y] 列表，可能 None"""
    if loc is None or (isinstance(loc, float) and np.isnan(loc)):
        return (None, None)
    if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2:
        return (float(loc[0]), float(loc[1]))
    return (None, None)


def learn_state_matrices(events_df) -> Dict[str, np.ndarray]:
    """
    学习 4 个网格量：
      shot_freq[i,j]    = P(在格内事件 = 射门 | 在格内有事件)
      shot_xg[i,j]      = E[xG | 射门发生于格内]
      pass_total[i,j]   = 在格内的传球总数（用于 m_p 归一）
      transition_p[i,j, i2,j2] = 从 (i,j) 传球/盘带到 (i2,j2) 的频率
    """
    print("⏳ 学习状态矩阵...")

    actions_count = np.zeros((N_X, N_Y), dtype=np.int64)  # 该格所有事件数
    shot_count = np.zeros((N_X, N_Y), dtype=np.int64)
    shot_xg_sum = np.zeros((N_X, N_Y), dtype=np.float64)

    # 转移频次：(i,j) -> (i2,j2)
    move_count = np.zeros((N_X, N_Y, N_X, N_Y), dtype=np.int64)

    n_events = len(events_df)
    n_processed = 0

    # 只看有 location 的事件
    relevant_types = {"Pass", "Carry", "Shot"}
    for _, ev in events_df.iterrows():
        n_processed += 1
        if n_processed % 100000 == 0:
            print(f"  processed {n_processed:,}/{n_events:,}")

        ev_type = ev.get("type")
        if ev_type not in relevant_types:
            continue

        x, y = _parse_loc(ev.get("location"))
        i, j = cell_of(x, y)
        if i < 0:
            continue

        actions_count[i, j] += 1

        if ev_type == "Shot":
            shot_count[i, j] += 1
            xg = ev.get("shot_statsbomb_xg")
            if xg and not np.isnan(xg):
                shot_xg_sum[i, j] += float(xg)
        elif ev_type == "Pass":
            x2, y2 = _parse_loc(ev.get("pass_end_location"))
            i2, j2 = cell_of(x2, y2)
            if i2 >= 0:
                move_count[i, j, i2, j2] += 1
        elif ev_type == "Carry":
            x2, y2 = _parse_loc(ev.get("carry_end_location"))
            i2, j2 = cell_of(x2, y2)
            if i2 >= 0:
                move_count[i, j, i2, j2] += 1

    # 计算频率
    shot_freq = np.zeros((N_X, N_Y))
    shot_xg = np.zeros((N_X, N_Y))
    for i in range(N_X):
        for j in range(N_Y):
            if actions_count[i, j] > 0:
                shot_freq[i, j] = shot_count[i, j] / actions_count[i, j]
            if shot_count[i, j] > 0:
                shot_xg[i, j] = shot_xg_sum[i, j] / shot_count[i, j]

    # 转移概率：以"格内的所有传球+盘带"为分母
    move_prob = np.zeros_like(move_count, dtype=np.float64)
    for i in range(N_X):
        for j in range(N_Y):
            total_moves = move_count[i, j].sum()
            if total_moves > 0:
                move_prob[i, j] = move_count[i, j] / total_moves

    return {
        "shot_freq": shot_freq,
        "shot_xg": shot_xg,
        "move_prob": move_prob,  # 形状 (N_X, N_Y, N_X, N_Y)
        "actions_count": actions_count,
    }


# ============ 2. 迭代求解 xT 矩阵 ============
def solve_xt(matrices: Dict[str, np.ndarray],
              n_iter: int = 5,
              eps: float = 1e-5) -> np.ndarray:
    """
    Karun Singh 迭代公式：
      xT_{t+1}(s) = m_s(s)·xG_s(s) + (1 - m_s(s))·Σ_{s'} P(s→s')·xT_t(s')
    
    其中 m_s = shot_freq, xG_s = shot_xg, P = move_prob
    """
    shot_freq = matrices["shot_freq"]
    shot_xg = matrices["shot_xg"]
    move_prob = matrices["move_prob"]

    xt = np.zeros((N_X, N_Y))
    for it in range(n_iter):
        # 射门贡献项
        shot_term = shot_freq * shot_xg
        # 转移贡献：(N_X*N_Y, N_X*N_Y) @ (N_X*N_Y,) → (N_X*N_Y,)
        # 把 4D 展平成 2D
        flat_xt = xt.flatten()
        flat_move = move_prob.reshape(N_X * N_Y, N_X * N_Y)
        move_term_flat = flat_move @ flat_xt
        move_term = move_term_flat.reshape(N_X, N_Y)

        new_xt = shot_term + (1 - shot_freq) * move_term
        diff = np.abs(new_xt - xt).max()
        xt = new_xt
        print(f"  iter {it+1}: max_diff={diff:.5f}, max_xt={xt.max():.4f}")
        if diff < eps:
            break

    return xt


# ============ 3. 球队 xt_per_90 ============
def compute_team_xt(events_df, xt_grid: np.ndarray) -> Dict[str, dict]:
    """
    对每队每场，累加成功传球+盘带的 ΔxT；除以场数 / 90 分钟。
    """
    print("⏳ 计算球队 xt_per_90...")
    team_stats = defaultdict(lambda: {"xt_total": 0.0, "match_ids": set()})

    n_total = len(events_df)
    for n_idx, (_, ev) in enumerate(events_df.iterrows(), 1):
        if n_idx % 100000 == 0:
            print(f"  {n_idx:,}/{n_total:,}")

        ev_type = ev.get("type")
        if ev_type not in ("Pass", "Carry"):
            continue

        x, y = _parse_loc(ev.get("location"))
        i, j = cell_of(x, y)
        if i < 0:
            continue

        if ev_type == "Pass":
            x2, y2 = _parse_loc(ev.get("pass_end_location"))
        else:
            x2, y2 = _parse_loc(ev.get("carry_end_location"))
        i2, j2 = cell_of(x2, y2)
        if i2 < 0:
            continue

        delta_xt = xt_grid[i2, j2] - xt_grid[i, j]
        # 只累加正向 ΔxT（向前推进的价值）
        if delta_xt > 0:
            team = ev.get("team")
            if team:
                team_stats[team]["xt_total"] += delta_xt
                team_stats[team]["match_ids"].add(ev.get("match_id"))

    out = {}
    for team, s in team_stats.items():
        n_matches = len(s["match_ids"])
        if n_matches >= 2:  # 至少 2 场才统计
            xt_per_match = s["xt_total"] / n_matches
            out[team] = {
                "xt_per_90": round(xt_per_match, 3),  # 一场≈90分钟
                "matches": n_matches,
                "xt_total": round(s["xt_total"], 2),
            }
    return out


# ============ 4. 写回 teams.json ============
TEAM_NAME_MAP = {
    # StatsBomb name → teams.json name
    "Czech Republic": "Czechia",
    "South Korea": "Korea Republic",
    "Korea": "Korea Republic",
    "Türkiye": "Turkiye",
    "Turkey": "Turkiye",
    "USA": "United States",
}


def write_to_teams_json(team_xt: Dict[str, dict]) -> int:
    """合并到 data/raw/teams.json 的 teams 字典"""
    teams_path = DATA_RAW / "teams.json"
    data = json.load(open(teams_path))
    teams = data["teams"]

    n_updated = 0
    for sb_name, info in team_xt.items():
        # 名字标准化
        canonical = TEAM_NAME_MAP.get(sb_name, sb_name)
        if canonical in teams:
            teams[canonical]["xt_per_90"] = info["xt_per_90"]
            teams[canonical]["xt_n_matches"] = info["matches"]
            n_updated += 1

    # 加来源声明
    sources = data.get("_sources", [])
    if "StatsBomb" not in sources:
        sources.append("StatsBomb")
        data["_sources"] = sources

    teams_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return n_updated


# ============ 主流程 ============
def train_pipeline() -> np.ndarray:
    """完整训练管道：加载事件 → 学矩阵 → 解 xT → 持久化"""
    df = _load_all_events()
    print(f"✓ 加载 {len(df):,} 事件 from {df['_source'].nunique()} 源")

    matrices = learn_state_matrices(df)
    print(f"✓ 状态矩阵学习完成")

    xt_grid = solve_xt(matrices, n_iter=5)
    print(f"✓ xT 收敛, 最大值 = {xt_grid.max():.4f}")

    # 持久化
    out = DATA_OUTPUTS / "xt_grid.json"
    out.write_text(json.dumps({
        "shape": [N_X, N_Y],
        "pitch_length": PITCH_LENGTH,
        "pitch_width": PITCH_WIDTH,
        "max_xt": float(xt_grid.max()),
        "min_xt": float(xt_grid.min()),
        "grid": xt_grid.tolist(),
    }, ensure_ascii=False, indent=2))
    print(f"✓ 矩阵写入 {out.relative_to(DATA_OUTPUTS.parent.parent)}")

    return xt_grid


def compute_pipeline(xt_grid: Optional[np.ndarray] = None):
    """计算各队 xt_per_90 并写回 teams.json"""
    if xt_grid is None:
        # 从磁盘读
        path = DATA_OUTPUTS / "xt_grid.json"
        if not path.exists():
            print("✗ xt_grid.json 不存在，请先 train")
            return
        d = json.load(open(path))
        xt_grid = np.array(d["grid"])

    df = _load_all_events()
    team_xt = compute_team_xt(df, xt_grid)

    # 写出原始
    out = DATA_RAW / "teams_xt.json"
    out.write_text(json.dumps(team_xt, ensure_ascii=False, indent=2))

    # 合并到 teams.json
    n = write_to_teams_json(team_xt)
    print(f"✓ 写入 {n} 队的 xt_per_90 到 teams.json")

    # 打印 Top 12
    sorted_t = sorted(team_xt.items(), key=lambda x: -x[1]["xt_per_90"])[:12]
    print("\n📊 Top 12 by xt_per_90:")
    for name, info in sorted_t:
        print(f"  {name:<22} {info['xt_per_90']:.3f}  ({info['matches']} 场)")


def viz_grid(xt_grid: Optional[np.ndarray] = None):
    """文本可视化 xT 网格"""
    if xt_grid is None:
        path = DATA_OUTPUTS / "xt_grid.json"
        if not path.exists():
            print("✗ 先 train")
            return
        xt_grid = np.array(json.load(open(path))["grid"])

    print(f"\n  xT Grid ({N_X} × {N_Y}, 值 × 1000)")
    print("  防守方  ←──────────────  →  进攻方")
    print("  " + "─" * (N_X * 4))
    # 每列是 x（沿着进攻方向），每行是 y
    for j in range(N_Y - 1, -1, -1):
        row = []
        for i in range(N_X):
            v = xt_grid[i, j] * 1000
            row.append(f"{v:>3.0f}")
        print(f"  {' '.join(row)}")
    print()


# ============ CLI ============
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "train":
        xt_grid = train_pipeline()
        viz_grid(xt_grid)
    elif cmd == "compute":
        compute_pipeline()
    elif cmd == "viz":
        viz_grid()
    elif cmd == "all":
        xt_grid = train_pipeline()
        viz_grid(xt_grid)
        compute_pipeline(xt_grid)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
