"""
pi-ratings — Constantinou & Fenton (2013)
==========================================

Opponent-adjusted Elo 变体。每队维护两套评分：
  R_H[team]: 主场评分（home rating）
  R_A[team]: 客场评分（away rating）

更新公式（每场比赛后）：
  expected_diff = (R_H[home] - R_A[away]) / c    # c=3 为缩放
  expected_goals_diff = b × tanh(expected_diff)  # b=10
  
  实际净胜 g_actual = home_score - away_score
  误差 e = g_actual - expected_goals_diff
  
  weighted_e_home = e × λ × γ
  weighted_e_away = e × λ × (1-γ)
  
  R_H[home] += weighted_e_home
  R_A[home] += weighted_e_home × ψ          # 主场表现部分泛化到客场（ψ=0.3-0.5）
  R_A[away] -= weighted_e_away
  R_H[away] -= weighted_e_away × ψ

参数（参考报告 / 论文推荐）：
  λ = 0.054   学习率
  γ = 0.79    主场偏置（主队误差权重大）
  ψ = 0.40    主客场泛化系数
  c = 3       缩放
  b = 10      tanh 幅度

输出 7 维特征：
  R_H_home, R_A_home, R_H_away, R_A_away,
  diff_H_A, diff_H_H, recent_form_diff
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
from collections import defaultdict, deque


DATA_HIST = Path(__file__).parent.parent.parent / "data" / "historical"
DATA_OUTPUTS = Path(__file__).parent.parent.parent / "data" / "outputs"


# ============ 默认参数 ============
LAMBDA = 0.054
GAMMA = 0.79
PSI = 0.40
SCALE_C = 3.0
TANH_B = 10.0


# ============ 比赛权重（按赛事级别）============
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.5,
    "FIFA World Cup qualification": 1.2,
    "UEFA Euro": 1.4,
    "UEFA Euro qualification": 1.0,
    "Copa América": 1.4,
    "Africa Cup of Nations": 1.2,
    "AFC Asian Cup": 1.1,
    "Friendly": 0.6,
    "UEFA Nations League": 1.0,
}


def expected_goal_diff(r_home: float, r_away: float,
                         c: float = SCALE_C, b: float = TANH_B) -> float:
    """期望净胜球（主队视角）"""
    raw = (r_home - r_away) / c
    return b * math.tanh(raw)


# ============ 增量更新 ============
def update_step(R_H: Dict[str, float], R_A: Dict[str, float],
                 home: str, away: str,
                 home_score: int, away_score: int,
                 weight: float = 1.0,
                 neutral: bool = False,
                 lam: float = LAMBDA, gamma: float = GAMMA,
                 psi: float = PSI) -> None:
    """单场更新（in-place 修改 R_H / R_A）"""
    R_H.setdefault(home, 0.0)
    R_A.setdefault(home, 0.0)
    R_H.setdefault(away, 0.0)
    R_A.setdefault(away, 0.0)

    # 中立场地：双方都用主-客均值
    if neutral:
        r_h = (R_H[home] + R_A[home]) / 2.0
        r_a = (R_H[away] + R_A[away]) / 2.0
    else:
        r_h = R_H[home]
        r_a = R_A[away]

    expected = expected_goal_diff(r_h, r_a)
    actual = home_score - away_score
    err = actual - expected

    # 误差缩放（避免大屠杀过度拉升评级）
    sign = 1 if err >= 0 else -1
    err_smoothed = sign * math.log(1 + abs(err))

    delta = weight * lam * err_smoothed

    # 主队主场更新（GAMMA 权重）
    R_H[home] += delta * gamma
    R_A[home] += delta * gamma * psi
    # 客队客场更新（1-GAMMA 权重）
    R_A[away] -= delta * (1 - gamma)
    R_H[away] -= delta * (1 - gamma) * psi


# ============ 全量训练（扫 5 万场）============
def train_pi_ratings(csv_path: Optional[Path] = None,
                      max_rows: Optional[int] = None,
                      since_year: int = 1990) -> Tuple[Dict[str, float], Dict[str, float], Dict]:
    """
    扫历史数据训练 pi-ratings
    
    Returns:
        (R_H, R_A, meta)
        meta = {
            "n_matches": ...,
            "n_teams": ...,
            "form_5": {team: {"wins":int, "draws":int, "losses":int, "goal_diff":int}}
        }
    """
    csv_path = csv_path or (DATA_HIST / "results.csv")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= since_year].sort_values("date").reset_index(drop=True)
    if max_rows:
        df = df.head(max_rows)

    R_H: Dict[str, float] = {}
    R_A: Dict[str, float] = {}
    # 保存最近 5 场用于 form 特征
    recent_results: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        try:
            hs, ascore = int(row["home_score"]), int(row["away_score"])
        except Exception:
            continue
        tournament = row.get("tournament", "Friendly")
        weight = TOURNAMENT_WEIGHTS.get(tournament, 0.8)
        neutral = bool(row.get("neutral", False))

        update_step(R_H, R_A, home, away, hs, ascore,
                     weight=weight, neutral=neutral)

        # 记录 form
        if hs > ascore:
            recent_results[home].append(("W", hs - ascore))
            recent_results[away].append(("L", ascore - hs))
        elif hs < ascore:
            recent_results[home].append(("L", hs - ascore))
            recent_results[away].append(("W", ascore - hs))
        else:
            recent_results[home].append(("D", 0))
            recent_results[away].append(("D", 0))

    # 汇总 form_5
    form_5: Dict[str, Dict] = {}
    for team, results in recent_results.items():
        wins = sum(1 for r, _ in results if r == "W")
        draws = sum(1 for r, _ in results if r == "D")
        losses = sum(1 for r, _ in results if r == "L")
        gd = sum(g for _, g in results)
        form_5[team] = {
            "wins": wins, "draws": draws, "losses": losses,
            "goal_diff": gd,
            "form_score": (wins * 3 + draws) / (3 * max(1, len(results))),
        }

    meta = {
        "n_matches": len(df),
        "n_teams": len(R_H),
        "since_year": since_year,
        "form_5": form_5,
    }
    return R_H, R_A, meta


# ============ 持久化 ============
def save_ratings(R_H: Dict[str, float], R_A: Dict[str, float],
                  meta: Dict, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or (DATA_OUTPUTS / "pi_ratings.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "R_H": R_H,
        "R_A": R_A,
        "meta": meta,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out_path


def load_ratings(path: Optional[Path] = None) -> Tuple[Dict, Dict, Dict]:
    path = path or (DATA_OUTPUTS / "pi_ratings.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，先跑 train")
    d = json.load(open(path))
    return d["R_H"], d["R_A"], d.get("meta", {})


# ============ 特征构造（用于 CatBoost 输入）============
def build_features(R_H: Dict[str, float], R_A: Dict[str, float],
                    home: str, away: str, neutral: bool = False,
                    form_5: Optional[Dict] = None) -> Dict[str, float]:
    """
    给定一场比赛，返回 7 维 pi-rating 特征
    """
    rh_h = R_H.get(home, 0.0)
    ra_h = R_A.get(home, 0.0)
    rh_a = R_H.get(away, 0.0)
    ra_a = R_A.get(away, 0.0)

    if neutral:
        r_home_eff = (rh_h + ra_h) / 2.0
        r_away_eff = (rh_a + ra_a) / 2.0
    else:
        r_home_eff = rh_h
        r_away_eff = ra_a

    expected_gd = expected_goal_diff(r_home_eff, r_away_eff)

    form_5 = form_5 or {}
    form_h = form_5.get(home, {}).get("form_score", 0.5)
    form_a = form_5.get(away, {}).get("form_score", 0.5)

    return {
        "pi_diff": r_home_eff - r_away_eff,
        "pi_H_diff": rh_h - rh_a,
        "pi_A_diff": ra_h - ra_a,
        "pi_self_h": rh_h - ra_h,         # 主队的"主-客差"
        "pi_self_a": rh_a - ra_a,
        "expected_gd": expected_gd,
        "form_diff": form_h - form_a,
        "neutral": 1.0 if neutral else 0.0,
    }


# ============ CLI ============
if __name__ == "__main__":
    import sys, time
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "train":
        print("⏳ 训练 pi-ratings (1990+)...")
        t0 = time.time()
        R_H, R_A, meta = train_pi_ratings()
        print(f"✓ 完成: {meta['n_matches']:,} 场, "
              f"{meta['n_teams']} 队, 耗时 {time.time()-t0:.1f}s")
        save_ratings(R_H, R_A, meta)
        print(f"📄 写入 data/outputs/pi_ratings.json\n")

        # Top 12 by combined rating
        combined = {t: R_H[t] + R_A[t] for t in R_H}
        sorted_t = sorted(combined.items(), key=lambda x: -x[1])[:12]
        print("Top 12 by combined pi-rating:")
        for t, c in sorted_t:
            print(f"  {t:<22} R_H={R_H[t]:+.3f} R_A={R_A[t]:+.3f}  "
                  f"form={meta['form_5'].get(t,{}).get('form_score',0):.2f}")

    elif cmd == "predict":
        # python3 pi_ratings.py predict Spain Argentina --neutral
        home, away = sys.argv[2], sys.argv[3]
        neutral = "--neutral" in sys.argv
        R_H, R_A, meta = load_ratings()
        feats = build_features(R_H, R_A, home, away, neutral, meta.get("form_5"))
        print(f"\n{home} vs {away} (neutral={neutral}):")
        for k, v in feats.items():
            print(f"  {k:<14} {v:+.4f}")

    else:
        print(__doc__)
