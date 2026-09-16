"""
历史比赛数据加载器（支持 ML 训练）

数据来源：
  martj42/international_results 
  - results.csv: 49,478 场国际比赛 (1872-2024)
  - shootouts.csv: 678 场点球大战
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import ROOT

HIST_DIR = ROOT / "data" / "historical"


def load_results(min_year: int = 1990) -> pd.DataFrame:
    """加载 results.csv，过滤近代比赛"""
    df = pd.read_csv(HIST_DIR / "results.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df[df["year"] >= min_year].copy()
    
    # 计算结果
    df["result"] = np.where(
        df["home_score"] > df["away_score"], "H",
        np.where(df["home_score"] < df["away_score"], "A", "D")
    )
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["goal_diff"] = df["home_score"] - df["away_score"]
    
    return df


def filter_competitive(df: pd.DataFrame, exclude_friendly: bool = True) -> pd.DataFrame:
    """过滤掉友谊赛（保留世界杯/欧洲杯/美洲杯/预选赛等）"""
    if exclude_friendly:
        df = df[df["tournament"] != "Friendly"].copy()
    return df


def get_team_form(df: pd.DataFrame, team: str, before_date, n: int = 5) -> dict:
    """
    获取某队在某日期前 N 场的战绩
    """
    team_matches = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    ].sort_values("date", ascending=False).head(n)
    
    if len(team_matches) == 0:
        return {"wins": 0, "draws": 0, "losses": 0, "goals_for": 0, "goals_against": 0, "n": 0}
    
    wins = draws = losses = goals_for = goals_against = 0
    for _, m in team_matches.iterrows():
        if m["home_team"] == team:
            gf, ga = m["home_score"], m["away_score"]
        else:
            gf, ga = m["away_score"], m["home_score"]
        goals_for += gf
        goals_against += ga
        if gf > ga: wins += 1
        elif gf < ga: losses += 1
        else: draws += 1
    
    n_matches = len(team_matches)
    return {
        "wins": wins, "draws": draws, "losses": losses,
        "goals_for": goals_for, "goals_against": goals_against,
        "n": n_matches,
        "win_rate": wins / n_matches,
        "ppg": (3*wins + draws) / n_matches,  # 场均积分
    }


def get_h2h(df: pd.DataFrame, team_a: str, team_b: str, before_date, years: int = 5) -> dict:
    """获取两队近 N 年交锋记录"""
    earliest = before_date - pd.Timedelta(days=365 * years)
    matches = df[
        (((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
         ((df["home_team"] == team_b) & (df["away_team"] == team_a))) &
        (df["date"] < before_date) & (df["date"] >= earliest)
    ]
    
    if len(matches) == 0:
        return {"a_wins": 0, "draws": 0, "b_wins": 0, "n": 0, "a_advantage": 0.0}
    
    a_wins = draws = b_wins = 0
    for _, m in matches.iterrows():
        if m["home_team"] == team_a:
            if m["home_score"] > m["away_score"]: a_wins += 1
            elif m["home_score"] < m["away_score"]: b_wins += 1
            else: draws += 1
        else:
            if m["away_score"] > m["home_score"]: a_wins += 1
            elif m["away_score"] < m["home_score"]: b_wins += 1
            else: draws += 1
    
    n = len(matches)
    return {
        "a_wins": a_wins, "draws": draws, "b_wins": b_wins, "n": n,
        "a_advantage": (a_wins - b_wins) / n,  # -1 ~ +1
    }


def build_training_dataset(min_year: int = 2000, exclude_friendly: bool = True) -> pd.DataFrame:
    """
    构建 ML 训练数据集（每场比赛一行，含特征 + 标签）
    
    返回字段：
        date, home_team, away_team, result, target_home_win
        + 特征列：home_form_ppg, away_form_ppg, h2h_advantage, ...
    """
    print(f"📊 构建训练数据集（{min_year}+, 非友谊赛={exclude_friendly}）")
    df = load_results(min_year=min_year)
    df_full = df.copy()  # 完整数据用于查询历史战绩
    
    if exclude_friendly:
        df = df[df["tournament"] != "Friendly"].copy()
    
    print(f"   过滤后比赛数: {len(df)}")
    
    rows = []
    for idx, m in df.iterrows():
        home_form = get_team_form(df_full, m["home_team"], m["date"], n=5)
        away_form = get_team_form(df_full, m["away_team"], m["date"], n=5)
        h2h = get_h2h(df_full, m["home_team"], m["away_team"], m["date"], years=5)
        
        rows.append({
            "date": m["date"],
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "tournament": m["tournament"],
            "neutral": m["neutral"],
            "home_score": m["home_score"],
            "away_score": m["away_score"],
            "result": m["result"],
            "home_win": 1 if m["result"] == "H" else 0,
            "draw": 1 if m["result"] == "D" else 0,
            "home_form_ppg": home_form["ppg"],
            "away_form_ppg": away_form["ppg"],
            "form_diff": home_form["ppg"] - away_form["ppg"],
            "h2h_n": h2h["n"],
            "h2h_advantage": h2h["a_advantage"],
            "home_advantage_flag": 0 if m["neutral"] else 1,
        })
        
        if (idx + 1) % 5000 == 0:
            print(f"   Progress: {idx+1}/{len(df)}")
    
    return pd.DataFrame(rows)


def quick_stats():
    """打印数据集快速统计"""
    df = load_results(min_year=1990)
    
    print("=" * 70)
    print(f"📊 历史比赛数据集（1990 - 至今）")
    print("=" * 70)
    print(f"\n总比赛数: {len(df):,}")
    print(f"日期范围: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\n按赛事分布（Top 10）：")
    tournament_counts = df["tournament"].value_counts().head(10)
    for t, c in tournament_counts.items():
        print(f"  {t:<35} {c:>6,}")
    
    print(f"\n按结果分布：")
    result_dist = df["result"].value_counts(normalize=True)
    print(f"  主胜 H:  {result_dist.get('H', 0)*100:.1f}%")
    print(f"  平局 D:  {result_dist.get('D', 0)*100:.1f}%")
    print(f"  客胜 A:  {result_dist.get('A', 0)*100:.1f}%")
    
    # 中立场比例
    neutral_pct = df["neutral"].mean() * 100
    print(f"\n中立场比例: {neutral_pct:.1f}%")
    
    # 平均进球
    print(f"平均总进球: {df['total_goals'].mean():.2f}")
    
    # 世界杯专属统计
    wc = df[df["tournament"].str.contains("FIFA World Cup", na=False)]
    print(f"\n世界杯比赛数: {len(wc)}")
    print(f"  主胜率: {(wc['result']=='H').mean()*100:.1f}%")
    print(f"  平局率: {(wc['result']=='D').mean()*100:.1f}%")
    print(f"  客胜率: {(wc['result']=='A').mean()*100:.1f}%")
    print(f"  场均进球: {wc['total_goals'].mean():.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        quick_stats()
    elif len(sys.argv) > 1 and sys.argv[1] == "build":
        df = build_training_dataset(min_year=2010)
        out = HIST_DIR / "training_set.csv"
        df.to_csv(out, index=False)
        print(f"\n✅ 已保存训练集: {out}")
        print(f"   行数: {len(df):,}, 列数: {len(df.columns)}")
    else:
        print("用法:")
        print("  python3 historical_loader.py stats    # 数据集统计")
        print("  python3 historical_loader.py build    # 构建 ML 训练集")
