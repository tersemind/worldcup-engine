"""
StatsBomb 开源数据集成（Layer 1）

参考 参考报告 2.1.1 表 2.4：StatsBomb L3 事件数据 ★★★★☆ 评级，
本研究"基于 StatsBomb 开源数据构建 xG 模型"。

可访问的开放数据：
  - FIFA World Cup 2018/2022 全场事件
  - UEFA Euro 2024 全场事件

输出：
  data/raw/statsbomb_events_<comp>_<season>.parquet
  data/raw/statsbomb_team_xt.json   ← 各国家队 xt_per_90 汇总（供 xt_engine 消费）

CLI：
  python3 statsbomb_loader.py fetch     # 拉 WC2022 + Euro2024
  python3 statsbomb_loader.py status    # 看本地缓存
  python3 statsbomb_loader.py xg        # 算各队场均 xG（验证）
"""
from __future__ import annotations
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# 抑制 statsbombpy 的友好警告
warnings.filterwarnings("ignore", category=UserWarning)

DATA_RAW = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


# ============ 关键赛事清单 ============
# (competition_id, season_id, label)
KEY_COMPETITIONS = [
    (43, 106, "FIFA_WC_2022"),
    (43,   3, "FIFA_WC_2018"),
    (55, 282, "UEFA_Euro_2024"),
]


def fetch_competition(comp_id: int, season_id: int, label: str,
                       force: bool = False) -> Path:
    """
    拉取一届赛事的全部事件，存为 parquet。
    返回保存路径。
    """
    out_path = DATA_RAW / f"statsbomb_events_{label}.parquet"
    if out_path.exists() and not force:
        print(f"  ✓ {label} 已存在，跳过（用 --force 重抓）")
        return out_path

    from statsbombpy import sb

    print(f"  ⏳ 拉取 {label}...")
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    n_matches = len(matches)
    print(f"     {n_matches} 场比赛")

    all_events = []
    for i, row in matches.iterrows():
        match_id = row["match_id"]
        try:
            ev = sb.events(match_id=match_id)
            ev["match_id"] = match_id
            ev["home_team"] = row["home_team"]
            ev["away_team"] = row["away_team"]
            ev["match_date"] = row["match_date"]
            all_events.append(ev)
            if (i + 1) % 10 == 0:
                print(f"     ... {i+1}/{n_matches}")
        except Exception as e:
            print(f"     ⚠ 跳过 match_id={match_id}: {e}")

    if not all_events:
        raise RuntimeError(f"没拿到任何事件: {label}")

    import pandas as pd
    df = pd.concat(all_events, ignore_index=True)

    # 只保留我们关心的列（事件级 xT 计算所需）
    keep_cols = [
        "match_id", "match_date", "home_team", "away_team",
        "team", "type", "minute", "second",
        "location", "pass_end_location", "carry_end_location",
        "shot_statsbomb_xg", "shot_outcome",
        "play_pattern", "duration", "under_pressure",
    ]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available]
    df.to_parquet(out_path, compression="snappy")
    print(f"  ✓ {label}: {len(df):,} 事件 → {out_path.relative_to(DATA_RAW.parent.parent)}")
    return out_path


def fetch_all(force: bool = False) -> List[Path]:
    """拉所有关键赛事"""
    paths = []
    for comp_id, season_id, label in KEY_COMPETITIONS:
        try:
            p = fetch_competition(comp_id, season_id, label, force=force)
            paths.append(p)
        except Exception as e:
            print(f"  ✗ {label} 失败: {e}")
    return paths


def status() -> Dict[str, dict]:
    """查看本地缓存状态"""
    info = {}
    for _, _, label in KEY_COMPETITIONS:
        p = DATA_RAW / f"statsbomb_events_{label}.parquet"
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            info[label] = {
                "path": str(p.relative_to(DATA_RAW.parent.parent)),
                "size_mb": round(size_mb, 1),
                "exists": True,
            }
        else:
            info[label] = {"exists": False}
    return info


def compute_team_xg_per_match() -> Dict[str, Dict[str, float]]:
    """
    从已下载事件计算每队场均 xG（验证管道正确性）
    输出：{team_name: {"xg_per_match": float, "matches": int}}
    """
    import pandas as pd
    team_stats = defaultdict(lambda: {"xg_total": 0.0, "match_ids": set()})

    for _, _, label in KEY_COMPETITIONS:
        p = DATA_RAW / f"statsbomb_events_{label}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        # 只看 Shot 事件
        shots = df[df["type"] == "Shot"].copy()
        for team, team_shots in shots.groupby("team"):
            team_stats[team]["xg_total"] += team_shots["shot_statsbomb_xg"].fillna(0).sum()
            team_stats[team]["match_ids"].update(team_shots["match_id"].unique())

    out = {}
    for team, s in team_stats.items():
        n = len(s["match_ids"])
        if n > 0:
            out[team] = {
                "xg_per_match": round(s["xg_total"] / n, 3),
                "matches": n,
                "xg_total": round(s["xg_total"], 2),
            }
    return out


# ============ CLI ============
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "fetch":
        force = "--force" in sys.argv
        print("📡 拉取 StatsBomb 开源数据\n")
        fetch_all(force=force)
        print()

    elif cmd == "status":
        print("📊 StatsBomb 本地缓存\n")
        info = status()
        for label, meta in info.items():
            mark = "✓" if meta["exists"] else "✗"
            extra = f" ({meta.get('size_mb', '?')} MB)" if meta["exists"] else ""
            print(f"  {mark} {label}{extra}")

    elif cmd == "xg":
        print("⚽ 各国家队场均 xG（StatsBomb 算法）\n")
        team_xg = compute_team_xg_per_match()
        sorted_teams = sorted(team_xg.items(),
                              key=lambda x: -x[1]["xg_per_match"])[:20]
        for name, s in sorted_teams:
            print(f"  {name:<25} {s['xg_per_match']:.2f} xG/场  "
                  f"({s['matches']} 场, 总 {s['xg_total']:.1f} xG)")

        # 保存
        out_path = DATA_RAW / "statsbomb_team_xg.json"
        out_path.write_text(json.dumps(team_xg, ensure_ascii=False, indent=2))
        print(f"\n📄 写入: {out_path.relative_to(DATA_RAW.parent.parent)}")

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
