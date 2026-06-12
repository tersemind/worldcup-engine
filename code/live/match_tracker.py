#!/usr/bin/env python3
"""
赛中实时调整引擎（#16 P4）

功能：
1. 接收实际比赛结果（手动输入或赛后回填）
2. 更新 teams.json 中的 Elo（按 K=60 淘汰赛权重）
3. 标记球队为已淘汰
4. 重跑剩余比赛的预测
5. 推送变化报告

使用场景：
  小组赛 → 实时更新 → 重算淘汰赛概率
  淘汰赛 → 实时更新 → 推动决赛对阵概率收敛
"""
import json
import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW, DATA_OUTPUTS, ROOT
from models.elo_engine import update_elo, get_k_value


LIVE_STATE = ROOT / "data" / "raw" / "live_state.json"


def init_live_state():
    """初始化实时状态文件"""
    if LIVE_STATE.exists():
        return load_live_state()
    state = {
        "tournament": "2026 FIFA World Cup",
        "started_at": None,
        "matches_played": [],
        "eliminated_teams": [],
        "group_standings": {},
        "qualified_to_r32": [],
        "current_stage": "group",  # group / r32 / r16 / qf / sf / final / done
        "last_updated": None,
    }
    save_live_state(state)
    return state


def load_live_state():
    if not LIVE_STATE.exists():
        return init_live_state()
    with open(LIVE_STATE, "r") as f:
        return json.load(f)


def save_live_state(state: dict):
    state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LIVE_STATE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def record_match_result(home_team: str, away_team: str,
                          home_score: int, away_score: int,
                          stage: str = "group", date: str = None):
    """
    记录一场比赛结果，自动：
    1. 更新两队 Elo
    2. 添加到 matches_played
    3. 如是淘汰赛，标记败者为 eliminated
    """
    print(f"⚽ 记录: {home_team} {home_score}-{away_score} {away_team} ({stage})")
    
    # 加载数据
    teams_path = DATA_RAW / "teams.json"
    with open(teams_path, "r") as f:
        teams_data = json.load(f)
    
    if home_team not in teams_data["teams"]:
        print(f"❌ 未知球队: {home_team}")
        return
    if away_team not in teams_data["teams"]:
        print(f"❌ 未知球队: {away_team}")
        return
    
    # 计算结果
    if home_score > away_score:
        result = 1.0
        winner, loser = home_team, away_team
    elif home_score < away_score:
        result = 0.0
        winner, loser = away_team, home_team
    else:
        result = 0.5
        winner, loser = None, None
    
    # 更新 Elo（按阶段决定 K 值）
    k_table = {
        "group": 50,
        "r32": 60,
        "r16": 60,
        "qf": 60,
        "sf": 60,
        "final": 60,
    }
    k = k_table.get(stage, 50)
    
    home_elo = teams_data["teams"][home_team]["elo"]
    away_elo = teams_data["teams"][away_team]["elo"]
    
    new_home, new_away = update_elo(home_elo, away_elo, result, k=k)
    
    teams_data["teams"][home_team]["elo"] = round(new_home, 1)
    teams_data["teams"][away_team]["elo"] = round(new_away, 1)
    
    # 记录时间戳
    if "_last_updated" not in teams_data:
        teams_data["_last_updated"] = {}
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    teams_data["_last_updated"].setdefault(home_team, {})["elo"] = timestamp
    teams_data["_last_updated"].setdefault(away_team, {})["elo"] = timestamp
    
    with open(teams_path, "w") as f:
        json.dump(teams_data, f, indent=2, ensure_ascii=False)
    
    print(f"   {home_team} Elo: {home_elo:.1f} → {new_home:.1f}")
    print(f"   {away_team} Elo: {away_elo:.1f} → {new_away:.1f}")
    
    # 更新 live state
    state = load_live_state()
    state["matches_played"].append({
        "date": date or time.strftime("%Y-%m-%d"),
        "stage": stage,
        "home": home_team,
        "away": away_team,
        "score": f"{home_score}-{away_score}",
        "winner": winner,
    })
    
    # 淘汰赛阶段：败者出局
    if stage in ("r32", "r16", "qf", "sf", "final") and loser:
        if loser not in state["eliminated_teams"]:
            state["eliminated_teams"].append(loser)
            print(f"   🚫 {loser} 已被淘汰")
    
    save_live_state(state)
    
    # 自动重跑预测
    print(f"\n🔄 自动重跑预测...")
    rerun_prediction()


def rerun_prediction(n_sim: int = 50000):
    """重跑蒙特卡洛 + 综合预测"""
    code_dir = ROOT / "code"
    
    print(f"   蒙特卡洛 {n_sim:,} 次...")
    subprocess.run(
        [sys.executable, str(code_dir / "models" / "finals_analyzer.py"), str(n_sim)],
        capture_output=True, text=True, timeout=120
    )
    
    print(f"   综合预测...")
    subprocess.run(
        [sys.executable, str(code_dir / "models" / "synthesizer.py")],
        capture_output=True, text=True, timeout=30
    )
    
    print(f"   ✅ 预测已更新")


def diff_with_previous():
    """对比当前预测与上一次的差异"""
    state = load_live_state()
    n_played = len(state.get("matches_played", []))
    
    print(f"\n📊 已记录 {n_played} 场比赛")
    print(f"   已淘汰球队: {len(state.get('eliminated_teams', []))}")
    
    # 读最新综合报告
    synth_path = DATA_OUTPUTS / "synthesizer_report.json"
    if not synth_path.exists():
        print("⚠️  尚未生成综合报告")
        return
    
    with open(synth_path) as f:
        curr = json.load(f)
    
    sorted_curr = sorted(curr.items(), key=lambda x: -x[1]["final_probability"])[:10]
    
    print(f"\n🏆 当前 Top 10 冠军预测：")
    eliminated = set(state.get("eliminated_teams", []))
    for team, d in sorted_curr:
        if team in eliminated:
            continue
        prob = d["final_probability"]
        ci = f"[{d['ci_lower']:.1f}-{d['ci_upper']:.1f}]"
        print(f"   {team:<14} {prob:>5.2f}%  CI {ci}")


def show_status():
    """显示赛事进度"""
    state = load_live_state()
    print("=" * 60)
    print(f"⚽ 实时状态: {state['tournament']}")
    print("=" * 60)
    print(f"  当前阶段: {state['current_stage']}")
    print(f"  已记录比赛: {len(state.get('matches_played', []))}")
    print(f"  已淘汰球队: {len(state.get('eliminated_teams', []))}")
    print(f"  最后更新: {state.get('last_updated', 'N/A')}")
    
    if state.get("matches_played"):
        print(f"\n📅 最近 5 场比赛：")
        for m in state["matches_played"][-5:]:
            print(f"  [{m['stage']}] {m['date']}  {m['home']} {m['score']} {m['away']}")
    
    if state.get("eliminated_teams"):
        print(f"\n🚫 已淘汰: {', '.join(state['eliminated_teams'])}")


def reset_live():
    """重置实时状态（赛事结束后）"""
    print("⚠️  这将清空所有实时记录，恢复到赛前状态")
    if input("确认？(yes/no): ").lower() != "yes":
        return
    init_live_state()
    print("✅ 已重置")


def main():
    if len(sys.argv) == 1 or sys.argv[1] == "status":
        show_status()
    elif sys.argv[1] == "record" and len(sys.argv) >= 6:
        # 用法：record <home> <away> <home_score> <away_score> [stage]
        home = sys.argv[2]
        away = sys.argv[3]
        h_score = int(sys.argv[4])
        a_score = int(sys.argv[5])
        stage = sys.argv[6] if len(sys.argv) > 6 else "group"
        record_match_result(home, away, h_score, a_score, stage=stage)
        diff_with_previous()
    elif sys.argv[1] == "diff":
        diff_with_previous()
    elif sys.argv[1] == "rerun":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
        rerun_prediction(n)
    elif sys.argv[1] == "reset":
        reset_live()
    elif sys.argv[1] == "init":
        init_live_state()
        print("✅ 实时状态已初始化")
    else:
        print("用法:")
        print("  python3 match_tracker.py status                              # 查看状态")
        print("  python3 match_tracker.py record <home> <away> <h> <a> [stage]  # 记录比赛")
        print("  python3 match_tracker.py diff                                # 当前 Top 10")
        print("  python3 match_tracker.py rerun [n]                           # 手动重跑")
        print("  python3 match_tracker.py reset                               # 重置")
        print("\nstage 可选：group / r32 / r16 / qf / sf / final")


if __name__ == "__main__":
    main()
