"""
伤病情报抓取器（Claude + WebSearch 联动）

工作流：
1. 提供关键球员清单 + 推荐搜索词给 Claude
2. Claude 用 WebSearch 抓取最新伤情
3. Claude 解析为结构化数据
4. 调用 update_injury() 写入伤病库
5. synthesizer.py 自动读取伤病库，调整 health 系数
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW

INJURIES_FILE = DATA_RAW / "injuries.json"


# 关键球员清单（按球队分类，标注不可替代性）
KEY_PLAYERS = {
    "Spain": [
        {"name": "Lamine Yamal", "position": "RW", "irreplaceability": "S", "notes": "腿筋伤情持续追踪"},
        {"name": "Rodri", "position": "DM", "irreplaceability": "S", "notes": "ACL 恢复期"},
        {"name": "Pedri", "position": "AM", "irreplaceability": "A", "notes": "状态稳定"},
    ],
    "France": [
        {"name": "Kylian Mbappe", "position": "ST", "irreplaceability": "S", "notes": "4月小腿伤"},
        {"name": "Ousmane Dembele", "position": "RW", "irreplaceability": "A", "notes": "健康"},
        {"name": "Aurelien Tchouameni", "position": "DM", "irreplaceability": "A", "notes": ""},
    ],
    "Argentina": [
        {"name": "Lionel Messi", "position": "AM", "irreplaceability": "S", "notes": "腿筋疲劳"},
        {"name": "Lautaro Martinez", "position": "ST", "irreplaceability": "A", "notes": ""},
        {"name": "Cristian Romero", "position": "CB", "irreplaceability": "A", "notes": "膝伤需关注"},
    ],
    "England": [
        {"name": "Bukayo Saka", "position": "RW", "irreplaceability": "A", "notes": "头部伤未达标"},
        {"name": "Harry Kane", "position": "ST", "irreplaceability": "S", "notes": ""},
        {"name": "Jude Bellingham", "position": "AM", "irreplaceability": "S", "notes": "状态迷雾"},
    ],
    "Brazil": [
        {"name": "Vinicius Junior", "position": "LW", "irreplaceability": "S", "notes": "巅峰健康"},
        {"name": "Rodrygo", "position": "RW", "irreplaceability": "A", "notes": "ACL 确认缺席"},
        {"name": "Neymar", "position": "AM", "irreplaceability": "A", "notes": "小腿伤"},
    ],
    "Germany": [
        {"name": "Jamal Musiala", "position": "AM", "irreplaceability": "S", "notes": ""},
        {"name": "Florian Wirtz", "position": "AM", "irreplaceability": "A", "notes": ""},
        {"name": "Manuel Neuer", "position": "GK", "irreplaceability": "A", "notes": "40岁年龄风险"},
    ],
    "Portugal": [
        {"name": "Bruno Fernandes", "position": "AM", "irreplaceability": "S", "notes": "创纪录赛季"},
        {"name": "Cristiano Ronaldo", "position": "ST", "irreplaceability": "B", "notes": "39岁体能管理"},
    ],
    "Netherlands": [
        {"name": "Jurrien Timber", "position": "RB", "irreplaceability": "A", "notes": "确认伤缺"},
        {"name": "Xavi Simons", "position": "AM", "irreplaceability": "S", "notes": "确认伤缺"},
        {"name": "Stefan de Vrij", "position": "CB", "irreplaceability": "A", "notes": "确认伤缺"},
    ],
}


# 状态分级 → health 调整因子（pp）
STATUS_TO_ADJ = {
    "out_for_tournament": -3.0,      # 整届缺席
    "out_for_knockouts": -2.5,        # 淘汰赛缺席
    "uncertain": -1.5,                # 不确定能否上场
    "70_percent": -1.2,               # 带伤 70% 状态
    "80_percent": -0.7,               # 带伤 80% 状态
    "90_percent": -0.3,               # 带伤 90% 状态
    "fit": 0.0,                       # 完全健康
}


def load_injuries():
    """加载现有伤病库"""
    if not INJURIES_FILE.exists():
        return {"_comment": "Injury database for World Cup teams", "_updated": "", "teams": {}}
    with open(INJURIES_FILE, "r") as f:
        return json.load(f)


def save_injuries(data: dict):
    """保存伤病库"""
    data["_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(INJURIES_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def update_injury(team: str, player: str, status: str, severity_pp: float = None):
    """
    更新单个球员伤情
    
    Args:
        team: 球队名（如 "Spain"）
        player: 球员名（如 "Lamine Yamal"）
        status: 状态 (fit/90_percent/80_percent/70_percent/uncertain/out_for_knockouts/out_for_tournament)
        severity_pp: 自定义对球队 health 的调整（pp），None 则用默认映射
    """
    data = load_injuries()
    if "teams" not in data:
        data["teams"] = {}
    if team not in data["teams"]:
        data["teams"][team] = {"injuries": [], "total_health_adj_pp": 0.0}
    
    adj = severity_pp if severity_pp is not None else STATUS_TO_ADJ.get(status, 0.0)
    
    # 移除旧记录（同一球员）
    data["teams"][team]["injuries"] = [
        inj for inj in data["teams"][team]["injuries"] if inj["player"] != player
    ]
    
    # 添加新记录
    data["teams"][team]["injuries"].append({
        "player": player,
        "status": status,
        "adj_pp": adj,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    
    # 重新计算总 health 调整
    total = sum(inj["adj_pp"] for inj in data["teams"][team]["injuries"])
    data["teams"][team]["total_health_adj_pp"] = round(total, 2)
    
    save_injuries(data)
    print(f"✅ {team} - {player}: {status} (adj {adj:+.2f}pp), 球队 total_health = {total:+.2f}pp")


def get_health_adjustments() -> dict:
    """
    返回 {team: total_health_adj_pp}，供 synthesizer.py 调用
    
    兼容两种格式：
      旧（手填）：{"teams": {team: {"total_health_adj_pp": ...}}}
      新（injury_radar）：{"injuries": {team: {"total_pp_impact": ...}}}
    """
    data = load_injuries()
    out = {}
    # 优先新格式（injury_radar 写入）
    if "injuries" in data:
        for team, info in data["injuries"].items():
            pp = info.get("total_pp_impact")
            if pp is not None:
                out[team] = pp
    # 兼容旧格式
    if "teams" in data:
        for team, info in data["teams"].items():
            if team not in out:  # 不覆盖新格式
                out[team] = info.get("total_health_adj_pp", 0.0)
    return out


def get_search_queries_for_claude() -> str:
    """生成给 Claude 的搜索词清单"""
    output = "📋 伤病情报搜索清单（按球队分组，给 Claude 用 WebSearch）：\n\n"
    
    for team, players in KEY_PLAYERS.items():
        output += f"## {team}\n"
        for p in players:
            query = f'"{p["name"]}" injury 2026 World Cup June'
            output += f'  - WebSearch: `{query}`\n'
            output += f"    球员: {p['name']} ({p['position']}, 不可替代性 {p['irreplaceability']})\n"
            if p["notes"]:
                output += f"    当前已知: {p['notes']}\n"
        output += "\n"
    
    output += """
🔧 抓取后调用方式：

  python3 ~/.codebuddy/worldcup-engine/code/data/injuries_fetcher.py update <team> <player> <status>

  status 可选值：
    fit              = 完全健康（adj 0pp）
    90_percent       = 带伤 90% 状态（adj -0.3pp）
    80_percent       = 带伤 80% 状态（adj -0.7pp）
    70_percent       = 带伤 70% 状态（adj -1.2pp）
    uncertain        = 不确定（adj -1.5pp）
    out_for_knockouts = 淘汰赛缺席（adj -2.5pp）
    out_for_tournament = 整届缺席（adj -3.0pp）
"""
    return output


def show_summary():
    """显示当前伤病库状态"""
    data = load_injuries()
    print(f"📅 最后更新: {data.get('_updated', 'N/A')}")
    print(f"📊 涉及球队: {len(data.get('teams', {}))}")
    print()
    
    teams = data.get("teams", {})
    if not teams:
        print("（伤病库为空）")
        return
    
    sorted_teams = sorted(teams.items(), key=lambda x: x[1].get("total_health_adj_pp", 0))
    
    print(f"{'Team':<14} {'Total Health Adj':<18} {'Injuries':<8}")
    print("-" * 50)
    for team, info in sorted_teams:
        adj = info.get("total_health_adj_pp", 0)
        n_inj = len(info.get("injuries", []))
        print(f"{team:<14} {adj:>+8.2f}pp{'':<10}{n_inj}")
    
    print("\n📋 详细伤情：")
    for team, info in sorted_teams:
        if info.get("injuries"):
            print(f"\n  {team}:")
            for inj in info["injuries"]:
                print(f"    - {inj['player']:<24} {inj['status']:<22} ({inj['adj_pp']:+.2f}pp)")


def main():
    if len(sys.argv) == 1 or sys.argv[1] == "show":
        show_summary()
        return
    
    if sys.argv[1] == "queries":
        print(get_search_queries_for_claude())
        return
    
    if sys.argv[1] == "update" and len(sys.argv) >= 5:
        team = sys.argv[2]
        player = sys.argv[3]
        status = sys.argv[4]
        custom_pp = float(sys.argv[5]) if len(sys.argv) > 5 else None
        update_injury(team, player, status, custom_pp)
        return
    
    if sys.argv[1] == "adjustments":
        adj = get_health_adjustments()
        print(json.dumps(adj, indent=2))
        return
    
    print("用法:")
    print("  python3 injuries_fetcher.py show                                # 显示伤病库")
    print("  python3 injuries_fetcher.py queries                              # 输出搜索清单")
    print("  python3 injuries_fetcher.py update <team> <player> <status>      # 更新伤情")
    print("  python3 injuries_fetcher.py adjustments                          # 返回各队 health 调整")


if __name__ == "__main__":
    main()
