"""
数据刷新辅助工具：
- 由 Claude 在 /wc-predict refresh 命令中调用
- 提供给 Claude 一个清单：当前 teams.json 数据 + 待更新字段 + 推荐数据源
- Claude 用 WebSearch/WebFetch 抓取后，调用 update_team_field() 写回
"""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_teams, DATA_RAW


def show_current_data(top_n: int = 16) -> dict:
    """
    返回当前 teams.json 的关键数据，供 Claude 对照刷新
    """
    teams = load_teams()
    qualified = [(name, data) for name, data in teams.items() if data.get("group") != "_"]
    
    # 按 Elo 排序
    qualified.sort(key=lambda x: -x[1]["elo"])
    
    output = {
        "snapshot_path": str(DATA_RAW / "teams.json"),
        "total_teams": len(qualified),
        "fields_to_refresh": [
            "elo (推荐源: eloratings.net)",
            "fifa (推荐源: FIFA Rankings 官方)",
            "xg_for / xg_against (推荐源: FBRef 近 10 场)",
            "market_implied (推荐源: Polymarket / Kalshi 实时)",
        ],
        "top_teams_current": []
    }
    
    for name, data in qualified[:top_n]:
        output["top_teams_current"].append({
            "team": name,
            "elo": data.get("elo"),
            "fifa": data.get("fifa"),
            "xg_for": data.get("xg_for"),
            "xg_against": data.get("xg_against"),
            "market_implied": data.get("market_implied"),
        })
    
    return output


def update_team_field(team_name: str, field: str, value, write: bool = True) -> bool:
    """
    更新某球队某字段，并保存到 teams.json
    
    用法：
        update_team_field("Spain", "elo", 2160)
        update_team_field("France", "market_implied", 0.175)
    """
    file_path = DATA_RAW / "teams.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if team_name not in data["teams"]:
        print(f"❌ 球队不存在: {team_name}")
        return False
    
    valid_fields = ["elo", "fifa", "xg_for", "xg_against", "market_implied"]
    if field not in valid_fields:
        print(f"❌ 不支持的字段: {field}（合法字段: {valid_fields}）")
        return False
    
    old_value = data["teams"][team_name].get(field)
    data["teams"][team_name][field] = value
    
    # 添加最后更新时间
    if "_last_updated" not in data:
        data["_last_updated"] = {}
    data["_last_updated"][team_name] = data["_last_updated"].get(team_name, {})
    data["_last_updated"][team_name][field] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if write:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ {team_name}.{field}: {old_value} → {value}")
    return True


def batch_update(updates: list, write: bool = True) -> int:
    """
    批量更新
    updates: [{"team": "Spain", "field": "elo", "value": 2160}, ...]
    """
    success = 0
    for u in updates:
        if update_team_field(u["team"], u["field"], u["value"], write=write):
            success += 1
    return success


def get_refresh_checklist() -> str:
    """生成给 Claude 的刷新清单（让 Claude 知道要搜什么）"""
    teams = load_teams()
    qualified = [(name, data) for name, data in teams.items() if data.get("group") != "_"]
    qualified.sort(key=lambda x: -x[1]["elo"])
    
    checklist = """
📋 数据刷新清单（共 4 项）：

1. Elo 评级（推荐源：eloratings.net/World）
   - 抓取 Top 30 国家队最新 Elo
   - 用 WebFetch: https://www.eloratings.net/World

2. 市场隐含概率（推荐源：Polymarket 实时）
   - 用 WebFetch: https://predictmarketcap.com/canonical/2026-fifa-world-cup-winner
   - 或: https://polymarket.com/event/world-cup-winner

3. 关键伤病（推荐源：ESPN/BBC 实时）
   - 用 WebSearch: "2026 World Cup [team name] injuries June 2026"
   - 重点关注：Yamal, Mbappe, Saka, Messi, Neuer

4. 预测市场赔率（推荐源：Kalshi）
   - 用 WebFetch: https://kalshi.com/markets/world-cup-winner

📊 当前 Top 16 球队数据快照：
"""
    for name, data in qualified[:16]:
        checklist += f"\n  {name:<18} Elo {data['elo']:<6} FIFA #{data['fifa']:<4} xG {data['xg_for']:.2f}/场  Market {data['market_implied']*100:>5.1f}%"
    
    checklist += """

🔧 更新方式：
  用 Claude 的 Edit 工具直接修改 ~/.codebuddy/worldcup-predict/data/raw/teams.json
  
  或调用 Python 脚本：
  python3 ~/.codebuddy/worldcup-predict/code/data/refresh_helper.py update <team> <field> <value>
"""
    return checklist


def main():
    if len(sys.argv) == 1 or sys.argv[1] == "show":
        # 默认显示当前数据
        print(get_refresh_checklist())
        return
    
    if sys.argv[1] == "update" and len(sys.argv) == 5:
        team = sys.argv[2]
        field = sys.argv[3]
        try:
            value = float(sys.argv[4])
            if field in ["elo", "fifa"]:
                value = int(value)
        except ValueError:
            value = sys.argv[4]
        update_team_field(team, field, value)
        return
    
    if sys.argv[1] == "snapshot":
        # JSON 输出供脚本调用
        snap = show_current_data(top_n=20)
        print(json.dumps(snap, indent=2, ensure_ascii=False))
        return
    
    print("用法:")
    print("  python3 refresh_helper.py show                           # 显示刷新清单")
    print("  python3 refresh_helper.py snapshot                       # JSON 快照")
    print("  python3 refresh_helper.py update <team> <field> <value>  # 更新单字段")


if __name__ == "__main__":
    main()
