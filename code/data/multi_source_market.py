"""
多源市场赔率聚合器
- 支持 Polymarket / Kalshi / Betfair / Pinnacle 等多平台
- 去除 overround（庄家抽水）后归一化
- 按交易量加权聚合
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW


# 平台权重（基于流动性/可信度）
PLATFORM_WEIGHTS = {
    "polymarket": 0.40,    # 最大流动性
    "kalshi": 0.30,        # CFTC 监管，权重高
    "betfair": 0.20,       # 传统交易所
    "pinnacle": 0.10,      # 锐价庄家
}


def remove_overround(odds_dict: dict) -> dict:
    """
    去除 overround（赔率隐含概率之和应为 1，超出部分是庄家抽水）
    
    输入：{team: implied_prob}
    输出：{team: normalized_prob}（和为 1.0）
    """
    total = sum(odds_dict.values())
    if total <= 0:
        return odds_dict
    return {team: prob / total for team, prob in odds_dict.items()}


def aggregate_market_odds(platforms: dict) -> dict:
    """
    多平台赔率加权聚合
    
    输入：
        platforms = {
            "polymarket": {"Spain": 0.175, "France": 0.16, ...},
            "kalshi":     {"Spain": 0.18,  "France": 0.16, ...},
            ...
        }
    
    输出：
        {team: weighted_avg_prob}
    """
    # 1. 每平台先去 overround
    normalized_per_platform = {}
    for platform, odds in platforms.items():
        if odds:
            normalized_per_platform[platform] = remove_overround(odds)
    
    # 2. 收集所有球队
    all_teams = set()
    for odds in normalized_per_platform.values():
        all_teams.update(odds.keys())
    
    # 3. 按平台权重加权聚合
    aggregated = {}
    for team in all_teams:
        weighted_sum = 0.0
        total_weight = 0.0
        for platform, odds in normalized_per_platform.items():
            if team in odds:
                w = PLATFORM_WEIGHTS.get(platform, 0.1)
                weighted_sum += odds[team] * w
                total_weight += w
        if total_weight > 0:
            aggregated[team] = weighted_sum / total_weight
    
    # 4. 最终归一化（保证和为 1.0）
    return remove_overround(aggregated)


def write_to_teams_json(aggregated: dict, dry_run: bool = False) -> int:
    """把聚合后的市场概率写回 teams.json"""
    file_path = DATA_RAW / "teams.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if "_last_updated" not in data:
        data["_last_updated"] = {}
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    success = 0
    
    for team, prob in aggregated.items():
        if team not in data["teams"]:
            continue
        old_value = data["teams"][team].get("market_implied", 0)
        # 四舍五入到 4 位小数
        new_value = round(prob, 4)
        if abs(old_value - new_value) > 0.001:  # 仅显著变化才更新
            data["teams"][team]["market_implied"] = new_value
            data["_last_updated"].setdefault(team, {})["market_implied"] = timestamp
            success += 1
    
    if not dry_run:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    return success


def get_template_for_claude() -> dict:
    """
    返回一个空模板，Claude 用 WebFetch 抓取后填入数据
    """
    return {
        "polymarket": {
            "_url": "https://predictmarketcap.com/canonical/2026-fifa-world-cup-winner",
            "_instruction": "用 WebFetch 抓取，提取每队的 Polymarket 概率（小数形式）",
            "data": {}
        },
        "kalshi": {
            "_url": "https://kalshi.com/markets/world-cup-winner",
            "_instruction": "用 WebFetch 抓取 Kalshi 的对应市场",
            "data": {}
        },
        "betfair": {
            "_url": "https://www.betfair.com/exchange/plus/football/competition/2026-world-cup",
            "_instruction": "用 WebSearch '2026 World Cup winner Betfair odds' 然后 WebFetch",
            "data": {}
        },
    }


def example_run():
    """示例运行（用模拟数据）"""
    platforms = {
        "polymarket": {
            "Spain": 0.175,
            "France": 0.16,
            "Argentina": 0.09,
            "England": 0.11,
            "Brazil": 0.085,
            "Portugal": 0.11,
        },
        "kalshi": {
            "Spain": 0.18,
            "France": 0.16,
            "Argentina": 0.09,
            "England": 0.11,
            "Brazil": 0.09,
            "Portugal": 0.11,
        },
    }
    
    print("📊 输入：")
    for p, odds in platforms.items():
        print(f"  {p}: 总和 {sum(odds.values()):.3f}")
    
    aggregated = aggregate_market_odds(platforms)
    
    print(f"\n📊 聚合后（已去 overround）：")
    for team, prob in sorted(aggregated.items(), key=lambda x: -x[1]):
        print(f"  {team:<14} {prob*100:>5.2f}%")
    print(f"\n  总和: {sum(aggregated.values())*100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "example":
        example_run()
    elif len(sys.argv) > 1 and sys.argv[1] == "template":
        print(json.dumps(get_template_for_claude(), indent=2, ensure_ascii=False))
    else:
        print("用法:")
        print("  python3 multi_source_market.py example      # 测试聚合算法")
        print("  python3 multi_source_market.py template     # 输出 Claude 抓取模板")
