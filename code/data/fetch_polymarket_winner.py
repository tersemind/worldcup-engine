"""
Polymarket 2026 FIFA World Cup Winner 实时抓取
================================================
来源: Polymarket Gamma API (无需 key)
事件: https://gamma-api.polymarket.com/events?slug=world-cup-winner

每个 market 是单队 "Will <Team> win the 2026 FIFA World Cup?"
outcomePrices = ["Yes_prob", "No_prob"]，取 Yes_prob 就是该队夺冠隐含概率。

写入: data/raw/external_predictions.json (models.polymarket.predictions)
"""
import sys
import json
import re
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW

GAMMA_URL = "https://gamma-api.polymarket.com/events?slug=world-cup-winner"
EXT_PATH = DATA_RAW / "external_predictions.json"

# Polymarket question 文本里的国名 → teams.json 标准名映射
POLY_TO_TEAM = {
    "Korea": "Korea Republic",
    "South Korea": "Korea Republic",
    "Ivory Coast": "Cote d'Ivoire",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Turkey": "Turkiye",
    "Türkiye": "Turkiye",
    "USA": "USA",
    "United States": "USA",
    "United States of America": "USA",
    "Curaçao": "Curacao",
    "Cabo Verde": "Cape Verde",
    "DR Congo": "DR Congo",
    "DRC": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Bosnia and Herzegovina": "Bosnia",
    "Bosnia & Herzegovina": "Bosnia",
}


def _parse_team(question: str) -> Optional[str]:
    """从 'Will Spain win the 2026 FIFA World Cup?' 提取 'Spain'"""
    m = re.match(r"^Will\s+(.+?)\s+win\s+the\s+2026", question, re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).strip()
    return POLY_TO_TEAM.get(raw, raw)


def fetch_polymarket_winner(timeout: int = 20) -> dict:
    """
    返回: {"team": prob_pct, ...} 比如 {"Spain": 16.95, "France": 16.0, ...}

    副作用：把每队市场的 slug 缓存到模块级 _LAST_SLUGS，给 write_to_external_predictions 用。
    """
    r = requests.get(GAMMA_URL, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError("Gamma API 返回空")
    event = data[0]
    markets = event.get("markets", [])
    if not markets:
        raise RuntimeError("event 无 markets")

    out = {}
    slugs = {}
    skipped = []
    for m in markets:
        if m.get("closed"):
            continue
        q = m.get("question", "")
        team = _parse_team(q)
        if not team:
            skipped.append(q)
            continue
        # outcomePrices 是字符串数组 ["Yes_prob", "No_prob"]
        prices_str = m.get("outcomePrices")
        if not prices_str:
            continue
        try:
            prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
            yes_prob = float(prices[0])
        except (ValueError, IndexError, TypeError):
            continue
        if yes_prob <= 0:
            continue
        out[team] = round(yes_prob * 100, 2)
        if m.get("slug"):
            slugs[team] = m["slug"]

    global _LAST_SLUGS
    _LAST_SLUGS = slugs

    if skipped:
        print(f"  ⚠ 跳过 {len(skipped)} 个无法解析的 question (前 3): {skipped[:3]}")
    return out


_LAST_SLUGS: dict = {}  # team -> polymarket market slug，由 fetch_polymarket_winner 填充


def write_to_external_predictions(preds: dict) -> Tuple[int, int]:
    """
    更新 external_predictions.json 的 models.polymarket.predictions
    返回 (n_changed, n_total)
    """
    if EXT_PATH.exists():
        data = json.load(open(EXT_PATH))
    else:
        data = {"models": {}}
    
    if "models" not in data:
        data["models"] = {}
    
    old = data["models"].get("polymarket", {}).get("predictions", {})
    
    # 统计变化
    n_changed = 0
    for t, p in preds.items():
        if abs(old.get(t, -999) - p) >= 0.5:  # 变化 ≥ 0.5pp 才算
            n_changed += 1
    
    data["models"]["polymarket"] = {
        "predictions": preds,
        "market_slugs": dict(_LAST_SLUGS),  # team -> slug，给前端深链入口用
        "event_url": "https://polymarket.com/event/world-cup-winner",
        "_meta": {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_url": "https://gamma-api.polymarket.com/events?slug=world-cup-winner",
            "method": "gamma_api",
            "n_teams": len(preds),
        }
    }
    
    EXT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return n_changed, len(preds)


def sync_to_teams_json(preds: dict) -> int:
    """把 Polymarket 实时夺冠概率回写到 teams.json[*].market_implied。

    设计：
      - preds 是百分数（如 Spain=16.95），teams.json 用小数（0.1695）
      - 仅在变化 ≥0.05pp（=0.0005）时写，避免频繁 IO
      - 完全吞异常（feedback_no_break_existing），失败不影响 external_predictions 写入
      - synthesizer 已优先从 teams.json[*].market_implied 取最新市场价
        （synthesizer.py:167-171），所以这步是把"陈旧 snapshot"→"实时"的关键链路

    Returns:
        实际更新的队数（0 表示无变化或失败）
    """
    teams_path = DATA_RAW / "teams.json"
    if not teams_path.exists():
        return 0
    try:
        data = json.load(open(teams_path))
        teams = data.get("teams", {})
        n_updated = 0
        for team_name, pct in preds.items():
            if team_name not in teams:
                continue
            new_val = round(pct / 100.0, 4)  # 16.95% → 0.1695
            old_val = teams[team_name].get("market_implied")
            if old_val is None or abs((old_val or 0) - new_val) >= 0.0005:
                teams[team_name]["market_implied"] = new_val
                n_updated += 1
        if n_updated > 0:
            # 顺手更新 _last_updated.<team>.market_implied 元数据
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            last_updated = data.setdefault("_last_updated", {})
            for team_name in preds:
                if team_name in teams:
                    last_updated.setdefault(team_name, {})["market_implied"] = now_str
            teams_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return n_updated
    except Exception as e:
        print(f"⚠ 回写 teams.json 失败（不影响主流程）: {type(e).__name__}: {e}")
        return 0


def main():
    print("=== Polymarket Winner 抓取 ===")
    try:
        preds = fetch_polymarket_winner()
    except Exception as e:
        print(f"✗ 抓取失败: {e}")
        sys.exit(1)
    
    print(f"✓ 抓到 {len(preds)} 队隐含概率")
    
    # 显示 Top 10
    top = sorted(preds.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10:")
    for t, p in top:
        print(f"  {t:<22s} {p:>5.2f}%")
    
    n_chg, n_tot = write_to_external_predictions(preds)
    print(f"\n写入 {EXT_PATH.name}: {n_tot} 队 (变化 ≥0.5pp 的 {n_chg} 队)")

    # 回写到 teams.json[*].market_implied，让 synthesizer/arbitrage 用实时市场价
    n_synced = sync_to_teams_json(preds)
    print(f"回写 teams.json: 更新 {n_synced} 队的 market_implied")


if __name__ == "__main__":
    main()
