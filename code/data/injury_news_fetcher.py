"""
伤病新闻自动抓取器（#5 P1 数据自动化）

工作流：
1. 从 RSS/News API 抓取头条
2. 用关键词过滤伤病新闻
3. 用规则匹配提取（球员/球队/状态）
4. 输出待 Claude 用 LLM 二次确认的清单

数据源：
- ESPN Soccer RSS: https://www.espn.com/espn/rss/soccer/news
- BBC Sport: https://feeds.bbci.co.uk/sport/football/rss.xml
- Skysports: 通过 WebSearch 触发

注：完整 LLM 解析需 API key，本模块输出"待解析队列"供 Claude 在 /wc-predict refresh injuries 时人工辅助处理
"""
import json
import re
import sys
import time
import warnings
import urllib.request
from pathlib import Path

warnings.filterwarnings("ignore", category=Warning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW
from data.injuries_fetcher import KEY_PLAYERS, STATUS_TO_ADJ


# 关键词 → 状态映射
INJURY_KEYWORDS = {
    "ruled out": "out_for_tournament",
    "out for the tournament": "out_for_tournament",
    "out for the world cup": "out_for_tournament",
    "miss the world cup": "out_for_tournament",
    "withdrawn": "out_for_tournament",
    "season-ending": "out_for_tournament",
    "torn acl": "out_for_tournament",
    "acl injury": "out_for_tournament",
    "acl tear": "out_for_tournament",
    
    "miss the rest": "out_for_knockouts",
    "knockout stage": "out_for_knockouts",
    "out for weeks": "out_for_knockouts",
    
    "doubtful": "uncertain",
    "uncertain": "uncertain",
    "could miss": "uncertain",
    "may miss": "uncertain",
    "race against time": "uncertain",
    
    "fitness concern": "80_percent",
    "managing injury": "80_percent",
    "tight": "80_percent",
    "limited training": "80_percent",
    
    "minor knock": "90_percent",
    "should be fit": "90_percent",
    "expected to be fit": "90_percent",
    
    "fully fit": "fit",
    "back in training": "fit",
    "available": "fit",
    "cleared to play": "fit",
}


# 球员名 → (球队, 标准化全名)
def build_player_lookup():
    """从 KEY_PLAYERS 构建反向索引"""
    lookup = {}
    for team, players in KEY_PLAYERS.items():
        for p in players:
            full_name = p["name"]
            # 全名 + 姓氏单独
            lookup[full_name.lower()] = (team, full_name)
            last_name = full_name.split()[-1].lower()
            if last_name not in lookup:  # 避免冲突（如 'Silva'）
                lookup[last_name] = (team, full_name)
    return lookup


def fetch_rss(url: str, timeout: int = 10) -> str:
    """简易 RSS 抓取（不依赖 feedparser）"""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  ⚠️  抓取失败: {url} ({e})")
        return ""


def parse_rss_titles(xml_text: str, max_items: int = 50) -> list:
    """从 RSS XML 提取标题（粗暴 regex 解析）"""
    titles = re.findall(r"<title[^>]*>(.*?)</title>", xml_text, re.DOTALL | re.IGNORECASE)
    # 去掉 CDATA 标记
    cleaned = []
    for t in titles[:max_items]:
        t = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", t).strip()
        if t and len(t) > 5:
            cleaned.append(t)
    return cleaned


def detect_injury_status(text: str) -> str:
    """从文本检测伤病状态关键词"""
    text_lower = text.lower()
    for keyword, status in INJURY_KEYWORDS.items():
        if keyword in text_lower:
            return status
    return None


def detect_player(text: str, lookup: dict) -> tuple:
    """从文本检测涉及的关键球员"""
    text_lower = text.lower()
    for name_key, (team, full_name) in lookup.items():
        # 至少 4 个字符的关键词，避免 "lee"/"li" 误匹配
        if len(name_key) >= 4 and name_key in text_lower:
            return (team, full_name)
    return (None, None)


def scan_news_titles(titles: list, lookup: dict) -> list:
    """扫描标题，返回检测到的伤情清单"""
    matches = []
    for title in titles:
        player_team, player_name = detect_player(title, lookup)
        if not player_name:
            continue
        status = detect_injury_status(title)
        if not status:
            continue
        matches.append({
            "title": title,
            "team": player_team,
            "player": player_name,
            "detected_status": status,
            "adj_pp": STATUS_TO_ADJ.get(status, 0.0),
        })
    return matches


def fetch_all_sources():
    """从多个 RSS 源抓取标题"""
    sources = {
        "ESPN Soccer": "https://www.espn.com/espn/rss/soccer/news",
        "BBC Football": "https://feeds.bbci.co.uk/sport/football/rss.xml",
        "Sky Sports Football": "https://www.skysports.com/rss/12040",
    }
    
    all_titles = []
    for source_name, url in sources.items():
        print(f"  📡 抓取 {source_name}...")
        xml = fetch_rss(url)
        if xml:
            titles = parse_rss_titles(xml)
            all_titles.extend([(source_name, t) for t in titles])
            print(f"     抓到 {len(titles)} 条")
    
    return all_titles


def main():
    print("=" * 70)
    print("🏥 Injury News Fetcher v1.0 — RSS 自动扫描")
    print("=" * 70)
    
    lookup = build_player_lookup()
    print(f"\n📋 监测球员清单: {len(set(t for t, _ in lookup.values()))} 队 × {len(lookup)} 项关键词")
    
    print(f"\n📡 抓取 RSS 源...")
    all_titles = fetch_all_sources()
    print(f"\n📊 共抓到 {len(all_titles)} 条标题")
    
    # 扫描
    print(f"\n🔍 扫描伤情关键词...")
    matches = []
    for source, title in all_titles:
        m = scan_news_titles([title], lookup)
        if m:
            for hit in m:
                hit["source"] = source
                matches.append(hit)
    
    if not matches:
        print(f"\n✅ 未检测到关键球员的伤情新闻")
        # 输出建议供 Claude
        print(f"\n💡 建议 Claude 用 WebSearch 主动搜索：")
        for team in ["Spain", "France", "Argentina", "England", "Brazil"]:
            players = KEY_PLAYERS.get(team, [])
            for p in players[:1]:
                print(f"   WebSearch: \"{p['name']} injury 2026 World Cup June\"")
        return
    
    print(f"\n🎯 检测到 {len(matches)} 条潜在伤情：\n")
    print(f"  {'Team':<14} {'Player':<22} {'Status':<22} {'Adj':<8}")
    print("  " + "-" * 75)
    for m in matches[:20]:
        print(f"  {m['team']:<14} {m['player']:<22} {m['detected_status']:<22} {m['adj_pp']:>+5.2f}pp")
        print(f"     {m['source']}: {m['title'][:80]}")
    
    # 保存到队列
    queue_file = DATA_RAW / "injury_queue.json"
    with open(queue_file, "w") as f:
        json.dump({
            "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_matches": len(matches),
            "matches": matches,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 已保存到队列: {queue_file}")
    print(f"💡 下一步：用 Claude 审核每条匹配，确认后调用：")
    print(f"   python3 injuries_fetcher.py update <team> <player> <status>")


if __name__ == "__main__":
    main()
