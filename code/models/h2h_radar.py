"""
双方近 5 场对阵（H2H）雷达
============================
对单对阵抓双方最近 5 次交手记录：
  1. Serper 搜 "<Team A> vs <Team B> head to head last 5 matches"
  2. LLM 把摘要解析成结构化 {date, competition, score, winner}
  3. 写入 data/raw/h2h.json（key = "A vs B @ YYYY-MM-DD"）

设计参考 lineup_radar.py：同样 Serper+LLM 模板、同样降级、同样 retry。
"""
import os
import sys
import json
import time
import urllib.request
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import get_default_client, LLMUnavailable
from llm.client import LLMClient

logger = logging.getLogger("worldcup.h2h_radar")
SERPER_URL = "https://google.serper.dev/search"


def _bootstrap_env():
    for var_name in ("SERPER_API_KEY", "LINGYA_API_KEY", "WORLDCUP_LLM_MODEL"):
        if os.environ.get(var_name):
            continue
        try:
            r = subprocess.run(["launchctl", "getenv", var_name],
                               capture_output=True, text=True, timeout=2)
            v = r.stdout.strip()
            if v:
                os.environ[var_name] = v
        except Exception:
            pass


_bootstrap_env()


def serper_search(query: str, max_results: int = 6, timeout: int = 15,
                   retry: int = 2) -> List[Dict[str, str]]:
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        logger.warning("SERPER_API_KEY 未设置")
        return []
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = json.dumps({"q": query, "num": max_results}).encode()
    req = urllib.request.Request(SERPER_URL, data=payload, headers=headers, method="POST")
    for attempt in range(retry + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                results = data.get("organic", [])
                return [
                    {"title": r.get("title", ""), "url": r.get("link", ""),
                     "snippet": r.get("snippet", ""), "source": r.get("source", ""),
                     "date": r.get("date", "")}
                    for r in results[:max_results]
                ]
        except Exception as e:
            if attempt < retry:
                time.sleep(2)
            else:
                logger.warning(f"Serper 失败 '{query}': {e}")
                return []
    return []


def build_search_summary(team_a: str, team_b: str) -> str:
    queries = [
        f"{team_a} vs {team_b} head to head last 5 matches",
        f"{team_a} {team_b} all time results history",
    ]
    sections = []
    for q in queries:
        rs = serper_search(q, max_results=6)
        if not rs:
            continue
        section = [f"## 查询：{q}"]
        for r in rs:
            title = r['title'][:120]
            snippet = (r['snippet'] or "")[:320]
            snippet = snippet.replace("\u00a0", " ").replace("·", "-")
            section.append(f"- {title}")
            if snippet:
                section.append(f"  {snippet}")
        sections.append("\n".join(section))
    return "\n\n".join(sections)


SYSTEM_PROMPT = """你是足球数据分析师。基于新闻搜索摘要，输出两支国家队**近 5 次正式比赛对阵**的结构化记录。

# 任务
读完所有摘要后，提取 team_a 与 team_b 之间最近 5 次正式比赛（World Cup / Continental Cup / 友谊赛 / 预选赛皆可，明确赛事名）的对阵记录。

# 严格要求
1. **必须 JSON 输出**，绝不空字符串、不拒答
2. 只输出摘要里明确出现的对阵；不要编造比分/日期
3. 没找到 5 场就少给几场，但每场必须 4 个字段齐全
4. winner 字段：team_a 的英文名 / team_b 的英文名 / "Draw"

# 字段定义
- matches: 数组，最多 5 个，按日期倒序
  - date: "YYYY-MM-DD"（如只知道年月，YYYY-MM-XX；只知年，YYYY-XX-XX）
  - competition: "World Cup 2014 group" / "Friendly" / "Copa America 2019 SF" 等
  - score: "2-1" 字符串（team_a 进球-team_b 进球）
  - winner: team_a 名 / team_b 名 / "Draw"
- summary_record: "team_a W-D-L" 形如 "Brazil 3-1-1"
- last_meeting_year: 最近一次交手年份（int），不确定填 null
- confidence: "高/中/低"
- notes: 不超过 80 字总结

# 输出 JSON Schema
{
  "matches": [
    {"date": "2022-12-09", "competition": "World Cup 2022 QF",
     "score": "1-1", "winner": "Draw"},
    ...
  ],
  "summary_record": "...",
  "last_meeting_year": 2022,
  "confidence": "中",
  "notes": "..."
}"""


def analyze_h2h(team_a: str, team_b: str,
                client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """对单对阵跑 H2H 雷达"""
    summary = build_search_summary(team_a, team_b)
    if len(summary) < 80:
        return {
            "team_a": team_a, "team_b": team_b,
            "matches": [], "summary_record": "", "last_meeting_year": None,
            "confidence": "低", "notes": "搜索结果不足",
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }

    try:
        c = client or get_default_client()
    except LLMUnavailable as e:
        return {"team_a": team_a, "team_b": team_b,
                "error": f"LLM 不可用: {e}",
                "raw_search_summary": summary,
                "checked_at": datetime.now().isoformat(timespec="seconds")}

    user_msg = f"""# Team A
{team_a}

# Team B
{team_b}

# 新闻摘要
{summary}

---
基于以上摘要，严格 JSON 输出 team_a 与 team_b 之间近 5 次对阵记录。"""

    data = None
    last_err = None
    attempts = [
        {"temp": 0.3, "client": c, "label": "primary T=0.3"},
        {"temp": 0.6, "client": c, "label": "primary T=0.6"},
        {"temp": 0.4, "client": None, "label": "fallback deepseek-v3", "fallback": True},
    ]
    for at in attempts:
        try:
            cur_client = at["client"]
            if at.get("fallback"):
                try:
                    cur_client = LLMClient(model="deepseek-v3")
                except Exception as e:
                    last_err = f"fallback 不可用: {e}"
                    continue
            text = cur_client.chat(system=SYSTEM_PROMPT, user=user_msg,
                                    json_mode=True, max_tokens=1200, temperature=at["temp"])
            data = LLMClient.extract_json(text)
            if data and "matches" in data:
                break
            last_err = f"[{at['label']}] 字段缺失: {str(data)[:80]}"
        except Exception as e:
            last_err = f"[{at['label']}] {e}"
            data = None

    if not data or "matches" not in data:
        logger.warning(f"{team_a} vs {team_b} LLM 失败: {last_err}")
        return {
            "team_a": team_a, "team_b": team_b,
            "error": f"LLM 重试失败: {last_err}",
            "raw_search_summary": summary[:1500],
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }

    data["team_a"] = team_a
    data["team_b"] = team_b
    data["checked_at"] = datetime.now().isoformat(timespec="seconds")
    data.setdefault("matches", [])
    data.setdefault("summary_record", "")
    data.setdefault("last_meeting_year", None)
    data.setdefault("confidence", "中")
    data.setdefault("notes", "")
    return data


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-a", required=True)
    ap.add_argument("--team-b", required=True)
    args = ap.parse_args()
    print(f"=== H2H 雷达：{args.team_a} vs {args.team_b} ===\n")
    r = analyze_h2h(args.team_a, args.team_b)
    print(json.dumps(r, ensure_ascii=False, indent=2))
