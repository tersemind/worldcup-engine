"""
阵容市值雷达（Transfermarkt via Serper + LLM）
==================================================
对单队抓 Transfermarkt 报告的市值总和：
  1. Serper 搜 "<Team> national team squad market value Transfermarkt 2026"
  2. LLM 从摘要里抽 total_value_eur / top_3_players / source_url

直接 scrape Transfermarkt 容易 429，所以走"搜索 + LLM 抽数"路线（同 lineup_radar 模板）。
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

logger = logging.getLogger("worldcup.squad_value_radar")
SERPER_URL = "https://google.serper.dev/search"


def _bootstrap_env():
    for v in ("SERPER_API_KEY", "LINGYA_API_KEY", "WORLDCUP_LLM_MODEL"):
        if os.environ.get(v):
            continue
        try:
            r = subprocess.run(["launchctl", "getenv", v],
                               capture_output=True, text=True, timeout=2)
            val = r.stdout.strip()
            if val:
                os.environ[v] = val
        except Exception:
            pass


_bootstrap_env()


def serper_search(query: str, max_results: int = 6, timeout: int = 15,
                   retry: int = 2) -> List[Dict[str, str]]:
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return []
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = json.dumps({"q": query, "num": max_results}).encode()
    req = urllib.request.Request(SERPER_URL, data=payload, headers=headers, method="POST")
    for attempt in range(retry + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                return [
                    {"title": r.get("title", ""), "url": r.get("link", ""),
                     "snippet": r.get("snippet", "")}
                    for r in data.get("organic", [])[:max_results]
                ]
        except Exception as e:
            if attempt < retry:
                time.sleep(2)
            else:
                logger.warning(f"Serper 失败 '{query}': {e}")
                return []
    return []


def build_summary(team: str, extra_query: Optional[str] = None) -> str:
    """3 路 query 拼成简报：
       1) site:transfermarkt.com 直锁源站
       2) site:en.wikipedia.org 国家队主页通常含 TM 引用
       3) 通用 Google 搜兜底（含 ESPN / FBref / 二手引用）
    """
    queries = [
        f"{team} national football team squad market value site:transfermarkt.com",
        f"{team} national football team squad value site:en.wikipedia.org",
        f"{team} men's national team market value euros 2025",
    ]
    if extra_query:
        queries.append(extra_query)
    parts = []
    for q in queries:
        rs = serper_search(q, max_results=5)
        if not rs:
            continue
        section = [f"## 查询：{q}"]
        for r in rs:
            t = r['title'][:120]
            s = (r['snippet'] or "")[:320].replace("\u00a0", " ")
            section.append(f"- {t}")
            if s:
                section.append(f"  {s}")
        parts.append("\n".join(section))
    return "\n\n".join(parts)


SYSTEM_PROMPT = """你是足球数据分析师。基于 Transfermarkt 相关新闻搜索摘要，抽取某国家队的阵容总市值。

# 任务
输出该国家队当前国家队阵容（national team squad）的市值（按 Transfermarkt 数据）。

# 严格要求
1. 必须 JSON 输出；找不到就 total_value_eur = null，confidence = "低"
2. 单位统一欧元（EUR）。"€1.2B" → 1_200_000_000；"€850M" → 850_000_000
3. 只取摘要中明确出现的数字，不要编造
4. top_players 最多 3 人，每人含 name + value_eur（如有）

# 字段定义
- total_value_eur: int 或 null
- top_players: [{"name": str, "value_eur": int|null}]
- as_of: 摘要中提到的日期/时间窗口，如 "Nov 2025"，没有就 ""
- source_url: 主要引用的 URL（首选 transfermarkt.* 域名）
- confidence: "高/中/低"
- notes: ≤60 字

# 输出 JSON
{
  "total_value_eur": 1200000000,
  "top_players": [{"name": "Vinícius Jr.", "value_eur": 200000000}, ...],
  "as_of": "Nov 2025",
  "source_url": "https://www.transfermarkt.com/...",
  "confidence": "中",
  "notes": "..."
}"""


def analyze_squad_value(team: str, client: Optional[LLMClient] = None) -> Dict[str, Any]:
    summary = build_summary(team)
    if len(summary) < 60:
        return {
            "team": team, "total_value_eur": None, "top_players": [],
            "as_of": "", "source_url": "", "confidence": "低",
            "notes": "搜索结果不足",
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }

    try:
        c = client or get_default_client()
    except LLMUnavailable as e:
        return {"team": team, "error": f"LLM 不可用: {e}",
                "raw_search_summary": summary,
                "checked_at": datetime.now().isoformat(timespec="seconds")}

    user_msg = f"# 球队\n{team}\n\n# 摘要\n{summary}\n\n---\n严格 JSON 输出阵容市值。"
    data = None
    last_err = None
    for at in [
        {"temp": 0.3, "client": c, "label": "primary T=0.3"},
        {"temp": 0.6, "client": c, "label": "primary T=0.6"},
        {"temp": 0.4, "client": None, "label": "fallback deepseek-v3", "fallback": True},
    ]:
        try:
            cur = at["client"]
            if at.get("fallback"):
                try:
                    cur = LLMClient(model="deepseek-v3")
                except Exception as e:
                    last_err = f"fallback 不可用: {e}"
                    continue
            text = cur.chat(system=SYSTEM_PROMPT, user=user_msg,
                             json_mode=True, max_tokens=600, temperature=at["temp"])
            data = LLMClient.extract_json(text)
            if data and "total_value_eur" in data:
                break
            last_err = f"[{at['label']}] 字段缺失: {str(data)[:80]}"
        except Exception as e:
            last_err = f"[{at['label']}] {e}"
            data = None

    if not data or "total_value_eur" not in data:
        logger.warning(f"{team} squad value LLM 失败: {last_err}")
        return {
            "team": team, "error": f"LLM 重试失败: {last_err}",
            "raw_search_summary": summary[:1500],
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }

    # 兜底：LLM 答 null 时换一组更具体的 query 再试一次
    if data.get("total_value_eur") is None:
        logger.info(f"{team}: LLM 答 null，换查询重试")
        extra_q = f"{team} football squad value November 2025 transfermarkt"
        summary2 = build_summary(team, extra_query=extra_q)
        if summary2 and summary2 != summary:
            try:
                text2 = c.chat(system=SYSTEM_PROMPT,
                                user=f"# 球队\n{team}\n\n# 摘要\n{summary2}\n\n---\n严格 JSON 输出阵容市值。",
                                json_mode=True, max_tokens=600, temperature=0.4)
                data2 = LLMClient.extract_json(text2)
                if data2 and data2.get("total_value_eur"):
                    data = data2  # 用新结果覆盖
            except Exception as e:
                logger.warning(f"{team} 二次重试失败: {e}")

    data["team"] = team
    data["checked_at"] = datetime.now().isoformat(timespec="seconds")
    data.setdefault("top_players", [])
    data.setdefault("as_of", "")
    data.setdefault("source_url", "")
    data.setdefault("confidence", "中")
    data.setdefault("notes", "")
    return data


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    args = ap.parse_args()
    print(json.dumps(analyze_squad_value(args.team), ensure_ascii=False, indent=2))
