"""
裁判任命雷达
==============
对单场比赛抓裁判信息（赛前 24h 内 FIFA 通常会公布）：
  1. Serper 搜 "<TeamA> vs <TeamB> referee World Cup 2026"
  2. LLM 抽取：name / nationality / yellow_per_match / red_per_match / penalty_tendency

下游用途：高黄牌密度裁判 → 红牌/点球场景概率上调；某些 ref 偏主队等。
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

logger = logging.getLogger("worldcup.referee_radar")
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


def build_summary(team_a: str, team_b: str, date_str: str) -> str:
    queries = [
        f"{team_a} vs {team_b} referee World Cup 2026 {date_str}",
        f"{team_a} {team_b} FIFA match official appointed",
    ]
    parts = []
    for q in queries:
        rs = serper_search(q, max_results=5)
        if not rs:
            continue
        sec = [f"## 查询：{q}"]
        for r in rs:
            t = r['title'][:120]
            s = (r['snippet'] or "")[:280].replace("\u00a0", " ")
            sec.append(f"- {t}")
            if s:
                sec.append(f"  {s}")
        parts.append("\n".join(sec))
    return "\n\n".join(parts)


SYSTEM_PROMPT = """你是足球裁判数据分析师。基于新闻搜索摘要，抽取一场比赛的主裁信息及风格。

# 任务
对指定比赛输出主裁判的基本信息和判罚风格统计。

# 严格要求
1. 必须 JSON 输出；找不到主裁就 name = null，confidence = "低"
2. 数字只取摘要里明确出现的，不要编造
3. yellow/red/penalty 是"每场平均"

# 字段定义
- name: 主裁名（如 "Anthony Taylor"），找不到 null
- nationality: ISO 三位国家码或全名
- yellow_per_match: float 或 null
- red_per_match: float 或 null
- penalty_per_match: float 或 null
- style: 短描述，如 "strict / lenient / card-happy / VAR-active"
- recent_high_profile: 近期执法的大赛事，最多 3 条字符串
- confidence: "高/中/低"
- notes: ≤80 字

# 输出 JSON
{
  "name": "...",
  "nationality": "...",
  "yellow_per_match": 4.2,
  "red_per_match": 0.15,
  "penalty_per_match": 0.3,
  "style": "...",
  "recent_high_profile": [...],
  "confidence": "中",
  "notes": "..."
}"""


def analyze_referee(team_a: str, team_b: str, date_str: str,
                    client: Optional[LLMClient] = None) -> Dict[str, Any]:
    summary = build_summary(team_a, team_b, date_str)
    if len(summary) < 60:
        return {
            "team_a": team_a, "team_b": team_b, "date": date_str,
            "name": None, "nationality": None,
            "yellow_per_match": None, "red_per_match": None, "penalty_per_match": None,
            "style": "", "recent_high_profile": [],
            "confidence": "低", "notes": "搜索结果不足",
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }

    try:
        c = client or get_default_client()
    except LLMUnavailable as e:
        return {"team_a": team_a, "team_b": team_b, "date": date_str,
                "error": f"LLM 不可用: {e}",
                "raw_search_summary": summary,
                "checked_at": datetime.now().isoformat(timespec="seconds")}

    user_msg = (f"# 比赛\n{team_a} vs {team_b} ({date_str})\n\n"
                f"# 摘要\n{summary}\n\n---\n严格 JSON 输出主裁信息。")
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
            if data and "name" in data:
                break
            last_err = f"[{at['label']}] 字段缺失: {str(data)[:80]}"
        except Exception as e:
            last_err = f"[{at['label']}] {e}"
            data = None

    if not data or "name" not in data:
        logger.warning(f"{team_a} vs {team_b} referee LLM 失败: {last_err}")
        return {
            "team_a": team_a, "team_b": team_b, "date": date_str,
            "error": f"LLM 重试失败: {last_err}",
            "raw_search_summary": summary[:1500],
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }

    data["team_a"] = team_a
    data["team_b"] = team_b
    data["date"] = date_str
    data["checked_at"] = datetime.now().isoformat(timespec="seconds")
    data.setdefault("nationality", None)
    data.setdefault("yellow_per_match", None)
    data.setdefault("red_per_match", None)
    data.setdefault("penalty_per_match", None)
    data.setdefault("style", "")
    data.setdefault("recent_high_profile", [])
    data.setdefault("confidence", "中")
    data.setdefault("notes", "")
    return data


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-a", required=True)
    ap.add_argument("--team-b", required=True)
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    print(json.dumps(analyze_referee(args.team_a, args.team_b, args.date),
                     ensure_ascii=False, indent=2))
