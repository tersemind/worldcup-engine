"""
赛前首发阵容雷达（LLM + Serper）
=================================
对单场比赛的双方在赛前 2-3h 抓首发名单：
  1. Serper Google 搜 "<Team> starting lineup vs <Opponent> 2026-MM-DD"
  2. LLM 把新闻摘要解析成结构化 starters/bench/captain/formation

与 injury_radar 区别：
  - injury_radar 关注"谁缺阵"（轮换+伤）
  - lineup_radar 关注"实际上场 11 人"（比 injury 更确定，但要等赛前 2-3h 教练公布）

下游消费：可让 synthesizer 在 starter 缺失时进一步下调实力。
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
from llm.client import LLMClient, reset_default_client

logger = logging.getLogger("worldcup.lineup_radar")
SERPER_URL = "https://google.serper.dev/search"


def _bootstrap_env():
    """从 launchctl 桥接所有需要的 key/config"""
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
                    {
                        "title":   r.get("title", ""),
                        "url":     r.get("link", ""),
                        "snippet": r.get("snippet", ""),
                        "source":  r.get("source", ""),
                        "date":    r.get("date", ""),
                    }
                    for r in results[:max_results]
                ]
        except Exception as e:
            if attempt < retry:
                time.sleep(2)
            else:
                logger.warning(f"Serper 失败 '{query}': {e}")
                return []
    return []


def build_search_summary(team: str, opponent: str, date_str: str) -> str:
    """对单队跑 2 个查询拼成简报（精简版，避免触发 LLM 过滤）"""
    queries = [
        f"{team} starting lineup vs {opponent} {date_str}",
        f"{team} predicted XI World Cup {date_str}",
    ]
    sections = []
    for q in queries:
        rs = serper_search(q, max_results=5)
        if not rs:
            continue
        section = [f"## 查询：{q}"]
        for r in rs:
            title = r['title'][:120]
            snippet = (r['snippet'] or "")[:280]
            # 去掉可能触发过滤的特殊字符
            snippet = snippet.replace("\u00a0", " ").replace("·", "-")
            section.append(f"- {title}")
            if snippet:
                section.append(f"  {snippet}")
        sections.append("\n".join(section))
    return "\n\n".join(sections)


SYSTEM_PROMPT = """你是足球战术分析师。基于新闻搜索摘要，输出某球队**最可能的首发 11 人**。

# 任务
读完所有摘要后，提取该球队在指定比赛中最可能的首发 11 人（starters）+ 可能轮换的 3-5 个替补（bench_key）。

# 严格要求
1. **必须 JSON 输出**，绝不空字符串、不拒答
2. **优先引用明确出现在摘要里的球员名**；摘要不够时可基于历史首发推断，但 confirmed 字段须如实标
3. starters 必须正好 11 人；若摘要不足以推 11 人，缺位用 "?" 占位
4. **位置** 用四大类: GK / DEF / MID / FWD（不要用具体位）

# 字段定义
- starters: 11 人列表
  - player: 球员名
  - position: GK / DEF / MID / FWD
  - is_captain: bool
- bench_key: 关键替补 3-5 人
  - player, position
- formation: 阵型字符串（如 "4-3-3", "4-2-3-1"），不确定填 ""
- confirmed: bool — 是否教练官方公布；推断的填 false
- key_changes_vs_previous: 与上场比相比的关键变动列表（如 "starter Pedri replaced by Olmo (rotation)"）
- confidence: "高/中/低"
- notes: 不超过 80 字总结

# 输出 JSON Schema
{
  "starters": [{"player": "...", "position": "...", "is_captain": false}, ...],
  "bench_key": [{"player": "...", "position": "..."}, ...],
  "formation": "4-3-3",
  "confirmed": false,
  "key_changes_vs_previous": [],
  "confidence": "中",
  "notes": "..."
}"""


def analyze_team_lineup(team: str, opponent: str, date_str: str,
                         client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """对单队跑首发雷达"""
    summary = build_search_summary(team, opponent, date_str)
    if len(summary) < 80:
        return {
            "team": team,
            "starters": [], "bench_key": [], "formation": "",
            "confirmed": False, "confidence": "低",
            "notes": "搜索结果不足",
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
    
    try:
        c = client or get_default_client()
    except LLMUnavailable as e:
        return {"team": team, "error": f"LLM 不可用: {e}",
                "raw_search_summary": summary,
                "checked_at": datetime.now().isoformat(timespec="seconds")}
    
    user_msg = f"""# 球队
{team}

# 对手
{opponent}

# 比赛日期
{date_str}

# 新闻摘要
{summary}

---
基于以上摘要，严格 JSON 输出该队首发 11 人。"""
    
    # 2 次重试（覆盖 LLM 偶发空响应 / 字段缺失），第 3 次切 deepseek-v3
    data = None
    last_err = None
    attempts = [
        {"temp": 0.3, "client": c, "label": "primary T=0.3"},
        {"temp": 0.7, "client": c, "label": "primary T=0.7"},
        {"temp": 0.4, "client": None, "label": "fallback deepseek-v3", "fallback": True},
    ]
    for at in attempts:
        try:
            cur_client = at["client"]
            if at.get("fallback"):
                # 强制切 deepseek-v3（记忆里记录: Lingya v4-flash-max 并发偶发空 JSON）
                # 直接构造 client（不走单例，避免 env 切换不生效）
                try:
                    cur_client = LLMClient(model="deepseek-v3")
                except Exception as e:
                    last_err = f"fallback 不可用: {e}"
                    continue
            text = cur_client.chat(system=SYSTEM_PROMPT, user=user_msg,
                                    json_mode=True, max_tokens=1500, temperature=at["temp"])
            data = LLMClient.extract_json(text)
            if data and "starters" in data:
                break
            last_err = f"[{at['label']}] 字段缺失: {str(data)[:80]}"
        except Exception as e:
            last_err = f"[{at['label']}] {e}"
            data = None
    
    if not data or "starters" not in data:
        logger.warning(f"{team} LLM 失败: {last_err}")
        return {
            "team": team, "error": f"LLM 重试失败: {last_err}",
            "raw_search_summary": summary[:1500],
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
    
    # 校验 + 整理
    data["team"] = team
    data["checked_at"] = datetime.now().isoformat(timespec="seconds")
    # 标准化字段
    data.setdefault("starters", [])
    data.setdefault("bench_key", [])
    data.setdefault("formation", "")
    data.setdefault("confirmed", False)
    data.setdefault("key_changes_vs_previous", [])
    data.setdefault("confidence", "中")
    data.setdefault("notes", "")
    return data


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    print(f"=== 首发雷达：{args.team} vs {args.opponent} ({args.date}) ===\n")
    r = analyze_team_lineup(args.team, args.opponent, args.date)
    print(json.dumps(r, ensure_ascii=False, indent=2))
