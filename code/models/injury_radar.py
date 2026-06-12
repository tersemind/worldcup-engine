"""
LLM 伤病雷达 Agent
==================
对单支球队跑：
  1. DuckDuckGo HTML 搜索（无需 key）抓 3-5 个查询的结果摘要
  2. LLM 解析摘要 → 结构化伤病信息
  3. 输出 {player, status, pp_impact, source} 列表

DuckDuckGo HTML 端点：https://html.duckduckgo.com/html/?q=<query>
返回纯 HTML（无需 JS），用 BeautifulSoup 或简单正则提取结果摘要。

输入：team_name, opponent (可选, 用于查"vs Y" 类型上下文), date
输出：结构化伤病字典（写入 injuries.json）
"""
from __future__ import annotations
import sys
import os
import json
import time
import urllib.request
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import get_default_client, LLMUnavailable
from llm.client import LLMClient

logger = logging.getLogger("worldcup.injury_radar")

# Serper Google web search
SERPER_URL = "https://google.serper.dev/search"


def _bootstrap_env():
    """从 launchctl 桥接所有需要的 key/config（macOS launchctl setenv 不自动传播）"""
    import subprocess
    for var_name in ("SERPER_API_KEY", "LINGYA_API_KEY", "WORLDCUP_LLM_MODEL"):
        if os.environ.get(var_name):
            continue
        try:
            result = subprocess.run(
                ["launchctl", "getenv", var_name],
                capture_output=True, text=True, timeout=2,
            )
            val = result.stdout.strip()
            if val:
                os.environ[var_name] = val
        except Exception:
            pass


_bootstrap_env()


# ============ 搜索工具 ============
def serper_search(query: str, max_results: int = 8, timeout: int = 15,
                    retry: int = 2) -> List[Dict[str, str]]:
    """
    Serper API（Google web search 代理）
    
    返回 [{title, url, snippet, source, date}, ...]
    
    免费额度 2500/月。无 key 或失败时返回空列表（调用方负责处理）。
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        logger.warning("SERPER_API_KEY 未设置")
        return []
    
    # 排除预测市场自身（避免循环引用）
    full_query = f"{query} -site:polymarket.com -site:kalshi.com"
    
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = json.dumps({"q": full_query, "num": max_results}).encode()
    req = urllib.request.Request(SERPER_URL, data=payload, headers=headers, method="POST")
    
    for attempt in range(retry + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                results = data.get("organic", [])
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "snippet": r.get("snippet", ""),
                        "source": r.get("source", ""),
                        "date": r.get("date", ""),
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


def build_search_summary(team: str, opponent: Optional[str], date_str: str) -> str:
    """对单队跑 2 个查询，汇总成给 LLM 的新闻简报"""
    queries = [
        f"{team} national team injury news {date_str}",
        f"{team} starting lineup vs {opponent}" if opponent else f"{team} world cup 2026 squad",
    ]
    
    sections = []
    for q in queries:
        results = serper_search(q, max_results=5)
        if not results:
            sections.append(f"### 查询「{q}」: 无结果\n")
            continue
        section = [f"### 查询「{q}」:"]
        for r in results:
            src = r.get("source", "")
            src_tag = f"[{src}]" if src else ""
            section.append(f"- **{src_tag}{r['title']}** ({r['url'][:80]})")
            if r['snippet']:
                section.append(f"  {r['snippet'][:300]}")
            if r.get('date'):
                section.append(f"  日期: {r['date']}")
        sections.append("\n".join(section))
    
    return "\n\n".join(sections)


# ============ LLM 解析 ============
SYSTEM_PROMPT = """你是足球伤病情报分析师。基于新闻搜索摘要，输出某球队的关键球员伤病/缺阵情况。

# 任务
读完所有摘要后，提取该球队**赛前 24-48h 内确实有伤病/缺阵情况**的关键球员。

# 严格要求
1. **必须 JSON 输出**，绝不空字符串、不拒答
2. **只引用明确出现在搜索摘要里的球员名 + 事实**，禁止编造
3. 如摘要里完全没有伤病相关信息 → 返回 `absences: []` 和 `total_pp_impact: 0` 即可
4. **不要把"全员可用"等正面信息当作伤病**

# 字段定义
- player: 球员名（英文）
- position: 大致位置（如 attacking midfielder, defender, goalkeeper, forward, midfielder）
- status: confirmed_out（确认缺阵）/ doubtful（疑问）/ fit（仍可上场）
- importance: high（首发核心）/ medium（轮换主力）/ low（替补）
- pp_impact: 单人对夺冠概率的影响（参考：核心 -1.0 ~ -2.0；中等 -0.3 ~ -0.8；替补 0 ~ -0.2）
- rationale: 不超过 50 字，说明伤病/缺阵原因
- source_url: 第一个明确报道此事的 URL

# 输出 JSON Schema
{
  "absences": [
    {"player": "...", "position": "...", "status": "...", "importance": "...",
     "pp_impact": -1.0, "rationale": "...", "source_url": "..."}
  ],
  "total_pp_impact": -1.0,
  "confidence": "高/中/低",
  "notes": "<总结性观察, 不超过 80 字>"
}"""


def analyze_team_injuries(team: str, opponent: Optional[str] = None,
                           date_str: Optional[str] = None,
                           client: Optional[LLMClient] = None) -> Dict[str, Any]:
    """对单队跑伤病雷达：DDG 搜索 + LLM 解析"""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    # 1. 抓搜索摘要
    summary = build_search_summary(team, opponent, date_str)
    if len(summary) < 100:
        logger.warning(f"{team}: 搜索结果过少，跳过")
        return {
            "team": team,
            "absences": [],
            "total_pp_impact": 0,
            "confidence": "低",
            "notes": "搜索无结果",
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
    
    # 2. LLM 解析
    try:
        c = client or get_default_client()
    except LLMUnavailable as e:
        return {
            "team": team, "error": f"LLM 不可用: {e}",
            "raw_search_summary": summary,    # 即使 LLM 挂了也保留搜索结果供人工核查
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
    
    user_msg = f"""# 球队
{team}

# 检查日期
{date_str}

{f"# 下一场对手: {opponent}" if opponent else ""}

# 搜索结果汇总
{summary}

# 任务
基于以上搜索结果，输出 {team} 队当前关键球员伤病/缺阵情况。严格 JSON。"""
    
    # 3 步尝试：primary T=0.3 → primary T=0.7 → fallback deepseek-v3 T=0.4
    # 记忆: Lingya v4-flash-max 并发场景偶发空 JSON，切 deepseek-v3 可救活
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
                # 直接构造一个 deepseek-v3 client（绕开 get_default_client 的单例缓存）
                try:
                    cur_client = LLMClient(model="deepseek-v3")
                except Exception as e:
                    last_err = f"[fallback] 不可用: {e}"
                    continue
            text = cur_client.chat(system=SYSTEM_PROMPT, user=user_msg,
                                    json_mode=True, max_tokens=1200, temperature=at["temp"])
            data = LLMClient.extract_json(text)
            if data and "absences" in data:
                break
            last_err = f"[{at['label']}] 字段缺失: {str(data)[:80]}"
        except Exception as e:
            last_err = f"[{at['label']}] {e}"
            data = None
    
    if not data or "absences" not in data:
        logger.warning(f"{team} LLM 失败: {last_err}")
        return {
            "team": team, "error": f"LLM 重试失败: {last_err}",
            "raw_search_summary": summary,  # 保留搜索结果供人工核查
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
    
    # 校验 + 整理
    absences = data.get("absences", []) or []
    # clip pp_impact 到合理范围
    for a in absences:
        try:
            a["pp_impact"] = max(-3.0, min(0.0, float(a.get("pp_impact", 0))))
        except (TypeError, ValueError):
            a["pp_impact"] = 0.0
    
    total_pp = round(sum(a.get("pp_impact", 0) for a in absences), 2)
    
    return {
        "team": team,
        "absences": absences,
        "total_pp_impact": total_pp,
        "confidence": data.get("confidence", "中"),
        "notes": data.get("notes", "")[:200],
        "checked_at": datetime.now().isoformat(timespec="seconds"),
    }


# ============ CLI 自测 ============
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--opponent", default=None)
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    
    print(f"=== 伤病雷达：{args.team} ===")
    if args.opponent:
        print(f"对手: {args.opponent}")
    print(f"日期: {args.date or 'today'}")
    print()
    print("跑 WebSearch + LLM...")
    
    result = analyze_team_injuries(args.team, args.opponent, args.date)
    print()
    print(json.dumps(result, ensure_ascii=False, indent=2))
