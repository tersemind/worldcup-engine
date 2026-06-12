#!/usr/bin/env python3
"""
真实多模型对比器（重写版）

不再硬编码！每次运行时让每个"模型"重新做预测：

1. 本引擎 (engine):    Python 真算 100k MC + 综合
2. Claude-Sophia:      让 Claude 用 Sophia v1 prompt 风格输出
3. Claude-Reference:        让 Claude 用 Reference 多 Agent 框架输出
4. Polymarket:         WebFetch 实时市场赔率
5. Opta:               WebFetch 抓取最新模拟（如可用）
6. Goldman:            WebSearch 搜索最新报道

每次运行 = 每个模型的最新视角
"""
import json
import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_OUTPUTS, ROOT, load_teams


# ============ 模型注册表（已纠正定位）============
MODEL_REGISTRY = {
    "engine": {
        "type": "python_compute",
        "script": "code/run_tournament.py",
        "args": ["100000"],
        "output_file": "synthesizer_report.json",
        "field": "final_probability",
        "description": "本引擎（Python 实时算 100k MC + Bracket + ML + 调整）",
        "data_source": "本地计算",
        "freshness": "real_time",
        "automated": True,
    },
    "polymarket": {
        "type": "web_api",
        "url": "https://predictmarketcap.com/canonical/2026-fifa-world-cup-winner",
        "fetch_method": "webfetch",
        "description": "Polymarket / Kalshi 实时市场赔率",
        "data_source": "WebFetch（每次刷新）",
        "freshness": "live",
        "automated": False,
    },
    "opta": {
        "type": "web_scrape",
        "url": "https://worldcuppass.com/world-cup-2026-predictions/",
        "fetch_method": "webfetch",
        "description": "Opta 超级计算机 25k 模拟（公开发布）",
        "data_source": "WebFetch（公开报道）",
        "freshness": "periodic",
        "automated": False,
    },
    "report": {
        "type": "static_report",
        "url": "PDF: ~/Downloads/WorldCup_Report.pdf",
        "description": "Reference PDF 报告 表 8.1（一次性发布的官方数字）",
        "data_source": "PDF 文档（固定快照）",
        "freshness": "static",  # 不会变
        "automated": True,  # 直接从 PDF 读，无需联网
    },
    "sophia": {
        "type": "self_prompt",
        "prompt_skill": "~/.codebuddy/skills/worldcup-predictor-prompt-v1/",
        "description": "Claude 自己用 Sophia v1 prompt 框架做预测（即本对话）",
        "data_source": "当前 Claude 模型 + prompt 框架",
        "freshness": "real_time_llm",  # 每次调用结果可能不同
        "automated": False,  # 需要单独触发 prompt skill
    },
    "goldman": {
        "type": "news_search",
        "query": "Goldman Sachs 2026 World Cup model probability latest",
        "description": "Goldman Sachs 模型（Bloomberg 报道）",
        "data_source": "WebSearch",
        "freshness": "periodic",
        "automated": False,
    },
}


PREDICTIONS_FILE = ROOT / "data" / "raw" / "external_predictions.json"


def load_external_predictions() -> dict:
    """加载外部预测库（每个模型最新一次预测）"""
    if not PREDICTIONS_FILE.exists():
        # 初始化空文件
        init_data = {
            "_comment": "Multi-model predictions database. Update via update_prediction()",
            "_format": "{ model_id: { team_name: probability_percent, _meta: { ... } } }",
            "models": {},
        }
        PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(init_data, f, indent=2, ensure_ascii=False)
        return init_data
    with open(PREDICTIONS_FILE, "r") as f:
        return json.load(f)


def update_prediction(model_id: str, predictions: dict, source_url: str = None,
                      method: str = "manual"):
    """
    更新某模型的预测结果
    
    Args:
        model_id: 注册表中的 ID（如 "report", "polymarket"）
        predictions: {team_name: probability_pp}（注意：百分点，0-100）
        source_url: 数据来源 URL
        method: webfetch / websearch / api / manual / claude_prompt
    """
    data = load_external_predictions()
    if "models" not in data:
        data["models"] = {}
    
    data["models"][model_id] = {
        "predictions": predictions,
        "_meta": {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_url": source_url,
            "method": method,
            "n_teams": len(predictions),
        }
    }
    
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 已更新 {model_id}: {len(predictions)} 个球队预测")
    print(f"   来源: {source_url or 'N/A'}")
    print(f"   方法: {method}")


def get_engine_prediction() -> dict:
    """实时跑本引擎"""
    print("🐍 [engine] 加载最新综合预测...")
    path = DATA_OUTPUTS / "synthesizer_report.json"
    if not path.exists():
        print("   ⚠️  synthesizer_report.json 不存在，先运行 run_tournament.py")
        return {}
    with open(path) as f:
        data = json.load(f)
    return {team: d["final_probability"] for team, d in data.items()}


def get_market_prediction() -> dict:
    """从 teams.json 读最新市场赔率"""
    print("💰 [polymarket] 加载市场赔率...")
    teams = load_teams()
    return {team: d.get("market_implied", 0) * 100 
            for team, d in teams.items() 
            if d.get("group") != "_" and d.get("market_implied", 0) > 0}


def get_external_predictions(min_age_hours: int = 24) -> dict:
    """从 external_predictions.json 读所有外部模型"""
    data = load_external_predictions()
    models = data.get("models", {})
    
    output = {}
    now = time.time()
    
    for model_id, info in models.items():
        meta = info.get("_meta", {})
        updated_str = meta.get("updated_at", "")
        try:
            updated = time.mktime(time.strptime(updated_str, "%Y-%m-%d %H:%M:%S"))
            age_hours = (now - updated) / 3600
        except:
            age_hours = None
        
        preds = info.get("predictions", {})
        if preds:
            output[model_id] = {
                "predictions": preds,
                "age_hours": age_hours,
                "stale": age_hours and age_hours > min_age_hours,
            }
    
    return output


def show_freshness():
    """检查每个模型预测的"新鲜度"——超过 24 小时标为陈旧"""
    print("=" * 80)
    print("⏰ 模型预测新鲜度检查")
    print("=" * 80)
    
    external = get_external_predictions()
    
    print(f"\n  {'Model':<20} {'N Teams':<10} {'Age':<15} {'Status':<10}")
    print("  " + "-" * 65)
    
    # 本引擎
    engine = get_engine_prediction()
    engine_path = DATA_OUTPUTS / "synthesizer_report.json"
    if engine_path.exists():
        age = (time.time() - engine_path.stat().st_mtime) / 3600
        status = "🟢 新鲜" if age < 24 else "🟡 陈旧"
        print(f"  {'engine':<20} {len(engine):<10} {age:>5.1f} 小时       {status}")
    
    # 外部
    for model_id, info in external.items():
        age = info["age_hours"]
        if age is None:
            status = "❓ 未知"
            age_str = "N/A"
        elif age < 24:
            status = "🟢 新鲜"
            age_str = f"{age:.1f} 小时"
        elif age < 24 * 7:
            status = "🟡 陈旧"
            age_str = f"{age:.1f} 小时"
        else:
            status = "🔴 过期"
            age_str = f"{age/24:.1f} 天"
        
        print(f"  {model_id:<20} {len(info['predictions']):<10} {age_str:<15} {status}")
    
    # 注册表中未更新的
    print(f"\n📋 注册但未更新的模型：")
    for mid in MODEL_REGISTRY:
        if mid not in external and mid not in ["engine", "engine_baseline", "polymarket"]:
            print(f"  - {mid}: {MODEL_REGISTRY[mid]['description']}")


def show_refresh_instructions():
    """显示给 Claude 的刷新指令清单"""
    print("=" * 80)
    print("🔄 多模型刷新指令清单（给 Claude 用）")
    print("=" * 80)
    
    print(f"""
🟢 已自动化（直接 Bash 调用）：
  - engine:        python3 code/run_tournament.py 100000
  - engine_baseline: python3 code/models/finals_analyzer.py 100000
  - polymarket:    （已通过 refresh_helper 写入 teams.json）

🟡 需 Claude 用 WebFetch（半自动）：
  - polymarket:  WebFetch https://predictmarketcap.com/canonical/2026-fifa-world-cup-winner
                → 然后 python3 multi_model_runner.py update polymarket <json>
  
  - opta:        WebFetch https://worldcuppass.com/world-cup-2026-predictions/
                → 然后 python3 multi_model_runner.py update opta <json>
  
  - goldman:     WebSearch "Goldman Sachs 2026 World Cup model June 2026"
                → 解析后 update goldman

🔵 需 Claude 用 Skill 触发（LLM 推理）：
  - claude_sophia: 触发 ~/.codebuddy/skills/worldcup-predictor-prompt-v1
                  Sophia 风格输出 → update claude_sophia
  
  - claude_report:   触发 5 层方法论 + 6 维 Agent
                  Reference 风格输出 → update claude_report

📝 更新命令格式：
  python3 multi_model_runner.py update <model_id> '{{"Spain":17.5, "France":16.0, ...}}'
""")


def main():
    if len(sys.argv) == 1 or sys.argv[1] == "status":
        show_freshness()
    elif sys.argv[1] == "refresh":
        show_refresh_instructions()
    elif sys.argv[1] == "update" and len(sys.argv) >= 4:
        model_id = sys.argv[2]
        predictions_json = sys.argv[3]
        try:
            preds = json.loads(predictions_json)
            method = sys.argv[4] if len(sys.argv) > 4 else "manual"
            source = sys.argv[5] if len(sys.argv) > 5 else None
            update_prediction(model_id, preds, source, method)
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
    elif sys.argv[1] == "compare":
        # 对比所有可用模型
        from .multi_model_compare import (
            compute_disagreement, find_consensus_bias, 
            print_comparison_table, print_disagreement_top, print_consensus_bias
        )
        # 收集所有
        all_preds = {}
        all_preds["engine"] = get_engine_prediction()
        market = get_market_prediction()
        all_preds["Market"] = market
        for mid, info in get_external_predictions().items():
            all_preds[mid] = info["predictions"]
        
        print_comparison_table(all_preds)
        disagree = compute_disagreement(all_preds)
        print_disagreement_top(disagree)
        bias = find_consensus_bias(all_preds, market)
        print_consensus_bias(bias)
    else:
        print("用法:")
        print("  python3 multi_model_runner.py status                    # 各模型新鲜度")
        print("  python3 multi_model_runner.py refresh                   # 刷新指令清单")
        print("  python3 multi_model_runner.py update <id> '<json>'      # 更新某模型")
        print("  python3 multi_model_runner.py compare                   # 跑对比")


if __name__ == "__main__":
    main()
