"""
多语言报告生成器（#17）

支持语言：zh / en / es
输出格式：Markdown
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_OUTPUTS, save_output


# ============ i18n 字典 ============
I18N = {
    "zh": {
        "title": "🏆 2026 世界杯冠军预测报告",
        "subtitle": "WorldCup Predict v1.2 · 真实计算引擎",
        "method": "**方法**：100,000 次蒙特卡洛 + 真实 FIFA Bracket + 三情景平行模拟 + 多平台市场聚合 + 伤病库自动注入",
        "data_source": "**数据源**：eloratings.net + Polymarket/Kalshi + 实时伤病情报",
        "tournament_top": "📊 冠军概率排名（Top 12）",
        "rank": "排名",
        "team": "球队",
        "prob": "概率",
        "ci": "95% 置信区间",
        "market": "市场",
        "bias": "偏差",
        "finals_top": "🏆 最可能决赛对阵 (Top 10)",
        "finals_appearance": "📌 进决赛概率 Top 8",
        "scenarios": "🎬 三情景对比",
        "bull": "乐观",
        "base": "基准",
        "bear": "悲观",
        "ev": "期望值",
        "key_insights": "💡 关键洞察",
        "risk_warning": "⚠️ 风险提示",
        "warning_text": "所有概率为低置信度（<40%）；48 队制首届赛事固有不确定性；伤病/赔率赛前 48h 内可能变化",
        "stage_final": "决赛",
        "stage_sf": "半决赛",
        "stage_qf": "8 强",
        "vs": "vs",
    },
    "en": {
        "title": "🏆 2026 World Cup Winner Prediction Report",
        "subtitle": "WorldCup Predict v1.2 · Real Compute Engine",
        "method": "**Method**: 100,000 Monte Carlo simulations + Real FIFA Bracket + Tri-scenario parallel simulation + Multi-platform market aggregation + Auto-injected injury database",
        "data_source": "**Data Sources**: eloratings.net + Polymarket/Kalshi + Real-time injury intel",
        "tournament_top": "📊 Championship Probability Rankings (Top 12)",
        "rank": "Rank",
        "team": "Team",
        "prob": "Prob.",
        "ci": "95% CI",
        "market": "Market",
        "bias": "Edge",
        "finals_top": "🏆 Most Likely Final Matchups (Top 10)",
        "finals_appearance": "📌 Probability of Reaching Final (Top 8)",
        "scenarios": "🎬 Three-Scenario Comparison",
        "bull": "Bull",
        "base": "Base",
        "bear": "Bear",
        "ev": "EV",
        "key_insights": "💡 Key Insights",
        "risk_warning": "⚠️ Risk Disclaimer",
        "warning_text": "All probabilities are low confidence (<40%); inherent uncertainty of first 48-team tournament; injuries/odds can shift within 48h pre-match",
        "stage_final": "Final",
        "stage_sf": "Semi-Final",
        "stage_qf": "Quarter-Final",
        "vs": "vs",
    },
    "es": {
        "title": "🏆 Informe de Predicción del Campeón de la Copa Mundial 2026",
        "subtitle": "WorldCup Predict v1.2 · Motor de Cálculo Real",
        "method": "**Método**: 100,000 simulaciones Monte Carlo + Bracket Real FIFA + Simulación paralela tri-escenario + Agregación de mercado multi-plataforma + Base de lesiones auto-inyectada",
        "data_source": "**Fuentes**: eloratings.net + Polymarket/Kalshi + Inteligencia de lesiones en tiempo real",
        "tournament_top": "📊 Ranking de Probabilidad de Campeonato (Top 12)",
        "rank": "Pos",
        "team": "Equipo",
        "prob": "Prob.",
        "ci": "IC 95%",
        "market": "Mercado",
        "bias": "Sesgo",
        "finals_top": "🏆 Finales Más Probables (Top 10)",
        "finals_appearance": "📌 Probabilidad de Llegar a Final (Top 8)",
        "scenarios": "🎬 Comparación Tres Escenarios",
        "bull": "Optimista",
        "base": "Base",
        "bear": "Pesimista",
        "ev": "VE",
        "key_insights": "💡 Hallazgos Clave",
        "risk_warning": "⚠️ Advertencia de Riesgo",
        "warning_text": "Todas las probabilidades son de baja confianza (<40%); incertidumbre inherente del primer torneo de 48 equipos; lesiones/cuotas pueden cambiar 48h antes del partido",
        "stage_final": "Final",
        "stage_sf": "Semifinal",
        "stage_qf": "Cuartos",
        "vs": "vs",
    },
}


def load_data():
    """加载所有需要的数据文件"""
    out = {}
    for fname in ["synthesizer_report.json", "finals_matchups.json", 
                   "three_scenarios.json"]:
        path = DATA_OUTPUTS / fname
        if path.exists():
            with open(path, "r") as f:
                out[fname.replace(".json", "")] = json.load(f)
    return out


def generate_report(lang: str = "zh") -> str:
    """生成 Markdown 报告"""
    if lang not in I18N:
        lang = "zh"
    t = I18N[lang]
    data = load_data()
    
    md = f"# {t['title']}\n\n"
    md += f"**{t['subtitle']}**  \n"
    md += f"{t['method']}  \n"
    md += f"{t['data_source']}\n\n"
    md += "---\n\n"
    
    # ============ 1. Top 12 冠军概率 ============
    md += f"## {t['tournament_top']}\n\n"
    if "synthesizer_report" in data:
        synth = data["synthesizer_report"]
        sorted_teams = sorted(synth.items(), key=lambda x: -x[1]["final_probability"])[:12]
        
        md += f"| {t['rank']} | {t['team']} | {t['prob']} | {t['ci']} | {t['market']} | {t['bias']} |\n"
        md += "|------|------|--------|--------|--------|--------|\n"
        for i, (team, d) in enumerate(sorted_teams, 1):
            bias = d["bias_pp"]
            sign = "+" if bias > 0 else ""
            md += f"| {i} | **{team}** | {d['final_probability']:.1f}% | [{d['ci_lower']:.1f}-{d['ci_upper']:.1f}] | {d['market_implied']:.1f}% | {sign}{bias:.1f}pp |\n"
    md += "\n---\n\n"
    # ============ 3. 决赛对阵 ============
    if "finals_matchups" in data:
        finals = data["finals_matchups"]
        md += f"## {t['finals_top']}\n\n"
        md += f"| {t['rank']} | {t['team']} A | {t['vs']} | {t['team']} B | {t['prob']} |\n"
        md += "|------|------|------|------|--------|\n"
        for i, m in enumerate(finals["top_matchups"][:10], 1):
            md += f"| {i} | **{m['team_a']}** | vs | **{m['team_b']}** | {m['probability']*100:.2f}% |\n"
        md += "\n"
        
        md += f"## {t['finals_appearance']}\n\n"
        appearances = list(finals["final_appearance"].items())[:8]
        n_sim = finals["n_simulations"]
        md += f"| {t['rank']} | {t['team']} | {t['prob']} |\n"
        md += "|------|------|--------|\n"
        for i, (team, count) in enumerate(appearances, 1):
            prob = count / n_sim * 100
            md += f"| {i} | **{team}** | {prob:.2f}% |\n"
        md += "\n---\n\n"
    
    # ============ 4. 三情景 ============
    if "three_scenarios" in data:
        sc = data["three_scenarios"]
        md += f"## {t['scenarios']}\n\n"
        md += f"| {t['team']} | {t['bull']}% | {t['base']}% | {t['bear']}% | {t['ev']}% |\n"
        md += "|------|------|------|------|------|\n"
        
        ev = sc["expected_value"]
        sorted_ev = sorted(ev.items(), key=lambda x: -x[1])[:8]
        for team, ev_val in sorted_ev:
            bull_p = sc["scenarios"]["bull"]["probs"][team]["champion"] * 100
            base_p = sc["scenarios"]["base"]["probs"][team]["champion"] * 100
            bear_p = sc["scenarios"]["bear"]["probs"][team]["champion"] * 100
            md += f"| **{team}** | {bull_p:.2f}% | {base_p:.2f}% | {bear_p:.2f}% | {ev_val*100:.2f}% |\n"
        md += "\n---\n\n"
    
    # ============ 5. 风险提示 ============
    md += f"## {t['risk_warning']}\n\n"
    md += f"> {t['warning_text']}\n\n"
    
    md += f"---\n\n*Generated by WorldCup Predict v1.2 · {Path(__file__).stem}.py · "
    md += f"data/outputs/synthesizer_report.json*\n"
    
    return md


def main(lang: str = "zh"):
    print(f"📝 生成 {lang.upper()} 报告...")
    md = generate_report(lang)
    
    # 保存
    out_path = DATA_OUTPUTS / f"report_{lang}.md"
    with open(out_path, "w") as f:
        f.write(md)
    
    print(f"✅ 已保存到: {out_path}")
    print(f"📊 字符数: {len(md):,}")
    
    # 打印前 50 行预览
    print(f"\n{'='*80}")
    print(f"📄 报告预览（前 60 行）：")
    print("=" * 80)
    for line in md.split("\n")[:60]:
        print(line)
    print("...")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        lang = sys.argv[1]
        if lang == "all":
            for l in ["zh", "en", "es"]:
                main(l)
                print()
        else:
            main(lang)
    else:
        main("zh")
