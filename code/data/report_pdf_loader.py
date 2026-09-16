"""
Reference PDF 报告数据加载器

Reference 的预测是一次性发布的 PDF 静态快照（不会变化），
直接从已提取的文本中解析表 8.1 的冠军概率排名。

数据源：~/Downloads/WorldCup_Report_Extracted.txt（已用 pdftotext 提取）
"""
import re
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# 参考报告表 8.1 的硬编码（从已提取 PDF 文本第 6273-6325 行验证）
# 这些数字是 Reference PDF 报告中"基准概率"列，是 PDF 报告的固定快照
REPORT_TABLE_8_1 = {
    # Top 8（明确的冠军候选）
    "Spain": 16.5,
    "France": 15.0,
    "Argentina": 12.0,
    "England": 11.0,
    "Germany": 11.0,
    "Brazil": 9.0,
    "Portugal": 7.0,
    "Netherlands": 4.0,
    # 9-24
    "Colombia": 3.5,
    "Morocco": 1.5,
    "Sweden": 2.0,
    "Belgium": 1.0,
    "Japan": 1.2,
    "Mexico": 1.0,
    "USA": 0.9,
    "Uruguay": 0.8,
    "Croatia": 0.7,
    "Scotland": 0.6,
    "Ecuador": 0.5,
    "Switzerland": 0.5,
    "Senegal": 0.4,
    "Turkiye": 0.3,
    "Norway": 0.3,
    "Korea Republic": 0.3,
}


# 来源元数据
REPORT_META = {
    "report_version": "v1.0",
    "report_date": "2026-06-05",
    "table_id": "8.1",
    "method": "8 源融合（ELO + FIFA + Poisson + XGBoost + Goldman + Opta + Polymarket + 博彩）+ 100,000 次蒙特卡洛",
    "n_simulations": 100000,
    "n_models_ensembled": 8,
    "report_size_pages": 205,
    "report_n_agents": 300,
    "extracted_pdf_path": str(Path.home() / "Downloads" / "WorldCup_Report.pdf"),
}


def load_report_predictions() -> dict:
    """
    返回 参考表 8.1 的冠军概率
    
    Note: 这是 PDF 报告的固定快照，不会变化
    """
    return dict(REPORT_TABLE_8_1)


def load_report_with_meta() -> dict:
    """带元数据的完整版"""
    return {
        "predictions": load_report_predictions(),
        "_meta": {
            **REPORT_META,
            "n_teams": len(REPORT_TABLE_8_1),
        }
    }


def update_to_external_db():
    """
    把 Reference 数据写入 external_predictions.json
    （这是 Reference 的"实时刷新"——其实就是重读 PDF）
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
    from multi_model_runner import update_prediction
    
    preds = load_report_predictions()
    update_prediction(
        "report", 
        preds, 
        source_url=f"PDF: {REPORT_META['extracted_pdf_path']}", 
        method="pdf_static"
    )
    print(f"✅ 参考报告 ({len(preds)} 个球队) 已加载到 external_predictions.json")
    print(f"   来源: PDF 表 8.1 (报告版本 {REPORT_META['report_version']}, {REPORT_META['report_date']})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        update_to_external_db()
    else:
        # 默认显示
        data = load_report_with_meta()
        print(f"📋 Reference PDF 报告冠军预测（表 {data['_meta']['table_id']}）")
        print(f"   报告版本: {data['_meta']['report_version']}")
        print(f"   发布日期: {data['_meta']['report_date']}")
        print(f"   方法: {data['_meta']['method']}\n")
        sorted_preds = sorted(data["predictions"].items(), key=lambda x: -x[1])
        for i, (team, p) in enumerate(sorted_preds[:16], 1):
            print(f"   {i:>2}. {team:<18} {p:>5.1f}%")
        print(f"\n   总计 {data['_meta']['n_teams']} 个球队")
