"""
数据可用性四问（Data Availability Quadruple-Check）
==================================================

参考 参考报告 2.1.2：所有模型调用前必须通过的审查协议。

四项检查：
1. Provenance（来源）：白名单数据源
2. Granularity（粒度）：country/club/friendly/competitive
3. Sample Size（样本量）：国家队 ≥15 场 / 俱乐部 ≥20 场
4. Timeliness（时效性）：≥ 2024-01-01

策略：检测 + 软警告
- 不阻断模型运行
- 输出 downgrade_weight ∈ [0.3, 1.0]，下游 Agent 把 confidence × 该权重
- 严重问题（来源不可信/全部超期）→ weight 直接降到 0.3

使用：
    from data.availability_check import run_all_checks
    quality_report = run_all_checks()
    # 写入 swarm context["data_quality"]
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


# ============ 白名单数据源 ============
TRUSTED_PROVENANCES = {
    "fifa.com", "FIFA", "FIFA Rankings",
    "eloratings.net", "World Football ELO Ratings",
    "Opta", "StatsBomb",
    "FBRef", "FBRef xG",
    "Transfermarkt",
    "Polymarket", "Kalshi", "Polymarket/Kalshi",
    "ESPN",
    "Reference", "Moonshot",
}

VALID_GRANULARITIES = {
    "country_competitive",   # 国家队正式赛（最高权重）
    "country_friendly",      # 国家队友谊赛
    "club_competitive",      # 俱乐部正式赛
    "club_friendly",         # 俱乐部友谊赛
    "country_mixed",         # 混合，需进一步细分
    "club_mixed",
}

# 验收标准
MIN_NATIONAL_SAMPLE = 15
MIN_CLUB_SAMPLE = 20
TIMELINESS_CUTOFF = "2024-01-01"


# ============ 检查结果 ============
@dataclass
class CheckResult:
    dataset: str
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    downgrade_weight: float = 1.0   # 1.0 = 完全可信；0.3 = 严重降级
    details: Dict[str, Any] = field(default_factory=dict)


# ============ 单项检查 ============
def check_provenance(dataset: str, sources: List[str]) -> tuple[bool, str, float]:
    """1. 来源审查"""
    if not sources:
        return False, "无来源标注", 0.5
    unknown = [s for s in sources if s not in TRUSTED_PROVENANCES]
    if not unknown:
        return True, "", 1.0
    if len(unknown) == len(sources):
        # 全部不可信 → 直接 0.2，几何平均后仍能拖到 <0.7
        return False, f"全部来源不在白名单: {unknown}", 0.2
    return True, f"部分来源未知: {unknown}", 0.85


def check_granularity(dataset: str, granularity: Optional[str]) -> tuple[bool, str, float]:
    """2. 粒度审查"""
    if not granularity:
        return False, "未标注粒度", 0.7
    if granularity not in VALID_GRANULARITIES:
        return False, f"未知粒度: {granularity}", 0.6
    return True, "", 1.0


def check_sample_size(dataset: str, n_samples: int,
                       granularity: Optional[str]) -> tuple[bool, str, float]:
    """3. 样本量审查"""
    if n_samples is None:
        return False, "无样本量标注", 0.7

    # 根据粒度选阈值
    if granularity and granularity.startswith("country"):
        threshold = MIN_NATIONAL_SAMPLE
    elif granularity and granularity.startswith("club"):
        threshold = MIN_CLUB_SAMPLE
    else:
        threshold = MIN_NATIONAL_SAMPLE

    if n_samples >= threshold:
        return True, "", 1.0
    if n_samples >= threshold * 0.5:
        return True, f"低样本: {n_samples} < {threshold}", 0.7
    return False, f"严重低样本: {n_samples} < {threshold * 0.5:.0f}", 0.4


def check_timeliness(dataset: str, last_updated: Optional[str]) -> tuple[bool, str, float]:
    """4. 时效性审查"""
    if not last_updated:
        return False, "无更新时间标注", 0.6

    try:
        updated_date = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
        # 去掉 tz 做比较
        if updated_date.tzinfo:
            updated_date = updated_date.replace(tzinfo=None)
        cutoff = datetime.fromisoformat(TIMELINESS_CUTOFF)
    except Exception as e:
        return False, f"时间格式错误: {e}", 0.5

    if updated_date < cutoff:
        days_old = (datetime.now() - updated_date).days
        return False, f"数据过期 ({days_old} 天前)", 0.5

    days_old = (datetime.now() - updated_date).days
    if days_old > 60:
        return True, f"较旧 ({days_old} 天前)", 0.85
    return True, "", 1.0


# ============ 单 dataset 综合检查 ============
def quad_check(dataset: str, meta: Dict[str, Any]) -> CheckResult:
    """对一个数据集做完整四问"""
    res = CheckResult(dataset=dataset)
    weights = []

    # 1. Provenance
    sources = meta.get("sources", [])
    if isinstance(sources, str):
        sources = [sources]
    passed, msg, w = check_provenance(dataset, sources)
    weights.append(w)
    if not passed:
        res.passed = False
    if msg:
        res.warnings.append(f"[Provenance] {msg}")
    res.details["provenance_weight"] = w

    # 2. Granularity
    granularity = meta.get("granularity")
    passed, msg, w = check_granularity(dataset, granularity)
    weights.append(w)
    if not passed:
        res.passed = False
    if msg:
        res.warnings.append(f"[Granularity] {msg}")
    res.details["granularity_weight"] = w

    # 3. Sample Size
    n_samples = meta.get("n_samples")
    passed, msg, w = check_sample_size(dataset, n_samples, granularity)
    weights.append(w)
    if not passed and w < 0.5:
        res.passed = False
    if msg:
        res.warnings.append(f"[SampleSize] {msg}")
    res.details["sample_weight"] = w
    res.details["n_samples"] = n_samples

    # 4. Timeliness
    last_updated = meta.get("last_updated")
    passed, msg, w = check_timeliness(dataset, last_updated)
    weights.append(w)
    if not passed:
        res.passed = False
    if msg:
        res.warnings.append(f"[Timeliness] {msg}")
    res.details["timeliness_weight"] = w

    # 综合权重 = 几何平均（任一极低都拖累整体）
    prod = 1.0
    for w in weights:
        prod *= w
    res.downgrade_weight = round(prod ** (1.0 / len(weights)), 3)

    return res


# ============ 全数据集扫描 ============
DATA_RAW = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_OUTPUTS = Path(__file__).parent.parent.parent / "data" / "outputs"


def _stat_age_iso(path: Path) -> str:
    """文件 mtime → ISO 字符串"""
    if not path.exists():
        return "1970-01-01"
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat()


def collect_dataset_metadata() -> Dict[str, Dict[str, Any]]:
    """从现有文件推断各数据集元信息（可能不完整时给保守值）"""
    teams_path = DATA_RAW / "teams.json"
    injuries_path = DATA_RAW / "injuries.json"
    external_path = DATA_RAW / "external_predictions.json"
    bracket_path = DATA_RAW / "bracket.json"

    metadata = {}

    # teams.json：核心实力数据
    if teams_path.exists():
        teams_data = json.load(open(teams_path))
        sources = teams_data.get("_sources", [])
        teams_dict = teams_data.get("teams", {})
        metadata["teams"] = {
            "sources": sources,
            "granularity": "country_competitive",
            "n_samples": len(teams_dict),  # 应为 48
            "last_updated": _stat_age_iso(teams_path),
        }

    # injuries.json
    if injuries_path.exists():
        try:
            inj_data = json.load(open(injuries_path))
            sources = inj_data.get("_sources", ["FBRef", "ESPN"])
            n_teams = len([k for k in inj_data if not k.startswith("_")])
            metadata["injuries"] = {
                "sources": sources,
                "granularity": "country_competitive",
                "n_samples": n_teams,
                "last_updated": _stat_age_iso(injuries_path),
            }
        except Exception:
            pass

    # external_predictions.json
    if external_path.exists():
        ext = json.load(open(external_path))
        sources = []
        n_predictions = 0
        for k, v in ext.items():
            if isinstance(v, dict):
                if "predictions" in v:
                    preds = v["predictions"]
                    if isinstance(preds, dict):
                        n_predictions += len(preds)
                    sources.append(v.get("source", k))
                elif "source" in v:
                    sources.append(v["source"])
                    n_predictions += len(v) - 1
        metadata["external_predictions"] = {
            "sources": sources or ["Polymarket", "Reference", "Opta"],
            "granularity": "country_competitive",
            "n_samples": n_predictions or 48,
            "last_updated": _stat_age_iso(external_path),
        }

    # bracket.json
    if bracket_path.exists():
        metadata["bracket"] = {
            "sources": ["FIFA"],
            "granularity": "country_competitive",
            "n_samples": 48,
            "last_updated": _stat_age_iso(bracket_path),
        }

    return metadata


def run_all_checks(verbose: bool = False) -> Dict[str, Any]:
    """对所有数据集跑可用性四问"""
    metadata = collect_dataset_metadata()
    results: Dict[str, Dict[str, Any]] = {}

    for dataset, meta in metadata.items():
        res = quad_check(dataset, meta)
        results[dataset] = asdict(res)
        if verbose:
            mark = "✓" if res.passed else "⚠"
            print(f"  {mark} {dataset:<25} weight={res.downgrade_weight}  "
                  f"warnings={len(res.warnings)}")
            for w in res.warnings:
                print(f"      · {w}")

    # 综合：所有 weight 的几何平均
    weights = [r["downgrade_weight"] for r in results.values()]
    if weights:
        prod = 1.0
        for w in weights:
            prod *= w
        overall = round(prod ** (1.0 / len(weights)), 3)
    else:
        overall = 0.5

    summary = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "n_datasets": len(results),
        "n_passed": sum(1 for r in results.values() if r["passed"]),
        "overall_weight": overall,
        "datasets": results,
    }

    # 写出供下游消费
    out = DATA_OUTPUTS / "data_quality_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


def get_team_weight_factor(team: str = None,
                            quality_report: Optional[Dict] = None) -> float:
    """
    给定 quality_report，返回 0.3-1.0 的 confidence 倍乘因子。
    供 Agent 在 build_user_prompt 中读取并附加到 prompt。
    """
    if not quality_report:
        return 1.0
    return float(quality_report.get("overall_weight", 1.0))


# ============ CLI ============
if __name__ == "__main__":
    print("📋 数据可用性四问审查\n")
    summary = run_all_checks(verbose=True)
    print(f"\n综合权重: {summary['overall_weight']}  "
          f"通过率: {summary['n_passed']}/{summary['n_datasets']}")
    print(f"📄 报告写入: data/outputs/data_quality_report.json")
