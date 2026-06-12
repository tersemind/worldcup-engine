"""
环境因子动态注入引擎（#10 P2 模型升级）

每场比赛粒度注入：
1. WBGT 高温对体能/精度的影响
2. 海拔对 VO₂ max 的影响
3. 球队大洲适应度
4. 旅行疲劳（赛程密度）

输出：动态修正的 lambda（每场进球期望）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW


# 缓存
_VENUES_CACHE = None
_TEAM_CONTINENT_CACHE = None
_TOLERANCE_CACHE = None


def load_venues():
    global _VENUES_CACHE, _TEAM_CONTINENT_CACHE, _TOLERANCE_CACHE
    if _VENUES_CACHE is None:
        with open(DATA_RAW / "venues.json", "r") as f:
            data = json.load(f)
        _VENUES_CACHE = data["venues"]
        _TEAM_CONTINENT_CACHE = data["_team_continent"]
        _TOLERANCE_CACHE = data["_continental_heat_tolerance"]
    return _VENUES_CACHE, _TEAM_CONTINENT_CACHE, _TOLERANCE_CACHE


def get_team_continent(team: str) -> str:
    """返回球队大洲"""
    _, continents, _ = load_venues()
    return continents.get(team, "Europe")  # 默认欧洲（保守）


def get_heat_penalty(team: str, venue: dict) -> float:
    """
    返回单场比赛的"热税"系数（对该队 lambda 的乘法修正）
    
    Args:
        venue: venues.json 中单个场地数据
    
    Returns:
        系数（0.85-1.0），越低 = 受影响越大
    """
    _, _, tolerance = load_venues()
    continent = get_team_continent(team)
    
    wbgt = venue.get("wbgt_peak_c", 22)
    heat_tolerance = tolerance.get(continent, {}).get("heat", 0.7)
    
    # WBGT 阈值映射：
    #   <22°C: 无影响
    #   22-26°C: 轻微（-2-4%）
    #   26-28°C: 中等（-4-7%）
    #   28-32°C: 严重（-7-12%）
    #   >32°C: 极端（-12-15%）
    if wbgt < 22:
        base_penalty = 0.00
    elif wbgt < 26:
        base_penalty = 0.03
    elif wbgt < 28:
        base_penalty = 0.06
    elif wbgt < 32:
        base_penalty = 0.10
    else:
        base_penalty = 0.13
    
    # 适应度修正：欧洲队（0.5）受影响是非洲队（1.0）的 2 倍
    actual_penalty = base_penalty * (1.0 - heat_tolerance) * 2  # 乘 2 让差异显著
    
    # 室内场馆（BC Place）/ 有空调减半
    if venue.get("indoor", False):
        actual_penalty *= 0.3
    
    return 1.0 - actual_penalty  # 返回乘法系数


def get_altitude_penalty(team: str, venue: dict) -> float:
    """
    海拔修正系数
    
    海拔每升 1000m，VO₂ max 下降约 6-7%
    适应度高的球队（南美）影响减半
    """
    _, _, tolerance = load_venues()
    altitude = venue.get("altitude_m", 0)
    
    if altitude < 500:
        return 1.0  # 无影响
    
    continent = get_team_continent(team)
    alt_tolerance = tolerance.get(continent, {}).get("altitude", 0.4)
    
    # 海拔影响：每 1000m 损失 6%（满分非适应队）
    base_loss = (altitude / 1000) * 0.06
    actual_loss = base_loss * (1.0 - alt_tolerance) * 1.5
    
    return 1.0 - min(0.15, actual_loss)  # 上限 -15%


def get_environment_factors(team: str, venue_name: str) -> dict:
    """
    返回某球队在某场地的所有环境修正因子
    
    Returns:
        {
            "heat_factor": 0.92,    # 进攻强度乘法系数
            "altitude_factor": 0.95,
            "combined_factor": 0.87,  # heat × altitude
            "details": {...}
        }
    """
    venues, _, _ = load_venues()
    venue = venues.get(venue_name)
    if not venue:
        return {"heat_factor": 1.0, "altitude_factor": 1.0, "combined_factor": 1.0,
                "warning": f"Unknown venue: {venue_name}"}
    
    heat = get_heat_penalty(team, venue)
    altitude = get_altitude_penalty(team, venue)
    combined = heat * altitude
    
    return {
        "heat_factor": round(heat, 4),
        "altitude_factor": round(altitude, 4),
        "combined_factor": round(combined, 4),
        "details": {
            "venue": venue_name,
            "wbgt": venue.get("wbgt_peak_c"),
            "altitude_m": venue.get("altitude_m"),
            "heat_risk": venue.get("heat_risk"),
            "team_continent": get_team_continent(team),
        }
    }


def adjust_match_lambdas(team_a: str, team_b: str, lambda_a: float, lambda_b: float,
                          venue_name: str = None) -> tuple:
    """
    动态调整两队期望进球
    
    Args:
        team_a, team_b: 球队名
        lambda_a, lambda_b: 原始期望进球
        venue_name: 场地名（如 "AT&T Stadium"）
    
    Returns:
        (lambda_a_adj, lambda_b_adj, env_info)
    """
    if venue_name is None:
        return lambda_a, lambda_b, {"adjusted": False}
    
    factor_a = get_environment_factors(team_a, venue_name)
    factor_b = get_environment_factors(team_b, venue_name)
    
    # 进攻 lambda 受热税 + 海拔双重影响
    new_lambda_a = lambda_a * factor_a["combined_factor"]
    new_lambda_b = lambda_b * factor_b["combined_factor"]
    
    return new_lambda_a, new_lambda_b, {
        "adjusted": True,
        "venue": venue_name,
        "team_a_factor": factor_a["combined_factor"],
        "team_b_factor": factor_b["combined_factor"],
        "team_a_continent": factor_a["details"]["team_continent"],
        "team_b_continent": factor_b["details"]["team_continent"],
    }


def venue_risk_summary():
    """打印 16 场地风险摘要"""
    venues, _, _ = load_venues()
    
    print("=" * 90)
    print("🌡️  2026 World Cup 16 场地环境风险摘要")
    print("=" * 90)
    print(f"{'Venue':<24} {'City':<18} {'WBGT':<6} {'Alt':<8} {'Heat':<10} {'Indoor':<7}")
    print("-" * 90)
    
    sorted_venues = sorted(venues.items(), key=lambda x: -x[1].get("wbgt_peak_c", 0))
    for name, v in sorted_venues:
        indoor = "✓" if v.get("indoor") else " "
        print(f"{name:<24} {v['city']:<18} {v['wbgt_peak_c']:>3}°C  "
              f"{v['altitude_m']:>4}m  {v['heat_risk']:<10} {indoor:<7}")


def team_environment_profile(team: str):
    """打印某队在 16 场地的全场修正系数"""
    venues, continents, tolerance = load_venues()
    continent = continents.get(team, "Europe")
    tol = tolerance.get(continent, {})
    
    print("=" * 80)
    print(f"🌍 球队环境画像: {team}")
    print(f"   大洲: {continent} | 热适应: {tol.get('heat'):.2f} | 海拔适应: {tol.get('altitude'):.2f}")
    print("=" * 80)
    print(f"\n{'Venue':<24} {'Heat':<10} {'Alt':<10} {'Combined':<12}")
    print("-" * 60)
    
    profiles = []
    for name in venues:
        f = get_environment_factors(team, name)
        profiles.append((name, f))
    
    profiles.sort(key=lambda x: x[1]["combined_factor"])
    
    for name, f in profiles:
        h = f["heat_factor"]
        a = f["altitude_factor"]
        c = f["combined_factor"]
        # 标注影响等级
        if c < 0.92:
            mark = "🔴"
        elif c < 0.96:
            mark = "🟡"
        else:
            mark = "🟢"
        print(f"{name:<24} {h:<10.3f} {a:<10.3f} {c:<10.3f} {mark}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "venues":
            venue_risk_summary()
        elif sys.argv[1] == "team" and len(sys.argv) > 2:
            team_environment_profile(sys.argv[2])
        elif sys.argv[1] == "test":
            # 测试单场调整
            print("=== 测试：法国 vs 摩洛哥 在不同场地 ===\n")
            for venue in ["AT&T Stadium", "Estadio Azteca", "BC Place"]:
                la, lb, info = adjust_match_lambdas("France", "Morocco", 1.5, 1.2, venue)
                print(f"📍 {venue}")
                print(f"   France(欧洲): λ {1.5:.2f} → {la:.2f} ({info['team_a_factor']:.3f}×)")
                print(f"   Morocco(非洲): λ {1.2:.2f} → {lb:.2f} ({info['team_b_factor']:.3f}×)")
                print()
    else:
        venue_risk_summary()
        print("\n用法:")
        print("  python3 environment_engine.py venues          # 场地风险摘要")
        print("  python3 environment_engine.py team Spain      # 某队全场画像")
        print("  python3 environment_engine.py test            # 测试单场调整")
