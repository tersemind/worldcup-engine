"""
2026 FIFA World Cup 比赛场地天气抓取
=====================================
来源: Open-Meteo Forecast API (https://api.open-meteo.com，无需 key，免费 10k 调用/天)

逻辑:
  1. 从 group_schedule.json 读未来 N 小时的比赛场次
  2. 按 venue_city 去重，对每个城市查 lat/lon 后调天气
  3. 取比赛时刻附近 ±1h 的气象指标
  4. 输出到 data/raw/weather.json

输出格式:
  {
    "_metadata": {"updated_at": ..., "source": "open-meteo", "hours_ahead": 72},
    "matches": [
      {
        "match_id", "date", "time_local", "venue_city",
        "weather": {
          "temp_c", "feels_like_c", "humidity_pct", "wind_kmh",
          "precip_mm", "precip_prob_pct", "uv_index",
          "wbgt_estimate_c", "risk_level": "low/medium/high/extreme"
        }
      }, ...
    ]
  }

下游消费: environment_engine.py 可以读这个文件做实时温度/降水调整。
"""
import sys
import json
import math
import requests
import warnings
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_RAW

warnings.filterwarnings("ignore")

# 16 个场馆城市坐标 + group_schedule 里使用的区域别名
CITY_COORDS = {
    "Arlington":       (32.7357, -97.1081),   # Dallas/Arlington TX
    "Dallas":          (32.7357, -97.1081),   # 别名
    "Atlanta":         (33.7550, -84.4006),
    "East Rutherford": (40.8128, -74.0742),   # MetLife NJ
    "New York/New Jersey": (40.8128, -74.0742),  # 别名
    "Foxborough":      (42.0909, -71.2643),   # Gillette MA
    "Boston":          (42.0909, -71.2643),   # 别名
    "Philadelphia":    (39.9008, -75.1675),   # Lincoln Financial
    "Guadalajara":     (20.6810, -103.3450),
    "Houston":         (29.6847, -95.4107),
    "Inglewood":       (33.9617, -118.3531),  # SoFi LA
    "Los Angeles":     (33.9617, -118.3531),  # 别名
    "Kansas City":     (39.0489, -94.4839),
    "Mexico City":     (19.3030, -99.1500),
    "Miami Gardens":   (25.9580, -80.2389),
    "Miami":           (25.9580, -80.2389),   # 别名
    "Monterrey":       (25.6692, -100.2447),
    "Santa Clara":     (37.4030, -121.9700),  # Levi's Stadium
    "San Francisco Bay Area": (37.4030, -121.9700),  # 别名
    "Seattle":         (47.5952, -122.3316),
    "Toronto":         (43.6332, -79.4163),
    "Vancouver":       (49.2769, -123.1119),
}

OUT_PATH = DATA_RAW / "weather.json"
SCHED_PATH = DATA_RAW / "group_schedule.json"
HOURS_AHEAD = 72  # 抓未来 72h 内的比赛
API_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _wbgt_estimate(temp_c: float, humidity_pct: float) -> float:
    """简化 WBGT (Wet Bulb Globe Temperature) 估算
    
    Stull 2011 简化公式: Tw ≈ T * atan(0.151977*sqrt(RH+8.313659)) 
                           + atan(T+RH) - atan(RH-1.676331)
                           + 0.00391838*RH^1.5*atan(0.023101*RH) - 4.686035
    然后 WBGT ≈ 0.7 * Tw + 0.3 * T （阴影下）
    """
    if humidity_pct < 0 or humidity_pct > 100:
        return temp_c
    try:
        rh = humidity_pct
        tw = (
            temp_c * math.atan(0.151977 * math.sqrt(rh + 8.313659))
            + math.atan(temp_c + rh) - math.atan(rh - 1.676331)
            + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
            - 4.686035
        )
        wbgt = 0.7 * tw + 0.3 * temp_c
        return round(wbgt, 1)
    except (ValueError, OverflowError):
        return temp_c


def _risk_level(wbgt_c: float, precip_prob: int) -> str:
    """根据 WBGT + 降水概率判定风险"""
    if wbgt_c >= 31 or precip_prob >= 80:
        return "extreme"
    if wbgt_c >= 28 or precip_prob >= 60:
        return "high"
    if wbgt_c >= 25 or precip_prob >= 30:
        return "medium"
    return "low"


def fetch_forecast(lat: float, lon: float, target_dt: datetime,
                   timeout: int = 20, max_retry: int = 3) -> dict:
    """抓取 target_dt 附近 ±1h 的天气均值

    自动判断时间：过去用 archive API，未来用 forecast API。
    """
    now = datetime.now()
    is_past = target_dt < now - timedelta(hours=2)
    if is_past:
        # 历史天气 archive API
        date_str = target_dt.strftime("%Y-%m-%d")
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": date_str, "end_date": date_str,
            "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                      "precipitation,wind_speed_10m",
            "timezone": "auto",
        }
        url = ARCHIVE_URL
    else:
        # forecast API - 支持未来 16 天
        days_ahead = max(1, min(16, (target_dt - now).days + 2))
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                      "precipitation,precipitation_probability,wind_speed_10m,uv_index",
            "timezone": "auto",
            "forecast_days": days_ahead,
        }
        url = API_URL
    last_err = None
    for attempt in range(max_retry + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            break
        except Exception as e:
            last_err = e
            if attempt < max_retry:
                import time as _t
                _t.sleep(2 * (attempt + 1))
    else:
        raise last_err
    data = r.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return {}
    
    # 找最接近 target_dt 的 3 个小时（±1h），取均值
    target_str = target_dt.strftime("%Y-%m-%dT%H:00")
    # 简单找最近的 index
    best_idx = None
    best_dt = None
    for i, t in enumerate(times):
        try:
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M")
        except ValueError:
            continue
        if best_idx is None or abs((dt - target_dt).total_seconds()) < abs((best_dt - target_dt).total_seconds()):
            best_idx = i
            best_dt = dt
    
    if best_idx is None:
        return {}
    
    # 取 ±1 个小时（3 点平均）
    idxs = [i for i in [best_idx-1, best_idx, best_idx+1] if 0 <= i < len(times)]
    def _avg(field):
        vals = [hourly[field][i] for i in idxs if hourly.get(field) and hourly[field][i] is not None]
        return sum(vals) / len(vals) if vals else None
    
    temp = _avg("temperature_2m")
    humid = _avg("relative_humidity_2m")
    feels = _avg("apparent_temperature")
    precip = _avg("precipitation") or 0
    precip_prob = _avg("precipitation_probability") if hourly.get("precipitation_probability") else 0
    precip_prob = precip_prob or 0
    wind = _avg("wind_speed_10m")
    uv = _avg("uv_index") if hourly.get("uv_index") else None
    
    wbgt = _wbgt_estimate(temp, humid) if (temp is not None and humid is not None) else None
    risk = _risk_level(wbgt or 0, int(precip_prob))
    
    return {
        "temp_c":             round(temp, 1) if temp is not None else None,
        "feels_like_c":       round(feels, 1) if feels is not None else None,
        "humidity_pct":       round(humid, 0) if humid is not None else None,
        "wind_kmh":           round(wind, 1) if wind is not None else None,
        "precip_mm":          round(precip, 1),
        "precip_prob_pct":    int(precip_prob),
        "uv_index":           round(uv, 1) if uv is not None else None,
        "wbgt_estimate_c":    wbgt,
        "risk_level":         risk,
        "forecast_time":      best_dt.isoformat(timespec="hours") if best_dt else None,
    }


def main():
    if not SCHED_PATH.exists():
        print(f"✗ group_schedule.json 不存在")
        sys.exit(1)
    
    fetch_all = "--all" in sys.argv
    merge_existing = fetch_all  # --all 模式下与已有数据合并

    sched = json.load(open(SCHED_PATH))
    now = datetime.now()
    cutoff = now + timedelta(hours=HOURS_AHEAD)
    
    # 筛选比赛
    upcoming = []
    for m in sched.get("matches", []):
        try:
            dt = datetime.strptime(f"{m['date']} {m.get('time_local','15:00')}", "%Y-%m-%d %H:%M")
        except (ValueError, KeyError):
            continue
        if fetch_all:
            upcoming.append((dt, m))
        elif now - timedelta(hours=1) <= dt <= cutoff:
            upcoming.append((dt, m))
    
    print(f"=== Weather Fetch (OpenMeteo) ===")
    if fetch_all:
        print(f"--all 模式：全部 {len(upcoming)} 场比赛（含历史 archive + 未来 forecast）")
    else:
        print(f"未来 {HOURS_AHEAD}h 内 {len(upcoming)} 场比赛")
    
    if not upcoming:
        OUT_PATH.write_text(json.dumps({
            "_metadata": {
                "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "open-meteo",
                "hours_ahead": HOURS_AHEAD,
                "n_matches": 0,
            },
            "matches": [],
        }, ensure_ascii=False, indent=2))
        print(f"  无场次，写入空文件")
        return
    
    # 按城市去重缓存（同一城市同一时间附近只调一次 API）
    city_cache = {}  # (city, hour_bucket) -> weather dict
    results = []
    n_fail = 0
    
    for idx, (dt, m) in enumerate(upcoming, 1):
        city = m.get("venue_city", "")
        if city not in CITY_COORDS:
            print(f"  ⚠ [{idx}/{len(upcoming)}] {m['team_a']} vs {m['team_b']}: 城市 '{city}' 无坐标，跳过")
            continue
        if idx % 10 == 0 or idx == 1:
            print(f"  ... 进度 {idx}/{len(upcoming)} (已抓 {len(results)}, 失败 {n_fail})")
        
        # 缓存键: city + 3h bucket（同城市 ±3h 内复用）
        bucket_h = dt.hour // 3
        cache_key = (city, m["date"], bucket_h)
        
        if cache_key not in city_cache:
            try:
                lat, lon = CITY_COORDS[city]
                city_cache[cache_key] = fetch_forecast(lat, lon, dt)
                import time as _t; _t.sleep(0.3)  # 限速
            except Exception as e:
                print(f"  ✗ {city} {m['date']} API 失败: {e}")
                city_cache[cache_key] = None
                n_fail += 1
        
        weather = city_cache[cache_key]
        if not weather:
            continue
        
        results.append({
            "match_id":   m.get("match_id"),
            "date":       m["date"],
            "time_local": m.get("time_local"),
            "venue_city": city,
            "venue":      m.get("venue"),
            "team_a":     m["team_a"],
            "team_b":     m["team_b"],
            "weather":    weather,
        })
    
    # --all 模式：合并已有数据（按 match_id 去重，新数据覆盖）
    final_matches = results
    if merge_existing and OUT_PATH.exists():
        try:
            existing = json.load(open(OUT_PATH)).get("matches", [])
            new_ids = {r.get("match_id") for r in results}
            kept = [e for e in existing if e.get("match_id") not in new_ids]
            final_matches = kept + results
            print(f"  合并已有 {len(kept)} 场 + 新抓 {len(results)} 场 = {len(final_matches)} 场")
        except Exception as e:
            print(f"  ⚠ 合并已有数据失败: {e}，仅保存新抓数据")

    OUT_PATH.write_text(json.dumps({
        "_metadata": {
            "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "open-meteo",
            "hours_ahead": HOURS_AHEAD if not fetch_all else "all",
            "mode": "all" if fetch_all else "window",
            "n_matches": len(final_matches),
            "n_api_calls": len([v for v in city_cache.values() if v]),
            "n_failed": n_fail,
        },
        "matches": final_matches,
    }, ensure_ascii=False, indent=2))
    
    print(f"✓ 写入 {OUT_PATH.name}: 总共 {len(final_matches)} 场（本次新抓 {len(results)} 场，{len(city_cache)} API 调用）")
    # 显示前 3 场示例
    for r in results[:3]:
        w = r["weather"]
        print(f"  {r['date']} {r['time_local']} {r['venue_city']:18s} "
              f"{w.get('temp_c','?')}°C feels {w.get('feels_like_c','?')}°C "
              f"湿度 {w.get('humidity_pct','?')}% 降水 {w.get('precip_prob_pct','?')}% "
              f"WBGT {w.get('wbgt_estimate_c','?')} ➜ {w.get('risk_level','?')}")


if __name__ == "__main__":
    main()
