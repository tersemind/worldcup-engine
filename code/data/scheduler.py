"""
自动数据刷新调度 daemon
========================

按各数据源的"自然新鲜度"分级调度，独立于 web server 运行。

任务表（默认频率）：
  - Kalshi 单场赔率       : 每 30 min
  - Polymarket 夺冠概率   : 每 60 min
  - Elo (eloratings.net)  : 每 12 h
  - 比赛结果 (group_results): 每 60 min（赛季内）
  - 伤病雷达 (LLM, 全 48 队): 每 6 h
  - 级联重算 (synth→mc→bias): 任何上游变化后触发
  
设计要点：
  - 单进程长跑（threading.Timer or APScheduler）
  - 每个任务有独立 lock，不会并发自己跑自己
  - 失败仅 log，不中断 daemon
  - 末次成功时间持久化到 data/outputs/scheduler_state.json（重启不会立刻全跑）
  - 标记 "dirty" 触发 cascade 防止抖动（debounce 5 min）

用法:
  python3 code/data/scheduler.py                  # 前台跑（调试）
  python3 code/data/scheduler.py --status         # 看上次各任务的运行情况
  nohup python3 code/data/scheduler.py > /tmp/wc_scheduler.log 2>&1 &  # 后台
  python3 code/data/scheduler.py --once <task>    # 立即跑单个任务
"""
from __future__ import annotations
import sys
import os
import json
import time
import logging
import threading
import argparse
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Callable, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "code"))

STATE_PATH = ROOT / "data" / "outputs" / "scheduler_state.json"
LOG_PATH = ROOT / "data" / "outputs" / "scheduler.log"
CONFIG_PATH = ROOT / "data" / "raw" / "scheduler_config.json"

# 内置默认配置（CONFIG_PATH 不存在时使用）
DEFAULT_CONFIG = {
    "_cascade": {"debounce_min": 3},
    "tasks": {
        "kalshi":          {"interval_min": 30,   "enabled": True, "timeout_sec": 60,  "triggers_cascade": True,  "description": "Kalshi 单场赔率"},
        "polymarket":      {"interval_min": 60,   "enabled": True, "timeout_sec": 60,  "triggers_cascade": True,  "description": "Polymarket 夺冠概率"},
        "group_results":   {"interval_min": 60,   "enabled": True, "timeout_sec": 60,  "triggers_cascade": True,  "description": "比赛结果 + live_state 同步"},
        "elo":             {"interval_min": 720,  "enabled": True, "timeout_sec": 60,  "triggers_cascade": True,  "description": "Elo 评级"},
        "injuries":        {"interval_min": 360,  "enabled": True, "timeout_sec": 900, "triggers_cascade": True,  "description": "48 队伤病雷达"},
        "market_bias":     {"interval_min": 60,   "enabled": True, "timeout_sec": 60,  "triggers_cascade": False, "description": "夺冠盘偏差"},
        "upset_llm":       {"interval_min": 360,  "enabled": True, "timeout_sec": 600, "triggers_cascade": False, "description": "LLM 终审爆冷"},
        "critical_nodes":  {"interval_min": 1440, "enabled": True, "timeout_sec": 1800,"triggers_cascade": True,  "description": "17-Agent 关键节点"},
        "derived_outputs": {"interval_min": 120,  "enabled": True, "timeout_sec": 600, "triggers_cascade": False, "description": "衍生输出"},
        "pi_ratings":      {"interval_min": 360,  "enabled": True, "timeout_sec": 120, "triggers_cascade": True,  "description": "Pi-ratings 评级"},
        "catboost_train":  {"interval_min": 10080,"enabled": True, "timeout_sec": 1800,"triggers_cascade": False, "description": "CatBoost 3 分类周训（pi-ratings 7 维特征）"},
        "scenarios":       {"interval_min": 360,  "enabled": True, "timeout_sec": 600, "triggers_cascade": False, "description": "三情景预测"},
        "multi_model_compare": {"interval_min": 360, "enabled": True, "timeout_sec": 120, "triggers_cascade": False, "description": "多模型对比"},
        "data_quality":    {"interval_min": 180,  "enabled": True, "timeout_sec": 60,  "triggers_cascade": False, "description": "数据可用性检查"},
        "weather":         {"interval_min": 180,  "enabled": True, "timeout_sec": 120, "triggers_cascade": False, "description": "比赛场地天气（OpenMeteo）"},
        "lineups":         {"interval_min": 30,   "enabled": True, "timeout_sec": 600, "triggers_cascade": False, "description": "赛前首发阵容（Serper + LLM）"},
        "schedule_refresh":{"interval_min": 1440, "enabled": True, "timeout_sec": 60,  "triggers_cascade": False, "description": "ESPN 赛程校验（changelog 报告）"},
        "live_events":     {"interval_min": 5,    "enabled": True, "timeout_sec": 60,  "triggers_cascade": False, "description": "ESPN 赛中事件流（无 live 时秒退）"},
        "h2h":             {"interval_min": 240,  "enabled": True, "timeout_sec": 900, "triggers_cascade": True,  "description": "双方近 5 场对阵（赛前 24h 内抓取，7 天内不重抓，所以 4h 一次足够）；接入 synthesizer 心理压制调整"},
        "squad_value":     {"interval_min": 10080,"enabled": True, "timeout_sec": 1200,"triggers_cascade": True,  "description": "Transfermarkt 阵容市值（每周 1 次），回写 teams.json[*].squad_value_m_eur"},
        "referee":         {"interval_min": 120,  "enabled": True, "timeout_sec": 600, "triggers_cascade": True,  "description": "裁判任命（赛前 30h 内抓取，24h 不重抓）；接入 synthesizer 裁判风格调整"},
        "in_match":        {"interval_min": 10,   "enabled": True, "timeout_sec": 120, "triggers_cascade": False, "description": "赛中迭代（MRCA+ITA+SOA）。仅当 live_events.json 有 live_matches 时实际跑；写 in_match_update.json 给 web 直读"},
        "live_trading_tick":{"interval_min": 1,    "enabled": True, "timeout_sec": 30,  "triggers_cascade": False, "description": "quant 高频量化 tick（仅 live match 存在时跑）；写 in_match_live.json + in_match_ticks.jsonl"},
    }
}


def load_config() -> dict:
    """读取外部 config（不存在则回落到默认）。失败不抛，仅 log。"""
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG
    try:
        data = json.load(open(CONFIG_PATH))
        # 合并：缺失字段用 default 填
        out = {
            "_cascade": {**DEFAULT_CONFIG["_cascade"], **data.get("_cascade", {})},
            "tasks": {},
        }
        for name, default_cfg in DEFAULT_CONFIG["tasks"].items():
            user_cfg = data.get("tasks", {}).get(name, {})
            out["tasks"][name] = {**default_cfg, **user_cfg}
        return out
    except Exception as e:
        print(f"⚠ 读取 {CONFIG_PATH} 失败: {e}，使用默认配置", file=sys.stderr)
        return DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("scheduler")


# ─────────── env bootstrap (launchctl) ───────────
def _bootstrap_env():
    """从 launchctl 读环境变量（macOS 子进程不会自动继承）"""
    for var in ["SERPER_API_KEY", "LINGYA_API_KEY", "WORLDCUP_LLM_MODEL"]:
        if os.environ.get(var):
            continue
        try:
            r = subprocess.run(["launchctl", "getenv", var],
                               capture_output=True, text=True, timeout=3)
            val = r.stdout.strip()
            if val:
                os.environ[var] = val
        except Exception:
            pass


_bootstrap_env()


# ─────────── S1: 阶段感知（stage-aware scheduling）───────────
# FIFA 2026 World Cup 阶段日期边界（公开赛程，UTC 日期）
# 注：以"开赛日"为准，第一场该阶段比赛开赛即进入新阶段
STAGE_DATE_BOUNDARIES = [
    ("group_stage",       "2026-06-11"),  # 6/11 - 6/27 小组赛
    ("round_of_32",       "2026-06-28"),  # 6/28 - 7/03 32 强
    ("round_of_16",       "2026-07-04"),  # 7/04 - 7/07 16 强
    ("quarter_finals",    "2026-07-09"),  # 7/09 - 7/11 8 强
    ("semi_finals",       "2026-07-14"),  # 7/14 - 7/15 半决赛
    ("final",             "2026-07-19"),  # 7/19 决赛
    ("post_tournament",   "2026-07-20"),  # 7/20+
]
# knockout 是 R32/R16/QF/SF 的逻辑总称（不是日期阶段）
KNOCKOUT_STAGES = {"round_of_32", "round_of_16", "quarter_finals", "semi_finals"}

# 单进程缓存：避免每 tick 都读盘
_stage_cache = {"value": None, "ts": 0.0, "ttl_sec": 60.0}
_stage_lock = threading.Lock()


def _infer_current_stage(today_str: Optional[str] = None) -> str:
    """根据当前日期推断阶段。

    today_str: 测试用，默认取系统今天。

    Returns: pre_tournament / group_stage / round_of_32 / round_of_16 /
             quarter_finals / semi_finals / final / post_tournament
    """
    today = today_str or datetime.now().strftime("%Y-%m-%d")
    # 从最早到最晚遍历，找到"今天 >= 阶段起始日"的最后一个匹配
    current = "pre_tournament"
    for stage, start_date in STAGE_DATE_BOUNDARIES:
        if today >= start_date:
            current = stage
        else:
            break
    return current


def _read_current_stage() -> str:
    """读 live_state.json 的 current_stage；缺失或过期则自动推断。

    带 60s 缓存，避免每个 task.should_run 都读盘。
    """
    with _stage_lock:
        now_ts = time.time()
        if _stage_cache["value"] and (now_ts - _stage_cache["ts"]) < _stage_cache["ttl_sec"]:
            return _stage_cache["value"]

        stage = None
        try:
            live_path = ROOT / "data" / "raw" / "live_state.json"
            if live_path.exists():
                live = json.load(open(live_path))
                stage = live.get("current_stage")
                # 兼容旧值 "group" → "group_stage"
                if stage == "group":
                    stage = "group_stage"
        except Exception as e:
            log.warning(f"[stage] 读 live_state 失败: {e}")

        if not stage:
            stage = _infer_current_stage()

        _stage_cache["value"] = stage
        _stage_cache["ts"] = now_ts
        return stage


def _sync_current_stage_to_live_state():
    """把推断的 current_stage 写回 live_state.json（由 group_results 任务调用）。

    失败仅 log，绝不影响主流程。
    """
    try:
        live_path = ROOT / "data" / "raw" / "live_state.json"
        if live_path.exists():
            live = json.load(open(live_path))
        else:
            live = {"tournament": "2026 FIFA World Cup"}

        inferred = _infer_current_stage()
        raw_old = live.get("current_stage")
        # 兼容旧 "group" → "group_stage"
        normalized_old = "group_stage" if raw_old == "group" else raw_old

        # 写盘条件：
        #   (a) 字面值与推断不一致（含 raw_old == "group" 需正规化的情况）
        #   (b) 第一次写入（raw_old is None）
        need_write = (raw_old != inferred) or (raw_old is None)
        if need_write:
            live["current_stage"] = inferred
            live["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            live_path.write_text(json.dumps(live, ensure_ascii=False, indent=2))
            if raw_old is None:
                log.info(f"[stage] 初始化 current_stage = {inferred}")
            elif normalized_old != inferred:
                log.info(f"[stage] current_stage: {normalized_old} → {inferred}")
            else:
                log.info(f"[stage] 正规化 current_stage: {raw_old!r} → {inferred!r}")
            # 失效缓存
            with _stage_lock:
                _stage_cache["value"] = None
    except Exception as e:
        log.warning(f"[stage] 写回 live_state 失败: {e}")


# ─────────── 任务定义 ───────────
class Task:
    """单个调度任务"""
    def __init__(self, name: str, fn: Callable, cfg: dict):
        """
        cfg: {interval_min, enabled, timeout_sec, triggers_cascade, description}
        """
        self.name = name
        self.fn = fn
        self.apply_cfg(cfg)
        self.lock = threading.Lock()
        self.last_run: Optional[datetime] = None
        self.last_success: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.n_runs = 0
        self.n_failures = 0

    def apply_cfg(self, cfg: dict):
        """支持 config 热重载"""
        self.interval = timedelta(minutes=int(cfg.get("interval_min", 60)))
        self.enabled = bool(cfg.get("enabled", True))
        self.timeout_sec = int(cfg.get("timeout_sec", 60))
        self.triggers_cascade = bool(cfg.get("triggers_cascade", False))
        self.description = cfg.get("description", "")
        # S2: 阶段感知配置（无则保持向后兼容，所有阶段用同一 interval）
        # 形如 {"group_stage": 60, "knockout": 120, "final": 30, "post_tournament": -1}
        # -1 表示该阶段暂停；"knockout" 是 R32/R16/QF/SF 的简写
        self.stage_overrides: Dict[str, int] = dict(cfg.get("stage_overrides", {}))

    def _effective_interval(self) -> Optional[timedelta]:
        """根据 live_state 当前阶段计算实际 interval。

        Returns:
            timedelta — 实际间隔
            None     — 当前阶段被 override 为 -1（暂停）

        优先级（高 → 低）：
          1. match_aware 动态层（赛后/赛前窗口，按当天日程）
          2. stage_overrides（按赛事阶段：group_stage / knockout / final ...）
          3. 默认 interval

        失败容错：任何异常退回默认 interval（绝不阻塞调度）。
        """
        # ── 第 1 层：match_aware 动态调度（赛后/赛前窗口）──
        try:
            from match_aware import dynamic_interval as _dyn_interval
            base_min = int(self.interval.total_seconds() / 60)
            override_min = _dyn_interval(self.name, base_min)
            if override_min is not None:
                return timedelta(minutes=int(override_min))
        except Exception as e:
            # 失败完全不影响主流程，退回老逻辑
            pass

        # ── 第 2 层：stage_overrides ──
        if not self.stage_overrides:
            return self.interval  # 无配置 → 旧行为
        try:
            stage = _read_current_stage()
            # 优先级：精确匹配 > knockout 别名 > 默认
            cand = self.stage_overrides.get(stage)
            if cand is None and stage in KNOCKOUT_STAGES:
                cand = self.stage_overrides.get("knockout")
            if cand is None:
                return self.interval
            if cand < 0:
                return None  # 暂停
            return timedelta(minutes=int(cand))
        except Exception as e:
            log.warning(f"[{self.name}] stage override 计算失败，退回默认: {e}")
            return self.interval

    def should_run(self, now: datetime) -> bool:
        if not self.enabled:
            return False
        eff = self._effective_interval()
        if eff is None:
            return False  # 当前阶段被暂停
        if self.last_run is None:
            return True
        return (now - self.last_run) >= eff

    def run(self, scheduler: "Scheduler") -> bool:
        """执行任务。返回 True 表示成功且有数据变化（触发 cascade）"""
        if not self.lock.acquire(blocking=False):
            log.info(f"[{self.name}] 跳过：上一次还在跑")
            return False
        try:
            self.last_run = datetime.now()
            self.n_runs += 1
            log.info(f"[{self.name}] 开始 (timeout={self.timeout_sec}s)")
            t0 = time.time()
            
            # 把 timeout 注入 task 函数
            try:
                result = self.fn(self.timeout_sec)
            except TypeError:
                # 兼容旧签名（无参）
                result = self.fn()
            
            elapsed = time.time() - t0
            self.last_success = datetime.now()
            self.last_error = None
            
            changed = isinstance(result, dict) and result.get("changed", False)
            summary = result.get("summary", "ok") if isinstance(result, dict) else "ok"
            
            log.info(f"[{self.name}] ✓ {elapsed:.1f}s {summary}")
            
            if changed and self.triggers_cascade:
                scheduler.mark_dirty(self.name)
            
            return changed
        except Exception as e:
            self.n_failures += 1
            err = f"{type(e).__name__}: {e}"
            self.last_error = err
            log.error(f"[{self.name}] ✗ {err}")
            log.debug(traceback.format_exc())
            return False
        finally:
            self.lock.release()
            scheduler.save_state()


# ─────────── 具体任务函数 ───────────
# 约定：所有 task 函数签名 (timeout: int) -> dict，由 Task.run 传入 timeout
def _run_subprocess(cmd, timeout, label):
    """统一 subprocess 包装"""
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=timeout, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"{label} failed: {(r.stderr or r.stdout)[:200]}")
    return r.stdout


def task_elo(timeout: int = 60) -> dict:
    out = _run_subprocess(["python3", "code/data/elo_fetcher.py"], timeout, "elo_fetcher")
    changed = any(kw in out for kw in ["更新", "已更新", "changed"])
    return {"changed": changed, "summary": out.strip().split("\n")[-1][:80]}


# 套利 hook 并发锁：同源连续触发时仅一次在跑（市场每秒可能多次小变化）
_arb_hook_lock = threading.Lock()
_arb_kalshi_hook_lock = threading.Lock()


def _arbitrage_fast_hook_sync(source: str) -> str:
    """S2 同步实现：lib 调用 generate_signals 并写 arbitrage_signals.json。

    完全吞异常（feedback_no_break_existing）。返回 summary 文本，失败返空串。
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "code"))
        from models.arbitrage import generate_signals
        from utils.io import save_output, DATA_OUTPUTS, DATA_RAW

        synth_path = DATA_OUTPUTS / "synthesizer_report.json"
        teams_path = DATA_RAW / "teams.json"
        if not synth_path.exists():
            return ""  # 还没第一次 cascade，跳过

        signals = generate_signals(bankroll=10000)

        model_ts = datetime.fromtimestamp(synth_path.stat().st_mtime).isoformat(timespec="seconds")
        market_ts = datetime.fromtimestamp(teams_path.stat().st_mtime).isoformat(timespec="seconds") if teams_path.exists() else None

        save_output("arbitrage_signals.json", {
            "bankroll": 10000,
            "n_signals": len(signals),
            "signals": signals,
            "market": {
                "name": "2026 FIFA World Cup Winner",
                "venue": "Polymarket",
                "event_url": "https://polymarket.com/event/world-cup-winner",
                "note": "夺冠盘：每队为独立的 Yes/No 合约",
            },
            "_freshness": {
                "trigger": source,
                "refreshed_at": datetime.now().isoformat(timespec="seconds"),
                "model_snapshot_ts": model_ts,
                "market_snapshot_ts": market_ts,
            },
        })

        n_s = sum(1 for s in signals if s.get("grade") == "S")
        n_a = sum(1 for s in signals if s.get("grade") == "A")
        return f"arb({len(signals)}信号/{n_s}S/{n_a}A)"
    except Exception as e:
        log.warning(f"[arbitrage_hook] {source} 同步执行失败: {type(e).__name__}: {e}")
        return ""


def _arbitrage_fast_hook(source: str) -> str:
    """S2 异步入口：起后台线程跑 _arbitrage_fast_hook_sync，主调用方立即返回。

    设计要点：
      - 主调用（task_polymarket/task_kalshi）不再被套利计算阻塞，scheduler tick 永远秒级
      - 用 _arb_hook_lock try-acquire 防同源连击重复计算（市场连续小变化时合并）
      - 异常完全吞掉，不影响数据拉取主流程

    Returns:
        立即返回 " + arb(async)" 标记串（供 summary 拼接），实际结果在后台写文件
    """
    def _bg():
        if not _arb_hook_lock.acquire(blocking=False):
            log.info(f"[arbitrage_hook] {source} 已在跑，本次合并跳过")
            return
        try:
            t0 = time.time()
            result = _arbitrage_fast_hook_sync(source)
            if result:
                log.info(f"[arbitrage_hook] {source} ✓ {result} ({time.time()-t0:.2f}s)")
        finally:
            _arb_hook_lock.release()

    try:
        threading.Thread(target=_bg, daemon=True, name=f"arb_hook_{source}").start()
        return " + arb(async)"
    except Exception as e:
        log.warning(f"[arbitrage_hook] {source} 起线程失败: {type(e).__name__}: {e}")
        return ""


def _arbitrage_kalshi_hook_sync(source: str) -> str:
    """Kalshi 单场套利同步实现（依赖 match_bias.json 已存在）。

    完全吞异常。返回 summary 文本，失败返空串。
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "code"))
        from models.arbitrage_kalshi import generate_kalshi_signals
        from utils.io import save_output, DATA_OUTPUTS

        bias_path = DATA_OUTPUTS / "match_bias.json"
        kalshi_path = DATA_OUTPUTS / "kalshi_match_odds.json"
        if not bias_path.exists() or not kalshi_path.exists():
            return ""  # 首次 cascade 前 / 首次 kalshi 抓取前，跳过

        signals = generate_kalshi_signals(bankroll=10000, min_edge_pp=3.0, future_only=True)
        buys = [s for s in signals if s.get("type") == "BUY"]
        sells = [s for s in signals if s.get("type") == "SELL"]

        bias_ts = datetime.fromtimestamp(bias_path.stat().st_mtime).isoformat(timespec="seconds")
        kalshi_ts = datetime.fromtimestamp(kalshi_path.stat().st_mtime).isoformat(timespec="seconds")

        save_output("arbitrage_kalshi_signals.json", {
            "bankroll": 10000,
            "n_signals": len(signals),
            "n_buy": len(buys),
            "n_sell": len(sells),
            "signals": signals,
            "market": {
                "name": "FIFA World Cup 2026 — Single Game Winner (H/D/A)",
                "venue": "Kalshi",
                "series_ticker": "KXWCGAME",
                "series_url": "https://kalshi.com/markets/kxwcgame",
                "note": "单场胜平负盘：模型 P(H/D/A) vs Kalshi P(H/D/A)；min_edge=3pp",
            },
            "_freshness": {
                "trigger": source,
                "refreshed_at": datetime.now().isoformat(timespec="seconds"),
                "bias_snapshot_ts": bias_ts,
                "kalshi_snapshot_ts": kalshi_ts,
            },
        })

        n_s = sum(1 for s in buys if s.get("grade") == "S")
        n_a = sum(1 for s in buys if s.get("grade") == "A")
        return f"arb_kalshi({len(signals)}/{n_s}S/{n_a}A)"
    except Exception as e:
        log.warning(f"[arbitrage_kalshi_hook] {source} 同步执行失败: {type(e).__name__}: {e}")
        return ""


def _arbitrage_kalshi_hook(source: str) -> str:
    """Kalshi 套利异步入口，结构与 _arbitrage_fast_hook 一致。"""
    def _bg():
        if not _arb_kalshi_hook_lock.acquire(blocking=False):
            log.info(f"[arbitrage_kalshi_hook] {source} 已在跑，本次合并跳过")
            return
        try:
            t0 = time.time()
            result = _arbitrage_kalshi_hook_sync(source)
            if result:
                log.info(f"[arbitrage_kalshi_hook] {source} ✓ {result} ({time.time()-t0:.2f}s)")
        finally:
            _arb_kalshi_hook_lock.release()

    try:
        threading.Thread(target=_bg, daemon=True, name=f"arb_kalshi_hook_{source}").start()
        return " + arb_kalshi(async)"
    except Exception as e:
        log.warning(f"[arbitrage_kalshi_hook] {source} 起线程失败: {type(e).__name__}: {e}")
        return ""


def task_polymarket(timeout: int = 60) -> dict:
    out = _run_subprocess(["python3", "code/data/fetch_polymarket_winner.py"], timeout, "polymarket")
    changed = "变化 ≥0.5pp 的 0 队" not in out
    # 解析 "回写 teams.json: 更新 N 队" —— 真正驱动套利价值变化的链路
    import re
    m_sync = re.search(r"回写 teams\.json[:：]\s*更新\s*(\d+)\s*队", out)
    n_synced = int(m_sync.group(1)) if m_sync else 0

    last_line = [l for l in out.strip().split("\n") if "写入" in l]
    summary = last_line[-1] if last_line else "ok"
    if n_synced > 0:
        summary += f" → teams.json 同步 {n_synced} 队"
    # S2: 价格一变立刻刷套利信号（<1s lib call，失败吞掉）
    # changed=True 或 n_synced>0 都触发：external_predictions 变化必带 teams 变化
    if changed or n_synced > 0:
        summary += _arbitrage_fast_hook("polymarket")
    return {"changed": changed or n_synced > 0, "summary": summary}


def task_kalshi(timeout: int = 60) -> dict:
    _run_subprocess(["python3", "code/data/fetch_kalshi_match_odds.py"], timeout, "kalshi")
    # Kalshi 赔率几乎总在变
    # S2: 价格一变立刻刷套利信号（<1s lib call，失败吞掉）
    summary = ("kalshi refreshed"
               + _arbitrage_fast_hook("kalshi")           # 夺冠盘套利（间接受 Kalshi 影响小，但保持一致）
               + _arbitrage_kalshi_hook("kalshi"))         # 单场套利（直接用新 Kalshi 价 vs 旧模型）
    return {"changed": True, "summary": summary}


def task_group_results(timeout: int = 60) -> dict:
    out = _run_subprocess(["python3", "code/data/fetch_group_results.py"], timeout, "group_results")
    import re
    m = re.search(r"本次更新[:：]\s*(\d+)\s*场", out)
    n_new = int(m.group(1)) if m else 0
    # P0c: 顺手同步 live_state.json
    try:
        _sync_live_state()
    except Exception as e:
        log.warning(f"[group_results] live_state 同步失败: {e}")
    # S1: 同步 current_stage（按日期推断；首次写入或跨阶段时更新）
    try:
        _sync_current_stage_to_live_state()
    except Exception as e:
        log.warning(f"[group_results] current_stage 同步失败: {e}")
    return {"changed": n_new > 0, "summary": f"新增 {n_new} 场结果（+ live_state 同步）"}


def task_injuries(timeout: int = 900) -> dict:
    out = _run_subprocess(["python3", "code/models/run_injury_radar.py", "--all", "--workers", "8"],
                          timeout, "injury_radar")
    import re
    m = re.search(r"发现伤病[:：]\s*(\d+)\s*条", out)
    n_inj = int(m.group(1)) if m else 0
    m2 = re.search(r"失败[:：]\s*(\d+)", out)
    n_fail = int(m2.group(1)) if m2 else 0
    return {"changed": True, "summary": f"{n_inj} 伤 / {n_fail} 失败"}


def task_market_bias(timeout: int = 60) -> dict:
    """P0a: 夺冠盘级偏差检测（与 match_bias 区分，后者是单场）"""
    out = _run_subprocess(["python3", "code/models/market_bias_detector.py"], timeout, "market_bias")
    last = out.strip().split("\n")[-1][:80] if out.strip() else "ok"
    return {"changed": True, "summary": last}


def task_upset_llm(timeout: int = 600) -> dict:
    """P0b: LLM 终审爆冷场（依赖 match_bias 输出）"""
    out = _run_subprocess(["python3", "code/models/run_upset_analysis.py"], timeout, "upset_llm")
    # 解析 "分析完成 N 场"
    import re
    m = re.search(r"(\d+)\s*场", out)
    n = int(m.group(1)) if m else 0
    return {"changed": False, "summary": f"分析 {n} 场"}


def task_critical_nodes(timeout: int = 1800) -> dict:
    """P1a: 17-Agent 关键节点修正（LLM 重活）"""
    out = _run_subprocess(["python3", "code/models/run_critical_nodes.py"], timeout, "critical_nodes")
    last = out.strip().split("\n")[-1][:120] if out.strip() else "ok"
    return {"changed": True, "summary": last}


def task_derived_outputs(timeout: int = 600) -> dict:
    """P1b: 衍生输出并行计算（swarm + arbitrage + finals + group_rank + uncertainty + most_likely_bracket）

    这些彼此独立，可并行；MC 结果是它们的共同输入。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    steps = {
        "swarm":              ["python3", "code/agents/swarm.py"],
        "arbitrage":          ["python3", "code/models/arbitrage.py"],
        "finals_analyzer":    ["python3", "code/models/finals_analyzer.py"],
        "group_rank_dist":    ["python3", "code/models/group_rank_dist.py"],
        "uncertainty":        ["python3", "code/models/uncertainty.py"],
        "most_likely_bracket":["python3", "code/models/most_likely_bracket.py"],
    }
    
    results = {}
    errors = {}
    
    def _one(item):
        name, cmd = item
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, cwd=str(ROOT))
            if r.returncode != 0:
                # 失败时把完整 stderr+stdout 落盘，避免 120 字符截断丢真因
                err_log = ROOT / "logs" / f"derived_{name}.err"
                err_log.parent.mkdir(parents=True, exist_ok=True)
                err_log.write_text(
                    f"=== {name} failed at {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"(elapsed={time.time()-t0:.1f}s, rc={r.returncode}) ===\n"
                    f"--- STDERR ---\n{r.stderr}\n"
                    f"--- STDOUT (tail) ---\n{(r.stdout or '')[-2000:]}\n"
                )
                return name, None, f"rc={r.returncode}: {(r.stderr or r.stdout)[:120]} → see {err_log.name}"
            return name, time.time() - t0, None
        except subprocess.TimeoutExpired as e:
            err_log = ROOT / "logs" / f"derived_{name}.err"
            err_log.parent.mkdir(parents=True, exist_ok=True)
            err_log.write_text(
                f"=== {name} TIMEOUT at {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(timeout={timeout}s) ===\n"
                f"--- partial STDOUT ---\n{(e.stdout or b'').decode(errors='ignore')[-2000:]}\n"
                f"--- partial STDERR ---\n{(e.stderr or b'').decode(errors='ignore')[-2000:]}\n"
            )
            return name, None, f"timeout {timeout}s → see {err_log.name}"
        except Exception as e:
            return name, None, f"{type(e).__name__}: {str(e)[:100]}"
    
    # 6 子任务 → 6 worker，所有派生输出真正并发（之前 4 worker 会让 2 项排队）
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_one, item) for item in steps.items()]
        for fut in as_completed(futures):
            name, dt, err = fut.result()
            if err:
                errors[name] = err
            else:
                results[name] = dt
    
    ok = ", ".join(f"{n}({dt:.0f}s)" for n, dt in results.items())
    err_str = f" / 失败: {list(errors.keys())}" if errors else ""
    return {"changed": False, "summary": f"成功: {ok}{err_str}"}


def task_pi_ratings(timeout: int = 120) -> dict:
    """Pi-ratings 进攻/防守评级"""
    out = _run_subprocess(["python3", "code/models/pi_ratings.py", "train"], timeout, "pi_ratings")
    last = out.strip().split("\n")[-1][:80] if out.strip() else "ok"
    return {"changed": True, "summary": last}


def task_catboost_train(timeout: int = 1800) -> dict:
    """CatBoost 3 分类引擎（每周训一次，吃 pi-ratings 7 维特征）

    周训理由：训练集是 32k 历史 + 增量比赛，每周才有几十场新数据，
    日训意义不大且 CPU 重；周训既能让模型跟上最新形态又不浪费算力。

    输出：data/models/catboost_3way.cbm + meta JSON
    """
    out = _run_subprocess(["python3", "code/models/catboost_engine.py", "train"],
                          timeout, "catboost_train")
    # 解析 RPS / Brier / accuracy 指标行（catboost 输出 "rps: 0.3411" 小写带空格）
    import re
    m_rps = re.search(r"rps[:：]?\s*([0-9.]+)", out, re.IGNORECASE)
    m_brier = re.search(r"brier[:：]?\s*([0-9.]+)", out, re.IGNORECASE)
    m_acc = re.search(r"accuracy[:：]?\s*([0-9.]+)", out, re.IGNORECASE)
    rps = m_rps.group(1) if m_rps else "?"
    brier = m_brier.group(1) if m_brier else "?"
    acc = m_acc.group(1) if m_acc else "?"
    return {"changed": True, "summary": f"RPS={rps} / Brier={brier} / acc={acc}"}


def task_scenarios(timeout: int = 600) -> dict:
    """三情景预测（30k 平衡精度与耗时；100k 时常 600s 超时）"""
    out = _run_subprocess(["python3", "code/models/scenario_engine.py", "30000"], timeout, "scenarios")
    last = out.strip().split("\n")[-1][:80] if out.strip() else "ok"
    return {"changed": False, "summary": last}


def task_multi_model_compare(timeout: int = 120) -> dict:
    """多模型对比（Polymarket / Opta / Reference / 我们）"""
    out = _run_subprocess(["python3", "code/models/multi_model_compare.py"], timeout, "multi_model")
    last = out.strip().split("\n")[-1][:80] if out.strip() else "ok"
    return {"changed": False, "summary": last}


def task_data_quality(timeout: int = 60) -> dict:
    """数据可用性四项检查"""
    out = _run_subprocess(["python3", "code/data/availability_check.py"], timeout, "data_quality")
    last = out.strip().split("\n")[-1][:80] if out.strip() else "ok"
    return {"changed": False, "summary": last}


def task_weather(timeout: int = 120) -> dict:
    """OpenMeteo 比赛场地天气抓取"""
    script = ROOT / "code/data/fetch_weather.py"
    if not script.exists():
        return {"changed": False, "summary": "fetch_weather.py 未实现，跳过"}
    out = _run_subprocess(["python3", "code/data/fetch_weather.py"], timeout, "weather")
    last = out.strip().split("\n")[-1][:80] if out.strip() else "ok"
    return {"changed": False, "summary": last}


def task_schedule_refresh(timeout: int = 60) -> dict:
    """ESPN 公开 API 校验 group_schedule.json，识别 FIFA 微调"""
    out = _run_subprocess(["python3", "code/data/refresh_schedule.py"], timeout, "schedule_refresh")
    import re
    m = re.search(r"发现\s*(\d+)\s*处变化", out)
    n_chg = int(m.group(1)) if m else 0
    # 不触发 cascade（schedule 微调通常是 venue/time，对预测影响小；如要传播需手动）
    return {"changed": n_chg > 0, "summary": f"{n_chg} 处变化（详见 schedule_changelog.json）"}


def task_live_events(timeout: int = 60) -> dict:
    """ESPN 赛中事件流（无 live 时秒退）"""
    out = _run_subprocess(["python3", "code/data/fetch_live_events.py"], timeout, "live_events")
    import re
    m1 = re.search(r"正在进行[:：]?\s*(\d+)", out)
    m2 = re.search(r"刚结束[:：]?\s*(\d+)", out)
    n_live = int(m1.group(1)) if m1 else 0
    n_post = int(m2.group(1)) if m2 else 0
    return {"changed": n_live > 0, "summary": f"{n_live} 进行中 / {n_post} 刚结束"}


def task_h2h(timeout: int = 900) -> dict:
    """双方近 5 场对阵（赛前 24h 窗口）"""
    out = _run_subprocess(["python3", "code/data/fetch_h2h.py",
                            "--hours", "24", "--workers", "4"],
                          timeout, "h2h")
    import re
    m = re.search(r"成功[:：]\s*(\d+)\s*/\s*失败[:：]\s*(\d+)\s*/\s*跳过", out)
    if m:
        n_ok, n_fail = int(m.group(1)), int(m.group(2))
        return {"changed": n_ok > 0, "summary": f"{n_ok} 对阵抓到 / {n_fail} 失败"}
    return {"changed": False, "summary": "无即将开赛场次（24h 内）"}


def task_squad_value(timeout: int = 1200) -> dict:
    """48 队阵容市值（每周 1 次）

    跑完后自动调 sync_squad_value_to_teams.py，把 total_value_eur 回写到
    teams.json[*].squad_value_m_eur，给 synthesizer 用。
    """
    out = _run_subprocess(["python3", "code/data/fetch_squad_value.py",
                            "--workers", "4"],
                          timeout, "squad_value")
    import re
    m = re.search(r"成功[:：]\s*(\d+)\s*/\s*失败[:：]\s*(\d+)\s*/\s*跳过", out)
    if m:
        n_ok, n_fail = int(m.group(1)), int(m.group(2))
        # 仅在有新数据时同步回写（即使失败也不影响主流程）
        n_synced = 0
        if n_ok > 0:
            try:
                sync_out = _run_subprocess(["python3", "code/data/sync_squad_value_to_teams.py"],
                                            60, "squad_value_sync")
                ms = re.search(r"更新\s*(\d+)\s*个字段", sync_out)
                if ms:
                    n_synced = int(ms.group(1))
            except Exception as e:
                logger.warning(f"squad_value 回写失败（不影响主流程）: {e}")
        return {"changed": n_ok > 0, "summary": f"{n_ok} 队抓到 / {n_fail} 失败 / 回写 {n_synced} 队"}
    return {"changed": False, "summary": "无更新"}


def task_referee(timeout: int = 600) -> dict:
    """裁判任命（赛前 30h 窗口）"""
    out = _run_subprocess(["python3", "code/data/fetch_referee.py",
                            "--hours", "30", "--workers", "4"],
                          timeout, "referee")
    import re
    m = re.search(r"成功[:：]\s*(\d+)\s*/\s*失败[:：]\s*(\d+)\s*/\s*跳过", out)
    if m:
        n_ok, n_fail = int(m.group(1)), int(m.group(2))
        return {"changed": n_ok > 0, "summary": f"{n_ok} 场裁判 / {n_fail} 失败"}
    return {"changed": False, "summary": "无即将开赛场次（30h 内）"}


def task_lineups(timeout: int = 600) -> dict:
    """赛前首发阵容（LLM + Serper 真实抓取）
    
    脚本内部只抓"未来 3h 内"的场次。所以每 60min 跑一次：
      - 大部分时段无场次 → 秒退（写空 stub）
      - 接近开赛 → 自动触发 LLM 抓取（已 confirmed 的会跳过，避免重复消耗 quota）
    """
    out = _run_subprocess(["python3", "code/data/fetch_lineups.py",
                            "--hours", "3", "--workers", "8"],
                         timeout, "lineups")
    import re
    m = re.search(r"成功[:：]\s*(\d+)\s*/\s*失败[:：]\s*(\d+)", out)
    if m:
        n_ok, n_fail = int(m.group(1)), int(m.group(2))
        return {"changed": n_ok > 0, "summary": f"{n_ok} 个队抓到 / {n_fail} 失败"}
    # 无场次的情况
    return {"changed": False, "summary": "无即将开赛场次（3h 内）"}


def task_in_match(timeout: int = 120) -> dict:
    """赛中迭代（MRCA+ITA+SOA）—— 仅当 live_events.json 有 live_matches 时实际跑。

    不触发 cascade（避免高频抖动夺冠 MC）；产出 in_match_update.json 直接被 web 读取。

    失败必须吞掉，绝不破坏主流程。
    """
    live_path = ROOT / "data" / "raw" / "live_events.json"
    if not live_path.exists():
        return {"changed": False, "summary": "无 live_events.json"}
    try:
        live = json.load(open(live_path))
        n_live = len(live.get("live_matches", []))
    except Exception as e:
        return {"changed": False, "summary": f"live_events 读取失败: {e}"}

    if n_live == 0:
        return {"changed": False, "summary": "无进行中/刚结束场次，跳过"}

    try:
        # 直接 import 避免 subprocess 开销
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "code"))
        from agents.in_match import run_in_match_pipeline
        out = run_in_match_pipeline(verbose=False)
        if "error" in out:
            return {"changed": False, "summary": f"前置缺失: {out['error']}"}
        n_events = out.get("n_events", 0)
        n_inj = out.get("n_injuries", 0)
        max_swing = out.get("mrca", {}).get("summary", {}).get("max_swing_team", "?")
        max_pp = out.get("mrca", {}).get("summary", {}).get("max_swing_pp", 0)
        return {"changed": n_events > 0 or n_inj > 0,
                "summary": f"{n_live} 场 live / {n_events} 事件 / {n_inj} 伤病 / 最大波动 {max_swing} {max_pp:+.2f}pp"}
    except Exception as e:
        # 任何异常都不能影响 scheduler 主循环
        return {"changed": False, "summary": f"in_match 异常: {type(e).__name__}: {e}"}


def task_live_trading_tick(timeout: int = 30) -> dict:
    """quant 高频量化 tick（每 60s 跑一次）。

    仅当 has_live_match() == True 时实际跑（否则秒退）。
    写 in_match_live.json（snapshot, web 直读）+ in_match_ticks.jsonl（append）。

    任何异常都吞掉，绝不破坏主调度。
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "code"))
        from quant.live_trading_loop import run_tick
        out = run_tick(verbose=False)
        return {
            "changed": out.get("changed", False),
            "summary": out.get("summary", "?"),
        }
    except Exception as e:
        return {"changed": False, "summary": f"live_trading_tick 异常: {type(e).__name__}: {e}"}


def _sync_live_state():
    """把 group_results.json 同步到 live_state.json
    
    更新 matches_played 列表（去重）+ last_updated。
    """
    res_path = ROOT / "data" / "outputs" / "group_results.json"
    live_path = ROOT / "data" / "raw" / "live_state.json"
    if not res_path.exists():
        return
    
    results = json.load(open(res_path)).get("results", {})
    if live_path.exists():
        live = json.load(open(live_path))
    else:
        live = {"tournament": "2026 FIFA World Cup", "current_stage": "group"}
    
    matches_played = []
    for key, r in results.items():
        matches_played.append({
            "date": r.get("date"),
            "team_a": r.get("team_a"),
            "team_b": r.get("team_b"),
            "score": r.get("score_str"),
            "winner": r.get("winner"),
        })
    
    live["matches_played"] = matches_played
    live["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if matches_played and not live.get("started_at"):
        # 取最早的日期
        dates = [m["date"] for m in matches_played if m.get("date")]
        if dates:
            live["started_at"] = min(dates)
    
    live_path.write_text(json.dumps(live, ensure_ascii=False, indent=2))


# ─────────────────────── cascade 时间序列快照（feedback_no_break_existing） ───────────────────────
def _trim_timeseries(path: Path, max_lines: int = 500) -> None:
    """裁剪 jsonl 文件最多保留最新 max_lines 行"""
    try:
        if not path.exists():
            return
        lines = path.read_text().splitlines()
        if len(lines) > max_lines:
            path.write_text("\n".join(lines[-max_lines:]) + "\n")
    except Exception:
        pass


def _snapshot_cascade() -> None:
    """cascade 完成后追加一行时间序列快照（失败完全吞，不影响主流程）。

    输出：data/outputs/snapshots/timeseries.jsonl  （JSON Lines，每行一条）
    字段：ts, champion{team→pct}, final{team→pct}, semifinal{team→pct},
          group_first{group→{team→pct}}, final_top[{...}]
    """
    try:
        from utils.io import DATA_OUTPUTS
        snap_dir = DATA_OUTPUTS / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        record: dict = {"ts": datetime.now().isoformat(timespec="seconds")}

        # MC 100k：6 个阶段晋级率（0~1 → 百分比）
        mc_path = DATA_OUTPUTS / "mc_simulation_n100000.json"
        if mc_path.exists():
            try:
                mc = json.load(open(mc_path))
                stages = ("round_of_32", "round_of_16", "quarterfinal",
                          "semifinal", "final", "champion")
                stage_data = {s: {} for s in stages}
                for t, d in mc.items():
                    if not isinstance(d, dict):
                        continue
                    for s in stages:
                        if s in d:
                            stage_data[s][t] = round(d[s] * 100, 4)
                for s in stages:
                    if stage_data[s]:
                        record[s] = stage_data[s]
            except Exception as e:
                log.warning(f"[snapshot] mc_simulation 解析失败: {e}")

        # group_rank_dist
        gr_path = DATA_OUTPUTS / "group_rank_dist.json"
        if gr_path.exists():
            try:
                gr = json.load(open(gr_path))
                groups = gr.get("groups", {})
                gf = {}
                for gname, gdata in groups.items():
                    if not isinstance(gdata, dict):
                        continue
                    gf[gname] = {
                        team: round(info.get("first_pct", 0), 4)
                        for team, info in gdata.items()
                        if isinstance(info, dict)
                    }
                if gf:
                    record["group_first"] = gf
            except Exception as e:
                log.warning(f"[snapshot] group_rank_dist 解析失败: {e}")

        # finals_matchups Top 10（schema：{top_matchups: [{team_a, team_b, probability(小数)}]}）
        fn_path = DATA_OUTPUTS / "finals_matchups.json"
        if fn_path.exists():
            try:
                fn = json.load(open(fn_path))
                raw = fn.get("top_matchups") or fn.get("final_matchups") or []
                top = []
                for item in raw[:10]:
                    a = item.get("team_a"); b = item.get("team_b")
                    p = item.get("probability")
                    if p is None:
                        p = item.get("prob_pct", 0) / 100.0
                    if a and b:
                        top.append({"team_a": a, "team_b": b,
                                    "prob_pct": round(float(p) * 100, 4)})
                if top:
                    record["final_top"] = top
            except Exception as e:
                log.warning(f"[snapshot] finals_matchups 解析失败: {e}")

        # 追加写入 jsonl
        ts_path = snap_dir / "timeseries.jsonl"
        with open(ts_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 滚动保留
        _trim_timeseries(ts_path, max_lines=500)

        log.info(
            f"[cascade] 时间序列已记录 (teams={len(record.get('champion', {}))}, "
            f"groups={len(record.get('group_first', {}))})"
        )
    except Exception as e:
        log.warning(f"[cascade] 快照写盘失败（已忽略）: {e}")


def _freeze_pre_kickoff_predictions(window_min: int = 210) -> None:
    """cascade 完成后冻结「即将开赛」场次的对阵预测。

    诉求：比赛一开始就不允许再修改预测结果（仅锁单场对阵预测）。

    机制：
      - cascade 完成后，扫描 (kickoff - window_min, kickoff] 区间内尚未开赛的场次
      - 调 _quick_match_preview 拿到当下两个通道（base / ai_phase3）的对阵预测
      - 写入/合并到 data/outputs/frozen_predictions.json
      - 写入策略：「保留最早冻结值」——一旦该场首次写入，后续 cascade 不再覆盖
        （避免随 cascade 抖动覆盖；3h mark 触发的 cascade 是首个落入窗口的，会落盘）
      - kickoff 之后，API 层（tournament_api）读到 frozen 即覆盖 prediction

    所有异常吞掉，绝不影响主流程。
    window_min 取 210（3.5h）为缓冲：3h mark cascade 可能耗时 30-60s，
    且非 cascade 触发（如 kalshi 加密期间）也可能落入窗口。
    """
    try:
        from utils.io import DATA_OUTPUTS, DATA_RAW
        from match_aware import matches_to_freeze
        from web.tournament_api import _quick_match_preview

        # 赛前 pre_window 内（默认 210min ≈ 3.5h）+ 赛后 post_window 内（4h 兜底）
        # 后者是为了 kickoff 当下没刚好有 cascade 时，开赛后第一次 cascade 也能锁住
        candidates = matches_to_freeze(pre_window_min=window_min, post_window_min=240)
        if not candidates:
            return

        frozen_path = DATA_OUTPUTS / "frozen_predictions.json"
        if frozen_path.exists():
            try:
                store = json.load(open(frozen_path))
            except Exception:
                store = {"matches": {}}
        else:
            store = {"matches": {}}
        store.setdefault("matches", {})

        n_new = 0
        for m in candidates:
            ta, tb = m.get("team_a"), m.get("team_b")
            if not ta or not tb:
                continue
            key = f"{ta} vs {tb} @ {m.get('kickoff').strftime('%Y-%m-%d')}"
            # 保留最早值——已冻结过就跳过
            if key in store["matches"]:
                continue

            entry = {
                "team_a": ta,
                "team_b": tb,
                "match_id": m.get("match_id"),
                "kickoff": m["kickoff"].strftime("%Y-%m-%d %H:%M"),
                "frozen_at": datetime.now().isoformat(timespec="seconds"),
                "channels": {},
            }

            # 两个通道分别冻结（前端按 channel 切换）
            for ch in ("base", "ai_phase3"):
                try:
                    pred = _quick_match_preview(ta, tb,
                                                  venue_city=m.get("venue_city"),
                                                  channel=ch)
                    if "error" in pred:
                        continue
                    entry["channels"][ch] = {
                        "p_win_a": pred["p_win_a"],
                        "p_draw": pred["p_draw"],
                        "p_win_b": pred["p_win_b"],
                        "predicted_winner": pred["predicted_winner"],
                        "winner_confidence_pct": pred["winner_confidence_pct"],
                        "lambda_a": pred["lambda_a"],
                        "lambda_b": pred["lambda_b"],
                        "forecast_format": pred.get("forecast_format", {}),
                    }
                except Exception as e:
                    log.warning(f"[freeze] {ta} vs {tb} channel={ch} 预测失败: {e}")

            if entry["channels"]:
                store["matches"][key] = entry
                n_new += 1
                log.info(f"[freeze] 锁定预测 {ta} vs {tb} kickoff={entry['kickoff']}")

        if n_new > 0:
            store["updated_at"] = datetime.now().isoformat(timespec="seconds")
            store["n_total"] = len(store["matches"])
            with open(frozen_path, "w") as f:
                json.dump(store, f, ensure_ascii=False, indent=2)
            log.info(f"[freeze] frozen_predictions.json 新增 {n_new} 场（共 {len(store['matches'])} 场）")
    except Exception as e:
        log.warning(f"[freeze] 冻结预测失败（已忽略）: {e}")


def task_cascade() -> dict:
    """五层架构级联重算（P1: cascade 流水线扩展）

    L3 合成 → L4 模拟 → L5 后处理 三层流水线：

      Step 1 (L3): synthesizer                                       [序列，下游强依赖]
      Step 2 (L4+L5a): monte_carlo 100k  ‖  match_bias_detector      [并行]
      Step 3 (L5b): calibrator ‖ report_writer ‖ derived_outputs(6)  [并行，全包 try/except]
      Step 4 (L5c): scenarios ‖ multi_model_compare                  [并行，全包 try/except]

    Step 3 是"派生输出流水线"——MC 输出后立刻刷新 calibration / 三语报告 /
    swarm+arbitrage+finals+group_rank+uncertainty+most_likely_bracket（derived_outputs 内部 6 项并行）。
    Step 4 是"应用层流水线"——MC 输出后立刻刷新 scenario / 多模型对比，让 web 端的
    情景预测与对比图秒级跟上 MC。重活（critical_nodes 1800s / upset_llm 600s）
    保留独立 timer，不塞进 cascade，避免拖垮节奏。

    关键约束：Step 3/4 任一失败必须吞掉，绝不破坏 Step 1/2 主流程（feedback_no_break_existing）。
    Step 1/2 失败仍向上抛（它们是主管线，调用方需要知道）。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ─── Step 1 (L3): synthesizer 必须先跑 ───
    t0 = time.time()
    r = subprocess.run(["python3", "code/models/synthesizer.py"],
                       capture_output=True, text=True, timeout=120, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"cascade 'synthesizer' failed: {r.stderr[:200]}")
    t_synth = time.time() - t0

    # ─── Step 2 (L4 + L5a): MC 和 match_bias 并行 ───
    parallel_steps_2 = {
        "monte_carlo": ["python3", "code/models/monte_carlo.py", "100000"],
        "match_bias":  ["python3", "code/models/match_bias_detector.py"],
    }

    def _run_strict(name_cmd):
        """Step 2 用：失败抛异常（主管线，必须可靠）"""
        name, cmd = name_cmd
        ts = time.time()
        rr = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=600, cwd=str(ROOT))
        if rr.returncode != 0:
            raise RuntimeError(f"cascade '{name}' failed: {rr.stderr[:200]}")
        return name, time.time() - ts

    t1 = time.time()
    elapsed_2 = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_run_strict, item): item[0] for item in parallel_steps_2.items()}
        for fut in as_completed(futures):
            name, dt = fut.result()
            elapsed_2[name] = dt
    t_step2 = time.time() - t1

    # match_bias 刚刷完，异步触发 Kalshi 单场套利信号生成（用最新模型 vs 最新 Kalshi 价）
    try:
        _arbitrage_kalshi_hook("cascade")
    except Exception as e:
        log.warning(f"[cascade] kalshi_arb_hook 起线程失败: {type(e).__name__}: {e}")

    # ─── Step 2.5: AI 加权 MC 通道（phase3）───
    # 双通道架构：base 通道走纯 MC（Step 2），AI 通道走 phase3（分段 ELO_PER_PP，逐场注入 ΔE）
    # 失败吞掉——AI 通道是"增强"不是主管线，base 通道才是 cascade 的命脉
    # 命名约定：mc_simulation_n100000_ai_phase3.json + synthesizer_report_ai_phase3.json
    t_phase3_mc = 0.0
    t_phase3_synth = 0.0
    t_phase3_ml = 0.0
    t_phase3_grd = 0.0
    phase3_status = "skipped"
    try:
        t_p3_a = time.time()
        rr_p3_mc = subprocess.run(
            ["python3", "code/run_ai_weighted_mc.py", "100000", "10", "phase3"],
            capture_output=True, text=True, timeout=900, cwd=str(ROOT))
        t_phase3_mc = time.time() - t_p3_a
        if rr_p3_mc.returncode != 0:
            log.warning(f"[cascade] phase3 MC 失败 ({t_phase3_mc:.0f}s): {(rr_p3_mc.stderr or rr_p3_mc.stdout)[:200]}")
            phase3_status = "mc_failed"
        else:
            t_p3_b = time.time()
            rr_p3_synth = subprocess.run(
                ["python3", "code/models/synthesizer.py",
                 "mc_simulation_n100000_ai_phase3.json",
                 "synthesizer_report_ai_phase3.json"],
                capture_output=True, text=True, timeout=120, cwd=str(ROOT))
            t_phase3_synth = time.time() - t_p3_b
            if rr_p3_synth.returncode != 0:
                log.warning(f"[cascade] phase3 synth 失败 ({t_phase3_synth:.0f}s): {(rr_p3_synth.stderr or rr_p3_synth.stdout)[:200]}")
                phase3_status = "synth_failed"
            else:
                # ─── 新增 Step 2.5c: phase3 最可能剧本派生 ───
                # 让 r32/r16/qf/sf/final 全链路按 phase3 数据预测，而不仅是冠军榜
                t_p3_c = time.time()
                rr_p3_ml = subprocess.run(
                    ["python3", "code/models/most_likely_bracket.py",
                     "100000", "ai_phase3"],
                    capture_output=True, text=True, timeout=600, cwd=str(ROOT))
                t_phase3_ml = time.time() - t_p3_c
                if rr_p3_ml.returncode != 0:
                    log.warning(f"[cascade] phase3 most_likely 失败 ({t_phase3_ml:.0f}s): "
                                f"{(rr_p3_ml.stderr or rr_p3_ml.stdout)[:200]}")
                    phase3_status = "ml_failed"  # synth 已成功，仅剧本派生失败 → 冠军榜可用，对阵回退 base
                else:
                    # ─── 新增 Step 2.5d: phase3 小组第 1 名概率派生 ───
                    # 让小组赛 first_pct 在 phase3 通道也能体现 ΔE 校准（否则 base/phase3 first_pct 完全相同）
                    t_p3_d = time.time()
                    rr_p3_grd = subprocess.run(
                        ["python3", "code/models/group_rank_dist.py",
                         "100000", "ai_phase3"],
                        capture_output=True, text=True, timeout=600, cwd=str(ROOT))
                    t_phase3_grd = time.time() - t_p3_d
                    if rr_p3_grd.returncode != 0:
                        log.warning(f"[cascade] phase3 group_rank_dist 失败 ({t_phase3_grd:.0f}s): "
                                    f"{(rr_p3_grd.stderr or rr_p3_grd.stdout)[:200]}")
                        phase3_status = "grd_failed"  # most_likely 已成功，仅小组赛 first_pct 失败 → 小组 tab 回退 base
                    else:
                        phase3_status = "ok"
                        log.info(f"[cascade] phase3 通道 OK: MC {t_phase3_mc:.0f}s + "
                                 f"synth {t_phase3_synth:.0f}s + most_likely {t_phase3_ml:.0f}s + "
                                 f"group_rank_dist {t_phase3_grd:.0f}s")
    except Exception as e:
        log.warning(f"[cascade] phase3 通道异常: {type(e).__name__}: {e}")
        phase3_status = f"exception:{type(e).__name__}"

    # ─── Step 3 (L5b): 后处理流水线（calibrator + report_writer + derived_outputs）───
    # 这三组都依赖 Step 2 的 MC 输出，彼此独立可并行。
    # 任何一个失败都吞掉，不影响 cascade 整体（L5 后处理失败 != 主管线失败）。
    parallel_steps_3 = {
        "calibrator":     ["python3", "code/models/calibrator.py", "report"],
        "report_writer":  ["python3", "code/models/report_writer.py", "all"],
        # derived_outputs 自己内部已经是 6 项并行（swarm/arbitrage/finals/group_rank/uncertainty/most_likely_bracket）
        # 这里复用 task_derived_outputs 函数，避免重复维护清单
    }

    def _run_safe(name_cmd):
        """Step 3 用：失败吞掉返回 error 字段（后处理，绝不破坏主流程）"""
        name, cmd = name_cmd
        ts = time.time()
        try:
            rr = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600, cwd=str(ROOT))
            if rr.returncode != 0:
                return name, None, f"{(rr.stderr or rr.stdout)[:120]}"
            return name, time.time() - ts, None
        except Exception as e:
            return name, None, str(e)[:120]

    t2 = time.time()
    elapsed_3 = {}
    errors_3 = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        # 子进程任务（calibrator + report_writer）
        futures = {pool.submit(_run_safe, item): item[0] for item in parallel_steps_3.items()}
        # derived_outputs 作为函数调用（已自带 try/except 和并行）
        fut_derived = pool.submit(_safe_call, task_derived_outputs, 600)
        futures[fut_derived] = "derived_outputs"

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                if name == "derived_outputs":
                    result = fut.result()
                    if isinstance(result, dict):
                        elapsed_3[name] = time.time() - t2
                        # derived_outputs summary 自带错误信息
                    else:
                        errors_3[name] = "unexpected return"
                else:
                    n, dt, err = fut.result()
                    if err:
                        errors_3[n] = err
                    else:
                        elapsed_3[n] = dt
            except Exception as e:
                # 双保险：even if _run_safe 也炸了
                errors_3[name] = f"outer: {type(e).__name__}: {e}"
    t_step3 = time.time() - t2

    # ─── Step 4 (L5c): 轻量级应用层并行刷新（scenarios + multi_model_compare）───
    # 这两个都依赖 Step 2 的 MC 输出，写各自独立 JSON，单跑都很快（~30s 内）。
    # 重活（critical_nodes 1800s、upset_llm 600s）保留独立 timer，不塞进 cascade
    # 以免拖垮 cascade 节奏。
    # 失败全部吞掉（feedback_no_break_existing）。
    parallel_steps_4 = {
        # scenarios 降到 30k：100k 在 cascade 内常 600s 超时（与 derived_outputs 抢 CPU）
        "scenarios":           ["python3", "code/models/scenario_engine.py", "30000"],
        "multi_model_compare": ["python3", "code/models/multi_model_compare.py"],
    }

    t3 = time.time()
    elapsed_4 = {}
    errors_4 = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_run_safe, item): item[0] for item in parallel_steps_4.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                n, dt, err = fut.result()
                if err:
                    errors_4[n] = err
                else:
                    elapsed_4[n] = dt
            except Exception as e:
                errors_4[name] = f"outer: {type(e).__name__}: {e}"
    t_step4 = time.time() - t3

    # 组装 summary
    s3_ok = ",".join(f"{n}({elapsed_3[n]:.0f}s)" for n in elapsed_3) if elapsed_3 else "无"
    s3_err = f" / L5b失败:{list(errors_3.keys())}" if errors_3 else ""
    s4_ok = ",".join(f"{n}({elapsed_4[n]:.0f}s)" for n in elapsed_4) if elapsed_4 else "无"
    s4_err = f" / L5c失败:{list(errors_4.keys())}" if errors_4 else ""

    phase3_seg = (
        f" | AI通道 {phase3_status}({t_phase3_mc:.0f}s+{t_phase3_synth:.0f}s+{t_phase3_ml:.0f}s+{t_phase3_grd:.0f}s)"
        if phase3_status != "skipped" else ""
    )

    summary = (
        f"L3 synth {t_synth:.0f}s | "
        f"L4 mc {elapsed_2.get('monte_carlo',0):.0f}s ‖ L5a bias {elapsed_2.get('match_bias',0):.0f}s = {t_step2:.0f}s | "
        f"L5b {s3_ok} = {t_step3:.0f}s{s3_err} | "
        f"L5c {s4_ok} = {t_step4:.0f}s{s4_err}"
        f"{phase3_seg}"
    )
    return {"changed": False, "summary": summary}


def _safe_call(fn, *args):
    """包装函数调用，任何异常都返回 dict，不抛"""
    try:
        return fn(*args)
    except Exception as e:
        return {"changed": False, "summary": f"call failed: {type(e).__name__}: {e}"}


# ─────────── Scheduler 主类 ───────────
class Scheduler:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self._dirty_lock = threading.Lock()
        self._dirty: set = set()
        self._last_cascade: Optional[datetime] = None
        self._cascade_lock = threading.Lock()  # 防止并发 cascade
        self.running = True
        self._cfg = load_config()
        self._cfg_mtime: Optional[float] = None
        self._cascade_debounce = timedelta(minutes=int(self._cfg["_cascade"]["debounce_min"]))
        self._load_state()

    def register(self, task: Task):
        self.tasks[task.name] = task

    def _maybe_reload_config(self):
        """检测 config 文件 mtime 变化，热重载"""
        try:
            if not CONFIG_PATH.exists():
                return
            mt = CONFIG_PATH.stat().st_mtime
            if self._cfg_mtime is None:
                self._cfg_mtime = mt
                return
            if mt == self._cfg_mtime:
                return
            log.info(f"[config] 检测到 {CONFIG_PATH.name} 变化，热重载")
            self._cfg = load_config()
            self._cfg_mtime = mt
            self._cascade_debounce = timedelta(minutes=int(self._cfg["_cascade"]["debounce_min"]))
            for name, task in self.tasks.items():
                if name in self._cfg["tasks"]:
                    old_int = int(task.interval.total_seconds() / 60)
                    old_en = task.enabled
                    task.apply_cfg(self._cfg["tasks"][name])
                    new_int = int(task.interval.total_seconds() / 60)
                    if old_int != new_int or old_en != task.enabled:
                        log.info(f"  - {name}: interval {old_int}→{new_int}min, enabled {old_en}→{task.enabled}")
        except Exception as e:
            log.warning(f"[config] 重载失败: {e}")

    def mark_dirty(self, source: str):
        """标记下游需要 cascade"""
        with self._dirty_lock:
            self._dirty.add(source)
            log.info(f"[dirty] {source} → cascade pending")

    def _maybe_cascade(self):
        """如有 dirty 且距上次 cascade 超过 debounce，则跑一次 cascade
        
        改造：异步触发（独立线程），不阻塞主 tick；用 _cascade_lock 防并发。
        """
        with self._dirty_lock:
            if not self._dirty:
                return
            now = datetime.now()
            if self._last_cascade and (now - self._last_cascade) < self._cascade_debounce:
                return  # debounce
            sources = sorted(self._dirty)
            self._dirty.clear()
            self._last_cascade = now
        
        def _do_cascade():
            if not self._cascade_lock.acquire(blocking=False):
                log.info(f"[cascade] 已在跑，本次跳过（来源: {','.join(sources)}）")
                return
            try:
                log.info(f"[cascade] 触发（异步），来源: {','.join(sources)}")
                t0 = time.time()
                result = task_cascade()
                log.info(f"[cascade] ✓ {time.time()-t0:.1f}s {result.get('summary','')}")
                _snapshot_cascade()  # 时间序列快照（try/except 全包裹，失败不影响主流程）
                _freeze_pre_kickoff_predictions()  # 冻结即将开赛场次的预测（仅单场对阵）
            except Exception as e:
                log.error(f"[cascade] ✗ {e}")
            finally:
                self._cascade_lock.release()
        
        threading.Thread(target=_do_cascade, daemon=True).start()

    def _load_state(self):
        if not STATE_PATH.exists():
            return
        try:
            data = json.load(open(STATE_PATH))
            for name, info in data.get("tasks", {}).items():
                # 注意：tasks 在 register() 后才存在，这里把数据缓存到 _restored
            # noop：实际 restore 在 register 之后通过 apply_restored 处理
                pass
            self._restored = data.get("tasks", {})
        except Exception:
            self._restored = {}

    def apply_restored(self):
        """注册完任务后调用，把上次的 last_run / last_success 恢复进 task 对象"""
        for name, info in getattr(self, "_restored", {}).items():
            if name not in self.tasks:
                continue
            t = self.tasks[name]
            if info.get("last_run"):
                t.last_run = datetime.fromisoformat(info["last_run"])
            if info.get("last_success"):
                t.last_success = datetime.fromisoformat(info["last_success"])
            t.n_runs = info.get("n_runs", 0)
            t.n_failures = info.get("n_failures", 0)

    def save_state(self):
        data = {
            "_saved_at": datetime.now().isoformat(timespec="seconds"),
            "tasks": {
                name: {
                    "last_run": t.last_run.isoformat(timespec="seconds") if t.last_run else None,
                    "last_success": t.last_success.isoformat(timespec="seconds") if t.last_success else None,
                    "last_error": t.last_error,
                    "n_runs": t.n_runs,
                    "n_failures": t.n_failures,
                    "interval_min": int(t.interval.total_seconds() / 60),
                    "description": t.description,
                }
                for name, t in self.tasks.items()
            }
        }
        STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def bootstrap_all(self, include_heavy_llm: bool = False, max_workers: int = 6):
        """冷启动全量刷新：按依赖关系把所有板块跑一遍，然后再进入正常周期。

        分批策略（依赖关系驱动）：
          批 1 (并行 max_workers): 独立原子数据 — elo/weather/schedule_refresh/lineups/h2h/
              referee/squad_value/kalshi/polymarket/group_results/live_events/in_match/
              data_quality/pi_ratings/market_bias
          批 2 (串行): cascade — synthesizer → MC → match_bias → derived_outputs（含
              arbitrage + arbitrage_kalshi 异步 hook）→ scenarios ‖ multi_model_compare
          批 3 (并行 LLM, 仅 include_heavy_llm=True): injuries / upset_llm / critical_nodes

        所有失败完全吞掉（feedback_no_break_existing）；bootstrap 失败不影响主循环启动。
        """
        log.info("=" * 60)
        log.info("🚀 BOOTSTRAP: 冷启动全量刷新开始")
        log.info("=" * 60)
        t_total = time.time()

        # ─── 批 1：原子数据并行 ───
        # 顺序无关紧要，按 interval 短的优先（更可能秒退）
        batch1_names = [
            "live_events", "in_match", "data_quality",
            "kalshi", "polymarket", "group_results",
            "lineups", "weather", "h2h", "referee",
            "pi_ratings", "market_bias", "elo",
            "schedule_refresh", "squad_value",
        ]
        # 过滤：跳过未注册 / disabled / stage_overrides 设为 -1（当前阶段已禁用）
        from concurrent.futures import ThreadPoolExecutor, as_completed
        runnable = []
        for n in batch1_names:
            t = self.tasks.get(n)
            if not t or not t.enabled:
                log.info(f"  [bootstrap b1] {n}: 跳过（未注册/禁用）")
                continue
            # 检查 stage_overrides 是否 -1
            try:
                cur_stage = _read_current_stage()
                ov = (t.stage_overrides or {}).get(cur_stage)
                if ov == -1:
                    log.info(f"  [bootstrap b1] {n}: 跳过（阶段 {cur_stage} 暂停）")
                    continue
            except Exception:
                pass
            runnable.append(n)

        log.info(f"批 1: 并行刷新 {len(runnable)} 个原子任务（max_workers={max_workers}）")
        t_b1 = time.time()
        results_b1 = {}

        def _safe_run(name):
            t0 = time.time()
            try:
                self.tasks[name].run(self)
                return name, time.time() - t0, None
            except Exception as e:
                return name, time.time() - t0, f"{type(e).__name__}: {e}"

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_safe_run, n): n for n in runnable}
            for fut in as_completed(futs):
                name, dt, err = fut.result()
                results_b1[name] = (dt, err)
                if err:
                    log.warning(f"  [bootstrap b1] ✗ {name}: {err} ({dt:.1f}s)")
                else:
                    log.info(f"  [bootstrap b1] ✓ {name} ({dt:.1f}s)")

        log.info(f"批 1 完成: {sum(1 for _,e in results_b1.values() if e is None)}/{len(results_b1)} 成功，{time.time()-t_b1:.1f}s")

        # ─── 批 2：cascade ───
        log.info("批 2: 触发 cascade（synthesizer→MC→match_bias→derived/arbitrage/scenarios）")
        t_b2 = time.time()
        # 强制清 dirty 标记并直接调 task_cascade（绕过 debounce）
        try:
            with self._dirty_lock:
                self._dirty.clear()
                self._last_cascade = datetime.now()
            result = task_cascade()
            log.info(f"批 2 完成: cascade {time.time()-t_b2:.1f}s · {result.get('summary','')}")
        except Exception as e:
            log.warning(f"批 2 失败（不影响后续）: {type(e).__name__}: {e}")

        # ─── 批 3：重 LLM 任务（默认跳过）───
        if include_heavy_llm:
            batch3_names = ["injuries", "upset_llm", "critical_nodes"]
            log.info(f"批 3: 并行触发重 LLM 任务 {batch3_names}（异步，可能耗时数分钟）")
            for n in batch3_names:
                t = self.tasks.get(n)
                if not t or not t.enabled:
                    continue
                # 后台跑，不等
                threading.Thread(target=t.run, args=(self,), daemon=True, name=f"bootstrap_{n}").start()
                log.info(f"  [bootstrap b3] 已派发 {n} 到后台")
        else:
            log.info("批 3: 跳过重 LLM 任务（injuries/upset_llm/critical_nodes）；按各自周期触发即可")

        log.info("=" * 60)
        log.info(f"🚀 BOOTSTRAP 完成 · 总耗时 {time.time()-t_total:.1f}s · 进入正常周期")
        log.info("=" * 60)

    def run_loop(self, tick_sec: int = 60):
        """主循环：每 tick_sec 秒检查一次有无该跑的任务"""
        log.info(f"调度器启动，注册 {len(self.tasks)} 个任务，tick={tick_sec}s")
        for n, t in self.tasks.items():
            log.info(f"  - {n}: 每 {int(t.interval.total_seconds()/60)}min · {t.description}")
        
        # match_aware 「赛前 mark 已触发」去重表（避免同场同 mark 重复 mark_dirty）
        # key: (match_id, min_before)；进入下一日自动失效（清理逻辑见 _maybe_pre_match_cascade）
        self._pre_match_fired: set = set()

        while self.running:
            self._maybe_reload_config()  # 每 tick 检查 config 变化
            now = datetime.now()
            for task in self.tasks.values():
                if task.should_run(now):
                    # 在独立线程里跑，避免阻塞 tick
                    threading.Thread(
                        target=task.run, args=(self,), daemon=True
                    ).start()

            # match_aware：检查赛前 cascade mark（30/15/5min before kickoff）
            self._maybe_pre_match_cascade(now, tick_sec)

            # 检查 cascade（dirty 防抖触发）
            self._maybe_cascade()

            time.sleep(tick_sec)

    def _maybe_pre_match_cascade(self, now: datetime, tick_sec: int):
        """match_aware：当前 tick 若命中某场比赛的赛前 mark（30/15/5min），mark_dirty 触发 cascade。

        不阻塞主流程；任何异常都吞掉。
        """
        try:
            from match_aware import pre_match_cascade_marks
            marks = pre_match_cascade_marks(now=now, lookback_sec=tick_sec + 5)
            if not marks:
                return
            for min_before, m in marks:
                key = (m.get("match_id"), int(min_before), now.strftime("%Y-%m-%d"))
                if key in self._pre_match_fired:
                    continue
                self._pre_match_fired.add(key)
                log.info(
                    f"[match_aware] 赛前 {min_before}min 触发 cascade: "
                    f"{m.get('team_a')} vs {m.get('team_b')} @ "
                    f"{m['kickoff'].strftime('%Y-%m-%d %H:%M')}"
                )
                self.mark_dirty(f"pre_match_{min_before}min")
            # 当 fired 集合超过 200 项时清理，避免长期内存增长
            if len(self._pre_match_fired) > 200:
                today = now.strftime("%Y-%m-%d")
                self._pre_match_fired = {k for k in self._pre_match_fired if k[2] == today}
        except Exception as e:
            log.warning(f"[match_aware] pre-match cascade 检查失败（已忽略）: {e}")

    def status_report(self) -> str:
        # S2: 显示当前阶段（让用户看到 stage_overrides 是基于哪个阶段生效）
        current_stage = _read_current_stage()
        lines = [
            "=== Scheduler Status ===",
            f"配置: {CONFIG_PATH}",
            f"Cascade debounce: {int(self._cascade_debounce.total_seconds()/60)}min",
            f"🎯 当前阶段: {current_stage}",
            "",
        ]
        now = datetime.now()
        for name, t in self.tasks.items():
            interval_default = int(t.interval.total_seconds() / 60)
            eff = t._effective_interval()
            # 阶段被暂停
            if eff is None:
                lines.append(f"  💤 {name:<18} default {interval_default:>4}m · 阶段[{current_stage}]暂停 · {t.description[:40]}")
                continue
            interval_eff = int(eff.total_seconds() / 60)
            # 是否被 stage override（不等于默认）
            stage_marker = f"→{interval_eff}m" if interval_eff != interval_default else ""

            if not t.enabled:
                lines.append(f"  ⏸ {name:<18} every {interval_default:>4}m · DISABLED · {t.description}")
                continue
            last_run = t.last_run.strftime("%Y-%m-%d %H:%M") if t.last_run else "never"

            if t.last_run:
                next_run = t.last_run + eff
                eta_min = int((next_run - now).total_seconds() / 60)
                eta = f"in {eta_min}m" if eta_min >= 0 else f"overdue {-eta_min}m"
            else:
                eta = "ASAP"

            health = "✓" if (t.last_success and (now - t.last_success) < eff * 2) else "⚠"
            err = f" | err: {t.last_error[:60]}" if t.last_error else ""
            cas = " ⚡" if t.triggers_cascade else "  "
            lines.append(f"  {health}{cas} {name:<18} every {interval_default:>4}m{stage_marker:<6} · last: {last_run} · next: {eta} · runs: {t.n_runs} fails: {t.n_failures}{err}")
        return "\n".join(lines)


# ─────────── 入口 ───────────
# task_name → fn 映射（config 控制 interval/enabled/timeout）
TASK_REGISTRY = {
    "kalshi":              task_kalshi,
    "polymarket":          task_polymarket,
    "group_results":       task_group_results,
    "elo":                 task_elo,
    "injuries":            task_injuries,
    "market_bias":         task_market_bias,
    "upset_llm":           task_upset_llm,
    "critical_nodes":      task_critical_nodes,
    "derived_outputs":     task_derived_outputs,
    "pi_ratings":          task_pi_ratings,
    "catboost_train":      task_catboost_train,
    "scenarios":           task_scenarios,
    "multi_model_compare": task_multi_model_compare,
    "data_quality":        task_data_quality,
    "weather":             task_weather,
    "lineups":             task_lineups,
    "schedule_refresh":    task_schedule_refresh,
    "live_events":         task_live_events,
    "h2h":                 task_h2h,
    "squad_value":         task_squad_value,
    "referee":             task_referee,
    "in_match":            task_in_match,
    "live_trading_tick":   task_live_trading_tick,
}


def build_scheduler() -> Scheduler:
    s = Scheduler()
    for name, fn in TASK_REGISTRY.items():
        cfg = s._cfg["tasks"].get(name, {})
        s.register(Task(name, fn, cfg))
    s.apply_restored()
    s._cfg_mtime = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else None
    return s


def main():
    ap = argparse.ArgumentParser(
        description="WorldCup 数据自动调度器",
        epilog=f"频率配置: {CONFIG_PATH}（修改后无需重启，热重载）",
    )
    ap.add_argument("--status", action="store_true", help="显示状态后退出")
    ap.add_argument("--once", metavar="TASK", help="只跑一次某个任务后退出")
    ap.add_argument("--list-tasks", action="store_true", help="列出所有已注册任务")
    ap.add_argument("--match-aware", action="store_true",
                    help="诊断 match_aware：当天比赛 + 各任务的赛程感知决策")
    ap.add_argument("--tick", type=int, default=60,
                    help="主循环 tick 秒数（默认 60s）")
    ap.add_argument("--bootstrap", choices=["none", "default", "full"], default="default",
                    help="冷启动全量刷新: none=跳过, default=刷新所有非 LLM 板块（推荐）, "
                         "full=连重 LLM 也跑（injuries/upset_llm/critical_nodes，耗时数分钟）")
    args = ap.parse_args()
    
    if args.list_tasks:
        print(f"可用任务（共 {len(TASK_REGISTRY)}）:")
        for n in TASK_REGISTRY:
            print(f"  - {n}")
        print(f"\n配置文件: {CONFIG_PATH}")
        return

    if args.match_aware:
        try:
            from match_aware import describe_today, dynamic_interval, pre_match_cascade_marks
            d = describe_today()
            print(f"=== match_aware 诊断 · {d['now']} ===")
            print(f"未来 48h 比赛数: {d['n_matches_in_48h']}")
            for m in d['matches'][:20]:
                kw = "🟢 已开赛" if m['min_until_kickoff'] <= 0 else "🟡 待赛"
                print(f"  {kw} {m['kickoff_local']:<20s} {m['team_a']:<18s} vs {m['team_b']:<18s} (Δ {m['min_until_kickoff']:+.0f}min)")
            print()
            print("各任务当下的赛程感知 interval（None=用默认）：")
            for tn in ["group_results", "live_events", "lineups", "referee", "h2h", "kalshi", "polymarket", "cascade"]:
                v = dynamic_interval(tn, 60)
                tag = f"{v}min" if v is not None else "—"
                print(f"  {tn:<20s} {tag}")
            print()
            print(f"当前 tick 命中的赛前 cascade mark: {pre_match_cascade_marks(lookback_sec=120) or '无'}")
        except Exception as e:
            print(f"match_aware 诊断失败: {type(e).__name__}: {e}")
        return
    
    s = build_scheduler()
    
    if args.status:
        print(s.status_report())
        return
    
    if args.once:
        if args.once not in s.tasks:
            print(f"未知任务: {args.once}，可选: {list(s.tasks.keys())}")
            sys.exit(1)
        log.info(f"=== 单次运行: {args.once} ===")
        s.tasks[args.once].run(s)
        s._maybe_cascade()
        # 给 cascade 线程 1s 启动时间，然后等它完成（最多 10min）
        time.sleep(1)
        log.info("等待 cascade 线程完成 (若有)...")
        deadline = time.time() + 600
        while time.time() < deadline:
            # 用 acquire(blocking=False) 试探：能拿到说明无 cascade 在跑
            if s._cascade_lock.acquire(blocking=False):
                s._cascade_lock.release()
                break
            time.sleep(2)
        return
    
    # 冷启动全量刷新（在进入主循环之前）
    if args.bootstrap != "none":
        try:
            s.bootstrap_all(include_heavy_llm=(args.bootstrap == "full"))
        except Exception as e:
            log.error(f"[bootstrap] 失败（不影响主循环启动）: {type(e).__name__}: {e}")

    try:
        s.run_loop(tick_sec=args.tick)
    except KeyboardInterrupt:
        log.info("收到 Ctrl+C，退出")
        s.save_state()


if __name__ == "__main__":
    main()
