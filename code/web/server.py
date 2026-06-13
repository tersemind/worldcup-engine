#!/usr/bin/env python3
"""
WorldCup Engine Web 仪表盘（#11 P3）

技术栈：
- 后端：Python 内置 http.server（无需 FastAPI 依赖）
- 前端：单页 HTML + ECharts（CDN）
- API：返回 outputs/*.json

启动：
  python3 code/web/server.py
  浏览器访问：http://localhost:8088
"""
import json
import sys
import os
import subprocess
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import DATA_OUTPUTS, ROOT


def _bootstrap_env_from_launchctl():
    """启动时尝试从 launchctl getenv 拿到 LINGYA_API_KEY
    
    macOS `launchctl setenv` 设置的变量只对 launchd 子进程生效，
    从 Terminal/SSH 启动的进程默认拿不到。这里主动调一次桥接。
    
    优先级：os.environ > launchctl > 无
    """
    if os.environ.get("LINGYA_API_KEY"):
        return  # 已有，不覆盖
    try:
        result = subprocess.run(
            ["launchctl", "getenv", "LINGYA_API_KEY"],
            capture_output=True, text=True, timeout=2,
        )
        val = result.stdout.strip()
        if val:
            os.environ["LINGYA_API_KEY"] = val
            print(f"[server] 从 launchctl 加载 LINGYA_API_KEY (尾 4 位: ...{val[-4:]})")
    except Exception as e:
        print(f"[server] launchctl 桥接失败（可忽略）: {e}")


_bootstrap_env_from_launchctl()


PORT = 8088
WEB_DIR = ROOT / "code" / "web"


# ─────────────────────── /api/scheduler 调度仪表盘 ───────────────────────
def _serve_scheduler_state() -> dict:
    """聚合调度状态供前端展示：
      - tasks: 22 任务的 last_run/next_eta/runs/fails/interval/effective_interval/stage_overrides
      - current_stage: 当前阶段（从 live_state.json）
      - cascade_recent: 最近 5 次 cascade summary（从 scheduler.log tail）
      - now: 服务器时间，前端用来算 ETA
      - daemon_alive: scheduler daemon 是否在跑（ps grep）

    完全吞异常 —— 不能让 web 报错影响其它 API。
    """
    from datetime import datetime
    import re
    import subprocess

    out = {
        "now": datetime.now().isoformat(timespec="seconds"),
        "current_stage": "unknown",
        "daemon_alive": False,
        "tasks": [],
        "cascade_recent": [],
        "errors": [],
    }

    # 1. scheduler_state.json
    try:
        state_path = ROOT / "data" / "outputs" / "scheduler_state.json"
        if state_path.exists():
            state = json.load(open(state_path))
            out["state_saved_at"] = state.get("_saved_at")
            # config 用于读 stage_overrides
            cfg_path = ROOT / "data" / "raw" / "scheduler_config.json"
            cfg = {}
            if cfg_path.exists():
                cfg = json.load(open(cfg_path)).get("tasks", {})

            for name, info in state.get("tasks", {}).items():
                task_cfg = cfg.get(name, {})
                interval = info.get("interval_min", 60)
                stage_overrides = task_cfg.get("stage_overrides", {})
                # 计算 next_eta（按默认 interval，前端再叠加 stage override 显示）
                last_run = info.get("last_run")
                next_eta_sec = None
                if last_run:
                    try:
                        last_dt = datetime.fromisoformat(last_run)
                        diff = (last_dt - datetime.now()).total_seconds() + interval * 60
                        next_eta_sec = int(diff)
                    except Exception:
                        pass
                out["tasks"].append({
                    "name": name,
                    "interval_min": interval,
                    "last_run": last_run,
                    "last_success": info.get("last_success"),
                    "last_error": info.get("last_error"),
                    "n_runs": info.get("n_runs", 0),
                    "n_failures": info.get("n_failures", 0),
                    "description": info.get("description", ""),
                    "next_eta_sec": next_eta_sec,
                    "stage_overrides": stage_overrides,
                    "triggers_cascade": task_cfg.get("triggers_cascade", False),
                    "enabled": task_cfg.get("enabled", True),
                })
    except Exception as e:
        out["errors"].append(f"state: {type(e).__name__}: {e}")

    # 2. live_state.json → current_stage
    try:
        live_path = ROOT / "data" / "raw" / "live_state.json"
        if live_path.exists():
            live = json.load(open(live_path))
            out["current_stage"] = live.get("current_stage", "unknown")
            out["live_last_updated"] = live.get("last_updated")
    except Exception as e:
        out["errors"].append(f"live: {type(e).__name__}: {e}")

    # 3. 最近 cascade summary（从 scheduler.log tail 解析）
    try:
        log_path = ROOT / "data" / "outputs" / "scheduler.log"
        if log_path.exists():
            # 只看末尾 50KB 防大文件
            size = log_path.stat().st_size
            with open(log_path, "rb") as f:
                if size > 50000:
                    f.seek(-50000, 2)
                data = f.read().decode("utf-8", errors="ignore")
            # 匹配 "[cascade] ✓ XXs ..." 行
            pat = re.compile(r"^([\d-]+\s[\d:]+).*?\[cascade\]\s+✓\s+([\d.]+)s\s+(.*)$", re.MULTILINE)
            for m in list(pat.finditer(data))[-5:]:
                out["cascade_recent"].append({
                    "ts": m.group(1),
                    "duration_sec": float(m.group(2)),
                    "summary": m.group(3).strip(),
                })
    except Exception as e:
        out["errors"].append(f"log: {type(e).__name__}: {e}")

    # 4. daemon_alive 探测
    try:
        r = subprocess.run(["pgrep", "-f", "code/data/scheduler.py"],
                           capture_output=True, text=True, timeout=2)
        out["daemon_alive"] = bool(r.stdout.strip())
        out["daemon_pids"] = r.stdout.strip().split("\n") if r.stdout.strip() else []
    except Exception as e:
        out["errors"].append(f"daemon: {type(e).__name__}: {e}")

    return out


# ─────────────────────── /api/timeseries cascade 时间序列 ───────────────────────
def _serve_timeseries(query: dict) -> dict:
    """从 data/outputs/snapshots/timeseries.jsonl 读取并按 metric 抽取。

    query 参数：
      - metric: champion / final / semifinal / group_first / final_top （默认 champion）
      - teams:  逗号分隔的球队过滤（可选；group_first/final_top 时忽略）
      - limit:  返回最近 N 条（默认 200，最大 500）

    返回：{points: [{ts, values}], metric, teams: [...], total: N}
    """
    try:
        ts_path = DATA_OUTPUTS / "snapshots" / "timeseries.jsonl"
        if not ts_path.exists():
            return {"points": [], "metric": "", "teams": [],
                    "note": "尚无快照（等待下一轮 cascade）"}

        metric = (query.get("metric", ["champion"])[0] or "champion").strip()
        valid_metrics = {"champion", "final", "semifinal", "quarterfinal",
                         "round_of_16", "round_of_32",
                         "group_first", "final_top"}
        if metric not in valid_metrics:
            return {"error": f"metric 必须是 {sorted(valid_metrics)}",
                    "points": []}

        try:
            limit = int(query.get("limit", ["200"])[0])
        except Exception:
            limit = 200
        limit = max(1, min(limit, 500))

        teams_filter = (query.get("teams", [""])[0] or "").strip()
        teams_set = (
            {t.strip() for t in teams_filter.split(",") if t.strip()}
            if teams_filter else None
        )

        # 从尾部读 limit 行
        lines = ts_path.read_text().splitlines()
        if not lines:
            return {"points": [], "metric": metric, "teams": [],
                    "note": "snapshot 为空"}
        tail = lines[-limit:]

        points = []
        all_teams = set()
        for line in tail:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ts = rec.get("ts")
            if not ts:
                continue

            if metric in ("champion", "final", "semifinal", "quarterfinal",
                          "round_of_16", "round_of_32"):
                vals = rec.get(metric, {}) or {}
                if teams_set:
                    vals = {t: v for t, v in vals.items() if t in teams_set}
                all_teams.update(vals.keys())
                points.append({"ts": ts, "values": vals})

            elif metric == "group_first":
                # 嵌套 {group: {team: pct}}，平铺成 {team: pct} 用于绘图
                gf = rec.get("group_first", {}) or {}
                flat = {}
                for gname, gdict in gf.items():
                    if not isinstance(gdict, dict):
                        continue
                    for team, pct in gdict.items():
                        flat[team] = pct  # 队伍名唯一，直接平铺
                if teams_set:
                    flat = {t: v for t, v in flat.items() if t in teams_set}
                all_teams.update(flat.keys())
                points.append({"ts": ts, "values": flat})

            elif metric == "final_top":
                # final_top 是 list[{team_a, team_b, prob_pct, ...}]
                # 用 "A vs B" 做 key，prob_pct 做值
                top = rec.get("final_top", []) or []
                vals = {}
                for item in top:
                    a = item.get("team_a"); b = item.get("team_b")
                    p = item.get("prob_pct")
                    if a and b and p is not None:
                        vals[f"{a} vs {b}"] = p
                all_teams.update(vals.keys())
                points.append({"ts": ts, "values": vals})

        return {
            "points": points,
            "metric": metric,
            "teams": sorted(all_teams),
            "total": len(points),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "points": []}


class APIHandler(SimpleHTTPRequestHandler):
    """统一处理静态 HTML + JSON API"""
    
    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        
        # 路由
        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path.startswith("/api/"):
            self._serve_api(path)
        elif path.startswith("/static/"):
            self._serve_static(path)
        else:
            self.send_error(404, f"Not Found: {path}")
    
    def _serve_html(self):
        """主页面"""
        html_path = WEB_DIR / "index.html"
        if not html_path.exists():
            self.send_error(404, "index.html not found")
            return
        with open(html_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)
    
    def _serve_api(self, path):
        """API 路由（支持静态 JSON 文件 + 动态推演 API）"""
        url = urlparse(self.path)
        query = parse_qs(url.query)
        api_name = path.replace("/api/", "")

        # 静态 JSON 文件
        api_to_file = {
            "synthesizer": "synthesizer_report.json",
            "synthesizer_ai_phase3": "synthesizer_report_ai_phase3.json",
            "scenarios": "three_scenarios.json",
            "finals": "finals_matchups.json",
            "arbitrage": "arbitrage_signals.json",
            "arbitrage_kalshi": "arbitrage_kalshi_signals.json",
            "calibration": "calibration.json",
            "backtest": "backtest_result.json",
            "mc": "mc_simulation_n100000.json",
            "swarm": "swarm_consensus.json",
            "uncertainty": "uncertainty_decomposition.json",
            "in_match": "in_match_update.json",
            "data_quality": "data_quality_report.json",
            "kalshi": "kalshi_match_odds.json",
        }

        # 调度状态：聚合 scheduler_state.json + live_state.current_stage + 最近 cascade summary
        if api_name == "scheduler":
            return self._send_json(_serve_scheduler_state())

        # 时间序列：从 snapshots/timeseries.jsonl 抽取
        if api_name == "timeseries":
            return self._send_json(_serve_timeseries(query))

        # 动态推演 API（来自 tournament_api.py）
        dynamic_apis = {
            "groups", "r32", "r16", "qf", "sf", "final_match",
            "bracket", "match", "team_path", "path_distribution",
            "critical_compare", "group_schedule", "market_bias",
            "match_bias",
        }

        if api_name in dynamic_apis:
            return self._serve_dynamic_api(api_name, query)

        if api_name not in api_to_file:
            self._send_json({
                "error": "Unknown API",
                "static_apis": sorted(api_to_file.keys()),
                "dynamic_apis": sorted(dynamic_apis),
            }, status=404)
            return

        file_path = DATA_OUTPUTS / api_to_file[api_name]
        if not file_path.exists():
            self._send_json({"error": f"Data not found: {file_path.name}"}, status=404)
            return

        with open(file_path, "r") as f:
            data = json.load(f)
        self._send_json(data)

    def _serve_dynamic_api(self, api_name: str, query: dict):
        """动态计算 API（实时调 tournament_api.py）"""
        try:
            # lazy import 避免 server 启动慢
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from web import tournament_api as ta
        except ImportError as e:
            self._send_json({"error": f"tournament_api import failed: {e}"}, status=500)
            return

        # 双通道支持：?channel=base | ai_phase3（默认 base）
        # 仅赛程系列 API 接受此参数：groups/r32/r16/qf/sf/final_match/team_path/path_distribution
        channel = query.get("channel", ["base"])[0]
        if channel not in ("base", "ai_phase3"):
            channel = "base"

        try:
            if api_name == "groups":
                data = ta.api_groups(channel=channel)
            elif api_name == "r32":
                data = ta.api_r32(channel=channel)
            elif api_name == "r16":
                data = ta.api_r16(channel=channel)
            elif api_name == "qf":
                data = ta.api_qf(channel=channel)
            elif api_name == "sf":
                data = ta.api_sf(channel=channel)
            elif api_name == "final_match":
                data = ta.api_final_match(channel=channel)
            elif api_name == "bracket":
                data = ta.api_bracket()
            elif api_name == "match":
                a = (query.get("a", [None])[0])
                b = (query.get("b", [None])[0])
                with_llm = query.get("with_llm", ["0"])[0] in ("1", "true", "yes")
                if not (a and b):
                    self._send_json({"error": "需要 query: ?a=Spain&b=France[&with_llm=1]"}, status=400)
                    return
                data = ta.api_match(a, b, with_llm=with_llm)
            elif api_name == "team_path":
                t = query.get("team", [None])[0] or query.get("a", [None])[0]
                if not t:
                    self._send_json({"error": "需要 query: ?team=Spain"}, status=400)
                    return
                data = ta.api_team_path(t, channel=channel)
            elif api_name == "path_distribution":
                n = int(query.get("n", ["10"])[0])
                data = ta.api_path_distribution(top_n=n, channel=channel)
            elif api_name == "critical_compare":
                data = ta.api_critical_compare()
            elif api_name == "group_schedule":
                data = ta.api_group_schedule()
            elif api_name == "match_bias":
                refresh = query.get("refresh", ["0"])[0] in ("1", "true", "yes")
                data = ta.api_match_bias(refresh=refresh)
            elif api_name == "market_bias":
                refresh = query.get("refresh", ["0"])[0] in ("1", "true", "yes")
                data = ta.api_market_bias(refresh=refresh)
            else:
                self._send_json({"error": f"unknown dynamic API: {api_name}"}, status=404)
                return
        except Exception as e:
            import traceback
            self._send_json({
                "error": str(e),
                "trace": traceback.format_exc()[-500:],
            }, status=500)
            return

        self._send_json(data)
    
    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def _serve_static(self, path):
        """静态资源"""
        rel = path.replace("/static/", "")
        file_path = WEB_DIR / "static" / rel
        if not file_path.exists():
            self.send_error(404)
            return
        ext = file_path.suffix
        ct = {".js": "application/javascript", ".css": "text/css", 
              ".png": "image/png", ".svg": "image/svg+xml"}.get(ext, "text/plain")
        with open(file_path, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.end_headers()
        self.wfile.write(content)
    
    def log_message(self, format, *args):
        """简化日志"""
        sys.stderr.write(f"  → {self.command} {self.path}\n")


def _detect_lan_ip():
    """探测本机所有真实 LAN IP（排除 127.x / 169.254.x / VPN 点对点接口）"""
    import socket, subprocess
    ips = []
    # 用 ifconfig 拿所有真实接口
    try:
        out = subprocess.check_output(["ifconfig"], text=True, timeout=2)
        current_iface = None
        is_ptp = False  # 点对点（VPN/utun/tap）跳过
        for line in out.splitlines():
            if line and not line.startswith((' ', '\t')):
                current_iface = line.split(':')[0]
                # 同一行就含 POINTOPOINT（macOS ifconfig 格式）
                is_ptp = "POINTOPOINT" in line or current_iface.startswith(("utun", "ppp"))
            elif "POINTOPOINT" in line:
                is_ptp = True
            elif line.strip().startswith("inet ") and not is_ptp:
                parts = line.strip().split()
                if len(parts) >= 2:
                    ip = parts[1]
                    if (not ip.startswith("127.")
                        and not ip.startswith("169.254.")
                        and not ip.startswith("0.")):
                        ips.append(ip)
    except Exception:
        pass
    # 兜底：用 UDP socket trick
    if not ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ips.append(s.getsockname()[0])
            s.close()
        except Exception:
            pass
    return ips


def run_server(port: int = PORT, host: str = "0.0.0.0"):
    server = HTTPServer((host, port), APIHandler)
    lan_ips = _detect_lan_ip()

    print(f"🌐 WorldCup Engine Web 仪表盘启动")
    print(f"   Bind: {host}:{port}")
    print(f"   访问地址:")
    print(f"     - 本机:    http://localhost:{port}")
    print(f"     - 本机:    http://127.0.0.1:{port}")
    for ip in (lan_ips or []):
        print(f"     - 局域网:  http://{ip}:{port}")
    print(f"   API:")
    print(f"     /api/synthesizer  → 综合预测（base 通道：纯 MC + 8 项 AI 修正）")
    print(f"     /api/synthesizer_ai_phase3 → 综合预测（AI 加权 MC + 8 项修正，分段 ELO_PER_PP）")
    print(f"     /api/scenarios    → 三情景")
    print(f"     /api/finals       → 决赛对阵")
    print(f"     /api/arbitrage    → 套利信号")
    print(f"     /api/scheduler    → 调度仪表盘（22 任务 + cascade 进度）")
    print(f"     /api/backtest     → 回测结果")
    print(f"   按 Ctrl+C 停止\n")

    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 已停止")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="WorldCup Engine Web 仪表盘")
    ap.add_argument("--port", type=int, default=PORT, help=f"端口 (默认 {PORT})")
    ap.add_argument("--host", default="0.0.0.0",
                     help="绑定地址 (默认 0.0.0.0 = 所有网卡含局域网; 用 127.0.0.1 仅本机)")
    args = ap.parse_args()
    run_server(port=args.port, host=args.host)
