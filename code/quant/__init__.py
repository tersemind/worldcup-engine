"""worldcup-engine 量化交易模块（in-match 高频预测 + 套利信号）。

本模块独立于 code/data, code/models, code/agents——可被它们 import，但不被反向依赖。
任何 quant.* 失败都不会影响主调度（scheduler cascade、frozen_predictions 等）。

模块组件：
  - live_elo_solver       v1 硬事实反求（比分/红牌/换人/elapsed），保底路径
  - live_stats_features   v2 质量调整（xG/possession/penalty/key_player_off）
  - market_anomaly_detector  v3' Kalshi 价跳变监控
  - live_fetcher_extras   ESPN summary endpoint 扩展抓取
  - live_trading_loop     主调度 tick（每 60s 跑一次）
  - tick_writer           落盘工具（snapshot + jsonl）
"""
__version__ = "0.1.0"
