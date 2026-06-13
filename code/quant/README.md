# quant — In-Match 高频量化交易模块

本目录是 **worldcup-engine** 的量化交易子系统。专门处理：
> 比赛开赛 → 终场期间，60s 频率的实时模型预测 + 市场行情 + 套利信号生成。

## 设计原则

1. **完全独立**：可被 `code/data`、`code/models`、`code/agents` import，但不被反向依赖
2. **fail-soft**：任何 quant.* 失败都不能影响主调度（cascade、frozen_predictions 等）
3. **三档信号融合**：
   - **v1 硬事实**（保底）：比分/红牌/换人/elapsed → 反求 ΔE
   - **v2 质量调整**（可选）：xG/possession/penalty/key_player_off
   - **v3' 市场异动**（信号告警）：Kalshi 价跳变 → dirty flag

## 模块组件

| 文件 | 角色 | 状态 |
|------|------|------|
| `live_elo_solver.py` | v1 反求公式 | ✅ S1 完成 |
| `live_stats_features.py` | v2 ESPN summary 信号映射 | ✅ S4 完成 |
| `market_anomaly_detector.py` | v3' Kalshi 异动检测 | ✅ S5 完成 |
| `live_fetcher_extras.py` | ESPN summary endpoint 扩展 | ✅ S3 完成 |
| `live_trading_loop.py` | 主调度 tick（每 60s） | ✅ S2 完成 |
| `tick_writer.py` | 落盘工具（snapshot+jsonl） | ✅ S2 完成 |

## 测试

```bash
# v1 单元测试（20 cases）
python3 code/quant/tests/test_live_elo_solver.py

# v1 demo
python3 code/quant/live_elo_solver.py
```

## 数据流

```
live tick (60s):
  并行 fetch → ESPN scoreboard (4s) + Kalshi (1.7s) + ESPN summary (v2)
       ↓
  ΔE 合成 = v1_solver(硬事实) + v2_features(质量调整)
       ↓
  P(H/D/A) = match_probabilities(elo + ΔE)
       ↓
  arbitrage_kalshi 全量重算 (10ms)
       ↓
  v3' 市场异动检测（不进 ΔE，仅作 dirty flag）
       ↓
  落盘:
    - data/outputs/in_match_live.json  (snapshot, web 直读)
    - data/outputs/in_match_ticks.jsonl (append-only, 回测/审计)
```

## 不动的现有任务

| 任务 | 角色 |
|------|------|
| `live_events` (5min) | 提供 fallback 数据 |
| `in_match` (10min) | 写 in_match_update.json，被 cascade 引用 |
| `kalshi` (15min) | 提供 fallback Kalshi 价 |
| `cascade` | live tick 不触发，避免雪崩 |

## 实施进度

- [x] **S1** 模块骨架 + `live_elo_solver.py` v1（20 unit tests 全通过）
- [x] **S2** `live_trading_loop.py` + `tick_writer.py` + scheduler 注册（端到端 mock 验证通过）
- [x] **S3** `live_fetcher_extras.py`（ESPN summary endpoint + 30s 缓存）
- [x] **S4** `live_stats_features.py`（v2 量化映射 xG/possession/shots/penalty/corners，总 cap 4pp）
- [x] **S5** `market_anomaly_detector.py`（v3' Kalshi ≥3pp 跳变检测 5min 滑窗）
- [x] **S6** 前端 trading tab live banner + ⚡ 异动标记 + `/api/in_match_live`；tab 移到末位
- [ ] **S7** memory 更新 + 文档完善
