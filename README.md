# WorldCup Predict v1.1（路径 B：联网刷新版）

真实计算 + 实时数据 = 可复现的世界杯预测引擎

---

## 🎯 核心架构（路径 B）

```
                  ┌─────────────────────────────┐
                  │  Claude（智能编排层）        │
                  │  - WebFetch 抓 Polymarket   │
                  │  - WebSearch 抓伤病情报      │
                  │  - 解析 → 调用 Bash         │
                  └────────────┬────────────────┘
                               │
                  ┌────────────▼────────────────┐
                  │  Python 脚本层（计算/IO）    │
                  │  - refresh_helper.py 写库   │
                  │  - run_tournament.py 跑预测 │
                  └────────────┬────────────────┘
                               │
                  ┌────────────▼────────────────┐
                  │  本地数据层                  │
                  │  - teams.json（48 队 ）      │
                  │  - groups.json              │
                  │  - outputs/*.json           │
                  └─────────────────────────────┘
```

**分工**：
- **联网采集**：Claude（WebFetch/WebSearch）
- **数据落盘**：Python（保证完整性）
- **数学计算**：Python（保证可复现）
- **结果展示**：Claude（保证可读性）

---

## 🚀 使用流程

### 完整流程示例

```
1. /wc-predict refresh market       → Claude 抓 Polymarket → Python 写 teams.json
2. /wc-predict tournament 10000     → Python 跑蒙特卡洛 + 综合预测
3. /wc-predict match Spain vs France → Python 跑单场预测
```

### 子命令说明

| 命令 | 触发动作 | 涉及工具 |
|------|---------|---------|
| `/wc-predict tournament [N]` | 蒙特卡洛 + 综合预测 | Bash → Python |
| `/wc-predict match A vs B` | 单场胜平负 + 比分概率 | Bash → Python |
| `/wc-predict refresh` | 联网刷新全字段 | WebFetch + WebSearch + Bash |
| `/wc-predict refresh elo` | 仅 Elo（eloratings.net）| WebFetch + Bash |
| `/wc-predict refresh market` | 仅市场赔率（Polymarket）| WebFetch + Bash |
| `/wc-predict refresh injuries` | 仅伤病情报（ESPN/BBC）| WebSearch + 建议 |
| `/wc-predict update <T> <F> <V>` | 手动改单字段 | Bash → Python |
| `/wc-predict data` | 查看当前数据 | Bash → Python |

---

## 📊 测试结果（已用真实 Polymarket 数据刷新）

刷新前后对比：

| 球队 | 旧 market | 新 market（实时）| 模型概率 | 新偏差 |
|------|----------|-----------------|---------|-------|
| Spain | 18.2% | **17.5%** | 18.6% | +1.1pp |
| France | 16.7% | **16.0%** | 16.9% | +0.9pp |
| Argentina | 11.1% | **9.0%** | 10.6% | **+1.6pp** ⬆️ |
| Portugal | 8.3% | **11.0%** | 6.7% | -4.3pp |
| Brazil | 11.1% | **8.5%** | 5.0% | -3.5pp |

→ Argentina 在新数据下变成最大正向偏差候选（市场对阿根廷信心下滑）

---

## 📁 工程目录

```
~/.codebuddy/worldcup-predict/
├── README.md                    # 本文件
├── code/
│   ├── data/
│   │   ├── refresh_helper.py    # 数据刷新辅助
│   │   └── __init__.py
│   ├── models/
│   │   ├── elo_engine.py        # Elo 评级
│   │   ├── poisson_model.py     # Dixon-Coles
│   │   ├── monte_carlo.py       # 蒙特卡洛 10k
│   │   └── synthesizer.py       # 综合预测（读最新 market）
│   ├── utils/
│   │   └── io.py                # 数据加载
│   ├── run_tournament.py        # 主入口
│   └── predict_match.py         # 单场入口
├── data/
│   ├── raw/
│   │   ├── teams.json           # 48 队基础（含 _last_updated 时间戳）
│   │   └── groups.json          # 12 组分组
│   └── outputs/
│       ├── mc_simulation_n10000.json
│       └── synthesizer_report.json
└── logs/
```

---

## 🔧 开发者：直接运行 Python

```bash
# 查看当前数据
python3 ~/.codebuddy/worldcup-predict/code/data/refresh_helper.py show

# 手动更新单字段
python3 ~/.codebuddy/worldcup-predict/code/data/refresh_helper.py update Spain elo 2160

# 跑全锦标赛预测
python3 ~/.codebuddy/worldcup-predict/code/run_tournament.py 10000

# 跑单场
python3 ~/.codebuddy/worldcup-predict/code/predict_match.py "Spain" "France"

# 仅重跑综合预测（不重跑蒙特卡洛）
python3 ~/.codebuddy/worldcup-predict/code/models/synthesizer.py
```

---

## ⚙️ 数据更新机制

### 字段优先级

1. **手动 Edit** > **refresh_helper.py update** > **静态 teams.json**
2. **synthesizer.py 读 market_implied 优先级**：
   - 优先：`teams.json`（最新）
   - 后备：`mc_simulation_n*.json`（蒙特卡洛快照）

### 自动时间戳

每次 `update` 自动写入 `_last_updated` 字段：

```json
"_last_updated": {
  "Spain": {"market_implied": "2026-06-11 16:45:00"},
  "France": {"market_implied": "2026-06-11 16:45:00"}
}
```

---

## 📝 局限性

1. **Elo 自动抓取未实现**：当前只支持市场赔率自动刷新，Elo 需手动从 eloratings.net 抓
2. **伤病不直接写库**：refresh injuries 输出建议，需要用户手动调整 synthesizer.py
3. **时区/海拔/天气未联网**：通过 synthesizer 静态调整因子注入
4. **机器学习层未实现**：暂未训练 XGBoost（需历史比赛数据集）

---

## 🆚 三个版本对比

| 版本 | 计算 | 数据 | 目录 |
|------|------|------|------|
| Prompt v1（备份）| LLM 角色扮演 | 上下文记忆 | `worldcup-predictor-prompt-v1` |
| Prompt v1（当前）| LLM 角色扮演 | 上下文记忆 | `worldcup-predictor` |
| **Engine v1（路径 B）** ⭐ | **真 numpy/scipy** | **联网刷新** | **`worldcup-predict`** |

---

## 🎓 关键设计哲学

1. **静态计算用 Python**：保证可复现、可审计
2. **动态数据用 Claude**：保证最新、保证理解力
3. **数据落盘用 Python**：保证完整性、保证可回溯
4. **结果展示用 Claude**：保证可读性、保证上下文

→ 这是"AI + 工程"的最佳分工。
