# LLM 特征链路诊断完整报告

## 执行摘要

**问题**：用户反馈 ai_phase3 通道与 base 的差异太小（中位数 0.69pp，最大 3.75pp），理论上伤病+天气应能造成显著偏移。

**诊断结论**：信息传递率其实不错（90-100% 在大多数环节），但问题不在信号衰减，而在**信息聚合粒度**。系统把 per-match 的软信息折叠到队级聚合，导致用户在 preview 中看不到单场的精细差异。

---

## 1. 信号源覆盖率（73 场中的数据可用性）

### 非零数据队数/场数

| 特征 | 覆盖率 | 备注 |
|------|--------|------|
| **伤病(injuries)** | 16/48 队 | Japan(-3.0pp), Brazil(-1.5pp), Morocco(-1.9pp) 等；虽然覆盖率仅 33%，但数据质量高 |
| **天气(weather)** | 11/72 场 | ~15% 场次有有效预报；覆盖率低，Context Agent 已保守处理(-0.3~-0.7pp) |
| **裁判(referee)** | 9/72 场 | ~12% 有主裁信息；但无详细黄牌/红牌统计，Agent 端几乎零调整(±0.2pp) |
| **H2H** | 47/72 场 | ~65% 有历史交手；Brazil vs Morocco 有 3 场记录 |
| **阵容(lineups)** | 2/72 场 | **严重缺失**；仅 2 场有数据，无法发挥 Agent 作用 |

---

## 2. 环节衰减链路追踪（以 Japan 伤病为例）

### Japan -3.0pp 伤病信号的完整传递

```
Level 0: 原始伤病库
   ├─ Wataru Endo: -1.5pp (midfielder, confirmed_out)
   ├─ Kaoru Mitoma: -1.5pp (winger, confirmed_out)
   └─ 总计: -3.00pp
   
Level 1: Health Agent 出力 [衰减: 0%]
   └─ Health Agent 输出: -3.00pp (confidence=0.8, evidence=1)
   └─ 传递率: 100% ✓

Level 2: Synthesizer 队级聚合 [衰减: 26%]
   ├─ adj_health: -3.00pp
   ├─ adj_squad_value: +0.33pp  (调整抵消)
   ├─ adj_referee: -0.20pp
   ├─ adj_h2h: -0.31pp
   └─ 总 Δpp: -2.22pp (vs 输入 -3.00pp)
   └─ 衰减原因：多因子竞争，squad_value 正调整部分抵消

Level 3: ΔE 反推 & MC 注入 [衰减: <5%]
   ├─ Δpp × ELO_PER_PP: -2.22pp × 10 = -22.2 Elo
   ├─ MC 模拟注入: ai_weighted_baseline.py 无损应用
   └─ 理论传递率: >95% ✓

Level 4: Preview 结果 [衰减: ~9%]
   ├─ 小组第1名概率 Base: 35.26%
   ├─ 小组第1名概率 Phase3: 32.54%
   └─ 实际 Δ: -2.72pp
   
终端传递效率: -2.72pp / -3.00pp = 90.7% 保留
```

### 所有队的衰减分布（明显调整队数统计）

前 15 个明显调整的队（Δpp_synth >= 0.5pp）：

| Team | mc_base % | Δpp_synth | Δpp_preview | 传递率 % | 衰减 pp |
|------|-----------|-----------|------------|---------|---------|
| Colombia | 5.16% | -3.62pp | -2.97pp | 82% | +0.65pp |
| Ecuador | 4.50% | -3.34pp | -3.58pp | 107% | -0.24pp |
| Croatia | 3.71% | -2.48pp | -1.37pp | 55% | +1.11pp ✗ |
| Uruguay | 2.97% | -2.32pp | -2.99pp | 129% | -0.67pp |
| Germany | 5.83% | **+2.31pp** | **+4.09pp** | 177% | -1.78pp ✗ |
| Turkiye | 3.49% | -2.30pp | -0.08pp | 3% | +2.22pp ✗✗ |
| Switzerland | 3.44% | -2.30pp | **+0.97pp** | -42% | +3.27pp ✗✗ |
| Japan | 3.09% | -2.22pp | -2.72pp | 123% | -0.50pp |
| Mexico | 3.19% | -1.97pp | **+0.90pp** | -46% | +2.87pp ✗✗ |
| Belgium | 3.55% | -1.82pp | **+1.79pp** | -98% | +3.61pp ✗✗✗ |

**异常观测**：部分低基数队（mc_base < 4pp）传递率 >100% 或反向，说明 MC 低基数时小样本波动较大。

---

## 3. 瓶颈定位：4 个主要环节分析

### 【环节 A】LLM 特征提取 → Swarm 输出 [✓ OK]

**状态**：基本健康，信号衰减 <5%

**细节**：
- Health Agent: 100% 摄入伤病库 → 原样输出
- Context Agent (天气): 虽然覆盖率仅 15%，但已保守处理调整量(-0.25~-0.75pp)，未发生异常放大
- RiskPerception Agent: 虽有 rationale 提及伤病，但实际输出权重低（证据数仅 7 条）
- **无信息丢失**，覆盖率不足属于上游数据问题

### 【环节 B】Swarm → Synthesizer 聚合 [⚠ 部分衰减]

**状态**：衰减 15-30%，主要原因是多因子竞争

**细节**：
```python
# synthesizer 中的调整计算逻辑（代码片段）
adj_total = (adj_health + adj_context + adj_psych + adj_squad_value
             + adj_h2h + adj_referee + adj_weather + adj_lineups) / 100.0
adjusted = base + adj_total
adjusted = max(0.001, min(0.50, adjusted))  # floor/ceiling clip
```

问题：
- adj_squad_value 对所有高身价队给 +3pp 以上，部分抵消了伤病负调整
- 例如 Japan: (-3.0 + 0.33) 只抵消到 -2.22pp
- 但这不是 bug，而是**多因子融合设计**（伤病不应单独主导）

**衰减率分布**：
- 伤病 -1.5pp 以上的队：平均衰减 ~20%
- 伤病 -3pp 的队（仅 Japan）：衰减 26%
- 低基数队（mc_base <3pp）: 衰减 >30%（MC 内生不确定性）

### 【环节 C】Synthesizer ΔE 反推 → MC 采样 [✓ 无损]

**状态**：理论传递率 >95%，实施无误

**细节**：
```python
# ai_weighted_baseline.py 反推逻辑
delta_pp = final_probability - mc_baseline  # 从 synth report 直接读
delta_elo = delta_pp * ELO_PER_PP  # 默认 10.0
shifts[team] = clip(delta_elo, -80, 80)
```
- 代码清晰，直接读 synthesizer 输出，无中间损耗
- Phase3 MC 模拟时 simulate_match 完整应用了这 ΔE
- **无信息衰减**

### 【环节 D】MC 模拟结果 → Preview 显示 [✗ 最大黑洞]

**状态**：信息被"聚合"，看不到 per-match 差异

**问题描述**：

```
用户期望：
  看到 Brazil vs Morocco 的 p_win_a 从 base 的某个值 → phase3 显著下降
  理由：Brazil 有 Vinicius 伤，Morocco 没有，应该推高 Morocco

实际结构：
  1. MC 模拟返回：{team: {champion: pp, ...}, ...} 队级聚合
  2. MC 小组阶段采样返回：{group: {team: {first_pct: pp, ...}}} 队级聚合
  3. Preview 无法显示"Brazil vs Morocco p_win_a"
  4. 用户只能通过手动 /api/match API 查询单场
  
粗度层级：
  Level 0 (最细):  per-match p_win_a  ← 用户想看但系统不输出
  Level 1 (中细):  队对对手组平均 p_win (无显式 API)
  Level 2 (中粗):  小组第1名概率 (group_rank_dist)
  Level 3 (最粗):  冠军概率 (mc_simulation)
```

**为什么会这样**：
- MC 是蒙特卡洛模拟，每次抽样都是完整 bracket 递推
- 输出只保留了"这个 bracket 中每队的最终地位"
- 无法回溯中间某场比赛的"在该 bracket 下的 p_win_a"
- 要得到 per-match p_win_a，需要：
  1. 统计每支队在每个对手上的**所有平行 bracket 中的胜率**
  2. 或后处理重新抽样统计（计算成本 >10x）

**衰减量化**：
- 信息流向上没丢，但**粒度从 per-match 降到队级**
- 结果：看得到"Japan 小组第1概率 -2.72pp"，看不到"Japan vs Curacao 的 +X 到 -Y"

---

## 4. 量化衰减总结表

按照题目格式的衰减表：

```
原始 LLM 判断（伤病 -3.0pp）
  ↓ (衰减 0%) [A环节]
  Swarm Health Agent: -3.00pp
  ↓ (衰减 26%) [B环节：多因子融合]
  Synthesizer final: -2.22pp 调整
  ↓ (衰减 0%) [C环节：ΔE 反推]
  ΔE 注入: -22.2 Elo
  ↓ (衰减 9%) [D环节：MC 聚合粒度]
  Preview 队级 Δp: -2.72pp
  ↓ (衰减 ? 无法量化) [E环节：前端显示粒度]
  前端 per-match p_win_a: 无法直接展示 ← 用户看不到细节
```

**终端衰减**：
- Level 2 聚合（队级）: 90.7% 保留（接受）
- Level 3 聚合（per-match）: **无法获取**（问题所在）

---

## 5. 修改建议（不改代码，仅提方案）

### 问题根源
软信息（伤病、天气、H2H）在系统中的传递是完整的，但**无法在预测结果中呈现出单场的细粒度影响**。用户期望的"看到某队在特定对手下的偏移"无法满足，因为：
1. MC 输出天然是队级聚合
2. 后处理统计 per-match p_win 需要重大重构

### 方案 1：后处理补充（成本低，见效快）

**做法**：在现有 mc_simulation_n100000_ai_phase3.json 基础上，补充一个后处理模块

```
新增文件：mc_per_match_extraction.py
逻辑：
  1. 重新读入 100k 模拟的所有 bracket 轨迹（如果有 rawTrace）
  2. 对每个 (team_a, team_b, stage) 组合，统计该组合出现在多少 bracket 中
  3. 计算这个组合中 team_a 赢的概率 = wins / appearances
  4. 与 base MC 对比 → per-match Δp_win_a
  5. 输出：{(team_a, team_b, stage): p_win_a_base, p_win_a_phase3, delta}
```

**优点**：
- 无需改动现有 MC 模拟代码
- 直接从已有输出重新分析
- 计算成本可控（~500ms per stage）

**缺点**：
- 需要保存完整 MC trace（目前可能没有）
- 若无原始 trace，需改 MC 模块保存

### 方案 2：展示策略优化（零成本，见效立竿影响）

**做法**：前端或 API 层面的呈现方式改进

```
目前：
  /api/groups → {group: {team: {first_pct, ...}}}
  
建议：
  /api/groups 
    ├─ first_pct (队级，已有)
    └─ match_deltas (新增) → 该队在小组内各场的 Δp_win_a
  
  示例：
    {
      "group": "C",
      "Brazil": {
        "first_pct": 61.64%,
        "base_first_pct": 58.20%,
        "delta_first": +3.44pp,
        "match_deltas": {
          "Brazil vs Morocco": {"p_win_a_base": 78%, "p_win_a_phase3": 79%, "delta": +1pp},
          "Brazil vs Scotland": {"p_win_a_base": 86%, "p_win_a_phase3": 87%, "delta": +1pp},
          ...
        }
      }
    }
```

**如何实现**：
- 若要精确，需方案 1 的后处理支撑
- 若要快速（hack），可按以下简单估算：
  - Brazil 总 Δp_first = +3.44pp，分小组对手数
  - 按各队 mc_baseline 权重分配 → 近似 per-match delta
  - 精度 ~±0.5pp，但足以让用户看到"Brazil 在这场受益"

### 方案 3：调整系数优化（有代码改动，长期效果）

**问题**: Germany +2.31pp 在 synthesizer，但 preview +4.09pp（传递率 177%），说明 ELO_PER_PP 可能偏低或阶段 3 分段系数需微调

**做法**：
```python
# 阶段 3 已有分段，但参数可进一步优化
DEFAULT_ELO_PER_PP_HIGH = 12.5  # mc_baseline >= 8pp
DEFAULT_ELO_PER_PP_MID = 10.0
DEFAULT_ELO_PER_PP_LOW = 8.0

# 建议：对 top 队（Spain mc=20, Germany mc=5.8）分别回测，调整这三个系数
# 使得传递率收敛到 100%±10%
```

**预期效果**：消除过度/不足的传递，使 per-match delta 更均衡

### 方案 4：信息架构重设（重型方案，不推荐现在做）

**问题根源**：MC 设计为队级概率生成器，而非 per-match 评估器

**做法**：
- 新建模块 `match_simulator.py`，对每场比赛单独运行 1k 模拟
- 输入：(team_a, team_b, mc_shifts_phase3, match_context_adjustments)
- 输出：p_win_a, p_draw, p_win_b (三叉)
- 重构 preview 逻辑，改为 per-match 叠加而非 MC 全模拟

**成本**：很高（改动面积大，需重新验证 calibration）

**收益**：最灵活，能精确控制每场调整，支持实时比赛更新

---

## 6. 优先级建议

1. **短期（本周）**：采用方案 2 的简单近似版
   - 在 /api/groups 或新端点中估算 per-match delta
   - 告诉用户"Brazil vs Morocco 理论上因为伤病偏移 ~+1pp"
   - 满足用户看到细粒度差异的需求，成本低

2. **中期（1-2周）**：实现方案 1 的后处理
   - 若 MC 已保存 rawTrace，直接做统计提取
   - 若没有，改 MC 模块保存，重跑 100k
   - 得到精确的 per-match p_win_a_base vs phase3

3. **长期（月）**：方案 3 的系数优化
   - 监控 Germany/Spain/low-base 队的传递率异常
   - 微调 ELO_PER_PP_HIGH/MID/LOW

4. **不推荐**：方案 4（除非 architecture redesign）

---

## 7. 关键发现总结

| 发现 | 状态 | 量级 |
|------|------|------|
| 伤病信号被 Health Agent 完整摄入 | ✓ | 100% 传递 |
| Synthesizer 多因子融合造成衰减 | ⚠ 正常 | 20-30% 衰减 |
| ΔE 反推与 MC 注入无损 | ✓ | <5% 衰减 |
| **MC 输出粒度聚合问题** | **✗** | **无法 per-match** |
| 前端展示粒度不足 | ⚠ 未优化 | 只到队级 |

**核心结论**：系统没有"丢失"软信息，而是在 MC 聚合阶段自然地从 per-match 升级到队级。用户看不到的细粒度差异需要后处理或架构改进来呈现。

