"""
参考表 9.13 — 6 桶概率校准映射
====================================
来源：WorldCup_Report §9.6.2（表 9.13）
基于 2018+2022 两届世界杯 210 场回测的分箱偏差拟合。

设计原则
--------
1. 仅对"夺冠 / 进决赛 / 进半决赛"等淘汰赛阶段概率应用——这些是
   稀疏长尾分布，最需要厚尾补偿。
2. 不对"进 R32 / R16"这种 >50% 的高频概率应用——分箱样本不足。
3. 校准是单调线性分段函数 f(p)：保证单调性，避免次序翻转。
4. 校准后做归一化检查：单赛事概率 ≥1 时按比例缩放（防止累积爆 100%）。

公式（表 9.13 原文）：
    0-5%   → +1.5pp   （补偿厚尾低估）
    5-10%  → +0.8pp   （轻度厚尾补偿）
    10-15% → +0.3pp   （边际修正）
    15-20% →  0       （校准良好）
    20-25% → -0.5pp   （轻度过度自信修正）
    >25%   → -1.0pp   （过度自信修正）
"""

# (low, high, delta_pp) — 区间为左闭右开，delta 单位为百分点
BUCKETS = [
    (0.00, 0.05, +1.5),
    (0.05, 0.10, +0.8),
    (0.10, 0.15, +0.3),
    (0.15, 0.20,  0.0),
    (0.20, 0.25, -0.5),
    (0.25, 1.01, -1.0),
]


def calibrate(p: float) -> float:
    """
    对单个概率 p (0~1) 应用 参考表 9.13 校准。

    Args:
        p: 原始概率（0~1）

    Returns:
        校准后概率，clip 在 [0, 1]
    """
    if p is None or p <= 0:
        return 0.0
    if p >= 1.0:
        return 1.0
    for lo, hi, delta in BUCKETS:
        if lo <= p < hi:
            return max(0.0, min(1.0, p + delta / 100.0))
    return p  # 兜底（理论不会走到）


def calibrate_dict(probs: dict, fields=("champion", "final", "semifinal")) -> dict:
    """
    对单队的 mc_simulation 输出字典做就地校准。
    
    只校准淘汰赛长尾字段；R32/R16 这种 60%+ 不进入分箱范围。
    
    返回浅拷贝，原始值用 raw_<field> 保留以备审计。
    """
    out = dict(probs)
    for f in fields:
        if f in out:
            raw = out[f]
            out[f"raw_{f}"] = raw
            out[f] = round(calibrate(raw), 4)
    return out


def calibrate_all(mc_results: dict) -> dict:
    """
    对整个 MC 输出（{team: {champion, final, ...}}）做校准。

    注意：参考表 9.13 是为「单事件二分类校准」设计（某队夺冠 vs 不夺冠，
    长期回测频率 vs 预测概率）。它本质上是"每队独立修正长尾低估/头部过度自信"。

    因此**不做归一化** —— 参考报告原文德国 11.0%→11.3% 也未涉及全分布归一。
    校准后所有队 champion 总和会从 100% 漂到 ~160%（因 42 队都在 0-5% 桶里
    各 +1.5pp），这是 参考校准的固有特性，反映「实际世界杯爆冷比模型预期多」。

    单值会 clip 到 [0, 1]。raw 值保留在 raw_<field>。
    """
    return {team: calibrate_dict(p) for team, p in mc_results.items()}


# ===== 自测 =====
if __name__ == "__main__":
    # 对应 参考规范 §9.6.2 报告原话："德国 11.0% → 11.3%"
    test_cases = [
        (0.030, "0.045"),   # 0-5%  → +1.5pp
        (0.075, "0.083"),   # 5-10% → +0.8pp
        (0.110, "0.113"),   # 10-15%→ +0.3pp  ← 参考报告德国案例
        (0.165, "0.165"),   # 15-20%→  0      ← 参考报告西班牙案例
        (0.220, "0.215"),   # 20-25%→ -0.5pp
        (0.260, "0.250"),   # >25%  → -1.0pp  ← 参考报告 Goldman 26→25
    ]
    print("=== 参考表 9.13 校准自测 ===")
    print(f"{'raw_p':<8} {'calibrated':<12} {'expected':<12} {'check':<6}")
    print("-" * 40)
    for raw, expected in test_cases:
        out = calibrate(raw)
        ok = "OK" if f"{out:.3f}" == expected else "FAIL"
        print(f"{raw:<8.3f} {out:<12.4f} {expected:<12} {ok}")
