"""
Poisson 进球分布模型族（参考 Layer 3）
=========================================

实现 4 个 Karlis-Ntzoufras 教科书模型：

1. Basic Poisson（模型 4）
2. Dixon-Coles 修正（模型 5）：低比分相关性修正 ρ ∈ [-0.15, -0.03]
3. Bivariate Poisson（模型 6）：Karlis-Ntzoufras 2003，
   通过共享潜变量 λ3 显式建模两队进球协方差
4. ZIGP 零膨胀广义泊松（模型 7）：处理 0:0 过度离散
   π0 + (1-π0)·GP(λ, θ)

ensemble_score_matrix() 提供加权融合（默认 DC=0.5 + Bivariate=0.3 + ZIGP=0.2，
权重对应 参考报告各模型 Brier Score 的反向加权）
"""
from __future__ import annotations
import numpy as np
from scipy.stats import poisson
from math import factorial, exp


def basic_poisson_score_matrix(lambda_a: float, lambda_b: float, max_goals: int = 8) -> np.ndarray:
    """
    基础 Poisson 比分矩阵
    matrix[i][j] = P(A 进 i 球, B 进 j 球)
    """
    pa = poisson.pmf(np.arange(max_goals + 1), lambda_a)
    pb = poisson.pmf(np.arange(max_goals + 1), lambda_b)
    return np.outer(pa, pb)


def dixon_coles_correction(matrix: np.ndarray, lambda_a: float, lambda_b: float,
                            rho: float = -0.05) -> np.ndarray:
    """
    Dixon-Coles 低比分相关性修正（参考方法论 rho=-0.05）
    校正 0-0、1-1 被基础 Poisson 系统性低估的问题
    """
    m = matrix.copy()
    # τ 修正因子（仅作用于低比分 0-0, 0-1, 1-0, 1-1）
    m[0][0] *= 1 - lambda_a * lambda_b * rho
    m[0][1] *= 1 + lambda_a * rho
    m[1][0] *= 1 + lambda_b * rho
    m[1][1] *= 1 - rho
    # 重新归一化
    m = m / m.sum()
    return m


# ============================================================
# 模型 6: Bivariate Poisson (Karlis & Ntzoufras 2003)
# ============================================================
def bivariate_poisson_score_matrix(lambda_a: float, lambda_b: float,
                                     lambda_c: float = 0.1,
                                     max_goals: int = 8) -> np.ndarray:
    """
    Bivariate Poisson:
        X1 = Y1 + Y3,  X2 = Y2 + Y3
        Y1 ~ Poisson(λ_a - λ_c)
        Y2 ~ Poisson(λ_b - λ_c)
        Y3 ~ Poisson(λ_c)   ← 共同冲击（同一场比赛的协变量）

    联合 PMF：
        P(X1=x, X2=y) = exp(-λ1-λ2-λ3) · (λ1^x / x!) · (λ2^y / y!)
                        · Σ_{k=0}^{min(x,y)} C(x,k)·C(y,k)·k! · (λ3/(λ1·λ2))^k

    其中 λ1 = λ_a - λ_c, λ2 = λ_b - λ_c, λ3 = λ_c

    参数：
        lambda_a, lambda_b: 两队边缘进球期望
        lambda_c: 协方差参数（建议 [0.05, 0.20]，0 时退化为独立 Poisson）
    """
    # λ_c 不能超过任一边缘期望
    lambda_c = max(0.0, min(lambda_c, min(lambda_a, lambda_b) - 0.01))
    lam1 = lambda_a - lambda_c
    lam2 = lambda_b - lambda_c
    lam3 = lambda_c

    matrix = np.zeros((max_goals + 1, max_goals + 1))
    base = exp(-(lam1 + lam2 + lam3))

    # 预计算组合数与阶乘以提速
    fact = [factorial(i) for i in range(max_goals + 2)]

    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            kmax = min(x, y)
            inner = 0.0
            for k in range(kmax + 1):
                # C(x,k) * C(y,k) * k!
                c1 = fact[x] // (fact[k] * fact[x - k])
                c2 = fact[y] // (fact[k] * fact[y - k])
                term = c1 * c2 * fact[k]
                if lam1 > 0 and lam2 > 0:
                    term *= (lam3 / (lam1 * lam2)) ** k
                elif k > 0:
                    # lam1 或 lam2 = 0 时 k>=1 项为 0
                    term = 0.0
                inner += term
            matrix[x][y] = base * (lam1 ** x / fact[x]) * (lam2 ** y / fact[y]) * inner

    # 数值清理 + 归一化
    matrix = np.maximum(matrix, 0.0)
    s = matrix.sum()
    if s > 0:
        matrix /= s
    return matrix


# ============================================================
# 模型 7: ZIGP (Zero-Inflated Generalized Poisson)
# ============================================================
def _generalized_poisson_pmf(k: int, lam: float, theta: float) -> float:
    """
    Consul-Jain Generalized Poisson:
        P(K=k) = λ(λ+kθ)^(k-1) · exp(-λ-kθ) / k!
    
    θ ∈ (-1, 1) 调节 over/under-dispersion；θ=0 退化为标准 Poisson
    """
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    base = lam + k * theta
    if base <= 0:
        return 0.0
    return lam * (base ** (k - 1)) * exp(-base) / factorial(k)


def zigp_score_matrix(lambda_a: float, lambda_b: float,
                       theta_a: float = 0.05, theta_b: float = 0.05,
                       pi_zero: float = 0.04,
                       max_goals: int = 8) -> np.ndarray:
    """
    Zero-Inflated Generalized Poisson:
        - 概率 π0 是"结构性零"（赛事氛围紧绷不进球）
        - 概率 1-π0 走 GP(λ, θ)
    
    用于解决标准 Poisson 在 0:0 比分上的低估问题，
    参考报告 2.2.3 章节指出在低样本国家队场景中校准度优于标准 Poisson。

    参数：
        theta_a/b: GP 离散参数，建议 [0.02, 0.15]
        pi_zero: 结构零概率，建议 [0.02, 0.08]
    """
    theta_a = max(0.0, min(0.3, theta_a))
    theta_b = max(0.0, min(0.3, theta_b))
    pi_zero = max(0.0, min(0.15, pi_zero))

    # GP 边缘分布
    pa = np.array([_generalized_poisson_pmf(k, lambda_a, theta_a) for k in range(max_goals + 1)])
    pb = np.array([_generalized_poisson_pmf(k, lambda_b, theta_b) for k in range(max_goals + 1)])
    # 归一化（GP 在数值近似时可能不严格归一）
    pa = pa / pa.sum() if pa.sum() > 0 else pa
    pb = pb / pb.sum() if pb.sum() > 0 else pb

    # GP 联合分布（独立假设）
    gp_matrix = np.outer(pa, pb)

    # 零膨胀混合：以 π0 概率塞到 (0,0)
    matrix = (1 - pi_zero) * gp_matrix
    matrix[0][0] += pi_zero
    return matrix


# ============================================================
# 集成：DC + Bivariate + ZIGP 加权
# ============================================================
def ensemble_score_matrix(lambda_a: float, lambda_b: float,
                           rho: float = -0.05,
                           lambda_c: float = 0.10,
                           theta: float = 0.05, pi_zero: float = 0.04,
                           weights: dict = None,
                           max_goals: int = 8) -> dict:
    """
    参考报告 2.3.4 节"贝叶斯加权"：
      最终 P(score) = Σ w_i · P_i(score)
    
    默认权重基于报告中各模型的 Brier 表现：
      DC=0.5 / Bivariate=0.3 / ZIGP=0.2
    
    Returns:
        {
            "matrix": 集成比分矩阵,
            "components": {model_name: matrix},
            "weights": ...,
        }
    """
    if weights is None:
        weights = {"dc": 0.5, "bivariate": 0.3, "zigp": 0.2}

    # 各模型矩阵
    base = basic_poisson_score_matrix(lambda_a, lambda_b, max_goals)
    dc = dixon_coles_correction(base, lambda_a, lambda_b, rho)
    biv = bivariate_poisson_score_matrix(lambda_a, lambda_b, lambda_c, max_goals)
    zigp = zigp_score_matrix(lambda_a, lambda_b, theta, theta, pi_zero, max_goals)

    # 加权融合
    total = sum(weights.values())
    w = {k: v / total for k, v in weights.items()}
    ensemble = w["dc"] * dc + w["bivariate"] * biv + w["zigp"] * zigp

    # 归一化（避免数值漂移）
    ensemble = ensemble / ensemble.sum()

    return {
        "matrix": ensemble,
        "components": {"dc": dc, "bivariate": biv, "zigp": zigp},
        "weights": w,
    }


def expected_lambdas(elo_a: float, elo_b: float, xg_a: float, xg_b: float,
                     xg_against_a: float, xg_against_b: float,
                     home_advantage: float = 1.0) -> tuple:
    """
    根据 Elo + xG 数据计算两队的期望进球（lambda）
    
    参考方法论：xG 权重 0.7（70% 过程 + 30% Elo 实力差）
    """
    # 实力差 → 进球修正系数
    elo_diff = elo_a - elo_b
    elo_factor_a = 1.0 + elo_diff / 600.0 * 0.5  # Elo 差每 600 分修正 50%
    elo_factor_b = 1.0 - elo_diff / 600.0 * 0.5
    
    # 期望进球 = (本队进攻 xG + 对方失球 xG) / 2 × Elo 修正
    lambda_a = ((xg_a + xg_against_b) / 2.0) * elo_factor_a * home_advantage
    lambda_b = ((xg_b + xg_against_a) / 2.0) * elo_factor_b
    
    # 边界保护
    lambda_a = max(0.2, min(5.0, lambda_a))
    lambda_b = max(0.2, min(5.0, lambda_b))
    
    return lambda_a, lambda_b


def match_outcome_from_score_matrix(matrix: np.ndarray) -> dict:
    """从比分矩阵汇总胜平负概率"""
    n = matrix.shape[0]
    p_win_a = sum(matrix[i][j] for i in range(n) for j in range(n) if i > j)
    p_draw = sum(matrix[i][i] for i in range(n))
    p_win_b = sum(matrix[i][j] for i in range(n) for j in range(n) if i < j)
    return {
        "p_win_a": round(p_win_a, 4),
        "p_draw": round(p_draw, 4),
        "p_win_b": round(p_win_b, 4),
    }


def top_scorelines(matrix: np.ndarray, top_n: int = 5) -> list:
    """返回 Top N 最可能比分"""
    n = matrix.shape[0]
    scorelines = []
    for i in range(n):
        for j in range(n):
            scorelines.append(((i, j), matrix[i][j]))
    scorelines.sort(key=lambda x: -x[1])
    return [(score, round(p, 4)) for score, p in scorelines[:top_n]]


def predict_match(team_a: dict, team_b: dict, home_advantage: float = 1.0,
                   rho: float = -0.05, max_goals: int = 8) -> dict:
    """
    完整单场预测（Dixon-Coles 单模型版本，保留作回退/对照）
    返回胜平负 + 期望进球 + 最可能比分
    
    输入：
        team_a/b dict 含 elo, xg_for, xg_against
    """
    lambda_a, lambda_b = expected_lambdas(
        team_a["elo"], team_b["elo"],
        team_a["xg_for"], team_b["xg_for"],
        team_a["xg_against"], team_b["xg_against"],
        home_advantage,
    )
    
    # 基础 Poisson
    base_matrix = basic_poisson_score_matrix(lambda_a, lambda_b, max_goals)
    
    # Dixon-Coles 修正
    dc_matrix = dixon_coles_correction(base_matrix, lambda_a, lambda_b, rho)
    
    # 输出
    outcome = match_outcome_from_score_matrix(dc_matrix)
    scorelines = top_scorelines(dc_matrix, top_n=5)
    
    return {
        "lambda_a": round(lambda_a, 3),
        "lambda_b": round(lambda_b, 3),
        "outcome": outcome,
        "top_scorelines": scorelines,
        "model": "dixon-coles",
    }


def predict_match_ensemble(team_a: dict, team_b: dict, home_advantage: float = 1.0,
                            rho: float = -0.05, lambda_c: float = 0.10,
                            theta: float = 0.05, pi_zero: float = 0.04,
                            weights: dict = None, max_goals: int = 8) -> dict:
    """
    参考规范 §2.3.4 集成版单场预测：贝叶斯加权融合 3 个进球分布模型
    
        最终 P(score) = w_dc·DC + w_biv·Bivariate + w_zigp·ZIGP
    
    默认权重对应 参考报告各模型 Brier Score 反向加权：
        DC = 0.50  （Dixon-Coles 低比分修正）
        Bivariate = 0.30  （Karlis-Ntzoufras 显式协方差）
        ZIGP = 0.20  （零膨胀广义 Poisson，修复 0-0 过度离散）
    
    相比单 DC 模型的改善：
      - 0-0、1-1 概率更接近实际数据
      - 高比分尾部更现实（DC 单模型对 3-2/4-2 略低估）
      - 1-0 / 0-1 等"低破门胶着比赛"概率与历史频率更吻合
    """
    lambda_a, lambda_b = expected_lambdas(
        team_a["elo"], team_b["elo"],
        team_a["xg_for"], team_b["xg_for"],
        team_a["xg_against"], team_b["xg_against"],
        home_advantage,
    )
    
    ens = ensemble_score_matrix(
        lambda_a, lambda_b,
        rho=rho, lambda_c=lambda_c,
        theta=theta, pi_zero=pi_zero,
        weights=weights, max_goals=max_goals,
    )
    
    outcome = match_outcome_from_score_matrix(ens["matrix"])
    scorelines = top_scorelines(ens["matrix"], top_n=15)  # 取 Top 15 供下游做"主胜/客胜/平"分类查询
    
    return {
        "lambda_a": round(lambda_a, 3),
        "lambda_b": round(lambda_b, 3),
        "outcome": outcome,
        "top_scorelines": scorelines,
        "model": "ensemble (DC+Bivariate+ZIGP)",
        "weights": ens["weights"],
    }


# ===== 自测 =====
if __name__ == "__main__":
    spain = {"elo": 2155, "xg_for": 2.50, "xg_against": 0.65}
    france = {"elo": 2105, "xg_for": 2.45, "xg_against": 0.73}
    curacao = {"elo": 1480, "xg_for": 0.85, "xg_against": 1.55}
    
    print("=== Spain vs France ===")
    result = predict_match(spain, france)
    print(f"  λ_Spain = {result['lambda_a']}, λ_France = {result['lambda_b']}")
    print(f"  胜平负: {result['outcome']}")
    print(f"  Top 5 比分:")
    for score, p in result['top_scorelines']:
        print(f"    {score[0]}-{score[1]}: {p*100:.1f}%")
    
    print("\n=== Germany vs Curacao ===")
    germany = {"elo": 1932, "xg_for": 2.20, "xg_against": 0.95}
    result = predict_match(germany, curacao)
    print(f"  λ_Germany = {result['lambda_a']}, λ_Curacao = {result['lambda_b']}")
    print(f"  胜平负: {result['outcome']}")
    print(f"  Top 5 比分:")
    for score, p in result['top_scorelines']:
        print(f"    {score[0]}-{score[1]}: {p*100:.1f}%")
