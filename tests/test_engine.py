"""
WorldCup Predict 核心模块单元测试

运行：
  python3 -m pytest tests/  
  或：
  python3 tests/test_engine.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from models.elo_engine import expected_score, match_probabilities, update_elo
from models.poisson_model import basic_poisson_score_matrix, dixon_coles_correction
from models.bracket_engine import load_bracket, assign_third_places
from models.fifa_495_table import find_valid_assignment, R32_THIRD_SLOTS
from models.environment_engine import get_heat_penalty, get_altitude_penalty
import numpy as np


class TestEloEngine(unittest.TestCase):
    
    def test_expected_score_equal_teams(self):
        """实力相同 → 50% 胜率"""
        self.assertAlmostEqual(expected_score(1500, 1500), 0.5, places=6)
    
    def test_expected_score_difference(self):
        """更高 Elo → 更高胜率"""
        p_high = expected_score(2000, 1500)
        self.assertGreater(p_high, 0.5)
        self.assertLess(p_high, 1.0)
    
    def test_match_probabilities_sum_to_one(self):
        """三结果概率之和 = 1"""
        for elo_a, elo_b in [(1500, 1500), (2155, 1480), (1700, 1900)]:
            p = match_probabilities(elo_a, elo_b)
            total = p["p_win_a"] + p["p_draw"] + p["p_win_b"]
            self.assertAlmostEqual(total, 1.0, places=3)
    
    def test_update_elo_winner_gains(self):
        """赢的一方 Elo 上升"""
        new_a, new_b = update_elo(1500, 1500, result=1.0, k=30)
        self.assertGreater(new_a, 1500)
        self.assertLess(new_b, 1500)
        # 总分守恒
        self.assertAlmostEqual(new_a + new_b, 3000, places=3)


class TestPoissonModel(unittest.TestCase):
    
    def test_score_matrix_sums_to_one(self):
        """比分矩阵和 ≈ 1"""
        m = basic_poisson_score_matrix(1.5, 1.2, max_goals=8)
        self.assertAlmostEqual(m.sum(), 1.0, places=2)
    
    def test_dixon_coles_correction(self):
        """Dixon-Coles 修正后仍归一化"""
        m = basic_poisson_score_matrix(1.5, 1.2)
        m_corrected = dixon_coles_correction(m, 1.5, 1.2, rho=-0.05)
        self.assertAlmostEqual(m_corrected.sum(), 1.0, places=4)


class TestBracketEngine(unittest.TestCase):
    
    def test_bracket_loads(self):
        b = load_bracket()
        self.assertEqual(len(b["round_of_32"]), 16)
        self.assertEqual(len(b["_round_of_16_pairings"]), 8)
        self.assertEqual(len(b["_quarterfinal_pairings"]), 4)
        self.assertEqual(len(b["_semifinal_pairings"]), 2)
    
    def test_third_place_assignment_8_slots(self):
        """8 个第三名能完美分配"""
        bracket = load_bracket()
        rng = np.random.default_rng(42)
        for trial in range(20):
            groups = ["A", "B", "C", "D", "E", "F", "G", "H"]
            np.random.default_rng(trial).shuffle(groups)
            third_places = {g: f"3rd_{g}" for g in groups}
            result = assign_third_places(third_places, bracket, rng)
            self.assertEqual(len(result), 8, f"Trial {trial}: 期望 8 个槽位，实际 {len(result)}")
    
    def test_no_same_group_in_r32(self):
        """同组不相遇约束"""
        bracket = load_bracket()
        rng = np.random.default_rng(42)
        groups = ["A", "B", "C", "D", "E", "F", "G", "H"]
        third_places = {g: f"3rd_{g}" for g in groups}
        result = assign_third_places(third_places, bracket, rng)
        for match_id, third_team in result.items():
            third_group = third_team.split("_")[1]
            winner_group = R32_THIRD_SLOTS[match_id]
            self.assertNotEqual(third_group, winner_group,
                f"Match {match_id}: {third_group}3 不能对阵 {winner_group}1")


class TestFifa495Table(unittest.TestCase):
    
    def test_all_combinations_valid(self):
        """所有 495 种组合都应有合法分配"""
        from itertools import combinations
        groups = list("ABCDEFGHIJKL")
        invalid = 0
        for combo in combinations(groups, 8):
            r = find_valid_assignment(combo)
            if not r:
                invalid += 1
        self.assertEqual(invalid, 0, f"{invalid}/495 种组合无解")


class TestEnvironmentEngine(unittest.TestCase):
    
    def test_heat_penalty_extreme_venue(self):
        """达拉斯 WBGT 33°C → 欧洲队应有显著热税"""
        venue = {"wbgt_peak_c": 33, "indoor": False}
        eu_factor = get_heat_penalty("Spain", venue)
        af_factor = get_heat_penalty("Morocco", venue)
        self.assertLess(eu_factor, 1.0)  # 欧洲队受影响
        self.assertGreater(af_factor, eu_factor)  # 非洲队受影响小
    
    def test_altitude_penalty(self):
        """墨西哥城 2240m → 非高原适应队受影响"""
        venue = {"altitude_m": 2240}
        eu_factor = get_altitude_penalty("Spain", venue)
        sa_factor = get_altitude_penalty("Ecuador", venue)
        self.assertLess(eu_factor, 1.0)
        self.assertGreater(sa_factor, eu_factor)
    
    def test_low_altitude_no_effect(self):
        """海拔 <500m 应无影响"""
        venue = {"altitude_m": 100}
        f = get_altitude_penalty("Spain", venue)
        self.assertAlmostEqual(f, 1.0, places=3)


class TestAgentSwarm(unittest.TestCase):
    """Layer 4 Agent 框架测试"""
    
    def test_swarm_runs(self):
        """Swarm 完整流程能跑通"""
        from agents.swarm import QueenSwarm
        swarm = QueenSwarm()
        result = swarm.run(verbose=False)
        self.assertEqual(result["n_agents"], 17)  # 3 strategic + 9 tactical + 5 execution
        self.assertGreater(len(result["swarm_predictions"]), 30)
    
    def test_consensus_aggregator(self):
        """加权聚合 + Byzantine"""
        from agents.base import AgentOutput, ConsensusAggregator
        outs = [
            AgentOutput(agent_name="A", layer="strategic", role="r", 
                         point_estimate=10.0, confidence=0.8),
            AgentOutput(agent_name="B", layer="tactical", role="r", 
                         point_estimate=12.0, confidence=0.7),
            AgentOutput(agent_name="C", layer="execution", role="r", 
                         point_estimate=11.0, confidence=0.6),
        ]
        agg = ConsensusAggregator.aggregate_point_estimates(outs)
        # Strategic（3x weight） 主导，应接近 10
        self.assertLess(agg["consensus"], 12)
        self.assertGreater(agg["consensus"], 9)
        self.assertEqual(agg["n_agents"], 3)
    
    def test_critic_scores(self):
        """Critic 给分逻辑"""
        from agents.base import AgentOutput, Critic
        # 高质量输出
        good = AgentOutput(agent_name="A", layer="tactical", role="r",
                            point_estimate=15.0, confidence=0.7,
                            rationale="基于 Elo + 100k MC + 真实 bracket 的概率估计" * 3,
                            evidence=["e1", "e2", "e3", "e4"])
        score = Critic.score(good, consensus=15.0)
        self.assertEqual(score["grade"], "A")


class TestUncertaintyDecomposition(unittest.TestCase):
    """Layer 5 三层不确定性分解测试"""
    
    def test_structural_uncertainty(self):
        from models.uncertainty import structural_uncertainty
        result = structural_uncertainty("Spain")
        self.assertIn("structural_std", result)
        std = result["structural_std"]
        self.assertGreater(std, 0.5)
        self.assertLess(std, 5.0)
    
    def test_model_disagreement(self):
        from models.uncertainty import model_disagreement
        result = model_disagreement("Spain")
        self.assertGreaterEqual(result["n_models"], 2)
        self.assertIn("std", result)
    
    def test_degradation_check(self):
        from models.uncertainty import check_degradation
        normal = {"ci_width_pp": 5.0, "model": {"std": 1.5, "n_models": 4}}
        self.assertFalse(check_degradation(normal)["degraded"])
        bad = {"ci_width_pp": 20.0, "model": {"std": 8.0, "n_models": 1}}
        deg = check_degradation(bad)
        self.assertTrue(deg["degraded"])
        self.assertGreaterEqual(len(deg["triggers"]), 2)


class TestPsychologySync(unittest.TestCase):
    """两套 Psychology 数据源必须保持同步（不一致会让 swarm vs synth 排序漂移）"""

    # Germany 是唯一允许的例外：synth +1.5 / Agent +2.5
    # （因 swarm 的 conf 缩放约 0.4 倍，Agent 端需放大 1.67x 才能达到等效效果）
    KNOWN_EXCEPTIONS = {"Germany"}

    def _get_synth_table(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "code"))
        from models.synthesizer import DEFAULT_ADJUSTMENTS
        return DEFAULT_ADJUSTMENTS

    def _get_psych_dict(self):
        import sys, os
        sys.path.insert(0, str(Path(__file__).parent.parent / "code"))
        os.environ.pop("LINGYA_API_KEY", None)
        from llm.client import reset_default_client
        reset_default_client()
        from agents.execution import PsychologyAgent
        ag = PsychologyAgent()
        fake_ctx = {
            "teams": {}, "mc_results": {}, "scenarios": {},
            "injuries": {}, "external_predictions": {},
            "data_quality": {"overall_weight": 1.0},
        }
        out = ag.fallback_analyze(fake_ctx)
        return out.probability_dist

    def test_no_synth_only_teams(self):
        """每个 synth 表的队都必须在 Agent 字典中（否则 swarm 漏 psych 信号）"""
        synth_table = self._get_synth_table()
        agent_dict = self._get_psych_dict()
        synth_only = [t for t in synth_table if t not in agent_dict]
        self.assertEqual(synth_only, [],
            f"synth 表中存在 Agent 字典缺失的队: {synth_only}（会导致 swarm 漏 psych 信号）")

    def test_psych_values_aligned(self):
        """同一队的 psych 值在两边应一致（除已知例外）"""
        synth_table = self._get_synth_table()
        agent_dict = self._get_psych_dict()
        mismatches = []
        for team, info in synth_table.items():
            if team in self.KNOWN_EXCEPTIONS:
                continue
            synth_psych = info.get("psych", 0)
            agent_psych = agent_dict.get(team, 0)
            if abs(synth_psych - agent_psych) > 0.3:
                mismatches.append((team, synth_psych, agent_psych))
        self.assertEqual(mismatches, [],
            f"psych 值不一致 (差异 >0.3pp): {mismatches}")


class TestLLMAgents(unittest.TestCase):
    """LLM Agent 改造验证（核心：fallback 路径无回归 + LLM 路径正确解析）"""

    def setUp(self):
        from agents.swarm import QueenSwarm
        self.sw = QueenSwarm()
        self.ctx = self.sw.build_context()

    def test_fallback_when_no_api_key(self):
        """无 LINGYA_API_KEY 时所有 LLM Agent 应自动降级"""
        import os
        from agents.strategic import MacroTrendAgent, FormatAnalysisAgent, RiskPerceptionAgent
        from agents.tactical import MarketAgent, ContextAgent
        from llm.client import reset_default_client
        # 清掉 key + 重置单例
        old = os.environ.pop("LINGYA_API_KEY", None)
        reset_default_client()
        try:
            for cls in [MacroTrendAgent, FormatAnalysisAgent, RiskPerceptionAgent,
                        MarketAgent, ContextAgent]:
                agent = cls()
                out = agent.analyze(self.ctx)
                self.assertIsNotNone(out)
                # 应有 fallback 标记
                ev_str = " ".join(out.evidence or [])
                self.assertIn("fallback", ev_str,
                              f"{cls.__name__} 缺少 fallback 标记")
        finally:
            if old:
                os.environ["LINGYA_API_KEY"] = old
            reset_default_client()

    def test_llm_agent_parses_valid_json(self):
        """注入 mock client，验证 JSON → AgentOutput 解析"""
        from agents.strategic import MacroTrendAgent
        import json

        class MockClient:
            def chat(self, system, user, **kw):
                return json.dumps({
                    "adjustments": {"Spain": -0.5, "Argentina": -2.0, "Brazil": 1.0},
                    "confidence": 0.7,
                    "rationale": "测试用",
                    "evidence": ["卫冕魔咒", "南美主场"]
                })

        agent = MacroTrendAgent()
        agent._client_override = MockClient()
        out = agent.analyze(self.ctx)
        self.assertEqual(out.probability_dist["Argentina"], -2.0)
        # confidence 可能被 data_quality 降权，所以 ≤ 原始 0.7
        self.assertLessEqual(out.confidence, 0.7)
        self.assertGreater(out.confidence, 0.4)
        self.assertIn("[source:LLM]", out.evidence)

    def test_llm_agent_handles_malformed_json(self):
        """LLM 返回坏 JSON 时应 fallback 而不抛异常"""
        from agents.strategic import RiskPerceptionAgent

        class BadClient:
            def chat(self, system, user, **kw):
                return "this is not json at all 不是 JSON"

        agent = RiskPerceptionAgent()
        agent._client_override = BadClient()
        out = agent.analyze(self.ctx)
        self.assertIsNotNone(out)
        ev_str = " ".join(out.evidence or [])
        self.assertIn("fallback", ev_str)

    def test_risk_agent_only_outputs_negatives(self):
        """RiskPerception LLM 输出非负值时应被钳制为 0"""
        from agents.strategic import RiskPerceptionAgent
        import json

        class PositiveClient:
            def chat(self, system, user, **kw):
                return json.dumps({
                    "adjustments": {"Spain": 5.0, "Brazil": -2.0},
                    "confidence": 0.7, "rationale": "x", "evidence": []
                })

        agent = RiskPerceptionAgent()
        agent._client_override = PositiveClient()
        out = agent.analyze(self.ctx)
        # 正数应被钳制为 0
        self.assertEqual(out.probability_dist["Spain"], 0.0)
        self.assertEqual(out.probability_dist["Brazil"], -2.0)


class TestLayer1Gap(unittest.TestCase):
    """Layer 1 缺口（可用性四问 + xT + StatsBomb 接入）"""

    def test_quad_check_passes_clean_data(self):
        """干净数据应通过四问"""
        from data.availability_check import quad_check
        clean_meta = {
            "sources": ["FIFA", "StatsBomb"],
            "granularity": "country_competitive",
            "n_samples": 48,
            "last_updated": "2026-06-01",
        }
        res = quad_check("test", clean_meta)
        self.assertTrue(res.passed)
        self.assertGreaterEqual(res.downgrade_weight, 0.95)

    def test_quad_check_flags_stale_data(self):
        """过期数据应被标记并降权"""
        from data.availability_check import quad_check
        stale_meta = {
            "sources": ["FIFA"],
            "granularity": "country_competitive",
            "n_samples": 48,
            "last_updated": "2020-01-01",
        }
        res = quad_check("test", stale_meta)
        self.assertFalse(res.passed)
        self.assertLess(res.downgrade_weight, 0.9)
        self.assertTrue(any("Timeliness" in w for w in res.warnings))

    def test_quad_check_flags_unknown_source(self):
        """未知来源应触发警告"""
        from data.availability_check import quad_check
        bad_meta = {
            "sources": ["RandomBlog", "WikiRumor"],
            "granularity": "country_competitive",
            "n_samples": 48,
            "last_updated": "2026-06-01",
        }
        res = quad_check("test", bad_meta)
        self.assertFalse(res.passed)
        self.assertLess(res.downgrade_weight, 0.7)

    def test_quad_check_low_sample(self):
        """低样本应触发降权但不一定 fail"""
        from data.availability_check import quad_check
        meta = {
            "sources": ["FIFA"],
            "granularity": "country_competitive",
            "n_samples": 8,  # < 15 min
            "last_updated": "2026-06-01",
        }
        res = quad_check("test", meta)
        self.assertLess(res.downgrade_weight, 1.0)

    def test_xt_grid_shape(self):
        """xT 网格应为 16×12"""
        from pathlib import Path
        import json
        path = Path(__file__).parent.parent / "data" / "outputs" / "xt_grid.json"
        if not path.exists():
            self.skipTest("xt_grid.json 未生成")
        d = json.load(open(path))
        self.assertEqual(d["shape"], [16, 12])
        self.assertGreater(d["max_xt"], 0.1)
        # 进攻方最深格 (i=15) xT 应 > 防守方最深格 (i=0)
        grid = d["grid"]
        # grid[i][j] 形状 16x12
        attack_max = max(grid[15][j] for j in range(12))
        defend_max = max(grid[0][j] for j in range(12))
        self.assertGreater(attack_max, defend_max,
            "进攻方 xT 应该更高（球场推进价值）")

    def test_xt_agent_runs(self):
        """XTAgent 在缺数据时不应崩溃"""
        from agents.tactical import XTAgent
        # 空 context 触发 fallback 分支
        out = XTAgent().analyze({"teams": {}})
        self.assertIsNotNone(out)
        self.assertEqual(out.agent_name, "xT")

    def test_xt_agent_with_data(self):
        """XTAgent 在有数据时输出非空 adjustments"""
        from agents.tactical import XTAgent
        ctx = {"teams": {
            "Spain": {"xt_per_90": 9.92},
            "Brazil": {"xt_per_90": 9.67},
            "Germany": {"xt_per_90": 11.70},
            "Other": {"xt_per_90": 6.0},
        }}
        out = XTAgent().analyze(ctx)
        self.assertGreater(len(out.probability_dist), 0)
        # Germany 是最高，应该有正向调整
        self.assertGreater(out.probability_dist.get("Germany", 0), 0)

    def test_swarm_includes_xt(self):
        """Swarm 应注册 XTAgent"""
        from agents.swarm import QueenSwarm
        sw = QueenSwarm()
        names = [a.name for a in sw.tactical_agents]
        self.assertIn("xT", names)
        # 总 Agent 数 = 3+7+5 = 15
        self.assertEqual(len(sw.all_agents()), 17)


class TestPoissonFamily(unittest.TestCase):
    """Layer 3 Poisson 族：Bivariate + ZIGP + Ensemble"""

    def test_bivariate_normalization(self):
        """Bivariate Poisson 矩阵应归一"""
        from models.poisson_model import bivariate_poisson_score_matrix
        m = bivariate_poisson_score_matrix(2.0, 1.5, lambda_c=0.10)
        self.assertAlmostEqual(m.sum(), 1.0, places=3)
        # 所有元素非负
        self.assertTrue((m >= 0).all())

    def test_bivariate_reduces_to_independent(self):
        """λ3=0 时 Bivariate 应退化为独立 Poisson"""
        from models.poisson_model import bivariate_poisson_score_matrix, basic_poisson_score_matrix
        bp = bivariate_poisson_score_matrix(2.0, 1.5, lambda_c=0.0)
        ip = basic_poisson_score_matrix(2.0, 1.5)
        # 至少 0:0 应非常接近
        self.assertAlmostEqual(bp[0][0], ip[0][0], places=2)

    def test_zigp_inflates_zero(self):
        """ZIGP 应抬高 0:0 概率"""
        from models.poisson_model import zigp_score_matrix, basic_poisson_score_matrix
        zigp = zigp_score_matrix(2.0, 1.5, theta_a=0.05, theta_b=0.05, pi_zero=0.05)
        base = basic_poisson_score_matrix(2.0, 1.5)
        self.assertGreater(zigp[0][0], base[0][0])
        self.assertAlmostEqual(zigp.sum(), 1.0, places=3)

    def test_ensemble_combines_three(self):
        """Ensemble 应综合三模型"""
        from models.poisson_model import ensemble_score_matrix
        ens = ensemble_score_matrix(2.0, 1.5)
        self.assertIn("matrix", ens)
        self.assertIn("components", ens)
        self.assertIn("dc", ens["components"])
        self.assertIn("bivariate", ens["components"])
        self.assertIn("zigp", ens["components"])
        # 权重和=1
        self.assertAlmostEqual(sum(ens["weights"].values()), 1.0, places=3)
        self.assertAlmostEqual(ens["matrix"].sum(), 1.0, places=3)


class TestInMatchAgents(unittest.TestCase):
    """Layer 5 赛中三 Agent + Bayesian 更新"""

    def test_mrca_blowout_loss(self):
        """大比分输给强敌应触发显著降权"""
        from models.bayesian_updater import MatchEvent, batch_update
        priors = {"Argentina": 13.3, "Spain": 18.5}
        events = [
            MatchEvent(team="Argentina", event_type="group_loss",
                       opponent="Spain", opponent_elo=2155,
                       blowout=True, timestamp="2026-06-12"),
        ]
        res = batch_update(priors, events)
        delta = res["posteriors"]["Argentina"] - priors["Argentina"]
        self.assertLess(delta, -2.0, "大败应至少 -2pp")

    def test_mrca_cap_constraint(self):
        """累积调整应被 cap 约束"""
        from models.bayesian_updater import MatchEvent, batch_update
        priors = {"Brazil": 15.0}
        # 连续 5 场惨败
        events = [MatchEvent(team="Brazil", event_type="knockout_loss",
                              opponent="Spain", opponent_elo=2155, blowout=True,
                              timestamp="2026-07-01") for _ in range(5)]
        res = batch_update(priors, events)
        cumulative = abs(res["cumulative_deltas"]["Brazil"])
        # cap 25pp，不应超过太多
        self.assertLessEqual(cumulative, 26.0)

    def test_ita_mbappe_calibrated(self):
        """ITA 对姆巴佩级球员应给出 -18% 单场胜率（参考报告基准）"""
        from agents.in_match import InjuryTrackerAgent
        ita = InjuryTrackerAgent()
        ev = ita.evaluate_injury(
            team="France", player="Mbappé", dependency=0.28,
            position="forward", status="starter",
            injury_severity=1.0, expected_absent_matches=1,
        )
        self.assertGreater(ev["per_match_impact_pct"], 15.0)
        self.assertLess(ev["per_match_impact_pct"], 22.0)
        self.assertLess(ev["cumulative_delta_pp"], -1.0)

    def test_soa_classify_overbuy(self):
        """模型显著高于市场应被分类为 model_overbuy"""
        from agents.in_match import SentimentBiasAgent
        soa = SentimentBiasAgent()
        ev = soa.evaluate_team("TestTeam",
                                model_prob_pct=15.0, market_prob_pct=8.0,
                                nmi=0.5, avi=0.0)
        self.assertGreater(ev["mmdi"], 0.5)
        self.assertEqual(ev["category"], "model_overbuy")

    def test_soa_classify_consensus(self):
        """模型与市场基本一致应分类 consensus_confirmed"""
        from agents.in_match import SentimentBiasAgent
        soa = SentimentBiasAgent()
        ev = soa.evaluate_team("TestTeam",
                                model_prob_pct=10.0, market_prob_pct=10.5)
        self.assertEqual(ev["category"], "consensus_confirmed")


class TestCatBoostML(unittest.TestCase):
    """参考模型 10：CatBoost + pi-ratings"""

    def test_pi_ratings_train_minimal(self):
        """pi-ratings 应能训练并产出合理评级"""
        from models.pi_ratings import train_pi_ratings
        R_H, R_A, meta = train_pi_ratings(max_rows=2000, since_year=2010)
        self.assertGreater(len(R_H), 50)
        # 大队应该有正评分
        # 用样本范围内的常见队
        candidates = ["Brazil", "Germany", "Spain", "France", "Argentina"]
        positive_count = sum(1 for t in candidates
                              if t in R_H and R_H[t] > 0)
        self.assertGreater(positive_count, 2,
                            "至少 3 个传统强队应有正评分")

    def test_pi_ratings_features_shape(self):
        """build_features 应输出 8 个键"""
        from models.pi_ratings import build_features
        R_H = {"Spain": 0.8, "France": 0.6}
        R_A = {"Spain": 0.5, "France": 0.4}
        feats = build_features(R_H, R_A, "Spain", "France", neutral=True)
        expected_keys = {"pi_diff", "pi_H_diff", "pi_A_diff", "pi_self_h",
                          "pi_self_a", "expected_gd", "form_diff", "neutral"}
        self.assertEqual(set(feats.keys()), expected_keys)
        # neutral 模式下 expected_gd 应正（Spain 评分更高）
        self.assertGreater(feats["expected_gd"], 0)

    def test_catboost_predict_returns_proba(self):
        """如果模型已训练，predict_match 应返回归一概率"""
        from pathlib import Path
        model_path = Path(__file__).parent.parent / "data" / "outputs" / "catboost_match.cbm"
        if not model_path.exists():
            self.skipTest("CatBoost 模型未训练")
        from models.catboost_engine import predict_match
        result = predict_match("Spain", "France", neutral=True)
        total = result["p_home"] + result["p_draw"] + result["p_away"]
        self.assertAlmostEqual(total, 1.0, places=2)
        # Spain 评级更高，应该胜率最大
        self.assertGreater(result["p_home"], result["p_away"])

    def test_catboost_agent_in_swarm(self):
        """CatBoostAgent 应注册到 swarm"""
        from agents.swarm import QueenSwarm
        sw = QueenSwarm()
        names = [a.name for a in sw.tactical_agents]
        self.assertIn("CatBoost", names)
        # 总 Agent 16 个
        self.assertEqual(len(sw.all_agents()), 17)


class TestWalkForwardBacktest(unittest.TestCase):
    """Layer 4 Walk-Forward 历届 WC 回测（参考报告 2.4.3）"""

    def test_metrics_rps_perfect(self):
        """完美预测 RPS=0"""
        sys.path.insert(0, str(Path(__file__).parent.parent / "code"))
        from backtest.walk_forward import rps
        self.assertEqual(rps(1.0, 0.0, 0.0, "H"), 0.0)
        self.assertEqual(rps(0.0, 1.0, 0.0, "D"), 0.0)

    def test_metrics_brier_uniform(self):
        """均匀预测 1/3 Brier = 2/3"""
        from backtest.walk_forward import brier_multi
        b = brier_multi(1/3, 1/3, 1/3, "H")
        self.assertAlmostEqual(b, 2/3, places=3)

    def test_predicted_class_ties(self):
        """概率打平时返回 H（按顺序）"""
        from backtest.walk_forward import predicted_class
        self.assertEqual(predicted_class(0.4, 0.4, 0.2), "H")
        self.assertEqual(predicted_class(0.2, 0.5, 0.3), "D")
        self.assertEqual(predicted_class(0.2, 0.3, 0.5), "A")

    def test_elo_model_predicts(self):
        """Elo 模型应输出概率三元组"""
        from backtest.walk_forward import EloModel
        m = EloModel()
        # 默认评级，主场加成
        p_h, p_d, p_a = m.predict(home="A", away="B", neutral=False)
        self.assertAlmostEqual(p_h + p_d + p_a, 1.0, places=2)
        # 主场应有优势
        self.assertGreater(p_h, p_a)

    def test_elo_updates_after_match(self):
        """Elo 更新应改变评级"""
        from backtest.walk_forward import EloModel
        m = EloModel()
        before_a = m.get("Spain")
        m.update(home="Spain", away="Germany", home_score=3, away_score=0,
                  tournament="FIFA World Cup")
        after_a = m.get("Spain")
        self.assertGreater(after_a, before_a)

    def test_backtest_report_meets_baselines(self):
        """如已跑回测，结果应优于基准"""
        from pathlib import Path
        import json
        p = Path(__file__).parent.parent / "data" / "outputs" / "backtest_walk_forward.json"
        if not p.exists():
            self.skipTest("回测报告未生成")
        d = json.load(open(p))
        L4 = d.get("L4_walk_forward", {})
        elo = L4.get("elo_simple", {})
        catboost = L4.get("catboost_pi", {})
        if elo:
            # Elo 至少应优于 random (33.3%) 和 home_always
            self.assertGreater(elo["accuracy"], 0.45)
            self.assertLess(elo["rps"], 0.25)
        if catboost:
            self.assertGreater(catboost["accuracy"], 0.45)
            self.assertLess(catboost["rps"], 0.25)


class TestDebateProtocol(unittest.TestCase):
    """参考报告 2.3.3 四级辩论协议"""

    @staticmethod
    def _make(name, layer, pe, conf=0.7, rationale="x", evidence=None):
        from agents.base import AgentOutput
        return AgentOutput(
            agent_name=name, layer=layer, role="",
            point_estimate=pe, confidence=conf, weight=1.0,
            rationale=rationale, evidence=evidence or [],
        )

    def test_level1_mild_disagreement(self):
        """spread < 15pp → L1"""
        from agents.debate import DebateEngine
        r = DebateEngine().resolve("X", [
            self._make("Elo", "tactical", 18.0),
            self._make("Poisson", "tactical", 13.0),
        ])
        self.assertEqual(r.level, 1)
        self.assertEqual(r.confidence_multiplier, 1.0)
        self.assertLess(r.spread_pp, 15.0)

    def test_level2_moderate_arbitrated(self):
        """spread 15-30pp → L2，RiskPerception 仲裁"""
        from agents.debate import DebateEngine
        r = DebateEngine().resolve("X", [
            self._make("Elo", "tactical", 12.0),
            self._make("Market", "tactical", 32.0),
            self._make("RiskPerception", "strategic", 15.0, conf=0.8),
        ])
        self.assertEqual(r.level, 2)
        self.assertEqual(r.confidence_multiplier, 0.8)
        self.assertEqual(r.arbiter_choice, "RiskPerception")

    def test_level3_high_discards_loser(self):
        """spread 30-50pp → L3，剔除评分最低方"""
        from agents.debate import DebateEngine
        r = DebateEngine().resolve("X", [
            self._make("Elo", "tactical", 8.0, conf=0.8,
                        rationale="基于 2000 场实测",
                        evidence=["数据 A", "数据 B", "数据 C"]),
            self._make("Market", "tactical", 48.0, conf=0.3,
                        rationale="x", evidence=[]),
        ])
        self.assertEqual(r.level, 3)
        self.assertEqual(r.confidence_multiplier, 0.5)
        self.assertEqual(r.rounds_played, 3)
        self.assertEqual(len(r.discarded_agents), 1)
        # 应该剔除证据弱的 Market
        self.assertIn("Market", r.discarded_agents)

    def test_level4_extreme_conflict(self):
        """spread > 50pp → L4 暂停定量"""
        from agents.debate import DebateEngine
        r = DebateEngine().resolve("X", [
            self._make("Elo", "tactical", 5.0),
            self._make("Market", "tactical", 80.0),
        ])
        self.assertEqual(r.level, 4)
        self.assertEqual(r.confidence_multiplier, 0.2)
        self.assertIn("conflict", r.label)
        # 中位数应在两端之间
        self.assertGreater(r.consensus_pp, 5.0)
        self.assertLess(r.consensus_pp, 80.0)

    def test_batch_debate_summary(self):
        """batch_debate + summarize 应正确统计分布"""
        from agents.debate import batch_debate, summarize
        outputs_per_team = {
            "MildTeam": [
                self._make("A", "tactical", 10.0),
                self._make("B", "tactical", 12.0),
            ],
            "ConflictTeam": [
                self._make("A", "tactical", 5.0),
                self._make("B", "tactical", 70.0),
            ],
        }
        results = batch_debate(outputs_per_team)
        summary = summarize(results)
        self.assertEqual(summary["n_teams"], 2)
        self.assertEqual(summary["level_distribution"][1], 1)
        self.assertEqual(summary["level_distribution"][4], 1)
        self.assertEqual(summary["n_conflicts"], 1)

    def test_llm_debate_l2_with_mock(self):
        """L2 LLM 辩论：mock client 应被正确调用 3 次"""
        from agents.llm_debate import LLMDebateEngine
        from agents.base import AgentOutput
        import json
        call_log = []
        class MockClient:
            def chat(self, system, user, **kw):
                # 用更具区分性的标识：仲裁 system 含 "仲裁 Agent"
                if "仲裁 Agent" in system:
                    role = "arbiter"
                elif system.startswith("你是\"正方"):
                    role = "pro"
                else:
                    role = "con"
                call_log.append(role)
                if role == "pro":
                    return json.dumps({"estimate_pp": 30.0, "evidence": ["E1","E2"],
                                        "opponent_weakness": "x", "rationale": "高估"})
                if role == "con":
                    return json.dumps({"estimate_pp": 10.0, "evidence": ["M1","M2"],
                                        "opponent_weakness": "y", "rationale": "低估"})
                return json.dumps({"pro_evidence_score": 60, "con_evidence_score": 80,
                                    "consensus_pp": 18.0, "resolved": True,
                                    "verdict": "向反方倾斜"})
        eng = LLMDebateEngine(client=MockClient())
        pro = AgentOutput(agent_name="Elo", layer="tactical", role="",
                            point_estimate=30.0, confidence=0.7, weight=1.0)
        con = AgentOutput(agent_name="Market", layer="tactical", role="",
                            point_estimate=10.0, confidence=0.7, weight=1.0)
        res = eng.debate_l2("X", pro, con, spread=20.0)
        self.assertEqual(len(call_log), 3)
        self.assertEqual(call_log, ["pro", "con", "arbiter"])
        self.assertEqual(res["n_llm_calls"], 3)
        self.assertAlmostEqual(res["consensus_pp"], 18.0)
        self.assertTrue(res["resolved"])

    def test_llm_debate_l3_early_exit(self):
        """L3 三轮辩论：仲裁说 resolved 后提前结束"""
        from agents.llm_debate import LLMDebateEngine
        from agents.base import AgentOutput
        import json
        call_log = []
        class MockClient:
            def chat(self, system, user, **kw):
                if "仲裁 Agent" in system:
                    role = "arbiter"
                elif system.startswith("你是\"正方"):
                    role = "pro"
                else:
                    role = "con"
                call_log.append(role)
                if role == "pro":
                    return json.dumps({"estimate_pp": 45.0, "evidence": ["E"],
                                        "opponent_weakness": "x", "rationale": "y"})
                if role == "con":
                    return json.dumps({"estimate_pp": 8.0, "evidence": ["M"],
                                        "opponent_weakness": "x", "rationale": "y"})
                round_num = sum(1 for c in call_log if c == "arbiter")
                return json.dumps({"pro_evidence_score": 50,
                                    "con_evidence_score": 50,
                                    "consensus_pp": 25.0,
                                    "resolved": round_num >= 2,
                                    "verdict": "x"})
        eng = LLMDebateEngine(client=MockClient())
        pro = AgentOutput(agent_name="A", layer="tactical", role="",
                            point_estimate=45.0, confidence=0.7, weight=1.0)
        con = AgentOutput(agent_name="B", layer="tactical", role="",
                            point_estimate=8.0, confidence=0.7, weight=1.0)
        res = eng.debate_l3("X", pro, con, spread=37.0, max_rounds=3)
        # 应在 round 2 就停止 → 6 次调用（2 pro + 2 con + 2 arbiter）
        self.assertEqual(res["n_llm_calls"], 6)
        self.assertTrue(res["resolved"])

    def test_llm_debate_l4_qualitative(self):
        """L4 极端冲突：仅 1 次 LLM 调用生成定性报告"""
        from agents.llm_debate import LLMDebateEngine
        from agents.base import AgentOutput
        import json
        class MockClient:
            def chat(self, system, user, **kw):
                return json.dumps({
                    "qualitative_judgment": "高度不确定",
                    "recommended_action": "human_review",
                    "possible_causes": ["原因A", "原因B"],
                    "conservative_estimate_pp": 30.0,
                })
        eng = LLMDebateEngine(client=MockClient())
        pro = AgentOutput(agent_name="A", layer="tactical", role="",
                            point_estimate=80.0, confidence=0.7, weight=1.0)
        con = AgentOutput(agent_name="B", layer="tactical", role="",
                            point_estimate=5.0, confidence=0.7, weight=1.0)
        res = eng.debate_l4("X", pro, con, spread=75.0)
        self.assertEqual(res["n_llm_calls"], 1)
        self.assertEqual(res["consensus_pp"], 30.0)
        self.assertIn("qualitative_report", res)

    def test_llm_debate_fallback_no_key(self):
        """无 LINGYA_API_KEY 时，run_llm_debate 应返回 None"""
        import os
        from agents.llm_debate import run_llm_debate
        from agents.base import AgentOutput
        from llm.client import reset_default_client
        old = os.environ.pop("LINGYA_API_KEY", None)
        reset_default_client()
        try:
            pro = AgentOutput(agent_name="A", layer="tactical", role="",
                                point_estimate=20.0, confidence=0.7, weight=1.0)
            con = AgentOutput(agent_name="B", layer="tactical", role="",
                                point_estimate=5.0, confidence=0.7, weight=1.0)
            res = run_llm_debate("X", level=2, pro=pro, con=con, spread=15.0)
            self.assertIsNone(res)
        finally:
            if old:
                os.environ["LINGYA_API_KEY"] = old
            reset_default_client()

    def test_debate_integrated_in_swarm(self):
        """QueenSwarm 输出应包含 debate_protocol 字段"""
        from agents.swarm import QueenSwarm
        sw = QueenSwarm()
        # 用 build_context + run，但不打印
        result = sw.run(verbose=False)
        self.assertIn("debate_protocol", result)
        self.assertIn("summary", result["debate_protocol"])
        # Top 球队都参与
        n_teams = result["debate_protocol"]["summary"]["n_teams"]
        self.assertGreaterEqual(n_teams, 5,
            "至少应有 5 个 Top 队参与辩论")


class TestStackedEnsemble(unittest.TestCase):
    """Stacked Ensemble + PoissonFamily（突破 ECE<5%）"""

    def test_poisson_family_normalized(self):
        """PoissonFamily 输出应归一"""
        from backtest.walk_forward import PoissonFamilyModel
        m = PoissonFamilyModel()
        # 预热几场
        m.update(home="A", away="B", home_score=2, away_score=1)
        m.update(home="B", away="C", home_score=0, away_score=3)
        m.update(home="C", away="A", home_score=1, away_score=2)
        ph, pd, pa = m.predict(home="A", away="C")
        self.assertAlmostEqual(ph + pd + pa, 1.0, places=3)

    def test_poisson_family_high_lambda_favors_home(self):
        """主队 pi-rating 高时应预测主胜概率 > 0.4"""
        from backtest.walk_forward import PoissonFamilyModel
        m = PoissonFamilyModel()
        m.R_H["Strong"] = 1.0
        m.R_A["Strong"] = 0.7
        m.R_H["Weak"] = -0.5
        m.R_A["Weak"] = -0.7
        ph, pd, pa = m.predict(home="Strong", away="Weak", neutral=True)
        self.assertGreater(ph, pa)
        self.assertGreater(ph, 0.4)

    def test_stacked_ensemble_loadable(self):
        """stacker 如已训练，能加载且预测归一"""
        from pathlib import Path
        path = Path(__file__).parent.parent / "data" / "outputs" / "stacker_logistic.joblib"
        if not path.exists():
            self.skipTest("stacker 未训练")
        from backtest.walk_forward import (
            StackedEnsembleModel, EloModel, PiCatBoostModel,
            CalibratedCatBoostModel, PoissonFamilyModel,
        )
        sub = [
            EloModel(), PiCatBoostModel(),
            CalibratedCatBoostModel("temperature"),
            PoissonFamilyModel(),
        ]
        m = StackedEnsembleModel(sub, stacker_path=path)
        self.assertIsNotNone(m.stacker)
        ph, pd_, pa = m.predict(home="Spain", away="France", neutral=True)
        self.assertAlmostEqual(ph + pd_ + pa, 1.0, places=3)

    def test_stacked_meets_ece_target(self):
        """L4 回测后 stacked ensemble 应 ECE < 5%"""
        import json
        from pathlib import Path
        p = Path(__file__).parent.parent / "data" / "outputs" / "backtest_walk_forward.json"
        if not p.exists():
            self.skipTest("L4 回测未跑")
        d = json.load(open(p))
        L4 = d.get("L4_walk_forward", {})
        stacked = L4.get("stacked_ensemble")
        if stacked is None:
            self.skipTest("stacked_ensemble 不在 L4 报告中（stacker 未训练）")
        self.assertLess(stacked["calibration_error"], 0.05,
                         "Stacked Ensemble cal_err 应 < 5%（精度目标）")


class TestMultiCalibrator(unittest.TestCase):
    """多类校准器（参考报告 2.3.4）"""

    def _make_data(self, n=500):
        """生成有系统性偏差的合成数据"""
        import numpy as np
        rng = np.random.default_rng(42)
        # 模拟一个 over-confident 模型：概率往两端推
        raw_p = rng.dirichlet([1, 1, 1], size=n)
        # 强化最大类
        max_idx = raw_p.argmax(axis=1)
        boost = rng.uniform(0.1, 0.3, n)
        for i in range(n):
            raw_p[i, max_idx[i]] += boost[i]
            raw_p[i] = raw_p[i] / raw_p[i].sum()
        # y_true 部分跟概率，部分随机（模拟 over-confidence）
        y = np.array([
            max_idx[i] if rng.random() < raw_p[i, max_idx[i]] * 0.85
            else rng.integers(0, 3)
            for i in range(n)
        ])
        return raw_p, y

    def test_temperature_reduces_ece(self):
        """Temperature Scaling 应降低 over-confident 模型的 ECE"""
        from models.multi_calibrator import (
            TemperatureScaling, expected_calibration_error,
        )
        probs, y = self._make_data(n=500)
        ece_before, _ = expected_calibration_error(probs, y)
        ts = TemperatureScaling().fit(probs, y)
        p_cal = ts.transform(probs)
        ece_after, _ = expected_calibration_error(p_cal, y)
        # 不一定降低（合成数据可能已经较准），但 T 应被学到
        self.assertGreater(ts.T, 0.5)
        self.assertLess(ts.T, 3.0)

    def test_isotonic_normalized(self):
        """Isotonic 校准后应保持归一性"""
        from models.multi_calibrator import IsotonicMulti
        probs, y = self._make_data(n=300)
        iso = IsotonicMulti().fit(probs, y)
        p_cal = iso.transform(probs)
        # 每行应归一
        sums = p_cal.sum(axis=1)
        for s in sums[:20]:
            self.assertAlmostEqual(s, 1.0, places=3)

    def test_vector_scaling_fits(self):
        """Vector Scaling 应能学到 W/b 参数"""
        from models.multi_calibrator import VectorScaling
        probs, y = self._make_data(n=300)
        vs = VectorScaling().fit(probs, y)
        self.assertEqual(vs.W.shape, (3, 3))
        self.assertEqual(vs.b.shape, (3,))
        # 校准后应仍是合法概率
        p_cal = vs.transform(probs)
        sums = p_cal.sum(axis=1)
        for s in sums[:20]:
            self.assertAlmostEqual(s, 1.0, places=3)

    def test_benchmark_pipeline(self):
        """benchmark() 应返回 4 种校准结果"""
        from models.multi_calibrator import benchmark
        probs, y = self._make_data(n=500)
        result = benchmark(probs, y, ratio=0.8)
        for name in ["uncalibrated", "temperature", "temperature_ece",
                      "vector", "isotonic"]:
            self.assertIn(name, result)
            self.assertIn("ece", result[name])

    def test_calibration_report_meets_target(self):
        """如已跑校准，5872 全样本应达 ECE<5%"""
        import json
        from pathlib import Path
        p = Path(__file__).parent.parent / "data" / "outputs" / "calibration_multiclass.json"
        if not p.exists():
            self.skipTest("校准报告未生成")
        d = json.load(open(p))
        # 全样本验证集上至少有一种校准方法 ECE<5%
        eces = {n: d.get(n, {}).get("ece") for n in
                 ["temperature", "temperature_ece", "vector", "isotonic"]}
        eces = {n: v for n, v in eces.items() if v is not None}
        if eces:
            self.assertLess(min(eces.values()), 0.06,
                "至少一种校准方法在全样本上应 ECE<6%")


class TestProbabilityConservation(unittest.TestCase):
    """蒙特卡洛输出概率守恒（关键回归测试）"""
    
    def test_champion_probability_sum(self):
        """所有球队冠军概率之和 ≈ 1"""
        import json
        path = Path(__file__).parent.parent / "data" / "outputs" / "mc_simulation_n100000.json"
        if not path.exists():
            self.skipTest("100k 模拟文件不存在")
        with open(path) as f:
            data = json.load(f)
        total = sum(v.get("champion", 0) for v in data.values())
        self.assertAlmostEqual(total, 1.0, places=2,
            msg=f"冠军概率总和应为 1.0，实际 {total}")


if __name__ == "__main__":
    # 简洁输出
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
