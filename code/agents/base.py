"""
Agent 基类 + Hierarchical 三层架构核心

参考 参考方法论第 2.3 章：
- 战略层（Strategic Layer）：宏观判断、质量控制
- 战术层（Tactical Layer）：垂直领域分析
- 执行层（Execution Layer）：微观因子量化

Queen-led Swarm 协调机制：
- 战略层 = Queen 节点（3 倍权重）
- 战术 + 执行 = Worker 节点
- 共识算法：Byzantine 容错（2/3 多数）+ 加权共识
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import time


# ============ Agent 输出数据结构 ============
@dataclass
class AgentOutput:
    """单个 Agent 的标准化输出"""
    agent_name: str
    layer: str  # strategic / tactical / execution
    role: str   # 角色描述
    
    # 核心输出（任选其一）
    point_estimate: Optional[float] = None  # 点估计（如某队夺冠概率）
    probability_dist: Optional[Dict[str, float]] = None  # 球队 → 概率
    
    # 元数据
    confidence: float = 0.5      # 自信度 0-1
    weight: float = 1.0          # 权重（Queen 节点 3.0）
    rationale: str = ""          # 推理理由
    evidence: List[str] = field(default_factory=list)  # 证据列表
    contributions: Dict[str, float] = field(default_factory=dict)  # 各因子贡献
    
    # 不确定性
    uncertainty_pp: float = 0.0  # 不确定性（pp）
    
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


# ============ Agent 基类 ============
class Agent(ABC):
    """所有 Agent 的抽象基类"""
    
    def __init__(self, name: str, layer: str = "tactical", weight: float = 1.0):
        self.name = name
        self.layer = layer  # strategic / tactical / execution
        self.weight = weight
        self.role = self.__class__.__doc__.strip().split("\n")[0] if self.__class__.__doc__ else ""
    
    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> AgentOutput:
        """
        执行分析，返回 AgentOutput
        
        Args:
            context: 共享上下文（teams_data, mc_results, market, etc.）
        """
        pass


# ============ 共识聚合机制 ============
class ConsensusAggregator:
    """
    Byzantine 容错（2/3 多数）+ 加权共识
    
    参考方法论：
    - 战略层 = Queen 节点 = 3.0 倍权重
    - 战术层 + 执行层 = Worker = 1.0 倍权重
    """
    
    LAYER_WEIGHTS = {
        "strategic": 3.0,
        "tactical": 1.0,
        "execution": 1.0,
    }
    
    @classmethod
    def aggregate_point_estimates(cls, outputs: List[AgentOutput], 
                                   team: Optional[str] = None) -> Dict[str, Any]:
        """
        加权聚合点估计 + Byzantine 容错
        
        Returns:
            {
                "consensus": 加权均值,
                "std": 标准差,
                "min": 最低估计,
                "max": 最高估计,
                "n_agents": Agent 数,
                "confidence": 共识强度（1 - 标准差/均值）,
                "agent_breakdown": [...]
            }
        """
        # 提取所有有效估计
        estimates = []
        for out in outputs:
            value = None
            if out.point_estimate is not None:
                value = out.point_estimate
            elif team and out.probability_dist:
                value = out.probability_dist.get(team)
            
            if value is not None:
                # 计算 Layer 权重 × Agent 自身权重
                layer_w = cls.LAYER_WEIGHTS.get(out.layer, 1.0)
                effective_weight = layer_w * out.weight * out.confidence
                estimates.append({
                    "agent": out.agent_name,
                    "layer": out.layer,
                    "value": value,
                    "weight": effective_weight,
                    "rationale": out.rationale,
                })
        
        if not estimates:
            return {"consensus": None, "n_agents": 0}
        
        # 加权平均
        total_w = sum(e["weight"] for e in estimates)
        consensus = sum(e["value"] * e["weight"] for e in estimates) / total_w
        
        # 标准差
        values = [e["value"] for e in estimates]
        std = (sum((v - consensus)**2 for v in values) / len(values)) ** 0.5
        
        # Byzantine 容错检测：是否 ≥2/3 Agent 在 ±1 std 内一致？
        in_range = sum(1 for v in values if abs(v - consensus) <= std)
        byzantine_pass = (in_range / len(values)) >= 0.667
        
        # 共识强度
        confidence = 1.0 - min(1.0, std / max(1.0, abs(consensus)))
        
        return {
            "consensus": round(consensus, 3),
            "std": round(std, 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "n_agents": len(estimates),
            "byzantine_pass": byzantine_pass,
            "agreement_pct": round(in_range / len(values) * 100, 1),
            "confidence": round(confidence, 3),
            "agent_breakdown": estimates,
        }
    
    @classmethod
    def aggregate_distributions(cls, outputs: List[AgentOutput]) -> Dict[str, Dict[str, Any]]:
        """对多个球队的概率分布做加权聚合"""
        # 收集所有涉及的球队
        all_teams = set()
        for out in outputs:
            if out.probability_dist:
                all_teams.update(out.probability_dist.keys())
        
        # 对每个球队聚合
        result = {}
        for team in all_teams:
            agg = cls.aggregate_point_estimates(outputs, team=team)
            if agg.get("n_agents", 0) > 0:
                result[team] = agg
        
        return result


# ============ Critic 评分系统 ============
class Critic:
    """
    对抗性评分（参考报告第 2.3.4 节）
    
    评分维度：
    1. 数据质量（来源是否可信、时效性）
    2. 推理一致性（contributions 之和是否合理）
    3. 自信度校准（rationale 是否充分）
    4. 与共识偏差度（是否离群）
    """
    
    @classmethod
    def score(cls, output: AgentOutput, consensus: Optional[float] = None) -> Dict[str, Any]:
        scores = {}
        
        # 1. 推理充分性（rationale 长度 + evidence 数量）
        evidence_score = min(1.0, len(output.evidence) / 3.0)  # ≥3 条证据满分
        rationale_score = min(1.0, len(output.rationale) / 100.0)  # ≥100 字满分
        scores["evidence"] = round(evidence_score * 0.6 + rationale_score * 0.4, 2)
        
        # 2. 自信度校准（不应过度自信）
        scores["calibration"] = round(min(output.confidence, 0.85), 2)  # 永远不到 1.0
        
        # 3. 离群度（如有 consensus）
        if consensus is not None and output.point_estimate is not None:
            deviation = abs(output.point_estimate - consensus)
            scores["agreement"] = round(max(0, 1.0 - deviation / max(1.0, consensus)), 2)
        else:
            scores["agreement"] = 1.0
        
        # 综合分（0-100）
        total = (scores["evidence"] * 0.3 + 
                 scores["calibration"] * 0.3 + 
                 scores["agreement"] * 0.4) * 100
        scores["total"] = round(total, 1)
        
        # 评级
        if total >= 80:
            grade = "A"
        elif total >= 60:
            grade = "B"
        elif total >= 40:
            grade = "C"
        else:
            grade = "D"
        scores["grade"] = grade
        
        return scores
