"""
LLMAgent 基类
=============
将 参考报告 2.3.2 节描述的"三段式 Prompt"工程化：
  1. 角色定义与能力边界（system_prompt）
  2. 输出格式规范（JSON Schema 在 system 中声明）
  3. 合规约束（每个角色额外的硬约束）

调用流程：
    analyze(context)
        -> build_user_prompt(context)
        -> LLMClient.chat(system, user, json_mode=True)
        -> extract_json
        -> parse_response(data, context) -> AgentOutput

任何环节失败都会自动 fallback 到子类提供的 fallback_analyze()。
"""
from __future__ import annotations
import sys
import json
import logging
from pathlib import Path
from abc import abstractmethod
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.base import Agent, AgentOutput
from llm import get_default_client, LLMUnavailable
from llm.client import LLMClient

logger = logging.getLogger("worldcup.llm_agent")


class LLMAgent(Agent):
    """
    LLM 驱动 Agent 抽象基类。

    子类需实现：
        SYSTEM_PROMPT (类属性)        — 角色定义+合规约束
        OUTPUT_SCHEMA (类属性, dict)  — 期望的 JSON 字段结构（用于 prompt 注入）
        build_user_prompt(context)    — 把 context 序列化成 user message
        parse_response(data, context) — 把 LLM JSON 转成 AgentOutput
        fallback_analyze(context)     — LLM 不可用时的规则降级
    """

    SYSTEM_PROMPT: str = ""        # 子类覆盖
    OUTPUT_SCHEMA: Dict[str, Any] = {}  # 子类覆盖

    def __init__(self, name: str, layer: str = "tactical", weight: float = 1.0,
                 client: Optional[LLMClient] = None):
        super().__init__(name, layer, weight)
        self._client_override = client  # 测试注入

    # ============ 强制子类实现 ============
    @abstractmethod
    def build_user_prompt(self, context: Dict[str, Any]) -> str: ...

    @abstractmethod
    def parse_response(self, data: Dict[str, Any], context: Dict[str, Any]) -> AgentOutput: ...

    @abstractmethod
    def fallback_analyze(self, context: Dict[str, Any]) -> AgentOutput: ...

    # ============ 可被 Swarm 调度的入口 ============
    def analyze(self, context: Dict[str, Any]) -> AgentOutput:
        """
        三段式：调 LLM → 解析 → 转 AgentOutput。
        任意环节出错都会落到 fallback。
        """
        try:
            client = self._client_override or get_default_client()
        except LLMUnavailable as e:
            logger.warning(f"[{self.name}] LLM 不可用 → fallback: {e}")
            out = self.fallback_analyze(context)
            out.evidence = [f"[fallback:无LLM] {e}"] + (out.evidence or [])
            return out

        try:
            user = self.build_user_prompt(context)
            system = self._compose_system()
            text = client.chat(system=system, user=user, json_mode=True,
                               temperature=0.3, max_tokens=8000)
            data = LLMClient.extract_json(text)
            out = self.parse_response(data, context)
            # 标记是 LLM 输出
            out.evidence = (out.evidence or []) + ["[source:LLM]"]
            # Layer 1: 数据可用性降权
            dq = context.get("data_quality") or {}
            dq_weight = float(dq.get("overall_weight", 1.0))
            if dq_weight < 1.0:
                old_conf = out.confidence
                out.confidence = round(old_conf * dq_weight, 3)
                out.evidence.append(
                    f"[data_quality:{dq_weight}] conf {old_conf:.2f} → {out.confidence:.2f}"
                )
            return out
        except LLMUnavailable as e:
            logger.warning(f"[{self.name}] LLM 调用失败 → fallback: {e}")
            out = self.fallback_analyze(context)
            out.evidence = [f"[fallback:LLM失败] {e}"] + (out.evidence or [])
            return out
        except Exception as e:
            logger.exception(f"[{self.name}] 解析失败 → fallback")
            out = self.fallback_analyze(context)
            out.evidence = [f"[fallback:解析失败] {type(e).__name__}: {e}"] + (out.evidence or [])
            return out

    # ============ 内部：拼 system prompt ============
    def _compose_system(self) -> str:
        """把 SYSTEM_PROMPT + OUTPUT_SCHEMA + 通用合规约束拼起来"""
        schema_part = ""
        if self.OUTPUT_SCHEMA:
            schema_part = (
                "\n\n## 输出 JSON 结构\n"
                f"{json.dumps(self.OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}"
            )
        compliance = (
            "\n\n## 通用合规约束\n"
            "- 概率值用百分点 (pp) 表示；调整量为相对当前基准的偏移。\n"
            "- 禁用确定性表述：'必中''稳胆''铁定''绝对'。\n"
            "- 必须给出 confidence (0-1) 和 rationale。\n"
            "- 数据缺失时输出 reason='insufficient_data'，不要编造。"
        )
        return self.SYSTEM_PROMPT + schema_part + compliance
