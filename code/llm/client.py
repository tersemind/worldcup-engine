"""
Lingya LLM 客户端（OpenAI 兼容协议）

环境变量：
    LINGYA_API_KEY                必填
    LINGYA_BASE_URL               可选，默认 https://api.lingyaai.cn/v1
    WORLDCUP_LLM_MODEL            可选，默认 deepseek-v4-flash（主模型，便宜快）
    WORLDCUP_LLM_FALLBACK_MODEL   可选，默认 deepseek-v4-flash-max（备用，稳定但贵）
    WORLDCUP_LLM_FALLBACK_AFTER   可选，默认 3（主模型连续失败 N 次后进程级切换到备用）
    WORLDCUP_LLM_CACHE            可选，默认 1（开启磁盘缓存）

降级机制（两级）：
    1. 单次失败：主模型 chat 抛错 → 立刻用备用模型再试一次（同一次调用内）
    2. 连续失败：主模型连续失败超过 FALLBACK_AFTER 次 → 进程后续直接用备用模型

使用：
    from llm import get_default_client, LLMUnavailable
    try:
        client = get_default_client()
        text = client.chat(system="你是XX", user="分析YY", json_mode=True)
    except LLMUnavailable:
        # 降级到规则
        ...
"""
from __future__ import annotations
import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("worldcup.llm")


class LLMUnavailable(Exception):
    """LLM 不可用（缺 key、网络故障、超时等）—调用方应降级"""


# ============ 缓存目录 ============
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "outputs" / "llm_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class LLMClient:
    """轻量包装：OpenAI Python SDK + 磁盘缓存 + JSON 抽取"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        use_cache: bool = True,
    ):
        api_key = api_key or os.getenv("LINGYA_API_KEY")
        if not api_key:
            raise LLMUnavailable("LINGYA_API_KEY 未设置")

        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMUnavailable(f"openai 包未安装: {e}")

        self.base_url = base_url or os.getenv("LINGYA_BASE_URL", "https://api.lingyaai.cn/v1")
        self.model = model or os.getenv("WORLDCUP_LLM_MODEL", "deepseek-v4-flash")
        self.fallback_model = os.getenv("WORLDCUP_LLM_FALLBACK_MODEL", "deepseek-v4-flash-max")
        self.fallback_after = int(os.getenv("WORLDCUP_LLM_FALLBACK_AFTER", "3"))
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_cache = use_cache

        # 降级状态
        self._consecutive_failures = 0  # 主模型连续失败计数
        self._using_fallback = False    # 是否已进程级切换到备用模型

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    # ============ 缓存 key ============
    @staticmethod
    def _cache_key(model: str, system: str, user: str, json_mode: bool, temperature: float) -> str:
        h = hashlib.sha256()
        payload = f"{model}\x00{system}\x00{user}\x00{json_mode}\x00{temperature:.3f}"
        h.update(payload.encode("utf-8"))
        return h.hexdigest()[:24]

    def _cache_get(self, key: str) -> Optional[str]:
        if not self.use_cache:
            return None
        f = _CACHE_DIR / f"{key}.json"
        if f.exists():
            try:
                return json.loads(f.read_text())["content"]
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, content: str, meta: Dict[str, Any]) -> None:
        if not self.use_cache:
            return
        f = _CACHE_DIR / f"{key}.json"
        f.write_text(json.dumps({"content": content, "meta": meta, "ts": time.time()},
                                 ensure_ascii=False, indent=2))

    # ============ 模型选择 ============
    @property
    def active_model(self) -> str:
        """当前生效模型（已切换到备用则返回 fallback_model）"""
        return self.fallback_model if self._using_fallback else self.model

    def _try_one_model(self, model: str, system: str, user: str,
                        temperature: float, max_tokens: int):
        """单次尝试某模型；成功返回 (content, resp)，失败抛 Exception"""
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or ""
        return content, resp

    # ============ 主入口 ============
    def chat(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 8000,  # reasoning 模型（如 deepseek-v4-flash）需要充足额度
    ) -> str:
        """
        发送一次对话，返回模型文本。
        json_mode=True 时强制要求 JSON 输出（通过 system 指令 + 后处理抽取）。

        降级路径：
            1. 用 active_model（主或已切换的备用）查缓存→直接返回
            2. 调用 active_model；成功 → 重置失败计数
            3. 调用失败：
               a. 若 active 是主模型：立即用备用模型重试一次
                  - 备用成功 → 累计主模型失败；若达阈值则进程级切换
                  - 备用也失败 → raise
               b. 若 active 已是备用模型：直接 raise
        """
        if json_mode:
            system = system.rstrip() + (
                "\n\n严格要求：输出必须是单个合法 JSON 对象，"
                "不要包含 markdown 代码块标记（```），不要在 JSON 之外添加任何解释文字。"
            )

        active = self.active_model
        cache_key = self._cache_key(active, system, user, json_mode, temperature)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # 1. 试 active model
        try:
            content, resp = self._try_one_model(active, system, user, temperature, max_tokens)
            # 主模型成功 → 重置失败计数
            if active == self.model:
                self._consecutive_failures = 0
            self._cache_put(cache_key, content, {
                "model": active,
                "json_mode": json_mode,
                "temperature": temperature,
                "usage": getattr(resp, "usage", None) and resp.usage.model_dump(),
            })
            return content
        except Exception as e:
            primary_err = f"{type(e).__name__}: {e}"

        # 2. 主模型失败：试备用模型一次（仅当当前 active 是主模型时）
        if active == self.model and self.fallback_model and self.fallback_model != self.model:
            self._consecutive_failures += 1
            logger.warning(
                f"主模型 {self.model} 失败 ({self._consecutive_failures}/{self.fallback_after}) "
                f"→ 单次降级到 {self.fallback_model}: {primary_err}"
            )
            fb_cache_key = self._cache_key(self.fallback_model, system, user, json_mode, temperature)
            fb_cached = self._cache_get(fb_cache_key)
            if fb_cached is not None:
                self._maybe_switch_to_fallback()
                return fb_cached
            try:
                content, resp = self._try_one_model(
                    self.fallback_model, system, user, temperature, max_tokens
                )
                self._cache_put(fb_cache_key, content, {
                    "model": self.fallback_model,
                    "json_mode": json_mode,
                    "temperature": temperature,
                    "usage": getattr(resp, "usage", None) and resp.usage.model_dump(),
                    "via": "single_call_fallback",
                })
                self._maybe_switch_to_fallback()
                return content
            except Exception as e2:
                raise LLMUnavailable(
                    f"LLM 主+备用都失败: 主={primary_err}; "
                    f"备={type(e2).__name__}: {e2}"
                )

        # 3. 当前已经是备用模型，或没有备用 → 直接失败
        raise LLMUnavailable(f"LLM 调用失败: {primary_err}")

    def _maybe_switch_to_fallback(self) -> None:
        """主模型连续失败达阈值 → 进程级切换到备用模型"""
        if (
            not self._using_fallback
            and self._consecutive_failures >= self.fallback_after
        ):
            logger.warning(
                f"主模型 {self.model} 连续失败 {self._consecutive_failures} 次 "
                f"≥ 阈值 {self.fallback_after}，进程级切换到备用模型 {self.fallback_model}"
            )
            self._using_fallback = True

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """
        从 LLM 输出里抽 JSON。容错三种格式：
        1. 纯 JSON
        2. ```json ... ``` 包裹
        3. 嵌在解释文字里（取第一个 { ... } 配对）
        """
        text = text.strip()

        # 1. 纯 JSON
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except Exception:
                pass

        # 2. 代码块
        if "```" in text:
            import re
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass

        # 3. 第一个 { ... } 平衡括号匹配
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except Exception:
                            break

        raise LLMUnavailable(f"无法从 LLM 输出解析 JSON: {text[:200]!r}")


# ============ 全局单例（懒加载）============
_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    """返回进程级单例。无 key 时抛 LLMUnavailable。"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def reset_default_client() -> None:
    """供测试重置"""
    global _default_client
    _default_client = None
