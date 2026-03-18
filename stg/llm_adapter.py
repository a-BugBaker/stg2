"""
LLM 适配器模块

本模块封装了与 OpenAI 兼容 API 的交互逻辑，使 STG 系统能将结构化证据发送给任意
支持 OpenAI chat completions 接口的大语言模型（如 GPT-4、Qwen、DeepSeek 等）。

核心类：
    OpenAICompatibleLLMAdapter:
        - 初始化参数：api_base（API 端点）、api_key、model（模型名）、temperature
        - answer(prompts, evidence): 将 grounded prompt 发送给 LLM，解析并返回结构化答案
        - 如果 evidence 中既没有事件也没有实体，直接返回 insufficient evidence 结果
        - 自动解析 LLM 返回的 JSON 响应；若解析失败则保留原始文本

异常：
    LLMAdapterError: API 配置缺失或 openai 包未安装时抛出

注意：openai 包是可选依赖，未安装时系统仍可正常构建和检索，只是无法调用 LLM。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from .utils import extract_json_object


class LLMAdapterError(RuntimeError):
    pass


@dataclass
class OpenAICompatibleLLMAdapter:
    api_base: str
    api_key: str
    model: str
    temperature: float = 0.1

    def _client(self) -> Any:
        """创建 OpenAI 兼容客户端并校验必要配置。"""
        # 依赖检查：未安装 openai 时给出明确错误提示。
        if OpenAI is None:
            raise LLMAdapterError(
                "openai package is not available. Install it or run the script in an environment with OpenAI support."
            )
        # 配置检查：必须同时提供 base 与 key。
        if not self.api_base or not self.api_key:
            raise LLMAdapterError(
                "Missing API configuration. Provide both --api_base and --api_key, or use --dry_run to inspect the grounded prompt."
            )
        return OpenAI(base_url=self.api_base, api_key=self.api_key)

    def answer(self, prompts: Dict[str, str], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """调用 LLM 生成 grounded 答案；证据不足时直接短路返回。"""
        # 若检索不到事件与实体证据，直接返回证据不足，避免无根据生成。
        if not evidence.get("events") and not evidence.get("entities"):
            return {
                "answer": "Insufficient evidence to answer the question.",
                "sufficient_evidence": False,
                "used_event_ids": [],
                "used_entity_ids": [],
                "short_rationale": "No event or entity evidence was retrieved above threshold.",
            }

        # 组装标准 chat completion 请求（system + user）。
        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": prompts["system_prompt"]},
                {"role": "user", "content": prompts["user_prompt"]},
            ],
        )
        # 解析模型文本，标准化为约定 JSON 结构。
        content = response.choices[0].message.content or ""
        parsed = self._parse_response(content)
        return parsed

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """解析模型输出 JSON；失败时回退为保留原文的兜底结构。"""
        try:
            payload = extract_json_object(content)
        except Exception:
            # 非 JSON 输出兜底：保留原始文本并标记证据不足。
            payload = {
                "answer": content.strip(),
                "sufficient_evidence": False,
                "used_event_ids": [],
                "used_entity_ids": [],
                "short_rationale": "Model did not return valid JSON; raw text was preserved.",
            }

        # 字段补齐与类型规范化，保证上游调用稳定。
        payload.setdefault("answer", "")
        payload["sufficient_evidence"] = bool(payload.get("sufficient_evidence", False))
        payload["used_event_ids"] = [str(item) for item in payload.get("used_event_ids", [])]
        payload["used_entity_ids"] = [str(item) for item in payload.get("used_entity_ids", [])]
        payload.setdefault("short_rationale", "")
        return payload
