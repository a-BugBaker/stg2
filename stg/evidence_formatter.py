"""
证据格式化模块（Evidence Formatter）

本模块负责将检索和重排后的事件/实体证据，格式化为不同消费者所需的格式。

三个核心功能：
    1. build_bundle():          将 QueryParseResult + 事件列表 + 实体列表 + registry
                                组装成完整的 evidence bundle（含 evidence_text、evidence_json 等）
    2. format_evidence_for_llm(): 从 bundle 中提取 LLM 所需的精简 JSON 证据
                                （限制 max_events / max_entities，只保留关键字段）
    3. build_grounded_prompt(): 生成 system_prompt + user_prompt，要求 LLM：
                                - 仅基于提供的证据作答
                                - 返回标准 JSON 格式（含 answer、sufficient_evidence、
                                  used_event_ids、used_entity_ids、short_rationale）
                                - 证据不足时允许返回 sufficient_evidence=false

evidence_text 是面向人类可读的文本摘要，适合调试和日志。
evidence_json 是面向程序消费的结构化数据。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from .query_parser import QueryParseResult


class EvidenceFormatter:
    def build_bundle(
        self,
        query_info: QueryParseResult,
        events: Sequence[Dict[str, Any]],
        entities: Sequence[Dict[str, Any]],
        registry: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """组装完整 evidence bundle（文本版 + JSON 版）。"""
        # 先汇总统计信息，便于上层快速查看检索覆盖情况。
        summary_stats = {
            "num_event_evidence": len(events),
            "num_entity_evidence": len(entities),
            "num_tracked_entities": len(registry),
            "matched_entity_ids": list(query_info.entity_ids),
            "matched_entity_tags": list(query_info.entity_tags),
            "matched_entity_labels": list(query_info.entity_labels),
        }
        # 构建程序消费的结构化证据主对象。
        evidence_json = {
            "query": query_info.query,
            "normalized_query": query_info.normalized_query,
            "subqueries": list(query_info.subqueries),
            "query_hints": query_info.to_dict(),
            "events": list(events),
            "entities": list(entities),
            "summary_stats": summary_stats,
        }
        # 再生成可读文本版本，便于日志和调试展示。
        evidence_text = self.format_evidence_text(evidence_json)
        bundle = dict(evidence_json)
        bundle["evidence_text"] = evidence_text
        bundle["evidence_json"] = evidence_json
        return bundle

    def format_evidence_text(self, bundle: Dict[str, Any]) -> str:
        """将证据结构渲染为便于阅读的文本摘要。"""
        # 1) 输出查询与解析线索头部信息。
        lines: List[str] = [
            "=== STG Evidence Bundle ===",
            f"Query: {bundle['query']}",
            f"Normalized Query: {bundle['normalized_query']}",
            f"Subqueries: {', '.join(bundle.get('subqueries', [])) or '(none)'}",
        ]

        hints = bundle.get("query_hints", {})
        lines.append(
            "Hints: "
            f"entity_tags={hints.get('entity_tags', [])}, "
            f"entity_labels={hints.get('entity_labels', [])}, "
            f"relation_keywords={hints.get('relation_keywords', [])}, "
            f"temporal_keywords={hints.get('temporal_keywords', [])}, "
            f"intents={hints.get('query_intents', [])}"
        )
        lines.append("--- Event Evidence ---")
        events = bundle.get("events", [])
        if not events:
            lines.append("(no event evidence above threshold)")
        else:
            # 2) 逐条渲染事件证据（排名、分数、帧范围、摘要）。
            for rank, item in enumerate(events, start=1):
                lines.append(
                    f"{rank}. [{item.get('event_type', 'event')}] "
                    f"id={item.get('memory_id')} score={item.get('final_score', item.get('score', 0.0)):.3f} "
                    f"frames={item.get('frame_start')}-{item.get('frame_end')} "
                    f"entities={item.get('entity_tags', item.get('entities', []))} "
                    f"| {item.get('summary', '')}"
                )
        lines.append("--- Entity Evidence ---")
        entities = bundle.get("entities", [])
        if not entities:
            lines.append("(no entity evidence above threshold)")
        else:
            # 3) 逐条渲染实体状态证据。
            for rank, item in enumerate(entities, start=1):
                lines.append(
                    f"{rank}. [{item.get('entity_id', 'entity')}] "
                    f"id={item.get('memory_id')} score={item.get('final_score', item.get('score', 0.0)):.3f} "
                    f"frame={item.get('frame_index')} "
                    f"bbox={item.get('bbox')} "
                    f"| {item.get('description', '')}"
                )
        stats = bundle.get("summary_stats", {})
        lines.append("--- Summary Stats ---")
        # 4) 末尾附加机器可读统计 JSON。
        lines.append(json.dumps(stats, ensure_ascii=False, sort_keys=True))
        return "\n".join(lines)

    def format_evidence_for_llm(
        self,
        bundle: Dict[str, Any],
        *,
        max_events: int = 8,
        max_entities: int = 4,
    ) -> Dict[str, Any]:
        """裁剪证据字段与数量，生成 LLM 输入证据。"""
        # 事件证据：保留对回答最关键的字段，限制条数控制提示词长度。
        events = []
        for item in bundle.get("events", [])[:max_events]:
            events.append(
                {
                    "memory_id": item.get("memory_id"),
                    "memory_type": item.get("memory_type"),
                    "event_type": item.get("event_type"),
                    "frame_start": item.get("frame_start"),
                    "frame_end": item.get("frame_end"),
                    "entities": item.get("entities", []),
                    "entity_tags": item.get("entity_tags", []),
                    "entity_labels": item.get("entity_labels", []),
                    "summary": item.get("summary", ""),
                    "confidence": item.get("confidence"),
                    "source": item.get("source"),
                    "score": item.get("final_score", item.get("score")),
                }
            )
        # 实体证据：保留状态和时空属性相关字段。
        entities = []
        for item in bundle.get("entities", [])[:max_entities]:
            entities.append(
                {
                    "memory_id": item.get("memory_id"),
                    "memory_type": item.get("memory_type"),
                    "entity_id": item.get("entity_id"),
                    "tag": item.get("tag"),
                    "label": item.get("label"),
                    "frame_index": item.get("frame_index"),
                    "frame_start": item.get("frame_start"),
                    "frame_end": item.get("frame_end"),
                    "bbox": item.get("bbox"),
                    "attributes": item.get("attributes", []),
                    "relations": item.get("relations", []),
                    "total_displacement": item.get("total_displacement"),
                    "description": item.get("description", ""),
                    "confidence": item.get("confidence"),
                    "score": item.get("final_score", item.get("score")),
                }
            )
        # 返回统一 LLM 证据结构。
        return {
            "query": bundle.get("query"),
            "normalized_query": bundle.get("normalized_query"),
            "subqueries": bundle.get("subqueries", []),
            "summary_stats": bundle.get("summary_stats", {}),
            "events": events,
            "entities": entities,
        }

    def build_grounded_prompt(
        self,
        query: str,
        llm_evidence: Dict[str, Any],
    ) -> Dict[str, str]:
        """构建受约束问答提示词，要求模型仅依据证据作答。"""
        # system_prompt 定义角色、证据约束和输出 JSON 协议。
        system_prompt = (
            "You are a grounded spatio-temporal video QA assistant. "
            "Answer strictly from the provided STG evidence. "
            "If evidence is insufficient, say so explicitly. "
            "Return only JSON with this schema: "
            '{"answer": str, "sufficient_evidence": bool, "used_event_ids": [str], '
            '"used_entity_ids": [str], "short_rationale": str}.'
        )
        # user_prompt 注入问题与结构化证据，并重复规则以提高遵循率。
        user_prompt = (
            "Question:\n"
            f"{query}\n\n"
            "Structured Evidence JSON:\n"
            f"{json.dumps(llm_evidence, ensure_ascii=False, indent=2)}\n\n"
            "Rules:\n"
            "1. Use only the evidence above.\n"
            "2. Cite memory IDs in used_event_ids and used_entity_ids.\n"
            "3. If the evidence cannot support a reliable answer, set sufficient_evidence=false.\n"
            "4. Return JSON only.\n"
        )
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}
