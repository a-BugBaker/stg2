"""
查询解析模块（Query Parser）

本模块负责将用户的自然语言问题解析为结构化的查询信息 QueryParseResult，
供后续的检索和重排序环节使用。

解析内容：
    1. **查询归一化**：统一小写、去多余空格
    2. **子查询分解**：按连词（and/then/while）和标点（?!；等）拆分复合问题
    3. **实体匹配**：将查询中的词与 entity_registry 中的 tag/label 做概念 token 交集匹配，
       找出查询可能涉及的 entity_id / entity_tag / entity_label
    4. **关系关键词提取**：识别 relation/interact/near/holding 等关系线索词
    5. **时序关键词提取**：识别 before/after/first/last/during 等时序线索词
    6. **查询意图推断**：根据关键词命中情况推断意图类型
       （relation / motion / appearance / disappearance / attribute / general）
    7. **优先事件类型**：根据意图推断应优先检索哪些 event_type

输出：QueryParseResult dataclass，包含上述所有解析结果。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set

from .utils import concept_tokens, decompose_query, normalize_text


_TEMPORAL_HINTS = {
    "before",
    "after",
    "first",
    "last",
    "then",
    "while",
    "earlier",
    "later",
    "initial",
    "initially",
    "final",
    "finally",
    "start",
    "starting",
    "end",
    "ending",
    "during",
}

_RELATION_HINTS = {
    "relation",
    "relations",
    "interact",
    "interaction",
    "together",
    "near",
    "holding",
    "held",
    "approaching",
    "approach",
    "with",
    "between",
    "towards",
    "toward",
}

_MOTION_HINTS = {
    "move",
    "moved",
    "moving",
    "motion",
    "trajectory",
    "run",
    "running",
    "walk",
    "walking",
    "jump",
    "jumping",
}

_APPEAR_HINTS = {"appear", "appeared", "enter", "entered", "show", "shown"}
_DISAPPEAR_HINTS = {"disappear", "disappeared", "leave", "left", "gone", "vanish", "vanished"}
_ATTRIBUTE_HINTS = {"attribute", "attributes", "wear", "wearing", "look", "looking", "color", "state"}


@dataclass
class QueryParseResult:
    query: str
    normalized_query: str
    subqueries: List[str]
    entity_ids: List[str]
    entity_tags: List[str]
    entity_labels: List[str]
    relation_keywords: List[str]
    temporal_keywords: List[str]
    query_intents: List[str]
    preferred_event_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """导出查询解析结果为字典。"""
        return asdict(self)


class QueryParser:
    def _label_match(self, query_tokens: Set[str], candidate_text: str) -> bool:
        """判断候选标签 token 是否被查询 token 完整覆盖。"""
        candidate_tokens = concept_tokens(candidate_text)
        if not candidate_tokens:
            return False
        return candidate_tokens.issubset(query_tokens)

    def parse(self, query: str, registry: Sequence[Dict[str, Any]] | None = None) -> QueryParseResult:
        """将自然语言问题解析为结构化查询线索。"""
        # 1) 查询文本归一化并做子查询分解。
        normalized_query = normalize_text(query)
        subqueries = decompose_query(query) or [query]
        query_tokens = concept_tokens(normalized_query)

        entity_ids: List[str] = []
        entity_tags: Set[str] = set()
        entity_labels: Set[str] = set()

        # 2) 与实体注册表做 token 级匹配，抽取实体线索。
        for entity in registry or []:
            entity_id = str(entity.get("entity_id", ""))
            tag = str(entity.get("tag", ""))
            label = str(entity.get("label", ""))
            tag_tokens = concept_tokens(tag)
            label_tokens = concept_tokens(label)
            if query_tokens & tag_tokens:
                if entity_id:
                    entity_ids.append(entity_id)
                if tag:
                    entity_tags.add(tag)
                if label:
                    entity_labels.add(label)
                continue
            if label_tokens and self._label_match(query_tokens, label):
                if label:
                    entity_labels.add(label)

        relation_keywords = sorted(query_tokens & _RELATION_HINTS)
        temporal_keywords = sorted(query_tokens & _TEMPORAL_HINTS)

        # 3) 基于关键词命中推断查询意图集合。
        intents: List[str] = []
        if query_tokens & _RELATION_HINTS:
            intents.append("relation")
        if query_tokens & _MOTION_HINTS:
            intents.append("motion")
        if query_tokens & _APPEAR_HINTS:
            intents.append("appearance")
        if query_tokens & _DISAPPEAR_HINTS:
            intents.append("disappearance")
        if query_tokens & _ATTRIBUTE_HINTS:
            intents.append("attribute")
        if not intents:
            intents.append("general")

        # 4) 根据意图推断优先事件类型并输出标准结构。
        preferred_event_types = self._preferred_event_types(intents, query_tokens)
        entity_ids = sorted(set(entity_ids))

        return QueryParseResult(
            query=query,
            normalized_query=normalized_query,
            subqueries=subqueries,
            entity_ids=entity_ids,
            entity_tags=sorted(entity_tags),
            entity_labels=sorted(entity_labels),
            relation_keywords=relation_keywords,
            temporal_keywords=temporal_keywords,
            query_intents=intents,
            preferred_event_types=preferred_event_types,
        )

    def _preferred_event_types(self, intents: Sequence[str], query_tokens: Iterable[str]) -> List[str]:
        """根据查询意图推断优先检索的事件类型列表。"""
        preferred: List[str] = []
        intent_set = set(intents)

        if "appearance" in intent_set:
            preferred.extend(["entity_appeared", "initial_scene"])
        if "disappearance" in intent_set:
            preferred.append("entity_disappeared")
        if "motion" in intent_set:
            preferred.extend(["entity_moved", "trajectory_summary", "interaction"])
        if "relation" in intent_set:
            preferred.extend(["relation_changed", "interaction"])
        if "attribute" in intent_set:
            preferred.append("attribute_changed")

        if "general" in intent_set or {"what", "happened"} <= set(query_tokens):
            preferred.extend(
                [
                    "entity_moved",
                    "trajectory_summary",
                    "relation_changed",
                    "attribute_changed",
                    "entity_appeared",
                    "entity_disappeared",
                    "interaction",
                ]
            )

        # 去重同时保留原始优先级顺序。
        unique: List[str] = []
        seen = set()
        for item in preferred:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        return unique
