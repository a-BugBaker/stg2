"""
记忆管理器——STG 系统主控模块

本模块是 STG 系统的顶层入口，STGraphMemory 类编排了构建、检索、导出的全部流程。

■ 构建流程 (build)
    1. 加载并归一化场景图 JSON
    2. 逐帧调用 ImmediateUpdater 生成即时事件和实体状态
    3. 帧观测送入 BufferUpdater 缓冲区，满时 flush 生成轨迹和交互事件
    4. 持久化向量存储到磁盘
    5. 导出 entity_registry.json 和 stg_graph.json

■ 检索流程 (retrieve_evidence)
    1. 调用 QueryParser 将自然语言问题解析为结构化查询信息
    2. 如果启用子查询分解，对每个子查询分别编码并在 events/entities 分区做 dense 检索
    3. 合并去重后，调用 _rerank_hits() 进行启发式重排序：
       - 实体命中加分、关系关键词命中加分、时序线索加分、事件类型-意图匹配加分
    4. 通过 EvidenceFormatter 组装成完整的 evidence bundle

■ LLM 问答 (format_evidence_for_llm + build_grounded_prompt)
    1. 从 evidence bundle 提取精简 JSON 证据
    2. 构建 system_prompt + user_prompt，送入 LLM

■ 图导出 (export_stg_graph)
    - 将实体和事件组织为图节点
    - 生成三类边：event_to_entity_association、temporal_entity_chain、temporal_adjacency

公开接口：
    - build(scene_graph_path, sample_id) → 构建统计信息
    - retrieve_evidence(query, sample_id, ...) → evidence bundle
    - search(query, sample_id, ...) → 同 retrieve_evidence（别名）
    - get_context_for_qa(query, sample_id, ...) → evidence_text 字符串
    - format_evidence_for_llm(bundle, ...) → LLM 精简证据 JSON
    - build_grounded_prompt(query, llm_evidence) → {system_prompt, user_prompt}
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from .buffer_update import BufferUpdater
from .config import STGConfig
from .entity_tracker import EntityTracker
from .evidence_formatter import EvidenceFormatter
from .event_generator import EventGenerator
from .immediate_update import ImmediateUpdater
from .motion_analyzer import MotionAnalyzer
from .query_parser import QueryParseResult, QueryParser
from .schema import load_and_normalize_scene_graph
from .utils import (
    EmbeddingManager,
    compact_box,
    compute_displacement,
    concept_tokens,
    dump_json,
    load_json,
)
from .vector_store import VectorStore


class STGraphMemory:
    def __init__(self, config: STGConfig):
        """初始化 STG 主控组件并装配各子模块。"""
        self.config = config
        self.embedder = EmbeddingManager(config.embedding)
        self.store = VectorStore(config.store_path, dim=config.embedding.dim)
        self.tracker = EntityTracker(config.matching, self.embedder)
        self.event_generator = EventGenerator()
        self.motion_analyzer = MotionAnalyzer(config.trajectory, config.motion)
        self.query_parser = QueryParser()
        self.evidence_formatter = EvidenceFormatter()
        self.immediate_updater = ImmediateUpdater(
            config=config,
            tracker=self.tracker,
            event_generator=self.event_generator,
            embedder=self.embedder,
            store=self.store,
        )
        self.buffer_updater = BufferUpdater(
            config=config,
            motion_analyzer=self.motion_analyzer,
            event_generator=self.event_generator,
            embedder=self.embedder,
            store=self.store,
        )

    def _sample_output_dir(self, sample_id: str) -> Path:
        """返回样本输出目录，不存在则自动创建。"""
        path = Path(self.config.output_dir) / sample_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _registry_path(self, sample_id: str) -> Path:
        """返回实体注册表输出路径。"""
        return self._sample_output_dir(sample_id) / "entity_registry.json"

    def _graph_path(self, sample_id: str) -> Path:
        """返回 STG 图导出路径。"""
        return self._sample_output_dir(sample_id) / "stg_graph.json"

    def reset_build_state(self) -> None:
        """重置构建状态（实体跟踪器、缓冲区与即时更新器）。"""
        self.tracker.reset()
        self.buffer_updater.reset()
        self.immediate_updater.reset()

    def _load_frames(self, scene_graph_path: str | Path) -> List[Dict[str, Any]]:
        """加载并标准化场景图帧序列。"""
        return load_and_normalize_scene_graph(scene_graph_path)

    def build(self, scene_graph_path: str | Path, sample_id: str) -> Dict[str, Any]:
        """执行完整构建流程并返回统计信息。

        该流程会串联即时更新、缓冲更新、向量存储持久化和图结构导出。
        """
        # 1) 加载并标准化帧数据。
        frames = self._load_frames(scene_graph_path)
        # 2) 如配置要求，先清空该 sample 的既有存储，避免新旧数据混杂。
        if self.config.clear_existing_sample:
            self.store.clear_sample(sample_id)
        # 3) 重置内存态组件（跟踪器与缓冲区）。
        self.reset_build_state()

        # 4) 逐帧执行即时更新；缓冲区达到阈值时触发一次批量 flush。
        for frame in frames:
            frame_index = int(frame["frame_index"])
            objects = frame.get("objects", [])
            observations = self.immediate_updater.process_frame(sample_id, frame_index, objects)
            should_flush = self.buffer_updater.observe(observations)
            if should_flush:
                self.buffer_updater.flush(sample_id)

        # 5) 处理最后不足一个 buffer 的残留观测，并持久化向量存储。
        self.buffer_updater.flush(sample_id)
        self.store.save_sample(sample_id)
        # 6) 导出实体注册表与 STG 图，最后返回统计结果。
        registry = self.export_entity_registry(sample_id)
        graph = self.export_stg_graph(sample_id)
        return {
            "sample_id": sample_id,
            "num_frames": len(frames),
            "num_entities": len(registry),
            "num_event_memories": len(self.store.all_metadata(sample_id, "events")),
            "num_entity_memories": len(self.store.all_metadata(sample_id, "entities")),
            "num_graph_nodes": len(graph.get("nodes", [])),
            "num_graph_edges": len(graph.get("edges", [])),
        }

    def _search_partition(
        self,
        sample_id: str,
        key: str,
        query_vector: Any,
        top_k: int,
        similarity_threshold: float,
    ) -> List[Dict[str, Any]]:
        """在指定分区检索并按阈值过滤结果。"""
        results = self.store.search(sample_id, key, query_vector, top_k=top_k)
        return [result for result in results if result["score"] >= similarity_threshold]

    def _load_registry(self, sample_id: str) -> List[Dict[str, Any]]:
        """从磁盘加载实体注册表；不存在则返回空列表。"""
        path = self._registry_path(sample_id)
        if path.exists():
            return load_json(path)
        return []

    def _registry_index(self, sample_id: str) -> Dict[str, Dict[str, Any]]:
        """将注册表转为 entity_id 到实体信息的索引表。"""
        return {item["entity_id"]: item for item in self._load_registry(sample_id)}

    def _dedupe_hits(self, hits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按 memory_id 去重，保留 dense_score 更高的一条。"""
        best: Dict[str, Dict[str, Any]] = {}
        for item in hits:
            metadata = dict(item.get("metadata", {}))
            memory_id = str(metadata.get("memory_id", ""))
            if not memory_id:
                continue
            candidate = dict(metadata)
            candidate["dense_score"] = float(item.get("score", 0.0))
            candidate["memory_id"] = memory_id
            if memory_id not in best or candidate["dense_score"] > best[memory_id]["dense_score"]:
                best[memory_id] = candidate
        return list(best.values())

    def _metadata_tokens(self, item: Dict[str, Any], registry_index: Dict[str, Dict[str, Any]]) -> Set[str]:
        """提取候选记忆的概念 token 集合，供启发式重排使用。"""
        tokens: Set[str] = set()
        for key in ("summary", "description", "event_type", "tag", "label"):
            if item.get(key):
                tokens |= concept_tokens(str(item[key]))
        for value in item.get("entity_tags", []) or []:
            tokens |= concept_tokens(str(value))
        for value in item.get("entity_labels", []) or []:
            tokens |= concept_tokens(str(value))
        for value in item.get("attributes", []) or []:
            tokens |= concept_tokens(str(value))
        for rel in item.get("relations", []) or []:
            if isinstance(rel, dict):
                tokens |= concept_tokens(str(rel.get("name", "")))
                tokens |= concept_tokens(str(rel.get("object", "")))
        for entity_id in item.get("entities", []) or []:
            registry_item = registry_index.get(str(entity_id))
            if registry_item:
                tokens |= concept_tokens(str(registry_item.get("tag", "")))
                tokens |= concept_tokens(str(registry_item.get("label", "")))
        return tokens

    def _intent_bonus(self, item: Dict[str, Any], query_info: QueryParseResult) -> float:
        """根据查询意图与事件类型匹配程度计算加分。"""
        event_type = str(item.get("event_type", ""))
        if not event_type:
            if item.get("memory_type") == "entity_state":
                if "attribute" in query_info.query_intents:
                    return self.config.search.rerank_intent_bonus * 0.5
                if query_info.entity_ids or query_info.entity_tags or query_info.entity_labels:
                    return self.config.search.rerank_intent_bonus * 0.25
            return 0.0
        if event_type in query_info.preferred_event_types:
            return self.config.search.rerank_intent_bonus
        return 0.0

    def _temporal_bonus(self, item: Dict[str, Any], query_info: QueryParseResult, max_frame: int) -> float:
        """根据时序关键词与帧位置计算时序相关加分。"""
        if not query_info.temporal_keywords:
            return 0.0
        frame_start = int(item.get("frame_start", item.get("frame_index", 0)))
        frame_end = int(item.get("frame_end", frame_start))
        bonus = 0.0
        if {"while", "during"} & set(query_info.temporal_keywords) and frame_end > frame_start:
            bonus += self.config.search.rerank_temporal_bonus
        if {"first", "initial", "initially", "before", "start", "starting"} & set(query_info.temporal_keywords):
            closeness = 1.0 - (frame_start / max(max_frame, 1))
            bonus += self.config.search.rerank_temporal_bonus * max(0.0, closeness)
        if {"last", "final", "finally", "after", "later", "end", "ending"} & set(query_info.temporal_keywords):
            closeness = frame_end / max(max_frame, 1)
            bonus += self.config.search.rerank_temporal_bonus * max(0.0, closeness)
        return bonus

    def _rerank_hits(
        self,
        items: Sequence[Dict[str, Any]],
        query_info: QueryParseResult,
        registry_index: Dict[str, Dict[str, Any]],
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """融合 dense_score 与规则加分，输出最终排序结果。"""
        max_frame = 0
        for entity in registry_index.values():
            max_frame = max(max_frame, int(entity.get("last_frame", 0)))
        reranked: List[Dict[str, Any]] = []
        for item in items:
            tokens = self._metadata_tokens(item, registry_index)
            bonus = 0.0
            if query_info.entity_tags and any(concept_tokens(tag) & tokens for tag in query_info.entity_tags):
                bonus += self.config.search.rerank_entity_bonus
            if query_info.entity_labels and any(concept_tokens(label) & tokens for label in query_info.entity_labels):
                bonus += self.config.search.rerank_entity_bonus
            if query_info.relation_keywords and set(query_info.relation_keywords) & tokens:
                bonus += self.config.search.rerank_relation_bonus
            bonus += self._intent_bonus(item, query_info)
            bonus += self._temporal_bonus(item, query_info, max_frame=max_frame)
            final_score = float(item.get("dense_score", 0.0)) + bonus
            enriched = dict(item)
            enriched["rerank_score"] = round(bonus, 4)
            enriched["final_score"] = round(final_score, 4)
            reranked.append(enriched)
        reranked.sort(key=lambda item: item["final_score"], reverse=True)
        return reranked[:top_k]

    def retrieve_evidence(
        self,
        query: str,
        sample_id: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> Dict[str, Any]:
        """执行查询解析、分区检索、去重重排并产出 evidence bundle。"""
        # 1) 解析检索参数并应用默认配置。
        top_k = top_k or self.config.search.top_k
        entity_top_k = min(self.config.search.entity_top_k, top_k)
        similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.config.search.similarity_threshold
        )

        # 2) 加载实体注册表并解析自然语言查询。
        registry = self._load_registry(sample_id)
        registry_index = self._registry_index(sample_id)
        query_info = self.query_parser.parse(query, registry=registry)

        # 3) 子查询级 dense 检索：events/entities 双分区并行累积候选。
        subqueries = query_info.subqueries if self.config.search.enable_subquery_decomposition else [query]
        event_hits: List[Dict[str, Any]] = []
        entity_hits: List[Dict[str, Any]] = []
        candidate_multiplier = max(1, self.config.search.dense_candidate_multiplier)

        for subquery in subqueries:
            query_vector = self.embedder.embed(subquery)
            event_hits.extend(
                self._search_partition(
                    sample_id,
                    "events",
                    query_vector,
                    top_k=top_k * candidate_multiplier,
                    similarity_threshold=similarity_threshold,
                )
            )
            entity_hits.extend(
                self._search_partition(
                    sample_id,
                    "entities",
                    query_vector,
                    top_k=entity_top_k * candidate_multiplier,
                    similarity_threshold=similarity_threshold,
                )
            )

        # 4) 按 memory_id 去重后，结合查询意图/时序线索进行启发式重排。
        deduped_events = self._dedupe_hits(event_hits)
        deduped_entities = self._dedupe_hits(entity_hits)
        reranked_events = self._rerank_hits(deduped_events, query_info, registry_index, top_k=top_k)
        reranked_entities = self._rerank_hits(deduped_entities, query_info, registry_index, top_k=entity_top_k)

        # 5) 组装标准 evidence bundle（文本版 + 结构化 JSON）。
        return self.evidence_formatter.build_bundle(
            query_info=query_info,
            events=reranked_events,
            entities=reranked_entities,
            registry=registry,
        )

    def search_structured(
        self,
        query: str,
        sample_id: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> Dict[str, Any]:
        """retrieve_evidence 的结构化别名接口。"""
        return self.retrieve_evidence(
            query=query,
            sample_id=sample_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def search(
        self,
        query: str,
        sample_id: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> Dict[str, Any]:
        """retrieve_evidence 的简洁别名接口。"""
        # 保持别名语义：所有检索逻辑统一由 retrieve_evidence 承担。
        return self.retrieve_evidence(
            query=query,
            sample_id=sample_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def get_context_for_qa(self, query: str, sample_id: str, top_k: int | None = None) -> str:
        """返回面向阅读的证据文本，用于直接问答或调试。"""
        # 先检索完整 bundle，再提取人类可读的 evidence_text。
        bundle = self.retrieve_evidence(query, sample_id=sample_id, top_k=top_k)
        return bundle["evidence_text"]

    def format_evidence_for_llm(self, bundle: Dict[str, Any], *, max_events: int = 8, max_entities: int = 4) -> Dict[str, Any]:
        """将完整证据裁剪为 LLM 易消费的紧凑结构。"""
        return self.evidence_formatter.format_evidence_for_llm(bundle, max_events=max_events, max_entities=max_entities)

    def build_grounded_prompt(self, query: str, llm_evidence: Dict[str, Any]) -> Dict[str, str]:
        """构建受证据约束的 system/user 提示词。"""
        return self.evidence_formatter.build_grounded_prompt(query, llm_evidence)

    def export_entity_registry(self, sample_id: str) -> List[Dict[str, Any]]:
        """导出并持久化实体注册表。"""
        # tracker 中保存构建期实体全量状态，这里直接快照并落盘。
        registry = self.tracker.export_registry()
        dump_json(self._registry_path(sample_id), registry)
        return registry

    def export_stg_graph(self, sample_id: str) -> Dict[str, Any]:
        """导出 STG 图结构（实体节点、事件节点与时序/关联边）。"""
        # 1) 读取实体注册表与事件记忆元数据，准备图节点。
        registry = self._load_registry(sample_id)
        event_metadata = self.store.all_metadata(sample_id, "events")
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        edge_keys = set()

        # 2) 写入实体节点。
        for entity in registry:
            nodes.append(
                {
                    "id": entity["entity_id"],
                    "node_type": "entity",
                    "label": entity.get("label"),
                    "tag": entity.get("tag"),
                    "frame_start": entity.get("first_frame"),
                    "frame_end": entity.get("last_frame"),
                    "state": entity.get("state"),
                    "total_displacement": round(
                        compute_displacement(entity["first_bbox"], entity["last_bbox"]),
                        3,
                    ),
                }
            )

        # 3) 写入事件节点。
        for event in event_metadata:
            nodes.append(
                {
                    "id": event.get("memory_id"),
                    "node_type": "event",
                    "event_type": event.get("event_type"),
                    "frame_start": event.get("frame_start"),
                    "frame_end": event.get("frame_end"),
                    "summary": event.get("summary", ""),
                    "entities": event.get("entities", []),
                    "entity_tags": event.get("entity_tags", []),
                    "entity_labels": event.get("entity_labels", []),
                    "confidence": event.get("confidence"),
                    "source": event.get("source"),
                }
            )

        # 4) 建立 event_to_entity 关联边，并按实体聚合其事件序列。
        entity_to_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in event_metadata:
            event_id = event.get("memory_id")
            if not event_id:
                continue
            for entity_id in event.get("entities", []):
                edge_key = (str(entity_id), str(event_id), "event_to_entity_association")
                if edge_key not in edge_keys:
                    edges.append(
                        {
                            "source": str(entity_id),
                            "target": str(event_id),
                            "relation": "event_to_entity_association",
                        }
                    )
                    edge_keys.add(edge_key)
                entity_to_events[str(entity_id)].append(event)

        # 5) 对每个实体的事件序列构造 temporal_entity_chain 边。
        for entity_id, items in entity_to_events.items():
            items = sorted(items, key=lambda item: (item.get("frame_start", 0), item.get("frame_end", 0), item.get("memory_id", "")))
            for src, dst in zip(items[:-1], items[1:]):
                edge_key = (str(src["memory_id"]), str(dst["memory_id"]), "temporal_entity_chain", entity_id)
                if edge_key in edge_keys:
                    continue
                edges.append(
                    {
                        "source": str(src["memory_id"]),
                        "target": str(dst["memory_id"]),
                        "relation": "temporal_entity_chain",
                        "entity_id": entity_id,
                    }
                )
                edge_keys.add(edge_key)

        # 6) 在全局事件序列上构造 temporal_adjacency 边。
        sorted_events = sorted(
            [event for event in event_metadata if event.get("memory_id")],
            key=lambda item: (item.get("frame_start", 0), item.get("frame_end", 0), item.get("memory_id", "")),
        )
        for src, dst in zip(sorted_events[:-1], sorted_events[1:]):
            edge_key = (str(src["memory_id"]), str(dst["memory_id"]), "temporal_adjacency")
            if edge_key in edge_keys:
                continue
            edges.append(
                {
                    "source": str(src["memory_id"]),
                    "target": str(dst["memory_id"]),
                    "relation": "temporal_adjacency",
                }
            )
            edge_keys.add(edge_key)

        # 7) 汇总图统计并写出到磁盘。
        graph = {
            "sample_id": sample_id,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "num_entity_nodes": sum(1 for node in nodes if node["node_type"] == "entity"),
                "num_event_nodes": sum(1 for node in nodes if node["node_type"] == "event"),
                "num_edges": len(edges),
            },
        }
        dump_json(self._graph_path(sample_id), graph)
        return graph
