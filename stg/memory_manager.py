"""
记忆管理器——STG 系统主控模块

本模块是 STG 系统的顶层入口，STGraphMemory 类编排了构建、检索、导出的全部流程。

■ 构建流程 (build)
    1. 加载并归一化场景图 JSON
    2. 逐帧调用 ImmediateUpdater 生成即时事件和实体状态
    3. 帧观测送入 BufferUpdater 缓冲区，满时 flush 生成轨迹和交互事件
    4. 如果启用DAG，同时生成DAG节点和边
    5. 持久化向量存储到磁盘
    6. 导出 entity_registry.json 和 stg_graph.json

■ 检索流程 (retrieve_evidence)
    1. 调用 QueryParser 将自然语言问题解析为结构化查询信息
    2. 如果启用DAG闭包检索，使用 ClosureRetriever
    3. 否则使用传统 Top-K 检索 + 重排序
    4. 通过 EvidenceFormatter 组装成完整的 evidence bundle

■ DAG特性 (当 config.dag.enabled = True)
    - 使用 DAGManager 管理节点和边
    - 使用 DAGEventGenerator 生成带因果关系的事件节点
    - 使用 ClosureRetriever 进行闭包检索
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

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

# DAG模块导入
from .dag_manager import DAGManager
from .dag_event_generator import DAGEventGenerator
from .closure_retrieval import ClosureRetriever

logger = logging.getLogger(__name__)


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
        
        # DAG组件初始化（当启用时）
        self.dag_manager: Optional[DAGManager] = None
        self.dag_event_generator: Optional[DAGEventGenerator] = None
        self.closure_retriever: Optional[ClosureRetriever] = None
        
        if config.dag.enabled:
            self._init_dag_components()
    
    def _init_dag_components(self) -> None:
        """初始化DAG相关组件。"""
        logger.info("Initializing DAG components...")
        
        # 创建DAG管理器
        self.dag_manager = DAGManager(self.config)
        self.dag_manager.set_embed_func(self.embedder.embed)
        
        # 创建DAG事件生成器
        self.dag_event_generator = DAGEventGenerator(self.config, self.dag_manager)
        
        # 创建闭包检索器
        self.closure_retriever = ClosureRetriever(
            self.config,
            self.dag_manager,
            self.embedder.embed
        )
        
        # 将DAG组件注入到ImmediateUpdater和BufferUpdater
        self.immediate_updater.set_dag_components(self.dag_manager, self.dag_event_generator)
        self.buffer_updater.set_dag_components(self.dag_manager, self.dag_event_generator)
        
        logger.info("DAG components initialized successfully")

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
        # 重置DAG组件状态
        if self.dag_event_generator:
            self.dag_event_generator.reset()

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
        if self.dag_manager and self.config.dag.clear_sample_before_build:
            self.dag_manager.clear_sample_graph(sample_id)
        # 3) 重置内存态组件（跟踪器与缓冲区）。
        self.reset_build_state()
        if self.dag_manager:
            self.dag_manager.set_current_sample(sample_id)

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
        
        # 5.5) 如果启用DAG，保存DAG状态并构建闭包检索索引
        if self.dag_manager:
            dag_state_path = self._sample_output_dir(sample_id) / "dag_state.json"
            self.dag_manager.save_state(dag_state_path)
            if self.closure_retriever:
                self.closure_retriever.build_index()
                logger.info(f"DAG saved with {len(self.dag_manager.get_all_nodes())} nodes")
        
        # 6) 导出实体注册表与 STG 图，最后返回统计结果。
        registry = self.export_entity_registry(sample_id)
        graph = self.export_stg_graph(sample_id)
        
        stats = {
            "sample_id": sample_id,
            "num_frames": len(frames),
            "num_entities": len(registry),
            "num_event_memories": len(self.store.all_metadata(sample_id, "events")),
            "num_entity_memories": len(self.store.all_metadata(sample_id, "entities")),
            "num_graph_nodes": len(graph.get("nodes", [])),
            "num_graph_edges": len(graph.get("edges", [])),
        }
        
        # 添加DAG统计
        if self.dag_manager:
            stats["num_dag_nodes"] = len(self.dag_manager.get_all_nodes())
        
        return stats

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
        """执行闭包检索并产出 evidence bundle（仅 DAG 路径）。"""
        if not self.dag_manager or not self.closure_retriever:
            raise RuntimeError("DAG/ClosureRetriever is required. Please enable config.dag.enabled.")

        # similarity_threshold 参数保留接口兼容，但检索统一走闭包路径。
        _ = similarity_threshold

        top_k = top_k or self.config.search.top_k
        entity_top_k = min(self.config.search.entity_top_k, top_k)

        registry = self._load_registry(sample_id)
        registry_index = self._registry_index(sample_id)
        tag_to_entity_id = {
            str(item.get("tag", "")): str(item.get("entity_id", ""))
            for item in registry
            if item.get("tag") and item.get("entity_id")
        }
        query_info = self.query_parser.parse(query, registry=registry)

        # 闭包检索以DAG为唯一来源：按 sample 重新装载索引。
        self.dag_manager.set_current_sample(sample_id)
        self.closure_retriever.build_index_for_sample(sample_id)

        closure_result = self.closure_retriever.retrieve_with_context(
            query=query,
            top_k=top_k,
            max_depth=self.config.dag.closure_max_depth,
        )
        seed_score_map = {str(node_id): float(score) for node_id, score in closure_result.get("seeds", [])}
        structured_nodes = closure_result.get("context_structured", [])

        events: List[Dict[str, Any]] = []
        entities: List[Dict[str, Any]] = []

        for item in structured_nodes:
            node_id = str(item.get("node_id", ""))
            node_type = str(item.get("node_type", ""))
            metadata = dict(item.get("metadata", {}))
            seed_score = seed_score_map.get(node_id, 0.0)
            frame_start = int(item.get("frame_start", 0) or 0)
            frame_end = int(item.get("frame_end", frame_start) or frame_start)
            content = str(item.get("content", ""))

            if node_type == "entity_state":
                tag = str(metadata.get("entity_tag", ""))
                entities.append(
                    {
                        "memory_id": node_id,
                        "memory_type": "entity_state",
                        "entity_id": tag_to_entity_id.get(tag, tag),
                        "tag": tag,
                        "label": metadata.get("label", "unknown"),
                        "frame_index": frame_end,
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "bbox": metadata.get("bbox"),
                        "attributes": metadata.get("attributes", {}),
                        "relations": metadata.get("layer_mapping", []),
                        "description": content,
                        "source": "dag_closure",
                        "dense_score": seed_score,
                        "final_score": seed_score,
                    }
                )
                continue

            entity_tags: List[str] = []
            for key in ("entity_tag", "subject_tag", "object_tag"):
                value = metadata.get(key)
                if isinstance(value, str) and value:
                    entity_tags.append(value)
            involved = metadata.get("involved_entities", [])
            if isinstance(involved, list):
                entity_tags.extend([str(x) for x in involved if x])
            # Preserve order while deduplicating.
            entity_tags = list(dict.fromkeys(entity_tags))
            entity_ids = [tag_to_entity_id.get(tag, tag) for tag in entity_tags]

            events.append(
                {
                    "memory_id": node_id,
                    "memory_type": "event",
                    "event_type": node_type,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "summary": content,
                    "entities": entity_ids,
                    "entity_tags": entity_tags,
                    "entity_labels": [],
                    "confidence": metadata.get("confidence", 1.0),
                    "source": "dag_closure",
                    "dense_score": seed_score,
                    "final_score": seed_score,
                }
            )

        # 闭包内节点已按τ线性化，这里只做数量裁剪。
        events = events[:top_k]
        entities = entities[:entity_top_k]

        bundle = self.evidence_formatter.build_bundle(
            query_info=query_info,
            events=events,
            entities=entities,
            registry=registry,
        )
        bundle["closure_stats"] = {
            "num_seeds": len(closure_result.get("seeds", [])),
            "closure_size": int(closure_result.get("closure_size", 0)),
        }
        return bundle

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
        """导出 DAG 派生视图（保留实体注册表 + DAG 因果边）。"""
        registry = self._load_registry(sample_id)
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        edge_keys = set()
        tag_to_entity_id = {
            str(item.get("tag", "")): str(item.get("entity_id", ""))
            for item in registry
            if item.get("tag") and item.get("entity_id")
        }

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

        # 3) 从DAG加载节点与因果边，导出为直观视图。
        if self.dag_manager:
            dag_nodes = self.dag_manager.get_all_nodes(sample_id)
            for node in dag_nodes:
                metadata = dict(node.metadata)
                nodes.append(
                    {
                        "id": node.node_id,
                        "node_type": "dag_node",
                        "dag_node_type": node.node_type.value,
                        "content": node.content,
                        "tau": node.tau.to_tuple(),
                        "frame_start": node.frame_start,
                        "frame_end": node.frame_end,
                        "metadata": metadata,
                    }
                )

                # 因果边：父 -> 子。
                for parent_id in node.parent_ids:
                    edge_key = (str(parent_id), str(node.node_id), "causes")
                    if edge_key in edge_keys:
                        continue
                    edges.append(
                        {
                            "source": str(parent_id),
                            "target": str(node.node_id),
                            "relation": "causes",
                        }
                    )
                    edge_keys.add(edge_key)

                # 实体映射边：DAG节点关联到实体注册表（便于可视化）。
                entity_tags: List[str] = []
                for key in ("entity_tag", "subject_tag", "object_tag"):
                    value = metadata.get(key)
                    if isinstance(value, str) and value:
                        entity_tags.append(value)
                involved = metadata.get("involved_entities", [])
                if isinstance(involved, list):
                    entity_tags.extend([str(x) for x in involved if x])

                for tag in dict.fromkeys(entity_tags):
                    entity_id = tag_to_entity_id.get(tag)
                    if not entity_id:
                        continue
                    edge_key = (str(entity_id), str(node.node_id), "entity_context")
                    if edge_key in edge_keys:
                        continue
                    edges.append(
                        {
                            "source": str(entity_id),
                            "target": str(node.node_id),
                            "relation": "entity_context",
                            "entity_tag": tag,
                        }
                    )
                    edge_keys.add(edge_key)

        # 4) 汇总图统计并写出到磁盘。
        graph = {
            "sample_id": sample_id,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "num_entity_nodes": sum(1 for node in nodes if node["node_type"] == "entity"),
                "num_dag_nodes": sum(1 for node in nodes if node["node_type"] == "dag_node"),
                "num_edges": len(edges),
            },
        }
        dump_json(self._graph_path(sample_id), graph)
        return graph
