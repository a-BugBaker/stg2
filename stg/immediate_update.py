"""
即时更新模块（Immediate Update）

本模块负责逐帧处理：对每一帧的检测物体，执行实体关联并立即生成事件记忆和实体状态记忆。
这是 STG 构建流程中"实时事件生成"的核心环节。

处理流程（process_frame）：
    1. 按 detection_score_threshold 过滤低分物体
    2. 调用 EntityTracker.process_frame() 执行跨帧实体匹配
    3. 如果是首帧：
       - 生成 initial_scene 事件
       - 为每个新实体生成 entity_appeared 事件 + entity_state 记忆
    4. 对于已匹配的实体：
       - 位移超过阈值 → 生成 entity_moved 事件
       - 关系发生变化 → 生成 relation 事件（DAG模式）
       - 属性发生变化 → 生成 attribute_changed 事件
       - 更新该实体的 entity_state 记忆
    5. 对于新出现的实体 → 生成 entity_appeared + entity_state
    6. 对于消失的实体 → 生成 entity_disappeared 事件

当启用DAG时，同时生成DAG节点并建立因果边。
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from .config import STGConfig
from .entity_tracker import EntityTracker
from .event_generator import EventGenerator
from .utils import (
    EmbeddingManager,
    Relation,
    compute_displacement,
    diff_relations_semantic,
    diff_attributes_semantic,
    entity_state_description,
    filter_objects_by_score,
    normalize_attributes,
    normalize_relations,
)
from .vector_store import VectorStore

if TYPE_CHECKING:
    from .dag_manager import DAGManager
    from .dag_event_generator import DAGEventGenerator

logger = logging.getLogger(__name__)


class ChangeDebouncer:
    """关系/属性变化去抖动管理器。

    设计思路：
        - 新增的关系/属性：立即确认（不去抖）
        - 移除的关系/属性：连续 N 帧未出现才确认移除

    这样可以避免因检测器抖动导致的关系/属性"闪烁"误报。
    """

    def __init__(
        self,
        relation_debounce_frames: int = 3,
        attribute_debounce_frames: int = 3,
    ):
        """初始化去抖动器。

        Args:
            relation_debounce_frames: 关系移除需要连续多少帧未出现才确认
            attribute_debounce_frames: 属性移除需要连续多少帧未出现才确认
        """
        self.relation_debounce_frames = relation_debounce_frames
        self.attribute_debounce_frames = attribute_debounce_frames

        # 记录每个实体"待移除"关系的连续未出现帧数
        # {entity_id: {relation: 连续未出现帧数}}
        self._pending_relation_removals: Dict[str, Dict[Relation, int]] = defaultdict(dict)

        # 记录每个实体"待移除"属性的连续未出现帧数
        # {entity_id: {attribute: 连续未出现帧数}}
        self._pending_attribute_removals: Dict[str, Dict[str, int]] = defaultdict(dict)

        # 记录每个实体当前已确认的关系和属性（用于计算真实变化）
        self._confirmed_relations: Dict[str, Set[Relation]] = defaultdict(set)
        self._confirmed_attributes: Dict[str, Set[str]] = defaultdict(set)

    def reset(self) -> None:
        """重置所有状态。"""
        self._pending_relation_removals.clear()
        self._pending_attribute_removals.clear()
        self._confirmed_relations.clear()
        self._confirmed_attributes.clear()

    def process_relation_change(
        self,
        entity_id: str,
        prev_rels: Set[Relation],
        curr_rels: Set[Relation],
        semantic_added: List[Relation],
        semantic_removed: List[Relation],
    ) -> Dict[str, List[Relation]]:
        """处理关系变化，应用去抖动逻辑。

        Args:
            entity_id: 实体ID
            prev_rels: 上一帧的关系集合（归一化后）
            curr_rels: 当前帧的关系集合（归一化后）
            semantic_added: 语义比较后判定为新增的关系
            semantic_removed: 语义比较后判定为移除的关系

        Returns:
            {"added": 确认新增的关系, "removed": 确认移除的关系}
        """
        # 初始化该实体的确认关系集合（首次处理时）
        if entity_id not in self._confirmed_relations:
            self._confirmed_relations[entity_id] = prev_rels.copy()

        confirmed = self._confirmed_relations[entity_id]
        pending = self._pending_relation_removals[entity_id]

        # 1) 处理新增：立即确认
        final_added: List[Relation] = []
        for rel in semantic_added:
            if rel not in confirmed:
                confirmed.add(rel)
                final_added.append(rel)
                # 如果之前在待移除列表中，取消待移除
                pending.pop(rel, None)

        # 2) 处理移除：需要去抖动
        final_removed: List[Relation] = []
        for rel in semantic_removed:
            if rel in confirmed:
                # 累计连续未出现帧数
                pending[rel] = pending.get(rel, 0) + 1
                if pending[rel] >= self.relation_debounce_frames:
                    # 达到阈值，确认移除
                    confirmed.discard(rel)
                    final_removed.append(rel)
                    pending.pop(rel, None)

        # 3) 对于当前帧重新出现的关系，重置其待移除计数
        for rel in curr_rels:
            if rel in pending:
                pending.pop(rel, None)

        return {"added": sorted(final_added), "removed": sorted(final_removed)}

    def process_attribute_change(
        self,
        entity_id: str,
        prev_attrs: Set[str],
        curr_attrs: Set[str],
        semantic_added: List[str],
        semantic_removed: List[str],
    ) -> Dict[str, List[str]]:
        """处理属性变化，应用去抖动逻辑。

        Args:
            entity_id: 实体ID
            prev_attrs: 上一帧的属性集合
            curr_attrs: 当前帧的属性集合
            semantic_added: 语义比较后判定为新增的属性
            semantic_removed: 语义比较后判定为移除的属性

        Returns:
            {"added": 确认新增的属性, "removed": 确认移除的属性}
        """
        # 初始化该实体的确认属性集合（首次处理时）
        if entity_id not in self._confirmed_attributes:
            self._confirmed_attributes[entity_id] = prev_attrs.copy()

        confirmed = self._confirmed_attributes[entity_id]
        pending = self._pending_attribute_removals[entity_id]

        # 1) 处理新增：立即确认
        final_added: List[str] = []
        for attr in semantic_added:
            if attr not in confirmed:
                confirmed.add(attr)
                final_added.append(attr)
                pending.pop(attr, None)

        # 2) 处理移除：需要去抖动
        final_removed: List[str] = []
        for attr in semantic_removed:
            if attr in confirmed:
                pending[attr] = pending.get(attr, 0) + 1
                if pending[attr] >= self.attribute_debounce_frames:
                    confirmed.discard(attr)
                    final_removed.append(attr)
                    pending.pop(attr, None)

        # 3) 对于当前帧重新出现的属性，重置其待移除计数
        for attr in curr_attrs:
            if attr in pending:
                pending.pop(attr, None)

        return {"added": sorted(final_added), "removed": sorted(final_removed)}


class ImmediateUpdater:
    def __init__(
        self,
        config: STGConfig,
        tracker: EntityTracker,
        event_generator: EventGenerator,
        embedder: EmbeddingManager,
        store: VectorStore,
    ):
        """初始化即时更新器及其依赖组件。"""
        self.config = config
        self.tracker = tracker
        self.event_generator = event_generator
        self.embedder = embedder
        self.store = store
        # 初始化去抖动器
        self.debouncer = ChangeDebouncer(
            relation_debounce_frames=config.matching.relation_removal_debounce,
            attribute_debounce_frames=config.matching.attribute_removal_debounce,
        )
        # DAG组件（可选）
        self.dag_manager: Optional["DAGManager"] = None
        self.dag_event_generator: Optional["DAGEventGenerator"] = None
    
    def set_dag_components(
        self,
        dag_manager: "DAGManager",
        dag_event_generator: "DAGEventGenerator"
    ) -> None:
        """设置DAG组件。"""
        self.dag_manager = dag_manager
        self.dag_event_generator = dag_event_generator

    def reset(self) -> None:
        """重置即时更新器状态（包括去抖动器）。"""
        self.debouncer.reset()

    def _write_event(self, sample_id: str, event: Dict[str, Any]) -> None:
        """将事件摘要向量化并写入 events 分区。"""
        # 事件 summary 作为检索语义载体，编码后落入 events 分区。
        summary = event["summary"]
        vector = self.embedder.embed(summary)
        self.store.add(sample_id, "events", vector, event)

    def _entity_state_metadata(self, record: Any, frame_index: int) -> Dict[str, Any]:
        """构建实体状态记忆元数据，并生成去重键。"""
        # 1) 生成可读描述与结构化字段。
        description = entity_state_description(record)
        attrs = normalize_attributes(record.last_object.get("attributes", []))
        relations = list(record.last_object.get("relations", []))
        total_displacement = self.tracker.total_displacement(record)
        # 2) 以关键状态字段构造 dedupe_basis，确保同帧重复写入可被识别。
        dedupe_basis = {
            "entity_id": record.entity_id,
            "frame_index": frame_index,
            "bbox": list(record.last_bbox),
            "attributes": attrs,
            "relations": relations,
            "state": record.state,
        }
        dedupe_key = hashlib.sha1(json.dumps(dedupe_basis, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        # 3) 统一返回 entity_state 记忆元数据。
        return {
            "memory_type": "entity_state",
            "entity_id": record.entity_id,
            "tag": record.tag,
            "label": record.label,
            "frame_index": frame_index,
            "frame_start": record.first_frame,
            "frame_end": record.last_frame,
            "bbox": list(record.last_bbox),
            "attributes": attrs,
            "relations": relations,
            "total_displacement": float(total_displacement),
            "description": description,
            "confidence": float(record.last_object.get("score", 1.0)),
            "status": record.state,
            "source": "immediate",
            "dedupe_key": dedupe_key,
        }

    def _write_entity_state(self, sample_id: str, record: Any, frame_index: int) -> None:
        """写入 entity_state 记忆到 entities 分区。"""
        metadata = self._entity_state_metadata(record, frame_index)
        vector = self.embedder.embed(metadata["description"])
        self.store.add(sample_id, "entities", vector, metadata)

    def process_frame(
        self,
        sample_id: str,
        frame_index: int,
        objects: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """处理单帧并即时生成事件/实体状态记忆。

        返回当前帧活跃实体观测，供缓冲区做跨帧轨迹与交互分析。
        """
        # 1) 先按检测分数过滤低置信目标，减少噪声匹配。
        filtered_objects = filter_objects_by_score(objects, threshold=self.config.matching.detection_score_threshold)
        # 2) 执行跨帧关联，得到 matched/new/disappeared 三类结果。
        associations = self.tracker.process_frame(filtered_objects, frame_index)

        # 3) 首帧特殊处理：生成场景初始化事件 + 新实体出现与状态记忆。
        if associations.is_first_frame:
            if filtered_objects:
                self._write_event(
                    sample_id,
                    self.event_generator.gen_initial_scene_description(frame_index, filtered_objects),
                )
            for record in associations.new_entities:
                self._write_event(sample_id, self.event_generator.gen_entity_appeared(record, frame_index))
                self._write_entity_state(sample_id, record, frame_index)
                # DAG: 创建实体状态节点和出现事件
                self._dag_process_new_entity(record, frame_index, filtered_objects)
            return self.tracker.current_frame_observations(frame_index)

        # 4) 对已匹配实体，根据位移/关系/属性变化生成即时事件。
        for match in associations.matched:
            prev_obj = match.prev_snapshot["last_object"]
            curr_obj = match.curr_object
            record = self.tracker.registry[match.entity_id]
            
            # 位移检测
            displacement = compute_displacement(prev_obj["bbox"], curr_obj["bbox"])
            if displacement >= self.config.matching.movement_event_threshold:
                moved_event = self.event_generator.gen_entity_moved(
                    entity_id=match.entity_id,
                    tag=match.prev_snapshot["tag"],
                    label=match.prev_snapshot["label"],
                    prev_box=prev_obj["bbox"],
                    curr_box=curr_obj["bbox"],
                    displacement=displacement,
                    frame_index=frame_index,
                )
                self._write_event(sample_id, moved_event)
                # DAG: 创建移动事件节点
                self._dag_process_movement(match.prev_snapshot["tag"], frame_index, curr_obj["bbox"])

            # 关系变化检测：使用语义相似度 + 去抖动
            prev_rels = normalize_relations(prev_obj.get("relations", []))
            curr_rels = normalize_relations(curr_obj.get("relations", []))
            semantic_rel_delta = diff_relations_semantic(
                prev_obj,
                curr_obj,
                self.embedder,
                threshold=self.config.matching.relation_semantic_threshold,
            )
            # 应用去抖动：新增立即生效，移除需要连续 N 帧确认
            final_rel_delta = self.debouncer.process_relation_change(
                entity_id=match.entity_id,
                prev_rels=prev_rels,
                curr_rels=curr_rels,
                semantic_added=semantic_rel_delta["added"],
                semantic_removed=semantic_rel_delta["removed"],
            )
            if final_rel_delta["added"] or final_rel_delta["removed"]:
                relation_event = self.event_generator.gen_relation_changed(
                    entity_id=match.entity_id,
                    tag=match.prev_snapshot["tag"],
                    label=match.prev_snapshot["label"],
                    changes=final_rel_delta,
                    frame_index=frame_index,
                )
                self._write_event(sample_id, relation_event)
            
            # DAG: 处理关系事件（使用当前帧的所有关系）
            self._dag_process_relations(
                frame_index,
                match.prev_snapshot["tag"],
                curr_obj.get("relations", []),
            )

            # 属性变化检测：使用语义相似度 + 去抖动
            prev_attrs = normalize_attributes(prev_obj.get("attributes", []))
            curr_attrs = normalize_attributes(curr_obj.get("attributes", []))
            semantic_attr_delta = diff_attributes_semantic(
                prev_attrs,
                curr_attrs,
                self.embedder,
                threshold=self.config.matching.attribute_semantic_threshold,
            )
            # 应用去抖动：新增立即生效，移除需要连续 N 帧确认
            final_attr_delta = self.debouncer.process_attribute_change(
                entity_id=match.entity_id,
                prev_attrs=set(prev_attrs),
                curr_attrs=set(curr_attrs),
                semantic_added=semantic_attr_delta["added"],
                semantic_removed=semantic_attr_delta["removed"],
            )
            if final_attr_delta["added"] or final_attr_delta["removed"]:
                attr_event = self.event_generator.gen_attribute_changed(
                    entity_id=match.entity_id,
                    tag=match.prev_snapshot["tag"],
                    label=match.prev_snapshot["label"],
                    prev_attrs=prev_attrs,
                    curr_attrs=curr_attrs,
                    frame_index=frame_index,
                )
                self._write_event(sample_id, attr_event)
                # DAG: 创建属性变化事件
                self._dag_process_attribute_change(
                    match.prev_snapshot["tag"], frame_index,
                    dict(zip(prev_attrs, prev_attrs)),
                    dict(zip(curr_attrs, curr_attrs))
                )

            self._write_entity_state(sample_id, record, frame_index)
            # DAG: 更新实体状态节点
            self._dag_update_entity_state(record, frame_index, curr_obj)

        # 5) 新增实体写 appeared + entity_state。
        for record in associations.new_entities:
            self._write_event(sample_id, self.event_generator.gen_entity_appeared(record, frame_index))
            self._write_entity_state(sample_id, record, frame_index)
            # DAG: 创建实体状态节点和出现事件
            curr_obj = record.last_object
            self._dag_process_new_entity(record, frame_index, [curr_obj])

        # 6) 消失实体写 disappeared 事件。
        for snapshot in associations.disappeared_entities:
            self._write_event(
                sample_id,
                self.event_generator.gen_entity_disappeared(snapshot, frame_index),
            )
            # DAG: 创建消失事件
            self._dag_process_disappeared(snapshot, frame_index)

        # 7) 返回当前帧活跃观测，供 BufferUpdater 聚合跨帧信息。
        return self.tracker.current_frame_observations(frame_index)
    
    # ==================== DAG辅助方法 ====================
    
    def _dag_process_new_entity(
        self,
        record: Any,
        frame_index: int,
        objects: Sequence[Dict[str, Any]]
    ) -> None:
        """DAG: 处理新实体出现。"""
        if not self.dag_event_generator:
            return
        
        obj = record.last_object
        bbox = tuple(obj["bbox"])
        attributes = {attr: attr for attr in normalize_attributes(obj.get("attributes", []))}
        
        # 创建实体状态节点
        self.dag_event_generator.create_or_update_entity_state(
            entity_tag=record.tag,
            frame_idx=frame_index,
            label=record.label,
            attributes=attributes,
            bbox=bbox,
            layer_id=obj.get("layer_id"),
            layer_mapping=obj.get("layer_mapping")
        )
        
        # 创建出现事件
        self.dag_event_generator.create_appeared_event(
            entity_tag=record.tag,
            frame_idx=frame_index,
            label=record.label,
            bbox=bbox
        )
        
        # 处理layer_mapping
        layer_mapping = obj.get("layer_mapping")
        if layer_mapping:
            # 数据中常见格式是 list[{'tag': child}]，当前实体作为父节点。
            if isinstance(layer_mapping, list):
                normalized = [{"parent_tag": record.tag, **(item if isinstance(item, dict) else {"tag": item})} for item in layer_mapping]
                self.dag_event_generator.process_layer_mapping(normalized, frame_index)
            else:
                self.dag_event_generator.process_layer_mapping(layer_mapping, frame_index)
    
    def _dag_update_entity_state(
        self,
        record: Any,
        frame_index: int,
        curr_obj: Dict[str, Any]
    ) -> None:
        """DAG: 更新实体状态节点。"""
        if not self.dag_event_generator:
            return
        
        bbox = tuple(curr_obj["bbox"])
        attributes = {attr: attr for attr in normalize_attributes(curr_obj.get("attributes", []))}
        
        self.dag_event_generator.create_or_update_entity_state(
            entity_tag=record.tag,
            frame_idx=frame_index,
            label=record.label,
            attributes=attributes,
            bbox=bbox,
            layer_id=curr_obj.get("layer_id"),
            layer_mapping=curr_obj.get("layer_mapping")
        )
    
    def _dag_process_movement(
        self,
        entity_tag: str,
        frame_index: int,
        curr_bbox: Sequence[float]
    ) -> None:
        """DAG: 处理位移事件。"""
        if not self.dag_event_generator:
            return
        
        self.dag_event_generator.check_and_create_movement_event(
            entity_tag=entity_tag,
            frame_idx=frame_index,
            current_bbox=tuple(curr_bbox)
        )
    
    def _dag_process_relations(
        self,
        frame_index: int,
        subject_tag: str,
        relations: List[Dict[str, Any]]
    ) -> None:
        """DAG: 处理关系事件。"""
        if not self.dag_event_generator:
            return

        normalized_relations: List[Dict[str, Any]] = []
        for rel in relations:
            rel_name = rel.get("predicate") or rel.get("name") or rel.get("relation")
            object_tag = rel.get("object_tag") or rel.get("object") or rel.get("target")
            if not rel_name:
                continue
            if not object_tag:
                object_tag = "unknown"
            predicate = f"{subject_tag} {str(rel_name)} {str(object_tag)}"
            normalized_relations.append(
                {
                    "subject_tag": str(subject_tag),
                    "predicate": predicate,
                    "object_tag": str(object_tag),
                }
            )

        if normalized_relations:
            self.dag_event_generator.process_relations(frame_index, normalized_relations)
    
    def _dag_process_attribute_change(
        self,
        entity_tag: str,
        frame_index: int,
        prev_attrs: Dict[str, Any],
        curr_attrs: Dict[str, Any]
    ) -> None:
        """DAG: 处理属性变化事件。"""
        if not self.dag_event_generator:
            return
        
        self.dag_event_generator.check_and_create_attribute_changed_events(
            entity_tag=entity_tag,
            frame_idx=frame_index,
            current_attributes=curr_attrs
        )
    
    def _dag_process_disappeared(
        self,
        snapshot: Dict[str, Any],
        frame_index: int
    ) -> None:
        """DAG: 处理实体消失事件。"""
        if not self.dag_event_generator:
            return
        
        self.dag_event_generator.create_disappeared_event(
            entity_tag=snapshot.get("tag", ""),
            frame_idx=frame_index,
            last_known_bbox=tuple(snapshot.get("last_bbox", [0, 0, 0, 0]))
        )
