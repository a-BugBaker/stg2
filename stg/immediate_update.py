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
       - 关系发生变化 → 生成 relation_changed 事件
       - 属性发生变化 → 生成 attribute_changed 事件
       - 更新该实体的 entity_state 记忆
    5. 对于新出现的实体 → 生成 entity_appeared + entity_state
    6. 对于消失的实体 → 生成 entity_disappeared 事件

所有事件的 summary 文本会被向量化后写入 VectorStore。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence

from .config import STGConfig
from .entity_tracker import EntityTracker
from .event_generator import EventGenerator
from .utils import (
    EmbeddingManager,
    compute_displacement,
    diff_relations,
    entity_state_description,
    filter_objects_by_score,
    normalize_attributes,
)
from .vector_store import VectorStore


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
            return self.tracker.current_frame_observations(frame_index)

        # 4) 对已匹配实体，根据位移/关系/属性变化生成即时事件。
        for match in associations.matched:
            prev_obj = match.prev_snapshot["last_object"]
            curr_obj = match.curr_object
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

            relation_delta = diff_relations(prev_obj, curr_obj)
            if relation_delta["added"] or relation_delta["removed"]:
                relation_event = self.event_generator.gen_relation_changed(
                    entity_id=match.entity_id,
                    tag=match.prev_snapshot["tag"],
                    label=match.prev_snapshot["label"],
                    changes=relation_delta,
                    frame_index=frame_index,
                )
                self._write_event(sample_id, relation_event)

            prev_attrs = normalize_attributes(prev_obj.get("attributes", []))
            curr_attrs = normalize_attributes(curr_obj.get("attributes", []))
            if prev_attrs != curr_attrs:
                attr_event = self.event_generator.gen_attribute_changed(
                    entity_id=match.entity_id,
                    tag=match.prev_snapshot["tag"],
                    label=match.prev_snapshot["label"],
                    prev_attrs=prev_attrs,
                    curr_attrs=curr_attrs,
                    frame_index=frame_index,
                )
                self._write_event(sample_id, attr_event)

            record = self.tracker.registry[match.entity_id]
            self._write_entity_state(sample_id, record, frame_index)

        # 5) 新增实体写 appeared + entity_state。
        for record in associations.new_entities:
            self._write_event(sample_id, self.event_generator.gen_entity_appeared(record, frame_index))
            self._write_entity_state(sample_id, record, frame_index)

        # 6) 消失实体写 disappeared 事件。
        for snapshot in associations.disappeared_entities:
            self._write_event(
                sample_id,
                self.event_generator.gen_entity_disappeared(snapshot, frame_index),
            )

        # 7) 返回当前帧活跃观测，供 BufferUpdater 聚合跨帧信息。
        return self.tracker.current_frame_observations(frame_index)
