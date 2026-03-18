"""
事件生成器模块

本模块负责根据实体跟踪结果和运动分析结果，生成结构化的事件记忆（event memory）。
每条事件记忆包含统一的元数据格式：memory_type, event_type, frame_start/end, entities,
summary, confidence, source, details 等字段，并附带 dedupe_key 用于去重。

支持生成的事件类型：
    - initial_scene:        首帧场景描述（包含物体数量和类型组成）
    - entity_appeared:      某实体首次出现
    - entity_disappeared:   某实体消失
    - entity_moved:         某实体发生显著位移
    - relation_changed:     某实体的关系发生变化（新增/移除了哪些关系）
    - attribute_changed:    某实体的属性发生变化
    - trajectory_summary:   缓冲区级别的轨迹运动摘要
    - interaction:          缓冲区级别的两实体交互事件（接近/远离/同向移动）

每种事件的 summary 是自然语言文本，会被向量化后存入 FAISS 索引，用于后续检索。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .utils import compact_box, normalize_attributes, relations_to_serializable


class EventGenerator:
    def _make_dedupe_key(self, payload: Dict[str, Any]) -> str:
        """为事件生成稳定去重键，避免重复写入同类事件。"""
        basis = {
            "event_type": payload.get("event_type"),
            "frame_start": payload.get("frame_start"),
            "frame_end": payload.get("frame_end"),
            "entities": payload.get("entities", []),
            "summary": payload.get("summary", ""),
            "details": payload.get("details", {}),
        }
        return hashlib.sha1(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _base_event(
        self,
        *,
        event_type: str,
        frame_start: int,
        frame_end: int,
        entities: Sequence[str],
        summary: str,
        confidence: float,
        source: str,
        entity_tags: Sequence[str] | None = None,
        entity_labels: Sequence[str] | None = None,
        details: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """构造统一事件元数据骨架并附加 dedupe_key。"""
        # 统一事件字段，保证下游检索与图导出结构一致。
        payload: Dict[str, Any] = {
            "memory_type": "event",
            "event_type": event_type,
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "entities": list(entities),
            "entity_tags": list(entity_tags or []),
            "entity_labels": list(entity_labels or []),
            "summary": summary,
            "confidence": round(float(confidence), 4),
            "source": source,
            "details": details or {},
        }
        # 补充去重键，防止同一事件重复入库。
        payload["dedupe_key"] = self._make_dedupe_key(payload)
        return payload

    def gen_initial_scene_description(self, frame_index: int, objects: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """生成首帧场景组成事件（initial_scene）。"""
        # 统计首帧对象组成，并提取标签/类别集合。
        counts: Dict[str, int] = {}
        tags: List[str] = []
        labels: List[str] = []
        for obj in objects:
            label = str(obj.get("label", "object"))
            counts[label] = counts.get(label, 0) + 1
            if obj.get("tag"):
                tags.append(str(obj["tag"]))
            labels.append(label)
        composition = ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))
        # 生成自然语言摘要，供后续向量检索。
        summary = f"Initial scene at frame {frame_index} contains {len(objects)} objects: {composition}."
        return self._base_event(
            event_type="initial_scene",
            frame_start=frame_index,
            frame_end=frame_index,
            entities=[],
            entity_tags=sorted(set(tags)),
            entity_labels=sorted(set(labels)),
            summary=summary,
            confidence=1.0,
            source="immediate",
            details={"object_count": len(objects), "composition": counts},
        )

    def gen_entity_appeared(self, record: Any, frame_index: int) -> Dict[str, Any]:
        """生成实体出现事件（entity_appeared）。"""
        attrs = normalize_attributes(record.last_object.get("attributes", []))
        attr_text = ", ".join(attrs) if attrs else "no obvious attributes"
        summary = (
            f"{record.entity_id} {record.tag} ({record.label}) appeared at frame {frame_index}; "
            f"box={compact_box(record.last_bbox)}; attributes={attr_text}."
        )
        return self._base_event(
            event_type="entity_appeared",
            frame_start=frame_index,
            frame_end=frame_index,
            entities=[record.entity_id],
            entity_tags=[record.tag],
            entity_labels=[record.label],
            summary=summary,
            confidence=float(record.last_object.get("score", 0.95)),
            source="immediate",
            details={"bbox": list(record.last_bbox), "attributes": attrs},
        )

    def gen_entity_disappeared(self, snapshot: Dict[str, Any], frame_index: int) -> Dict[str, Any]:
        """生成实体消失事件（entity_disappeared）。"""
        confidence = max(0.5, 0.95 - 0.1 * float(snapshot.get("missed_frames", 0)))
        summary = (
            f"{snapshot['entity_id']} {snapshot['tag']} ({snapshot['label']}) disappeared at frame {frame_index}; "
            f"last_box={compact_box(snapshot['last_bbox'])}."
        )
        return self._base_event(
            event_type="entity_disappeared",
            frame_start=frame_index,
            frame_end=frame_index,
            entities=[snapshot["entity_id"]],
            entity_tags=[snapshot["tag"]],
            entity_labels=[snapshot["label"]],
            summary=summary,
            confidence=confidence,
            source="immediate",
            details={
                "last_bbox": list(snapshot["last_bbox"]),
                "missed_frames": snapshot.get("missed_frames", 0),
                "disappeared_frame": frame_index,
            },
        )

    def gen_entity_moved(
        self,
        entity_id: str,
        tag: str,
        label: str,
        prev_box: Sequence[float],
        curr_box: Sequence[float],
        displacement: float,
        frame_index: int,
    ) -> Dict[str, Any]:
        """生成显著位移事件（entity_moved）。"""
        summary = (
            f"{entity_id} {tag} ({label}) moved at frame {frame_index} from {compact_box(prev_box)} "
            f"to {compact_box(curr_box)} with displacement {displacement:.1f}px."
        )
        confidence = min(1.0, 0.60 + displacement / 100.0)
        return self._base_event(
            event_type="entity_moved",
            frame_start=frame_index,
            frame_end=frame_index,
            entities=[entity_id],
            entity_tags=[tag],
            entity_labels=[label],
            summary=summary,
            confidence=confidence,
            source="immediate",
            details={
                "prev_bbox": list(prev_box),
                "curr_bbox": list(curr_box),
                "displacement": float(displacement),
            },
        )

    def gen_relation_changed(
        self,
        entity_id: str,
        tag: str,
        label: str,
        changes: Dict[str, List[Tuple[str, str]]],
        frame_index: int,
    ) -> Dict[str, Any]:
        """生成关系变化事件（relation_changed）。"""
        # 将新增/删除关系转成可读字符串，写入摘要与 details。
        added = ", ".join([f"{name}->{target}" for name, target in changes.get("added", [])]) or "none"
        removed = ", ".join([f"{name}->{target}" for name, target in changes.get("removed", [])]) or "none"
        summary = (
            f"{entity_id} {tag} ({label}) relation changed at frame {frame_index}; "
            f"added={added}; removed={removed}."
        )
        return self._base_event(
            event_type="relation_changed",
            frame_start=frame_index,
            frame_end=frame_index,
            entities=[entity_id],
            entity_tags=[tag],
            entity_labels=[label],
            summary=summary,
            confidence=0.82,
            source="immediate",
            details={
                "added_relations": relations_to_serializable(changes.get("added", [])),
                "removed_relations": relations_to_serializable(changes.get("removed", [])),
            },
        )

    def gen_attribute_changed(
        self,
        entity_id: str,
        tag: str,
        label: str,
        prev_attrs: Sequence[str],
        curr_attrs: Sequence[str],
        frame_index: int,
    ) -> Dict[str, Any]:
        """生成属性变化事件（attribute_changed）。"""
        summary = (
            f"{entity_id} {tag} ({label}) attributes changed at frame {frame_index}; "
            f"from {', '.join(prev_attrs) or 'none'} to {', '.join(curr_attrs) or 'none'}."
        )
        return self._base_event(
            event_type="attribute_changed",
            frame_start=frame_index,
            frame_end=frame_index,
            entities=[entity_id],
            entity_tags=[tag],
            entity_labels=[label],
            summary=summary,
            confidence=0.78,
            source="immediate",
            details={"prev_attributes": list(prev_attrs), "curr_attributes": list(curr_attrs)},
        )

    def gen_trajectory_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """根据轨迹分析结果生成摘要事件（trajectory_summary）。"""
        # 轨迹类事件强调时间跨度、位移与速度。
        summary = (
            f"{analysis['entity_id']} {analysis['tag']} ({analysis['label']}) moved {analysis['mode']} from frame "
            f"{analysis['frame_start']} to {analysis['frame_end']}; total_displacement={analysis['total_displacement']:.1f}px; "
            f"avg_speed={analysis['avg_speed']:.2f}px/frame."
        )
        confidence = min(1.0, 0.65 + float(analysis["total_displacement"]) / 150.0)
        # details 中去掉重复主键字段，仅保留分析细节。
        details = {key: value for key, value in analysis.items() if key not in {"entity_id", "tag", "label"}}
        return self._base_event(
            event_type="trajectory_summary",
            frame_start=analysis["frame_start"],
            frame_end=analysis["frame_end"],
            entities=[analysis["entity_id"]],
            entity_tags=[analysis["tag"]],
            entity_labels=[analysis["label"]],
            summary=summary,
            confidence=confidence,
            source="buffer",
            details=details,
        )

    def gen_interaction(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """根据双实体交互分析生成 interaction 事件。"""
        # 交互类事件强调双方实体、时段与距离变化趋势。
        summary = (
            f"{analysis['entity_a']} and {analysis['entity_b']} had interaction {analysis['interaction_type']} "
            f"from frame {analysis['frame_start']} to {analysis['frame_end']}; distance {analysis['distance_start']:.1f}px -> "
            f"{analysis['distance_end']:.1f}px."
        )
        # 用距离变化比例估计置信度，并限制上界为 1.0。
        ratio = max(float(analysis["distance_start"]), 1e-6) / max(float(analysis["distance_end"]), 1e-6)
        confidence = min(1.0, 0.60 + min(ratio, 2.0) / 4.0)
        details = {key: value for key, value in analysis.items() if key not in {"entity_a", "entity_b"}}
        return self._base_event(
            event_type="interaction",
            frame_start=analysis["frame_start"],
            frame_end=analysis["frame_end"],
            entities=[analysis["entity_a"], analysis["entity_b"]],
            entity_tags=[analysis["tag_a"], analysis["tag_b"]],
            entity_labels=[analysis["label_a"], analysis["label_b"]],
            summary=summary,
            confidence=confidence,
            source="buffer",
            details=details,
        )
