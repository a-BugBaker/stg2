"""
场景图数据校验与归一化模块

本模块负责将原始 scene graph JSON 转换为统一、规范的内部格式。
主要完成以下工作：
    1. 校验 JSON 结构的合法性（frame 必须是 dict、object 必须包含 tag/label/bbox 等）
    2. 归一化字段值：标签/标记 → normalize_label/normalize_tag、bbox → 浮点数、关系 → 去重统一格式
    3. 兼容多种输入字段名（如 bbox/box、label/category、subject_relations/object_relations 等）
    4. 发现不合法数据时抛出 SceneGraphValidationError，附带精确的帧号和物体索引

核心函数：
    - load_and_normalize_scene_graph(path): 加载 JSON 文件并返回归一化后的帧列表
    - normalize_scene_graph_payload(payload): 对已加载的 payload 做归一化

异常：
    - SceneGraphValidationError: 场景图不符合预期格式时抛出
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from .utils import (
    frame_index_from_frame,
    load_json,
    normalize_attributes,
    normalize_label,
    normalize_relation_name,
    normalize_relation_target,
    normalize_tag,
)


class SceneGraphValidationError(ValueError):
    """Raised when scene-graph JSON does not satisfy the expected contract."""


def _error(message: str, *, frame_index: int | None = None, object_index: int | None = None) -> SceneGraphValidationError:
    # 统一拼装错误位置信息，便于定位坏数据。
    location = []
    if frame_index is not None:
        location.append(f"frame[{frame_index}]")
    if object_index is not None:
        location.append(f"object[{object_index}]")
    prefix = f"{' '.join(location)}: " if location else ""
    return SceneGraphValidationError(prefix + message)


def _as_float(value: Any, *, field_name: str, frame_index: int, object_index: int) -> float:
    try:
        # 对数值字段统一做 float 强制转换。
        return float(value)
    except (TypeError, ValueError) as exc:
        raise _error(
            f"field '{field_name}' must be numeric, got {value!r}",
            frame_index=frame_index,
            object_index=object_index,
        ) from exc


def _normalize_bbox(raw_bbox: Any, *, frame_index: int, object_index: int) -> List[float]:
    # 校验 bbox 结构必须是长度为 4 的列表/元组。
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise _error(
            "field 'bbox' must be a list/tuple of length 4",
            frame_index=frame_index,
            object_index=object_index,
        )
    # 逐坐标转浮点并附带可追踪的字段名。
    bbox = [
        _as_float(raw_bbox[0], field_name="bbox[0]", frame_index=frame_index, object_index=object_index),
        _as_float(raw_bbox[1], field_name="bbox[1]", frame_index=frame_index, object_index=object_index),
        _as_float(raw_bbox[2], field_name="bbox[2]", frame_index=frame_index, object_index=object_index),
        _as_float(raw_bbox[3], field_name="bbox[3]", frame_index=frame_index, object_index=object_index),
    ]
    x1, y1, x2, y2 = bbox
    # 校验边框坐标顺序合法。
    if x2 < x1 or y2 < y1:
        raise _error(
            f"invalid bbox ordering {bbox}; expected x2>=x1 and y2>=y1",
            frame_index=frame_index,
            object_index=object_index,
        )
    return bbox


def _normalize_relation(rel: Any, *, frame_index: int, object_index: int, rel_index: int) -> Dict[str, str]:
    # 兼容 dict / tuple / 简写字符串三类关系表示。
    if isinstance(rel, dict):
        raw_name = rel.get("name", rel.get("relation", rel.get("predicate", "")))
        raw_target = (
            rel.get("object")
            or rel.get("target")
            or rel.get("object_tag")
            or rel.get("object_id")
            or rel.get("subject_tag")
            or rel.get("source")
        )
    elif isinstance(rel, (list, tuple)) and len(rel) >= 2:
        raw_name = rel[0]
        raw_target = rel[1]
    else:
        raw_name = rel
        raw_target = "unknown"

    name = normalize_relation_name(raw_name)
    target = normalize_relation_target(raw_target)
    # 关系名为空视为非法关系。
    if not name:
        raise _error(
            f"relation[{rel_index}] is missing a valid relation name",
            frame_index=frame_index,
            object_index=object_index,
        )
    return {"name": name, "object": target or "unknown"}


def _coerce_relations(obj: Dict[str, Any], *, frame_index: int, object_index: int) -> List[Dict[str, str]]:
    # 若未提供标准 relations 字段，则尝试从兼容字段合成。
    raw_relations = obj.get("relations")
    if raw_relations is None:
        raw_relations = []
        for rel in obj.get("subject_relations", []) or []:
            raw_relations.append(
                {
                    "name": rel.get("name", rel.get("predicate", "")),
                    "object": rel.get("object", rel.get("object_tag", rel.get("target", "unknown"))),
                }
            )
        for rel in obj.get("object_relations", []) or []:
            raw_relations.append(
                {
                    "name": rel.get("name", rel.get("predicate", "")),
                    "object": rel.get("object", rel.get("subject_tag", rel.get("source", "unknown"))),
                }
            )
        for rel in obj.get("layer_mapping", []) or []:
            raw_relations.append(
                {
                    "name": rel.get("name", "contains"),
                    "object": rel.get("object", rel.get("tag", rel.get("target", "unknown"))),
                }
            )

    if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, (str, bytes)):
        raise _error(
            "field 'relations' must be a list if provided",
            frame_index=frame_index,
            object_index=object_index,
        )

    # 逐条归一化并去重。
    normalized: List[Dict[str, str]] = []
    seen = set()
    for rel_index, rel in enumerate(raw_relations):
        norm_rel = _normalize_relation(rel, frame_index=frame_index, object_index=object_index, rel_index=rel_index)
        key = (norm_rel["name"], norm_rel["object"])
        if key in seen:
            continue
        normalized.append(norm_rel)
        seen.add(key)
    return normalized


def _normalize_object(obj: Any, *, frame_index: int, object_index: int) -> Dict[str, Any]:
    # 1) 对象结构与必填字段校验。
    if not isinstance(obj, dict):
        raise _error("each object must be a JSON object", frame_index=frame_index, object_index=object_index)

    raw_tag = obj.get("tag", obj.get("name", obj.get("id")))
    raw_label = obj.get("label", obj.get("category"))
    raw_bbox = obj.get("bbox", obj.get("box"))

    if raw_tag is None:
        raise _error("missing required field 'tag'", frame_index=frame_index, object_index=object_index)
    if raw_label is None:
        raise _error("missing required field 'label'", frame_index=frame_index, object_index=object_index)
    if raw_bbox is None:
        raise _error("missing required field 'bbox' or 'box'", frame_index=frame_index, object_index=object_index)

    # 2) 归一化 tag/label 并校验非空。
    tag = normalize_tag(raw_tag)
    label = normalize_label(raw_label)
    if not tag:
        raise _error("field 'tag' cannot be empty after normalization", frame_index=frame_index, object_index=object_index)
    if not label:
        raise _error(
            "field 'label' cannot be empty after normalization",
            frame_index=frame_index,
            object_index=object_index,
        )

    # 3) 归一化 bbox/score/attributes/relations。
    bbox = _normalize_bbox(raw_bbox, frame_index=frame_index, object_index=object_index)
    score = obj.get("score", 1.0)
    try:
        score_value = float(score)
    except (TypeError, ValueError) as exc:
        raise _error("field 'score' must be numeric if provided", frame_index=frame_index, object_index=object_index) from exc

    # 4) 返回保留原字段的标准化对象。
    normalized = dict(obj)
    normalized["tag"] = tag
    normalized["label"] = label
    normalized["bbox"] = bbox
    normalized["score"] = score_value
    normalized["attributes"] = normalize_attributes(obj.get("attributes", []))
    normalized["relations"] = _coerce_relations(obj, frame_index=frame_index, object_index=object_index)
    return normalized


def normalize_scene_graph_payload(payload: Any) -> List[Dict[str, Any]]:
    # 1) 兼容两种顶层结构：list 或 {frames: list}。
    if isinstance(payload, dict) and "frames" in payload:
        frames = payload["frames"]
    elif isinstance(payload, list):
        frames = payload
    else:
        raise SceneGraphValidationError(
            "scene graph JSON must be either a list of frames or a dict with a top-level 'frames' field"
        )

    if not isinstance(frames, list):
        raise SceneGraphValidationError("top-level 'frames' must be a list")

    # 2) 逐帧校验并归一化 frame_index 与 objects。
    normalized_frames: List[Dict[str, Any]] = []
    seen_frame_indices = set()

    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise _error("each frame must be a JSON object", frame_index=idx)
        frame_copy = dict(frame)
        frame_index = frame_index_from_frame(frame_copy, idx)
        # 禁止重复 frame_index，避免时间轴歧义。
        if frame_index in seen_frame_indices:
            raise _error(f"duplicate frame_index {frame_index}", frame_index=frame_index)
        seen_frame_indices.add(frame_index)

        objects = frame_copy.get("objects", [])
        if objects is None:
            objects = []
        if not isinstance(objects, list):
            raise _error("field 'objects' must be a list", frame_index=frame_index)

        # 逐对象归一化并回填到 frame。
        normalized_objects = [
            _normalize_object(obj, frame_index=frame_index, object_index=object_index)
            for object_index, obj in enumerate(objects)
        ]

        normalized_frame = dict(frame_copy)
        normalized_frame["frame_index"] = int(frame_index)
        if "image_path" in normalized_frame and normalized_frame["image_path"] is not None:
            normalized_frame["image_path"] = str(normalized_frame["image_path"])
        normalized_frame["objects"] = normalized_objects
        normalized_frames.append(normalized_frame)

    # 3) 最终按 frame_index 排序，保证时间一致性。
    normalized_frames.sort(key=lambda item: int(item["frame_index"]))
    return normalized_frames


def load_and_normalize_scene_graph(scene_graph_path: str | Path) -> List[Dict[str, Any]]:
    # 文件加载与结构归一化的一站式入口。
    payload = load_json(scene_graph_path)
    try:
        return normalize_scene_graph_payload(payload)
    except SceneGraphValidationError as exc:
        raise SceneGraphValidationError(f"{Path(scene_graph_path)}: {exc}") from exc
