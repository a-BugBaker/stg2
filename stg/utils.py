"""
通用工具函数模块

本模块汇集了 STG 系统中被多个模块共用的底层工具，涵盖以下几类：

1. **IO 工具**
   - ensure_dir: 确保目录存在
   - dump_json / load_json: JSON 序列化/反序列化
   - extract_json_object: 从可能含有非 JSON 前缀/后缀的文本中提取 JSON 对象

2. **几何计算**
   - box_center / box_area / compute_iou / compute_iou_matrix: 边界框运算
   - euclidean_distance / compute_displacement: 点/框的欧氏距离
   - compute_direction / angle_difference_deg: 方向角与角度差

3. **文本归一化**
   - normalize_text / normalize_label / normalize_tag: 统一文本格式（小写、去多余空格）
   - normalize_relation_name / normalize_relation_target: 关系名与目标标签归一化
   - tokenize / concept_tokens: 分词与概念 token 提取

4. **嵌入管理**
   - EmbeddingManager: 统一的向量编码器，支持 sentence-transformers 和 hashing 两种后端
     - embed(text) / embed_batch(texts): 单条/批量文本编码
     - label_embedding(label): 带缓存的标签嵌入
     - cosine_similarity(a, b): 余弦相似度

5. **关系/属性处理**
   - normalize_relations / diff_relations / normalize_attributes: 关系集合的归一化、差集计算、属性归一化
   - relations_to_serializable: 关系集合 → 可 JSON 序列化格式

6. **场景图辅助**
   - filter_objects_by_score: 按检测分数过滤物体
   - frame_index_from_frame: 从帧 dict 提取帧编号
   - compact_box: 边界框转紧凑字符串
   - entity_state_description: 生成实体状态的自然语言描述

7. **查询拆分**
   - decompose_query: 将复合问题按连词/标点拆分为子问题列表
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

from .config import EmbeddingConfig

Box = Sequence[float]
Point = Tuple[float, float]
Relation = Tuple[str, str]

_TOKEN_RE = re.compile(r"[\w\-]+", flags=re.UNICODE)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def ensure_dir(path: str | Path) -> Path:
    # 统一将输入转换为 Path，并递归创建目录。
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def dump_json(path: str | Path, payload: Any) -> None:
    # 统一 UTF-8 + pretty JSON 输出，保证跨平台可读性。
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=to_jsonable),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_json_object(text: str) -> Dict[str, Any]:
    try:
        # 优先按“纯 JSON 文本”直接解析。
        return json.loads(text)
    except json.JSONDecodeError:
        # 兼容模型返回带前后缀说明文字的场景，抽取首个 JSON 块再解析。
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            raise
        return json.loads(match.group(0))


def box_center(box: Box) -> Point:
    x1, y1, x2, y2 = map(float, box)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = box_area((ix1, iy1, ix2, iy2))
    union = box_area(box_a) + box_area(box_b) - inter
    if union <= 1e-8:
        return 0.0
    return float(inter / union)


def compute_iou_matrix(boxes_a: Sequence[Box], boxes_b: Sequence[Box]) -> np.ndarray:
    # 任一侧为空时返回对应形状的空矩阵。
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    matrix = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    # 逐对计算 IoU，形成匹配所需代价矩阵。
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            matrix[i, j] = compute_iou(box_a, box_b)
    return matrix


def euclidean_distance(point_a: Point, point_b: Point) -> float:
    return float(math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2))


def compute_displacement(box_a: Box, box_b: Box) -> float:
    return euclidean_distance(box_center(box_a), box_center(box_b))


def compute_direction(point_a: Point, point_b: Point) -> float:
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    return float(math.degrees(math.atan2(dy, dx)))


def angle_difference_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def concept_tokens(text: str) -> Set[str]:
    pieces = tokenize(str(text).replace("_", " ").replace("/", " ").replace(">", " "))
    normalized: Set[str] = set()
    for piece in pieces:
        cleaned = re.sub(r"\d+", "", piece).strip("-_ ")
        if cleaned:
            normalized.add(cleaned)
        if piece:
            normalized.add(piece)
    return normalized


def normalize_label(label: Any) -> str:
    return normalize_text(str(label))


def normalize_tag(tag: Any) -> str:
    return normalize_text(str(tag))


def normalize_relation_name(name: Any) -> str:
    return normalize_text(str(name))


def normalize_relation_target(target: Any) -> str:
    return normalize_text(str(target))


def _stable_index(token: str, dim: int, salt: str) -> int:
    h = hashlib.blake2b(f"{salt}:{token}".encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) % dim


class EmbeddingManager:
    """Embedding helper with a deterministic lightweight fallback."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model: Optional[Any] = None
        self._label_cache: Dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self.config.dim

    def _get_backend(self) -> str:
        # auto 模式按依赖可用性自动选择后端。
        if self.config.backend == "auto":
            return "sentence_transformers" if SentenceTransformer is not None else "hashing"
        return self.config.backend

    def _load_model(self) -> Any:
        # 仅 sentence_transformers 后端需要加载外部模型。
        backend = self._get_backend()
        if backend == "sentence_transformers":
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is not installed, but backend='sentence_transformers' was requested."
                )
            if self._model is None:
                # 惰性加载，避免未使用时占用启动时间与显存。
                self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
            return self._model
        raise RuntimeError("Hashing backend does not use an external model.")

    def _hash_embed(self, text: str) -> np.ndarray:
        # 轻量回退嵌入：token -> 稀疏哈希向量，并可选归一化。
        vector = np.zeros(self.config.dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            idx = _stable_index(token, self.config.dim, salt="idx")
            sign = 1.0 if _stable_index(token, 2, salt="sign") == 0 else -1.0
            vector[idx] += sign
            idx2 = _stable_index(token, self.config.dim, salt="idx2")
            vector[idx2] += 0.5 * sign
        if self.config.normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector /= norm
        return vector.astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        # 单条文本编码入口：根据后端分发实现。
        backend = self._get_backend()
        if backend == "hashing":
            return self._hash_embed(text)
        model = self._load_model()
        vector = model.encode(text, normalize_embeddings=self.config.normalize)
        return np.asarray(vector, dtype=np.float32)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        # 批量编码入口：空输入返回标准空矩阵。
        if not texts:
            return np.zeros((0, self.config.dim), dtype=np.float32)
        backend = self._get_backend()
        if backend == "hashing":
            return np.stack([self._hash_embed(text) for text in texts], axis=0)
        model = self._load_model()
        vectors = model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def label_embedding(self, label: str) -> np.ndarray:
        # 标签嵌入带缓存，减少重复计算开销。
        key = normalize_label(label)
        if key not in self._label_cache:
            self._label_cache[key] = self.embed(key)
        return self._label_cache[key]

    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        return float(np.dot(emb_a, emb_b))


def normalize_relations(relations: Any) -> Set[Relation]:
    # 将多种关系输入格式统一为 (name, object) 二元组集合。
    normalized: Set[Relation] = set()
    if not relations:
        return normalized
    for rel in relations:
        if isinstance(rel, dict):
            name = normalize_relation_name(rel.get("name", rel.get("relation", "")))
            target = normalize_relation_target(
                rel.get("object")
                or rel.get("target")
                or rel.get("object_tag")
                or rel.get("object_id")
                or "unknown"
            )
            if name:
                normalized.add((name, target))
        elif isinstance(rel, (list, tuple)) and len(rel) >= 2:
            name = normalize_relation_name(rel[0])
            target = normalize_relation_target(rel[1])
            if name:
                normalized.add((name, target))
        else:
            name = normalize_relation_name(rel)
            if name:
                normalized.add((name, "unknown"))
    return normalized


def relations_to_serializable(relations: Iterable[Relation]) -> List[Dict[str, str]]:
    return [{"name": name, "object": target} for name, target in sorted(set(relations))]


def diff_relations(prev_obj: Dict[str, Any], curr_obj: Dict[str, Any]) -> Dict[str, List[Relation]]:
    # 基于集合差运算计算关系新增/移除。
    prev_rels = normalize_relations(prev_obj.get("relations", []))
    curr_rels = normalize_relations(curr_obj.get("relations", []))
    added = sorted(curr_rels - prev_rels)
    removed = sorted(prev_rels - curr_rels)
    return {"added": added, "removed": removed}


def normalize_attributes(attributes: Any) -> List[str]:
    if not attributes:
        return []
    if isinstance(attributes, str):
        value = normalize_text(attributes)
        return [value] if value else []
    normalized: List[str] = []
    for attr in attributes:
        value = normalize_text(str(attr))
        if value:
            normalized.append(value)
    return sorted(set(normalized))


def filter_objects_by_score(objects: Sequence[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    # 按检测分过滤，减少低置信目标带来的跟踪误匹配。
    filtered: List[Dict[str, Any]] = []
    for obj in objects:
        score = float(obj.get("score", 1.0))
        if score >= threshold:
            filtered.append(obj)
    return filtered


def frame_index_from_frame(frame: Dict[str, Any], default: int) -> int:
    if "frame_index" in frame:
        return int(frame["frame_index"])
    image_path = str(frame.get("image_path", ""))
    match = re.search(r"(\d+)", Path(image_path).stem)
    if match:
        return int(match.group(1))
    return default


def compact_box(box: Box) -> str:
    x1, y1, x2, y2 = map(int, map(round, box))
    return f"[{x1}, {y1}, {x2}, {y2}]"


def entity_state_description(record: Any) -> str:
    attrs = normalize_attributes(record.last_object.get("attributes", []))
    rels = sorted(
        [f"{name}->{target}" for name, target in normalize_relations(record.last_object.get("relations", []))]
    )
    start_center = box_center(record.first_bbox)
    end_center = box_center(record.last_bbox)
    displacement = euclidean_distance(start_center, end_center)
    attr_text = ", ".join(attrs[:5]) if attrs else "none"
    rel_text = ", ".join(rels[:5]) if rels else "none"
    return (
        f"{record.entity_id}: {record.tag} ({record.label}); frames {record.first_frame}-{record.last_frame}; "
        f"last_box={compact_box(record.last_bbox)}; total_displacement={displacement:.1f}px; "
        f"attributes={attr_text}; relations={rel_text}; state={record.state}"
    )


def decompose_query(query: str) -> List[str]:
    # 先做空白清理，空串直接返回空列表。
    query = query.strip()
    if not query:
        return []
    # 按连词/标点拆分复合问句。
    pieces = re.split(r"\s+(?:and|then|while)\s+|[；;。!?！？]\s*|\s*和\s*|\s*并且\s*", query)
    parts = [part.strip() for part in pieces if part.strip()]
    unique_parts: List[str] = []
    seen = set()
    # 去重但保留原出现顺序。
    for part in parts:
        lowered = part.lower()
        if lowered not in seen:
            unique_parts.append(part)
            seen.add(lowered)
    return unique_parts or [query]
