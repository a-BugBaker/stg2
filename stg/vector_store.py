"""
向量存储模块（Vector Store）

本模块提供了一个轻量级的持久化向量存储系统，支持可选的 FAISS 加速。
核心设计：
    - 按 (sample_id, key) 二级分区组织数据（如 sample_id="video_001", key="events"/"entities"）
    - 每个分区是一个 VectorPartition，内含向量列表和元数据列表
    - 支持 dedupe_key 机制防止重复写入

VectorPartition:
    - add(vector, metadata):    添加一条向量及其元数据，自动跳过重复 dedupe_key
    - search(query, top_k):     执行 top-k 最近邻搜索（FAISS IndexFlatIP 或 NumPy 内积回退）

VectorStore:
    - add(sample_id, key, vector, metadata):        向指定分区添加数据
    - search(sample_id, key, query, top_k):         在指定分区搜索最相似的 top_k 条
    - save_sample(sample_id):                       持久化所有分区到磁盘（.npy + _meta.json + .index）
    - clear_sample(sample_id):                      清除内存和磁盘上该 sample 的所有数据
    - available_keys(sample_id):                    列出该 sample 已有的所有分区 key
    - all_metadata(sample_id, key):                 返回分区的全部元数据列表

持久化文件格式：
    outputs/store/<sample_id>/<key>.npy             向量矩阵
    outputs/store/<sample_id>/<key>_meta.json       元数据 JSON
    outputs/store/<sample_id>/<key>.index            FAISS 索引（可选）
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore

from .utils import dump_json, ensure_dir, load_json


class VectorPartition:
    def __init__(self, dim: int):
        """初始化单分区向量容器与去重索引。"""
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self._index = None
        self._dirty = True
        self._dedupe_keys: Dict[str, str] = {}

    def add(self, vector: np.ndarray, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """添加单条向量与元数据；若 dedupe_key 重复则跳过。"""
        # 1) 基于 dedupe_key 做幂等写入保护，避免同一事件重复落库。
        dedupe_key = str(metadata.get("dedupe_key", "")).strip()
        if dedupe_key and dedupe_key in self._dedupe_keys:
            return None
        # 2) 统一向量为 float32 一维数组，并做维度校验。
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected vector dim {self.dim}, got {vec.shape[0]}")
        # 3) 追加向量和元数据，同时记录 dedupe_key 到 memory_id 的映射。
        self.vectors.append(vec)
        self.metadata.append(metadata)
        if dedupe_key:
            self._dedupe_keys[dedupe_key] = str(metadata.get("memory_id", ""))
        # 4) 标记索引脏状态，后续 search/save 时触发重建。
        self._dirty = True
        return metadata

    def _build_index(self) -> None:
        """重建 FAISS 内积索引；无 FAISS 时保持 NumPy 回退路径。"""
        if faiss is None:
            self._index = None
            self._dirty = False
            return
        index = faiss.IndexFlatIP(self.dim)
        if self.vectors:
            matrix = np.stack(self.vectors, axis=0).astype(np.float32)
            index.add(matrix)
        self._index = index
        self._dirty = False

    def search(self, query: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """执行 top-k 相似检索并返回分数、元数据与排名。"""
        # 空分区直接返回，避免不必要计算。
        if not self.vectors:
            return []
        # 统一查询向量形状为 (1, dim) 并校验维度一致性。
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"Expected query dim {self.dim}, got {q.shape[1]}")
        # 优先走 FAISS；不可用时回退到 NumPy 内积检索。
        if faiss is not None:
            if self._index is None or self._dirty:
                self._build_index()
            scores, indices = self._index.search(q, min(top_k, len(self.vectors)))
            scores_arr = scores[0].tolist()
            idx_arr = indices[0].tolist()
        else:
            matrix = np.stack(self.vectors, axis=0).astype(np.float32)
            all_scores = (matrix @ q[0]).astype(np.float32)
            idx_arr = np.argsort(all_scores)[::-1][:top_k].tolist()
            scores_arr = [float(all_scores[idx]) for idx in idx_arr]
        # 组装标准化返回结构（score/metadata/rank）。
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores_arr, idx_arr):
            if idx < 0:
                continue
            results.append(
                {
                    "score": float(score),
                    "metadata": self.metadata[int(idx)],
                    "rank": len(results) + 1,
                }
            )
        return results

    def vectors_matrix(self) -> np.ndarray:
        """导出当前分区向量矩阵（空分区返回 0 行矩阵）。"""
        if not self.vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(self.vectors, axis=0).astype(np.float32)


class VectorStore:
    """A lightweight persistent vector store with optional FAISS acceleration."""

    def __init__(self, root_dir: str | Path, dim: int):
        """初始化按 sample/key 双层组织的向量存储。"""
        self.root_dir = ensure_dir(root_dir)
        self.dim = dim
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, VectorPartition]] = {}

    def _sample_dir(self, sample_id: str) -> Path:
        """返回 sample 的存储目录。"""
        return ensure_dir(self.root_dir / sample_id)

    def _paths(self, sample_id: str, key: str) -> Dict[str, Path]:
        """返回分区向量、元数据和 FAISS 索引文件路径。"""
        base = self._sample_dir(sample_id) / key
        return {
            "vectors": base.with_suffix(".npy"),
            "metadata": base.with_name(f"{base.name}_meta.json"),
            "faiss": base.with_suffix(".index"),
        }

    def _ensure_partition(self, sample_id: str, key: str) -> VectorPartition:
        """确保目标分区已初始化并返回该分区对象。"""
        if sample_id not in self._data:
            self._data[sample_id] = {}
        if key not in self._data[sample_id]:
            self._data[sample_id][key] = VectorPartition(dim=self.dim)
        return self._data[sample_id][key]

    def clear_sample(self, sample_id: str) -> None:
        """清空指定样本的内存与磁盘数据。"""
        with self._lock:
            # 先删内存，保证后续读取触发磁盘惰性加载而不是命中旧缓存。
            self._data.pop(sample_id, None)
            sample_dir = self._sample_dir(sample_id)
            if sample_dir.exists():
                # 逐文件删除该 sample 的持久化产物（向量/元数据/索引）。
                for path in sample_dir.iterdir():
                    if path.is_file():
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            continue

    def add(self, sample_id: str, key: str, vector: np.ndarray, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """向指定分区追加一条记忆；缺失 memory_id 时自动分配。"""
        with self._lock:
            # 1) 确保目标 sample/key 分区存在。
            partition = self._ensure_partition(sample_id, key)
            # 2) 复制一份 metadata，避免调用方后续原地修改影响存储数据。
            metadata = dict(metadata)
            # 3) 如未提供 memory_id，则按分区当前长度自动分配稳定 ID。
            if "memory_id" not in metadata:
                metadata["memory_id"] = f"{key}_{len(partition.metadata):06d}"
            return partition.add(vector, metadata)

    def add_batch(self, sample_id: str, key: str, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        """批量写入向量与元数据。"""
        # 使用单条 add 复用去重与 ID 分配逻辑，保持行为一致。
        for vector, metadata in zip(vectors, metadatas):
            self.add(sample_id, key, vector, metadata)

    def save_sample(self, sample_id: str) -> None:
        """将样本全部分区持久化到磁盘。"""
        # 若 sample 尚未在内存中构建，不做落盘。
        if sample_id not in self._data:
            return
        for key, partition in self._data[sample_id].items():
            paths = self._paths(sample_id, key)
            # 向量与元数据分别持久化，便于后续独立加载。
            np.save(paths["vectors"], partition.vectors_matrix())
            dump_json(paths["metadata"], partition.metadata)
            if faiss is not None:
                # 若启用 FAISS，同步写出索引文件用于快速检索。
                partition._build_index()
                faiss.write_index(partition._index, str(paths["faiss"]))

    def _load_partition(self, sample_id: str, key: str) -> VectorPartition:
        """按需惰性加载分区（若尚未加载且磁盘文件存在）。"""
        partition = self._ensure_partition(sample_id, key)
        if partition.metadata or partition.vectors:
            return partition
        paths = self._paths(sample_id, key)
        if not paths["vectors"].exists() or not paths["metadata"].exists():
            return partition
        matrix = np.load(paths["vectors"])
        metadata = load_json(paths["metadata"])
        for vector, meta in zip(matrix, metadata):
            partition.add(np.asarray(vector, dtype=np.float32), meta)
        partition._dirty = True
        return partition

    def search(self, sample_id: str, key: str, query: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """在指定分区执行相似检索。"""
        # 按需加载分区后，复用分区级检索实现。
        partition = self._load_partition(sample_id, key)
        return partition.search(query, top_k=top_k)

    def all_metadata(self, sample_id: str, key: str) -> List[Dict[str, Any]]:
        """获取指定分区全部元数据。"""
        # 返回副本，避免调用方原地修改内部状态。
        partition = self._load_partition(sample_id, key)
        return list(partition.metadata)

    def available_keys(self, sample_id: str) -> List[str]:
        """列出样本当前可用的分区键。"""
        # 合并内存中的 key 与磁盘上的 key，得到完整可见分区列表。
        keys = set(self._data.get(sample_id, {}).keys())
        sample_dir = self._sample_dir(sample_id)
        if sample_dir.exists():
            for path in sample_dir.iterdir():
                if path.name.endswith("_meta.json"):
                    keys.add(path.name[:-10])
                elif path.suffix in {".npy", ".index"}:
                    keys.add(path.stem)
        return sorted(keys)
