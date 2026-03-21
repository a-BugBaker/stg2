"""
DAG存储层模块

本模块实现DAG的三层存储架构：
    - Neo4jGraphStore: 图索引层，存储节点ID及入边关系
    - JSONMetaStore: 元数据存储层，存储节点Content和τ等信息
    - 与现有FAISS向量索引集成

设计原则：
    - 图索引用于快速的拓扑查询（父节点、子节点、可达性）
    - JSON用于持久化节点详细信息
    - FAISS用于语义相似度检索
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .dag_core import DAGNode, EventType, LogicalClock

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Neo4j图索引存储层。
    
    存储节点ID及其入边列表（哪些节点是它的父节点）。
    用于高效的拓扑查询：父节点、子节点、祖先、可达性检查。
    
    如果Neo4j不可用，会回退到内存字典存储（仅用于测试/开发）。
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """初始化Neo4j连接。
        
        Args:
            uri: Neo4j连接URI，如 "bolt://localhost:7687"
            user: 用户名
            password: 密码
        """
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None
        self._fallback_mode = False
        
        # 内存回退存储（当Neo4j不可用时）
        self._memory_nodes: Dict[str, Dict[str, Any]] = {}  # node_id -> {type, tau}
        self._memory_edges: Dict[str, Set[str]] = {}  # child_id -> set of parent_ids
        self._memory_children: Dict[str, Set[str]] = {}  # parent_id -> set of child_ids
        
        self._connect()
    
    def _connect(self) -> None:
        """尝试连接Neo4j，失败则启用回退模式。"""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self._uri, 
                auth=(self._user, self._password)
            )
            # 测试连接
            with self._driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self._uri}")
            self._fallback_mode = False
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}. Using in-memory fallback.")
            self._fallback_mode = True
            self._driver = None
    
    def close(self) -> None:
        """关闭Neo4j连接。"""
        if self._driver:
            self._driver.close()
    
    def create_node(self, node_id: str, node_type: str, tau: tuple) -> None:
        """在图中创建节点。
        
        Args:
            node_id: 节点唯一ID
            node_type: 节点类型
            tau: 逻辑时钟元组 (frame_idx, seq)
        """
        if self._fallback_mode:
            self._memory_nodes[node_id] = {
                "type": node_type,
                "tau_frame": tau[0],
                "tau_seq": tau[1]
            }
            if node_id not in self._memory_edges:
                self._memory_edges[node_id] = set()
            if node_id not in self._memory_children:
                self._memory_children[node_id] = set()
            return
        
        with self._driver.session() as session:
            session.run(
                """
                MERGE (n:DAGNode {node_id: $node_id})
                SET n.type = $node_type, n.tau_frame = $tau_frame, n.tau_seq = $tau_seq
                """,
                node_id=node_id,
                node_type=node_type,
                tau_frame=tau[0],
                tau_seq=tau[1]
            )
    
    def create_edge(self, parent_id: str, child_id: str) -> None:
        """创建从父节点到子节点的边。
        
        Args:
            parent_id: 父节点ID
            child_id: 子节点ID
        """
        if self._fallback_mode:
            if child_id not in self._memory_edges:
                self._memory_edges[child_id] = set()
            self._memory_edges[child_id].add(parent_id)
            if parent_id not in self._memory_children:
                self._memory_children[parent_id] = set()
            self._memory_children[parent_id].add(child_id)
            return
        
        with self._driver.session() as session:
            session.run(
                """
                MATCH (p:DAGNode {node_id: $parent_id})
                MATCH (c:DAGNode {node_id: $child_id})
                MERGE (p)-[:CAUSES]->(c)
                """,
                parent_id=parent_id,
                child_id=child_id
            )
    
    def delete_edge(self, parent_id: str, child_id: str) -> None:
        """删除边。
        
        Args:
            parent_id: 父节点ID
            child_id: 子节点ID
        """
        if self._fallback_mode:
            if child_id in self._memory_edges:
                self._memory_edges[child_id].discard(parent_id)
            if parent_id in self._memory_children:
                self._memory_children[parent_id].discard(child_id)
            return
        
        with self._driver.session() as session:
            session.run(
                """
                MATCH (p:DAGNode {node_id: $parent_id})-[r:CAUSES]->(c:DAGNode {node_id: $child_id})
                DELETE r
                """,
                parent_id=parent_id,
                child_id=child_id
            )
    
    def delete_all_edges_to(self, node_id: str) -> None:
        """删除指向某节点的所有入边。
        
        Args:
            node_id: 目标节点ID
        """
        if self._fallback_mode:
            old_parents = self._memory_edges.get(node_id, set()).copy()
            self._memory_edges[node_id] = set()
            for parent_id in old_parents:
                if parent_id in self._memory_children:
                    self._memory_children[parent_id].discard(node_id)
            return
        
        with self._driver.session() as session:
            session.run(
                """
                MATCH ()-[r:CAUSES]->(n:DAGNode {node_id: $node_id})
                DELETE r
                """,
                node_id=node_id
            )
    
    def get_parents(self, node_id: str) -> List[str]:
        """获取节点的所有父节点ID。
        
        Args:
            node_id: 节点ID
            
        Returns:
            父节点ID列表
        """
        if self._fallback_mode:
            return list(self._memory_edges.get(node_id, set()))
        
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:DAGNode)-[:CAUSES]->(n:DAGNode {node_id: $node_id})
                RETURN p.node_id AS parent_id
                """,
                node_id=node_id
            )
            return [record["parent_id"] for record in result]
    
    def get_children(self, node_id: str) -> List[str]:
        """获取节点的所有子节点ID。
        
        Args:
            node_id: 节点ID
            
        Returns:
            子节点ID列表
        """
        if self._fallback_mode:
            return list(self._memory_children.get(node_id, set()))
        
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (n:DAGNode {node_id: $node_id})-[:CAUSES]->(c:DAGNode)
                RETURN c.node_id AS child_id
                """,
                node_id=node_id
            )
            return [record["child_id"] for record in result]
    
    def get_ancestors(self, node_id: str, max_depth: int = 100) -> Set[str]:
        """获取节点的所有祖先节点ID（向上递归遍历）。
        
        用于闭包扩展。
        
        Args:
            node_id: 起始节点ID
            max_depth: 最大遍历深度
            
        Returns:
            祖先节点ID集合
        """
        if self._fallback_mode:
            ancestors = set()
            queue = [node_id]
            depth = 0
            while queue and depth < max_depth:
                next_queue = []
                for nid in queue:
                    for parent_id in self._memory_edges.get(nid, set()):
                        if parent_id not in ancestors:
                            ancestors.add(parent_id)
                            next_queue.append(parent_id)
                queue = next_queue
                depth += 1
            return ancestors
        
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (n:DAGNode {node_id: $node_id})<-[:CAUSES*1..$max_depth]-(ancestor:DAGNode)
                RETURN DISTINCT ancestor.node_id AS ancestor_id
                """,
                node_id=node_id,
                max_depth=max_depth
            )
            return {record["ancestor_id"] for record in result}
    
    def path_exists(self, from_id: str, to_id: str, max_depth: int = 100) -> bool:
        """检查是否存在从from_id到to_id的路径。
        
        用于传递规约：检查候选父节点是否可通过其他路径到达。
        
        Args:
            from_id: 起始节点ID
            to_id: 目标节点ID
            max_depth: 最大搜索深度
            
        Returns:
            是否存在路径
        """
        if from_id == to_id:
            return True
        
        if self._fallback_mode:
            visited = set()
            queue = [from_id]
            depth = 0
            while queue and depth < max_depth:
                next_queue = []
                for nid in queue:
                    if nid == to_id:
                        return True
                    if nid in visited:
                        continue
                    visited.add(nid)
                    for child_id in self._memory_children.get(nid, set()):
                        if child_id not in visited:
                            next_queue.append(child_id)
                queue = next_queue
                depth += 1
            return False
        
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH path = (from:DAGNode {node_id: $from_id})-[:CAUSES*1..$max_depth]->(to:DAGNode {node_id: $to_id})
                RETURN count(path) > 0 AS exists
                """,
                from_id=from_id,
                to_id=to_id,
                max_depth=max_depth
            )
            record = result.single()
            return record["exists"] if record else False
    
    def get_all_node_ids(self) -> List[str]:
        """获取所有节点ID。"""
        if self._fallback_mode:
            return list(self._memory_nodes.keys())
        
        with self._driver.session() as session:
            result = session.run("MATCH (n:DAGNode) RETURN n.node_id AS node_id")
            return [record["node_id"] for record in result]
    
    def clear(self) -> None:
        """清除所有数据。"""
        if self._fallback_mode:
            self._memory_nodes.clear()
            self._memory_edges.clear()
            self._memory_children.clear()
            return
        
        with self._driver.session() as session:
            session.run("MATCH (n:DAGNode) DETACH DELETE n")
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典（用于持久化内存回退数据）。"""
        return {
            "nodes": self._memory_nodes,
            "edges": {k: list(v) for k, v in self._memory_edges.items()},
            "children": {k: list(v) for k, v in self._memory_children.items()},
        }
    
    def load_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载（用于恢复内存回退数据）。"""
        self._memory_nodes = data.get("nodes", {})
        self._memory_edges = {k: set(v) for k, v in data.get("edges", {}).items()}
        self._memory_children = {k: set(v) for k, v in data.get("children", {}).items()}


class JSONMetaStore:
    """JSON元数据存储层。
    
    存储每个节点的Content、τ等详细元信息。
    每个节点存储为一个独立的JSON文件，支持增量更新。
    """
    
    def __init__(self, base_dir: Path):
        """初始化存储目录。
        
        Args:
            base_dir: 存储根目录
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        
        # 内存缓存，减少IO
        self._cache: Dict[str, DAGNode] = {}
        self._dirty: Set[str] = set()  # 需要写入磁盘的节点
    
    def _node_path(self, sample_id: str, node_id: str) -> Path:
        """获取节点的JSON文件路径。
        
        Args:
            sample_id: 样本ID
            node_id: 节点ID
        """
        # 按sample_id分目录，使用node_id前2位作为子目录
        sample_dir = self._base_dir / sample_id
        prefix = node_id[:2] if len(node_id) >= 2 else "00"
        subdir = sample_dir / prefix
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{node_id}.json"
    
    def _cache_key(self, sample_id: str, node_id: str) -> str:
        """生成缓存key。"""
        return f"{sample_id}:{node_id}"
    
    def save_node(self, sample_id: str, node: DAGNode) -> None:
        """保存节点到缓存（延迟写入磁盘）。
        
        Args:
            sample_id: 样本ID
            node: DAG节点
        """
        cache_key = self._cache_key(sample_id, node.node_id)
        self._cache[cache_key] = node
        self._dirty.add((sample_id, node.node_id))
    
    def load_node(self, sample_id: str, node_id: str) -> Optional[DAGNode]:
        """加载节点。
        
        Args:
            sample_id: 样本ID
            node_id: 节点ID
            
        Returns:
            DAGNode实例，不存在则返回None
        """
        cache_key = self._cache_key(sample_id, node_id)
        
        # 先查缓存
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 从磁盘加载
        path = self._node_path(sample_id, node_id)
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            node = DAGNode.from_dict(data)
            self._cache[cache_key] = node
            return node
        except Exception as e:
            logger.error(f"Failed to load node {node_id}: {e}")
            return None
    
    def update_node(self, sample_id: str, node_id: str, updates: Dict[str, Any]) -> Optional[DAGNode]:
        """更新节点的部分字段。
        
        Args:
            sample_id: 样本ID
            node_id: 节点ID
            updates: 要更新的字段字典
            
        Returns:
            更新后的节点，不存在则返回None
        """
        node = self.load_node(sample_id, node_id)
        if node is None:
            return None
        
        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)
        
        self._dirty.add((sample_id, node_id))
        return node
    
    def delete_node(self, sample_id: str, node_id: str) -> None:
        """删除节点（实际是标记为tombstone）。
        
        Args:
            sample_id: 样本ID
            node_id: 节点ID
        """
        node = self.load_node(sample_id, node_id)
        if node:
            node.is_tombstone = True
            self._dirty.add((sample_id, node_id))
    
    def flush(self) -> None:
        """将脏数据写入磁盘。"""
        for sample_id, node_id in self._dirty:
            cache_key = self._cache_key(sample_id, node_id)
            if cache_key in self._cache:
                node = self._cache[cache_key]
                path = self._node_path(sample_id, node_id)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(node.to_dict(), f, ensure_ascii=False, indent=2)
        self._dirty.clear()
    
    def list_nodes_by_type(self, node_type: EventType) -> List[str]:
        """按类型列出节点ID。
        
        Args:
            node_type: 节点类型
            
        Returns:
            节点ID列表
        """
        result = []
        for node_id, node in self._cache.items():
            if node.node_type == node_type and not node.is_tombstone:
                result.append(node_id)
        return result
    
    def get_all_cached_nodes(self) -> Dict[str, DAGNode]:
        """获取所有缓存的节点。"""
        return {k: v for k, v in self._cache.items() if not v.is_tombstone}
    
    def clear_cache(self) -> None:
        """清除缓存（不删除磁盘文件）。"""
        self._cache.clear()
        self._dirty.clear()
    
    def load_all_from_disk(self) -> int:
        """从磁盘加载所有节点到缓存。
        
        Returns:
            加载的节点数量
        """
        count = 0
        for subdir in self._base_dir.iterdir():
            if subdir.is_dir():
                for json_file in subdir.glob("*.json"):
                    node_id = json_file.stem
                    if node_id not in self._cache:
                        self.load_node(node_id)
                        count += 1
        return count
