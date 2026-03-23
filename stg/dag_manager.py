"""
DAG管理器模块

本模块实现DAG的核心管理逻辑：
    - 节点的插入、更新、删除
    - 传递规约算法（保持最小因果骨架）
    - 边的管理
    - 与三层存储的协调

设计原则：
    - 每次插入节点时立即执行传递规约
    - 边 p → v 表示v在语义上依赖于p，τ(p) < τ(v)
    - 保持最小因果骨架：边被保留当且仅当去掉后p无法通过其他路径到达v
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .config import STGConfig
from .dag_core import DAGNode, EventType, LogicalClock, LogicalClockManager
from .dag_storage import JSONMetaStore, Neo4jGraphStore

logger = logging.getLogger(__name__)


class DAGManager:
    """DAG管理器，负责节点和边的管理。
    
    核心功能：
        - 节点的CRUD操作
        - 传递规约算法
        - 逻辑时钟管理
        - 与存储层的协调
    
    Attributes:
        config: STG配置
        clock_manager: 逻辑时钟管理器
        graph_store: Neo4j图索引存储
        meta_store: JSON元数据存储
        embed_func: 嵌入函数（延迟设置）
    """
    
    def __init__(self, config: STGConfig):
        """初始化DAG管理器。
        
        Args:
            config: STG配置
        """
        self.config = config
        self.clock_manager = LogicalClockManager()
        
        # 初始化存储层
        self.graph_store = Neo4jGraphStore(
            uri=config.dag.neo4j_uri,
            user=config.dag.neo4j_user,
            password=config.dag.neo4j_password,
            allow_fallback=config.dag.allow_memory_fallback,
            store_content=config.dag.neo4j_store_content,
        )
        
        self.meta_store = JSONMetaStore(config.dag_node_meta_path)
        
        # 嵌入函数（需要外部设置）
        self._embed_func: Optional[Callable[[str], np.ndarray]] = None
        
        # 实体状态节点映射：entity_tag -> node_id
        self._entity_state_nodes: Dict[str, str] = {}
        
        # 事件链映射：用于追踪同一实体/关系的连续事件
        # key格式："{event_type}:{entity_tag}" 或 "{event_type}:{subject}:{object}"
        self._event_chains: Dict[str, str] = {}  # chain_key -> last_node_id
        
        # 连续关系追踪：用于处理"连续重复则首尾创建"的逻辑
        # key: relation_key (如 "man3:stand_near:man8")
        # value: {"start_node_id": ..., "start_frame": ..., "last_frame": ...}
        self._continuous_relations: Dict[str, Dict[str, Any]] = {}
        
        # 当前处理的sample_id
        self._current_sample_id: str = "default"
    
    def set_current_sample(self, sample_id: str) -> None:
        """设置当前处理的样本ID。
        
        Args:
            sample_id: 样本ID
        """
        self._current_sample_id = sample_id
    
    def set_embed_func(self, func: Callable[[str], np.ndarray]) -> None:
        """设置嵌入函数。
        
        Args:
            func: 接收文本返回向量的函数
        """
        self._embed_func = func
    
    def create_tau(self, frame_idx: int) -> LogicalClock:
        """为指定帧创建新的逻辑时钟。
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            新的LogicalClock
        """
        return self.clock_manager.create_tau(frame_idx)
    
    def create_tau_after_all(self, frame_idx: int) -> LogicalClock:
        """创建一个τ，保证大于指定帧内所有已创建的τ。
        
        用于阶段性描述节点。
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            新的LogicalClock
        """
        return self.clock_manager.create_tau_after_all(frame_idx)
    
    # ==================== 节点操作 ====================
    
    def insert_node(
        self,
        node_type: EventType,
        content: str,
        frame_idx: int,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compute_embedding: bool = True
    ) -> DAGNode:
        """插入新节点到DAG（标准方式）。
        
        流程：
            1. 创建节点（分配τ）
            2. 计算嵌入（可选）
            3. 对候选父节点执行传递规约
            4. 建立边
            5. 存储节点
        
        Args:
            node_type: 节点类型
            content: 自然语言内容
            frame_idx: 帧索引
            parent_ids: 候选父节点ID列表（会执行传递规约）
            metadata: 额外元信息
            compute_embedding: 是否计算嵌入向量
            
        Returns:
            创建的DAGNode
        """
        # 1. 创建节点
        tau = self.create_tau(frame_idx)
        node = DAGNode(
            node_type=node_type,
            content=content,
            tau=tau,
            metadata=metadata or {},
        )
        
        # 2. 计算嵌入
        if compute_embedding and self._embed_func is not None:
            try:
                node.embedding = self._embed_func(content)
            except Exception as e:
                logger.warning(f"Failed to compute embedding: {e}")
        
        # 3. 传递规约
        if parent_ids:
            if self.config.dag.transitive_reduction:
                reduced_parents = self._transitive_reduction(node.node_id, parent_ids)
            else:
                reduced_parents = list(parent_ids)
            for pid in reduced_parents:
                node.add_parent(pid)
        
        # 4. 存储到图索引
        self.graph_store.create_node(
            node_id=node.node_id,
            node_type=node_type.value,
            tau=tau.to_tuple(),
            content=node.content,
            metadata=node.metadata,
            is_tombstone=node.is_tombstone,
        )
        
        # 5. 建立边
        for parent_id in node.parents:
            self.graph_store.create_edge(parent_id, node.node_id)
        
        # 6. 存储元数据
        self.meta_store.save_node(self._current_sample_id, node)
        
        logger.debug(f"Inserted node {node.node_id[:8]}... type={node_type.value} τ={tau}")
        return node
    
    def insert_existing_node(self, sample_id: str, node: DAGNode) -> DAGNode:
        """插入已构造好的节点。
        
        用于直接插入外部构造的DAGNode对象。
        
        Args:
            sample_id: 样本ID
            node: 要插入的节点
            
        Returns:
            插入的节点
        """
        self._current_sample_id = sample_id
        
        # 存储到图索引
        self.graph_store.create_node(
            node_id=node.node_id,
            node_type=node.event_type.value,
            tau=node.tau.to_tuple(),
            content=node.content,
            metadata=node.metadata,
            is_tombstone=node.is_tombstone,
        )
        
        # 建立边
        for parent_id in node.parents:
            self.graph_store.create_edge(parent_id, node.node_id)
        
        # 存储元数据
        self.meta_store.save_node(self._current_sample_id, node)
        
        logger.debug(f"Inserted node {node.node_id[:8]}... type={node.event_type.value} τ={node.tau}")
        return node
    
    def get_node(self, sample_id_or_node_id: str, node_id: Optional[str] = None) -> Optional[DAGNode]:
        """获取节点。
        
        支持两种调用方式：
        - get_node(node_id) - 从当前sample获取
        - get_node(sample_id, node_id) - 从指定sample获取
        
        Args:
            sample_id_or_node_id: 如果node_id为None，这是node_id；否则这是sample_id
            node_id: 节点ID（可选）
            
        Returns:
            DAGNode实例，不存在则返回None
        """
        if node_id is None:
            # 只传了一个参数，作为node_id使用
            actual_node_id = sample_id_or_node_id
        else:
            # 传了两个参数，设置sample_id
            self._current_sample_id = sample_id_or_node_id
            actual_node_id = node_id
        
        return self.meta_store.load_node(self._current_sample_id, actual_node_id)
    
    def update_node_content(self, node_id: str, new_content: str) -> Optional[DAGNode]:
        """更新节点内容（τ保持不变）。
        
        用于entity_state节点的轨迹更新。
        
        Args:
            node_id: 节点ID
            new_content: 新内容
            
        Returns:
            更新后的节点，不存在则返回None
        """
        node = self.meta_store.load_node(self._current_sample_id, node_id)
        if node is None:
            return None
        
        node.content = new_content
        
        # 重新计算嵌入
        if self._embed_func is not None:
            try:
                node.embedding = self._embed_func(new_content)
            except Exception as e:
                logger.warning(f"Failed to compute embedding: {e}")
        
        self.meta_store.save_node(self._current_sample_id, node)
        self.graph_store.create_node(
            node_id=node.node_id,
            node_type=node.event_type.value,
            tau=node.tau.to_tuple(),
            content=node.content,
            metadata=node.metadata,
            is_tombstone=node.is_tombstone,
        )
        return node
    
    def delete_node(self, node_id: str) -> bool:
        """软删除节点（标记为tombstone）。
        
        删除后，其子节点需要重新挂载（调用方负责）。
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否成功
        """
        node = self.meta_store.load_node(self._current_sample_id, node_id)
        if node is None:
            return False
        
        node.is_tombstone = True
        self.meta_store.save_node(self._current_sample_id, node)
        
        # 获取子节点（调用方需要处理孤儿节点）
        children = self.graph_store.get_children(node_id)
        if children:
            logger.info(f"Node {node_id[:8]}... deleted, orphan children: {len(children)}")
        
        return True
    
    # ==================== 传递规约 ====================
    
    def _transitive_reduction(self, new_node_id: str, candidate_parents: List[str]) -> List[str]:
        """对候选父节点集执行传递规约，返回最小父节点集。
        
        规则：若候选父节点p1可通过其他候选父节点到达new_node（即存在路径p1→...→p2），
        则p1作为父节点是冗余的，应该移除。
        
        算法：
            1. 对于每对候选父节点(p1, p2)，检查是否存在p1→...→p2的路径
            2. 如果存在，则p1是冗余的（因为p1已经通过p2间接连接到new_node）
            3. 保留不可化约的父节点
        
        Args:
            new_node_id: 新节点ID（用于日志）
            candidate_parents: 候选父节点ID列表
            
        Returns:
            最小父节点集
        """
        if len(candidate_parents) <= 1:
            return candidate_parents
        
        # 找出所有冗余的父节点
        redundant = set()
        
        for i, p1 in enumerate(candidate_parents):
            if p1 in redundant:
                continue
            for j, p2 in enumerate(candidate_parents):
                if i == j or p2 in redundant:
                    continue
                # 检查是否存在 p1 → ... → p2 的路径
                # 如果存在，说明p1可以通过p2到达new_node，p1是冗余的
                if self.graph_store.path_exists(p1, p2):
                    redundant.add(p1)
                    break
        
        reduced = [p for p in candidate_parents if p not in redundant]
        
        if redundant:
            logger.debug(f"Transitive reduction: {len(candidate_parents)} -> {len(reduced)} parents")
        
        return reduced
    
    # ==================== 实体状态节点管理 ====================
    
    def get_or_create_entity_state(
        self,
        entity_tag: str,
        frame_idx: int,
        initial_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[DAGNode, bool]:
        """获取或创建实体状态节点。
        
        每个实体只有一个entity_state节点。
        
        Args:
            entity_tag: 实体标签
            frame_idx: 帧索引（仅用于创建时）
            initial_content: 初始内容（仅用于创建时）
            metadata: 元信息（仅用于创建时）
            
        Returns:
            (节点, 是否新创建)
        """
        if entity_tag in self._entity_state_nodes:
            node_id = self._entity_state_nodes[entity_tag]
            node = self.get_node(node_id)
            if node and not node.is_tombstone:
                return node, False
        
        # 创建新的实体状态节点
        meta = metadata or {}
        meta["entity_tag"] = entity_tag
        meta["frame_start"] = frame_idx
        
        node = self.insert_node(
            node_type=EventType.ENTITY_STATE,
            content=initial_content,
            frame_idx=frame_idx,
            parent_ids=[],  # 实体状态节点没有父节点
            metadata=meta
        )
        
        self._entity_state_nodes[entity_tag] = node.node_id
        return node, True
    
    def update_entity_state_trajectory(
        self,
        entity_tag: str,
        new_trajectory_content: str
    ) -> Optional[DAGNode]:
        """更新实体状态节点的轨迹（τ不变）。
        
        Args:
            entity_tag: 实体标签
            new_trajectory_content: 新的轨迹内容
            
        Returns:
            更新后的节点，不存在则返回None
        """
        if entity_tag not in self._entity_state_nodes:
            return None
        
        node_id = self._entity_state_nodes[entity_tag]
        return self.update_node_content(node_id, new_trajectory_content)
    
    def get_entity_state_node_id(self, entity_tag: str) -> Optional[str]:
        """获取实体的状态节点ID。
        
        Args:
            entity_tag: 实体标签
            
        Returns:
            节点ID，不存在则返回None
        """
        return self._entity_state_nodes.get(entity_tag)
    
    # ==================== 事件链管理 ====================
    
    def get_last_event_node_id(self, chain_key: str) -> Optional[str]:
        """获取事件链的最后一个节点ID。
        
        Args:
            chain_key: 事件链键，如 "entity_moved:man3"
            
        Returns:
            最后一个节点ID，不存在则返回None
        """
        return self._event_chains.get(chain_key)
    
    def update_event_chain(self, chain_key: str, node_id: str) -> None:
        """更新事件链的最后一个节点。
        
        Args:
            chain_key: 事件链键
            node_id: 新的最后节点ID
        """
        self._event_chains[chain_key] = node_id
    
    # ==================== 连续关系追踪 ====================
    
    def start_continuous_relation(
        self,
        relation_key: str,
        start_node_id: str,
        start_frame: int
    ) -> None:
        """开始追踪一个连续关系。
        
        Args:
            relation_key: 关系键，如 "man3:stand_near:man8"
            start_node_id: 起始节点ID
            start_frame: 起始帧
        """
        self._continuous_relations[relation_key] = {
            "start_node_id": start_node_id,
            "start_frame": start_frame,
            "last_frame": start_frame
        }
    
    def update_continuous_relation(self, relation_key: str, current_frame: int) -> None:
        """更新连续关系的最后帧。
        
        Args:
            relation_key: 关系键
            current_frame: 当前帧
        """
        if relation_key in self._continuous_relations:
            self._continuous_relations[relation_key]["last_frame"] = current_frame
    
    def end_continuous_relation(self, relation_key: str) -> Optional[Dict[str, Any]]:
        """结束连续关系追踪，返回追踪信息。
        
        Args:
            relation_key: 关系键
            
        Returns:
            追踪信息字典，不存在则返回None
        """
        return self._continuous_relations.pop(relation_key, None)
    
    def get_continuous_relation(self, relation_key: str) -> Optional[Dict[str, Any]]:
        """获取连续关系的当前状态。
        
        Args:
            relation_key: 关系键
            
        Returns:
            追踪信息字典，不存在则返回None
        """
        return self._continuous_relations.get(relation_key)
    
    # ==================== layer_mapping处理 ====================
    
    def process_layer_mapping(
        self,
        parent_tag: str,
        child_tag: str,
        frame_idx: int
    ) -> None:
        """处理layer_mapping依赖关系。
        
        规则：parent_tag → child_tag（child依赖parent）
        例如：man → shoes
        
        在两个实体的entity_state节点之间建立边。
        
        Args:
            parent_tag: 上层实体标签（如 "man"）
            child_tag: 下层实体标签（如 "shoes"）
            frame_idx: 帧索引
        """
        parent_node_id = self.get_entity_state_node_id(parent_tag)
        child_node_id = self.get_entity_state_node_id(child_tag)
        
        if parent_node_id and child_node_id:
            # 检查边是否已存在
            existing_parents = self.graph_store.get_parents(child_node_id)
            if parent_node_id not in existing_parents:
                self.graph_store.create_edge(parent_node_id, child_node_id)
                logger.debug(f"Layer mapping edge: {parent_tag} -> {child_tag}")
    
    # ==================== 持久化 ====================
    
    def flush(self) -> None:
        """将所有脏数据写入磁盘。"""
        self.meta_store.flush()
    
    def save_state(self, path: Path) -> None:
        """保存管理器状态。
        
        Args:
            path: 状态文件路径
        """
        import json
        state = {
            "clock_manager": self.clock_manager.to_dict(),
            "entity_state_nodes": self._entity_state_nodes,
            "event_chains": self._event_chains,
            "continuous_relations": self._continuous_relations,
            "graph_store_fallback": self.graph_store.to_dict() if self.graph_store._fallback_mode else None
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        self.flush()
    
    def load_state(self, path: Path) -> None:
        """加载管理器状态。
        
        Args:
            path: 状态文件路径
        """
        import json
        if not path.exists():
            return
        
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        self.clock_manager.load_dict(state.get("clock_manager", {}))
        self._entity_state_nodes = state.get("entity_state_nodes", {})
        self._event_chains = state.get("event_chains", {})
        self._continuous_relations = state.get("continuous_relations", {})
        
        if state.get("graph_store_fallback") and self.graph_store._fallback_mode:
            self.graph_store.load_dict(state["graph_store_fallback"])
        
        # 加载所有节点元数据到缓存
        self.meta_store.load_all_from_disk()
    
    def close(self) -> None:
        """关闭管理器，释放资源。"""
        self.flush()
        self.graph_store.close()
    
    # ==================== 查询辅助 ====================
    
    def get_parents(self, node_id: str) -> List[str]:
        """获取节点的父节点ID列表。"""
        return self.graph_store.get_parents(node_id)
    
    def get_children(self, node_id: str) -> List[str]:
        """获取节点的子节点ID列表。"""
        return self.graph_store.get_children(node_id)
    
    def get_ancestors(self, node_id: str, max_depth: Optional[int] = None) -> Set[str]:
        """获取节点的所有祖先节点ID。"""
        depth = max_depth or self.config.dag.closure_max_depth
        return self.graph_store.get_ancestors(node_id, depth)
    
    def get_all_nodes(self, sample_id: Optional[str] = None) -> List[DAGNode]:
        """获取所有非tombstone节点。
        
        Args:
            sample_id: 样本ID（可选，默认使用当前sample）
            
        Returns:
            节点列表
        """
        sid = sample_id or self._current_sample_id
        nodes = []
        for node_id in self.graph_store.get_all_node_ids():
            node = self.meta_store.load_node(sid, node_id)
            if node and not node.is_tombstone:
                nodes.append(node)
        return nodes
