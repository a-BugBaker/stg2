"""
闭包检索模块

本模块实现基于DAG的闭包检索（Closure-based Retrieval）：
    - 种子识别：用FAISS检索最相关的种子节点
    - 闭包扩展：从种子向上递归遍历，收集所有祖先节点
    - 上下文线性化：按τ排序，序列化为线性文本

目标：检索一个最小闭合子图，包含最相关的证据节点及其所有必要的历史前提。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .config import STGConfig
from .dag_core import DAGNode, EventType, LogicalClock
from .dag_manager import DAGManager

logger = logging.getLogger(__name__)


class ClosureRetriever:
    """闭包检索器。
    
    实现三步骤：
        1. 种子识别（Seed Identification）
        2. 闭包扩展（Closure Expansion）
        3. 上下文线性化（Context Linearization）
    
    Attributes:
        config: STG配置
        dag_manager: DAG管理器
        embed_func: 嵌入函数
        node_embeddings: 节点ID到嵌入的映射
    """
    
    def __init__(
        self,
        config: STGConfig,
        dag_manager: DAGManager,
        embed_func: Optional[Callable[[str], np.ndarray]] = None
    ):
        """初始化闭包检索器。
        
        Args:
            config: STG配置
            dag_manager: DAG管理器
            embed_func: 嵌入函数
        """
        self.config = config
        self.dag_manager = dag_manager
        self.embed_func = embed_func
        
        # 节点嵌入缓存（用于快速检索）
        self._node_embeddings: Dict[str, np.ndarray] = {}
        self._node_ids_list: List[str] = []  # 保持顺序，与FAISS索引对应
        
        # FAISS索引
        self._faiss_index = None
    
    def set_embed_func(self, func: Callable[[str], np.ndarray]) -> None:
        """设置嵌入函数。"""
        self.embed_func = func
    
    # ==================== 索引构建 ====================
    
    def build_index(self) -> int:
        """构建FAISS索引。
        
        从DAG管理器中获取所有节点，构建向量索引。
        
        Returns:
            索引的节点数量
        """
        try:
            import faiss
        except ImportError:
            logger.error("FAISS not installed. Please install faiss-cpu or faiss-gpu.")
            return 0
        
        nodes = self.dag_manager.get_all_nodes()
        if not nodes:
            return 0
        
        # 收集嵌入
        embeddings = []
        self._node_ids_list = []
        self._node_embeddings = {}
        
        for node in nodes:
            if node.embedding is not None:
                embeddings.append(node.embedding)
                self._node_ids_list.append(node.node_id)
                self._node_embeddings[node.node_id] = node.embedding
            elif self.embed_func is not None:
                # 计算嵌入
                try:
                    emb = self.embed_func(node.content)
                    embeddings.append(emb)
                    self._node_ids_list.append(node.node_id)
                    self._node_embeddings[node.node_id] = emb
                except Exception as e:
                    logger.warning(f"Failed to compute embedding for node {node.node_id}: {e}")
        
        if not embeddings:
            return 0
        
        # 构建FAISS索引
        embedding_dim = embeddings[0].shape[0]
        embedding_matrix = np.vstack(embeddings).astype(np.float32)
        
        # 使用L2索引（如果向量已归一化，L2等价于余弦相似度）
        self._faiss_index = faiss.IndexFlatIP(embedding_dim)  # 内积索引
        self._faiss_index.add(embedding_matrix)
        
        logger.info(f"Built FAISS index with {len(embeddings)} nodes")
        return len(embeddings)
    
    def add_node_to_index(self, node: DAGNode) -> bool:
        """增量添加节点到索引。
        
        Args:
            node: DAG节点
            
        Returns:
            是否成功添加
        """
        if node.embedding is None:
            if self.embed_func is None:
                return False
            try:
                node.embedding = self.embed_func(node.content)
            except Exception as e:
                logger.warning(f"Failed to compute embedding: {e}")
                return False
        
        self._node_embeddings[node.node_id] = node.embedding
        self._node_ids_list.append(node.node_id)
        
        if self._faiss_index is not None:
            emb = node.embedding.astype(np.float32).reshape(1, -1)
            self._faiss_index.add(emb)
        
        return True
    
    # ==================== 种子识别 ====================
    
    def identify_seeds(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """识别种子节点。
        
        用Top-K嵌入相似度找到最相关的若干种子节点。
        
        Args:
            query: 查询文本
            top_k: 返回的节点数量
            similarity_threshold: 相似度阈值
            
        Returns:
            [(node_id, similarity_score), ...] 列表
        """
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            logger.warning("FAISS index is empty or not built")
            return []
        
        if self.embed_func is None:
            logger.warning("Embedding function not set")
            return []
        
        k = top_k or self.config.dag.closure_top_k_seeds
        
        # 计算查询嵌入
        try:
            query_emb = self.embed_func(query).astype(np.float32).reshape(1, -1)
        except Exception as e:
            logger.error(f"Failed to compute query embedding: {e}")
            return []
        
        # FAISS检索
        actual_k = min(k, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(query_emb, actual_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._node_ids_list):
                continue
            if score < similarity_threshold:
                continue
            
            node_id = self._node_ids_list[idx]
            results.append((node_id, float(score)))
        
        return results
    
    # ==================== 闭包扩展 ====================
    
    def expand_closure(
        self,
        seed_ids: List[str],
        max_depth: Optional[int] = None,
        include_seeds: bool = True
    ) -> Set[str]:
        """闭包扩展。
        
        从种子节点出发，沿依赖边向上递归遍历，收集所有可达的祖先节点。
        
        Args:
            seed_ids: 种子节点ID列表
            max_depth: 最大遍历深度
            include_seeds: 是否包含种子节点本身
            
        Returns:
            闭包内所有节点ID的集合
        """
        depth = max_depth or self.config.dag.closure_max_depth
        
        closure = set()
        
        for seed_id in seed_ids:
            if include_seeds:
                closure.add(seed_id)
            
            # 获取所有祖先
            ancestors = self.dag_manager.get_ancestors(seed_id, depth)
            closure.update(ancestors)
        
        return closure
    
    def expand_closure_with_relevance_filter(
        self,
        seed_ids: List[str],
        query_embedding: np.ndarray,
        max_depth: Optional[int] = None,
        min_relevance: float = 0.3
    ) -> Set[str]:
        """带相关性过滤的闭包扩展。
        
        在向上遍历时，过滤掉与查询相关性过低的节点。
        
        Args:
            seed_ids: 种子节点ID列表
            query_embedding: 查询嵌入向量
            max_depth: 最大遍历深度
            min_relevance: 最小相关性阈值
            
        Returns:
            闭包内所有节点ID的集合
        """
        depth = max_depth or self.config.dag.closure_max_depth
        
        closure = set()
        visited = set()
        
        # BFS遍历
        current_level = set(seed_ids)
        closure.update(seed_ids)
        
        for _ in range(depth):
            if not current_level:
                break
            
            next_level = set()
            for node_id in current_level:
                if node_id in visited:
                    continue
                visited.add(node_id)
                
                # 获取父节点
                parent_ids = self.dag_manager.get_parents(node_id)
                
                for parent_id in parent_ids:
                    if parent_id in visited:
                        continue
                    
                    # 检查相关性
                    if parent_id in self._node_embeddings:
                        parent_emb = self._node_embeddings[parent_id]
                        relevance = np.dot(query_embedding, parent_emb)
                        if relevance < min_relevance:
                            continue
                    
                    closure.add(parent_id)
                    next_level.add(parent_id)
            
            current_level = next_level
        
        return closure
    
    # ==================== 上下文线性化 ====================
    
    def linearize_context(
        self,
        nodes_or_ids: Set[str] | List[DAGNode] | List[str],
        max_tokens: Optional[int] = None,
        include_metadata: bool = False
    ) -> str:
        """上下文线性化。
        
        将闭包内所有节点按τ排序，序列化为线性文本。
        
        Args:
            nodes_or_ids: 节点ID集合、节点列表或节点ID列表
            max_tokens: 最大token数（近似）
            include_metadata: 是否包含元数据
            
        Returns:
            线性化的上下文文本
        """
        if not nodes_or_ids:
            return ""
        
        # 判断输入类型
        first_item = next(iter(nodes_or_ids))
        if isinstance(first_item, DAGNode):
            # 直接是节点列表
            nodes = list(nodes_or_ids)
        else:
            # 是节点ID，需要加载
            nodes = []
            for node_id in nodes_or_ids:
                node = self.dag_manager.get_node(node_id)
                if node and not node.is_tombstone:
                    nodes.append(node)
        
        nodes.sort(key=lambda n: (n.tau.frame_idx, n.tau.seq))
        
        # 构建文本
        lines = []
        total_chars = 0
        max_chars = max_tokens * 4 if max_tokens else float('inf')  # 粗略估计
        
        for node in nodes:
            if include_metadata:
                line = f"[{node.event_type.value}|τ={node.tau}] {node.content}"
            else:
                line = node.content
            
            if total_chars + len(line) > max_chars:
                break
            
            lines.append(line)
            total_chars += len(line)
        
        return "\n".join(lines)
    
    def linearize_context_structured(
        self,
        node_ids: Set[str]
    ) -> List[Dict[str, Any]]:
        """结构化的上下文线性化。
        
        返回按τ排序的节点列表（包含完整信息）。
        
        Args:
            node_ids: 节点ID集合
            
        Returns:
            节点信息字典列表
        """
        if not node_ids:
            return []
        
        nodes = []
        for node_id in node_ids:
            node = self.dag_manager.get_node(node_id)
            if node and not node.is_tombstone:
                nodes.append(node)
        
        nodes.sort(key=lambda n: (n.tau.frame_idx, n.tau.seq))
        
        result = []
        for node in nodes:
            result.append({
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "content": node.content,
                "tau": node.tau.to_tuple(),
                "frame_start": node.frame_start,
                "frame_end": node.frame_end,
                "metadata": node.metadata
            })
        
        return result
    
    # ==================== 完整检索接口 ====================
    
    def closure_retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_depth: Optional[int] = None,
        similarity_threshold: float = 0.0,
        return_structured: bool = False
    ) -> str | List[Dict[str, Any]]:
        """闭包检索主接口。
        
        三步骤：种子识别 → 闭包扩展 → 上下文线性化
        
        Args:
            query: 查询文本
            top_k: 种子节点数量
            max_depth: 闭包扩展深度
            similarity_threshold: 种子相似度阈值
            return_structured: 是否返回结构化结果
            
        Returns:
            线性化的上下文文本，或结构化的节点列表
        """
        # 1. 种子识别
        seeds = self.identify_seeds(query, top_k, similarity_threshold)
        
        if not seeds:
            logger.info("No seeds found for query")
            return [] if return_structured else ""
        
        seed_ids = [s[0] for s in seeds]
        logger.info(f"Found {len(seed_ids)} seeds with similarities: {[f'{s[1]:.3f}' for s in seeds]}")
        
        # 2. 闭包扩展
        closure = self.expand_closure(seed_ids, max_depth)
        logger.info(f"Closure expanded to {len(closure)} nodes")
        
        # 3. 上下文线性化
        if return_structured:
            return self.linearize_context_structured(closure)
        else:
            return self.linearize_context(closure)
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """检索并返回完整上下文信息。
        
        Args:
            query: 查询文本
            top_k: 种子节点数量
            max_depth: 闭包扩展深度
            
        Returns:
            包含种子、闭包、线性化上下文的字典
        """
        # 种子识别
        seeds = self.identify_seeds(query, top_k)
        seed_ids = [s[0] for s in seeds]
        
        # 闭包扩展
        closure = self.expand_closure(seed_ids, max_depth) if seed_ids else set()
        
        # 线性化
        context_text = self.linearize_context(closure)
        context_structured = self.linearize_context_structured(closure)
        
        return {
            "query": query,
            "seeds": seeds,
            "closure_size": len(closure),
            "closure_node_ids": list(closure),
            "context_text": context_text,
            "context_structured": context_structured
        }
    
    def retrieve(
        self,
        sample_id: str,
        query: str,
        top_k: int = 3,
        max_depth: int = 5
    ) -> List[DAGNode]:
        """简化的检索接口，返回排序后的节点列表。
        
        Args:
            sample_id: 样本ID
            query: 查询文本
            top_k: 种子节点数量
            max_depth: 闭包扩展深度
            
        Returns:
            按τ排序的节点列表
        """
        # 设置当前sample
        self.dag_manager.set_current_sample(sample_id)
        
        # 确保索引已建立
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            self.build_index_for_sample(sample_id)
        
        # 种子识别
        seeds = self.identify_seeds(query, top_k)
        
        if not seeds:
            # 返回所有节点（按τ排序）
            all_nodes = self.dag_manager.get_all_nodes(sample_id)
            all_nodes.sort(key=lambda n: (n.tau.frame_idx, n.tau.seq))
            return all_nodes[:top_k]
        
        seed_ids = [s[0] for s in seeds]
        
        # 闭包扩展
        closure = self.expand_closure(seed_ids, max_depth)
        
        # 加载节点并排序
        nodes = []
        for node_id in closure:
            node = self.dag_manager.get_node(sample_id, node_id)
            if node and not node.is_tombstone:
                nodes.append(node)
        
        nodes.sort(key=lambda n: (n.tau.frame_idx, n.tau.seq))
        return nodes
    
    def build_index_for_sample(self, sample_id: str) -> int:
        """为指定样本构建FAISS索引。
        
        Args:
            sample_id: 样本ID
            
        Returns:
            索引的节点数量
        """
        # 获取所有节点
        nodes = self.dag_manager.get_all_nodes(sample_id)
        
        if not nodes:
            logger.warning(f"No nodes found for sample {sample_id}")
            return 0
        
        # 重置索引
        self._node_embeddings.clear()
        self._node_ids_list.clear()
        
        # 确定维度
        dim = self.config.embedding.dim
        if nodes[0].embedding is not None:
            dim = len(nodes[0].embedding)
        
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(dim)
        except ImportError:
            logger.warning("FAISS not available, using fallback")
            self._faiss_index = None
        
        # 添加节点
        embeddings = []
        for node in nodes:
            emb = node.embedding
            if emb is None and self.embed_func is not None:
                try:
                    emb = self.embed_func(node.content)
                    node.embedding = emb
                except Exception as e:
                    logger.warning(f"Failed to compute embedding for node {node.node_id}: {e}")
                    continue
            if emb is None:
                continue
            self._node_embeddings[node.node_id] = emb
            self._node_ids_list.append(node.node_id)
            embeddings.append(np.asarray(emb, dtype=np.float32))
        
        if embeddings and self._faiss_index is not None:
            emb_matrix = np.vstack(embeddings)
            self._faiss_index.add(emb_matrix)
        
        logger.info(f"Built index with {len(embeddings)} nodes for sample {sample_id}")
        return len(embeddings)
