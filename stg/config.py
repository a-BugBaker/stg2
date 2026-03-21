"""
全局配置模块

本模块使用 dataclass 定义了 STG 系统的所有超参数，按功能域拆分为多个子配置：
    - EmbeddingConfig:        文本向量化后端与模型参数
    - EntityMatchingConfig:   跨帧实体关联的匹配阈值（IoU、标签相似度等）
    - TrajectoryConfig:       轨迹分析相关阈值（移动判定、静止判定、跳跃判定等）
    - BufferConfig:           缓冲区大小，决定多少帧做一次 flush
    - MotionConfig:           运动交互分析参数（接近/远离比、同向角度等）
    - SearchConfig:           在线检索参数（top_k、相似度阈值、重排加分等）
    - STGConfig:              顶层配置，聚合上述所有子配置

使用方式：
    config = STGConfig(output_dir="./outputs")
    config.embedding.backend = "hashing"   # 切换到轻量确定性嵌入
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EmbeddingConfig:
    """文本嵌入（Embedding）后端配置。

    backend 取值说明：
        - "auto":                  优先使用 sentence-transformers，不可用时回退到 hashing
        - "sentence_transformers": 强制使用 SentenceTransformer 模型
        - "hashing":               轻量级确定性本地回退方案，不需要 GPU 或预训练模型

    主要字段：
        model_name:   sentence-transformers 模型名称（仅 backend != "hashing" 时生效）
        dim:          向量维度，默认 1024（BGE-M3 维度）
        normalize:    是否对输出向量做 L2 归一化（归一化后内积 = 余弦相似度）
        batch_size:   批量编码时的 batch 大小
        device:       推理设备，"cpu" 或 "cuda"
        random_seed:  hashing 模式下的随机种子
    """

    backend: str = "auto"
    model_name: str = "BAAI/bge-m3"
    dim: int = 1024
    normalize: bool = True
    batch_size: int = 8
    device: str = "cpu"
    random_seed: int = 42


@dataclass
class EntityMatchingConfig:
    """跨帧实体关联超参数。

    这些阈值用于离线构建阶段的实体匹配，与在线检索阈值分开管理。

    主要字段：
        detection_score_threshold: 检测分数低于此值的物体会被过滤，不参与跟踪
        iou_weight:                IoU 在综合匹配分中的权重（1 - iou_weight 为标签相似度权重）
        combined_threshold:        综合匹配分低于此值的候选对不予匹配
        label_threshold:           标签嵌入余弦相似度低于此值的候选对不予匹配
        min_iou_threshold:         IoU 低于此值且 tag 也不完全相同时直接拒绝匹配
        movement_event_threshold:  位移超过此像素值时生成 entity_moved 事件
        miss_tolerance:            实体最多允许连续丢失多少帧仍保持 inactive 状态
        relation_semantic_threshold: 关系语义相似度阈值，超过此值视为相同关系
        attribute_semantic_threshold: 属性语义相似度阈值，超过此值视为相同属性
        relation_removal_debounce: 关系移除去抖动帧数，连续 N 帧未出现才确认移除
        attribute_removal_debounce: 属性移除去抖动帧数，连续 N 帧未出现才确认移除
    """

    detection_score_threshold: float = 0.35
    iou_weight: float = 0.50
    combined_threshold: float = 0.40
    label_threshold: float = 0.35
    min_iou_threshold: float = 0.01
    movement_event_threshold: float = 10.0
    miss_tolerance: int = 0
    relation_semantic_threshold: float = 0.85
    attribute_semantic_threshold: float = 0.85
    relation_removal_debounce: int = 3
    attribute_removal_debounce: int = 3


@dataclass
class TrajectoryConfig:
    """轨迹分析相关阈值。

    主要字段：
        movement_threshold:       总位移低于此值的实体不生成轨迹摘要
        static_threshold:         dx/dy 均低于此值判定为"静止"
        jump_vertical_threshold:  垂直位移超过此值且向上运动时判定为"跳跃"
        min_frames_for_summary:   至少需要多少帧观测才能生成轨迹摘要
    """
    movement_threshold: float = 10.0
    static_threshold: float = 15.0
    jump_vertical_threshold: float = 25.0
    min_frames_for_summary: int = 2


@dataclass
class BufferConfig:
    """缓冲区配置。

    buffer_size: 每积累多少帧观测后做一次缓冲区刷新（flush），
                 生成轨迹摘要和交互事件。
    """
    buffer_size: int = 5


@dataclass
class MotionConfig:
    """运动交互分析参数。

    用于判断两个实体之间的空间交互类型：
        approach_distance_ratio:    结束距离/起始距离 < 此值 → 判定为"接近"
        depart_distance_ratio:      结束距离/起始距离 > 此值 → 判定为"远离"
        moving_together_angle_deg:  两实体运动方向夹角 < 此角度 → 判定为"同向移动"
        direction_change_angle_deg: 相邻步的方向夹角 > 此角度 → 计为一次方向突变
        min_interaction_distance:   起始距离低于此值的实体对不分析交互
    """
    approach_distance_ratio: float = 0.70
    depart_distance_ratio: float = 1.43
    moving_together_angle_deg: float = 30.0
    direction_change_angle_deg: float = 45.0
    min_interaction_distance: float = 5.0


@dataclass
class SearchConfig:
    """在线检索与重排序参数。

    主要字段：
        top_k:                       最终返回的事件证据条数
        similarity_threshold:        余弦相似度低于此值的候选直接丢弃
        entity_top_k:                最终返回的实体状态证据条数
        dense_candidate_multiplier:  初筛候选倍数（实际从 FAISS 取 top_k * multiplier 条再重排）
        enable_subquery_decomposition: 是否启用复合问题拆分为子查询
        rerank_entity_bonus:         重排时：查询实体命中加分
        rerank_relation_bonus:       重排时：关系关键词命中加分
        rerank_temporal_bonus:       重排时：时序关键词命中加分
        rerank_intent_bonus:         重排时：事件类型与查询意图匹配加分
    """
    top_k: int = 8
    similarity_threshold: float = 0.15
    entity_top_k: int = 4
    dense_candidate_multiplier: int = 3
    enable_subquery_decomposition: bool = True
    rerank_entity_bonus: float = 0.12
    rerank_relation_bonus: float = 0.10
    rerank_temporal_bonus: float = 0.08
    rerank_intent_bonus: float = 0.12


@dataclass
class DAGConfig:
    """DAG（有向无环图）相关配置。
    
    事件开关用于消融实验，可以选择性关闭某些事件类型的记录。
    
    主要字段：
        enabled:                      是否启用DAG存储结构
        neo4j_uri:                    Neo4j数据库连接URI
        neo4j_user:                   Neo4j用户名
        neo4j_password:               Neo4j密码
        node_meta_dir:                节点元数据JSON存储目录
        enable_entity_appeared:       是否记录实体出现事件
        enable_entity_moved:          是否记录实体移动事件
        enable_relation:              是否记录关系事件
        enable_attribute_changed:     是否记录属性变化事件
        enable_interaction:           是否记录交互事件（缓冲区级）
        enable_occlusion:             是否记录遮挡事件（缓冲区级）
        enable_entity_disappeared:    是否记录实体消失事件
        enable_periodic_description:  是否记录阶段性描述（缓冲区级）
        movement_threshold:           位移阈值，与上次记录位置相比超过此值才产生移动事件
        closure_max_depth:            闭包检索最大深度
        transitive_reduction:         是否执行传递规约（每次插入时）
    """
    enabled: bool = True
    
    # Neo4j配置
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # 节点元数据存储目录（相对于output_dir）
    node_meta_dir: str = "dag_nodes"
    
    # 事件开关（用于消融实验）
    enable_entity_state: bool = True       # 实体状态节点
    enable_entity_appeared: bool = True
    enable_entity_moved: bool = True
    enable_relation: bool = True
    enable_attribute_changed: bool = True
    enable_interaction: bool = True        # 缓冲区级
    enable_occlusion: bool = True          # 缓冲区级
    enable_entity_disappeared: bool = True
    enable_periodic_description: bool = True  # 缓冲区级
    
    # 位移检测阈值（与上次记录位置比较）
    movement_threshold: float = 50.0
    
    # 关系去抖动帧数（关系消失后等待N帧确认）
    relation_debounce_frames: int = 3
    
    # 闭包检索配置
    closure_max_depth: int = 10
    closure_top_k_seeds: int = 5
    
    # 传递规约
    transitive_reduction: bool = True


@dataclass
class STGConfig:
    """顶层配置——聚合所有子配置。

    主要字段：
        output_dir:             输出根目录（entity_registry.json、stg_graph.json 等存放于此）
        store_dir:              向量存储目录（FAISS 索引 + 元数据），默认为 output_dir/store
        clear_existing_sample:  build 时是否清除同名 sample 的旧数据
        dag:                    DAG相关配置（事件开关、Neo4j等）
    """
    output_dir: str = "./outputs"
    store_dir: Optional[str] = None
    clear_existing_sample: bool = True
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    matching: EntityMatchingConfig = field(default_factory=EntityMatchingConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    dag: DAGConfig = field(default_factory=DAGConfig)

    def __post_init__(self) -> None:
        # 未显式指定 store_dir 时，默认放到 output_dir/store。
        if self.store_dir is None:
            self.store_dir = str(Path(self.output_dir) / "store")

    @property
    def output_path(self) -> Path:
        # 提供 Path 形式输出目录，方便上层统一路径操作。
        return Path(self.output_dir)

    @property
    def store_path(self) -> Path:
        # 提供 Path 形式向量存储目录。
        return Path(self.store_dir)
    
    @property
    def dag_node_meta_path(self) -> Path:
        # 提供DAG节点元数据存储目录。
        return Path(self.output_dir) / self.dag.node_meta_dir
