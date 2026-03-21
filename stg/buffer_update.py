"""
缓冲区更新模块（Buffer Update）

本模块实现了帧级观测数据的缓冲与定期刷新机制。
与 ImmediateUpdater 逐帧生成事件不同，BufferUpdater 积累若干帧后一次性分析：

工作流程：
    1. observe(frame_observations): 每帧调用，将当前帧的所有活跃实体观测加入缓冲区
    2. 当缓冲区满（达到 buffer_size 帧）后返回 True，由外层调用 flush()
    3. flush(sample_id):
       a. 将缓冲区内所有观测按 entity_id 分组组织轨迹
       b. 对每个实体调用 MotionAnalyzer.analyze_single_entity() 生成轨迹摘要
       c. 调用 MotionAnalyzer.analyze_all_interactions() 分析实体间交互
       d. 将生成的 trajectory_summary 和 interaction 事件向量化并写入 VectorStore
       e. 如果启用DAG，生成 interaction/occlusion/periodic_description 节点
       f. 清空缓冲区
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from .config import STGConfig
from .event_generator import EventGenerator
from .motion_analyzer import MotionAnalyzer
from .utils import EmbeddingManager
from .vector_store import VectorStore

if TYPE_CHECKING:
    from .dag_manager import DAGManager
    from .dag_event_generator import DAGEventGenerator

logger = logging.getLogger(__name__)


class BufferUpdater:
    def __init__(
        self,
        config: STGConfig,
        motion_analyzer: MotionAnalyzer,
        event_generator: EventGenerator,
        embedder: EmbeddingManager,
        store: VectorStore,
    ):
        """初始化缓冲更新器及其运动分析依赖。"""
        self.config = config
        self.motion_analyzer = motion_analyzer
        self.event_generator = event_generator
        self.embedder = embedder
        self.store = store
        self.buffer: List[List[Dict[str, Any]]] = []
        self._buffer_frame_start: int = 0  # 当前缓冲区起始帧
        
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
        """清空缓冲区状态。"""
        self.buffer.clear()
        self._buffer_frame_start = 0

    def observe(self, frame_observations: Sequence[Dict[str, Any]]) -> bool:
        """写入一帧观测并返回是否达到 flush 条件。"""
        # 记录缓冲区起始帧
        if not self.buffer and frame_observations:
            self._buffer_frame_start = frame_observations[0].get("frame_index", 0)
        # 按帧追加观测，维持时间顺序。
        self.buffer.append(list(frame_observations))
        # 达到 buffer_size 时由外层触发 flush。
        return len(self.buffer) >= self.config.buffer.buffer_size

    def _write_event(self, sample_id: str, event: Dict[str, Any]) -> None:
        """将缓冲阶段事件写入向量存储。"""
        vector = self.embedder.embed(event["summary"])
        self.store.add(sample_id, "events", vector, event)

    def flush(self, sample_id: str) -> None:
        """刷新缓冲区并生成轨迹摘要与交互事件。

        该函数是 buffer 阶段的核心入口：分组轨迹 -> 运动分析 -> 事件写入 -> 清空缓冲。
        """
        # 空缓冲直接返回，避免产生空事件。
        if not self.buffer:
            return
        
        # 计算缓冲区帧范围
        frame_start = self._buffer_frame_start
        frame_end = frame_start + len(self.buffer) - 1

        # 1) 将多帧观测按 entity_id 聚合为轨迹，并提取实体基本信息。
        trajectories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        entity_info: Dict[str, Dict[str, Any]] = {}
        involved_entities: List[str] = []
        
        for frame_observations in self.buffer:
            for obs in frame_observations:
                entity_id = obs["entity_id"]
                trajectories[entity_id].append(obs)
                entity_info[entity_id] = {
                    "entity_id": entity_id,
                    "tag": obs["tag"],
                    "label": obs["label"],
                }
                if obs["tag"] not in involved_entities:
                    involved_entities.append(obs["tag"])

        # 2) 对每个实体轨迹做单体运动分析，生成 trajectory_summary 事件。
        for entity_id, trajectory in trajectories.items():
            analysis = self.motion_analyzer.analyze_single_entity(entity_info[entity_id], trajectory)
            if analysis is None:
                continue
            event = self.event_generator.gen_trajectory_summary(analysis)
            self._write_event(sample_id, event)

        # 3) 在实体两两组合上做交互分析，生成 interaction 事件。
        interaction_events = self.motion_analyzer.analyze_all_interactions(trajectories, entity_info)
        for interaction in interaction_events:
            event = self.event_generator.gen_interaction(interaction)
            self._write_event(sample_id, event)
            # DAG: 创建交互事件节点
            self._dag_process_interaction(interaction, frame_start, frame_end)

        # 4) DAG: 生成阶段性描述节点
        self._dag_generate_periodic_description(frame_start, frame_end, involved_entities, entity_info)

        # 5) 刷新完成后清空缓冲，进入下一轮积累。
        self.buffer.clear()
        self._buffer_frame_start = frame_end + 1
    
    # ==================== DAG辅助方法 ====================
    
    def _dag_process_interaction(
        self,
        interaction: Dict[str, Any],
        frame_start: int,
        frame_end: int
    ) -> None:
        """DAG: 处理交互事件。"""
        if not self.dag_event_generator:
            return
        
        interaction_type = interaction.get("interaction_type", "unknown")
        entity1_tag = interaction.get("entity1_tag", "")
        entity2_tag = interaction.get("entity2_tag", "")
        
        if entity1_tag and entity2_tag:
            self.dag_event_generator.create_interaction_event(
                entity1_tag=entity1_tag,
                entity2_tag=entity2_tag,
                interaction_type=interaction_type,
                frame_start=frame_start,
                frame_end=frame_end,
                details=interaction
            )
    
    def _dag_generate_periodic_description(
        self,
        frame_start: int,
        frame_end: int,
        involved_entities: List[str],
        entity_info: Dict[str, Dict[str, Any]]
    ) -> None:
        """DAG: 生成阶段性描述节点。"""
        if not self.dag_event_generator:
            return
        
        if not involved_entities:
            return
        
        # 构建简单的场景描述（基于规则）
        num_entities = len(involved_entities)
        entity_labels = list(set(info.get("label", "object") for info in entity_info.values()))
        
        description = f"Scene contains {num_entities} entities ({', '.join(entity_labels[:3])}"
        if len(entity_labels) > 3:
            description += f" and {len(entity_labels) - 3} more types"
        description += f") observed over frames {frame_start}-{frame_end}."
        
        self.dag_event_generator.create_periodic_description(
            frame_start=frame_start,
            frame_end=frame_end,
            involved_entities=involved_entities,
            description=description,
            description_type="scene"
        )
