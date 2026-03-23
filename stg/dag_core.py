"""
DAG核心数据结构模块

本模块定义了STG系统DAG（有向无环图）的核心数据结构：
    - LogicalClock: 逻辑时钟，用于节点的因果排序
    - DAGNode: DAG节点，存储记忆单元
    - EventType: 事件类型枚举

设计原则：
    - τ使用复合值 (frame_idx, seq)，保证后一帧所有节点 > 前一帧所有节点
    - 节点内容与图拓扑解耦，支持独立更新
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EventType(Enum):
    """事件类型枚举。
    
    按修改要求定义的8种节点/事件类型：
        - ENTITY_STATE: 实体状态节点（每个实体一个）
        - ENTITY_APPEARED: 实体首次出现事件
        - ENTITY_MOVED: 实体位移事件
        - RELATION: 关系事件
        - ATTRIBUTE_CHANGED: 属性变化事件
        - INTERACTION: 缓冲区级两实体交互事件（接近/远离/同向移动）
        - OCCLUSION: 缓冲区级实体遮挡事件
        - ENTITY_DISAPPEARED: 实体消失事件
        - PERIODIC_DESCRIPTION: 缓冲区级阶段性描述
    """
    ENTITY_STATE = "entity_state"
    ENTITY_APPEARED = "entity_appeared"
    ENTITY_MOVED = "entity_moved"
    RELATION = "relation"
    ATTRIBUTE_CHANGED = "attribute_changed"
    INTERACTION = "interaction"
    OCCLUSION = "occlusion"
    ENTITY_DISAPPEARED = "entity_disappeared"
    PERIODIC_DESCRIPTION = "periodic_description"


@dataclass(frozen=True, order=True)
class LogicalClock:
    """逻辑时钟，用于节点的因果排序。
    
    采用复合值 (frame_idx, seq)：
        - frame_idx: 帧索引，保证后一帧所有节点 > 前一帧所有节点
        - seq: 同一帧内的序列号，用于区分同帧内多个节点的顺序
    
    比较规则：先比较frame_idx，相同则比较seq
    
    Attributes:
        frame_idx: 帧索引
        seq: 帧内序列号
    
    Example:
        >>> t1 = LogicalClock(10, 0)
        >>> t2 = LogicalClock(10, 1)
        >>> t3 = LogicalClock(11, 0)
        >>> t1 < t2 < t3
        True
    """
    frame_idx: int
    seq: int = 0
    
    def to_tuple(self) -> Tuple[int, int]:
        """转换为元组，用于序列化。"""
        return (self.frame_idx, self.seq)
    
    @classmethod
    def from_tuple(cls, t: Tuple[int, int]) -> LogicalClock:
        """从元组创建，用于反序列化。"""
        return cls(frame_idx=t[0], seq=t[1])
    
    def __str__(self) -> str:
        return f"τ({self.frame_idx}, {self.seq})"
    
    def __repr__(self) -> str:
        return f"LogicalClock(frame_idx={self.frame_idx}, seq={self.seq})"


@dataclass
class DAGNode:
    """DAG节点，存储记忆单元。
    
    每个节点包含：
        - content: 自然语言描述（语义内容）
        - embedding: 嵌入向量（用于相似度检索）
        - tau: 逻辑时钟（因果排序）
        - metadata: 额外元信息
    
    节点类型：
        - entity_state: 实体状态（每个实体一个，轨迹更新时τ不变）
        - 各类事件节点
    
    Attributes:
        node_id: 唯一标识符（UUID）
        node_type: 节点类型（EventType枚举值）
        content: 自然语言描述
        tau: 逻辑时钟
        embedding: 嵌入向量（可选，延迟计算）
        metadata: 额外元信息字典
        is_tombstone: 软删除标记
        parent_ids: 父节点ID列表（入边来源）
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: EventType = EventType.ENTITY_STATE
    content: str = ""
    tau: LogicalClock = field(default_factory=lambda: LogicalClock(0, 0))
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_tombstone: bool = False
    parent_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """确保node_type是EventType枚举。"""
        if isinstance(self.node_type, str):
            self.node_type = EventType(self.node_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，用于JSON存储。
        
        注意：embedding不包含在序列化中，需要单独存储到FAISS。
        """
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": self.content,
            "tau": self.tau.to_tuple(),
            "metadata": self.metadata,
            "is_tombstone": self.is_tombstone,
            "parent_ids": self.parent_ids,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DAGNode:
        """从字典反序列化。"""
        return cls(
            node_id=d["node_id"],
            node_type=EventType(d["node_type"]),
            content=d["content"],
            tau=LogicalClock.from_tuple(tuple(d["tau"])),
            metadata=d.get("metadata", {}),
            is_tombstone=d.get("is_tombstone", False),
            parent_ids=d.get("parent_ids", []),
        )
    
    def update_content(self, new_content: str) -> None:
        """更新节点内容（τ保持不变）。
        
        用于entity_state节点的轨迹更新。
        """
        self.content = new_content
        # 注意：更新内容后需要重新计算embedding
        self.embedding = None
    
    @property
    def entity_tag(self) -> Optional[str]:
        """获取关联的实体标签（如果有）。"""
        return self.metadata.get("entity_tag")
    
    @property
    def frame_start(self) -> int:
        """获取起始帧。"""
        return self.metadata.get("frame_start", self.tau.frame_idx)
    
    @property
    def frame_end(self) -> int:
        """获取结束帧。"""
        return self.metadata.get("frame_end", self.tau.frame_idx)
    
    @property
    def event_type(self) -> EventType:
        """node_type的别名，兼容不同调用方式。"""
        return self.node_type
    
    @property
    def parents(self) -> List[str]:
        """parent_ids的别名，兼容不同调用方式。"""
        return self.parent_ids
    
    def add_parent(self, parent_id: str) -> None:
        """添加父节点。"""
        if parent_id not in self.parent_ids:
            self.parent_ids.append(parent_id)
    
    def remove_parent(self, parent_id: str) -> None:
        """移除父节点。"""
        if parent_id in self.parent_ids:
            self.parent_ids.remove(parent_id)
    
    def __hash__(self) -> int:
        return hash(self.node_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DAGNode):
            return False
        return self.node_id == other.node_id


class LogicalClockManager:
    """逻辑时钟管理器，负责生成和维护τ。
    
    确保：
        - 同一帧内的节点按创建顺序递增seq
        - 后一帧的所有节点τ > 前一帧的所有节点τ
    
    Attributes:
        _frame_seq_counters: 每帧的序列号计数器
    """
    
    def __init__(self):
        self._frame_seq_counters: Dict[int, int] = {}
    
    def create_tau(self, frame_idx: int) -> LogicalClock:
        """为指定帧创建新的逻辑时钟。
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            新的LogicalClock实例
        """
        if frame_idx not in self._frame_seq_counters:
            self._frame_seq_counters[frame_idx] = 0
        
        seq = self._frame_seq_counters[frame_idx]
        self._frame_seq_counters[frame_idx] += 1
        
        return LogicalClock(frame_idx=frame_idx, seq=seq)
    
    def create_tau_after_all(self, frame_idx: int) -> LogicalClock:
        """创建一个τ，保证大于该帧内所有已创建的τ。
        
        用于阶段性描述节点，需要τ > 缓冲区内所有其他节点。
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            新的LogicalClock实例
        """
        # 在同一frame上分配下一个seq，保证τ.frame等于该缓冲区最大帧号，
        # 且τ仍大于该frame内已创建的所有节点。
        if frame_idx not in self._frame_seq_counters:
            self._frame_seq_counters[frame_idx] = 0
        seq = self._frame_seq_counters[frame_idx]
        self._frame_seq_counters[frame_idx] += 1
        return LogicalClock(frame_idx=frame_idx, seq=seq)
    
    def get_current_seq(self, frame_idx: int) -> int:
        """获取指定帧当前的序列号。"""
        return self._frame_seq_counters.get(frame_idx, 0)
    
    def reset(self) -> None:
        """重置所有计数器。"""
        self._frame_seq_counters.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化状态。"""
        return {"frame_seq_counters": self._frame_seq_counters}
    
    def load_dict(self, d: Dict[str, Any]) -> None:
        """加载状态。"""
        self._frame_seq_counters = {
            int(k): v for k, v in d.get("frame_seq_counters", {}).items()
        }
