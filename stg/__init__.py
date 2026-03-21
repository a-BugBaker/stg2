"""
stg 包入口模块

本包实现了 STG（Spatio-Temporal Graph Memory，时空图记忆）系统的全部核心功能。
系统从逐帧场景图 JSON 出发，离线构建持久化的实体记忆和事件记忆，
在线阶段通过向量检索和启发式重排序返回结构化证据，供下游 LLM 问答使用。

DAG重构后新增：
    - DAGNode, EventType, LogicalClock: DAG核心数据结构
    - DAGManager: DAG管理器，负责节点/边操作和传递规约
    - DAGEventGenerator: DAG事件生成器
    - ClosureRetriever: 闭包检索器

对外暴露的核心接口：
    - STGConfig: 全局配置数据类，包含嵌入、匹配、轨迹、缓冲、运动、检索、DAG等所有超参数
    - STGraphMemory: 主控类，封装 build / retrieve_evidence / format_evidence_for_llm 等完整流程
    - DAGManager: DAG管理器
    - ClosureRetriever: 闭包检索器
"""

from .config import STGConfig
from .memory_manager import STGraphMemory

# DAG相关模块
from .dag_core import DAGNode, EventType, LogicalClock, LogicalClockManager
from .dag_manager import DAGManager
from .dag_event_generator import DAGEventGenerator
from .closure_retrieval import ClosureRetriever

__all__ = [
    "STGConfig",
    "STGraphMemory",
    # DAG相关
    "DAGNode",
    "EventType", 
    "LogicalClock",
    "LogicalClockManager",
    "DAGManager",
    "DAGEventGenerator",
    "ClosureRetriever",
]

print("STG package loaded. Available classes: STGConfig, STGraphMemory, DAGManager, ClosureRetriever")