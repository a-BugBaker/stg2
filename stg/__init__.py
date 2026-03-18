"""
stg 包入口模块

本包实现了 STG（Spatio-Temporal Graph Memory，时空图记忆）系统的全部核心功能。
系统从逐帧场景图 JSON 出发，离线构建持久化的实体记忆和事件记忆，
在线阶段通过向量检索和启发式重排序返回结构化证据，供下游 LLM 问答使用。

对外暴露两个核心接口：
    - STGConfig: 全局配置数据类，包含嵌入、匹配、轨迹、缓冲、运动、检索等所有超参数
    - STGraphMemory: 主控类，封装 build / retrieve_evidence / format_evidence_for_llm 等完整流程
"""

from .config import STGConfig
from .memory_manager import STGraphMemory

__all__ = ["STGConfig", "STGraphMemory"]
print("STG package loaded. Available classes: STGConfig, STGraphMemory")