# STG Reference v2

中文 | [English](README.md)

STG Reference v2 是一个可运行的时空图记忆（Spatio-Temporal Graph Memory, STG）原型，面向长视频的 memory-first 问答流程。

这个项目重点不是端到端视频模型训练，而是可落地的记忆系统：

- 输入：逐帧 scene graph JSON。
- 构建：持久化实体记忆、事件记忆和 DAG 状态。
- 检索：返回结构化证据包，供下游推理使用。
- 输出：可机器读取、可调试、可分析的产物文件。

## 项目核心内容

真正的核心价值集中在 [stg](stg) 目录：

- [stg/memory_manager.py](stg/memory_manager.py)：STGraphMemory 总控编排。
- [stg/schema.py](stg/schema.py)：scene graph 严格校验与归一化。
- [stg/entity_tracker.py](stg/entity_tracker.py)：IoU + 标签相似度 + 匈牙利匹配的跨帧跟踪。
- [stg/dag_manager.py](stg/dag_manager.py)：DAG 节点/边管理与传递规约。
- [stg/closure_retrieval.py](stg/closure_retrieval.py)：从种子节点到祖先闭包的因果检索。
- [stg/vector_store.py](stg/vector_store.py)：支持可选 FAISS 的持久化向量分区存储。
- [stg/config.py](stg/config.py)：embedding、匹配、缓冲、检索、DAG 的模块化配置体系。

## 端到端流程

### 1. Build（离线构建）

- 归一化原始 scene graph 输入。
- 进行跨帧实体关联，并维护生命周期：active / inactive / disappeared。
- 生成即时事件与缓冲区事件。
- 将事件/实体状态写入向量分区。
- 持久化导出注册表、图结构和 DAG 状态。

### 2. Retrieve（在线检索）

- 将自然语言问题解析为结构化查询提示。
- 在启用 DAG 时走闭包检索路径。
- 输出稳定、结构化的 evidence bundle，便于 QA/LLM 使用。

## 项目亮点

- 因果记忆图：采用 DAG 表达事件依赖，并通过传递规约保持最小因果骨架。
- 闭包检索：不仅返回最相似节点，还补齐必要历史祖先，减少上下文断裂。
- 稳健降级：缺少 sentence-transformers 或 FAISS 时可退化到轻量本地方案。
- 产物可观测：每次构建都会导出可检查文件，便于定位问题和复现实验。
- 架构可消融：通过配置开关可按模块做实验和能力对比。

## 仓库结构

```text
stg_reference_v2/
├── stg/                    # 核心记忆引擎
├── scripts/                # 构建/演示/测试辅助脚本
├── data/                   # 示例 scene-graph 输入
├── outputs/                # 构建产物输出目录
├── test/                   # 端到端与模块测试
├── docs/                   # 设计说明与测试文档
└── telemem-main/           # 集成的 TeleMem 相关代码
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

用样例 scene graph 构建 STG 记忆：

```bash
python scripts/build_stg.py \
  --scene_graph_path data/less_move_2frames.json \
  --sample_id less_move_2frames_debug \
  --output_dir outputs \
  --embedding_backend sentence_transformers \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --neo4j_store_content \
  --clear_neo4j_sample_before_build \
  --export_match_debug
```

或者直接运行最小演示：

```bash
python scripts/demo_minimal.py
```

## 主要输出文件

构建完成后，核心产物位于 outputs/sample_id 下：

- entity_registry.json：实体状态与生命周期历史。
- stg_graph.json：用于分析的图结构导出。
- dag_state.json：DAG 运行态序列化结果。
- debug_entity_matches.json：可选的跨帧匹配调试记录。
- outputs/store/sample_id/：事件与实体记忆向量分区文件。

## 代码调用示例

```python
from stg import STGConfig, STGraphMemory

config = STGConfig(output_dir="./outputs")
config.embedding.backend = "hashing"

stg = STGraphMemory(config)
stg.build("data/less_move_2frames.json", sample_id="demo_sample")

bundle = stg.retrieve_evidence(
    query="What happened around man3?",
    sample_id="demo_sample",
)

llm_evidence = stg.format_evidence_for_llm(bundle)
prompt_parts = stg.build_grounded_prompt("What happened around man3?", llm_evidence)
```

## 当前定位

该仓库目前是一个偏研究工程化的高可读原型，强调可解释、可扩展、可调试的记忆检索流程。
