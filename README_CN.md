# STG Reference v2

`stg_reference_v2` 是一个可运行的 **STG（Spatio-Temporal Graph Memory，时空图记忆）** 原型系统，面向长视频、memory-first 的问答场景。

这个系统不是端到端视频模型，也不会训练新的神经网络。它的职责是：

1. 以逐帧 scene graph JSON 作为输入，
2. 在离线阶段构建持久化的实体记忆和事件记忆，
3. 导出结构化的 STG graph 和 entity registry，
4. 在在线阶段检索结构化 evidence，用于下游 grounded QA，
5. 在需要时把这些 evidence 传给任意 OpenAI-compatible LLM。

当前实现的目标是：

- 可在 CPU 上运行，
- 模块化清晰，
- 对缺失的重依赖具备较强的降级能力，
- 适合作为算法系统原型，而不只是一个 demo 脚本。

## 系统概览

系统主要包含三条链路。

### 1. Build

从 scene graph 离线构建 memory：

- scene graph 校验与归一化，
- 跨帧实体关联，
- 基于 `active / inactive / disappeared` 的生命周期管理，
- 即时事件生成，
- buffer 级摘要与交互事件生成，
- 对事件记忆和实体状态记忆做向量化并存储，
- 导出 `entity_registry.json` 和 `stg_graph.json`。

### 2. Query

结构化 evidence 检索：

- query 归一化，
- subquery 分解，
- entity / relation / temporal 提示词抽取，
- 在事件记忆和实体记忆分区上做 dense retrieval，
- symbolic / heuristic rerank，
- 输出稳定的 evidence bundle，而不是只输出平铺文本。

### 3. LLM QA

基于证据的答案生成：

- 将 evidence bundle 转换成适合 LLM 使用的 JSON evidence，
- 构建带 memory ID 的 grounded prompt，
- 要求模型输出 JSON，并引用使用到的事件/实体 memory ID，
- 允许模型明确返回 `insufficient evidence`。

## 目录结构

```text
stg_reference_v2/
├── README.md
├── requirements.txt
├── data/
│   └── toy_scene_graphs.json
├── experiments/
│   └── EXPERIMENT_PLAN.md
├── scripts/
│   ├── build_stg.py
│   ├── query_stg.py
│   ├── llm_qa.py
│   └── demo_minimal.py
└── stg/
    ├── __init__.py
    ├── config.py
    ├── utils.py
    ├── schema.py
    ├── entity_tracker.py
    ├── event_generator.py
    ├── motion_analyzer.py
    ├── immediate_update.py
    ├── buffer_update.py
    ├── vector_store.py
    ├── query_parser.py
    ├── evidence_formatter.py
    ├── llm_adapter.py
    └── memory_manager.py
```

## 安装

```bash
cd stg_reference_v2
pip install -r requirements.txt
```

系统支持平滑降级：

- `sentence-transformers` 不可用时 -> 使用本地确定性的 `hashing` embedding
- `faiss` 不可用时 -> 使用基于 NumPy 的内积检索

如果你想保证完全本地可运行，建议使用 `--embedding_backend hashing`。

## 输入 scene graph 契约

build 阶段接受两种输入形式：

- 顶层为 `{"frames": [...]}` 的 JSON 对象，
- 或直接是 frame list `[...]`。

每个归一化后的 frame 至少需要包含：

- `frame_index`
- `image_path`（可选）
- `objects`

每个 object 至少需要包含：

- `tag`
- `label`
- `bbox` 或 `box`

可选字段：

- `score`
- `attributes`
- `relations`

归一化规则：

- `label`、`tag`、relation name 和 attribute 会被统一到稳定的文本形式
- 如果存在 `subject_relations`、`object_relations` 和 `layer_mapping`，会被合并进 `relations`
- 如果 schema 非法，会抛出清晰的校验错误，而不是静默失败

### 最小输入示例

```json
{
  "frames": [
    {
      "frame_index": 0,
      "image_path": "frame_0000.jpg",
      "objects": [
        {
          "tag": "player_1",
          "label": "person",
          "bbox": [20, 120, 70, 220],
          "score": 0.98,
          "attributes": ["standing"],
          "relations": [
            {"name": "near", "object": "basketball_1"}
          ]
        }
      ]
    }
  ]
}
```

## 事件与记忆类型

### 事件类型

当前系统会生成以下事件类型：

- `initial_scene`
- `entity_appeared`
- `entity_disappeared`
- `entity_moved`
- `relation_changed`
- `attribute_changed`
- `trajectory_summary`
- `interaction`

每条事件记忆都包含结构化 metadata，至少包括：

- `memory_id`
- `memory_type="event"`
- `event_type`
- `frame_start`
- `frame_end`
- `entities`
- `entity_tags`
- `entity_labels`
- `summary`
- `confidence`
- `source`
- `details`

### Entity-state memory

每条实体状态记忆至少包含：

- `memory_id`
- `memory_type="entity_state"`
- `entity_id`
- `tag`
- `label`
- `frame_index`
- `frame_start`
- `frame_end`
- `bbox`
- `attributes`
- `relations`
- `total_displacement`
- `description`
- `confidence`
- `status`

## Tracking 语义

实体追踪保留了当前的 `IoU + label similarity + Hungarian` 主体结构，并补充了更完整的生命周期管理，以支撑真正的 memory 构建。

实体状态含义如下：

- `active`：当前帧匹配成功
- `inactive`：当前暂时未匹配，但仍处于 `miss_tolerance` 允许范围内
- `disappeared`：连续未匹配时间超过 `miss_tolerance`

如果某个实体短暂漏检后，又在允许的 miss 窗口内重新匹配成功，它会被重新激活，而不是被当成一个新实体。

## 最小端到端 demo

```bash
cd stg_reference_v2
PYTHONPATH=. python scripts/demo_minimal.py
```

这个 demo 会：

- 从 `data/toy_scene_graphs.json` 构建 STG memory，
- 打印 build 统计信息，
- 跑几个示例 query，
- 打印 retrieval pipeline 返回的 evidence 文本。

## 构建 STG memory

```bash
cd stg_reference_v2
PYTHONPATH=. python scripts/build_stg.py \
  --scene_graph_path data/toy_scene_graphs.json \
  --sample_id toy_video \
  --output_dir ./outputs \
  --embedding_backend hashing
```

预期输出：

- `outputs/toy_video/entity_registry.json`
- `outputs/toy_video/stg_graph.json`
- `outputs/store/toy_video/events.*`
- `outputs/store/toy_video/entities.*`

`build_stg.py` 会打印如下摘要信息：

- `num_frames`
- `num_entities`
- `num_event_memories`
- `num_entity_memories`
- `num_graph_nodes`
- `num_graph_edges`

## 查询结构化 evidence

```bash
cd stg_reference_v2
PYTHONPATH=. python scripts/query_stg.py \
  --sample_id toy_video \
  --query "What happened to the basketball?" \
  --output_dir ./outputs \
  --embedding_backend hashing \
  --json
```

如果不加 `--json`，脚本会输出紧凑的人类可读 evidence 视图。

如果加上 `--json`，它会返回结构化 evidence bundle，包含：

- `query`
- `normalized_query`
- `subqueries`
- `query_hints`
- `events`
- `entities`
- `summary_stats`
- `evidence_text`
- `evidence_json`

当前 retrieval pipeline 组合了：

- query 归一化，
- subquery 分解，
- 在事件和实体两个 memory partition 上做 dense retrieval，
- 基于 entity / relation / temporal hints 的 query-aware rerank，
- 稳定的 evidence formatting。

## Grounded LLM QA

```bash
cd stg_reference_v2
PYTHONPATH=. python scripts/llm_qa.py \
  --sample_id toy_video \
  --query "What happened to the basketball?" \
  --output_dir ./outputs \
  --embedding_backend hashing \
  --api_base "https://api.openai.com/v1" \
  --api_key "YOUR_KEY" \
  --model "gpt-4o-mini"
```

如果只想在本地检查 evidence 和 prompt，而不真正调用 API：

```bash
cd stg_reference_v2
PYTHONPATH=. python scripts/llm_qa.py \
  --sample_id toy_video \
  --query "What happened to the basketball?" \
  --output_dir ./outputs \
  --embedding_backend hashing \
  --dry_run
```

这个 grounded QA 脚本会：

- 检索结构化 evidence，
- 把 evidence 格式化成适合 LLM 使用的形式，
- 构建 grounded prompt，
- 要求模型输出如下 schema 的 JSON：

```json
{
  "answer": "string",
  "sufficient_evidence": true,
  "used_event_ids": ["memory_id_1"],
  "used_entity_ids": ["memory_id_2"],
  "short_rationale": "string"
}
```

如果证据不足，模型可以明确返回 `sufficient_evidence=false`。

## 输出产物

### `entity_registry.json`

持久化导出的实体级状态文件。每条实体记录包含：

- `entity_id`
- `tag`
- `label`
- `first_frame`
- `last_frame`
- `first_bbox`
- `last_bbox`
- `trajectory`
- `attributes_history`
- `relations_history`
- `status_history`
- `state`
- `missed_frames`

### `stg_graph.json`

面向后续分析的图结构导出。当前包含：

- entity nodes
- event nodes
- `event_to_entity_association` edges
- `temporal_entity_chain` edges
- `temporal_adjacency` edges

### `outputs/store/<sample_id>/`

向量存储相关产物，包含：

- `events`
- `entities`

具体文件形式取决于后端，可能包括 `.npy`、`.json` 和 `.index`。

## 代码调用方式

```python
from stg import STGConfig, STGraphMemory

config = STGConfig(output_dir="./outputs")
config.embedding.backend = "hashing"

stg = STGraphMemory(config)

stg.build("data/toy_scene_graphs.json", sample_id="toy_video")
bundle = stg.retrieve_evidence("What happened to the basketball?", sample_id="toy_video")
llm_evidence = stg.format_evidence_for_llm(bundle)
prompts = stg.build_grounded_prompt("What happened to the basketball?", llm_evidence)
```

## 实现要点

- 这个系统是 memory framework，不是训练流水线。
- retrieval threshold 和 tracking threshold 被有意分开设计。
- query 结果不再只是平铺文本，而是结构化 evidence bundle。
- LLM QA 基于 evidence JSON 做 grounding，而不是直接喂一段拼接的 debug 文本。
- graph export 是可分析的图结构，而不只是事件摘要列表。

## 当前限制

现在它已经是一个完整的系统原型，但仍然是研究型原型。

已知限制：

- entity association 仍然是启发式的，还没有 appearance-aware 能力
- relation 语义质量依赖上游 scene graph 质量
- retrieval rerank 目前是 symbolic / heuristic，而不是学习式 reranker
- buffer 级摘要逻辑刻意保持轻量
- 这个包里不包含 benchmark / evaluation / experiment pipeline
- 按设计也不包含训练代码

## 推荐使用流程

1. 先确认你的 scene graph JSON 满足文档中的输入契约。
2. 运行 `build_stg.py` 为每个 sample 构建 memory。
3. 调试 tracking 或 event logic 时，优先检查 `entity_registry.json` 和 `stg_graph.json`。
4. 用 `query_stg.py --json` 查看结构化 evidence 的质量。
5. 在接入真实 LLM API 前，先用 `llm_qa.py --dry_run` 检查 prompt 和证据组织是否合理。
