# STG Reference v2

`stg_reference_v2` is a runnable prototype of **STG (Spatio-Temporal Graph Memory)** for long-video, memory-first question answering.

The system is not an end-to-end video model and does not train new neural networks. Its role is to:

1. take per-frame scene graph JSON as input,
2. build persistent entity memories and event memories offline,
3. export a structured STG graph and entity registry,
4. retrieve structured evidence online for downstream grounded QA,
5. optionally pass that evidence to any OpenAI-compatible LLM.

The current implementation is intended to be:

- CPU runnable,
- modular,
- robust to missing heavy dependencies,
- suitable as an algorithm-system prototype instead of only a demo script.

## System overview

The system has three main chains.

### 1. Build

Offline memory construction from scene graphs:

- scene graph validation and normalization,
- cross-frame entity association,
- lifecycle management with `active / inactive / disappeared`,
- immediate event generation,
- buffer-level summary and interaction generation,
- vectorization and storage of event memories and entity-state memories,
- export of `entity_registry.json` and `stg_graph.json`.

### 2. Query

Structured evidence retrieval:

- query normalization,
- subquery decomposition,
- entity / relation / temporal hint extraction,
- dense retrieval from event and entity memory partitions,
- symbolic / heuristic rerank,
- output of a stable evidence bundle instead of only flat text.

### 3. LLM QA

Grounded answer generation:

- convert evidence bundle into LLM-friendly JSON evidence,
- build a grounded prompt with memory IDs,
- require JSON output with citations to used event/entity memory IDs,
- allow explicit `insufficient evidence` answers.

## Directory structure

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

## Installation

```bash
cd stg_reference_v2
pip install -r requirements.txt
```

The system supports graceful fallback:

- `sentence-transformers` unavailable -> deterministic local `hashing` embeddings
- `faiss` unavailable -> NumPy-based inner-product retrieval

For a guaranteed local run, use `--embedding_backend hashing`.

## Input scene graph contract

The build step accepts either:

- a JSON object with top-level `{"frames": [...]}`, or
- a raw frame list `[...]`.

Each normalized frame must provide:

- `frame_index`
- `image_path` (optional)
- `objects`

Each object must provide at least:

- `tag`
- `label`
- `bbox` or `box`

Optional object fields:

- `score`
- `attributes`
- `relations`

Normalization rules:

- `label`, `tag`, relation names, and attributes are normalized to a stable text form
- `subject_relations`, `object_relations`, and `layer_mapping` are merged into `relations` when present
- invalid schema raises a clear validation error instead of failing silently

### Minimal input example

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

## Event and memory types

### Event types

The current system emits these event types:

- `initial_scene`
- `entity_appeared`
- `entity_disappeared`
- `entity_moved`
- `relation_changed`
- `attribute_changed`
- `trajectory_summary`
- `interaction`

Each event memory contains structured metadata including:

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

Each entity-state memory contains at least:

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

## Tracking semantics

Entity tracking keeps the existing `IoU + label similarity + Hungarian` structure and adds lifecycle handling that is usable for real memory construction.

Entity state meanings:

- `active`: matched in the current frame
- `inactive`: temporarily unmatched but still within `miss_tolerance`
- `disappeared`: unmatched for longer than `miss_tolerance`

If an entity is missed and later matched again within the allowed miss window, it is reactivated instead of being treated as a new entity.

## Minimal end-to-end demo

```bash
cd stg_reference_v2
PYTHONPATH=. python scripts/demo_minimal.py
```

This demo:

- builds STG memory from `data/toy_scene_graphs.json`,
- prints build statistics,
- runs a few example queries,
- prints the evidence text returned by the retrieval pipeline.

## Build STG memory

```bash
cd stg_reference_v2
PYTHONPATH=. python scripts/build_stg.py \
  --scene_graph_path data/toy_scene_graphs.json \
  --sample_id toy_video \
  --output_dir ./outputs \
  --embedding_backend hashing
```

Expected outputs:

- `outputs/toy_video/entity_registry.json`
- `outputs/toy_video/stg_graph.json`
- `outputs/store/toy_video/events.*`
- `outputs/store/toy_video/entities.*`

`build_stg.py` prints a summary with:

- `num_frames`
- `num_entities`
- `num_event_memories`
- `num_entity_memories`
- `num_graph_nodes`
- `num_graph_edges`

## Query structured evidence

```bash
cd stg_reference_v2
PYTHONPATH=. python scripts/query_stg.py \
  --sample_id toy_video \
  --query "What happened to the basketball?" \
  --output_dir ./outputs \
  --embedding_backend hashing \
  --json
```

Without `--json`, the script prints a compact human-readable evidence view.

With `--json`, it returns a structured evidence bundle containing:

- `query`
- `normalized_query`
- `subqueries`
- `query_hints`
- `events`
- `entities`
- `summary_stats`
- `evidence_text`
- `evidence_json`

The retrieval pipeline combines:

- query normalization,
- subquery decomposition,
- dense retrieval over both event and entity partitions,
- query-aware rerank with entity / relation / temporal hints,
- stable evidence formatting.

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

For local inspection without an API call:

```bash
cd stg_reference_v2
PYTHONPATH=. python scripts/llm_qa.py \
  --sample_id toy_video \
  --query "What happened to the basketball?" \
  --output_dir ./outputs \
  --embedding_backend hashing \
  --dry_run
```

The grounded QA script:

- retrieves structured evidence,
- formats evidence for LLM consumption,
- builds a grounded prompt,
- requires JSON output with this schema:

```json
{
  "answer": "string",
  "sufficient_evidence": true,
  "used_event_ids": ["memory_id_1"],
  "used_entity_ids": ["memory_id_2"],
  "short_rationale": "string"
}
```

If evidence is weak or missing, the model is allowed to return `sufficient_evidence=false`.

## Output artifacts

### `entity_registry.json`

Persistent entity-level state export. Each entity record includes:

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

A graph-oriented export for later analysis. It currently contains:

- entity nodes
- event nodes
- `event_to_entity_association` edges
- `temporal_entity_chain` edges
- `temporal_adjacency` edges

### `outputs/store/<sample_id>/`

Vector store artifacts for:

- `events`
- `entities`

Depending on the backend, this may include `.npy`, `.json`, and `.index` files.

## Programmatic usage

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

## Key implementation notes

- The system is a memory framework, not a training pipeline.
- Retrieval threshold and tracking threshold are intentionally separated.
- Query results are no longer only flat text; they are structured evidence bundles.
- LLM QA is grounded on evidence JSON instead of a raw concatenated debug context.
- The graph export is a real analyzable graph structure rather than just a list of event summaries.

## Current limitations

This is now a complete system prototype, but it is still a research prototype.

Known limits:

- entity association is still heuristic and appearance-unaware
- relation semantics depend on upstream scene graph quality
- retrieval rerank is symbolic / heuristic rather than learned
- buffer-level summarization is intentionally lightweight
- no benchmark / evaluation / experiment pipeline is included in this package
- no training code is included by design

## Recommended workflow

1. Validate your scene graph JSON against the documented contract.
2. Run `build_stg.py` to construct memory for each sample.
3. Inspect `entity_registry.json` and `stg_graph.json` when debugging tracking or event logic.
4. Use `query_stg.py --json` to inspect structured evidence quality.
5. Use `llm_qa.py --dry_run` before attaching a real LLM API.

