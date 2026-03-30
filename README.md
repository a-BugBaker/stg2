# STG Reference v2

[中文](README_CN.md) | English

STG Reference v2 is a runnable prototype of Spatio-Temporal Graph Memory (STG) for long-video, memory-first QA pipelines.

This project focuses on the memory system itself, not end-to-end video model training:

- Input: per-frame scene graph JSON.
- Build: persistent entity memories, event memories, and DAG state.
- Retrieve: structured evidence bundles for downstream reasoning.
- Output: machine-readable artifacts for debugging, analysis, and LLM grounding.

## What This Project Actually Contains

The core value is in the [stg](stg) package:

- [stg/memory_manager.py](stg/memory_manager.py): top-level orchestration via STGraphMemory.
- [stg/schema.py](stg/schema.py): strict scene-graph validation and normalization.
- [stg/entity_tracker.py](stg/entity_tracker.py): cross-frame tracking with IoU + label similarity + Hungarian matching.
- [stg/dag_manager.py](stg/dag_manager.py): DAG node/edge management and transitive reduction.
- [stg/closure_retrieval.py](stg/closure_retrieval.py): closure-based retrieval from seed nodes to causal context.
- [stg/vector_store.py](stg/vector_store.py): persistent vector partitions with optional FAISS acceleration.
- [stg/config.py](stg/config.py): modular configuration for embedding, matching, buffer, search, and DAG toggles.

## End-to-End Pipeline

### 1. Build (offline)

- Normalize raw scene graph payloads.
- Associate entities across frames and manage lifecycle: active / inactive / disappeared.
- Generate immediate events and buffered events.
- Write vectors to event/entity partitions.
- Persist registry, graph, and DAG artifacts.

### 2. Retrieve (online)

- Parse user query into structured hints.
- Retrieve from DAG closure path (when DAG enabled).
- Format a stable evidence bundle for QA/LLM consumption.

## Project Highlights

- Causal memory graph: a DAG representation with transitive reduction keeps compact causal structure.
- Closure retrieval: not only nearest hits, but also required historical ancestors for context completeness.
- Robust fallback design: sentence-transformers and FAISS can degrade to lightweight local paths.
- Clear artifact system: every build exports inspectable files for debugging and reproducibility.
- Modular architecture: each stage can be tuned or ablated through config switches.

## Repository Map

```text
stg_reference_v2/
├── stg/                    # Core memory engine
├── scripts/                # Build/demo/testing helpers
├── data/                   # Example scene-graph inputs
├── outputs/                # Build artifacts
├── test/                   # End-to-end and module tests
├── docs/                   # Design and test notes
└── telemem-main/           # Integrated TeleMem-related code
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Build STG memory from a sample scene graph:

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

Or run the minimal demo:

```bash
python scripts/demo_minimal.py
```

## Main Outputs

After build, key artifacts are under outputs/sample_id:

- entity_registry.json: persistent entity states and lifecycle history.
- stg_graph.json: graph-oriented export for analysis.
- dag_state.json: serialized DAG runtime state.
- debug_entity_matches.json: optional cross-frame matching debug records.
- outputs/store/sample_id/: vector partitions for event/entity memories.

## Programmatic Usage

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

## Status

This repository is a strong research-engineering prototype for memory-centric video understanding workflows, with emphasis on inspectability, modularity, and grounded retrieval.
