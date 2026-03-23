#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Ensure `python scripts/*.py` can import the sibling `stg` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stg import STGConfig, STGraphMemory


def parse_queries(raw: str) -> List[str]:
    queries = [q.strip() for q in raw.split("|") if q.strip()]
    return queries or ["man1发生了什么"]


def build_stg(output_dir: str, embedding_backend: str, embedding_model: str, embedding_dim: int, allow_fallback: bool) -> STGraphMemory:
    config = STGConfig(output_dir=output_dir)
    config.embedding.backend = embedding_backend
    if embedding_backend == "sentence_transformers":
        config.embedding.model_name = embedding_model
        if embedding_dim > 0:
            config.embedding.dim = embedding_dim
        elif "all-minilm-l6-v2" in embedding_model.lower():
            config.embedding.dim = 384
    elif embedding_dim > 0:
        config.embedding.dim = embedding_dim

    config.dag.allow_memory_fallback = allow_fallback
    config.dag.enabled = True
    config.dag.enable_entity_state = True
    config.dag.enable_entity_appeared = True
    config.dag.enable_entity_moved = True
    config.dag.enable_relation = True
    config.dag.enable_attribute_changed = True
    config.dag.enable_interaction = True
    config.dag.enable_occlusion = True
    config.dag.enable_entity_disappeared = True
    config.dag.enable_periodic_description = True

    return STGraphMemory(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test vector/closure retrieval quality on a built sample.")
    parser.add_argument("--sample_id", default="video_001_first10", help="Built sample id.")
    parser.add_argument("--output_dir", default="outputs_retest", help="Build output directory.")
    parser.add_argument(
        "--queries",
        default="man1发生了什么|ball发生了什么|有哪些交互事件",
        help="Multiple queries separated by '|'.",
    )
    parser.add_argument("--top_k", type=int, default=8, help="Top-k evidence count.")
    parser.add_argument(
        "--embedding_backend",
        default="hashing",
        choices=["auto", "sentence_transformers", "hashing"],
        help="Embedding backend.",
    )
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name.",
    )
    parser.add_argument("--embedding_dim", type=int, default=0, help="Embedding dimension override.")
    parser.add_argument("--allow_neo4j_fallback", action="store_true", help="Allow in-memory fallback when Neo4j is unavailable.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if any query has zero event evidence.")
    parser.add_argument("--save_json", default="", help="Optional path to save JSON report.")
    args = parser.parse_args()

    stg = build_stg(
        output_dir=args.output_dir,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        allow_fallback=args.allow_neo4j_fallback,
    )

    queries = parse_queries(args.queries)
    report = {
        "sample_id": args.sample_id,
        "queries": [],
    }
    failed = 0

    # Build closure index once and reuse for all queries.
    stg.dag_manager.set_current_sample(args.sample_id)
    indexed = stg.closure_retriever.build_index_for_sample(args.sample_id)
    print(f"[info] indexed_nodes={indexed}")

    for q in queries:
        seeds = stg.closure_retriever.identify_seeds(q, top_k=min(args.top_k, 5), similarity_threshold=0.0)
        bundle = stg.retrieve_evidence(query=q, sample_id=args.sample_id, top_k=args.top_k)
        stats = bundle.get("summary_stats", {})
        closure = bundle.get("closure_stats", {})

        event_n = int(stats.get("num_event_evidence", 0))
        entity_n = int(stats.get("num_entity_evidence", 0))
        closure_size = int(closure.get("closure_size", 0))

        print("=" * 72)
        print(f"query: {q}")
        print(f"seed_count: {len(seeds)}")
        print(f"event_evidence: {event_n}, entity_evidence: {entity_n}, closure_size: {closure_size}")
        if bundle.get("events"):
            print(f"top_event: {bundle['events'][0].get('event_type')} | {bundle['events'][0].get('summary', '')}")

        report["queries"].append(
            {
                "query": q,
                "seed_count": len(seeds),
                "seeds": [{"node_id": sid, "score": score} for sid, score in seeds],
                "summary_stats": stats,
                "closure_stats": closure,
                "top_event": bundle.get("events", [None])[0],
                "top_entity": bundle.get("entities", [None])[0],
            }
        )

        if args.strict and event_n <= 0:
            failed += 1

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {save_path}")

    if args.strict and failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
