#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure `python scripts/*.py` can import the sibling `stg` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stg import STGConfig, STGraphMemory
from stg.llm_adapter import LLMAdapterError, OpenAICompatibleLLMAdapter


def build_stg(output_dir: str, embedding_backend: str, embedding_model: str, embedding_dim: int, allow_fallback: bool, top_k: int) -> STGraphMemory:
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
    config.search.top_k = top_k
    return STGraphMemory(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LLM integration chain on a built sample.")
    parser.add_argument("--sample_id", default="video_001_first10", help="Built sample id.")
    parser.add_argument("--query", default="man1发生了什么", help="Query used for retrieval and QA.")
    parser.add_argument("--output_dir", default="outputs_retest", help="Build output directory.")
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

    # LLM call options: by default this script stays in dry-run mode.
    parser.add_argument("--call_llm", action="store_true", help="Actually call LLM API. If absent, dry-run only.")
    parser.add_argument("--api_base", default="", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api_key", default="", help="OpenAI-compatible API key.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name.")

    parser.add_argument(
        "--save_json",
        default="outputs_retest/video_001_first10/llm_integration_test_result.json",
        help="Path to save test payload JSON.",
    )
    args = parser.parse_args()

    stg = build_stg(
        output_dir=args.output_dir,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        allow_fallback=args.allow_neo4j_fallback,
        top_k=args.top_k,
    )

    bundle = stg.retrieve_evidence(query=args.query, sample_id=args.sample_id, top_k=args.top_k)
    llm_evidence = stg.format_evidence_for_llm(bundle)
    prompts = stg.build_grounded_prompt(args.query, llm_evidence)

    payload = {
        "mode": "llm_call" if args.call_llm else "dry_run",
        "sample_id": args.sample_id,
        "query": args.query,
        "summary_stats": bundle.get("summary_stats", {}),
        "closure_stats": bundle.get("closure_stats", {}),
        "evidence": llm_evidence,
        "system_prompt": prompts["system_prompt"],
        "user_prompt": prompts["user_prompt"],
    }

    if args.call_llm:
        if not args.api_base or not args.api_key:
            raise SystemExit("--call_llm requires both --api_base and --api_key")
        adapter = OpenAICompatibleLLMAdapter(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
        )
        try:
            answer = adapter.answer(prompts, llm_evidence)
        except LLMAdapterError as exc:
            raise SystemExit(f"LLM adapter error: {exc}") from exc
        payload["model"] = args.model
        payload["answer"] = answer

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== LLM Integration Test ===")
    print(f"mode: {payload['mode']}")
    print(f"sample_id: {args.sample_id}")
    print(f"query: {args.query}")
    print(f"event_evidence: {payload['summary_stats'].get('num_event_evidence', 0)}")
    print(f"entity_evidence: {payload['summary_stats'].get('num_entity_evidence', 0)}")
    print(f"closure_size: {payload.get('closure_stats', {}).get('closure_size', 0)}")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
