#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure `python scripts/*.py` can import the sibling `stg` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stg import STGConfig, STGraphMemory


def main() -> None:
    """命令行入口：构建指定样本的 STG 记忆并打印产物路径。"""
    # 1) 解析命令行参数。
    parser = argparse.ArgumentParser(description="Build an STG memory from scene-graph JSON.")
    parser.add_argument("--scene_graph_path", required=True, help="Path to scene graph JSON.")
    parser.add_argument("--sample_id", required=True, help="Sample/video id.")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory.")
    parser.add_argument(
        "--embedding_backend",
        default="sentence_transformers",
        choices=["auto", "sentence_transformers", "hashing"],
        help="Embedding backend.",
    )
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name (used when backend=sentence_transformers).",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=0,
        help="Embedding dimension override. 0 means auto by backend/model.",
    )
    parser.add_argument(
        "--allow_neo4j_fallback",
        action="store_true",
        help="Allow in-memory DAG graph fallback when Neo4j is unavailable.",
    )
    parser.add_argument(
        "--neo4j_store_content",
        dest="neo4j_store_content",
        action="store_true",
        help="Store content/metadata fields in Neo4j nodes for debugging.",
    )
    parser.add_argument(
        "--neo4j_no_store_content",
        dest="neo4j_store_content",
        action="store_false",
        help="Store only graph index fields in Neo4j nodes (production-lean mode).",
    )
    parser.set_defaults(neo4j_store_content=True)
    args = parser.parse_args()

    # 2) 构建配置并初始化 STG 主控。
    config = STGConfig(output_dir=args.output_dir)
    config.embedding.backend = args.embedding_backend
    if args.embedding_backend == "sentence_transformers":
        config.embedding.model_name = args.embedding_model
        if args.embedding_dim > 0:
            config.embedding.dim = args.embedding_dim
        elif "all-minilm-l6-v2" in args.embedding_model.lower():
            config.embedding.dim = 384
    elif args.embedding_dim > 0:
        config.embedding.dim = args.embedding_dim
    config.dag.allow_memory_fallback = args.allow_neo4j_fallback
    config.dag.neo4j_store_content = args.neo4j_store_content
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
    stg = STGraphMemory(config)
    # 3) 执行构建流程并获取统计信息。
    stats = stg.build(scene_graph_path=args.scene_graph_path, sample_id=args.sample_id)

    output_root = Path(args.output_dir).resolve()
    sample_root = output_root / args.sample_id

    # 4) 打印统计信息和输出产物路径。
    print("=== Build Summary ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("=== Artifacts ===")
    print(f"entity_registry: {sample_root / 'entity_registry.json'}")
    print(f"stg_graph: {sample_root / 'stg_graph.json'}")
    print(f"vector_store_dir: {output_root / 'store' / args.sample_id}")


if __name__ == "__main__":
    main()
