#!/usr/bin/env python3
"""
在线检索脚本

对已构建的 STG 记忆执行自然语言检索，返回结构化证据（evidence bundle）。

用法：
    cd stg_reference_v2
    PYTHONPATH=. python scripts/query_stg.py \\
        --sample_id toy_video \\
        --query "What happened to the basketball?" \\
        --output_dir ./outputs \\
        --top_k 8

参数说明：
    --sample_id:          之前 build 时使用的 sample_id
    --query:              自然语言问题
    --output_dir:         与 build 时相同的输出目录
    --top_k:              返回的事件证据条数
    --embedding_backend:  嵌入后端（需与 build 时一致）
    --json:               加上此标志后输出完整 JSON 而非人类可读文本

输出：
    默认输出人类可读的 evidence_text，加 --json 后输出完整的 evidence bundle JSON。
"""
from __future__ import annotations

import argparse
import json

from stg import STGConfig, STGraphMemory


def main() -> None:
    """命令行入口：执行检索并输出证据文本或 JSON。"""
    # 1) 解析命令行参数。
    parser = argparse.ArgumentParser(description="Query an STG memory and return structured evidence.")
    parser.add_argument("--sample_id", required=True, help="Sample/video id.")
    parser.add_argument("--query", required=True, help="Natural language question.")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory.")
    parser.add_argument("--top_k", type=int, default=8, help="Top-k retrieval count.")
    parser.add_argument(
        "--embedding_backend",
        default="auto",
        choices=["auto", "sentence_transformers", "hashing"],
        help="Embedding backend.",
    )
    parser.add_argument("--json", action="store_true", help="Print structured evidence JSON.")
    args = parser.parse_args()

    # 2) 初始化配置与 STG 主控。
    config = STGConfig(output_dir=args.output_dir)
    config.embedding.backend = args.embedding_backend
    config.search.top_k = args.top_k
    stg = STGraphMemory(config)

    # 3) 执行检索并按输出模式渲染结果。
    bundle = stg.retrieve_evidence(query=args.query, sample_id=args.sample_id, top_k=args.top_k)
    if args.json:
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
    else:
        print(bundle["evidence_text"])


if __name__ == "__main__":
    main()
