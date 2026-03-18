#!/usr/bin/env python3
"""
LLM 问答脚本

先检索 STG 结构化证据，再将证据传给 OpenAI 兼容 LLM 生成 grounded 答案。

用法（实际调用 LLM）：
    cd stg_reference_v2
    PYTHONPATH=. python scripts/llm_qa.py \\
        --sample_id video_001 \\
        --query "What happened to the basketball?" \\
        --output_dir ./outputs \\
        --api_base "https://dashscope.aliyuncs.com/compatible-mode/v1" \\
        --api_key "sk-78fece650be14d39af245674475a8f71" \\
        --model "qwen-plus-2025-07-14"

    $env:PYTHONPATH="."; python scripts/llm_qa.py --sample_id video_001 --query "What happened to the basketball?" --output_dir ./outputs --api_base "https://dashscope.aliyuncs.com/compatible-mode/v1" --api_key "sk-78fece650be14d39af245674475a8f71" --model "qwen-plus-2025-07-14"

用法（干跑模式，不调用 LLM，仅查看 prompt 和证据）：
    PYTHONPATH=. python scripts/llm_qa.py \\
        --sample_id video_001 \\
        --query "..." \\
        --dry_run

参数说明：
    --sample_id:          之前 build 时使用的 sample_id
    --query:              自然语言问题
    --output_dir:         与 build 时相同的输出目录
    --api_base:           OpenAI 兼容 API 端点
    --api_key:            API 密钥
    --model:              模型名称
    --top_k:              检索事件证据条数
    --embedding_backend:  嵌入后端（需与 build 时一致）
    --dry_run:            仅输出证据和 prompt，不实际调用 LLM
    --json:               以 JSON 格式输出最终答案
"""
from __future__ import annotations

import argparse
import json
import sys

from stg import STGConfig, STGraphMemory
from stg.llm_adapter import LLMAdapterError, OpenAICompatibleLLMAdapter


def main() -> None:
    """命令行入口：检索 STG 证据并执行受约束的 LLM 问答。"""
    # 1) 解析命令行参数。
    parser = argparse.ArgumentParser(
        description="Retrieve STG evidence and run grounded QA with an OpenAI-compatible LLM."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--api_base", default="", help="OpenAI-compatible API base.")
    parser.add_argument("--api_key", default="", help="OpenAI-compatible API key.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument(
        "--embedding_backend",
        default="auto",
        choices=["auto", "sentence_transformers", "hashing"],
        help="Embedding backend.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the grounded prompt and evidence without calling an LLM.")
    parser.add_argument("--json", action="store_true", help="Print the final answer payload as JSON.")
    args = parser.parse_args()

    # 2) 初始化 STG 并检索证据。
    config = STGConfig(output_dir=args.output_dir)
    config.embedding.backend = args.embedding_backend
    config.search.top_k = args.top_k
    stg = STGraphMemory(config)

    bundle = stg.retrieve_evidence(query=args.query, sample_id=args.sample_id, top_k=args.top_k)
    llm_evidence = stg.format_evidence_for_llm(bundle)
    prompts = stg.build_grounded_prompt(args.query, llm_evidence)

    # 3) dry_run 模式仅输出提示词与证据，不调用 LLM。
    if args.dry_run:
        payload = {
            "mode": "dry_run",
            "evidence": llm_evidence,
            "system_prompt": prompts["system_prompt"],
            "user_prompt": prompts["user_prompt"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    # 4) 初始化适配器并请求 LLM。
    adapter = OpenAICompatibleLLMAdapter(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
    )

    try:
        answer = adapter.answer(prompts, llm_evidence)
    except LLMAdapterError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print("Hint: pass --dry_run to inspect the evidence and prompt without an API call.", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - external API failures
        print(f"[ERROR] LLM request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # 5) 输出最终结果（JSON 或文本模式）。
    payload = {
        "query": args.query,
        "model": args.model,
        "evidence": llm_evidence,
        "answer": answer,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("=== Evidence ===")
        print(bundle["evidence_text"])
        print()
        print("=== Grounded Answer JSON ===")
        print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
