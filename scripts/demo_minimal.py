#!/usr/bin/env python3
"""
最小化端到端演示脚本

使用内置的 data/toy_scene_graphs.json 进行完整的 build → query 流程演示。
强制使用 hashing 嵌入后端，确保无需 GPU 或外部模型即可运行。

用法：
    cd stg_reference_v2
    PYTHONPATH=. python scripts/demo_minimal.py

流程：
    1. 从 toy_scene_graphs.json 构建 STG 记忆
    2. 输出构建统计信息
    3. 对多个示例问题执行检索
    4. 打印每个问题的 evidence_text
"""
from __future__ import annotations

from pathlib import Path

from stg import STGConfig, STGraphMemory


def main() -> None:
    """最小化演示入口：本地构建并执行若干示例查询。"""
    # 1) 定位演示数据与样本 ID。
    root = Path(__file__).resolve().parents[1]
    scene_graph_path = root / "data" / "toy_scene_graphs.json"
    sample_id = "toy_video"

    # 2) 构建一套轻量配置（hashing 后端，方便本地快速运行）。
    config = STGConfig(output_dir=str(root / "outputs"))
    config.embedding.backend = "hashing"  # guaranteed local demo
    config.buffer.buffer_size = 3
    config.search.top_k = 6
    config.search.entity_top_k = 3
    config.search.similarity_threshold = 0.05

    # 3) 执行 build 并输出统计。
    stg = STGraphMemory(config)
    stats = stg.build(scene_graph_path=scene_graph_path, sample_id=sample_id)
    print("=== Demo Build Stats ===")
    print(stats)
    print()

    # 4) 对示例问题逐条检索并打印证据文本。
    questions = [
        "What happened to the player?",
        "What happened to the basketball?",
        "Did the player and basketball move together?",
        "Which relations changed?",
    ]
    for question in questions:
        print("=" * 100)
        print(f"Question: {question}")
        print(stg.get_context_for_qa(question, sample_id=sample_id, top_k=6))
        print()


if __name__ == "__main__":
    main()
