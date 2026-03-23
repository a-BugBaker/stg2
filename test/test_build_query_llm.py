"""
构建-检索-LLM接入集成测试（本地可离线运行）

目标：
1. 使用轻量 hashing 嵌入构建 STG（DAG 全开）。
2. 执行查询并验证 evidence bundle。
3. 验证 LLM 接口接入链路（不依赖外部 API）：
   - prompt 构建
   - 证据不足短路返回
   - JSON 解析逻辑
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from stg import STGConfig, STGraphMemory
from stg.llm_adapter import OpenAICompatibleLLMAdapter


def _write_minimal_scene_graph(path: Path) -> None:
    payload = {
        "frames": [
            {
                "frame_index": 0,
                "image_path": "frame_000000.png",
                "objects": [
                    {
                        "tag": "man1",
                        "label": "man",
                        "bbox": [10, 20, 110, 220],
                        "score": 0.98,
                        "attributes": ["standing", "blue_shirt"],
                        "relations": [{"name": "near", "object": "shoe1"}],
                    },
                    {
                        "tag": "shoe1",
                        "label": "shoe",
                        "bbox": [20, 190, 70, 230],
                        "score": 0.97,
                        "attributes": ["black"],
                        "relations": [],
                    },
                ],
            },
            {
                "frame_index": 1,
                "image_path": "frame_000001.png",
                "objects": [
                    {
                        "tag": "man1",
                        "label": "man",
                        "bbox": [80, 20, 180, 220],
                        "score": 0.98,
                        "attributes": ["standing", "blue_shirt"],
                        "relations": [{"name": "near", "object": "shoe1"}],
                    },
                    {
                        "tag": "shoe1",
                        "label": "shoe",
                        "bbox": [90, 190, 140, 230],
                        "score": 0.97,
                        "attributes": ["black"],
                        "relations": [],
                    },
                ],
            },
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _full_on_config(output_dir: Path) -> STGConfig:
    cfg = STGConfig(output_dir=str(output_dir))
    cfg.embedding.backend = "hashing"
    cfg.embedding.dim = 256
    cfg.buffer.buffer_size = 2
    cfg.matching.movement_event_threshold = 5.0
    cfg.dag.movement_threshold = 5.0

    # 全功能开启 + DAG 开启
    cfg.dag.enabled = True
    cfg.dag.enable_entity_state = True
    cfg.dag.enable_entity_appeared = True
    cfg.dag.enable_entity_moved = True
    cfg.dag.enable_relation = True
    cfg.dag.enable_attribute_changed = True
    cfg.dag.enable_interaction = True
    cfg.dag.enable_occlusion = True
    cfg.dag.enable_entity_disappeared = True
    cfg.dag.enable_periodic_description = True
    return cfg


def run_integration_test() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        scene_path = root / "mini_scene.json"
        _write_minimal_scene_graph(scene_path)

        config = _full_on_config(root / "outputs")
        stg = STGraphMemory(config)

        sample_id = "mini_video"
        stats = stg.build(scene_graph_path=scene_path, sample_id=sample_id)

        assert stats["num_frames"] == 2
        assert stats["num_entities"] >= 2
        assert stats.get("num_dag_nodes", 0) > 0

        bundle = stg.retrieve_evidence("What happened to man1 and shoe1?", sample_id=sample_id, top_k=6)
        assert "evidence_text" in bundle
        assert isinstance(bundle["events"], list)

        llm_evidence = stg.format_evidence_for_llm(bundle)
        prompts = stg.build_grounded_prompt("What happened to man1 and shoe1?", llm_evidence)
        assert "system_prompt" in prompts and "user_prompt" in prompts

        # 不调用外部 API，验证证据不足短路逻辑
        adapter = OpenAICompatibleLLMAdapter(api_base="", api_key="", model="dummy-model")
        short_circuit = adapter.answer(prompts, {"events": [], "entities": []})
        assert short_circuit["sufficient_evidence"] is False

        # 验证 JSON 解析逻辑
        parsed = adapter._parse_response(
            '{"answer":"man1 moved","sufficient_evidence":true,"used_event_ids":["e1"],"used_entity_ids":["entity_0001"],"short_rationale":"supported by event"}'
        )
        assert parsed["sufficient_evidence"] is True
        assert parsed["used_event_ids"] == ["e1"]


if __name__ == "__main__":
    run_integration_test()
    print("test_build_query_llm passed")
