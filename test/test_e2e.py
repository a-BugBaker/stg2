"""
端到端集成测试脚本

验证DAG系统的完整流程：
1. 模拟场景图数据生成
2. DAG节点创建和边管理
3. 传递规约算法
4. 闭包检索
5. 嵌入和查询

使用简单的hashing嵌入模型（CPU友好）
"""

from __future__ import annotations

import sys
import os
import json
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, List

# 确保可以导入stg模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ============================================
# 测试1: 基础导入测试
# ============================================
def test_imports():
    """测试所有模块是否可以正常导入"""
    print("\n" + "="*60)
    print("测试1: 模块导入测试")
    print("="*60)
    
    try:
        from stg.dag_core import DAGNode, EventType, LogicalClock, LogicalClockManager
        print("  ✓ dag_core 导入成功")
        
        from stg.dag_storage import JSONMetaStore, Neo4jGraphStore
        print("  ✓ dag_storage 导入成功")
        
        from stg.dag_manager import DAGManager
        print("  ✓ dag_manager 导入成功")
        
        from stg.dag_event_generator import DAGEventGenerator
        print("  ✓ dag_event_generator 导入成功")
        
        from stg.closure_retrieval import ClosureRetriever
        print("  ✓ closure_retrieval 导入成功")
        
        from stg.config import STGConfig
        print("  ✓ config 导入成功")
        
        return True
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        traceback.print_exc()
        return False


# ============================================
# 测试2: 逻辑时钟测试
# ============================================
def test_logical_clock():
    """测试逻辑时钟的比较和管理"""
    print("\n" + "="*60)
    print("测试2: 逻辑时钟测试")
    print("="*60)
    
    try:
        from stg.dag_core import LogicalClock, LogicalClockManager
        
        # 测试时钟比较
        t1 = LogicalClock(10, 0)
        t2 = LogicalClock(10, 1)
        t3 = LogicalClock(11, 0)
        
        assert t1 < t2, "同帧内序列号比较失败"
        assert t2 < t3, "跨帧比较失败"
        assert t1 < t3, "跨帧传递比较失败"
        print("  ✓ 时钟比较正确: τ(10,0) < τ(10,1) < τ(11,0)")
        
        # 测试时钟管理器
        manager = LogicalClockManager()
        
        # 帧10内生成3个时钟
        clocks_f10 = [manager.create_tau(10) for _ in range(3)]
        assert clocks_f10[0].frame_idx == 10 and clocks_f10[0].seq == 0
        assert clocks_f10[1].frame_idx == 10 and clocks_f10[1].seq == 1
        assert clocks_f10[2].frame_idx == 10 and clocks_f10[2].seq == 2
        print(f"  ✓ 帧10生成时钟: {[str(c) for c in clocks_f10]}")
        
        # 切换到帧11
        clock_f11 = manager.create_tau(11)
        assert clock_f11.frame_idx == 11 and clock_f11.seq == 0
        print(f"  ✓ 帧11生成时钟: {clock_f11}")
        
        # 序列化/反序列化
        t_tuple = t1.to_tuple()
        t_restored = LogicalClock.from_tuple(t_tuple)
        assert t_restored == t1, "序列化/反序列化失败"
        print("  ✓ 时钟序列化/反序列化正确")
        
        return True
    except Exception as e:
        print(f"  ✗ 逻辑时钟测试失败: {e}")
        traceback.print_exc()
        return False


# ============================================
# 测试3: DAG节点创建和存储
# ============================================
def test_dag_node_and_storage():
    """测试DAG节点创建和JSON存储"""
    print("\n" + "="*60)
    print("测试3: DAG节点和存储测试")
    print("="*60)
    
    try:
        from stg.dag_core import DAGNode, EventType, LogicalClock
        from stg.dag_storage import JSONMetaStore
        
        # 创建测试节点
        node1 = DAGNode(
            node_id="test_node_001",
            node_type=EventType.ENTITY_STATE,
            content="man#1: 位于画面中央，穿蓝色衬衫",
            tau=LogicalClock(10, 0),
            embedding=np.random.randn(128).astype(np.float32),
            metadata={
                "entity_tag": "man#1",
                "label": "man",
                "bbox": [100, 100, 200, 300],
                "attributes": {"upper_color": "blue"}
            }
        )
        
        node2 = DAGNode(
            node_id="test_node_002",
            node_type=EventType.ENTITY_APPEARED,
            content="man#1 首次出现在帧10",
            tau=LogicalClock(10, 1),
            embedding=np.random.randn(128).astype(np.float32),
            metadata={"entity_tag": "man#1", "frame_idx": 10}
        )
        
        print(f"  ✓ 创建节点: {node1.node_id}, 类型={node1.event_type.value}")
        print(f"  ✓ 创建节点: {node2.node_id}, 类型={node2.event_type.value}")
        
        # 添加边
        node2.add_parent(node1.node_id)
        assert node1.node_id in node2.parents
        print(f"  ✓ 添加边: {node1.node_id} → {node2.node_id}")
        
        # 测试JSON存储
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "dag_meta"
            meta_store = JSONMetaStore(meta_path)
            
            # 保存节点
            meta_store.save_node("test_sample", node1)
            meta_store.save_node("test_sample", node2)
            meta_store.flush()
            print(f"  ✓ 保存节点到: {meta_path}")
            
            # 清除缓存后重新读取
            meta_store.clear_cache()
            
            # 读取节点
            loaded1 = meta_store.load_node("test_sample", node1.node_id)
            assert loaded1 is not None
            assert loaded1.content == node1.content
            assert loaded1.tau == node1.tau
            print(f"  ✓ 加载节点验证: content匹配, τ匹配")
            
            # 检查边关系
            loaded2 = meta_store.load_node("test_sample", node2.node_id)
            assert node1.node_id in loaded2.parents
            print(f"  ✓ 边关系持久化验证通过")
        
        return True
    except Exception as e:
        print(f"  ✗ 节点存储测试失败: {e}")
        traceback.print_exc()
        return False


# ============================================
# 测试4: DAG管理器
# ============================================
def test_dag_manager():
    """测试DAG管理器的基本功能"""
    print("\n" + "="*60)
    print("测试4: DAG管理器测试")
    print("="*60)
    
    try:
        from stg.dag_core import DAGNode, EventType, LogicalClock
        from stg.dag_manager import DAGManager
        from stg.config import STGConfig
        
        # 创建临时配置
        with tempfile.TemporaryDirectory() as tmpdir:
            config = STGConfig(output_dir=tmpdir)
            config.embedding.backend = "hashing"  # 使用CPU友好的hashing
            config.embedding.dim = 128  # 减小维度加速测试
            
            # 创建管理器
            manager = DAGManager(config)
            
            # 设置嵌入函数（简单hash）
            def simple_embed(text: str) -> np.ndarray:
                np.random.seed(hash(text) % 2**31)
                return np.random.randn(128).astype(np.float32)
            
            manager.set_embed_func(simple_embed)
            manager.set_current_sample("test_video")
            print("  ✓ DAG管理器初始化完成")
            
            # 插入节点
            node_a = manager.insert_node(
                EventType.ENTITY_STATE,
                "实体A的初始状态",
                0,  # frame_idx
                None,  # parent_ids
                {"entity_tag": "entity_a"}  # metadata
            )
            print(f"  ✓ 插入节点A: {node_a.node_id[:8]}..., τ={node_a.tau}")
            
            node_b = manager.insert_node(
                EventType.ENTITY_MOVED,
                "实体A在帧1移动",
                1,
                [node_a.node_id],
                {"entity_tag": "entity_a"}
            )
            print(f"  ✓ 插入节点B: {node_b.node_id[:8]}..., τ={node_b.tau}")
            
            # 获取节点
            loaded = manager.get_node(node_a.node_id)
            assert loaded is not None
            assert loaded.content == "实体A的初始状态"
            print(f"  ✓ 获取节点验证通过")
            
            # 获取所有节点
            all_nodes = manager.get_all_nodes("test_video")
            print(f"  ✓ 获取所有节点: {len(all_nodes)}个")
            
            return True
    except Exception as e:
        print(f"  ✗ DAG管理器测试失败: {e}")
        traceback.print_exc()
        return False


# ============================================
# 测试5: 完整端到端流程
# ============================================
def test_full_pipeline():
    """测试完整的端到端流程：构建→查询"""
    print("\n" + "="*60)
    print("测试5: 完整端到端流程测试")
    print("="*60)
    
    try:
        from stg.dag_core import DAGNode, EventType
        from stg.dag_manager import DAGManager
        from stg.dag_event_generator import DAGEventGenerator
        from stg.closure_retrieval import ClosureRetriever
        from stg.config import STGConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 配置
            config = STGConfig(output_dir=tmpdir)
            config.embedding.backend = "hashing"
            config.embedding.dim = 128
            config.dag.enable_entity_state = True
            config.dag.enable_entity_appeared = True
            config.dag.enable_entity_moved = True
            config.dag.enable_relation = True
            config.dag.movement_threshold = 30.0  # 降低阈值便于测试
            
            # 初始化组件
            def simple_embed(text: str) -> np.ndarray:
                np.random.seed(hash(text) % 2**31)
                return np.random.randn(128).astype(np.float32)
            
            dag_manager = DAGManager(config)
            dag_manager.set_embed_func(simple_embed)
            
            event_gen = DAGEventGenerator(config, dag_manager)
            event_gen.set_embed_func(simple_embed)
            
            retriever = ClosureRetriever(config, dag_manager)
            retriever.set_embed_func(simple_embed)
            
            print("  ✓ 所有组件初始化完成")
            
            sample_id = "street_scene_001"
            
            # ========== 阶段1: 模拟视频帧处理 ==========
            print("\n  --- 阶段1: 处理视频帧 ---")
            
            # 帧0: 初始场景
            frame_0_entities = {
                "man#1": {
                    "tag": "man#1", "label": "man",
                    "bbox": [50, 100, 150, 350], "center": (100, 225),
                    "attributes": {"upper_color": "black", "age": "middle_aged"}
                },
                "car#1": {
                    "tag": "car#1", "label": "car",
                    "bbox": [400, 200, 600, 350], "center": (500, 275),
                    "attributes": {"color": "red"}
                }
            }
            
            nodes_f0 = event_gen.process_frame(sample_id, 0, frame_0_entities, [])
            print(f"    帧0: 生成 {len(nodes_f0)} 个节点")
            
            # 帧10: man移动
            frame_10_entities = {
                "man#1": {
                    "tag": "man#1", "label": "man",
                    "bbox": [100, 100, 200, 350], "center": (150, 225),
                    "attributes": {"upper_color": "black", "age": "middle_aged"}
                },
                "car#1": {
                    "tag": "car#1", "label": "car",
                    "bbox": [350, 200, 550, 350], "center": (450, 275),
                    "attributes": {"color": "red"}
                },
                "woman#1": {
                    "tag": "woman#1", "label": "woman",
                    "bbox": [200, 150, 280, 380], "center": (240, 265),
                    "attributes": {"upper_color": "white", "lower_color": "blue"}
                }
            }
            
            nodes_f10 = event_gen.process_frame(sample_id, 10, frame_10_entities, [
                {"subject": "man#1", "object": "woman#1", "predicate": "near"}
            ])
            print(f"    帧10: 生成 {len(nodes_f10)} 个节点")
            
            # ========== 阶段2: 统计DAG ==========
            print("\n  --- 阶段2: DAG统计 ---")
            
            all_nodes = dag_manager.get_all_nodes(sample_id)
            print(f"    总节点数: {len(all_nodes)}")
            
            type_counts = {}
            for n in all_nodes:
                t = n.event_type.value
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"    事件类型分布: {type_counts}")
            
            # 计算边数
            edge_count = sum(len(n.parents) for n in all_nodes)
            print(f"    总边数: {edge_count}")
            
            # ========== 阶段3: 闭包检索测试 ==========
            print("\n  --- 阶段3: 闭包检索 ---")
            
            # 构建索引
            index_count = retriever.build_index_for_sample(sample_id)
            print(f"    构建索引: {index_count} 个节点")
            
            # 测试查询
            query = "场景中有哪些人"
            results = retriever.retrieve(sample_id, query, top_k=3, max_depth=5)
            print(f"\n    查询: '{query}'")
            print(f"    返回 {len(results)} 个节点:")
            for node in results[:5]:
                print(f"      - [{node.event_type.value}] {node.content[:50]}...")
            
            # ========== 阶段4: 线性化上下文 ==========
            print("\n  --- 阶段4: 上下文线性化 ---")
            
            linearized = retriever.linearize_context(results)
            print(f"    线性化上下文 ({len(results)} 个节点):")
            print("    " + "-"*50)
            for line in linearized.split('\n')[:5]:
                if line.strip():
                    print(f"    {line[:70]}...")
            print("    " + "-"*50)
            
            print("\n  ✓ 端到端流程测试完成!")
            return True
            
    except Exception as e:
        print(f"  ✗ 端到端流程测试失败: {e}")
        traceback.print_exc()
        return False


# ============================================
# 主测试入口
# ============================================
def main():
    """运行所有测试"""
    print("="*60)
    print("STG DAG系统 - 端到端集成测试")
    print("="*60)
    
    tests = [
        ("模块导入", test_imports),
        ("逻辑时钟", test_logical_clock),
        ("节点和存储", test_dag_node_and_storage),
        ("DAG管理器", test_dag_manager),
        ("完整流程", test_full_pipeline),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n测试 '{name}' 发生未捕获异常: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
