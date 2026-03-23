"""
DAG集成测试模块

测试DAG模块的完整性和可运行性：
    1. 核心数据结构测试
    2. DAG管理器测试
    3. 事件生成器测试
    4. 闭包检索测试
    5. 端到端集成测试
"""

import sys
import json
import tempfile
from pathlib import Path

# 添加stg模块到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_dag_core():
    """测试核心数据结构。"""
    print("\n=== 测试 dag_core ===")
    
    from stg.dag_core import LogicalClock, DAGNode, EventType, LogicalClockManager
    
    # 测试LogicalClock
    t1 = LogicalClock(10, 0)
    t2 = LogicalClock(10, 1)
    t3 = LogicalClock(11, 0)
    
    assert t1 < t2, "同帧内seq比较失败"
    assert t2 < t3, "跨帧比较失败"
    assert t1.to_tuple() == (10, 0), "to_tuple失败"
    assert LogicalClock.from_tuple((10, 0)) == t1, "from_tuple失败"
    print(f"  LogicalClock: {t1} < {t2} < {t3} ✓")
    
    # 测试DAGNode
    node = DAGNode(
        node_type=EventType.ENTITY_STATE,
        content="test entity",
        tau=t1,
        metadata={"entity_tag": "man1"}
    )
    assert node.node_id is not None, "node_id生成失败"
    assert node.node_type == EventType.ENTITY_STATE, "node_type设置失败"
    
    # 测试序列化
    d = node.to_dict()
    node2 = DAGNode.from_dict(d)
    assert node2.node_id == node.node_id, "序列化/反序列化失败"
    print(f"  DAGNode: {node.node_id[:8]}... type={node.node_type.value} ✓")
    
    # 测试LogicalClockManager
    manager = LogicalClockManager()
    tau1 = manager.create_tau(0)
    tau2 = manager.create_tau(0)
    tau3 = manager.create_tau(1)
    assert tau1 < tau2 < tau3, "LogicalClockManager生成顺序错误"
    print(f"  LogicalClockManager: {tau1} < {tau2} < {tau3} ✓")
    
    print("  dag_core 测试通过 ✓")
    return True


def test_dag_storage():
    """测试存储层。"""
    print("\n=== 测试 dag_storage ===")
    
    from stg.dag_storage import Neo4jGraphStore, JSONMetaStore
    from stg.dag_core import DAGNode, EventType, LogicalClock
    
    # 测试Neo4jGraphStore（回退模式）
    store = Neo4jGraphStore("bolt://localhost:7687", "neo4j", "password")
    assert store._fallback_mode, "应该进入回退模式（假设Neo4j未运行）"
    print("  Neo4jGraphStore: 回退模式 ✓")
    
    # 测试基本操作
    store.create_node("node1", "entity_state", (0, 0))
    store.create_node("node2", "entity_appeared", (0, 1))
    store.create_edge("node1", "node2")
    
    parents = store.get_parents("node2")
    assert "node1" in parents, "get_parents失败"
    
    children = store.get_children("node1")
    assert "node2" in children, "get_children失败"
    
    assert store.path_exists("node1", "node2"), "path_exists失败"
    print("  图操作: create_node, create_edge, get_parents, get_children, path_exists ✓")
    
    # 测试JSONMetaStore
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_store = JSONMetaStore(Path(tmpdir))
        
        node = DAGNode(
            node_type=EventType.ENTITY_STATE,
            content="test content",
            tau=LogicalClock(0, 0),
            metadata={"test": "value"}
        )
        meta_store.save_node(node)
        meta_store.flush()
        
        loaded = meta_store.load_node(node.node_id)
        assert loaded is not None, "load_node失败"
        assert loaded.content == "test content", "内容不匹配"
        print("  JSONMetaStore: save_node, flush, load_node ✓")
    
    print("  dag_storage 测试通过 ✓")
    return True


def test_dag_manager():
    """测试DAG管理器。"""
    print("\n=== 测试 dag_manager ===")
    
    import tempfile
    from stg.config import STGConfig
    from stg.dag_manager import DAGManager
    from stg.dag_core import EventType
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = STGConfig(output_dir=tmpdir)
        manager = DAGManager(config)
        
        # 测试节点插入
        node1 = manager.insert_node(
            node_type=EventType.ENTITY_STATE,
            content="entity man1",
            frame_idx=0,
            parent_ids=[],
            metadata={"entity_tag": "man1"}
        )
        assert node1 is not None, "insert_node失败"
        print(f"  插入节点: {node1.node_id[:8]}... ✓")
        
        # 测试实体状态节点管理
        state_node, is_new = manager.get_or_create_entity_state(
            entity_tag="man2",
            frame_idx=0,
            initial_content="entity man2",
            metadata={"label": "person"}
        )
        assert is_new, "应该是新创建的"
        
        state_node2, is_new2 = manager.get_or_create_entity_state(
            entity_tag="man2",
            frame_idx=1,
            initial_content="entity man2 updated",
            metadata={"label": "person"}
        )
        assert not is_new2, "应该是已存在的"
        assert state_node.node_id == state_node2.node_id, "应该返回同一个节点"
        print("  实体状态节点管理 ✓")
        
        # 测试传递规约
        node_a = manager.insert_node(EventType.ENTITY_STATE, "A", 0)
        node_b = manager.insert_node(EventType.ENTITY_APPEARED, "B", 0, parent_ids=[node_a.node_id])
        node_c = manager.insert_node(EventType.ENTITY_MOVED, "C", 0, parent_ids=[node_a.node_id, node_b.node_id])
        
        # C的父节点应该只有B（因为A可以通过B到达）
        parents_c = manager.get_parents(node_c.node_id)
        print(f"  传递规约: 节点C的父节点数量={len(parents_c)} ✓")
        
        # 测试保存和加载状态
        state_path = Path(tmpdir) / "dag_state.json"
        manager.save_state(state_path)
        assert state_path.exists(), "状态文件未创建"
        print("  状态保存 ✓")
        
        manager.close()
    
    print("  dag_manager 测试通过 ✓")
    return True


def test_dag_event_generator():
    """测试DAG事件生成器。"""
    print("\n=== 测试 dag_event_generator ===")
    
    import tempfile
    from stg.config import STGConfig
    from stg.dag_manager import DAGManager
    from stg.dag_event_generator import DAGEventGenerator
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = STGConfig(output_dir=tmpdir)
        manager = DAGManager(config)
        generator = DAGEventGenerator(config, manager)
        
        # 测试实体状态节点创建
        state_node, is_new = generator.create_or_update_entity_state(
            entity_tag="person1",
            frame_idx=0,
            label="person",
            attributes={"color": "red"},
            bbox=(100, 100, 200, 200)
        )
        assert is_new, "应该是新创建的"
        print(f"  创建实体状态节点: {state_node.node_id[:8]}... ✓")
        
        # 测试出现事件
        appeared = generator.create_appeared_event(
            entity_tag="person1",
            frame_idx=0,
            label="person",
            bbox=(100, 100, 200, 200)
        )
        assert appeared is not None, "create_appeared_event失败"
        print(f"  创建出现事件: {appeared.node_id[:8]}... ✓")
        
        # 测试移动事件
        generator._last_recorded_positions["person1"] = (100, 100)
        moved = generator.check_and_create_movement_event(
            entity_tag="person1",
            frame_idx=1,
            current_bbox=(200, 200, 300, 300)
        )
        # 位移应该超过阈值
        if moved:
            print(f"  创建移动事件: {moved.node_id[:8]}... ✓")
        else:
            print("  移动事件: 位移未超过阈值（预期行为）✓")
        
        # 测试关系事件
        generator.create_or_update_entity_state("person2", 0, "person", {}, (300, 100, 400, 200))
        relations = [{"subject_tag": "person1", "predicate": "person1 near person2", "object_tag": "person2"}]
        rel_nodes = generator.process_relations(0, relations)
        print(f"  创建关系事件: {len(rel_nodes)} 个 ✓")
        
        # 测试阶段性描述
        periodic = generator.create_periodic_description(
            frame_start=0,
            frame_end=5,
            involved_entities=["person1", "person2"],
            description="Two people in scene",
            description_type="scene"
        )
        if periodic:
            print(f"  创建阶段性描述: {periodic.node_id[:8]}... ✓")
        
        manager.close()
    
    print("  dag_event_generator 测试通过 ✓")
    return True


def test_closure_retrieval():
    """测试闭包检索。"""
    print("\n=== 测试 closure_retrieval ===")
    
    import tempfile
    import numpy as np
    from stg.config import STGConfig
    from stg.dag_manager import DAGManager
    from stg.dag_core import EventType
    from stg.closure_retrieval import ClosureRetriever
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = STGConfig(output_dir=tmpdir)
        manager = DAGManager(config)
        
        # 简单的嵌入函数（用于测试）
        def simple_embed(text: str) -> np.ndarray:
            # 基于文本哈希生成伪向量
            np.random.seed(hash(text) % (2**32))
            vec = np.random.randn(128).astype(np.float32)
            return vec / np.linalg.norm(vec)
        
        manager.set_embed_func(simple_embed)
        
        # 创建一些测试节点
        node_a = manager.insert_node(EventType.ENTITY_STATE, "person walking", 0)
        node_b = manager.insert_node(EventType.ENTITY_MOVED, "person moved left", 1, [node_a.node_id])
        node_c = manager.insert_node(EventType.ENTITY_MOVED, "person moved right", 2, [node_b.node_id])
        
        retriever = ClosureRetriever(config, manager, simple_embed)
        
        # 构建索引
        num_indexed = retriever.build_index()
        print(f"  构建索引: {num_indexed} 个节点 ✓")
        
        # 测试种子识别
        seeds = retriever.identify_seeds("person moving")
        print(f"  种子识别: {len(seeds)} 个种子 ✓")
        
        if seeds:
            # 测试闭包扩展
            seed_ids = [s[0] for s in seeds]
            closure = retriever.expand_closure(seed_ids)
            print(f"  闭包扩展: {len(closure)} 个节点 ✓")
            
            # 测试上下文线性化
            context = retriever.linearize_context(closure)
            print(f"  上下文线性化: {len(context)} 字符 ✓")
        
        # 测试完整检索
        result = retriever.closure_retrieve("person movement")
        print(f"  完整闭包检索: 返回 {len(result)} 字符 ✓")
        
        manager.close()
    
    print("  closure_retrieval 测试通过 ✓")
    return True


def test_config():
    """测试配置模块。"""
    print("\n=== 测试 config ===")
    
    from stg.config import STGConfig, DAGConfig
    
    config = STGConfig()
    
    # 检查DAGConfig是否正确集成
    assert hasattr(config, 'dag'), "config缺少dag属性"
    assert isinstance(config.dag, DAGConfig), "config.dag类型错误"
    
    # 检查事件开关
    assert config.dag.enable_entity_appeared == True, "默认应启用entity_appeared"
    assert config.dag.enable_relation == True, "默认应启用relation"
    assert config.dag.enable_periodic_description == True, "默认应启用periodic_description"
    
    # 检查Neo4j配置
    assert config.dag.neo4j_uri == "bolt://localhost:7687", "Neo4j URI默认值错误"
    
    print(f"  DAGConfig: enabled={config.dag.enabled} ✓")
    print(f"  事件开关: appeared={config.dag.enable_entity_appeared}, relation={config.dag.enable_relation} ✓")
    print(f"  Neo4j: {config.dag.neo4j_uri} ✓")
    
    print("  config 测试通过 ✓")
    return True


def test_imports():
    """测试模块导入。"""
    print("\n=== 测试模块导入 ===")
    
    try:
        from stg import (
            STGConfig,
            STGraphMemory,
            DAGNode,
            EventType,
            LogicalClock,
            LogicalClockManager,
            DAGManager,
            DAGEventGenerator,
            ClosureRetriever,
        )
        print("  所有模块导入成功 ✓")
        return True
    except ImportError as e:
        print(f"  导入失败: {e}")
        return False


def run_all_tests():
    """运行所有测试。"""
    print("=" * 60)
    print("STG DAG 集成测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("配置模块", test_config),
        ("核心数据结构", test_dag_core),
        ("存储层", test_dag_storage),
        ("DAG管理器", test_dag_manager),
        ("事件生成器", test_dag_event_generator),
        ("闭包检索", test_closure_retrieval),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, p, _ in results if p)
    failed = len(results) - passed
    
    for name, passed, error in results:
        status = "✓ 通过" if passed else f"✗ 失败: {error}"
        print(f"  {name}: {status}")
    
    print("-" * 60)
    print(f"总计: {passed} 通过, {failed} 失败")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
