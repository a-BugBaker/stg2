"""
DAG事件生成器模块

本模块实现基于DAG结构的事件节点生成逻辑：
    - 实体状态节点（每个实体一个）
    - 8种事件类型的节点生成
    - 因果链构建规则实现

事件类型（按修改要求）：
    1. entity_appeared: 实体首次出现
    2. entity_moved: 实体位移（与上次记录位置比较）
    3. relation: 关系事件
    4. attribute_changed: 属性变化
    5. interaction: 缓冲区级交互事件
    6. occlusion: 缓冲区级遮挡事件
    7. entity_disappeared: 实体消失
    8. periodic_description: 阶段性描述

因果链规则：
    - 实体首次出现: entity_appeared → entity_state
    - 首次位移: entity_state → movement_event; 后续: prev_movement → curr_movement
    - 关系事件: subject_state + object_state → relation; 连续重复: start → end
    - 属性变化: entity_state → first_change; 后续: prev_change → curr_change
    - 消失事件: entity_state → entity_disappeared
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .config import STGConfig
from .dag_core import DAGNode, EventType
from .dag_manager import DAGManager

logger = logging.getLogger(__name__)


class DAGEventGenerator:
    """DAG事件生成器。
    
    负责根据帧数据生成各类DAG事件节点，并建立正确的因果链。
    
    Attributes:
        config: STG配置
        dag_manager: DAG管理器
        _last_recorded_positions: 每个实体上次记录的位置（用于位移检测）
        _last_recorded_attributes: 每个实体上次记录的属性（用于属性变化检测）
        _active_relations: 当前活跃的关系（用于关系变化检测）
    """
    
    def __init__(self, config: STGConfig, dag_manager: DAGManager):
        """初始化事件生成器。
        
        Args:
            config: STG配置
            dag_manager: DAG管理器
        """
        self.config = config
        self.dag_manager = dag_manager
        
        # 位移检测：记录每个实体上次记录的位置
        self._last_recorded_positions: Dict[str, Tuple[float, float]] = {}
        
        # 属性变化检测：记录每个实体上次记录的属性
        self._last_recorded_attributes: Dict[str, Dict[str, Any]] = {}
        
        # 关系状态追踪
        # key: relation_key (如 "man3:stand_near:man8")
        # value: {"active": bool, "missing_frames": int, "start_frame": int}
        self._relation_states: Dict[str, Dict[str, Any]] = {}
    
    # ==================== 实体状态节点 ====================
    
    def create_or_update_entity_state(
        self,
        entity_tag: str,
        frame_idx: int,
        label: str,
        attributes: Dict[str, Any],
        bbox: Tuple[float, float, float, float],
        layer_id: Optional[int] = None,
        layer_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[DAGNode, bool]:
        """创建或更新实体状态节点。
        
        每个实体只有一个entity_state节点，轨迹更新时τ不变。
        
        Args:
            entity_tag: 实体标签
            frame_idx: 帧索引
            label: 实体类别标签
            attributes: 属性字典
            bbox: 边界框 (x1, y1, x2, y2)
            layer_id: 层级ID
            layer_mapping: 层级映射
            
        Returns:
            (节点, 是否新创建)
        """
        # 构建内容
        content = self._build_entity_state_content(
            entity_tag, label, attributes, bbox, layer_id, layer_mapping
        )
        
        # 构建元数据
        metadata = {
            "entity_tag": entity_tag,
            "label": label,
            "attributes": attributes,
            "layer_id": layer_id,
            "layer_mapping": layer_mapping,
            "frame_start": frame_idx,
            "latest_bbox": list(bbox)
        }
        
        node, is_new = self.dag_manager.get_or_create_entity_state(
            entity_tag=entity_tag,
            frame_idx=frame_idx,
            initial_content=content,
            metadata=metadata
        )
        
        if not is_new:
            # 更新轨迹（τ不变）
            node.metadata["latest_bbox"] = list(bbox)
            node.metadata["attributes"] = attributes
            self.dag_manager.update_node_content(node.node_id, content)
        
        return node, is_new
    
    def _build_entity_state_content(
        self,
        entity_tag: str,
        label: str,
        attributes: Dict[str, Any],
        bbox: Tuple[float, float, float, float],
        layer_id: Optional[int],
        layer_mapping: Optional[Any]
    ) -> str:
        """构建实体状态的自然语言描述。"""
        parts = [f"{entity_tag} is a {label}"]
        
        if attributes:
            attr_strs = [f"{k}: {v}" for k, v in attributes.items()]
            parts.append(f"with attributes [{', '.join(attr_strs)}]")
        
        # 计算中心位置
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        parts.append(f"at position ({cx:.1f}, {cy:.1f})")
        
        mapping_pairs = self._normalize_layer_mapping(entity_tag, layer_mapping)
        if mapping_pairs:
            mapping_strs = [f"{parent}->{child}" for parent, child in mapping_pairs]
            parts.append(f"layer mapping: {', '.join(mapping_strs)}")
        
        return " ".join(parts)

    def _normalize_layer_mapping(self, parent_tag: str, layer_mapping: Optional[Any]) -> List[Tuple[str, str]]:
        """将 layer_mapping 统一为 (parent, child) 列表。

        支持格式：
        - dict: {child: parent} 或 {child: <any>}（若 parent 非字符串，则回退为当前实体）
        - list[dict]: [{"tag": "child"}, {"child": "x"}, {"object_tag": "x"}]
        - list[str]: ["child1", "child2"]
        """
        if not layer_mapping:
            return []

        pairs: List[Tuple[str, str]] = []

        if isinstance(layer_mapping, dict):
            for child, mapped_parent in layer_mapping.items():
                child_tag = str(child).strip()
                if not child_tag:
                    continue
                parent = str(mapped_parent).strip() if isinstance(mapped_parent, str) and mapped_parent.strip() else parent_tag
                pairs.append((parent, child_tag))
            return pairs

        if isinstance(layer_mapping, list):
            for item in layer_mapping:
                if isinstance(item, dict):
                    child = item.get("tag") or item.get("child") or item.get("object_tag") or item.get("target")
                    if child:
                        pairs.append((parent_tag, str(child).strip()))
                elif isinstance(item, str):
                    child = item.strip()
                    if child:
                        pairs.append((parent_tag, child))
            return pairs

        return pairs
    
    # ==================== entity_appeared事件 ====================
    
    def create_appeared_event(
        self,
        entity_tag: str,
        frame_idx: int,
        label: str,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[DAGNode]:
        """创建实体出现事件节点。
        
        因果链：entity_appeared → entity_state
        
        Args:
            entity_tag: 实体标签
            frame_idx: 帧索引
            label: 实体类别
            bbox: 边界框
            
        Returns:
            事件节点，如果事件被禁用则返回None
        """
        if not self.config.dag.enable_entity_appeared:
            return None
        
        # 内容
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        content = f"{entity_tag} ({label}) appeared at frame {frame_idx}, position ({cx:.1f}, {cy:.1f})"
        
        # 元数据
        metadata = {
            "entity_tag": entity_tag,
            "label": label,
            "frame_start": frame_idx,
            "frame_end": frame_idx,
            "position": (cx, cy)
        }
        
        # 创建节点（无父节点，因为是实体状态的前驱）
        node = self.dag_manager.insert_node(
            node_type=EventType.ENTITY_APPEARED,
            content=content,
            frame_idx=frame_idx,
            parent_ids=[],
            metadata=metadata
        )
        
        # 建立 appeared → entity_state 的因果链
        entity_state_id = self.dag_manager.get_entity_state_node_id(entity_tag)
        if entity_state_id:
            self.dag_manager.graph_store.create_edge(node.node_id, entity_state_id)
        
        logger.debug(f"Created appeared event for {entity_tag} at frame {frame_idx}")
        return node
    
    # ==================== entity_moved事件 ====================
    
    def check_and_create_movement_event(
        self,
        entity_tag: str,
        frame_idx: int,
        current_bbox: Tuple[float, float, float, float]
    ) -> Optional[DAGNode]:
        """检查位移并创建移动事件。
        
        位移判定：与上次记录位置（不是上一帧）比较，超过阈值才创建。
        
        因果链：
            - 首次位移: entity_state → movement_event
            - 后续位移: prev_movement → curr_movement
        
        Args:
            entity_tag: 实体标签
            frame_idx: 帧索引
            current_bbox: 当前边界框
            
        Returns:
            事件节点，如果没有显著位移或事件被禁用则返回None
        """
        if not self.config.dag.enable_entity_moved:
            return None
        
        # 计算当前位置
        curr_x = (current_bbox[0] + current_bbox[2]) / 2
        curr_y = (current_bbox[1] + current_bbox[3]) / 2
        current_pos = (curr_x, curr_y)
        
        # 检查是否有上次记录的位置
        if entity_tag not in self._last_recorded_positions:
            # 首次记录位置，不产生移动事件
            self._last_recorded_positions[entity_tag] = current_pos
            return None
        
        last_pos = self._last_recorded_positions[entity_tag]
        
        # 计算位移
        displacement = ((curr_x - last_pos[0]) ** 2 + (curr_y - last_pos[1]) ** 2) ** 0.5
        
        if displacement < self.config.dag.movement_threshold:
            return None
        
        # 计算方向
        direction = self._compute_direction(last_pos, current_pos)
        
        # 更新记录位置
        self._last_recorded_positions[entity_tag] = current_pos
        
        # 内容
        content = (
            f"{entity_tag} moved from ({last_pos[0]:.1f}, {last_pos[1]:.1f}) "
            f"to ({curr_x:.1f}, {curr_y:.1f}), displacement={displacement:.1f}, direction={direction}"
        )
        
        # 元数据
        metadata = {
            "entity_tag": entity_tag,
            "frame_start": frame_idx,
            "frame_end": frame_idx,
            "from_position": last_pos,
            "to_position": current_pos,
            "displacement": displacement,
            "direction": direction
        }
        
        # 确定父节点
        chain_key = f"entity_moved:{entity_tag}"
        last_movement_id = self.dag_manager.get_last_event_node_id(chain_key)
        
        if last_movement_id:
            # 后续位移：prev_movement → curr_movement
            parent_ids = [last_movement_id]
        else:
            # 首次位移：entity_state → movement_event
            entity_state_id = self.dag_manager.get_entity_state_node_id(entity_tag)
            parent_ids = [entity_state_id] if entity_state_id else []
        
        # 创建节点
        node = self.dag_manager.insert_node(
            node_type=EventType.ENTITY_MOVED,
            content=content,
            frame_idx=frame_idx,
            parent_ids=parent_ids,
            metadata=metadata
        )
        
        # 更新事件链
        self.dag_manager.update_event_chain(chain_key, node.node_id)
        
        logger.debug(f"Created movement event for {entity_tag}: displacement={displacement:.1f}")
        return node
    
    def _compute_direction(
        self,
        from_pos: Tuple[float, float],
        to_pos: Tuple[float, float]
    ) -> str:
        """计算移动方向。"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # 简化为8个方向
        if abs(dx) < 1 and abs(dy) < 1:
            return "stationary"
        
        import math
        angle = math.atan2(-dy, dx) * 180 / math.pi  # 注意y轴向下
        
        if -22.5 <= angle < 22.5:
            return "right"
        elif 22.5 <= angle < 67.5:
            return "up-right"
        elif 67.5 <= angle < 112.5:
            return "up"
        elif 112.5 <= angle < 157.5:
            return "up-left"
        elif angle >= 157.5 or angle < -157.5:
            return "left"
        elif -157.5 <= angle < -112.5:
            return "down-left"
        elif -112.5 <= angle < -67.5:
            return "down"
        else:
            return "down-right"
    
    # ==================== relation事件 ====================
    
    def process_relations(
        self,
        frame_idx: int,
        current_relations: List[Dict[str, Any]]
    ) -> List[DAGNode]:
        """处理关系，生成relation事件节点。
        
        规则：
            - 新出现的关系：创建节点，开始追踪
            - 连续出现的关系：更新追踪状态
            - 消失的关系（连续重复）：在结束帧创建节点，由开始帧指向结束帧
        
        因果链：subject_state + object_state → relation
        
        Args:
            frame_idx: 帧索引
            current_relations: 当前帧的关系列表
            
        Returns:
            创建的事件节点列表
        """
        if not self.config.dag.enable_relation:
            return []
        
        created_nodes = []
        
        # 提取当前关系的key集合
        current_keys = set()
        for rel in current_relations:
            subject_tag = (
                rel.get("subject_tag")
                or rel.get("subject")
                or rel.get("source")
                or rel.get("subject_id")
            )
            raw_predicate = rel.get("predicate") or rel.get("name") or rel.get("relation")
            object_tag = (
                rel.get("object_tag")
                or rel.get("object")
                or rel.get("target")
                or rel.get("object_id")
            )

            if not raw_predicate:
                continue
            if not subject_tag:
                subject_tag = self._extract_subject_from_predicate(str(raw_predicate))
            if not object_tag:
                object_tag = "unknown"

            predicate_text = str(raw_predicate)
            predicate = self._normalize_predicate(predicate_text)
            relation_key = f"{subject_tag}:{predicate}:{object_tag}"
            current_keys.add(relation_key)
            
            # 处理新出现或继续的关系
            node = self._process_single_relation(
                frame_idx, relation_key, str(subject_tag), str(object_tag), predicate_text
            )
            if node:
                created_nodes.append(node)
        
        # 检查消失的关系
        ended_keys = []
        for relation_key in list(self._relation_states.keys()):
            if relation_key not in current_keys:
                state = self._relation_states[relation_key]
                if state["active"]:
                    # 关系结束
                    end_node = self._end_relation(frame_idx, relation_key, state)
                    if end_node:
                        created_nodes.append(end_node)
                    ended_keys.append(relation_key)
        
        # 清理已结束的关系
        for key in ended_keys:
            del self._relation_states[key]
        
        return created_nodes
    
    def process_relation(
        self,
        subject_tag: str,
        object_tag: str,
        predicate: str,
        frame_idx: int
    ) -> Optional[DAGNode]:
        """处理单个关系事件。
        
        这是外部调用的接口，内部使用_process_single_relation实现。
        
        Args:
            subject_tag: 主体标签
            object_tag: 客体标签
            predicate: 谓词
            frame_idx: 帧索引
            
        Returns:
            创建的事件节点（如果是新关系）
        """
        if not self.config.dag.enable_relation:
            return None
        
        # 规范化谓词并构建relation_key
        norm_predicate = self._normalize_predicate(predicate)
        relation_key = f"{subject_tag}:{norm_predicate}:{object_tag}"
        
        return self._process_single_relation(
            frame_idx=frame_idx,
            relation_key=relation_key,
            subject_tag=subject_tag,
            object_tag=object_tag,
            predicate=predicate
        )
    
    def _process_single_relation(
        self,
        frame_idx: int,
        relation_key: str,
        subject_tag: str,
        object_tag: str,
        predicate: str
    ) -> Optional[DAGNode]:
        """处理单个关系。"""
        if relation_key in self._relation_states:
            # 关系继续存在
            state = self._relation_states[relation_key]
            state["last_frame"] = frame_idx
            state["active"] = True
            return None
        
        # 新关系出现
        content = predicate
        
        # 获取父节点：两个实体的状态节点
        subject_state_id = self.dag_manager.get_entity_state_node_id(subject_tag)
        object_state_id = self.dag_manager.get_entity_state_node_id(object_tag)
        
        parent_ids = []
        if subject_state_id:
            parent_ids.append(subject_state_id)
        if object_state_id:
            parent_ids.append(object_state_id)
        
        # 检查是否是同一对实体的后续关系事件
        pair_key = f"relation:{subject_tag}:{object_tag}"
        last_relation_id = self.dag_manager.get_last_event_node_id(pair_key)
        if last_relation_id:
            parent_ids.append(last_relation_id)
        
        # 元数据
        metadata = {
            "subject_tag": subject_tag,
            "object_tag": object_tag,
            "predicate": predicate,
            "frame_start": frame_idx,
            "frame_end": frame_idx,
            "is_continuous_start": True
        }
        
        # 创建节点
        node = self.dag_manager.insert_node(
            node_type=EventType.RELATION,
            content=content,
            frame_idx=frame_idx,
            parent_ids=parent_ids,
            metadata=metadata
        )
        
        # 开始追踪
        self._relation_states[relation_key] = {
            "active": True,
            "start_frame": frame_idx,
            "last_frame": frame_idx,
            "start_node_id": node.node_id,
            "subject_tag": subject_tag,
            "object_tag": object_tag,
            "predicate": predicate
        }
        
        # 更新事件链
        self.dag_manager.update_event_chain(pair_key, node.node_id)
        
        logger.debug(f"Created relation event: {predicate}")
        return node
    
    def _end_relation(
        self,
        frame_idx: int,
        relation_key: str,
        state: Dict[str, Any]
    ) -> Optional[DAGNode]:
        """结束一个连续关系，创建结束节点。
        
        规则：连续重复则在开始帧和结束帧都创建，开始→结束
        """
        start_frame = state["start_frame"]
        last_frame = state["last_frame"]
        
        # 如果只持续一帧，不创建结束节点
        if last_frame == start_frame:
            return None
        
        content = f"{state['predicate']} (continuous from frame {start_frame} to {last_frame})"
        
        metadata = {
            "subject_tag": state["subject_tag"],
            "object_tag": state["object_tag"],
            "predicate": state["predicate"],
            "frame_start": start_frame,
            "frame_end": last_frame,
            "is_continuous_end": True,
            "duration_frames": last_frame - start_frame + 1
        }
        
        # 父节点是开始节点
        parent_ids = [state["start_node_id"]]
        
        node = self.dag_manager.insert_node(
            node_type=EventType.RELATION,
            content=content,
            frame_idx=last_frame,
            parent_ids=parent_ids,
            metadata=metadata
        )
        
        logger.debug(f"Created relation end event: {state['predicate']} (frames {start_frame}-{last_frame})")
        return node
    
    def _extract_subject_from_predicate(self, predicate: str) -> str:
        """从谓词中提取主语。如 'man3 stand near man8' -> 'man3'"""
        parts = predicate.split()
        return parts[0] if parts else "unknown"
    
    def _normalize_predicate(self, predicate: str) -> str:
        """标准化谓词用作key。"""
        return predicate.lower().replace(" ", "_")
    
    # ==================== attribute_changed事件 ====================
    
    def check_and_create_attribute_changed_events(
        self,
        entity_tag: str,
        frame_idx: int,
        current_attributes: Dict[str, Any]
    ) -> List[DAGNode]:
        """检查属性变化并创建事件。
        
        因果链：
            - 首次变化: entity_state → attr_changed
            - 后续变化: prev_attr_changed → curr_attr_changed
        
        Args:
            entity_tag: 实体标签
            frame_idx: 帧索引
            current_attributes: 当前属性
            
        Returns:
            创建的事件节点列表
        """
        if not self.config.dag.enable_attribute_changed:
            return []
        
        created_nodes = []
        
        # 获取上次记录的属性
        last_attrs = self._last_recorded_attributes.get(entity_tag, {})
        
        # 检查每个属性的变化
        for attr_name, new_value in current_attributes.items():
            old_value = last_attrs.get(attr_name)
            
            if old_value is None:
                # 新属性，记录但不创建事件
                continue
            
            if old_value != new_value:
                # 属性发生变化
                node = self._create_attribute_changed_event(
                    entity_tag, frame_idx, attr_name, old_value, new_value
                )
                if node:
                    created_nodes.append(node)
        
        # 更新记录
        self._last_recorded_attributes[entity_tag] = current_attributes.copy()
        
        return created_nodes
    
    def _create_attribute_changed_event(
        self,
        entity_tag: str,
        frame_idx: int,
        attr_name: str,
        old_value: Any,
        new_value: Any
    ) -> DAGNode:
        """创建属性变化事件节点。"""
        content = f"{entity_tag}'s {attr_name} changed from '{old_value}' to '{new_value}'"
        
        metadata = {
            "entity_tag": entity_tag,
            "attribute_name": attr_name,
            "old_value": old_value,
            "new_value": new_value,
            "frame_start": frame_idx,
            "frame_end": frame_idx
        }
        
        # 确定父节点
        chain_key = f"attribute_changed:{entity_tag}:{attr_name}"
        last_change_id = self.dag_manager.get_last_event_node_id(chain_key)
        
        if last_change_id:
            # 后续变化
            parent_ids = [last_change_id]
        else:
            # 首次变化
            entity_state_id = self.dag_manager.get_entity_state_node_id(entity_tag)
            parent_ids = [entity_state_id] if entity_state_id else []
        
        node = self.dag_manager.insert_node(
            node_type=EventType.ATTRIBUTE_CHANGED,
            content=content,
            frame_idx=frame_idx,
            parent_ids=parent_ids,
            metadata=metadata
        )
        
        # 更新事件链
        self.dag_manager.update_event_chain(chain_key, node.node_id)
        
        logger.debug(f"Created attribute_changed event: {entity_tag}.{attr_name}")
        return node
    
    # ==================== entity_disappeared事件 ====================
    
    def create_disappeared_event(
        self,
        entity_tag: str,
        frame_idx: int,
        last_known_bbox: Tuple[float, float, float, float]
    ) -> Optional[DAGNode]:
        """创建实体消失事件。
        
        因果链：entity_state → entity_disappeared
        
        Args:
            entity_tag: 实体标签
            frame_idx: 帧索引
            last_known_bbox: 最后已知的边界框
            
        Returns:
            事件节点，如果事件被禁用则返回None
        """
        if not self.config.dag.enable_entity_disappeared:
            return None
        
        cx = (last_known_bbox[0] + last_known_bbox[2]) / 2
        cy = (last_known_bbox[1] + last_known_bbox[3]) / 2
        
        content = f"{entity_tag} disappeared at frame {frame_idx}, last position ({cx:.1f}, {cy:.1f})"
        
        metadata = {
            "entity_tag": entity_tag,
            "frame_start": frame_idx,
            "frame_end": frame_idx,
            "last_position": (cx, cy)
        }
        
        # 父节点是实体状态
        entity_state_id = self.dag_manager.get_entity_state_node_id(entity_tag)
        parent_ids = [entity_state_id] if entity_state_id else []
        
        node = self.dag_manager.insert_node(
            node_type=EventType.ENTITY_DISAPPEARED,
            content=content,
            frame_idx=frame_idx,
            parent_ids=parent_ids,
            metadata=metadata
        )
        
        logger.debug(f"Created disappeared event for {entity_tag}")
        return node
    
    # ==================== 缓冲区级事件（interaction, occlusion, periodic_description）====================
    
    def create_interaction_event(
        self,
        entity1_tag: str,
        entity2_tag: str,
        interaction_type: str,
        frame_start: int,
        frame_end: int,
        details: Optional[Dict[str, Any]] = None
    ) -> Optional[DAGNode]:
        """创建交互事件（缓冲区级）。
        
        因果链：entity1_state + entity2_state → interaction
        
        Args:
            entity1_tag: 实体1标签
            entity2_tag: 实体2标签
            interaction_type: 交互类型（approaching, departing, moving_together）
            frame_start: 起始帧
            frame_end: 结束帧
            details: 额外细节
            
        Returns:
            事件节点
        """
        if not self.config.dag.enable_interaction:
            return None
        
        content = f"{entity1_tag} and {entity2_tag} {interaction_type} (frames {frame_start}-{frame_end})"
        
        metadata = {
            "entity1_tag": entity1_tag,
            "entity2_tag": entity2_tag,
            "interaction_type": interaction_type,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "details": details or {}
        }
        
        # 获取父节点
        parent_ids = []
        state1_id = self.dag_manager.get_entity_state_node_id(entity1_tag)
        state2_id = self.dag_manager.get_entity_state_node_id(entity2_tag)
        if state1_id:
            parent_ids.append(state1_id)
        if state2_id:
            parent_ids.append(state2_id)
        
        # 检查同对实体的上次交互
        pair_key = f"interaction:{min(entity1_tag, entity2_tag)}:{max(entity1_tag, entity2_tag)}"
        last_interaction_id = self.dag_manager.get_last_event_node_id(pair_key)
        if last_interaction_id:
            parent_ids.append(last_interaction_id)
        
        node = self.dag_manager.insert_node(
            node_type=EventType.INTERACTION,
            content=content,
            frame_idx=frame_end,
            parent_ids=parent_ids,
            metadata=metadata
        )
        
        self.dag_manager.update_event_chain(pair_key, node.node_id)
        
        logger.debug(f"Created interaction event: {entity1_tag} {interaction_type} {entity2_tag}")
        return node
    
    def create_occlusion_event(
        self,
        occluder_tag: str,
        occluded_tag: str,
        frame_start: int,
        frame_end: int
    ) -> Optional[DAGNode]:
        """创建遮挡事件（缓冲区级）。
        
        因果链：occluder_state + occluded_state → occlusion
        
        Args:
            occluder_tag: 遮挡者标签
            occluded_tag: 被遮挡者标签
            frame_start: 起始帧
            frame_end: 结束帧
            
        Returns:
            事件节点
        """
        if not self.config.dag.enable_occlusion:
            return None
        
        content = f"{occluder_tag} occludes {occluded_tag} (frames {frame_start}-{frame_end})"
        
        metadata = {
            "occluder_tag": occluder_tag,
            "occluded_tag": occluded_tag,
            "frame_start": frame_start,
            "frame_end": frame_end
        }
        
        parent_ids = []
        occluder_state_id = self.dag_manager.get_entity_state_node_id(occluder_tag)
        occluded_state_id = self.dag_manager.get_entity_state_node_id(occluded_tag)
        if occluder_state_id:
            parent_ids.append(occluder_state_id)
        if occluded_state_id:
            parent_ids.append(occluded_state_id)
        
        # 检查同对实体的上次遮挡
        pair_key = f"occlusion:{occluder_tag}:{occluded_tag}"
        last_occlusion_id = self.dag_manager.get_last_event_node_id(pair_key)
        if last_occlusion_id:
            parent_ids.append(last_occlusion_id)
        
        node = self.dag_manager.insert_node(
            node_type=EventType.OCCLUSION,
            content=content,
            frame_idx=frame_end,
            parent_ids=parent_ids,
            metadata=metadata
        )
        
        self.dag_manager.update_event_chain(pair_key, node.node_id)
        
        logger.debug(f"Created occlusion event: {occluder_tag} occludes {occluded_tag}")
        return node
    
    def create_periodic_description(
        self,
        frame_start: int,
        frame_end: int,
        involved_entities: List[str],
        description: str,
        description_type: str = "scene"
    ) -> Optional[DAGNode]:
        """创建阶段性描述节点（缓冲区级）。
        
        τ > 缓冲区内所有其他节点的τ
        因果链：涉及的所有实体状态 → periodic_description
        
        Args:
            frame_start: 起始帧
            frame_end: 结束帧
            involved_entities: 涉及的实体标签列表
            description: 描述内容
            description_type: 描述类型（scene/event/relation）
            
        Returns:
            事件节点
        """
        if not self.config.dag.enable_periodic_description:
            return None
        
        content = f"[{description_type}] {description}"
        
        metadata = {
            "description_type": description_type,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "involved_entities": involved_entities
        }
        
        # 获取所有涉及实体的状态节点作为父节点
        parent_ids = []
        for entity_tag in involved_entities:
            state_id = self.dag_manager.get_entity_state_node_id(entity_tag)
            if state_id:
                parent_ids.append(state_id)
        
        # 使用特殊的τ创建方法，确保大于缓冲区内所有节点
        tau = self.dag_manager.create_tau_after_all(frame_end)
        
        node = DAGNode(
            node_type=EventType.PERIODIC_DESCRIPTION,
            content=content,
            tau=tau,
            metadata=metadata,
            parent_ids=[]  # 先设为空，后面处理
        )
        
        # 传递规约
        if parent_ids and self.config.dag.transitive_reduction:
            reduced_parents = self.dag_manager._transitive_reduction(node.node_id, parent_ids)
        else:
            reduced_parents = parent_ids
        node.parent_ids = reduced_parents
        
        # 计算嵌入
        if self.dag_manager._embed_func:
            try:
                node.embedding = self.dag_manager._embed_func(content)
            except Exception as e:
                logger.warning(f"Failed to compute embedding: {e}")
        
        # 存储
        self.dag_manager.graph_store.create_node(
            node_id=node.node_id,
            node_type=EventType.PERIODIC_DESCRIPTION.value,
            tau=tau.to_tuple()
        )
        for parent_id in node.parent_ids:
            self.dag_manager.graph_store.create_edge(parent_id, node.node_id)
        self.dag_manager.meta_store.save_node(self.dag_manager._current_sample_id, node)
        
        logger.debug(f"Created periodic description: {description_type}")
        return node
    
    # ==================== layer_mapping处理 ====================
    
    def process_layer_mapping(
        self,
        layer_mapping: Any,
        frame_idx: int
    ) -> None:
        """处理layer_mapping依赖关系。
        
        规则：man → shoes（shoes依赖man）
        
        Args:
            layer_mapping: 层级映射字典，如 {"shoes1": "man3"}
            frame_idx: 帧索引
        """
        # 支持 dict/list 两种 layer_mapping 表达，调用方多传当前实体时优先使用。
        pairs: List[Tuple[str, str]] = []
        if isinstance(layer_mapping, dict):
            for child_tag, parent_tag in layer_mapping.items():
                if isinstance(parent_tag, str) and str(parent_tag).strip():
                    pairs.append((str(parent_tag).strip(), str(child_tag).strip()))
        elif isinstance(layer_mapping, list):
            for item in layer_mapping:
                if isinstance(item, dict):
                    parent_tag = item.get("parent") or item.get("parent_tag")
                    child_tag = item.get("tag") or item.get("child") or item.get("object_tag")
                    if parent_tag and child_tag:
                        pairs.append((str(parent_tag).strip(), str(child_tag).strip()))

        for parent_tag, child_tag in pairs:
            if parent_tag and child_tag:
                self.dag_manager.process_layer_mapping(parent_tag, child_tag, frame_idx)
    
    # ==================== 帧级处理入口 ====================
    
    def process_frame(
        self,
        sample_id: str,
        frame_idx: int,
        entities: Dict[str, Dict[str, Any]],
        relations: List[Dict[str, Any]],
        layer_mapping: Optional[Dict[str, str]] = None
    ) -> List[DAGNode]:
        """处理单帧数据，生成所有相关事件节点。
        
        这是帧级处理的主入口，负责：
        1. 处理每个实体：创建/更新entity_state，检测appeared/moved/attribute_changed
        2. 处理关系：创建relation事件
        3. 处理layer_mapping依赖
        
        Args:
            sample_id: 样本ID
            frame_idx: 帧索引
            entities: 实体字典，格式为 {tag: {tag, label, bbox, center, attributes, ...}}
            relations: 关系列表，格式为 [{subject, object, predicate}, ...]
            layer_mapping: 可选的层级映射
            
        Returns:
            本帧生成的所有节点列表
        """
        # 设置当前sample_id
        self.dag_manager.set_current_sample(sample_id)
        
        generated_nodes: List[DAGNode] = []
        
        # 1. 处理每个实体
        current_entity_tags = set(entities.keys())
        
        for tag, entity_data in entities.items():
            # 提取实体信息
            label = entity_data.get("label", "unknown")
            bbox = entity_data.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox, list) and len(bbox) == 4:
                bbox = tuple(bbox)
            else:
                bbox = (0, 0, 0, 0)
            
            center = entity_data.get("center")
            if center is None and len(bbox) == 4:
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            attributes = entity_data.get("attributes", {})
            layer_id = entity_data.get("layer_id")
            entity_layer_mapping = entity_data.get("layer_mapping")
            
            # 创建或更新entity_state
            state_node, is_new = self.create_or_update_entity_state(
                entity_tag=tag,
                frame_idx=frame_idx,
                label=label,
                attributes=attributes,
                bbox=bbox,
                layer_id=layer_id,
                layer_mapping=entity_layer_mapping
            )
            
            if is_new:
                generated_nodes.append(state_node)
                
                # 创建entity_appeared事件
                appeared_node = self.create_appeared_event(
                    entity_tag=tag,
                    frame_idx=frame_idx,
                    label=label,
                    bbox=bbox
                )
                if appeared_node:
                    generated_nodes.append(appeared_node)
            
            # 检测位移
            if center:
                movement_node = self.check_and_create_movement_event(
                    entity_tag=tag,
                    frame_idx=frame_idx,
                    current_bbox=bbox
                )
                if movement_node:
                    generated_nodes.append(movement_node)
            
            # 检测属性变化
            attr_nodes = self.check_and_create_attribute_changed_events(
                entity_tag=tag,
                frame_idx=frame_idx,
                current_attributes=attributes
            )
            generated_nodes.extend(attr_nodes)
        
        # 2. 处理关系
        current_relations = set()
        for rel in relations:
            subject = rel.get("subject")
            obj = rel.get("object")
            predicate = rel.get("predicate")
            
            if subject and obj and predicate:
                # 构建关系key
                relation_key = f"{subject}:{predicate}:{obj}"
                current_relations.add(relation_key)
                
                # 处理关系
                rel_node = self.process_relation(
                    subject_tag=subject,
                    object_tag=obj,
                    predicate=predicate,
                    frame_idx=frame_idx
                )
                if rel_node:
                    generated_nodes.append(rel_node)
        
        # 检测结束的关系
        for relation_key, state in list(self._relation_states.items()):
            if state.get("active") and relation_key not in current_relations:
                # 关系不再存在，检查去抖动
                missing_frames = state.get("missing_frames", 0) + 1
                
                debounce_threshold = self.config.dag.relation_debounce_frames
                if missing_frames >= debounce_threshold:
                    # 确认关系结束
                    end_node = self._end_relation(frame_idx, relation_key, state)
                    if end_node:
                        generated_nodes.append(end_node)
                    state["active"] = False
                else:
                    state["missing_frames"] = missing_frames
        
        # 更新活跃关系的last_frame
        for relation_key in current_relations:
            if relation_key in self._relation_states:
                self._relation_states[relation_key]["last_frame"] = frame_idx
                self._relation_states[relation_key]["missing_frames"] = 0
        
        # 3. 处理layer_mapping
        if layer_mapping:
            self.process_layer_mapping(layer_mapping, frame_idx)
        
        return generated_nodes
    
    def set_embed_func(self, func) -> None:
        """设置嵌入函数。"""
        self.dag_manager.set_embed_func(func)
    
    # ==================== 状态重置 ====================
    
    def reset(self) -> None:
        """重置所有内部状态。"""
        self._last_recorded_positions.clear()
        self._last_recorded_attributes.clear()
        self._relation_states.clear()
