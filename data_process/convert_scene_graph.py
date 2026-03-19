# -*- coding: utf-8 -*-
"""
场景图数据格式转换脚本
将 source_template 格式转换为 target_template 格式

源格式 (source_template):
- objects 包含 boxes, labels, scores 三个列表（一一对应）
- hierarchy 包含 layer1_nodes, layer2_mapping, layer3_mapping
- attributes 是一个字典，key是idx(字符串)，value是描述
- relations 是一个列表，每个元素包含 idx[subject, object], subject_label, object_label, predicate, confidence

目标格式 (target_template):
- objects 是一个列表，每个元素是一个对象字典，包含:
  - idx: 整数（该帧唯一）
  - box: [x1,y1,x2,y2]
  - score: 浮点数
  - label: 字符串
  - tag: label+数字（如果同类只有一个则无数字）
  - attributes: 字符串描述
  - layer_id: 层级（1为顶层）
  - layer_mapping: 指向下层实体的列表
  - subject_relations: 该对象作为主体的关系列表
  - object_relations: 该对象作为客体的关系列表
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def generate_tag(label: str, count: int, total: int) -> str:
    """
    生成对象的tag
    
    Args:
        label: 对象的标签
        count: 当前是该标签的第几个实例（从1开始）
        total: 该标签总共有多少个实例
    
    Returns:
        tag字符串，如果只有一个实例则不加数字
    """
    if total == 1:
        return label
    return f"{label}{count}"


def build_label_to_indices(labels: List[str]) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    """
    构建标签到索引的映射，以及索引到tag的映射
    
    Args:
        labels: 标签列表
    
    Returns:
        - label_indices: 标签 -> 该标签对应的所有idx列表
        - idx_to_tag: idx -> 对应的tag
    """
    # 统计每个标签出现的次数和对应的索引
    label_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    
    # 为每个idx分配tag
    idx_to_tag = {}
    for label, indices in label_indices.items():
        total = len(indices)
        for count, idx in enumerate(indices, start=1):
            idx_to_tag[idx] = generate_tag(label, count, total)
    
    return dict(label_indices), idx_to_tag


def parse_hierarchy_tag(tag: str) -> Tuple[str, int]:
    """
    解析hierarchy中的tag，提取label和编号
    
    Args:
        tag: 如 "man1", "basketball court", "shoes14"
    
    Returns:
        (label, number) 如果没有数字则number为0
    """
    # 从后往前找数字
    i = len(tag) - 1
    while i >= 0 and tag[i].isdigit():
        i -= 1
    
    if i == len(tag) - 1:
        # 没有数字后缀
        return tag, 0
    else:
        label = tag[:i+1]
        number = int(tag[i+1:])
        return label, number


def build_tag_to_idx(labels: List[str]) -> Dict[str, int]:
    """
    构建tag到idx的映射
    
    Args:
        labels: 标签列表
    
    Returns:
        tag -> idx 的映射
    """
    label_counts = defaultdict(int)
    label_totals = defaultdict(int)
    
    # 先统计总数
    for label in labels:
        label_totals[label] += 1
    
    # 再分配tag
    tag_to_idx = {}
    label_counts = defaultdict(int)
    for idx, label in enumerate(labels):
        label_counts[label] += 1
        tag = generate_tag(label, label_counts[label], label_totals[label])
        tag_to_idx[tag] = idx
    
    return tag_to_idx


def determine_layer_id(tag: str, hierarchy: Dict) -> int:
    """
    根据hierarchy确定对象的层级
    
    Args:
        tag: 对象的tag
        hierarchy: 层次结构信息
    
    Returns:
        层级ID（1为顶层，2为第二层，3为第三层）
    """
    # 检查是否在第一层
    layer1_nodes = hierarchy.get("layer1_nodes", [])
    if tag in layer1_nodes:
        return 1
    
    # 检查是否在第二层（作为layer2_mapping的值出现）
    layer2_mapping = hierarchy.get("layer2_mapping", {})
    for parent, children in layer2_mapping.items():
        if tag in children:
            return 2
    
    # 检查是否在第三层（作为layer3_mapping的值出现）
    layer3_mapping = hierarchy.get("layer3_mapping", {})
    for parent, children in layer3_mapping.items():
        if tag in children:
            return 3
    
    # 默认为第一层
    return 1


def build_layer_mapping(tag: str, hierarchy: Dict, tag_to_idx: Dict[str, int]) -> List[Dict]:
    """
    构建layer_mapping：该对象指向的下层实体列表
    
    Args:
        tag: 对象的tag
        hierarchy: 层次结构信息
        tag_to_idx: tag到idx的映射
    
    Returns:
        layer_mapping列表，每个元素是 {"idx": int, "tag": str}
    """
    layer_mapping = []
    
    # 检查layer2_mapping（该对象是否指向第二层对象）
    layer2_mapping = hierarchy.get("layer2_mapping", {})
    if tag in layer2_mapping:
        for child_tag in layer2_mapping[tag]:
            if child_tag in tag_to_idx:
                layer_mapping.append({
                    "idx": tag_to_idx[child_tag],
                    "tag": child_tag
                })
    
    # 检查layer3_mapping（该对象是否指向第三层对象）
    layer3_mapping = hierarchy.get("layer3_mapping", {})
    if tag in layer3_mapping:
        for child_tag in layer3_mapping[tag]:
            if child_tag in tag_to_idx:
                layer_mapping.append({
                    "idx": tag_to_idx[child_tag],
                    "tag": child_tag
                })
    
    return layer_mapping


def convert_frame(frame_data: Dict) -> Dict:
    """
    转换单帧数据
    
    Args:
        frame_data: 源格式的单帧数据
    
    Returns:
        目标格式的单帧数据
    """
    # 提取基本信息
    image_path = frame_data.get("image_path", "")
    objects_data = frame_data.get("objects", {})
    hierarchy = frame_data.get("hierarchy", {})
    attributes = frame_data.get("attributes", {})
    relations = frame_data.get("relations", [])
    
    # 提取boxes, labels, scores
    boxes = objects_data.get("boxes", [])
    labels = objects_data.get("labels", [])
    scores = objects_data.get("scores", [])
    
    # 构建tag相关映射
    label_indices, idx_to_tag = build_label_to_indices(labels)
    tag_to_idx = build_tag_to_idx(labels)
    
    # 初始化每个对象的关系列表
    subject_relations_map = defaultdict(list)  # idx -> subject_relations
    object_relations_map = defaultdict(list)   # idx -> object_relations
    
    # 处理relations
    for rel in relations:
        rel_idx = rel.get("idx", [])
        if len(rel_idx) != 2:
            continue
        
        subject_idx, object_idx = rel_idx
        predicate = rel.get("predicate", "")
        confidence = rel.get("confidence", 1.0)
        subject_label = rel.get("subject_label", "")
        object_label = rel.get("object_label", "")
        
        # 添加到subject的subject_relations
        if subject_idx < len(labels):
            object_tag = idx_to_tag.get(object_idx, object_label)
            subject_relations_map[subject_idx].append({
                "idx": object_idx,
                "predicate": predicate,
                "object_tag": object_tag,
                "confidence": confidence
            })
        
        # 添加到object的object_relations
        if object_idx < len(labels):
            subject_tag = idx_to_tag.get(subject_idx, subject_label)
            object_relations_map[object_idx].append({
                "idx": subject_idx,
                "predicate": predicate,
                "subject_tag": subject_tag,
                "confidence": confidence
            })
    
    # 构建目标格式的objects列表
    target_objects = []
    for idx in range(len(boxes)):
        tag = idx_to_tag.get(idx, labels[idx] if idx < len(labels) else "unknown")
        label = labels[idx] if idx < len(labels) else "unknown"
        
        obj = {
            "idx": idx,
            "box": boxes[idx],
            "score": scores[idx] if idx < len(scores) else 0.0,
            "label": label,
            "tag": tag,
            "attributes": attributes.get(str(idx), ""),
            "layer_id": determine_layer_id(tag, hierarchy),
            "layer_mapping": build_layer_mapping(tag, hierarchy, tag_to_idx),
            "subject_relations": subject_relations_map.get(idx, []),
            "object_relations": object_relations_map.get(idx, [])
        }
        target_objects.append(obj)
    
    return {
        "image_path": image_path,
        "objects": target_objects
    }


def convert_scene_graphs(input_file: str, output_file: str):
    """
    转换整个场景图数据文件
    
    Args:
        input_file: 输入文件路径（源格式）
        output_file: 输出文件路径（目标格式）
    """
    print(f"正在读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    print(f"共有 {len(source_data)} 帧数据需要转换")
    
    target_data = []
    for i, frame in enumerate(source_data):
        if (i + 1) % 100 == 0:
            print(f"正在处理第 {i + 1}/{len(source_data)} 帧...")
        target_frame = convert_frame(frame)
        target_data.append(target_frame)
    
    print(f"正在写入输出文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, ensure_ascii=False, indent=2)
    
    print("转换完成!")


def main():
    """主函数"""
    import os
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义输入输出文件路径
    input_file = os.path.join(script_dir, "scene_graphs(1).json")
    output_file = os.path.join(script_dir, "converted_scene_graphs.json")
    
    # 执行转换
    convert_scene_graphs(input_file, output_file)


if __name__ == "__main__":
    main()
