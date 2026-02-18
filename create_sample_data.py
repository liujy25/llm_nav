#!/usr/bin/env python3
"""
示例脚本：创建测试数据
"""

import json
import os
from pathlib import Path


def create_sample_test_data(output_dir: str = "./sample_test_data"):
    """
    创建示例测试数据
    
    这个函数会创建一个示例数据集结构，你需要：
    1. 将实际的图片放入这个目录
    2. 根据实际情况修改JSON标注
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 示例标注数据
    samples = [
        {
            "image_name": "scene_001",
            "targets": ["chair", "table", "bottle"],
            "num_actions": 5
        },
        {
            "image_name": "scene_002",
            "targets": ["door", "window"],
            "num_actions": 6
        },
        {
            "image_name": "scene_003",
            "targets": ["sofa", "lamp", "tv"],
            "num_actions": 4
        }
    ]
    
    print(f"创建示例测试数据到: {output_dir}")
    print("=" * 60)
    
    for sample in samples:
        json_file = output_path / f"{sample['image_name']}.json"
        
        annotation = {
            "targets": sample["targets"],
            "num_actions": sample["num_actions"]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 创建标注文件: {json_file.name}")
        print(f"  目标: {', '.join(sample['targets'])}")
        print(f"  Waypoint数量: {sample['num_actions']}")
        print(f"  请添加对应的图片: {sample['image_name']}.jpg 或 .png")
        print()
    
    print("=" * 60)
    print("下一步:")
    print("1. 将实际的图片文件复制到该目录")
    print("2. 确保图片文件名与JSON文件名匹配（除扩展名外）")
    print("3. 根据实际情况修改JSON中的targets和num_actions")
    print(f"4. 运行测试: python test_prompt_comparison.py --data_dir {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='创建示例测试数据')
    parser.add_argument('--output_dir', type=str, default='./sample_test_data',
                        help='输出目录')
    
    args = parser.parse_args()
    create_sample_test_data(args.output_dir)
