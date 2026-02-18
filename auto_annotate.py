#!/usr/bin/env python3
"""
自动标注工具：使用LLM分析标注图片并生成JSON标注文件

使用方式：
1. 准备带waypoint标注的图片（图片上已经标注了数字1, 2, 3...）
2. 运行脚本自动生成JSON标注文件
3. JSON文件包含：场景描述、action数量、目标物体列表

示例：
python auto_annotate.py --image_dir ./annotated_images --output_dir ./annotations
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List
import time

from vlm import OpenAIVLM


class AutoAnnotator:
    """自动标注器：使用LLM分析图片并生成标注"""
    
    def __init__(self, vlm_config: Dict):
        """
        初始化自动标注器
        
        Parameters
        ----------
        vlm_config : Dict
            VLM配置
        """
        self.vlm_config = vlm_config
        
        # 初始化VLM
        system_instruction = (
            "You are a visual scene analyzer. You observe images with waypoint annotations "
            "and provide detailed scene descriptions and object lists."
        )
        self.vlm = OpenAIVLM(
            model=vlm_config['model'],
            system_instruction=system_instruction,
            api_key=vlm_config['api_key'],
            base_url=vlm_config['base_url'],
            timeout=vlm_config['timeout']
        )
    
    def construct_annotation_prompt(self) -> str:
        """构建标注prompt"""
        prompt = """Please analyze this image carefully. The image shows a scene with numbered waypoint annotations (1, 2, 3, etc.).

Please provide the following information:

1. Scene Description:
Describe what you see in this scene. Include:
- The type of room or environment
- Major furniture and objects visible
- The layout and spatial arrangement
- Any notable features

2. Number of Actions:
Count how many numbered waypoints are visible in the image (e.g., if you see markers 1, 2, 3, 4, 5, then there are 5 actions).

3. Target Objects:
List objects that are:
a) Visible in the current scene
b) Not visible but likely to exist in this type of room

For each object, indicate whether it's visible or not.

Please provide your response in the following JSON format:
{
  "scene_description": "detailed description of the scene",
  "num_actions": <number of waypoint markers>,
  "visible_objects": ["object1", "object2", ...],
  "potential_objects": ["object3", "object4", ...]
}

Make sure to output valid JSON format."""
        
        return prompt
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64 URL格式"""
        import base64
        from io import BytesIO
        from PIL import Image
        
        # 读取图片
        image = Image.open(image_path)
        
        # 编码为base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"
    
    def call_vlm(self, image_path: str) -> str:
        """调用VLM分析图片"""
        prompt = self.construct_annotation_prompt()
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.encode_image_to_base64(image_path)
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # 调用VLM
        response = self.vlm.chat(messages)
        return response
    
    def parse_response(self, response: str) -> Dict:
        """
        解析VLM响应，提取JSON数据
        
        Parameters
        ----------
        response : str
            VLM响应文本
            
        Returns
        -------
        Dict
            解析后的标注数据
        """
        # 尝试提取JSON
        # 方法1: 查找JSON代码块
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 方法2: 查找花括号包围的内容
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # 方法3: 整个响应可能就是JSON
                json_str = response
        
        try:
            data = json.loads(json_str)
            
            # 验证必需字段
            required_fields = ['scene_description', 'num_actions', 'visible_objects', 'potential_objects']
            for field in required_fields:
                if field not in data:
                    print(f"警告: 缺少字段 {field}，使用默认值")
                    if field == 'scene_description':
                        data[field] = "No description provided"
                    elif field == 'num_actions':
                        data[field] = 5
                    else:
                        data[field] = []
            
            return data
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"响应内容: {response[:500]}")
            
            # 返回默认结构
            return {
                "scene_description": "Failed to parse response",
                "num_actions": 5,
                "visible_objects": [],
                "potential_objects": [],
                "raw_response": response
            }
    
    def create_test_annotation(self, annotation_data: Dict) -> Dict:
        """
        根据标注数据创建测试用的JSON文件格式
        
        Parameters
        ----------
        annotation_data : Dict
            从VLM获取的标注数据
            
        Returns
        -------
        Dict
            测试用的JSON格式
        """
        # 合并可见和潜在物体作为测试目标
        all_targets = annotation_data['visible_objects'] + annotation_data['potential_objects']
        
        return {
            "scene_description": annotation_data['scene_description'],
            "num_actions": annotation_data['num_actions'],
            "targets": all_targets,
            "visible_objects": annotation_data['visible_objects'],
            "potential_objects": annotation_data['potential_objects']
        }
    
    def annotate_image(self, image_path: str, output_path: str, save_raw: bool = True):
        """
        标注单张图片
        
        Parameters
        ----------
        image_path : str
            图片路径
        output_path : str
            输出JSON路径
        save_raw : bool
            是否保存原始VLM响应
        """
        print(f"处理图片: {Path(image_path).name}")
        
        # 调用VLM
        print("  调用VLM分析...")
        response = self.call_vlm(image_path)
        
        # 解析响应
        print("  解析响应...")
        annotation_data = self.parse_response(response)
        
        # 创建测试标注
        test_annotation = self.create_test_annotation(annotation_data)
        
        # 保存JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_annotation, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 保存标注: {Path(output_path).name}")
        print(f"    场景: {test_annotation['scene_description'][:60]}...")
        print(f"    Action数量: {test_annotation['num_actions']}")
        print(f"    目标数量: {len(test_annotation['targets'])}")
        print(f"    可见物体: {', '.join(test_annotation['visible_objects'][:5])}")
        
        # 保存原始响应（用于调试）
        if save_raw:
            raw_path = output_path.replace('.json', '_raw.txt')
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(response)
        
        return test_annotation
    
    def batch_annotate(self, image_dir: str, output_dir: str, save_raw: bool = True):
        """
        批量标注图片
        
        Parameters
        ----------
        image_dir : str
            图片目录
        output_dir : str
            输出目录
        save_raw : bool
            是否保存原始响应
        """
        image_path = Path(image_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 查找所有图片
        image_files = sorted(list(image_path.glob("*.jpg")) + list(image_path.glob("*.png")))
        
        if not image_files:
            print(f"错误: 在 {image_dir} 中未找到图片文件")
            return
        
        print(f"\n开始批量标注，共 {len(image_files)} 张图片")
        print("=" * 80)
        
        results = []
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]")
            
            # 生成输出路径
            json_file = output_path / f"{img_file.stem}.json"
            
            try:
                # 标注图片
                annotation = self.annotate_image(str(img_file), str(json_file), save_raw)
                results.append({
                    'image': img_file.name,
                    'json': json_file.name,
                    'success': True,
                    'annotation': annotation
                })
                
                # 避免请求过快
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                results.append({
                    'image': img_file.name,
                    'json': json_file.name,
                    'success': False,
                    'error': str(e)
                })
        
        # 保存批量处理摘要
        summary_path = output_path / "annotation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total': len(image_files),
                'success': sum(1 for r in results if r['success']),
                'failed': sum(1 for r in results if not r['success']),
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 80)
        print("批量标注完成！")
        print(f"成功: {sum(1 for r in results if r['success'])}/{len(image_files)}")
        print(f"输出目录: {output_dir}")
        print(f"摘要文件: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='自动标注工具：使用LLM分析图片并生成JSON标注')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='标注图片目录（包含带waypoint标记的图片）')
    parser.add_argument('--output_dir', type=str, default='./annotations',
                        help='输出标注文件目录')
    parser.add_argument('--model', type=str,
                        default='/data/sea_disk0/liujy/models/Qwen/Qwen3-VL-8B-Instruct/',
                        help='VLM模型路径')
    parser.add_argument('--api_key', type=str, default='EMPTY',
                        help='API密钥')
    parser.add_argument('--base_url', type=str, default='http://10.15.89.71:34134/v1/',
                        help='API基础URL')
    parser.add_argument('--timeout', type=int, default=30,
                        help='API超时时间（秒）')
    parser.add_argument('--no_raw', action='store_true',
                        help='不保存原始VLM响应')
    
    args = parser.parse_args()
    
    # VLM配置
    vlm_config = {
        'model': args.model,
        'api_key': args.api_key,
        'base_url': args.base_url,
        'timeout': args.timeout
    }
    
    # 创建标注器并运行
    annotator = AutoAnnotator(vlm_config)
    annotator.batch_annotate(args.image_dir, args.output_dir, save_raw=not args.no_raw)


if __name__ == '__main__':
    main()
