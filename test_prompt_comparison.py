#!/usr/bin/env python3
"""
测试脚本：比较关键帧prompt和非关键帧prompt的LLM输出差异

使用方式：
1. 准备测试数据文件夹，包含：
   - 图片文件（如 image_001.jpg）
   - 对应的JSON标注文件（如 image_001.json），格式：
     {
       "targets": ["目标1", "目标2", ...],
       "waypoints": [...] (可选)
     }

2. 运行脚本：
   python test_prompt_comparison.py --data_dir /path/to/test_data --output_dir ./results
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import time
from datetime import datetime

from vlm import OpenAIVLM


class PromptComparisonTester:
    """比较两种prompt模式的测试器"""
    
    def __init__(self, vlm_config: Dict):
        """
        初始化测试器
        
        Parameters
        ----------
        vlm_config : Dict
            VLM配置，包含model, api_key, base_url, timeout等
        """
        self.vlm_config = vlm_config
        self.turn_angle_deg = 30.0  # 默认转向角度
        
        # 初始化VLM
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You are only allowed to search for the goal object in the room you are in now. "
            "You cannot go to other rooms. You cannot move through doors."
        )
        self.vlm = OpenAIVLM(
            model=vlm_config['model'],
            system_instruction=system_instruction,
            api_key=vlm_config['api_key'],
            base_url=vlm_config['base_url'],
            timeout=vlm_config['timeout']
        )
        
    def construct_keyframe_prompt(self, goal: str, num_actions: int, iter: int = 1) -> str:
        """构建关键帧prompt（深度推理版本）"""
        prompt = f"""--- Iteration {iter} (KEYFRAME #1) ---

CRITICAL DECISION POINT: This is a keyframe where you need to make a strategic decision.

Current observation: {num_actions} available MOVE waypoints shown (numbered 1..{num_actions}).
Additionally, there are always two TURN actions:
- Action -1: TURN LEFT by {self.turn_angle_deg:.0f} degrees in place
- Action -2: TURN RIGHT by {self.turn_angle_deg:.0f} degrees in place

================================================================================

Please think step by step and answer the following questions. 
In your response, you should: first, give your answer of all the questions, 
then provide your final decision in this exact format: {{'action': <action_number>}}.

Question 1: Historical Review
This is the first keyframe. What is your initial exploration strategy?

Question 2: Current Observation
What do you see in the current image?
- Navigable areas (floor, corridors, open spaces)
- Obstacles (walls, furniture, objects)
- Potential target objects related to "{goal}"
- Unexplored directions

Question 3: Strategic Reasoning
Based on your current observation:
- Which areas are most promising for finding "{goal}"?
- What is your exploration plan?

Question 4: Action Selection
Which action number achieves your strategic goal best?
Consider all available actions (1 to {num_actions}, -1, -2).

Question 5: Action Justification
Why did you choose this specific action?
- How does it align with your strategy from Q3?
- What do you expect to discover or achieve?
"""
        return prompt
    
    def construct_nonkeyframe_prompt(self, goal: str, num_actions: int, iter: int = 1) -> str:
        """构建非关键帧prompt（快速执行版本）"""
        prompt = f"""--- Iteration {iter} (NON-KEYFRAME, step 1 in current cycle) ---

EXECUTION MODE: Continue following the strategy from keyframe #1 (iter 1).

Current observation: {num_actions} available MOVE waypoints shown (numbered 1..{num_actions}).
Additionally, there are always two TURN actions:
- Action -1: TURN LEFT by {self.turn_angle_deg:.0f} degrees in place
- Action -2: TURN RIGHT by {self.turn_angle_deg:.0f} degrees in place

================================================================================

CONTEXT REVIEW:
This is the first action after the initial keyframe. Continue with the exploration strategy.

Based on the parent keyframe's strategy and your current observation, select the best action to continue the plan.

================================================================================

Provide your action selection in this exact format:
{{'action': <action_number>}}
"""
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
    
    def call_vlm(self, image_path: str, prompt: str) -> Tuple[str, float]:
        """
        调用VLM获取响应
        
        Returns
        -------
        Tuple[str, float]
            (响应文本, 推理时间)
        """
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
        start_time = time.time()
        response = self.vlm.chat(messages)
        inference_time = time.time() - start_time
        
        return response, inference_time
    
    def extract_action(self, response: str) -> int:
        """从响应中提取action编号"""
        import re
        
        # 尝试匹配 {'action': N} 或 {"action": N}
        patterns = [
            r"['\"]action['\"]:\s*(-?\d+)",
            r"action['\"]?\s*:\s*(-?\d+)",
            r"选择.*?动作\s*(-?\d+)",
            r"Action\s*(-?\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def test_single_image(self, image_path: str, targets: List[str], num_actions: int = 5) -> Dict:
        """
        测试单张图片的所有目标
        
        Parameters
        ----------
        image_path : str
            图片路径
        targets : List[str]
            目标列表
        num_actions : int
            可用的waypoint数量
            
        Returns
        -------
        Dict
            测试结果
        """
        results = {
            'image': image_path,
            'num_actions': num_actions,
            'targets': []
        }
        
        for target in targets:
            print(f"  测试目标: {target}")
            
            # 1. 关键帧prompt测试
            kf_prompt = self.construct_keyframe_prompt(target, num_actions)
            kf_response, kf_time = self.call_vlm(image_path, kf_prompt)
            kf_action = self.extract_action(kf_response)
            
            print(f"    关键帧模式: action={kf_action}, time={kf_time:.2f}s")
            
            # 2. 非关键帧prompt测试
            nkf_prompt = self.construct_nonkeyframe_prompt(target, num_actions)
            nkf_response, nkf_time = self.call_vlm(image_path, nkf_prompt)
            nkf_action = self.extract_action(nkf_response)
            
            print(f"    非关键帧模式: action={nkf_action}, time={nkf_time:.2f}s")
            
            # 记录结果
            target_result = {
                'target': target,
                'keyframe': {
                    'prompt': kf_prompt,
                    'response': kf_response,
                    'action': kf_action,
                    'inference_time': kf_time
                },
                'non_keyframe': {
                    'prompt': nkf_prompt,
                    'response': nkf_response,
                    'action': nkf_action,
                    'inference_time': nkf_time
                },
                'action_match': kf_action == nkf_action,
                'action_diff': None if kf_action == nkf_action else f"{kf_action} vs {nkf_action}"
            }
            
            results['targets'].append(target_result)
        
        return results
    
    def run_tests(self, data_dir: str, output_dir: str):
        """
        运行所有测试
        
        Parameters
        ----------
        data_dir : str
            测试数据目录
        output_dir : str
            输出结果目录
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 查找所有图片和对应的JSON文件
        image_files = sorted(data_path.glob("*.jpg")) + sorted(data_path.glob("*.png"))
        
        all_results = []
        stats = {
            'total_tests': 0,
            'action_matches': 0,
            'action_diffs': 0,
            'total_kf_time': 0.0,
            'total_nkf_time': 0.0
        }
        
        print(f"\n开始测试，共找到 {len(image_files)} 张图片")
        print("=" * 80)
        
        for img_file in image_files:
            json_file = img_file.with_suffix('.json')
            
            if not json_file.exists():
                print(f"警告: 未找到 {json_file}，跳过")
                continue
            
            print(f"\n处理图片: {img_file.name}")
            
            # 读取JSON标注
            with open(json_file, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            
            targets = annotation.get('targets', [])
            num_actions = annotation.get('num_actions', 5)
            
            if not targets:
                print(f"  警告: 没有目标，跳过")
                continue
            
            # 测试这张图片
            result = self.test_single_image(str(img_file), targets, num_actions)
            all_results.append(result)
            
            # 更新统计
            for target_result in result['targets']:
                stats['total_tests'] += 1
                if target_result['action_match']:
                    stats['action_matches'] += 1
                else:
                    stats['action_diffs'] += 1
                stats['total_kf_time'] += target_result['keyframe']['inference_time']
                stats['total_nkf_time'] += target_result['non_keyframe']['inference_time']
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"comparison_results_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': all_results,
                'statistics': stats
            }, f, ensure_ascii=False, indent=2)
        
        # 生成统计报告
        self.generate_report(all_results, stats, output_path / f"report_{timestamp}.txt")
        
        print("\n" + "=" * 80)
        print("测试完成！")
        print(f"详细结果已保存到: {result_file}")
        
    def generate_report(self, results: List[Dict], stats: Dict, report_path: Path):
        """生成统计报告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Prompt模式对比测试报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 总体统计
            f.write("总体统计:\n")
            f.write(f"  总测试数: {stats['total_tests']}\n")
            f.write(f"  Action一致: {stats['action_matches']} ({stats['action_matches']/stats['total_tests']*100:.1f}%)\n")
            f.write(f"  Action不同: {stats['action_diffs']} ({stats['action_diffs']/stats['total_tests']*100:.1f}%)\n")
            f.write(f"  关键帧平均推理时间: {stats['total_kf_time']/stats['total_tests']:.2f}s\n")
            f.write(f"  非关键帧平均推理时间: {stats['total_nkf_time']/stats['total_tests']:.2f}s\n")
            f.write("\n")
            
            # 详细差异列表
            f.write("Action不同的案例:\n")
            f.write("-" * 80 + "\n")
            
            for result in results:
                for target_result in result['targets']:
                    if not target_result['action_match']:
                        f.write(f"\n图片: {Path(result['image']).name}\n")
                        f.write(f"目标: {target_result['target']}\n")
                        f.write(f"关键帧Action: {target_result['keyframe']['action']}\n")
                        f.write(f"非关键帧Action: {target_result['non_keyframe']['action']}\n")
                        f.write(f"关键帧响应摘要: {target_result['keyframe']['response'][:200]}...\n")
                        f.write(f"非关键帧响应摘要: {target_result['non_keyframe']['response'][:200]}...\n")
                        f.write("-" * 80 + "\n")
        
        print(f"统计报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='测试关键帧和非关键帧prompt的LLM输出差异')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='测试数据目录（包含图片和JSON标注）')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='输出结果目录')
    parser.add_argument('--model', type=str, 
                        default='/data/sea_disk0/liujy/models/Qwen/Qwen3-VL-8B-Instruct/',
                        help='VLM模型路径')
    parser.add_argument('--api_key', type=str, default='EMPTY',
                        help='API密钥')
    parser.add_argument('--base_url', type=str, default='http://10.15.89.71:34134/v1/',
                        help='API基础URL')
    parser.add_argument('--timeout', type=int, default=30,
                        help='API超时时间（秒）')
    
    args = parser.parse_args()
    
    # VLM配置
    vlm_config = {
        'model': args.model,
        'api_key': args.api_key,
        'base_url': args.base_url,
        'timeout': args.timeout
    }
    
    # 创建测试器并运行
    tester = PromptComparisonTester(vlm_config)
    tester.run_tests(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
