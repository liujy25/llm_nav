#!/usr/bin/env python3
"""
测试stop检测功能
测试server的 /check_stop endpoint和VLM的判断能力
"""
import os
import sys
import argparse
import requests
from PIL import Image
from io import BytesIO

def test_stop_check(server_url: str, image_path: str, goal: str, goal_description: str = ""):
    """
    测试stop检测
    
    Args:
        server_url: Server URL (e.g., http://10.19.126.158:1874)
        image_path: 测试图片路径
        goal: 导航目标 (e.g., "door", "chair")
        goal_description: 目标描述 (可选)
    """
    server_url = server_url.rstrip('/')
    
    print("=" * 60)
    print("Stop Detection Test")
    print("=" * 60)
    print(f"Server: {server_url}")
    print(f"Goal: {goal}")
    if goal_description:
        print(f"Description: {goal_description}")
    print(f"Test Image: {image_path}")
    print("=" * 60)
    
    # Step 1: Initialize navigation (必须先reset)
    print("\n[1/3] Initializing navigation...")
    reset_url = f'{server_url}/navigation_reset'
    reset_data = {
        'goal': goal,
        'goal_description': goal_description,
        'confidence_threshold': 0.5
    }
    
    try:
        response = requests.post(reset_url, json=reset_data, timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"Navigation initialized: {result}")
    except Exception as e:
        print(f"Failed to initialize navigation: {e}")
        return
    
    # Step 2: Load test image
    print(f"\n[2/3] Loading test image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    try:
        # Load and prepare image
        img = Image.open(image_path).convert('RGB')
        print(f"Image loaded: {img.size[0]}x{img.size[1]}")
        
        # Convert to JPEG bytes
        img_buffer = BytesIO()
        img.save(img_buffer, format='JPEG', quality=85)
        img_buffer.seek(0)
    except Exception as e:
        print(f"Failed to load image: {e}")
        return
    
    # Step 3: Call /check_stop
    print(f"\n[3/3] Calling /check_stop endpoint...")
    check_url = f'{server_url}/check_stop'
    files = {'rgb': ('test.jpg', img_buffer, 'image/jpeg')}
    
    try:
        print("Sending request to VLM (this may take a few seconds)...")
        response = requests.post(check_url, files=files, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        
        should_stop = result.get('should_stop', False)
        raw_response = result.get('raw_response', '')
        
        if should_stop:
            print("VLM判断: 应该停止 (STOP)")
        else:
            print("VLM判断: 继续前进 (CONTINUE)")
        
        print(f"\nVLM原始回复: \"{raw_response}\"")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nRequest failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                print(f"Server error: {error_data}")
            except:
                print(f"Server response: {e.response.text}")


def main():
    parser = argparse.ArgumentParser(description='Test Stop Detection')
    parser.add_argument('--server', type=str, default='http://10.19.126.158:1874',
                        help='Server URL')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--goal', type=str, required=True,
                        help='Navigation goal (e.g., "door", "chair")')
    parser.add_argument('--goal-description', type=str, default='',
                        help='Optional goal description')
    
    args = parser.parse_args()
    
    test_stop_check(
        server_url=args.server,
        image_path=args.image,
        goal=args.goal,
        goal_description=args.goal_description
    )


if __name__ == '__main__':
    main()

