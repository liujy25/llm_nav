#!/usr/bin/env python3
"""
Test script for HabitatNavAgent integration in habitat_server.py
"""

import numpy as np
import requests
import json
import cv2
from PIL import Image
import io

# Server configuration
SERVER_URL = "http://10.19.126.158:1874"

def test_navigation_reset():
    """Test /navigation_reset endpoint"""
    print("=" * 60)
    print("Testing /navigation_reset")
    print("=" * 60)

    url = f"{SERVER_URL}/navigation_reset"
    data = {
        "goal": "chair",
        "goal_description": "a red chair in the living room",
        "confidence_threshold": 0.7
    }

    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_navigation_step():
    """Test /navigation_step endpoint"""
    print("\n" + "=" * 60)
    print("Testing /navigation_step")
    print("=" * 60)

    # Create mock RGB and depth images
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)

    # Add some obstacles
    depth[200:280, 300:340] = 0.8
    depth[100:150, 100:200] = 1.5

    # Convert to image files
    rgb_pil = Image.fromarray(rgb)
    rgb_bytes = io.BytesIO()
    rgb_pil.save(rgb_bytes, format='JPEG')
    rgb_bytes.seek(0)

    depth_pil = Image.fromarray((depth * 1000).astype(np.uint16))
    depth_bytes = io.BytesIO()
    depth_pil.save(depth_bytes, format='PNG')
    depth_bytes.seek(0)

    # Camera intrinsic
    intrinsic = np.array([
        [320, 0, 320],
        [0, 320, 240],
        [0, 0, 1]
    ], dtype=np.float32)

    # Pose matrices
    T_cam_odom = np.eye(4, dtype=np.float32)
    T_odom_base = np.eye(4, dtype=np.float32)
    T_odom_base[0, 3] = 1.0  # x = 1.0m
    T_odom_base[1, 3] = 0.5  # y = 0.5m

    # Prepare form data
    files = {
        'rgb': ('rgb.jpg', rgb_bytes, 'image/jpeg'),
        'depth': ('depth.png', depth_bytes, 'image/png')
    }

    form_data = {
        'intrinsic': json.dumps(intrinsic.flatten().tolist()),
        'T_cam_odom': json.dumps(T_cam_odom.flatten().tolist()),
        'T_odom_base': json.dumps(T_odom_base.flatten().tolist()),
        'is_keyframe': 'true'
    }

    url = f"{SERVER_URL}/navigation_step"
    response = requests.post(url, files=files, data=form_data)

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Action Type: {result.get('action_type')}")
        print(f"Finished: {result.get('finished')}")
        print(f"Iteration: {result.get('iteration')}")
        if result.get('goal_pose'):
            pose = result['goal_pose']['pose']
            print(f"Goal Position: ({pose['position']['x']:.2f}, {pose['position']['y']:.2f}, {pose['position']['z']:.2f})")
        print(f"Timing: {json.dumps(result.get('timing'), indent=2)}")
    else:
        print(f"Error: {response.text}")

    return response.status_code == 200


def test_health():
    """Test /health endpoint"""
    print("\n" + "=" * 60)
    print("Testing /health")
    print("=" * 60)

    url = f"{SERVER_URL}/health"
    response = requests.get(url)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def main():
    """Run all tests"""
    print("=" * 60)
    print("HabitatNavAgent Integration Test")
    print("=" * 60)
    print(f"Server URL: {SERVER_URL}")
    print()

    # Test health first
    if not test_health():
        print("\n[ERROR] Server health check failed!")
        return

    # Test navigation reset
    if not test_navigation_reset():
        print("\n[ERROR] Navigation reset failed!")
        return

    # Test navigation step
    for i in range(3):
        print(f"\n--- Navigation Step {i+1} ---")
        if not test_navigation_step():
            print(f"\n[ERROR] Navigation step {i+1} failed!")
            break

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
