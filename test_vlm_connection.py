#!/usr/bin/env python3
"""
Test VLM connection with actual NavAgent configuration
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vlm import OpenAIVLM
import numpy as np
from PIL import Image

# Use same config as nav_server.py
config = {
    'vlm_model': '/data/sea_disk0/liujy/models/Qwen/Qwen3.5-27B-GPTQ-Int4/',
    'vlm_api_key': 'EMPTY',
    'vlm_base_url': 'http://10.15.89.71:34134/v1/',
    'vlm_timeout': 600,
}

print("=" * 60)
print("Testing VLM Connection")
print("=" * 60)
print(f"Model: {config['vlm_model']}")
print(f"Base URL: {config['vlm_base_url']}")
print(f"Timeout: {config['vlm_timeout']}s")
print()

# Create VLM instance
print("Creating VLM instance...")
vlm = OpenAIVLM(
    model=config['vlm_model'],
    system_instruction="You are a helpful assistant.",
    api_key=config['vlm_api_key'],
    base_url=config['vlm_base_url'],
    timeout=config['vlm_timeout']
)
print("VLM instance created successfully.")
print()

# Create a test image
print("Creating test image...")
test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
print(f"Test image shape: {test_image.shape}")
print()

# Test 1: Simple text-only call
print("Test 1: Text-only call")
print("-" * 60)
try:
    response = vlm.call([], "Hello, can you hear me?")
    print(f"Response: {response}")
    print("SUCCESS!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 2: Call with image
print("Test 2: Call with image")
print("-" * 60)
try:
    response = vlm.call([test_image], "What do you see in this image?")
    print(f"Response: {response}")
    print("SUCCESS!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 3: Chat with history
print("Test 3: Chat with history")
print("-" * 60)
try:
    vlm.reset(initial_prompt="You are navigating a robot.")
    response = vlm.call_chat(history=5, images=[test_image], text_prompt="Where should I go?")
    print(f"Response: {response}")
    print("SUCCESS!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
print()

print("=" * 60)
print("Test completed")
print("=" * 60)
