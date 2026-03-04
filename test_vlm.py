#!/usr/bin/env python3
"""
Test script to verify VLM model is working correctly
"""
import numpy as np
from PIL import Image
from vlm import OpenAIVLM

def create_test_image():
    """Create a simple test image"""
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return img

def test_basic_call():
    """Test basic call without history"""
    print("=" * 80)
    print("Test 1: Basic call (no history)")
    print("=" * 80)

    vlm = OpenAIVLM(
        model="/data/sea_disk0/liujy/models/Qwen/Qwen3.5-9B/",
        api_key="EMPTY",
        base_url="http://10.15.89.71:32054/v1/",
        timeout=30
    )

    test_image = create_test_image()
    prompt = "Describe what you see in this image in one sentence."

    print(f"Prompt: {prompt}")
    print("Calling VLM...")

    response = vlm.call([test_image], prompt)

    print(f"Response: {response}")
    print(f"Response type: {type(response)}")
    print(f"Response length: {len(response) if response else 0}")
    print()

    return response

def test_with_different_params():
    """Test with different parameter combinations"""
    print("=" * 80)
    print("Test 2: Testing different parameter combinations")
    print("=" * 80)

    test_image = create_test_image()
    prompt = "What color is dominant in this image?"

    param_sets = [
        {
            "name": "Default (from official example)",
            "params": {
                "temperature": 1.0,
                "top_p": 0.95,
                "presence_penalty": 1.5,
                "extra_body": {"top_k": 20}
            }
        },
        {
            "name": "Conservative",
            "params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "presence_penalty": 0.0,
            }
        },
        {
            "name": "Minimal",
            "params": {
                "temperature": 1.0,
            }
        },
    ]

    for param_set in param_sets:
        print(f"\nTesting: {param_set['name']}")
        print(f"Parameters: {param_set['params']}")

        from openai import OpenAI
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://10.15.89.71:32054/v1/",
            timeout=30
        )

        from utils import append_mime_tag, encode_image_b64

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": append_mime_tag(encode_image_b64(Image.fromarray(test_image, mode='RGB')))
                    }
                }
            ]
        }]

        try:
            response = client.chat.completions.create(
                model="/data/sea_disk0/liujy/models/Qwen/Qwen3.5-9B/",
                messages=messages,
                max_tokens=100,
                **param_set['params']
            )

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            matched_stop = getattr(response.choices[0], 'matched_stop', None)

            print(f"  Response: {content}")
            print(f"  Finish reason: {finish_reason}")
            print(f"  Matched stop: {matched_stop}")
            print(f"  Tokens: {response.usage.completion_tokens}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print()

def test_navigation_prompt():
    """Test with actual navigation prompt format"""
    print("=" * 80)
    print("Test 3: Navigation prompt format")
    print("=" * 80)

    vlm = OpenAIVLM(
        model="/data/sea_disk0/liujy/models/Qwen/Qwen3.5-9B/",
        api_key="EMPTY",
        base_url="http://10.15.89.71:32054/v1/",
        timeout=30
    )

    test_image = create_test_image()

    prompt = """--- Iteration 1 ---

Current observation:

RGB Image (First): No MOVE waypoints available (only TURN actions)

BEV Map (Second)

Available actions:
- MOVE: None available
- TURN: -1 (left 30.0°)2 (right 30.0°)

Recent actions: None (first iteration)

================================================================================

Respond in this format:

**Analysis:**
1. What do you observe in RGB and BEV?
2. Is target "sofa" visible? Where are unexplored areas (GRAY)?
3. What is the best action to find "sofa"?

**Decision:** {'action': <number>}"""

    print("Calling VLM with navigation prompt...")
    response = vlm.call([test_image], prompt)

    print(f"Response: {response}")
    print(f"Response length: {len(response) if response else 0}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("VLM Model Test Script")
    print("=" * 80 + "\n")

    try:
        # Test 1: Basic call
        test_basic_call()

        # Test 2: Different parameters
        test_with_different_params()

        # Test 3: Navigation prompt
        test_navigation_prompt()

        print("=" * 80)
        print("All tests completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
