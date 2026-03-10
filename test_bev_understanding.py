#!/usr/bin/env python3
"""
Test script to chat with VLM using RGB and BEV images.
Load RGB visualization and BEV map, then chat with VLM using the same logic as nav_agent.
"""

import os
import sys
import cv2
import json
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vlm import OpenAIVLM


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file.

    Parameters
    ----------
    image_path : str
        Path to image file

    Returns
    -------
    np.ndarray
        Image in RGB format
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb


def load_prompts(prompts_path: str) -> dict:
    """
    Load prompts from JSON file.

    Parameters
    ----------
    prompts_path : str
        Path to prompts.json

    Returns
    -------
    dict
        Prompts dictionary
    """
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    return prompts


def construct_iteration_prompt(
    prompts: dict,
    iter_num: int,
    num_actions: int,
    goal: str,
    enable_bev: bool = True,
    turn_angle: int = 30,
    current_trajectory_index: int = None,
) -> str:
    """
    Construct iteration prompt similar to nav_agent.

    Parameters
    ----------
    prompts : dict
        Prompts dictionary from prompts.json
    iter_num : int
        Current iteration number
    num_actions : int
        Number of available MOVE actions
    goal : str
        Navigation goal
    enable_bev : bool
        Whether BEV map is enabled
    turn_angle : int
        Turn angle in degrees
    current_trajectory_index : int
        Current trajectory point index

    Returns
    -------
    str
        Constructed prompt
    """
    # Select template based on enable_bev
    if enable_bev:
        template_lines = prompts['iteration_prompt_template']
    else:
        template_lines = prompts['iteration_prompt_template_no_bev']

    # Turn actions text (simplified: assume both directions available)
    turn_actions_text = f"- TURN: -1 (left {turn_angle}°) or -2 (right {turn_angle}°)"

    # Current position text
    if current_trajectory_index is not None:
        if enable_bev:
            current_position_text = f"Current robot position: Trajectory point {current_trajectory_index} (marked with red circle in BEV map)"
        else:
            current_position_text = f"Current robot position: Trajectory point {current_trajectory_index}"
    else:
        current_position_text = "Current robot position: Starting position"

    prompt = '\n'.join(template_lines).format(
        iter=iter_num,
        num_actions=num_actions,
        turn_angle=turn_angle,
        goal=goal,
        turn_actions=turn_actions_text,
        current_position=current_position_text
    )

    return prompt


def test_vlm_chat(
    rgb_path: str,
    bev_path: str = None,
    num_actions: int = 5,
    goal: str = "chair",
    goal_description: str = "",
    iter_num: int = 0,
    current_trajectory_index: int = 1,
    enable_bev: bool = True,
    turn_angle: int = 30,
    vlm_model: str = '/data/sea_disk0/liujy/models/Qwen/Qwen3.5-27B-GPTQ-Int4/',
    vlm_base_url: str = 'http://10.15.89.71:34134/v1/',
    vlm_api_key: str = 'EMPTY',
    vlm_timeout: int = 120,
    history: int = 5,
):
    """
    Test VLM chat with RGB and BEV images.

    Parameters
    ----------
    rgb_path : str
        Path to RGB visualization image
    bev_path : str, optional
        Path to BEV map image (required if enable_bev=True)
    num_actions : int
        Number of available MOVE actions
    goal : str
        Navigation goal
    goal_description : str
        Additional goal description
    iter_num : int
        Current iteration number
    current_trajectory_index : int
        Current trajectory point index
    enable_bev : bool
        Whether to use BEV map
    turn_angle : int
        Turn angle in degrees
    vlm_model : str
        VLM model name/path
    vlm_base_url : str
        VLM API base URL
    vlm_api_key : str
        VLM API key
    vlm_timeout : int
        Request timeout in seconds
    history : int
        Number of conversation rounds to keep in context
    """
    print("="*80)
    print("TESTING VLM CHAT")
    print("="*80)
    print(f"RGB image: {rgb_path}")
    if enable_bev and bev_path:
        print(f"BEV image: {bev_path}")
    print(f"Goal: {goal}")
    print(f"Number of actions: {num_actions}")
    print(f"Enable BEV: {enable_bev}")
    print(f"VLM model: {vlm_model}")
    print(f"History: {history} rounds")
    print("="*80 + "\n")

    # Load prompts
    prompts_path = os.path.join(os.path.dirname(__file__), 'prompts.json')
    prompts = load_prompts(prompts_path)

    # Initialize VLM
    system_instruction = prompts['system_instruction']
    vlm = OpenAIVLM(
        model=vlm_model,
        system_instruction=system_instruction,
        api_key=vlm_api_key,
        base_url=vlm_base_url,
        timeout=vlm_timeout
    )

    # Build initial prompt
    if enable_bev:
        initial_template = prompts['initial_prompt_template']
    else:
        initial_template = prompts['initial_prompt_template_no_bev']

    initial_prompt = '\n'.join(initial_template).format(
        goal=goal,
        description_text=goal_description if goal_description else "",
        turn_angle=turn_angle
    )

    # Reset VLM with initial prompt
    vlm.reset(initial_prompt=initial_prompt)
    print("VLM initialized with task briefing\n")

    # Load RGB image
    try:
        rgb_image = load_image(rgb_path)
        print(f"Loaded RGB image: {rgb_image.shape}")
    except Exception as e:
        print(f"ERROR loading RGB: {e}")
        return None

    # Prepare images list
    images = [rgb_image]

    # Load BEV image if enabled
    if enable_bev:
        if bev_path is None:
            print("ERROR: enable_bev=True but bev_path is None")
            return None
        try:
            bev_image = load_image(bev_path)
            images.append(bev_image)
            print(f"Loaded BEV image: {bev_image.shape}")
        except Exception as e:
            print(f"ERROR loading BEV: {e}")
            return None

    # Construct iteration prompt
    prompt = construct_iteration_prompt(
        prompts=prompts,
        iter_num=iter_num,
        num_actions=num_actions,
        goal=goal,
        enable_bev=enable_bev,
        turn_angle=turn_angle,
        current_trajectory_index=current_trajectory_index,
    )

    print(f"\nIteration Prompt:\n{'-'*80}\n{prompt}\n{'-'*80}\n")
    print("Calling VLM...\n")

    # Call VLM
    try:
        response = vlm.call_chat(
            history=history,
            images=images,
            text_prompt=prompt
        )

        print(f"VLM Response:\n{'='*80}\n{response}\n{'='*80}\n")

        return response

    except Exception as e:
        print(f"ERROR: VLM call failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main function with configuration.
    """
    # ========== CONFIGURATION ==========
    # Image paths
    RGB_PATH = "/home/liujy/mobile_manipulation/model_server/logs/run_20260309_195808/iter_0022_rgb_vis.jpg"
    BEV_PATH = "/home/liujy/mobile_manipulation/model_server/logs/run_20260309_195808/iter_0022_bev_map.jpg"

    # Navigation parameters
    NUM_ACTIONS = 1  # Number of available MOVE actions
    GOAL = "sofa"
    GOAL_DESCRIPTION = ""
    ITER_NUM = 6
    CURRENT_TRAJECTORY_INDEX = 6  # Current trajectory point number
    TURN_ANGLE = 30

    # BEV control
    ENABLE_BEV = True  # Set to False to test without BEV

    # VLM configuration
    VLM_MODEL = '/data/sea_disk0/liujy/models/Qwen/Qwen3.5-27B-GPTQ-Int4/'
    VLM_BASE_URL = 'http://10.15.89.71:34134/v1/'
    VLM_API_KEY = 'EMPTY'
    VLM_TIMEOUT = 120
    HISTORY = 5  # Number of conversation rounds to keep

    # ========== RUN TEST ==========
    response = test_vlm_chat(
        rgb_path=RGB_PATH,
        bev_path=BEV_PATH if ENABLE_BEV else None,
        num_actions=NUM_ACTIONS,
        goal=GOAL,
        goal_description=GOAL_DESCRIPTION,
        iter_num=ITER_NUM,
        current_trajectory_index=CURRENT_TRAJECTORY_INDEX,
        enable_bev=ENABLE_BEV,
        turn_angle=TURN_ANGLE,
        vlm_model=VLM_MODEL,
        vlm_base_url=VLM_BASE_URL,
        vlm_api_key=VLM_API_KEY,
        vlm_timeout=VLM_TIMEOUT,
        history=HISTORY,
    )

    if response:
        print("\nTest completed successfully!")
        return 0
    else:
        print("\nTest failed!")
        return 1


if __name__ == "__main__":
    exit(main())
