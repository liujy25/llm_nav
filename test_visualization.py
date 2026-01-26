#!/usr/bin/env python3
"""
Test script to visualize action annotations without running VLM inference.
This tests the turn left/right buttons and move action arrows.
"""
import pickle
import cv2
import numpy as np
from nav_agent import NavAgent


def test_visualization(data_path='./test_data.pkl'):
    """Test the action visualization"""
    
    # Load test data
    print(f"Loading test data from: {data_path}")
    try:
        data = pickle.load(open(data_path, 'rb'))
        print("✓ Test data loaded successfully")
    except FileNotFoundError:
        print(f"✗ Test data not found at {data_path}")
        print("Please provide a valid path to test data")
        return
    
    # Initialize agent
    print("\nInitializing NavAgent...")
    cfg = {
        'turn_angle_deg': 30.0,
        'clip_dist': 2.0,
        'num_theta': 40,
    }
    agent = NavAgent(cfg=cfg)
    agent.reset(goal='test_object')
    print(f"✓ Agent initialized with turn_angle={cfg['turn_angle_deg']}°")
    
    # Test visualization only (no VLM call)
    print("\nGenerating action visualization...")
    
    # Step 1: Get navigability
    a_initial = agent._navigability(data)
    print(f"✓ Navigability computed: {len(a_initial)} initial directions")
    
    # Step 2: Propose actions
    a_final = agent._action_proposer(a_initial, data['base_to_odom_matrix'])
    print(f"✓ Actions proposed: {len(a_final)} final actions")
    
    # Step 3: Project onto image (without chosen action)
    a_final_projected, rgb_vis = agent._projection(a_final, data)
    print(f"✓ Actions projected to image: {len(a_final_projected)} visible actions")
    print(f"  Action mapping: {a_final_projected}")
    
    # Save visualization
    output_path = '/tmp/nav_action_visualization_no_selection.png'
    bgr_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_vis)
    print(f"\n✓ Image saved to: {output_path}")
    
    # Test with a selected action (highlight action 1)
    if len(a_final_projected) > 0:
        test_action = 1
        print(f"\nGenerating visualization with selected action {test_action}...")
        _, rgb_vis_selected = agent._projection(a_final, data, chosen_action=test_action)
        output_path_selected = '/tmp/nav_action_visualization_action1_selected.png'
        bgr_vis_selected = cv2.cvtColor(rgb_vis_selected, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path_selected, bgr_vis_selected)
        print(f"✓ Image with action {test_action} selected saved to: {output_path_selected}")
    
    # Test turn left selected
    print(f"\nGenerating visualization with TURN LEFT (-1) selected...")
    _, rgb_vis_left = agent._projection(a_final, data, chosen_action=-1)
    output_path_left = '/tmp/nav_action_visualization_turn_left_selected.png'
    bgr_vis_left = cv2.cvtColor(rgb_vis_left, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path_left, bgr_vis_left)
    print(f"✓ Image with TURN LEFT selected saved to: {output_path_left}")
    
    # Test turn right selected
    print(f"\nGenerating visualization with TURN RIGHT (-2) selected...")
    _, rgb_vis_right = agent._projection(a_final, data, chosen_action=-2)
    output_path_right = '/tmp/nav_action_visualization_turn_right_selected.png'
    bgr_vis_right = cv2.cvtColor(rgb_vis_right, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path_right, bgr_vis_right)
    print(f"✓ Image with TURN RIGHT selected saved to: {output_path_right}")
    
    print("\n" + "="*60)
    print("VISUALIZATION TEST COMPLETE")
    print("="*60)
    print("\nGenerated images:")
    print("  1. No selection: /tmp/nav_action_visualization_no_selection.png")
    if len(a_final_projected) > 0:
        print("  2. Action 1 selected: /tmp/nav_action_visualization_action1_selected.png")
    print("  3. Turn left selected: /tmp/nav_action_visualization_turn_left_selected.png")
    print("  4. Turn right selected: /tmp/nav_action_visualization_turn_right_selected.png")
    print("\nExpected visualization:")
    print("  - Red arrows with numbered circles (1, 2, 3, ...) for forward actions")
    print("  - Left side: '-1' button with 'TURN LEFT 30°' label")
    print("  - Right side: '-2' button with 'TURN RIGHT 30°' label")
    print("  - Selected action should have GREEN circle background")


if __name__ == "__main__":
    import sys
    
    # Allow custom data path from command line
    if len(sys.argv) > 1:
        test_visualization(sys.argv[1])
    else:
        test_visualization()

