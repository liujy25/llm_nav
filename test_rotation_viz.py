#!/usr/bin/env python3
import cv2
import numpy as np
from nav_agent import NavAgent

rgb_path = "/home/liujy/mobile_manipulation/model_server/logs/run_20260115_233019/iter_0001_rgb.jpg"
rgb = cv2.imread(rgb_path)
H, W = rgb.shape[:2]

agent = NavAgent()
obs = {
    'rgb': cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
    'depth': np.ones((H, W), dtype=np.float32) * 2.0,
    'intrinsic': np.array([[500, 0, W/2], [0, 500, H/2], [0, 0, 1]], dtype=np.float32),
    'extrinsic': np.eye(4, dtype=np.float32),
    'base_to_odom_matrix': np.eye(4, dtype=np.float32)
}

a_final = [(1.5, -0.3), (1.2, 0.0), (1.8, 0.3)]
projected, rgb_vis = agent._projection(a_final, obs, chosen_action=None, allow_turnaround=True)

bgr_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
output_path = "/home/liujy/mobile_manipulation/model_server/nav/test_rotation_output.jpg"
cv2.imwrite(output_path, bgr_vis)
print(f"✅ Saved: {output_path}")

