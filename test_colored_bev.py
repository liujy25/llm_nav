#!/usr/bin/env python3
"""
Test colored BEV map functionality
"""
import numpy as np
import cv2
from PIL import Image
from obstacle_map import ObstacleMap

# Load snapshot_0004 data
snapshot_dir = "/home/liujy/mobile_manipulation/model_server/nav/snapshot_0001"

# Load RGB
rgb = np.array(Image.open(f"{snapshot_dir}/rgb.png"))
print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")

# Load Depth
depth_pil = Image.open(f"{snapshot_dir}/depth.png")
depth = np.array(depth_pil).astype(np.float32) / 1000.0  # mm to m
print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")

# Load params
params = np.load(f"{snapshot_dir}/params.npz")
intrinsic = params['intrinsic']
T_cam_odom = params['T_cam_odom']
T_odom_base = params['T_odom_base']

print("\n" + "="*60)
print("Testing Colored BEV Map")
print("="*60)

# Create ObstacleMap
obstacle_map = ObstacleMap(
    min_height=0.15,
    max_height=2.0,
    agent_radius=0.1,
    area_thresh=0.3,
    size=4000,  # Smaller size for testing
    pixels_per_meter=20,
)

# Update obstacle map (this sets episode origin)
print("\n1. Updating obstacle map...")
obstacle_map.update_map(
    depth=depth,
    intrinsic=intrinsic,
    extrinsic=T_cam_odom,
    base_to_odom=T_odom_base,
    min_depth=0.5,
    max_depth=10.0,
)

# Update colored BEV map
print("\n2. Updating colored BEV map...")
obstacle_map.update_colored_bev(
    rgb=rgb,
    depth=depth,
    intrinsic=intrinsic,
    extrinsic=T_cam_odom,
    base_to_odom=T_odom_base,
    min_depth=0.5,
    max_depth=10.0,
)

# Create some test waypoints
robot_pos = T_odom_base[:2, 3]
print(f"\nRobot position: {robot_pos}")

# Generate test waypoints (in a circle around robot)
num_waypoints = 5
angles = np.linspace(0, 2*np.pi, num_waypoints, endpoint=False)
radius = 2.0  # 2 meters
waypoints = np.array([
    robot_pos + radius * np.array([np.cos(a), np.sin(a)])
    for a in angles
])
print(f"Test waypoints:\n{waypoints}")

# Visualize
print("\n3. Generating visualizations...")

# Original obstacle map visualization
vis_obstacle = obstacle_map.visualize(robot_pos=robot_pos, show_frontiers=False)
cv2.imwrite(f"{snapshot_dir}/test_obstacle_map.png", cv2.cvtColor(vis_obstacle, cv2.COLOR_RGB2BGR))
print(f"Saved: {snapshot_dir}/test_obstacle_map.png")

# Colored BEV visualization
vis_colored = obstacle_map.visualize_colored_bev(robot_pos=robot_pos, waypoints=waypoints)
cv2.imwrite(f"{snapshot_dir}/test_colored_bev.png", cv2.cvtColor(vis_colored, cv2.COLOR_RGB2BGR))
print(f"Saved: {snapshot_dir}/test_colored_bev.png")

# Statistics
valid_pixels = np.isfinite(obstacle_map.height_map).sum()
total_pixels = obstacle_map.height_map.size
print(f"\n" + "="*60)
print("Statistics")
print("="*60)
print(f"Valid colored BEV pixels: {valid_pixels} / {total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
print(f"Height range: [{np.nanmin(obstacle_map.height_map):.3f}, {np.nanmax(obstacle_map.height_map):.3f}]")
print(f"RGB map non-zero pixels: {(obstacle_map.rgb_map.sum(axis=2) > 0).sum()}")

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)
