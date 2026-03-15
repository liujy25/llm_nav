#!/usr/bin/env python3
"""
Debug test script for ObstacleMap BEV maintenance.
Replays testdata frames and saves intermediate visualizations.

Diagnoses:
1. Explored area leaking through walls (穿墙)
2. Door blocked after obstacle dilation
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

from obstacle_map import ObstacleMap
from geometry_utils import get_point_cloud, transform_points, filter_points_by_height, extract_yaw

TESTDATA_DIR = Path("./testdata")
OUTPUT_DIR = Path("./debug_output")


def load_frame(req_dir: Path):
    """Load a single test frame."""
    with open(req_dir / "metadata.json") as f:
        meta = json.load(f)

    # Load depth as float32 meters (16-bit PNG, millimeters)
    depth_raw = cv2.imread(str(req_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"depth.png not found in {req_dir}")
    depth = depth_raw.astype(np.float32) / 1000.0  # mm -> m

    rgb = cv2.imread(str(req_dir / "rgb.jpg"))

    intrinsic = np.array(meta["intrinsic"])
    T_cam_odom = np.array(meta["T_cam_odom"])
    T_odom_base = np.array(meta["T_odom_base"])

    return depth, rgb, intrinsic, T_cam_odom, T_odom_base, meta


def crop_around_center(img, center_px, half_size=300):
    """Crop a region around center for visualization."""
    cx, cy = int(center_px[0]), int(center_px[1])
    h, w = img.shape[:2]
    x0 = max(0, cx - half_size)
    x1 = min(w, cx + half_size)
    y0 = max(0, cy - half_size)
    y1 = min(h, cy + half_size)
    return img[y0:y1, x0:x1]


def visualize_map(bool_map, robot_px, label, half_size=300):
    """Convert bool map to BGR visualization cropped around robot."""
    vis = np.zeros((*bool_map.shape, 3), dtype=np.uint8)
    vis[bool_map] = (255, 255, 255)
    vis[~bool_map] = (40, 40, 40)
    # Draw robot
    rx, ry = int(robot_px[0]), int(robot_px[1])
    cv2.circle(vis, (rx, ry), 5, (0, 0, 255), -1)
    cropped = crop_around_center(vis, robot_px, half_size)
    # Add label
    cv2.putText(cropped, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return cropped


def visualize_obstacle_detail(obstacle_map, navigable_map, robot_px, half_size=300):
    """Visualize obstacle vs navigable map overlay."""
    vis = np.zeros((*obstacle_map.shape, 3), dtype=np.uint8)
    vis[:] = (40, 40, 40)  # background = dark gray
    # Navigable = green
    vis[navigable_map] = (0, 100, 0)
    # Raw obstacle = red
    vis[obstacle_map] = (0, 0, 255)
    # Dilated obstacle (non-navigable & non-raw-obstacle) = orange
    dilated_only = ~navigable_map & ~obstacle_map
    vis[dilated_only] = (0, 128, 255)
    # Robot
    rx, ry = int(robot_px[0]), int(robot_px[1])
    cv2.circle(vis, (rx, ry), 5, (255, 255, 0), -1)
    cropped = crop_around_center(vis, robot_px, half_size)
    cv2.putText(cropped, "red=obs orange=dilated green=nav", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return cropped


def visualize_explored_detail(explored, navigable_map, obstacle_map, fov_mask, robot_px, half_size=300):
    """Visualize explored area with FOV overlay."""
    vis = np.zeros((*explored.shape, 3), dtype=np.uint8)
    vis[:] = (40, 40, 40)
    # Explored = white
    vis[explored] = (255, 255, 255)
    # Obstacle = red
    vis[obstacle_map] = (0, 0, 200)
    # FOV boundary
    fov_contour = fov_mask.astype(np.uint8)
    contours, _ = cv2.findContours(fov_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    # Robot
    rx, ry = int(robot_px[0]), int(robot_px[1])
    cv2.circle(vis, (rx, ry), 5, (255, 255, 0), -1)
    cropped = crop_around_center(vis, robot_px, half_size)
    cv2.putText(cropped, "white=explored red=obs green=fov", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return cropped


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Collect and sort request directories
    req_dirs = sorted(TESTDATA_DIR.glob("req_*"))
    print(f"Found {len(req_dirs)} test frames")

    # Create obstacle map with same params as production
    omap = ObstacleMap(
        min_height=0.15,
        max_height=1.5,
        agent_radius=0.18,
        area_thresh=3.0,
        size=5000,
        pixels_per_meter=100,
    )

    for i, req_dir in enumerate(req_dirs):
        print(f"\n{'='*60}")
        print(f"Processing {req_dir.name}")
        print(f"{'='*60}")

        depth, rgb, intrinsic, T_cam_odom, T_odom_base, meta = load_frame(req_dir)

        print(f"Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]m")
        print(f"Depth zero count: {(depth == 0).sum()}, inf count: {np.isinf(depth).sum()}")

        # --- Step A: Generate point cloud and check coordinate transform ---
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        min_depth, max_depth = 0.5, 10.0

        reliable_mask = np.isfinite(depth) & (depth > min_depth + 0.05) & (depth < max_depth)
        pc_optical = get_point_cloud(depth, reliable_mask, fx, fy, cx, cy)
        print(f"Point cloud (optical): {len(pc_optical)} points")

        if len(pc_optical) > 0:
            print(f"  optical X: [{pc_optical[:,0].min():.2f}, {pc_optical[:,0].max():.2f}]")
            print(f"  optical Y: [{pc_optical[:,1].min():.2f}, {pc_optical[:,1].max():.2f}]")
            print(f"  optical Z: [{pc_optical[:,2].min():.2f}, {pc_optical[:,2].max():.2f}]")

        # The code copies optical -> camera directly (no frame conversion)
        pc_camera = pc_optical.copy()

        T_odom_cam = np.linalg.inv(T_cam_odom)
        pc_odom = transform_points(T_odom_cam, pc_camera)

        if len(pc_odom) > 0:
            print(f"  odom X: [{pc_odom[:,0].min():.2f}, {pc_odom[:,0].max():.2f}]")
            print(f"  odom Y: [{pc_odom[:,1].min():.2f}, {pc_odom[:,1].max():.2f}]")
            print(f"  odom Z: [{pc_odom[:,2].min():.2f}, {pc_odom[:,2].max():.2f}]")

        # Check obstacle cloud
        obs_cloud = filter_points_by_height(pc_odom, 0.15, 2.0)
        print(f"Obstacle cloud: {len(obs_cloud)} points (height filtered)")

        # --- Step B: Run obstacle map update ---
        omap.update_map(
            depth=depth,
            intrinsic=intrinsic,
            extrinsic=T_cam_odom,
            base_to_odom=T_odom_base,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        # --- Step C: Get robot pixel for visualization ---
        robot_pos_odom = T_odom_base[:2, 3]
        robot_px = omap._xy_to_px(robot_pos_odom.reshape(1, 2))[0]
        robot_yaw = extract_yaw(T_odom_base)
        print(f"Robot pos odom: {robot_pos_odom}, pixel: {robot_px}, yaw: {np.rad2deg(robot_yaw):.1f} deg")

        # --- Step D: Visualize intermediate results ---
        H, W = depth.shape
        fov_h = 2 * np.arctan(W / (2 * fx))
        fov_mask = omap._create_fov_mask(robot_pos_odom, robot_yaw, fov_h, max_depth)

        # D1: Obstacle map (raw)
        vis_obs = visualize_map(omap._obstacle_map, robot_px, f"obstacle_map frame={i}")
        # D2: Navigable map (after dilation)
        vis_nav = visualize_map(omap._navigable_map, robot_px, f"navigable_map frame={i}")
        # D3: Explored area
        vis_exp = visualize_map(omap.explored_area, robot_px, f"explored_area frame={i}")
        # D4: Obstacle detail (raw vs dilated)
        vis_detail = visualize_obstacle_detail(omap._obstacle_map, omap._navigable_map, robot_px)
        # D5: Explored detail with FOV
        vis_exp_detail = visualize_explored_detail(
            omap.explored_area, omap._navigable_map, omap._obstacle_map, fov_mask, robot_px
        )

        # D6: Scatter plot of obstacle points on BEV
        vis_scatter = np.zeros((600, 600, 3), dtype=np.uint8)
        vis_scatter[:] = (40, 40, 40)
        if len(obs_cloud) > 0:
            # Center on robot position
            xy = obs_cloud[:, :2] - robot_pos_odom
            scale = 50  # pixels per meter
            px = (xy[:, 0] * scale + 300).astype(int)
            py = (-xy[:, 1] * scale + 300).astype(int)  # flip Y for image
            valid = (px >= 0) & (px < 600) & (py >= 0) & (py < 600)
            for x, y in zip(px[valid], py[valid]):
                cv2.circle(vis_scatter, (x, y), 1, (0, 0, 255), -1)
        cv2.circle(vis_scatter, (300, 300), 5, (0, 255, 0), -1)
        cv2.putText(vis_scatter, f"obs_scatter frame={i} ({len(obs_cloud)}pts)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save all visualizations
        frame_dir = OUTPUT_DIR / f"frame_{i:02d}"
        frame_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(frame_dir / "1_obstacle_map.png"), vis_obs)
        cv2.imwrite(str(frame_dir / "2_navigable_map.png"), vis_nav)
        cv2.imwrite(str(frame_dir / "3_explored_area.png"), vis_exp)
        cv2.imwrite(str(frame_dir / "4_obstacle_detail.png"), vis_detail)
        cv2.imwrite(str(frame_dir / "5_explored_detail.png"), vis_exp_detail)
        cv2.imwrite(str(frame_dir / "6_obs_scatter.png"), vis_scatter)

        # Save RGB and depth for reference
        if rgb is not None:
            cv2.imwrite(str(frame_dir / "0_rgb.jpg"), rgb)
        depth_vis = (np.clip(depth, 0, max_depth) / max_depth * 255).astype(np.uint8)
        cv2.imwrite(str(frame_dir / "0_depth.png"), depth_vis)

        # --- Step E: Check for wall leaking ---
        # Count explored pixels that are behind obstacles (in non-navigable areas)
        explored_in_non_nav = omap.explored_area & ~omap._navigable_map
        if explored_in_non_nav.sum() > 0:
            print(f"[WARN] Explored area in non-navigable region: {explored_in_non_nav.sum()} pixels")

        # --- Step F: Check obstacle continuity in FOV ---
        # Look for gaps in obstacle walls within FOV
        obs_in_fov = omap._obstacle_map & fov_mask
        if obs_in_fov.sum() > 0:
            # Morphological closing to see how many gaps exist
            kernel_close = np.ones((5, 5), np.uint8)
            obs_closed = cv2.morphologyEx(obs_in_fov.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
            gap_pixels = obs_closed.astype(bool) & ~obs_in_fov
            print(f"[INFO] Obstacle gaps in FOV (closable by 5x5): {gap_pixels.sum()} pixels")

        print(f"[STATS] obstacle={omap._obstacle_map.sum()}, "
              f"navigable={omap._navigable_map.sum()}, "
              f"explored={omap.explored_area.sum()}, "
              f"frontiers={len(omap.frontiers)}")

    print(f"\nVisualization saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
