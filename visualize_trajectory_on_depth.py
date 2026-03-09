#!/usr/bin/env python3
"""Visualize camera trajectory projected onto point cloud from depth map."""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from geometry_utils import get_point_cloud


def load_depth_image(depth_path: str) -> np.ndarray:
    """Load depth image from file.

    Parameters
    ----------
    depth_path : str
        Path to depth image file

    Returns
    -------
    np.ndarray
        Depth map with shape (H, W), values in meters
    """
    # Read depth image
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth_img is None:
        raise ValueError(f"Failed to load depth image from {depth_path}")

    print(f"Loaded depth image: shape={depth_img.shape}, dtype={depth_img.dtype}")
    print(f"Depth range: [{depth_img.min()}, {depth_img.max()}]")

    # Convert to float and normalize if needed
    if depth_img.dtype == np.uint16:
        # Assume 16-bit depth in millimeters
        depth = depth_img.astype(np.float32) / 1000.0  # Convert to meters
    elif depth_img.dtype == np.uint8:
        # Assume 8-bit normalized depth, need to scale
        # This is a guess - adjust based on actual depth range
        depth = depth_img.astype(np.float32) / 255.0 * 10.0  # Scale to 0-10m
    else:
        depth = depth_img.astype(np.float32)

    print(f"Depth in meters: [{depth.min():.3f}, {depth.max():.3f}]")

    return depth


def visualize_trajectory_on_pointcloud(
    points: np.ndarray,
    trajectory: np.ndarray,
    output_path: str = None
):
    """Visualize trajectory projected onto point cloud.

    Parameters
    ----------
    points : np.ndarray
        Point cloud with shape (N, 3)
    trajectory : np.ndarray
        Camera trajectory with shape (T, 3)
    output_path : str, optional
        Path to save visualization
    """
    fig = plt.figure(figsize=(15, 5))

    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')

    # Downsample points for visualization
    if len(points) > 50000:
        indices = np.random.choice(len(points), 50000, replace=False)
        points_vis = points[indices]
    else:
        points_vis = points

    ax1.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
                c=points_vis[:, 2], cmap='viridis', s=0.1, alpha=0.5)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
             'r-o', linewidth=2, markersize=4, label='Trajectory')
    ax1.plot([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]],
             'go', markersize=10, label='Start')
    ax1.plot([trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]],
             'bo', markersize=10, label='End')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D View')
    ax1.legend()

    # Top-down view (XZ plane)
    ax2 = fig.add_subplot(132)
    ax2.scatter(points_vis[:, 0], points_vis[:, 2], c=points_vis[:, 1],
                cmap='viridis', s=0.1, alpha=0.5)
    ax2.plot(trajectory[:, 0], trajectory[:, 2], 'r-o', linewidth=2, markersize=4)
    ax2.plot(trajectory[0, 0], trajectory[0, 2], 'go', markersize=10)
    ax2.plot(trajectory[-1, 0], trajectory[-1, 2], 'bo', markersize=10)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Top-Down View (XZ)')
    ax2.axis('equal')
    ax2.grid(True)

    # Side view (XY plane)
    ax3 = fig.add_subplot(133)
    ax3.scatter(points_vis[:, 0], points_vis[:, 1], c=points_vis[:, 2],
                cmap='viridis', s=0.1, alpha=0.5)
    ax3.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', linewidth=2, markersize=4)
    ax3.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10)
    ax3.plot(trajectory[-1, 0], trajectory[-1, 1], 'bo', markersize=10)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Side View (XY)')
    ax3.grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def print_statistics(points: np.ndarray, trajectory: np.ndarray):
    """Print statistics about point cloud and trajectory."""
    print("\n" + "=" * 60)
    print("Point Cloud Statistics")
    print("=" * 60)
    print(f"Number of points: {len(points)}")
    print(f"X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] m")
    print(f"Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] m")
    print(f"Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m")

    print("\n" + "=" * 60)
    print("Trajectory Statistics")
    print("=" * 60)
    print(f"Number of waypoints: {len(trajectory)}")
    print(f"X range: [{trajectory[:, 0].min():.3f}, {trajectory[:, 0].max():.3f}] m")
    print(f"Y range: [{trajectory[:, 1].min():.3f}, {trajectory[:, 1].max():.3f}] m")
    print(f"Z range: [{trajectory[:, 2].min():.3f}, {trajectory[:, 2].max():.3f}] m")

    if len(trajectory) > 1:
        diffs = np.diff(trajectory, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        total_distance = distances.sum()
        print(f"\nTotal trajectory length: {total_distance:.3f} m")
        print(f"Average step size: {distances.mean():.3f} m")

    print("=" * 60 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize trajectory on depth map point cloud')
    parser.add_argument('--depth', type=str, default='/home/liujy/mobile_manipulation/model_server/nav/depth.png',
                        help='Path to depth image')
    parser.add_argument('--trajectory', type=str, default='/home/liujy/mobile_manipulation/model_server/nav/test_trajectory.npy',
                        help='Path to trajectory file')
    parser.add_argument('--output', type=str, default='/home/liujy/mobile_manipulation/model_server/nav/trajectory_on_depth_vis.png',
                        help='Output visualization path')
    parser.add_argument('--no-align', action='store_true',
                        help='Skip trajectory alignment (use if trajectory is already aligned)')
    parser.add_argument('--no-flip', action='store_true',
                        help='Skip coordinate flip (use if coordinates are already correct)')
    args = parser.parse_args()

    # Paths
    depth_path = args.depth
    trajectory_path = args.trajectory
    output_path = args.output

    # Load depth image
    print("Loading depth image...")
    depth = load_depth_image(depth_path)

    # Load trajectory
    print("\nLoading trajectory...")
    trajectory = np.load(trajectory_path)
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"\nOriginal first 5 waypoints:")
    for i in range(min(5, len(trajectory))):
        print(f"  [{i}] X={trajectory[i, 0]:.3f}, Y={trajectory[i, 1]:.3f}, Z={trajectory[i, 2]:.3f}")

    # Transform trajectory to first frame camera coordinate system
    # Since depth map is from first frame, we need trajectory relative to first camera
    if not args.no_align:
        print("\nAligning trajectory to first frame camera (setting first point as origin)...")
        trajectory_aligned = trajectory - trajectory[0]
    else:
        print("\nSkipping trajectory alignment (using as-is)...")
        trajectory_aligned = trajectory

    # Fix coordinate system: Pi3's output seems to have inverted X and Z
    # User reports: camera moves forward then left
    # Current trajectory shows: X increases (right), Z decreases (backward)
    # Solution: flip X and Z to match actual motion
    if not args.no_flip:
        print("\nApplying coordinate transform (flipping X and Z)...")
        trajectory_aligned[:, 0] = -trajectory_aligned[:, 0]  # Flip X: right -> left
        trajectory_aligned[:, 2] = -trajectory_aligned[:, 2]  # Flip Z: backward -> forward
    else:
        print("\nSkipping coordinate flip (using as-is)...")

    print(f"\nTransformed first 5 waypoints:")
    for i in range(min(5, len(trajectory_aligned))):
        print(f"  [{i}] X={trajectory_aligned[i, 0]:.3f}, Y={trajectory_aligned[i, 1]:.3f}, Z={trajectory_aligned[i, 2]:.3f}")

    trajectory = trajectory_aligned

    # Camera intrinsics (rough estimate - adjust if you have actual values)
    H, W = depth.shape
    fx = fy = 500.0  # Focal length in pixels
    cx = W / 2.0
    cy = H / 2.0

    print(f"\nUsing camera intrinsics:")
    print(f"  fx={fx}, fy={fy}")
    print(f"  cx={cx}, cy={cy}")
    print(f"  Image size: {W}x{H}")

    # Create valid mask - filter to reasonable depth range
    # Based on depth statistics: median ~3m, 99th percentile ~8m
    # Keep only points within trajectory depth range for better visualization
    max_depth = max(10.0, trajectory[:, 2].max() * 3)  # At least 3x trajectory depth
    print(f"\nFiltering point cloud to depth range: [0.5, {max_depth:.1f}] meters")
    valid_mask = (depth > 0.5) & (depth < max_depth)

    # Convert depth to point cloud
    print("\nConverting depth to point cloud...")
    points_valid = get_point_cloud(depth, valid_mask, fx, fy, cx, cy)

    print(f"Valid points: {len(points_valid)}")

    # Check coordinate system
    print("\nCoordinate system check:")
    print("Point cloud (from depth map):")
    print(f"  - Should be in camera frame (OpenCV: x=right, y=down, z=forward)")
    print(f"  - Z values should all be positive (in front of camera)")
    print(f"  - Z range: [{points_valid[:, 2].min():.3f}, {points_valid[:, 2].max():.3f}]")
    print("\nTrajectory (from Pi3):")
    print(f"  - Should be camera positions in world frame (first frame as origin)")
    print(f"  - First camera should be near origin if first frame is reference")
    print(f"  - First point: X={trajectory[0, 0]:.3f}, Y={trajectory[0, 1]:.3f}, Z={trajectory[0, 2]:.3f}")

    # Print statistics
    print_statistics(points_valid, trajectory)

    # Visualize
    print("Creating visualization...")
    visualize_trajectory_on_pointcloud(points_valid, trajectory, output_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
