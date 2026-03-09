#!/usr/bin/env python3
"""
Generate trajectory from video using Pi3 decoder with MoGe-2 scale recovery.

This script implements the depth alignment and scale recovery pipeline described
in depth_refine.md, combining Pi3 for geometry and MoGe-2 for metric scale.
"""

import argparse
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import time

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Add MoGe module to path
moge_path = os.path.expanduser('~/mobile_manipulation/perception_modules/MoGe')
if os.path.exists(moge_path):
    sys.path.insert(0, moge_path)

from pi3_decoder_impl import Pi3Decoder
from local_planning_pipeline import PI3DecodeInput, VideoPrediction
from pointcloud_utils import save_ply, filter_points_by_confidence, downsample_points
from geometry_utils import get_point_cloud


def extract_trajectory_relative_to_first_frame(camera_poses: np.ndarray) -> np.ndarray:
    """
    Extract trajectory relative to first frame camera coordinate system.

    Parameters
    ----------
    camera_poses : np.ndarray
        Camera-to-world transformation matrices, shape (T, 4, 4)

    Returns
    -------
    np.ndarray
        Trajectory in first frame camera coordinates, shape (T, 3)
    """
    T = len(camera_poses)
    trajectory = np.zeros((T, 3), dtype=np.float32)

    # Get first frame pose and compute inverse
    T0 = camera_poses[0]
    T0_inv = np.linalg.inv(T0)

    # For each frame, compute relative transformation
    for i in range(T):
        # T_relative = T0^(-1) @ Ti
        # This gives transformation from frame i to frame 0
        T_relative = T0_inv @ camera_poses[i]

        # Extract translation (position of camera i in frame 0 coordinates)
        trajectory[i] = T_relative[:3, 3]

    return trajectory


def visualize_trajectory_on_depth(
    depth: np.ndarray,
    trajectory_pi3: np.ndarray,
    trajectory_moge: np.ndarray,
    trajectory_final: np.ndarray,
    fx: float = 500.0,
    fy: float = 500.0,
    output_path: str = None
):
    """
    Visualize three trajectories projected onto point cloud from depth map (top-down view only).

    Parameters
    ----------
    depth : np.ndarray
        Depth map with shape (H, W), values in meters
    trajectory_pi3 : np.ndarray
        Trajectory scaled by S2 (Pi3→MoGe), shape (T, 3)
    trajectory_moge : np.ndarray
        Trajectory scaled by S1 (MoGe→Real), shape (T, 3)
    trajectory_final : np.ndarray
        Final trajectory scaled by S_total, shape (T, 3)
    fx, fy : float
        Camera focal lengths
    output_path : str, optional
        Path to save visualization
    """
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0

    # Create valid mask - filter to reasonable depth range
    max_depth = max(10.0, trajectory_final[:, 2].max() * 3)
    valid_mask = (depth > 0.5) & (depth < max_depth)

    # Convert depth to point cloud
    points = get_point_cloud(depth, valid_mask, fx, fy, cx, cy)

    print(f"\n[Depth Visualization] Point cloud: {len(points)} points")
    print(f"[Depth Visualization] Depth range: [{depth[valid_mask].min():.3f}, {depth[valid_mask].max():.3f}] m")

    # Create visualization - top-down view only
    fig = plt.figure(figsize=(10, 10))

    # Downsample points for visualization
    if len(points) > 50000:
        indices = np.random.choice(len(points), 50000, replace=False)
        points_vis = points[indices]
    else:
        points_vis = points

    # Top-down view (XZ plane)
    ax = fig.add_subplot(111)
    ax.scatter(points_vis[:, 0], points_vis[:, 2], c=points_vis[:, 1],
                cmap='viridis', s=0.5, alpha=0.3)

    # Plot three trajectories
    ax.plot(trajectory_pi3[:, 0], trajectory_pi3[:, 2], 'b-o', linewidth=2, markersize=3,
            label='Pi3 Scale (S2)', alpha=0.7)
    ax.plot(trajectory_moge[:, 0], trajectory_moge[:, 2], 'g-o', linewidth=2, markersize=3,
            label='MoGe Scale (S1)', alpha=0.7)
    ax.plot(trajectory_final[:, 0], trajectory_final[:, 2], 'r-o', linewidth=2, markersize=4,
            label='Final (S_total)', alpha=0.9)

    # Mark start and end for final trajectory
    ax.plot(trajectory_final[0, 0], trajectory_final[0, 2], 'go', markersize=10, label='Start', zorder=10)
    ax.plot(trajectory_final[-1, 0], trajectory_final[-1, 2], 'mo', markersize=10, label='End', zorder=10)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Top-Down View (XZ) - Trajectory Comparison')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Depth Visualization] Saved to: {output_path}")

    plt.close()


def visualize_trajectory(traj_3d: np.ndarray, output_path: str = None):
    """Visualize 3D trajectory with multiple views."""
    if len(traj_3d) == 0:
        print("Empty trajectory, nothing to visualize")
        return

    fig = plt.figure(figsize=(15, 5))

    # XZ view (top-down, horizontal plane in OpenCV coordinates)
    ax1 = fig.add_subplot(131)
    ax1.plot(traj_3d[:, 0], traj_3d[:, 2], 'b-o', markersize=3)
    ax1.plot(traj_3d[0, 0], traj_3d[0, 2], 'go', markersize=10, label='Start')
    ax1.plot(traj_3d[-1, 0], traj_3d[-1, 2], 'ro', markersize=10, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('Top-Down View (XZ)')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')

    # XY view (side, shows height variation in OpenCV coordinates)
    ax2 = fig.add_subplot(132)
    ax2.plot(traj_3d[:, 0], traj_3d[:, 1], 'b-o', markersize=3)
    ax2.plot(traj_3d[0, 0], traj_3d[0, 1], 'go', markersize=10, label='Start')
    ax2.plot(traj_3d[-1, 0], traj_3d[-1, 1], 'ro', markersize=10, label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Side View (XY)')
    ax2.grid(True)
    ax2.legend()

    # 3D view
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], 'b-o', markersize=3)
    ax3.plot([traj_3d[0, 0]], [traj_3d[0, 1]], [traj_3d[0, 2]], 'go', markersize=10, label='Start')
    ax3.plot([traj_3d[-1, 0]], [traj_3d[-1, 1]], [traj_3d[-1, 2]], 'ro', markersize=10, label='End')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('3D View')
    ax3.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory visualization to: {output_path}")

    plt.close()


def print_trajectory_stats(traj_3d: np.ndarray):
    """Print trajectory statistics."""
    if len(traj_3d) == 0:
        print("Empty trajectory")
        return

    print("\n" + "=" * 60)
    print("Trajectory Statistics")
    print("=" * 60)
    print(f"Number of points: {len(traj_3d)}")
    print(f"\nX range: [{traj_3d[:, 0].min():.3f}, {traj_3d[:, 0].max():.3f}] m")
    print(f"Y range: [{traj_3d[:, 1].min():.3f}, {traj_3d[:, 1].max():.3f}] m")
    print(f"Z range: [{traj_3d[:, 2].min():.3f}, {traj_3d[:, 2].max():.3f}] m")

    if len(traj_3d) > 1:
        diffs = np.diff(traj_3d, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        total_distance = distances.sum()
        print(f"\nTotal path length: {total_distance:.3f} m")
        print(f"Average step size: {distances.mean():.3f} m")
        print(f"Max step size: {distances.max():.3f} m")

    print("=" * 60 + "\n")


def load_video(video_path: str, num_frames: int = 16, target_size: tuple = (448, 448)) -> np.ndarray:
    """Load video file and sample frames at equal intervals.

    Returns
    -------
    np.ndarray
        Video frames with shape (num_frames, H, W, 3), RGB uint8
    """
    print(f"Loading video from: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {total_frames} frames, {fps:.2f} fps")
    print(f"Sampling {num_frames} frames, resizing to {target_size[1]}x{target_size[0]}")

    # Calculate frame indices
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        max_frame = min(total_frames - 2, total_frames - 1)
        frame_indices = np.linspace(0, max_frame, num_frames, dtype=int).tolist()

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {idx}")
            continue

        frame_resized = cv2.resize(frame, (target_size[1], target_size[0]))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    frames_array = np.array(frames, dtype=np.uint8)
    print(f"Loaded {len(frames)} frames, shape: {frames_array.shape}")

    return frames_array


def extract_pi3_depth(pointmaps: np.ndarray) -> np.ndarray:
    """
    Extract depth maps from Pi3 pointmaps.

    Parameters
    ----------
    pointmaps : np.ndarray
        Pi3 pointmaps with shape (T, H, W, 3) in camera frame

    Returns
    -------
    np.ndarray
        Depth maps with shape (T, H, W), Z coordinate (forward depth)
    """
    # In OpenCV coordinates: x=right, y=down, z=forward
    # Depth is the Z coordinate (index 2)
    depth_pred = pointmaps[:, :, :, 2]

    print(f"[Pi3 Depth] Extracted depth from pointmaps")
    print(f"  Shape: {depth_pred.shape}")
    print(f"  Range: [{depth_pred.min():.3f}, {depth_pred.max():.3f}]")

    return depth_pred


def run_moge2(frames: np.ndarray, checkpoint: str = 'Ruicheng/moge-2-vitl', device: str = 'cuda') -> np.ndarray:
    """
    Run MoGe-2 to get metric depth maps.

    Parameters
    ----------
    frames : np.ndarray
        Video frames with shape (T, H, W, 3), RGB uint8
    checkpoint : str
        MoGe-2 checkpoint path or Hugging Face repo ID
    device : str
        Device to run inference on

    Returns
    -------
    np.ndarray
        Metric depth maps with shape (T, H, W) in meters
    """
    print(f"\n[MoGe-2] Loading model from: {checkpoint}")

    from moge.model import import_model_class_by_version

    # Load MoGe-2 model
    MoGeModel = import_model_class_by_version('v2')
    model = MoGeModel.from_pretrained(checkpoint).to(device).eval()

    print(f"[MoGe-2] Running inference on {len(frames)} frames...")

    T, H, W, _ = frames.shape
    depth_ref = np.zeros((T, H, W), dtype=np.float32)

    # Process each frame
    for t in range(T):
        # Prepare input: (H, W, 3) -> (1, 3, H, W), normalize to [0, 1]
        img = torch.from_numpy(frames[t]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img = img.to(device)

        # Run inference
        with torch.no_grad():
            # MoGe-2 expects num_tokens parameter
            num_tokens = (H // 14) * (W // 14)  # Assuming patch size 14
            output = model(img, num_tokens=num_tokens)

        # Extract depth from points (Z coordinate)
        # output['points'] has shape (1, H, W, 3)
        points = output['points'][0].cpu().numpy()  # (H, W, 3)
        depth_ref[t] = points[:, :, 2]  # Z coordinate is depth

        if (t + 1) % 10 == 0:
            print(f"  Processed {t + 1}/{T} frames")

    print(f"[MoGe-2] Depth range: [{depth_ref.min():.3f}, {depth_ref.max():.3f}] meters")

    return depth_ref


def compute_scale_factor(
    depth_pred: np.ndarray,
    depth_ref: np.ndarray,
    tau_min: float = 0.5,
    tau_max: float = 10.0,
    name: str = "Scale"
) -> float:
    """
    Compute scale factor by comparing two depth maps.

    Parameters
    ----------
    depth_pred : np.ndarray
        Predicted depth to be scaled, shape (H, W) or (T, H, W)
    depth_ref : np.ndarray
        Reference depth, shape (H, W) or (T, H, W)
    tau_min, tau_max : float
        Valid depth range in meters
    name : str
        Name for logging

    Returns
    -------
    float
        Scale factor S = depth_ref / depth_pred
    """
    print(f"\n[{name}] Computing scale factor...")

    # Handle both single frame and multi-frame
    if depth_pred.ndim == 2:
        depth_pred = depth_pred[np.newaxis, ...]
        depth_ref = depth_ref[np.newaxis, ...]

    T, H, W = depth_pred.shape
    all_ratios = []

    for t in range(T):
        # Create valid mask
        valid_mask = (
            (depth_ref[t] > tau_min) &
            (depth_ref[t] < tau_max) &
            (depth_pred[t] > 0)
        )

        num_valid = valid_mask.sum()
        if num_valid == 0:
            print(f"  Frame {t}: No valid pixels")
            continue

        # Compute ratios for valid pixels
        ratios_t = depth_ref[t][valid_mask] / depth_pred[t][valid_mask]
        all_ratios.append(ratios_t)

        if T <= 5:  # Only print details for few frames
            print(f"  Frame {t}: {num_valid} valid pixels, "
                  f"ratio range [{ratios_t.min():.2f}, {ratios_t.max():.2f}]")

    # Concatenate all ratios and take median
    all_ratios = np.concatenate(all_ratios)
    S = np.median(all_ratios)

    print(f"\n[{name}] Scale factor: S = {S:.4f}")
    print(f"  Total valid pixels: {len(all_ratios)}")
    print(f"  Ratio statistics:")
    print(f"    Mean: {all_ratios.mean():.4f}")
    print(f"    Median: {S:.4f}")
    print(f"    Std: {all_ratios.std():.4f}")

    return S


def load_real_depth(depth_path: str, target_size: tuple = None) -> np.ndarray:
    """
    Load real depth map from file.

    Parameters
    ----------
    depth_path : str
        Path to depth image
    target_size : tuple, optional
        (H, W) to resize to

    Returns
    -------
    np.ndarray
        Depth map in meters, shape (H, W)
    """
    print(f"\n[Real Depth] Loading from: {depth_path}")

    import cv2
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth_img is None:
        raise ValueError(f"Cannot load depth image: {depth_path}")

    # Convert to meters (assuming uint16 in millimeters)
    if depth_img.dtype == np.uint16:
        depth = depth_img.astype(np.float32) / 1000.0
    else:
        depth = depth_img.astype(np.float32)

    # Resize if needed
    if target_size is not None and depth.shape != target_size:
        depth = cv2.resize(depth, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        print(f"[Real Depth] Resized to: {target_size}")

    print(f"[Real Depth] Shape: {depth.shape}")
    print(f"[Real Depth] Range: [{depth.min():.3f}, {depth.max():.3f}] meters")

    return depth


def main():
    parser = argparse.ArgumentParser(
        description='Generate trajectory from video with MoGe-2 scale recovery'
    )
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Pi3 checkpoint')
    parser.add_argument('--real-depth', type=str, required=True, help='Path to real depth map (first frame)')
    parser.add_argument('--moge-checkpoint', type=str, default='Ruicheng/moge-2-vitl',
                        help='MoGe-2 checkpoint path or Hugging Face repo ID (default: Ruicheng/moge-2-vitl)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--num-frames', type=int, default=60, help='Number of frames to sample')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Pi3 dtype')
    parser.add_argument('--resize', type=int, nargs=2, default=[448, 448],
                        metavar=('HEIGHT', 'WIDTH'), help='Frame size (must be divisible by 14)')

    parser.add_argument('--save-scene', action='store_true',
                        help='Save full 3D scene reconstruction as point cloud')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                        help='Confidence threshold for filtering points (default: 0.3)')
    parser.add_argument('--voxel-size', type=float, default=0.02,
                        help='Voxel size for downsampling point cloud in meters (default: 0.02)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return 1

    target_h, target_w = args.resize
    if target_h % 14 != 0 or target_w % 14 != 0:
        print(f"Error: Resize dimensions must be divisible by 14")
        return 1

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.video).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Trajectory Generation with Scale Recovery")
    print("=" * 70)

    # Initialize timing dictionary
    timings = {}

    # Step 1: Load video
    t_start = time.time()
    frames = load_video(args.video, num_frames=args.num_frames, target_size=(target_h, target_w))
    timings['1_load_video'] = time.time() - t_start

    # Step 2: Run Pi3
    print("\n" + "=" * 70)
    print("Step 1: Running Pi3 for geometry reconstruction")
    print("=" * 70)

    video_pred = VideoPrediction(
        frames=frames,
        fps=30.0,
        metadata={'source': args.video}
    )

    # Camera intrinsics (rough estimate)
    intrinsic = np.array([
        [500.0, 0.0, frames.shape[2] / 2],
        [0.0, 500.0, frames.shape[1] / 2],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    extrinsic = np.eye(4, dtype=np.float32)
    base_to_odom = np.eye(4, dtype=np.float32)

    decode_input = PI3DecodeInput(
        video_prediction=video_pred,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        base_to_odom=base_to_odom,
    )

    decoder = Pi3Decoder(
        checkpoint_path=args.checkpoint,
        device=args.device,
        dtype=args.dtype
    )

    # Get full scene reconstruction from Pi3
    t_start = time.time()
    result_pi3 = decoder.decode_full_scene(decode_input)
    pointmaps = result_pi3['points']  # (T, H, W, 3)

    # Extract camera poses and compute trajectory correctly
    # Need to get camera_poses from decoder
    print(f"\n[Pi3] Extracting camera poses for trajectory...")
    decoder._load_model()
    frames_preprocessed = decoder._preprocess_video(frames)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=decoder.dtype):
            pred = decoder.model(frames_preprocessed)

    camera_poses = pred['camera_poses'][0].cpu().numpy()  # (T, 4, 4)

    # Extract trajectory relative to first frame (correct method)
    waypoints_norm = extract_trajectory_relative_to_first_frame(camera_poses)
    timings['2_pi3_reconstruction'] = time.time() - t_start

    print(f"\n[Pi3] Got normalized waypoints: {waypoints_norm.shape}")
    print(f"[Pi3] Got pointmaps: {pointmaps.shape}")

    # Step 3: Extract Pi3 depth
    print("\n" + "=" * 70)
    print("Step 2: Extracting depth from Pi3 pointmaps")
    print("=" * 70)

    t_start = time.time()
    depth_pred = extract_pi3_depth(pointmaps)
    timings['3_extract_pi3_depth'] = time.time() - t_start

    # Step 4: Run MoGe-2
    print("\n" + "=" * 70)
    print("Step 3: Running MoGe-2 for metric depth")
    print("=" * 70)

    t_start = time.time()
    depth_ref = run_moge2(frames, checkpoint=args.moge_checkpoint, device=args.device)
    timings['4_moge2_inference'] = time.time() - t_start

    # Step 5: Load real depth
    print("\n" + "=" * 70)
    print("Step 4: Loading real depth map")
    print("=" * 70)

    t_start = time.time()
    depth_real = load_real_depth(args.real_depth, target_size=(target_h, target_w))
    timings['5_load_real_depth'] = time.time() - t_start

    # Step 6: Two-stage scale alignment
    print("\n" + "=" * 70)
    print("Step 5: Two-stage scale alignment")
    print("=" * 70)

    # Stage 1: Align MoGe-2 to real depth (first frame only)
    print("\n--- Stage 1: MoGe-2 → Real Depth ---")
    t_start = time.time()
    S1 = compute_scale_factor(
        depth_ref[0],  # MoGe-2 first frame
        depth_real,    # Real depth first frame
        tau_min=0.5,
        tau_max=10.0,
        name="S1 (MoGe→Real)"
    )
    timings['6_scale_alignment_s1'] = time.time() - t_start

    # Stage 2: Align Pi3 to MoGe-2 (first frame only)
    print("\n--- Stage 2: Pi3 → MoGe-2 ---")
    t_start = time.time()
    S2 = compute_scale_factor(
        depth_pred[0],  # Pi3 first frame
        depth_ref[0],   # MoGe-2 first frame
        tau_min=0.5,
        tau_max=10.0,
        name="S2 (Pi3→MoGe)"
    )
    timings['7_scale_alignment_s2'] = time.time() - t_start

    # Total scale factor
    S_total = S1 * S2

    print("\n" + "=" * 70)
    print("Scale Factor Summary")
    print("=" * 70)
    print(f"S1 (MoGe→Real): {S1:.4f}")
    print(f"S2 (Pi3→MoGe):  {S2:.4f}")
    print(f"S_total:        {S_total:.4f}")
    print("=" * 70)

    # Step 7: Scale waypoints and pointmaps
    print("\n" + "=" * 70)
    print("Step 6: Scaling outputs to metric space")
    print("=" * 70)

    t_start = time.time()
    waypoints_metric = S_total * waypoints_norm
    pointmaps_metric = S_total * pointmaps  # Scale pointmaps too
    timings['8_apply_scaling'] = time.time() - t_start

    print(f"\n[Result] Waypoints before scaling:")
    print(f"  Range: X=[{waypoints_norm[:, 0].min():.3f}, {waypoints_norm[:, 0].max():.3f}], "
          f"Y=[{waypoints_norm[:, 1].min():.3f}, {waypoints_norm[:, 1].max():.3f}], "
          f"Z=[{waypoints_norm[:, 2].min():.3f}, {waypoints_norm[:, 2].max():.3f}]")

    print(f"\n[Result] Waypoints after scaling (metric):")
    print(f"  Range: X=[{waypoints_metric[:, 0].min():.3f}, {waypoints_metric[:, 0].max():.3f}], "
          f"Y=[{waypoints_metric[:, 1].min():.3f}, {waypoints_metric[:, 1].max():.3f}], "
          f"Z=[{waypoints_metric[:, 2].min():.3f}, {waypoints_metric[:, 2].max():.3f}]")

    # Print statistics
    print_trajectory_stats(waypoints_metric)

    # Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs")
    print("=" * 70)

    video_stem = Path(args.video).stem

    # Save trajectory
    traj_file = output_dir / f"{video_stem}_trajectory_scaled.npy"
    np.save(traj_file, waypoints_metric)
    print(f"Saved scaled trajectory to: {traj_file}")

    # Save scale factors
    scale_file = output_dir / f"{video_stem}_scale_factor.txt"
    with open(scale_file, 'w') as f:
        f.write(f"Two-stage scale alignment:\n")
        f.write(f"S1 (MoGe→Real): {S1:.6f}\n")
        f.write(f"S2 (Pi3→MoGe):  {S2:.6f}\n")
        f.write(f"S_total:        {S_total:.6f}\n")
    print(f"Saved scale factors to: {scale_file}")

    # Visualize trajectory
    vis_file = output_dir / f"{video_stem}_trajectory_scaled.png"
    visualize_trajectory(waypoints_metric, str(vis_file))

    # Save point cloud if requested
    if args.save_scene:
        print("\nProcessing point cloud...")

        # Flatten pointmaps
        T, H, W = pointmaps_metric.shape[:3]
        points_flat = pointmaps_metric.reshape(-1, 3)
        colors_flat = result_pi3['colors'].reshape(-1, 3)
        conf_flat = result_pi3['confidence'].reshape(-1)

        print(f"Total points before filtering: {len(points_flat)}")

        # Filter by confidence
        points_filtered, colors_filtered, conf_filtered = filter_points_by_confidence(
            points_flat, colors_flat, conf_flat, threshold=args.conf_threshold
        )
        print(f"Points after confidence filtering (>={args.conf_threshold}): {len(points_filtered)}")

        # Downsample
        points_down, colors_down, conf_down = downsample_points(
            points_filtered, colors_filtered, conf_filtered, voxel_size=args.voxel_size
        )
        print(f"Points after downsampling (voxel_size={args.voxel_size}m): {len(points_down)}")

        # Save point cloud
        ply_file = output_dir / f"{video_stem}_scene_scaled.ply"
        save_ply(str(ply_file), points_down, colors_down, conf_down)

    # Save intermediate results for debugging
    np.save(output_dir / f"{video_stem}_depth_pi3.npy", depth_pred)
    np.save(output_dir / f"{video_stem}_depth_moge2.npy", depth_ref)
    print(f"Saved intermediate depth maps for debugging")

    # Visualize trajectory on depth map
    print("\n" + "=" * 70)
    print("Step 7: Visualizing trajectory on depth map")
    print("=" * 70)

    # Compute three trajectories for comparison
    waypoints_pi3_scale = S2 * waypoints_norm  # Pi3→MoGe scale
    waypoints_moge_scale = S1 * waypoints_norm  # MoGe→Real scale
    # waypoints_metric already computed as S_total * waypoints_norm

    depth_vis_file = output_dir / f"{video_stem}_trajectory_on_depth.png"
    visualize_trajectory_on_depth(
        depth_real,
        waypoints_pi3_scale,
        waypoints_moge_scale,
        waypoints_metric,
        fx=intrinsic[0, 0],
        fy=intrinsic[1, 1],
        output_path=str(depth_vis_file)
    )

    # Print timing statistics
    print("\n" + "=" * 70)
    print("Timing Statistics (Core Algorithm Steps)")
    print("=" * 70)

    timing_labels = {
        '1_load_video': 'Load Video',
        '2_pi3_reconstruction': 'Pi3 Reconstruction + Trajectory Extraction',
        '3_extract_pi3_depth': 'Extract Pi3 Depth',
        '4_moge2_inference': 'MoGe-2 Inference',
        '5_load_real_depth': 'Load Real Depth',
        '6_scale_alignment_s1': 'Scale Alignment S1 (MoGe→Real)',
        '7_scale_alignment_s2': 'Scale Alignment S2 (Pi3→MoGe)',
        '8_apply_scaling': 'Apply Scaling'
    }

    total_time = 0.0
    for key in sorted(timings.keys()):
        t = timings[key]
        total_time += t
        label = timing_labels.get(key, key)
        print(f"{label:45s}: {t:8.3f}s")

    print("-" * 70)
    print(f"{'Total Time':45s}: {total_time:8.3f}s")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
