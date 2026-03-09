#!/usr/bin/env python3
"""
Visualize and compare point clouds from Pi3, MoGe-2, and depth map for the first frame.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Add Pi3 and MoGe paths
pi3_path = os.path.expanduser('~/mobile_manipulation/perception_modules/Pi3')
moge_path = os.path.expanduser('~/mobile_manipulation/perception_modules/MoGe')
if os.path.exists(pi3_path):
    sys.path.insert(0, pi3_path)
if os.path.exists(moge_path):
    sys.path.insert(0, moge_path)

from geometry_utils import get_point_cloud


def load_and_run_pi3(image_path: str, checkpoint: str, device: str = 'cuda') -> np.ndarray:
    """Run Pi3 on a single image and return pointmap."""
    print("[Pi3] Loading model...")

    from pi3.models.pi3 import Pi3

    # Load model - handle both local path and HF repo
    if os.path.exists(checkpoint):
        print(f"[Pi3] Loading from local checkpoint: {checkpoint}")
        # Load checkpoint manually
        if checkpoint.endswith('.safetensors'):
            from safetensors.torch import load_file
            weights = load_file(checkpoint)
        else:
            weights = torch.load(checkpoint, map_location='cpu', weights_only=False)

        # Create model with default config
        model = Pi3(pos_type='rope100', decoder_size='large')
        model.load_state_dict(weights, strict=False)
        model = model.to(device).eval()
    else:
        # Try as HF repo ID
        model = Pi3.from_pretrained(checkpoint).to(device).eval()

    # Load image
    img = np.array(Image.open(image_path))
    if img.shape[-1] == 4:  # Remove alpha channel if present
        img = img[:, :, :3]

    # Preprocess: (H, W, 3) -> (1, 1, 3, H, W), normalize to [0, 1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    print(f"[Pi3] Running inference on image {img.shape}...")

    # Run inference
    with torch.no_grad():
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast('cuda', dtype=dtype):
            output = model(img_tensor)

    # Extract pointmap for first frame
    pointmap = output['points'][0, 0].cpu().numpy()  # (H, W, 3)

    print(f"[Pi3] Pointmap shape: {pointmap.shape}")
    print(f"[Pi3] Range: X=[{pointmap[:,:,0].min():.3f}, {pointmap[:,:,0].max():.3f}], "
          f"Y=[{pointmap[:,:,1].min():.3f}, {pointmap[:,:,1].max():.3f}], "
          f"Z=[{pointmap[:,:,2].min():.3f}, {pointmap[:,:,2].max():.3f}]")

    return pointmap


def load_and_run_moge2(image_path: str, checkpoint: str = 'Ruicheng/moge-2-vitl', device: str = 'cuda') -> np.ndarray:
    """Run MoGe-2 on a single image and return pointmap."""
    print("[MoGe-2] Loading model...")

    from moge.model import import_model_class_by_version

    # Load model
    MoGeModel = import_model_class_by_version('v2')
    model = MoGeModel.from_pretrained(checkpoint).to(device).eval()

    # Load image
    img = np.array(Image.open(image_path))
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    H, W = img.shape[:2]

    # Preprocess: (H, W, 3) -> (1, 3, H, W), normalize to [0, 1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    print(f"[MoGe-2] Running inference on image {img.shape}...")

    # Run inference
    with torch.no_grad():
        num_tokens = (H // 14) * (W // 14)
        output = model(img_tensor, num_tokens=num_tokens)

    # Extract pointmap
    pointmap = output['points'][0].cpu().numpy()  # (H, W, 3)

    print(f"[MoGe-2] Pointmap shape: {pointmap.shape}")
    print(f"[MoGe-2] Range: X=[{pointmap[:,:,0].min():.3f}, {pointmap[:,:,0].max():.3f}], "
          f"Y=[{pointmap[:,:,1].min():.3f}, {pointmap[:,:,1].max():.3f}], "
          f"Z=[{pointmap[:,:,2].min():.3f}, {pointmap[:,:,2].max():.3f}]")

    return pointmap


def load_depth_pointcloud(depth_path: str, fx: float = 500.0, fy: float = 500.0) -> np.ndarray:
    """Load depth image and convert to point cloud."""
    print("[Depth] Loading depth image...")

    depth_img = np.array(Image.open(depth_path))

    # Convert to meters (assuming uint16 in millimeters)
    if depth_img.dtype == np.uint16:
        depth = depth_img.astype(np.float32) / 1000.0
    else:
        depth = depth_img.astype(np.float32)

    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0

    # Create valid mask
    valid_mask = (depth > 0.5) & (depth < 10.0)

    # Convert to point cloud
    points = get_point_cloud(depth, valid_mask, fx, fy, cx, cy)

    print(f"[Depth] Point cloud shape: {points.shape}")
    print(f"[Depth] Range: X=[{points[:,0].min():.3f}, {points[:,0].max():.3f}], "
          f"Y=[{points[:,1].min():.3f}, {points[:,1].max():.3f}], "
          f"Z=[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")

    return points


def visualize_pointclouds(pi3_points: np.ndarray, moge_points: np.ndarray, depth_points: np.ndarray,
                          output_path: str = None):
    """Visualize three point clouds side by side."""
    fig = plt.figure(figsize=(18, 6))

    # Downsample for visualization
    max_points = 10000

    def downsample(points):
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            return points[indices]
        return points

    # Pi3 pointcloud
    ax1 = fig.add_subplot(131, projection='3d')
    pi3_flat = pi3_points.reshape(-1, 3)
    pi3_vis = downsample(pi3_flat)
    ax1.scatter(pi3_vis[:, 0], pi3_vis[:, 1], pi3_vis[:, 2],
                c=pi3_vis[:, 2], cmap='viridis', s=0.5, alpha=0.6)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Pi3 Point Cloud')
    ax1.view_init(elev=20, azim=45)

    # MoGe-2 pointcloud
    ax2 = fig.add_subplot(132, projection='3d')
    moge_flat = moge_points.reshape(-1, 3)
    moge_vis = downsample(moge_flat)
    ax2.scatter(moge_vis[:, 0], moge_vis[:, 1], moge_vis[:, 2],
                c=moge_vis[:, 2], cmap='viridis', s=0.5, alpha=0.6)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('MoGe-2 Point Cloud')
    ax2.view_init(elev=20, azim=45)

    # Depth pointcloud
    ax3 = fig.add_subplot(133, projection='3d')
    depth_vis = downsample(depth_points)
    ax3.scatter(depth_vis[:, 0], depth_vis[:, 1], depth_vis[:, 2],
                c=depth_vis[:, 2], cmap='viridis', s=0.5, alpha=0.6)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('Depth Map Point Cloud')
    ax3.view_init(elev=20, azim=45)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {output_path}")

    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare point clouds from Pi3, MoGe-2, and depth map')
    parser.add_argument('--image', type=str, required=True, help='Path to first frame image')
    parser.add_argument('--depth', type=str, required=True, help='Path to depth image')
    parser.add_argument('--pi3-checkpoint', type=str, required=True, help='Path to Pi3 checkpoint')
    parser.add_argument('--moge-checkpoint', type=str, default='Ruicheng/moge-2-vitl',
                        help='MoGe-2 checkpoint')
    parser.add_argument('--output', type=str, default='pointcloud_comparison.png',
                        help='Output visualization path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    print("=" * 70)
    print("Point Cloud Comparison: Pi3 vs MoGe-2 vs Depth Map")
    print("=" * 70)

    # Run Pi3
    print("\n" + "=" * 70)
    print("Step 1: Running Pi3")
    print("=" * 70)
    pi3_pointmap = load_and_run_pi3(args.image, args.pi3_checkpoint, args.device)

    # Run MoGe-2
    print("\n" + "=" * 70)
    print("Step 2: Running MoGe-2")
    print("=" * 70)
    moge_pointmap = load_and_run_moge2(args.image, args.moge_checkpoint, args.device)

    # Load depth
    print("\n" + "=" * 70)
    print("Step 3: Loading depth map")
    print("=" * 70)
    depth_points = load_depth_pointcloud(args.depth)

    # Visualize
    print("\n" + "=" * 70)
    print("Step 4: Creating visualization")
    print("=" * 70)
    visualize_pointclouds(pi3_pointmap, moge_pointmap, depth_points, args.output)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
