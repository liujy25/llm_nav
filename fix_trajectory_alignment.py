#!/usr/bin/env python3
"""
Fix trajectory alignment by properly handling camera pose rotations.
"""

import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from pi3_decoder_impl import Pi3Decoder
from local_planning_pipeline import PI3DecodeInput, VideoPrediction
import cv2


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Re-extract trajectory with proper alignment')
    parser.add_argument('--video', type=str, required=True, help='Path to video')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Pi3 checkpoint')
    parser.add_argument('--output', type=str, default='test_trajectory_fixed.npy',
                        help='Output trajectory file')
    parser.add_argument('--num-frames', type=int, default=60, help='Number of frames')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Dtype')

    args = parser.parse_args()

    print("=" * 70)
    print("Re-extracting trajectory with proper camera pose alignment")
    print("=" * 70)

    # Load video
    print("\nLoading video...")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, min(total_frames - 2, total_frames - 1),
                                args.num_frames, dtype=int).tolist()

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_resized = cv2.resize(frame, (448, 448))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    frames = np.array(frames, dtype=np.uint8)
    print(f"Loaded {len(frames)} frames")

    # Run Pi3
    print("\nRunning Pi3...")
    video_pred = VideoPrediction(frames=frames, fps=30.0, metadata={'source': args.video})

    intrinsic = np.array([
        [500.0, 0.0, 224.0],
        [0.0, 500.0, 224.0],
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

    # Get camera poses
    print("\nExtracting camera poses...")
    result = decoder.decode_full_scene(decode_input)

    # Access raw camera poses from decoder
    # We need to re-run inference to get camera_poses
    decoder._load_model()

    frames_preprocessed = decoder._preprocess_video(frames)

    import torch
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=decoder.dtype):
            pred = decoder.model(frames_preprocessed)

    camera_poses = pred['camera_poses'][0].cpu().numpy()  # (T, 4, 4)

    print(f"Got camera poses: {camera_poses.shape}")
    print(f"\nFirst frame camera pose:")
    print(camera_poses[0])

    # Extract trajectory with proper alignment
    print("\nExtracting trajectory relative to first frame...")
    trajectory_fixed = extract_trajectory_relative_to_first_frame(camera_poses)

    print(f"\nFixed trajectory:")
    print(f"  Shape: {trajectory_fixed.shape}")
    print(f"  First 5 points:")
    for i in range(min(5, len(trajectory_fixed))):
        print(f"    [{i}] X={trajectory_fixed[i, 0]:.3f}, "
              f"Y={trajectory_fixed[i, 1]:.3f}, Z={trajectory_fixed[i, 2]:.3f}")

    print(f"\n  Range:")
    print(f"    X: [{trajectory_fixed[:, 0].min():.3f}, {trajectory_fixed[:, 0].max():.3f}]")
    print(f"    Y: [{trajectory_fixed[:, 1].min():.3f}, {trajectory_fixed[:, 1].max():.3f}]")
    print(f"    Z: [{trajectory_fixed[:, 2].min():.3f}, {trajectory_fixed[:, 2].max():.3f}]")

    # Save
    np.save(args.output, trajectory_fixed)
    print(f"\nSaved fixed trajectory to: {args.output}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
