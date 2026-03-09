#!/usr/bin/env python3
"""Test script for Pi3Decoder implementation."""

import numpy as np
import sys
import os

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from pi3_decoder_impl import Pi3Decoder
from local_planning_pipeline import PI3DecodeInput, VideoPrediction


def test_coordinate_transformation():
    """Test coordinate transformation logic."""
    print("Testing coordinate transformation...")

    # Create a simple trajectory in camera frame
    traj_cam = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0]
    ], dtype=np.float64)

    # Create identity transforms for testing
    extrinsic = np.eye(4, dtype=np.float32)  # odom->cam (identity means cam == odom)
    base_to_odom = np.eye(4, dtype=np.float32)

    # Create decoder (without loading model)
    decoder = Pi3Decoder(
        checkpoint_path='/tmp/dummy.pt',  # Won't be loaded in this test
        device='cpu'
    )

    # Test transformation
    traj_odom = decoder._transform_to_odom(traj_cam, extrinsic, base_to_odom)

    print(f"Input trajectory (camera frame):\n{traj_cam}")
    print(f"Output trajectory (odom frame):\n{traj_odom}")

    # With identity transform, output should equal input
    assert np.allclose(traj_cam, traj_odom), "Identity transform failed"
    print("Identity transform test passed!")

    # Test with translation
    extrinsic_translated = np.eye(4, dtype=np.float32)
    extrinsic_translated[0, 3] = 1.0  # Translate 1m in x

    traj_odom_translated = decoder._transform_to_odom(traj_cam, extrinsic_translated, base_to_odom)
    print(f"\nWith translation (odom->cam has +1m in x):")
    print(f"Output trajectory:\n{traj_odom_translated}")

    # Camera at x=1 in odom, so points should be shifted by -1 (inverse transform)
    expected = traj_cam.copy()
    expected[:, 0] -= 1.0
    assert np.allclose(traj_odom_translated, expected), "Translation transform failed"
    print("Translation transform test passed!")

    print("\nAll coordinate transformation tests passed!")


def test_video_preprocessing():
    """Test video preprocessing."""
    print("\nTesting video preprocessing...")

    # Create synthetic video frames
    T, H, W = 16, 224, 224
    frames = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)

    decoder = Pi3Decoder(
        checkpoint_path='/tmp/dummy.pt',
        device='cpu'
    )

    # Preprocess
    imgs = decoder._preprocess_video(frames)

    print(f"Input shape: {frames.shape}, dtype: {frames.dtype}")
    print(f"Output shape: {imgs.shape}, dtype: {imgs.dtype}")
    print(f"Output range: [{imgs.min():.3f}, {imgs.max():.3f}]")

    # Check shape
    assert imgs.shape == (1, T, 3, H, W), f"Expected (1, {T}, 3, {H}, {W}), got {imgs.shape}"

    # Check range
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0, "Values should be in [0, 1]"

    print("Video preprocessing test passed!")


def test_mock_inference():
    """Test with mock video data (no actual model loading)."""
    print("\nTesting with mock video data...")

    # This test only checks the data flow, not actual inference
    # To test actual inference, you need a real checkpoint file

    T, H, W = 16, 224, 224
    frames = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)

    video_pred = VideoPrediction(
        frames=frames,
        fps=10.0,
        metadata={}
    )

    intrinsic = np.eye(3, dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    base_to_odom = np.eye(4, dtype=np.float32)

    decode_input = PI3DecodeInput(
        video_prediction=video_pred,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        base_to_odom=base_to_odom,
    )

    print(f"Created PI3DecodeInput with video shape: {frames.shape}")
    print("Mock inference test setup complete!")


if __name__ == '__main__':
    print("=" * 60)
    print("Pi3Decoder Unit Tests")
    print("=" * 60)

    test_coordinate_transformation()
    test_video_preprocessing()
    test_mock_inference()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nTo test with actual Pi3 model inference:")
    print("1. Download or locate a Pi3 checkpoint file")
    print("2. Run: python test_pi3_decoder.py --checkpoint /path/to/checkpoint.pt")
