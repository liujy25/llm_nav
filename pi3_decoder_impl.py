#!/usr/bin/env python3
"""PI3Decoder implementation using Pi3 model for trajectory reconstruction."""

import os
import sys
import numpy as np
import torch
from typing import Optional

# Add Pi3 module to path
pi3_module_path = os.path.join(os.path.dirname(__file__), '../../perception_modules/Pi3')
if os.path.exists(pi3_module_path):
    sys.path.insert(0, pi3_module_path)

from local_planning_pipeline import PI3DecodeInput
from geometry_utils import transform_points


class Pi3Decoder:
    """Decoder that uses Pi3 model to reconstruct 3D trajectory from video."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        dtype: str = 'bfloat16',
        pos_type: str = 'rope100',
        decoder_size: str = 'large'
    ):
        """
        Initialize Pi3Decoder with lazy loading.

        Parameters
        ----------
        checkpoint_path : str
            Path to Pi3 checkpoint file (.pt or .safetensors)
        device : str
            Device to run inference on ('cuda' or 'cpu')
        dtype : str
            Data type for inference ('bfloat16' or 'float16')
        pos_type : str
            Position encoding type for Pi3 model
        decoder_size : str
            Decoder size ('small', 'base', or 'large')
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.pos_type = pos_type
        self.decoder_size = decoder_size

        self.model: Optional[torch.nn.Module] = None

    def _load_model(self):
        """Lazy load Pi3 model from checkpoint."""
        if self.model is not None:
            return

        print(f"[Pi3Decoder] Loading Pi3 model from {self.checkpoint_path}")

        # Import Pi3 model
        from pi3.models.pi3 import Pi3

        # Create model
        self.model = Pi3(pos_type=self.pos_type, decoder_size=self.decoder_size)

        # Load checkpoint
        if self.checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            weights = load_file(self.checkpoint_path)
        else:
            weights = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(weights, strict=False)
        self.model = self.model.to(self.device).eval()

        print(f"[Pi3Decoder] Model loaded successfully on {self.device}")

    def _preprocess_video(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess video frames for Pi3 model.

        Parameters
        ----------
        frames : np.ndarray
            Video frames with shape (T, H, W, 3), RGB uint8

        Returns
        -------
        torch.Tensor
            Preprocessed tensor with shape (1, T, 3, H, W), float32 in [0, 1]
        """
        # Convert to float and normalize to [0, 1]
        frames_float = frames.astype(np.float32) / 255.0

        # Rearrange dimensions: (T, H, W, 3) -> (1, T, 3, H, W)
        frames_tensor = torch.from_numpy(frames_float).permute(0, 3, 1, 2).unsqueeze(0)

        return frames_tensor.to(self.device)

    def _extract_trajectory(self, camera_poses: torch.Tensor) -> np.ndarray:
        """
        Extract translation trajectory from camera poses.

        Parameters
        ----------
        camera_poses : torch.Tensor
            Camera poses with shape (1, N, 4, 4)

        Returns
        -------
        np.ndarray
            Trajectory with shape (N, 3) in camera frame
        """
        # Extract translation: (1, N, 4, 4) -> (N, 3)
        traj_cam = camera_poses[0, :, :3, 3].cpu().numpy()
        return traj_cam

    def _transform_to_odom(
        self,
        traj_cam: np.ndarray,
        extrinsic: np.ndarray,
        base_to_odom: np.ndarray
    ) -> np.ndarray:
        """
        Transform trajectory from camera frame to odom frame.

        Parameters
        ----------
        traj_cam : np.ndarray
            Trajectory in Pi3's relative camera frame, shape (N, 3)
            Note: Pi3 outputs poses relative to the first frame
        extrinsic : np.ndarray
            Extrinsic matrix for the first frame (odom->cam), shape (4, 4)
            This defines where the first camera frame is in the world
        base_to_odom : np.ndarray
            Base to odom transform, shape (4, 4) (not used currently)

        Returns
        -------
        np.ndarray
            Trajectory in odom frame, shape (N, 3)
        """
        # Pi3 outputs poses relative to first frame
        # extrinsic defines first frame's pose in world: T_world_to_cam0
        # We need: T_world_to_cam0^-1 to get cam0_to_world
        T_cam0_to_world = np.linalg.inv(extrinsic)

        # Transform trajectory points from cam0 frame to world frame
        traj_world = transform_points(T_cam0_to_world, traj_cam)

        return traj_world

    def decode(self, data: PI3DecodeInput) -> np.ndarray:
        """
        Decode 3D trajectory from video prediction.

        Parameters
        ----------
        data : PI3DecodeInput
            Input data containing video prediction and transforms

        Returns
        -------
        np.ndarray
            Local trajectory with shape (N, 3) in odom frame
        """
        # Lazy load model
        self._load_model()

        # Preprocess video frames
        frames = data.video_prediction.frames  # (T, H, W, 3) RGB uint8
        imgs = self._preprocess_video(frames)  # (1, T, 3, H, W)

        print(f"[Pi3Decoder] Running inference on video with {frames.shape[0]} frames")

        # Run inference
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                pred = self.model(imgs)

        # Extract camera poses
        camera_poses = pred['camera_poses']  # (1, T, 4, 4)

        # Extract trajectory from camera poses
        traj_cam = self._extract_trajectory(camera_poses)  # (T, 3)

        print(f"[Pi3Decoder] Extracted trajectory with {traj_cam.shape[0]} points")

        # Transform to odom frame
        traj_odom = self._transform_to_odom(
            traj_cam,
            data.extrinsic,
            data.base_to_odom
        )

        print(f"[Pi3Decoder] Trajectory range: x=[{traj_odom[:, 0].min():.2f}, {traj_odom[:, 0].max():.2f}], "
              f"y=[{traj_odom[:, 1].min():.2f}, {traj_odom[:, 1].max():.2f}], "
              f"z=[{traj_odom[:, 2].min():.2f}, {traj_odom[:, 2].max():.2f}]")

        return traj_odom

    def decode_full_scene(self, data: PI3DecodeInput) -> dict:
        """
        Decode full 3D scene reconstruction from video prediction.

        Parameters
        ----------
        data : PI3DecodeInput
            Input data containing video prediction and transforms

        Returns
        -------
        dict
            Dictionary containing:
            - 'trajectory': Camera trajectory with shape (N, 3) in odom frame
            - 'points': Dense 3D point cloud with shape (N, H, W, 3) in odom frame
            - 'colors': RGB colors for each point with shape (N, H, W, 3)
            - 'confidence': Confidence scores with shape (N, H, W)
            - 'camera_poses': Camera poses with shape (N, 4, 4) in odom frame
        """
        # Lazy load model
        self._load_model()

        # Preprocess video frames
        frames = data.video_prediction.frames  # (T, H, W, 3) RGB uint8
        imgs = self._preprocess_video(frames)  # (1, T, 3, H, W)

        print(f"[Pi3Decoder] Running full scene reconstruction on {frames.shape[0]} frames")

        # Run inference
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                pred = self.model(imgs)

        # Extract outputs
        points = pred['points'].cpu().numpy()  # (1, T, H, W, 3)
        conf = pred['conf'].cpu().numpy()  # (1, T, H, W, 1)
        camera_poses = pred['camera_poses'].cpu().numpy()  # (1, T, 4, 4)

        # Remove batch dimension
        points = points[0]  # (T, H, W, 3)
        conf = conf[0, ..., 0]  # (T, H, W)
        camera_poses = camera_poses[0]  # (T, 4, 4)

        # Extract trajectory
        trajectory = camera_poses[:, :3, 3]  # (T, 3)

        # Pi3 outputs points and camera_poses already in world frame (relative to first frame)
        # Only apply extrinsic transform if it's not identity (i.e., if we want a different world frame)
        if not np.allclose(data.extrinsic, np.eye(4)):
            # Transform points to odom frame
            T_cam_to_odom = np.linalg.inv(data.extrinsic)
            T, H, W = points.shape[:3]
            points_flat = points.reshape(-1, 3)
            points_odom_flat = transform_points(T_cam_to_odom, points_flat)
            points = points_odom_flat.reshape(T, H, W, 3)

            # Transform trajectory to odom frame
            trajectory = transform_points(T_cam_to_odom, trajectory)

            # Transform camera poses to odom frame
            camera_poses_odom = np.zeros_like(camera_poses)
            for i in range(T):
                camera_poses_odom[i] = T_cam_to_odom @ camera_poses[i]
            camera_poses = camera_poses_odom

        T, H, W = points.shape[:3]
        print(f"[Pi3Decoder] Reconstructed {T} frames with {H}x{W} points each")
        print(f"[Pi3Decoder] Total points: {T * H * W}")
        print(f"[Pi3Decoder] Scene range: x=[{points[..., 0].min():.2f}, {points[..., 0].max():.2f}], "
              f"y=[{points[..., 1].min():.2f}, {points[..., 1].max():.2f}], "
              f"z=[{points[..., 2].min():.2f}, {points[..., 2].max():.2f}]")

        return {
            'trajectory': trajectory,
            'points': points,
            'colors': frames,
            'confidence': conf,
            'camera_poses': camera_poses,
        }
