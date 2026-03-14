"""
Detic Object Detector Wrapper for Goal Detection
Provides object detection and 3D localization for navigation goals
"""

import sys
import os
import numpy as np
import torch
import cv2
from typing import List, Dict, Optional

# Add Detic paths
DETIC_ROOT = '/home/liujy/mobile_manipulation/perception_modules/Detic'
sys.path.insert(0, DETIC_ROOT)
sys.path.insert(0, os.path.join(DETIC_ROOT, 'third_party/CenterNet2/'))

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder


class DeticDetector:
    """
    Wrapper for Detic object detection with custom vocabulary
    """

    def __init__(self, config_file, model_weights, device='cuda', confidence_threshold=0.5):
        """
        Initialize Detic detector

        Args:
            config_file: Path to Detic config YAML
            model_weights: Path to model weights (.pth)
            device: 'cuda' or 'cpu'
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.goal_name: Optional[str] = None
        self.goal_pool: List[str] = []
        self.goal_to_class: Dict[str, int] = {}
        self.active_goal_class: Optional[int] = None

        # Setup config
        cfg = get_cfg()
        cfg.MODEL.DEVICE = device
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(config_file)

        # Set confidence thresholds
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # Load later
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

        # Load model weights
        cfg.MODEL.WEIGHTS = model_weights
        cfg.freeze()

        # Initialize predictor (need to be in Detic root directory for relative paths)
        original_cwd = os.getcwd()
        try:
            os.chdir(DETIC_ROOT)
            self.predictor = DefaultPredictor(cfg)
            self.cpu_device = torch.device("cpu")
        finally:
            os.chdir(original_cwd)

        # Text encoder for custom vocabulary
        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()

        print(f"[DeticDetector] Initialized with confidence_threshold={confidence_threshold}")

    def set_goal_pool(self, goal_names: List[str]):
        """
        Set a pool of candidate goal classes and update classifier once.

        Args:
            goal_names: List of object class names
        """
        if goal_names is None:
            raise ValueError("goal_names is required")

        # Remove empty names and keep order while deduplicating.
        vocabulary = []
        seen = set()
        for name in goal_names:
            n = str(name).strip()
            if not n or n in seen:
                continue
            seen.add(n)
            vocabulary.append(n)

        if len(vocabulary) == 0:
            raise ValueError("goal_names is empty after cleanup")

        # Generate CLIP embeddings for custom vocabulary
        texts = ['a ' + x for x in vocabulary]
        with torch.no_grad():
            emb = self.text_encoder(texts).detach().permute(1, 0).contiguous().cpu()

        # Reset classifier with new vocabulary
        num_classes = len(vocabulary)
        reset_cls_test(self.predictor.model, emb, num_classes)
        self.goal_pool = vocabulary
        self.goal_to_class = {name: idx for idx, name in enumerate(vocabulary)}

        # If previous active goal is no longer valid, clear it.
        if self.goal_name not in self.goal_to_class:
            self.goal_name = None
            self.active_goal_class = None

        print(f"[DeticDetector] Set goal pool: {self.goal_pool}")

    def set_active_goal(self, goal_name: str):
        """
        Select one active class from preloaded goal pool.

        Args:
            goal_name: Name of active object class for filtering detections
        """
        if len(self.goal_pool) == 0:
            raise ValueError("Goal pool not set. Call set_goal_pool() first.")
        if goal_name not in self.goal_to_class:
            raise ValueError(f"Goal '{goal_name}' not in goal pool: {self.goal_pool}")

        self.goal_name = goal_name
        self.active_goal_class = int(self.goal_to_class[goal_name])
        print(f"[DeticDetector] Active goal set to: '{goal_name}' (class_id={self.active_goal_class})")

    def get_available_goals(self) -> List[str]:
        """Return available preloaded goal classes."""
        return list(self.goal_pool)

    def set_goal(self, goal_name: str):
        """
        Backward-compatible single-goal API.
        """
        self.set_goal_pool([goal_name])
        self.set_active_goal(goal_name)

    def detect(self, rgb_image: np.ndarray):
        """
        Run detection on RGB image

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)

        Returns:
            List of (bbox, confidence, mask) tuples for detected goal objects
            bbox format: [x1, y1, x2, y2]
            mask format: (H, W) bool array
        """
        if self.active_goal_class is None:
            raise ValueError("Active goal not set. Call set_active_goal() first.")

        # Convert RGB to BGR for Detic
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Run inference
        predictions = self.predictor(bgr_image)

        # Extract instances
        if "instances" not in predictions:
            return []

        instances = predictions["instances"].to(self.cpu_device)
        if not instances.has("pred_masks"):
            raise RuntimeError("Detic output does not contain pred_masks. Check MASK_ON/ROI_MASK_HEAD in config.")

        # Filter by confidence and active goal class
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy().astype(bool)

        # Filter detections
        detections = []
        for box, score, cls, mask in zip(boxes, scores, classes, masks):
            if int(cls) == int(self.active_goal_class) and score >= self.confidence_threshold:
                # Filter out very small detections (likely noise)
                width = box[2] - box[0]
                height = box[3] - box[1]
                if width >= 20 and height >= 20:
                    detections.append((box, score, mask))

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x[1], reverse=True)

        return detections

    def get_3d_position(self, mask, depth_image, intrinsic, T_cam_odom, T_odom_base):
        """
        Calculate 3D position of detected object in base frame using instance mask.

        Args:
            mask: Instance mask (H, W), bool
            depth_image: Depth image (H, W) in meters
            intrinsic: Camera intrinsic matrix (3, 3)
            T_cam_odom: Odom to camera transform (4, 4)
            T_odom_base: Base to odom transform (4, 4)

        Returns:
            3D position [x, y, z] in base frame
        """
        if mask is None:
            raise ValueError("Mask is required for 3D localization")

        if mask.shape != depth_image.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match depth shape {depth_image.shape}")

        mask_bool = mask.astype(bool)

        # Keep only valid depths inside object mask
        valid_depth_mask = (depth_image > 0) & (depth_image < 10.0) & np.isfinite(depth_image)
        object_valid_mask = mask_bool & valid_depth_mask

        if object_valid_mask.sum() == 0:
            raise ValueError("No valid depth values in object mask")

        valid_depths = depth_image[object_valid_mask]

        # Keep near-surface points to reduce background leakage from imperfect masks.
        z_ref = float(np.percentile(valid_depths, 30))
        z_window = 0.25  # meters
        near_surface_mask = object_valid_mask & (np.abs(depth_image - z_ref) <= z_window)
        if near_surface_mask.sum() < 30:
            near_surface_mask = object_valid_mask

        ys, xs = np.where(near_surface_mask)
        zs = depth_image[ys, xs].astype(np.float64)

        print(
            f"[DeticDetector] Mask valid pixels: {object_valid_mask.sum()}, "
            f"near-surface pixels: {near_surface_mask.sum()}, z_ref(p30)={z_ref:.3f}m"
        )
        print(
            f"[DeticDetector] Depth stats (mask): median={np.median(valid_depths):.3f}m, "
            f"min={valid_depths.min():.3f}m, max={valid_depths.max():.3f}m"
        )

        # Unproject near-surface mask points to 3D camera coordinates and use median 3D point.
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx_intrinsic = intrinsic[0, 2]
        cy_intrinsic = intrinsic[1, 2]

        x_cam_pts = (xs.astype(np.float64) - cx_intrinsic) * zs / fx
        y_cam_pts = (ys.astype(np.float64) - cy_intrinsic) * zs / fy
        z_cam_pts = zs

        x_cam = float(np.median(x_cam_pts))
        y_cam = float(np.median(y_cam_pts))
        z_cam = float(np.median(z_cam_pts))

        print(f"[DeticDetector] Camera frame: x={x_cam:.3f}, y={y_cam:.3f}, z={z_cam:.3f}")

        # Point in camera frame (homogeneous coordinates)
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])

        # Transform to odom frame.
        # T_cam_odom is odom->cam, so we need its inverse (cam->odom).
        T_odom_cam = np.linalg.inv(T_cam_odom)
        point_odom = T_odom_cam @ point_cam

        print(f"[DeticDetector] Odom frame: x={point_odom[0]:.3f}, y={point_odom[1]:.3f}, z={point_odom[2]:.3f}")

        # Transform to base frame
        T_base_odom = np.linalg.inv(T_odom_base)
        point_base = T_base_odom @ point_odom

        print(f"[DeticDetector] Base frame: x={point_base[0]:.3f}, y={point_base[1]:.3f}, z={point_base[2]:.3f}")

        return point_base[:3]
