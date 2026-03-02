"""
Detic Object Detector Wrapper for Goal Detection
Provides object detection and 3D localization for navigation goals
"""

import sys
import os
import numpy as np
import torch
import cv2

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
        self.goal_name = None

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

    def set_goal(self, goal_name: str):
        """
        Set the goal object to detect and update classifier

        Args:
            goal_name: Name of the object to detect (e.g., "chair", "door")
        """
        self.goal_name = goal_name
        vocabulary = [goal_name]

        # Generate CLIP embeddings for custom vocabulary
        texts = ['a ' + x for x in vocabulary]
        with torch.no_grad():
            emb = self.text_encoder(texts).detach().permute(1, 0).contiguous().cpu()

        # Reset classifier with new vocabulary
        num_classes = len(vocabulary)
        reset_cls_test(self.predictor.model, emb, num_classes)

        print(f"[DeticDetector] Set goal to: '{goal_name}'")

    def detect(self, rgb_image: np.ndarray):
        """
        Run detection on RGB image

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)

        Returns:
            List of (bbox, confidence) tuples for detected goal objects
            bbox format: [x1, y1, x2, y2]
        """
        if self.goal_name is None:
            raise ValueError("Goal not set. Call set_goal() first.")

        # Convert RGB to BGR for Detic
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Run inference
        predictions = self.predictor(bgr_image)

        # Extract instances
        if "instances" not in predictions:
            return []

        instances = predictions["instances"].to(self.cpu_device)

        # Filter by confidence and goal class (class 0 since we have single-class vocabulary)
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        # Filter detections
        detections = []
        for box, score, cls in zip(boxes, scores, classes):
            if cls == 0 and score >= self.confidence_threshold:
                # Filter out very small detections (likely noise)
                width = box[2] - box[0]
                height = box[3] - box[1]
                if width >= 20 and height >= 20:
                    detections.append((box, score))

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x[1], reverse=True)

        return detections

    def get_3d_position(self, bbox, depth_image, intrinsic, T_cam_odom, T_odom_base):
        """
        Calculate 3D position of detected object in base frame

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            depth_image: Depth image (H, W) in meters
            intrinsic: Camera intrinsic matrix (3, 3)
            T_cam_odom: Camera to odom transform (4, 4)
            T_odom_base: Odom to base transform (4, 4)

        Returns:
            3D position [x, y, z] in base frame
        """
        x1, y1, x2, y2 = bbox.astype(int)

        # Extract depth values within bounding box
        depth_roi = depth_image[y1:y2, x1:x2]

        # Filter invalid depths
        valid_depths = depth_roi[(depth_roi > 0) & (depth_roi < 10.0) & np.isfinite(depth_roi)]

        if len(valid_depths) == 0:
            raise ValueError("No valid depth values in bounding box")

        # Use median depth (robust to outliers)
        depth = np.median(valid_depths)

        # Calculate bbox center in image coordinates
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Unproject to 3D camera coordinates
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx_intrinsic = intrinsic[0, 2]
        cy_intrinsic = intrinsic[1, 2]

        x_cam = (cx - cx_intrinsic) * depth / fx
        y_cam = (cy - cy_intrinsic) * depth / fy
        z_cam = depth

        # Point in camera frame (homogeneous coordinates)
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])

        # Transform to odom frame
        point_odom = T_cam_odom @ point_cam

        # Transform to base frame
        T_base_odom = np.linalg.inv(T_odom_base)
        point_base = T_base_odom @ point_odom

        return point_base[:3]
