#!/usr/bin/python3
"""
Target detection module using Detic
Detects specific objects in images
"""
import sys
import os
import cv2
import numpy as np

# Add Detic paths
DETIC_PATH = '/home/liujy/mobile_manipulation/perception_modules/Detic'
sys.path.insert(0, DETIC_PATH)
sys.path.insert(0, os.path.join(DETIC_PATH, 'third_party/CenterNet2/'))

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.predictor import VisualizationDemo


class TargetDetector:
    """Wrapper for Detic object detection"""
    
    def __init__(self, 
                 target_name="banana",
                 confidence_threshold=0.5,
                 config_file="configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
                 model_weights="models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"):
        """
        Initialize Detic detector
        
        Args:
            target_name: Name of object to detect
            confidence_threshold: Minimum confidence score
            config_file: Path to Detic config (relative to DETIC_PATH)
            model_weights: Path to model weights (relative to DETIC_PATH)
        """
        self.target_name = target_name
        self.confidence_threshold = confidence_threshold
        
        # Save current directory
        self.original_cwd = os.getcwd()
        
        # Setup paths
        self.config_file = os.path.join(DETIC_PATH, config_file)
        self.model_weights = os.path.join(DETIC_PATH, model_weights)
        
        # Check if files exist
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        if not os.path.exists(self.model_weights):
            raise FileNotFoundError(f"Model weights not found: {self.model_weights}")
        
        # Change to Detic directory (required for relative paths to metadata)
        os.chdir(DETIC_PATH)
        
        try:
            # Setup config
            self.cfg = self._setup_cfg()
            
            # Initialize detector
            self.demo = VisualizationDemo(self.cfg, self._create_args())
            
            print(f"TargetDetector initialized for '{target_name}' with threshold {confidence_threshold}")
        finally:
            # Restore original directory
            os.chdir(self.original_cwd)
    
    def _create_args(self):
        """Create args object compatible with VisualizationDemo"""
        class Args:
            def __init__(self):
                self.vocabulary = 'custom'
                self.custom_vocabulary = ''
                self.pred_all_class = False
        
        args = Args()
        args.custom_vocabulary = self.target_name
        return args
    
    def _setup_cfg(self):
        """Setup Detectron2 config"""
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cuda"  # Use GPU
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(self.config_file)
        
        # Set thresholds
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.confidence_threshold
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        
        # Set model weights
        cfg.MODEL.WEIGHTS = self.model_weights
        
        cfg.freeze()
        return cfg
    
    def detect(self, image_path):
        """
        Detect target in image file
        """
        image_path = os.path.abspath(image_path)
        img = read_image(image_path, format="BGR")
        return self._detect_core(img, output_path=None)
    
    def _parse_predictions(self, predictions, image_shape):
        """Parse Detectron2 predictions into simple dict"""
        result = {
            'detected': False,
            'num_instances': 0,
            'boxes': [],
            'scores': [],
            'center_x': 0.0,
            'center_y': 0.0,
            'bbox': [],
            'score': 0.0
        }
        
        if 'instances' not in predictions:
            return result
        
        instances = predictions['instances']
        num_instances = len(instances)
        
        if num_instances == 0:
            return result
        
        # Get boxes and scores
        boxes = instances.pred_boxes.tensor.cpu().numpy()  # [N, 4]
        scores = instances.scores.cpu().numpy()  # [N]
        
        # Convert to lists
        result['detected'] = True
        result['num_instances'] = num_instances
        result['boxes'] = boxes.tolist()
        result['scores'] = scores.tolist()
        
        # Get highest confidence detection
        max_idx = np.argmax(scores)
        best_box = boxes[max_idx]
        best_score = scores[max_idx]
        
        # Calculate center
        x1, y1, x2, y2 = best_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        result['bbox'] = best_box.tolist()
        result['score'] = float(best_score)
        result['center_x'] = float(center_x)
        result['center_y'] = float(center_y)
        
        return result
    
    def detect_from_cv_image(self, cv_image, output_path=None):
        """
        Detect target directly from an OpenCV BGR image
        """
        if cv_image is None:
            raise ValueError("cv_image cannot be None")
        return self._detect_core(cv_image, output_path=output_path)

    def _detect_core(self, cv_image, output_path=None):
        """Run detector on provided BGR image"""
        current_dir = os.getcwd()
        os.chdir(DETIC_PATH)
        try:
            predictions, visualized_output = self.demo.run_on_image(cv_image)
            if output_path:
                visualized_output.save(output_path)
            return self._parse_predictions(predictions, cv_image.shape)
        finally:
            os.chdir(current_dir)
    
    def visualize_detection(self, image_path, output_path=None):
        """
        Run detection on file path (kept for compatibility)
        """
        img = read_image(os.path.abspath(image_path), format="BGR")
        return self.detect_from_cv_image(img, output_path)


# Test code
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--target", default="banana", help="Target object name")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", help="Output visualization path")
    args = parser.parse_args()
    
    # Create detector
    detector = TargetDetector(
        target_name=args.target,
        confidence_threshold=args.threshold
    )
    
    # Run detection
    if args.output:
        result = detector.visualize_detection(args.image, args.output)
        print(f"Visualization saved to: {args.output}")
    else:
        result = detector.detect(args.image)
    
    # Print results
    print("\nDetection Results:")
    print("=" * 50)
    print(f"Target: {args.target}")
    print(f"Detected: {result['detected']}")
    print(f"Number of instances: {result['num_instances']}")
    
    if result['detected']:
        print(f"Best detection score: {result['score']:.3f}")
        print(f"Bounding box: {result['bbox']}")
        print(f"Center: ({result['center_x']:.1f}, {result['center_y']:.1f})")
        print(f"\nAll detections:")
        for i, (box, score) in enumerate(zip(result['boxes'], result['scores'])):
            print(f"  {i+1}. Score: {score:.3f}, Box: {box}")
    
    print("=" * 50)

