#!/usr/bin/env python3
"""
Flask Server for LLM Navigation
Server端负责推理（NavAgent），返回Path或Action
"""
import os
import sys
import math
import json
import time
import re
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from nav_agent import NavAgent
from detic_detector import DeticDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'llm_nav_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
nav_state = {
    'agent': None,
    'detic_detector': None,
    'detic_goal_pool': [],
    'detic_init_confidence': 0.7,
    'goal': None,
    'goal_description': '',
    'iteration_count': 0,
    'log_dir': None,
    'latest_state': None,  # Store latest visualization state
    'blocked_detections': [],  # Rejected detection regions for current episode
}

DETIC_CONFIG_PATH = '/home/liujy/mobile_manipulation/perception_modules/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
DETIC_WEIGHTS_PATH = '/home/liujy/mobile_manipulation/perception_modules/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'


def setup_log_directory(goal: str = '', goal_description: str = ''):
    """Create log directory for this session and save metadata"""
    base_dir = os.path.join(script_dir, '..', 'logs')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Save goal and description metadata
    metadata = {
        'goal': goal,
        'goal_description': goal_description,
        'timestamp': timestamp
    }
    metadata_path = os.path.join(log_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return log_dir


def yaw_to_quaternion(yaw: float) -> Dict[str, float]:
    """Convert yaw to quaternion dict"""
    return {
        'x': 0.0,
        'y': 0.0,
        'z': math.sin(yaw / 2.0),
        'w': math.cos(yaw / 2.0)
    }


def mat_to_yaw(T_4x4: np.ndarray) -> float:
    """Extract yaw from 4x4 transformation matrix"""
    Rm = T_4x4[:3, :3].astype(np.float64)
    return math.atan2(Rm[1, 0], Rm[0, 0])


def quat_dict_to_yaw(q: Dict[str, float]) -> float:
    """Convert quaternion dict to yaw"""
    return math.atan2(
        2.0 * (q['w'] * q['z'] + q['x'] * q['y']),
        1.0 - 2.0 * (q['y'] * q['y'] + q['z'] * q['z'])
    )


def build_goal_pose(x: float, y: float, z: float, yaw: float) -> Dict[str, Any]:
    """
    Build a single goal pose
    Returns a pose dict with position and orientation
    """
    return {
            'position': {'x': float(x), 'y': float(y), 'z': float(z)},
        'orientation': yaw_to_quaternion(float(yaw))
    }


def compute_bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    xa1, ya1, xa2, ya2 = [float(v) for v in box_a]
    xb1, yb1, xb2, yb2 = [float(v) for v in box_b]
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if a.shape != b.shape:
        return 0.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def is_detection_blocked(
    bbox: np.ndarray,
    mask: np.ndarray,
    blocked_detections: List[Dict[str, Any]],
    iou_thresh: float = 0.7,
) -> bool:
    """Check whether current detection overlaps with blocked regions."""
    if mask is None or len(blocked_detections) == 0:
        return False

    curr_bbox = np.array(bbox, dtype=np.float32)
    curr_mask = mask.astype(bool)
    for blocked in blocked_detections:
        blocked_bbox = np.array(blocked['bbox'], dtype=np.float32)
        # Fast prefilter on bbox IoU first.
        if compute_bbox_iou(curr_bbox, blocked_bbox) <= 0.01:
            continue
        blocked_mask = blocked['mask'].astype(bool)
        miou = compute_mask_iou(curr_mask, blocked_mask)
        if miou >= iou_thresh:
            return True
    return False


def build_detic_verify_prompt(goal: str, goal_description: str = "") -> str:
    """Build strict binary verification prompt for Detic hit."""
    desc = goal_description.strip()
    desc_text = desc if desc else "N/A"
    return (
        "You are verifying an object detection for robot navigation.\n"
        f"Target object: {goal}\n"
        f"Target description: {desc_text}\n"
        "Input image: RGB with one highlighted bounding box.\n"
        "Task: Decide whether the highlighted box is truly the target object.\n"
        "Output must be ONLY JSON, no extra text:\n"
        '{"is_target": true}\n'
        "or\n"
        '{"is_target": false}'
    )


def parse_detic_verify_result(response_text: str) -> Optional[bool]:
    """Parse VLM verification response. Returns True/False/None(parse fail)."""
    if response_text is None:
        return None
    text = str(response_text).strip()
    if len(text) == 0:
        return None

    # Try JSON object extraction first.
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group(0))
            value = data.get("is_target", None)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                low = value.strip().lower()
                if low in {"true", "yes", "y"}:
                    return True
                if low in {"false", "no", "n"}:
                    return False
        except Exception:
            pass

    low = text.lower()
    if "is_target" in low and "true" in low:
        return True
    if "is_target" in low and "false" in low:
        return False
    if " yes" in f" {low} " and " no" not in f" {low} ":
        return True
    if " no" in f" {low} " and " yes" not in f" {low} ":
        return False
    return None


def parse_detic_goal_pool(goal_spec: Any) -> List[str]:
    """Parse Detic goal pool from list/tuple/string."""
    if goal_spec is None:
        return []
    if isinstance(goal_spec, (list, tuple)):
        raw_items = [str(x) for x in goal_spec]
    else:
        raw_items = str(goal_spec).split(',')

    goals: List[str] = []
    seen = set()
    for item in raw_items:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        goals.append(name)
    return goals


def ensure_detic_detector(
    selected_goal: str,
    confidence_threshold: float,
    requested_goal_pool: Optional[List[str]] = None,
) -> (bool, str):
    """
    Ensure Detic is initialized with a goal pool and set active goal.
    Returns (ok, message).
    """
    global nav_state

    if requested_goal_pool:
        if nav_state.get('detic_goal_pool'):
            if requested_goal_pool != nav_state['detic_goal_pool']:
                print(
                    f"[Server] Requested detic_goal_pool={requested_goal_pool} differs from "
                    f"startup pool={nav_state['detic_goal_pool']}; keeping startup pool"
                )
        else:
            nav_state['detic_goal_pool'] = requested_goal_pool

    if nav_state['detic_detector'] is None:
        try:
            init_conf = float(nav_state.get('detic_init_confidence', confidence_threshold))
            detector = DeticDetector(
                config_file=DETIC_CONFIG_PATH,
                model_weights=DETIC_WEIGHTS_PATH,
                device='cuda',
                confidence_threshold=init_conf
            )
            goal_pool = nav_state.get('detic_goal_pool', [])
            if len(goal_pool) == 0:
                goal_pool = [selected_goal]
            detector.set_goal_pool(goal_pool)
            nav_state['detic_detector'] = detector
            nav_state['detic_goal_pool'] = detector.get_available_goals()
            print(f"[Server] Detic detector initialized with goal pool: {nav_state['detic_goal_pool']}")
        except Exception as e:
            nav_state['detic_detector'] = None
            return False, f"Failed to initialize Detic: {e}"

    detector = nav_state['detic_detector']
    detector.confidence_threshold = float(confidence_threshold)
    available_goals = detector.get_available_goals()
    if selected_goal not in available_goals:
        return False, (
            f"Selected goal '{selected_goal}' not in Detic goal pool: {available_goals}. "
            f"Restart server with --detic-goals to include it."
        )

    try:
        detector.set_active_goal(selected_goal)
    except Exception as e:
        return False, f"Failed to set active Detic goal '{selected_goal}': {e}"

    return True, ""


@app.route('/navigation_reset', methods=['POST'])
def navigation_reset():
    """
    Initialize navigation session
    Request JSON: {
        "goal": str,  # navigation goal (e.g., "door", "chair")
        "goal_description": str,  # optional description of goal location
        "confidence_threshold": float,  # optional, default 0.5
    }
    """
    global nav_state
    
    data = request.get_json()
    goal = data.get('goal')
    goal_description = data.get('goal_description', '')
    confidence_threshold = data.get('confidence_threshold', 0.7)
    requested_goal_pool = parse_detic_goal_pool(data.get('detic_goal_pool', None))
    
    if not goal:
        return jsonify({'error': 'goal is required'}), 400
    
    # Initialize components
    nav_state['goal'] = goal
    nav_state['goal_description'] = goal_description or ''
    nav_state['iteration_count'] = 0
    nav_state['log_dir'] = setup_log_directory(goal, goal_description or '')
    nav_state['blocked_detections'] = []
    
    # Build agent config
    # agent_cfg = {
    #     'vlm_model': '/data/sea_disk0/liujy/models/Qwen/Qwen3.5-27B-GPTQ-Int4/',

    #     'vlm_api_key': 'EMPTY',
    #     'vlm_base_url': 'http://10.15.89.71:32054/v1/',
    #     # 'vlm_base_url': 'http://10.15.89.71:34134/v1/',

    #     # Navigation parameters
    #     'enable_bev_map': True,
    #     'bev_map_size': 600,
    #     'bev_pixels_per_meter': 20,
    #     'clip_dist': 4.0,

    #     # Increase timeout for complex vision requests
    #     'vlm_history': 10,
    #     'vlm_timeout': 600,  # 120 seconds for vision + navigation requests
    # }

    agent_cfg = {
        'vlm_model': 'gpt-5.2',
        'vlm_api_key': os.getenv('OPENAI_API_KEY'),
        'vlm_base_url': 'https://api.openai.com/v1/',
    
        # Navigation parameters
        'enable_bev_map': True,
        'bev_map_size': 800,
        'bev_pixels_per_meter': 20,
        'clip_dist': 4.0,
        'bev_min_depth': 0.51,
        'bev_max_depth': 4.99,
        'frontier_visibility_max_depth': 4.99,
    
        # Increase timeout for complex vision requests
        'vlm_history': 0,
        'vlm_timeout': 600,
    }
    
    # Initialize NavAgent (always recreate to ensure config is updated)
    nav_state['agent'] = NavAgent(cfg=agent_cfg)
    nav_state['agent'].reset(goal=goal, goal_description=goal_description or '')

    ok, detic_msg = ensure_detic_detector(
        selected_goal=goal,
        confidence_threshold=float(confidence_threshold),
        requested_goal_pool=requested_goal_pool if len(requested_goal_pool) > 0 else None,
    )
    if not ok:
        return jsonify({'error': detic_msg}), 400

    print(f"[Server] Navigation reset: goal='{goal}', description='{goal_description}', log_dir={nav_state['log_dir']}")
    print(f"[Server] NavAgent reset with goal information, VLM initialized with full task briefing")
    
    return jsonify({
        'status': 'success',
        'goal': goal,
        'goal_description': goal_description,
        'log_dir': nav_state['log_dir'],
        'detic_available_goals': nav_state['detic_detector'].get_available_goals() if nav_state['detic_detector'] else [],
        'detic_active_goal': goal,
    })


@app.route('/navigation_step', methods=['POST'])
def navigation_step():
    """
    Single navigation step with VLM planning
    
    Request:
        - Files: 'rgb' (JPEG/PNG), 'depth' (PNG 16-bit or NPY)
        - Form data:
            'intrinsic': JSON string, 3x3 matrix flattened
            'T_cam_odom': JSON string, 4x4 matrix flattened (odom->cam)
            'T_odom_base': JSON string, 4x4 matrix flattened (base->odom)
    
    Returns JSON: {
        'action_type': 'nav_step' | 'turn_around' | 'none',
        'goal_pose': {
        'frame_id': 'odom',
            'pose': {
                'position': {'x': float, 'y': float, 'z': float},
                'orientation': {'x': float, 'y': float, 'z': float, 'w': float}
            }
        } | None,
        'iteration': int,
        'finished': bool,
        'timing': {
            'total_server_time': float,
            'vlm_projection_time': float,
            'vlm_inference_time': float
        }
    }
    
    Note: Client should use nav2's NavigateToPose action with the returned goal_pose
    """
    global nav_state
    
    # Start timing the entire server processing
    t_server_start = time.time()
    
    if nav_state['goal'] is None:
        return jsonify({'error': 'Navigation not initialized. Call /navigation_reset first.'}), 400
    
    nav_state['iteration_count'] += 1
    iteration = nav_state['iteration_count']
    
    print(f"\n[Server] === ITERATION {iteration} ===")
    
    # Parse image files
    try:
        rgb_file = request.files['rgb']
        depth_file = request.files['depth']
        
        # Load RGB
        rgb_pil = Image.open(rgb_file.stream).convert('RGB')
        rgb = np.asarray(rgb_pil)  # (H, W, 3)
        
        # Load Depth
        if depth_file.filename.endswith('.npy'):
            depth = np.load(depth_file.stream).astype(np.float32)
        else:
            depth_pil = Image.open(depth_file.stream)
            if depth_pil.mode == 'I':  # 32-bit int
                depth = np.asarray(depth_pil).astype(np.float32) / 1000.0  # mm to m
            elif depth_pil.mode == 'I;16':  # 16-bit
                depth = np.asarray(depth_pil).astype(np.float32) / 1000.0
            else:
                depth = np.asarray(depth_pil).astype(np.float32)
        
        # Handle invalid depth
        depth[~np.isfinite(depth)] = 0.0
        
    except Exception as e:
        return jsonify({'error': f'Failed to load images: {str(e)}'}), 400
    
    # Parse transforms
    try:
        form_data = request.form
        intrinsic = np.array(json.loads(form_data['intrinsic']), dtype=np.float32).reshape(3, 3)
        T_cam_odom = np.array(json.loads(form_data['T_cam_odom']), dtype=np.float32).reshape(4, 4)
        T_odom_base = np.array(json.loads(form_data['T_odom_base']), dtype=np.float32).reshape(4, 4)
    except Exception as e:
        return jsonify({'error': f'Failed to parse transforms: {str(e)}'}), 400

    # === DETIC GOAL DETECTION (SYNCHRONOUS) ===
    # Check if goal is visible before running VLM navigation
    t_detic_start = time.time()
    t_detic_end = t_detic_start
    t_detic_verify = 0.0
    detic_verify_result = 'not_triggered'
    if nav_state['detic_detector'] is not None:
        try:
            detections = nav_state['detic_detector'].detect(rgb)
            t_detic_end = time.time()

            if len(detections) > 0:
                # Only use top-1 detection as configured.
                bbox, confidence, mask = detections[0]
                print(f"[Server] Detic detected goal '{nav_state['goal']}' with confidence {confidence:.3f}")

                if is_detection_blocked(
                    bbox=bbox,
                    mask=mask,
                    blocked_detections=nav_state.get('blocked_detections', []),
                    iou_thresh=0.7,
                ):
                    detic_verify_result = 'blocked'
                    print("[Server] Detic detection overlapped blocked region (mask IoU >= 0.7), skipping")
                else:
                    # VLM second-stage verification with bbox-highlighted RGB.
                    verify_rgb = rgb.copy()
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(verify_rgb, (x1, y1), (x2, y2), (255, 165, 0), 3)
                    cv2.putText(
                        verify_rgb,
                        f"Detic {confidence:.2f}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 165, 0),
                        2
                    )

                    verify_prompt = build_detic_verify_prompt(
                        nav_state['goal'],
                        nav_state.get('goal_description', ''),
                    )
                    verify_result_bool: Optional[bool] = None
                    verify_response_text = ''
                    t_verify_start = time.time()
                    try:
                        verify_response_text = nav_state['agent'].actionVLM.call(
                            images=[verify_rgb],
                            text_prompt=verify_prompt,
                        )
                        verify_result_bool = parse_detic_verify_result(verify_response_text)
                        if verify_result_bool is True:
                            detic_verify_result = 'yes'
                        elif verify_result_bool is False:
                            detic_verify_result = 'no'
                        else:
                            detic_verify_result = 'verify_parse_error'
                    except Exception as e:
                        print(f"[Server] VLM detic verification failed: {e}")
                        detic_verify_result = 'verify_error'
                    t_detic_verify = time.time() - t_verify_start

                    with open(
                        os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_detic_verify.txt'),
                        'w',
                        encoding='utf-8'
                    ) as f:
                        f.write(
                            f"goal={nav_state['goal']}\n"
                            f"confidence={confidence:.4f}\n"
                            f"verify_result={detic_verify_result}\n"
                            f"verify_response={verify_response_text}\n"
                        )

                    # yes -> keep original approach/finished logic
                    if verify_result_bool is True:
                        try:
                            print(f"[Server] === Transformation Matrices ===")
                            print(f"[Server] T_cam_odom (odom->camera):\n{T_cam_odom}")
                            print(f"[Server] T_odom_base (base->odom):\n{T_odom_base}")
                            print(f"[Server] Robot position (odom): {T_odom_base[:3, 3]}")
                            print(f"[Server] Robot rotation (odom):\n{T_odom_base[:3, :3]}")

                            target_pos_base = nav_state['detic_detector'].get_3d_position(
                                mask, depth, intrinsic, T_cam_odom, T_odom_base
                            )

                            distance = np.linalg.norm(target_pos_base[:2])
                            angle_to_target = np.arctan2(target_pos_base[1], target_pos_base[0])
                            print(f"[Server] Goal verified by VLM at distance {distance:.2f}m, angle {np.rad2deg(angle_to_target):.1f}°")

                            final_distance = 0.8
                            current_pos_odom = T_odom_base[:3, 3]
                            target_pos_odom = current_pos_odom + T_odom_base[:3, :3] @ target_pos_base

                            print(f"[Server] Target in base frame: [{target_pos_base[0]:.3f}, {target_pos_base[1]:.3f}, {target_pos_base[2]:.3f}]")
                            print(f"[Server] Target in odom frame: [{target_pos_odom[0]:.3f}, {target_pos_odom[1]:.3f}, {target_pos_odom[2]:.3f}]")
                            print(f"[Server] Robot to target vector (odom): [{target_pos_odom[0]-current_pos_odom[0]:.3f}, "
                                  f"{target_pos_odom[1]-current_pos_odom[1]:.3f}]")

                            direction = target_pos_odom[:2] - current_pos_odom[:2]
                            direction_norm = direction / np.linalg.norm(direction)
                            approach_pos = target_pos_odom[:2] - direction_norm * final_distance
                            approach_yaw = np.arctan2(
                                target_pos_odom[1] - approach_pos[1],
                                target_pos_odom[0] - approach_pos[0]
                            )

                            goal_pose = build_goal_pose(
                                float(approach_pos[0]),
                                float(approach_pos[1]),
                                float(current_pos_odom[2]),
                                float(approach_yaw)
                            )

                            print(f"[Server] Final pose: position=({approach_pos[0]:.2f}, {approach_pos[1]:.2f}), "
                                  f"yaw={np.rad2deg(approach_yaw):.1f}°, distance to target={final_distance}m")

                            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_rgb.jpg'), bgr)

                            rgb_vis = rgb.copy()
                            cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(rgb_vis, f"{nav_state['goal']} {confidence:.2f}",
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.9, (0, 255, 0), 2)
                            cv2.putText(rgb_vis, f"Distance: {distance:.2f}m",
                                       (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.7, (0, 255, 0), 2)
                            cv2.imwrite(
                                os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_rgb_vis.jpg'),
                                cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
                            )

                            if nav_state['agent'].obstacle_map is not None:
                                robot_pos = T_odom_base[:2, 3]
                                bev_map = nav_state['agent'].obstacle_map.visualize(robot_pos=robot_pos, show_frontiers=False)
                                cv2.imwrite(
                                    os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_bev_map.jpg'),
                                    cv2.cvtColor(bev_map, cv2.COLOR_RGB2BGR)
                                )

                            response_text = (
                                f"[Detic+VLM Verification - GOAL FOUND]\n\n"
                                f"Goal: {nav_state['goal']}\n"
                                f"Detic confidence: {confidence:.3f}\n"
                                f"Verification: YES\n"
                                f"Distance: {distance:.2f}m\n"
                                f"Angle: {np.rad2deg(angle_to_target):.1f}°\n\n"
                                f"Final approach pose:\n"
                                f"  Position: ({approach_pos[0]:.2f}, {approach_pos[1]:.2f})\n"
                                f"  Yaw: {np.rad2deg(approach_yaw):.1f}°\n"
                                f"  Distance to target: {final_distance}m\n\n"
                                f"Navigation finished successfully!"
                            )
                            with open(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_response.txt'), 'w') as f:
                                f.write(response_text)

                            timing_data = {
                                'iteration': iteration,
                                'action_type': 'nav_step',
                                'detic_detection_time': float(t_detic_end - t_detic_start),
                                'detic_verify_time': float(t_detic_verify),
                                'detic_verify_result': detic_verify_result,
                                'blocked_regions_count': int(len(nav_state.get('blocked_detections', []))),
                                'total_server_time': float(time.time() - t_server_start),
                                'finished': True
                            }
                            timing_path = os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_timing.json')
                            with open(timing_path, 'w') as f:
                                json.dump(timing_data, f, indent=2)

                            log_name = os.path.basename(nav_state['log_dir'])
                            viz_state = {
                                'iteration': iteration,
                                'goal': nav_state['goal'],
                                'goal_description': nav_state['goal_description'],
                                'action_type': 'nav_step',
                                'images': {
                                    'rgb': f'/logs/{log_name}/iter_{iteration:04d}_rgb.jpg',
                                    'rgb_vis': f'/logs/{log_name}/iter_{iteration:04d}_rgb_vis.jpg',
                                    'bev_map': f'/logs/{log_name}/iter_{iteration:04d}_bev_map.jpg'
                                },
                                'llm_response': response_text,
                                'current_cycle': [],
                                'historical_keyframes': []
                            }
                            nav_state['latest_state'] = viz_state
                            socketio.emit('navigation_update', viz_state)

                            print(f"[Server] Navigation finished! Goal '{nav_state['goal']}' verified and approached.")
                            return jsonify({
                                'action_type': 'nav_step',
                                'goal_pose': {
                                    'frame_id': 'odom',
                                    'pose': goal_pose
                                },
                                'iteration': iteration,
                                'finished': True,
                                'timing': {
                                    'total_server_time': time.time() - t_server_start,
                                    'vlm_projection_time': 0.0,
                                    'vlm_inference_time': 0.0,
                                    'detic_detection_time': t_detic_end - t_detic_start,
                                    'detic_verify_time': t_detic_verify,
                                    'detic_verify_result': detic_verify_result,
                                    'blocked_regions_count': int(len(nav_state.get('blocked_detections', []))),
                                }
                            })

                        except Exception as e:
                            print(f"[Server] Failed to calculate 3D position after verification: {e}")
                            # Continue with normal navigation on error.
                    elif verify_result_bool is False:
                        if mask is not None:
                            nav_state.setdefault('blocked_detections', []).append({
                                'bbox': [float(v) for v in bbox],
                                'mask': mask.astype(bool),
                            })
                            print(
                                f"[Server] VLM rejected Detic hit, added blocked region. "
                                f"blocked_count={len(nav_state['blocked_detections'])}"
                            )
                        else:
                            print("[Server] VLM rejected Detic hit but mask is None, skip blocking this region.")
                    else:
                        print("[Server] VLM verification unavailable/parse-failed, skipping this detection without blocking.")

            else:
                print(f"[Server] Detic: No detection of '{nav_state['goal']}', continuing with VLM navigation")

        except Exception as e:
            print(f"[Server] Detic detection failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue with VLM navigation on error

    # If no detection or Detic not initialized, continue with normal VLM navigation...

    # Prepare obs dict (same format as original code)
    obs = {
        'rgb': rgb,
        'depth': depth,
        'intrinsic': intrinsic,
        'extrinsic': T_cam_odom,
        'base_to_odom_matrix': T_odom_base
    }
    
    # Save images for debugging
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_rgb.jpg'), bgr)
    
    # LLM Navigation Step
    print("[Server] Calling NavAgent for decision...")
    response, rgb_vis, action, nav_timing, bev_map = nav_state['agent']._nav(
        obs, nav_state['goal'], iteration, goal_description=nav_state['goal_description']
    )

    print(f"[Server] NavAgent action: {action}")

    # Save visualization
    if rgb_vis is not None:
        cv2.imwrite(
            os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_rgb_vis.jpg'),
            cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
        )
    if response is not None:
        with open(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_response.txt'), 'w') as f:
            f.write(response)

    # Save BEV map (the actual BEV map from obstacle_map, not the old explored_map)
    if bev_map is not None:
        cv2.imwrite(
            os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_bev_map.jpg'),
            cv2.cvtColor(bev_map, cv2.COLOR_RGB2BGR)
        )

    # Save frontier visualization
    if nav_state['agent'].obstacle_map is not None:
        robot_pos = T_odom_base[:2, 3]
        frontier_vis = nav_state['agent'].obstacle_map.visualize_frontiers(robot_pos=robot_pos)
        cv2.imwrite(
            os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_frontiers.jpg'),
            cv2.cvtColor(frontier_vis, cv2.COLOR_RGB2BGR)
        )

    # Save prompt for debugging (if available in timing info)
    if nav_timing.get('prompt'):
        with open(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_prompt.txt'), 'w', encoding='utf-8') as f:
            f.write(nav_timing['prompt'])

    # Save voxel map for debugging (optional, deprecated)
    # Note: voxel_map and explored_map are deprecated in favor of ObstacleMap
    # Keeping this for backward compatibility, but it's no longer updated when using BEV
    voxel_map = nav_state['agent'].voxel_map
    if voxel_map is not None and voxel_map.sum() > 0:  # Only save if it has data
        cv2.imwrite(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_voxel_map.jpg'), voxel_map)
    
    # Save timing information (will be updated at the end with total time and action type)
    timing_data = {
        'iteration': iteration,
        'detic_detection_time': float(t_detic_end - t_detic_start),
        'detic_verify_time': float(t_detic_verify),
        'detic_verify_result': detic_verify_result,
        'blocked_regions_count': int(len(nav_state.get('blocked_detections', []))),
        'vlm_projection_time': float(nav_timing.get('projection_time', 0.0)),
        'vlm_inference_time': float(nav_timing.get('vlm_inference_time', 0.0))
    }
    timing_path = os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_timing.json')
    with open(timing_path, 'w') as f:
        json.dump(timing_data, f, indent=2)
    
    # Prepare visualization state for Socket.IO
    log_name = os.path.basename(nav_state['log_dir'])
    
    # Build history: show last 4 iterations as current cycle
    current_cycle = []
    for prev_iter in range(max(1, iteration - 3), iteration):
        hist_img_path = f'/logs/{log_name}/iter_{prev_iter:04d}_rgb_vis.jpg'
        hist_img_full_path = os.path.join(nav_state['log_dir'], f'iter_{prev_iter:04d}_rgb_vis.jpg')
        if os.path.exists(hist_img_full_path):
            # Try to read action_type from timing file
            action_type_str = None
            timing_file = os.path.join(nav_state['log_dir'], f'iter_{prev_iter:04d}_timing.json')
            if os.path.exists(timing_file):
                try:
                    with open(timing_file, 'r') as f:
                        timing_data_prev = json.load(f)
                        action_type_str = timing_data_prev.get('action_type', None)
                except:
                    pass
            
            current_cycle.append({
                'iteration': prev_iter,
                'action': action_type_str,
                'image_path': hist_img_path
            })
    
    viz_state = {
        'iteration': iteration,
        'goal': nav_state['goal'],
        'goal_description': nav_state['goal_description'],
        'action_type': None,
        'images': {},
        'llm_response': response if response else '',
        'current_cycle': current_cycle,
        'historical_keyframes': []
    }
    
    # Add image paths
    if os.path.exists(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_rgb.jpg')):
        viz_state['images']['rgb'] = f'/logs/{log_name}/iter_{iteration:04d}_rgb.jpg'
    if os.path.exists(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_rgb_vis.jpg')):
        viz_state['images']['rgb_vis'] = f'/logs/{log_name}/iter_{iteration:04d}_rgb_vis.jpg'
    if os.path.exists(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_bev_map.jpg')):
        viz_state['images']['bev_map'] = f'/logs/{log_name}/iter_{iteration:04d}_bev_map.jpg'
    
    if action is None:
        viz_state['action_type'] = 'none'
        nav_state['latest_state'] = viz_state
        socketio.emit('navigation_update', viz_state)
        
        # Calculate total server time
        t_server_end = time.time()
        total_server_time = t_server_end - t_server_start
        
        timing_info = {
            'total_server_time': float(total_server_time),
            'vlm_projection_time': float(nav_timing.get('projection_time', 0.0)),
            'vlm_inference_time': float(nav_timing.get('vlm_inference_time', 0.0)),
            'detic_detection_time': float(t_detic_end - t_detic_start),
            'detic_verify_time': float(t_detic_verify),
            'detic_verify_result': detic_verify_result,
            'blocked_regions_count': int(len(nav_state.get('blocked_detections', []))),
        }
        
        # Update timing file with total server time and action type
        timing_path = os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_timing.json')
        if os.path.exists(timing_path):
            with open(timing_path, 'r') as f:
                timing_data = json.load(f)
            timing_data['total_server_time'] = float(total_server_time)
            timing_data['action_type'] = 'none'
            # Keyframe info already saved above
            with open(timing_path, 'w') as f:
                json.dump(timing_data, f, indent=2)
        
        print(f"[Server] Timing - Total: {total_server_time:.3f}s, "
              f"VLM Projection: {timing_info['vlm_projection_time']:.3f}s, "
              f"VLM Inference: {timing_info['vlm_inference_time']:.3f}s")
        
        return jsonify({
            'action_type': 'none',
            'goal_pose': None,
            'iteration': iteration,
            'finished': False,
            'timing': timing_info
        })
    
    # Current base state
    T_odom_base = obs['base_to_odom_matrix'].astype(np.float64)
    x0 = float(T_odom_base[0, 3])
    y0 = float(T_odom_base[1, 3])
    z0 = float(T_odom_base[2, 3])
    yaw0 = mat_to_yaw(T_odom_base)
    
    # Check action type: ('turn', angle) or (r, theta)
    if isinstance(action, tuple) and len(action) == 2 and action[0] == 'turn':
        # Rotation action: ('turn', angle_in_degrees)
        angle_deg = float(action[1])
        angle_rad = math.radians(angle_deg)
        
        if angle_deg > 0:
            print(f"[Server] Action is TURN LEFT: {angle_deg:.1f} degrees")
            action_type = 'turn_left'
        else:
            print(f"[Server] Action is TURN RIGHT: {abs(angle_deg):.1f} degrees")
            action_type = 'turn_right'
        
        # Rotate at current position
        yaw_goal = yaw0 + angle_rad
        goal_pose = build_goal_pose(x0, y0, z0, yaw_goal)
    
    else:
        # Forward movement action: (r, theta) or (r, theta, normal)
        # Extract normal vector if present
        if isinstance(action, tuple) and len(action) == 3:
            r, theta, normal_odom = action
            r, theta = float(r), float(theta)
        elif isinstance(action, tuple) and len(action) == 2:
            r, theta = action
            r, theta = float(r), float(theta)
            normal_odom = None
        else:
            r, theta = float(action[0]), float(action[1])
            normal_odom = None

        print(f"[Server] Action is NAV STEP: r={r:.2f}, theta={theta:.2f}")

        dis = r
        # Compute goal in odom frame
        x_b = dis * math.cos(theta)
        y_b = dis * math.sin(theta)
        p_base = np.array([x_b, y_b, 0.0, 1.0], dtype=np.float64)
        p_goal_odom = T_odom_base @ p_base
        xg = float(p_goal_odom[0])
        yg = float(p_goal_odom[1])

        # Compute goal orientation from normal vector
        if normal_odom is not None:
            # Normal points towards unexplored area (desired facing direction)
            yawg = math.atan2(normal_odom[1], normal_odom[0])
            print(f"[Server] Using normal-based orientation: {math.degrees(yawg):.1f}°")
        else:
            # Fallback: keep current orientation
            yawg = yaw0
            print(f"[Server] Using current orientation: {math.degrees(yawg):.1f}°")

        goal_pose = build_goal_pose(xg, yg, z0, yawg)
        action_type = 'nav_step'
    
    # Update visualization state and emit to clients
    viz_state['action_type'] = action_type
    nav_state['latest_state'] = viz_state
    socketio.emit('navigation_update', viz_state)
    
    # Calculate total server time
    t_server_end = time.time()
    total_server_time = t_server_end - t_server_start
    
    timing_info = {
        'total_server_time': float(total_server_time),
        'vlm_projection_time': float(nav_timing.get('projection_time', 0.0)),
        'vlm_inference_time': float(nav_timing.get('vlm_inference_time', 0.0)),
        'detic_detection_time': float(t_detic_end - t_detic_start),
        'detic_verify_time': float(t_detic_verify),
        'detic_verify_result': detic_verify_result,
        'blocked_regions_count': int(len(nav_state.get('blocked_detections', []))),
    }
    
    # Update timing file with total server time and action type
    timing_path = os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_timing.json')
    if os.path.exists(timing_path):
        with open(timing_path, 'r') as f:
            timing_data = json.load(f)
        timing_data['total_server_time'] = float(total_server_time)
        timing_data['action_type'] = action_type
        # Keyframe info already saved above
        with open(timing_path, 'w') as f:
            json.dump(timing_data, f, indent=2)
    
    print(f"[Server] Timing - Total: {total_server_time:.3f}s, "
          f"VLM Projection: {timing_info['vlm_projection_time']:.3f}s, "
          f"VLM Inference: {timing_info['vlm_inference_time']:.3f}s")
    
    # Return goal pose for nav2 NavigateToPose
    return jsonify({
        'action_type': action_type,
        'goal_pose': {
            'frame_id': 'odom',
            'pose': goal_pose
        },
        'iteration': iteration,
        'finished': False,
        'timing': timing_info
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'goal': nav_state['goal'],
        'iteration': nav_state['iteration_count']
    })


@app.route('/detic_goals', methods=['GET'])
def detic_goals():
    """Get Detic goal pool and current active goal."""
    detector = nav_state.get('detic_detector', None)
    if detector is None:
        return jsonify({
            'initialized': False,
            'available_goals': nav_state.get('detic_goal_pool', []),
            'active_goal': None,
        })
    return jsonify({
        'initialized': True,
        'available_goals': detector.get_available_goals(),
        'active_goal': detector.goal_name,
    })


@app.route('/')
def dashboard():
    """Render visualization dashboard"""
    return render_template('dashboard.html')


@app.route('/api/list_logs', methods=['GET'])
def list_logs():
    """List all available log directories"""
    base_dir = os.path.join(script_dir, '..', 'logs')
    if not os.path.exists(base_dir):
        return jsonify({'logs': []})
    
    logs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('run_'):
            # Count iterations
            files = os.listdir(item_path)
            iterations = set()
            for f in files:
                if f.startswith('iter_') and f.endswith('_rgb.jpg'):
                    iter_num = int(f.split('_')[1])
                    iterations.add(iter_num)
            
            logs.append({
                'name': item,
                'path': item_path,
                'num_iterations': len(iterations)
            })
    
    # Sort by name (most recent first)
    logs.sort(key=lambda x: x['name'], reverse=True)
    return jsonify({'logs': logs})


@app.route('/api/get_iteration', methods=['GET'])
def get_iteration():
    """Get data for a specific iteration from a log directory"""
    log_name = request.args.get('log_name')
    iteration = request.args.get('iteration', type=int)
    
    if not log_name or iteration is None:
        return jsonify({'error': 'log_name and iteration are required'}), 400
    
    base_dir = os.path.join(script_dir, '..', 'logs')
    log_dir = os.path.join(base_dir, log_name)
    
    if not os.path.exists(log_dir):
        return jsonify({'error': 'Log directory not found'}), 404
    
    # Load metadata
    metadata_path = os.path.join(log_dir, 'metadata.json')
    goal = ''
    goal_description = ''
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                goal = metadata.get('goal', '')
                goal_description = metadata.get('goal_description', '')
        except:
            pass
    
    # Build file paths
    iter_prefix = f'iter_{iteration:04d}'
    detection_img = f'{iter_prefix}_detection.jpg'
    rgb_vis_img = f'{iter_prefix}_rgb_vis.jpg'
    bev_map_img = f'{iter_prefix}_bev_map.jpg'
    response_txt = f'{iter_prefix}_response.txt'
    
    # Check which files exist
    result = {
        'iteration': iteration,
        'log_name': log_name,
        'goal': goal,
        'goal_description': goal_description,
        'images': {},
        'llm_response': '',
        'current_cycle': [],
        'historical_keyframes': []
    }
    
    if os.path.exists(os.path.join(log_dir, detection_img)):
        result['images']['detection'] = f'/logs/{log_name}/{detection_img}'

    rgb_img = f'{iter_prefix}_rgb.jpg'
    if os.path.exists(os.path.join(log_dir, rgb_img)):
        result['images']['rgb'] = f'/logs/{log_name}/{rgb_img}'

    if os.path.exists(os.path.join(log_dir, rgb_vis_img)):
        result['images']['rgb_vis'] = f'/logs/{log_name}/{rgb_vis_img}'

    if os.path.exists(os.path.join(log_dir, bev_map_img)):
        result['images']['bev_map'] = f'/logs/{log_name}/{bev_map_img}'
    
    response_path = os.path.join(log_dir, response_txt)
    if os.path.exists(response_path):
        with open(response_path, 'r', encoding='utf-8') as f:
            result['llm_response'] = f.read()
    
    # Build history: show last 4 iterations as current cycle
    for prev_iter in range(max(1, iteration - 3), iteration):
        hist_img_path = f'/logs/{log_name}/iter_{prev_iter:04d}_rgb_vis.jpg'
        hist_img_full_path = os.path.join(log_dir, f'iter_{prev_iter:04d}_rgb_vis.jpg')
        if os.path.exists(hist_img_full_path):
            # Try to read action_type from timing file
            action_type_str = None
            timing_file = os.path.join(log_dir, f'iter_{prev_iter:04d}_timing.json')
            if os.path.exists(timing_file):
                try:
                    with open(timing_file, 'r') as f:
                        timing_data = json.load(f)
                        action_type_str = timing_data.get('action_type', None)
                except:
                    pass
            
            result['current_cycle'].append({
                'iteration': prev_iter,
                'action': action_type_str,
                'image_path': hist_img_path
            })
    
    return jsonify(result)


@app.route('/logs/<path:filepath>')
def serve_log_file(filepath):
    """Serve log files (images, etc.)"""
    base_dir = os.path.join(script_dir, '..', 'logs')
    return send_from_directory(base_dir, filepath)


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('[SocketIO] Client connected')
    # Send current state if available
    if nav_state['latest_state'] is not None:
        emit('navigation_update', nav_state['latest_state'])


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('[SocketIO] Client disconnected')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Navigation Flask Server')
    parser.add_argument('--port', type=int, default=1874, help='Server port')
    parser.add_argument('--host', type=str, default='10.19.126.158', help='Server host')
    parser.add_argument(
        '--detic-goals',
        type=str,
        default=os.getenv('DETIC_GOALS', ''),
        help='Comma-separated Detic goal classes to preload at server startup'
    )
    parser.add_argument(
        '--detic-confidence',
        type=float,
        default=0.7,
        help='Initial Detic confidence threshold used at startup'
    )
    args = parser.parse_args()

    nav_state['detic_goal_pool'] = parse_detic_goal_pool(args.detic_goals)
    nav_state['detic_init_confidence'] = float(args.detic_confidence)
    if len(nav_state['detic_goal_pool']) > 0:
        ok, msg = ensure_detic_detector(
            selected_goal=nav_state['detic_goal_pool'][0],
            confidence_threshold=float(args.detic_confidence),
            requested_goal_pool=nav_state['detic_goal_pool'],
        )
        if ok:
            print(f"[Server] Preloaded Detic goals at startup: {nav_state['detic_goal_pool']}")
        else:
            print(f"[Server] WARNING: Failed to preload Detic goal pool: {msg}")
    
    print("=" * 60)
    print("LLM Navigation Server (Flask + SocketIO)")
    print("=" * 60)
    print(f"Starting server on {args.host}:{args.port}")
    print("\nEndpoints:")
    print("  GET  /                    - Visualization Dashboard")
    print("  POST /navigation_reset    - Initialize navigation")
    print("  POST /navigation_step     - Execute navigation step")
    print("  GET  /health              - Health check")
    print("  GET  /detic_goals         - Get Detic goal pool / active goal")
    print("  GET  /api/list_logs       - List available logs")
    print("  GET  /api/get_iteration   - Get specific iteration data")
    print("=" * 60)
    print(f"\nOpen dashboard: http://{args.host}:{args.port}/")
    print("=" * 60)
    
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
