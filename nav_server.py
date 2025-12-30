#!/usr/bin/env python3
"""
Flask Server for LLM Navigation
Server端负责推理（NavAgent + TargetDetector），返回Path或Action
"""
import os
import sys
import math
import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from nav_agent import NavAgent
from target_detector import TargetDetector

app = Flask(__name__)

# Global state
nav_state = {
    'agent': None,
    'detector': None,
    'goal': None,
    'goal_description': '',
    'iteration_count': 0,
    'log_dir': None
}


def setup_log_directory():
    """Create log directory for this session"""
    base_dir = os.path.join(script_dir, '..', 'logs')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
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


def build_path_poses(x0: float, y0: float, z0: float, yaw0: float,
                     xg: float, yg: float, yawg: float,
                     num_poses: int = 20) -> List[Dict[str, Any]]:
    """
    Build interpolated poses from start to goal
    Returns list of pose dicts (not ROS Path, just data)
    """
    poses = []
    for i in range(1, num_poses + 1):
        s = i / float(num_poses)
        xi = x0 + s * (xg - x0)
        yi = y0 + s * (yg - y0)
        zi = z0  # Keep same z
        yawi = yaw0 + s * (yawg - yaw0)
        
        poses.append({
            'position': {'x': float(xi), 'y': float(yi), 'z': float(zi)},
            'orientation': yaw_to_quaternion(float(yawi))
        })
    return poses


def build_turnaround_poses(x: float, y: float, z: float, yaw0: float,
                          num_poses: int = 20) -> List[Dict[str, Any]]:
    """Build poses for 180-degree turn around"""
    yawg = yaw0 + math.pi
    poses = []
    for i in range(1, num_poses + 1):
        s = i / float(num_poses)
        yawi = yaw0 + s * (yawg - yaw0)
        
        poses.append({
            'position': {'x': float(x), 'y': float(y), 'z': float(z)},
            'orientation': yaw_to_quaternion(float(yawi))
        })
    return poses


@app.route('/navigation_reset', methods=['POST'])
def navigation_reset():
    """
    Initialize navigation session
    Request JSON: {
        "goal": str,  # navigation goal (e.g., "door", "chair")
        "goal_description": str,  # optional description of goal location
        "confidence_threshold": float  # optional, default 0.5
    }
    """
    global nav_state
    
    data = request.get_json()
    goal = data.get('goal')
    goal_description = data.get('goal_description', '')
    confidence_threshold = data.get('confidence_threshold', 0.5)
    
    if not goal:
        return jsonify({'error': 'goal is required'}), 400
    
    # Initialize components
    nav_state['goal'] = goal
    nav_state['goal_description'] = goal_description or ''
    nav_state['iteration_count'] = 0
    nav_state['log_dir'] = setup_log_directory()
    
    # Initialize NavAgent
    if nav_state['agent'] is None:
        nav_state['agent'] = NavAgent()
    
    # Initialize TargetDetector
    nav_state['detector'] = TargetDetector(
        target_name=goal,
        confidence_threshold=confidence_threshold
    )
    
    print(f"[Server] Navigation reset: goal='{goal}', description='{goal_description}', log_dir={nav_state['log_dir']}")
    
    return jsonify({
        'status': 'success',
        'goal': goal,
        'goal_description': goal_description,
        'log_dir': nav_state['log_dir']
    })


@app.route('/navigation_step', methods=['POST'])
def navigation_step():
    """
    Single navigation step with detection and planning
    
    Request:
        - Files: 'rgb' (JPEG/PNG), 'depth' (PNG 16-bit or NPY)
        - Form data:
            'intrinsic': JSON string, 3x3 matrix flattened
            'T_cam_odom': JSON string, 4x4 matrix flattened (odom->cam)
            'T_odom_base': JSON string, 4x4 matrix flattened (base->odom)
    
    Returns JSON: {
        'action_type': 'nav_step' | 'visual_servo' | 'turn_around' | 'finished',
        'poses': [...],  # List of 20 poses in odom frame
        'frame_id': 'odom',
        'detected': bool,
        'detection_score': float,
        'iteration': int
    }
    """
    global nav_state
    
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
    
    # Step 1: Target Detection
    print(f"[Server] Running detection for '{nav_state['goal']}'...")
    detection_vis_path = os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_detection.jpg')
    detection_result = nav_state['detector'].detect_from_cv_image(bgr, detection_vis_path)
    
    detected = detection_result.get('detected', False)
    score = detection_result.get('score', 0.0)
    
    print(f"[Server] Detection: detected={detected}, score={score:.3f}")
    
    # Step 2: Visual Servo if detected with high confidence and within 1.5m
    target_distance = None
    if detected and score > 0.5:
        # Compute target position in odom frame to check distance
        depth_val = obs['depth']
        K = obs['intrinsic']
        T_cam_odom = obs['extrinsic']
        T_odom_base = obs['base_to_odom_matrix']
        
        u = int(round(detection_result['center_x']))
        v = int(round(detection_result['center_y']))
        h, w = depth_val.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        
        d = float(depth_val[v, u])
        
        if d > 1e-6:
            # Unproject to camera frame
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            x_c = (u - cx) / fx * d
            y_c = (v - cy) / fy * d
            z_c = d
            p_cam = np.array([x_c, y_c, z_c, 1.0], dtype=np.float64)
            
            # Transform to odom frame
            T_odom_cam = np.linalg.inv(T_cam_odom.astype(np.float64))
            p_odom = T_odom_cam @ p_cam
            tx, ty = float(p_odom[0]), float(p_odom[1])
            
            # Current base position
            bx, by, bz = float(T_odom_base[0, 3]), float(T_odom_base[1, 3]), float(T_odom_base[2, 3])
            
            # Calculate distance in xy plane
            target_distance = math.hypot(tx - bx, ty - by)
            print(f"[Server] Target distance: {target_distance:.3f}m")
            
            # Only execute visual servo if target is within 1.5m
            if target_distance <= 1.5:
                print("[Server] Target detected within 1.5m -> VISUAL SERVO")
                yaw0 = mat_to_yaw(T_odom_base)
                
                # Compute goal pose (0.55m away from target)
                dx = tx - bx
                dy = ty - by
                norm = math.hypot(dx, dy)
                if norm < 1e-9:
                    norm = 1e-9
                ux, uy = dx / norm, dy / norm
                
                d0 = 0.55
                x_new = tx - d0 * ux
                y_new = ty - d0 * uy
                yaw_new = math.atan2(ty - y_new, tx - x_new) + math.radians(15)
                
                # Build path poses
                poses = build_path_poses(bx, by, bz, yaw0, x_new, y_new, yaw_new, num_poses=20)
                
                # Return complete path data (all processing done on server)
                return jsonify({
                    'action_type': 'visual_servo',
                    'path': {
                        'frame_id': 'odom',
                        'poses': poses
                    },
                    'detected': True,
                    'detection_score': score,
                    'target_distance': target_distance,
                    'iteration': iteration,
                    'finished': True
                })
            else:
                print(f"[Server] Target detected but too far ({target_distance:.3f}m > 1.5m) -> LLM NAV")
    
    # Step 3: LLM Navigation Step
    print("[Server] Calling NavAgent for decision...")
    response, rgb_vis, action = nav_state['agent']._nav(
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
    
    # Save voxel maps
    voxel_map = nav_state['agent'].voxel_map
    explored_map = nav_state['agent'].explored_map
    if voxel_map is not None:
        cv2.imwrite(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_voxel_map.jpg'), voxel_map)
    if explored_map is not None:
        cv2.imwrite(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_explored_map.jpg'), explored_map)
    
    if action is None:
        return jsonify({
            'action_type': 'none',
            'path': {
                'frame_id': 'odom',
                'poses': []
            },
            'detected': detected,
            'detection_score': score,
            'iteration': iteration,
            'finished': False
        })
    
    r, theta = float(action[0]), float(action[1])
    
    # Current base state
    T_odom_base = obs['base_to_odom_matrix'].astype(np.float64)
    x0 = float(T_odom_base[0, 3])
    y0 = float(T_odom_base[1, 3])
    z0 = float(T_odom_base[2, 3])
    yaw0 = mat_to_yaw(T_odom_base)
    
    # Action = 0 -> Turn around
    if abs(r) < 1e-9 and abs(theta) < 1e-9:
        print("[Server] Action is TURN AROUND")
        poses = build_turnaround_poses(x0, y0, z0, yaw0, num_poses=20)
        action_type = 'turn_around'
    else:
        # Normal navigation step
        print(f"[Server] Action is NAV STEP: r={r:.2f}, theta={theta:.2f}")
        
        # Compute goal in odom frame
        x_b = r * math.cos(theta)
        y_b = r * math.sin(theta)
        p_base = np.array([x_b, y_b, 0.0, 1.0], dtype=np.float64)
        p_goal_odom = T_odom_base @ p_base
        xg = float(p_goal_odom[0])
        yg = float(p_goal_odom[1])
        yawg = yaw0 + float(theta)
        
        poses = build_path_poses(x0, y0, z0, yaw0, xg, yg, yawg, num_poses=20)
        action_type = 'nav_step'
    
    # Return complete path data (all processing done on server)
    return jsonify({
        'action_type': action_type,
        'path': {
            'frame_id': 'odom',
            'poses': poses
        },
        'detected': detected,
        'detection_score': score,
        'iteration': iteration,
        'finished': False
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'goal': nav_state['goal'],
        'iteration': nav_state['iteration_count']
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Navigation Flask Server')
    parser.add_argument('--port', type=int, default=1874, help='Server port')
    parser.add_argument('--host', type=str, default='10.19.126.158', help='Server host')
    args = parser.parse_args()
    
    print("=" * 60)
    print("LLM Navigation Server (Flask)")
    print("=" * 60)
    print(f"Starting server on {args.host}:{args.port}")
    print("\nEndpoints:")
    print("  POST /navigation_reset - Initialize navigation")
    print("  POST /navigation_step  - Execute navigation step")
    print("  GET  /health          - Health check")
    print("=" * 60)
    
    app.run(host=args.host, port=args.port, debug=False)

