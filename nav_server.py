#!/usr/bin/env python3
"""
Flask Server for LLM Navigation
Server端负责推理（NavAgent + TargetDetector），返回Path或Action
"""
import os
import sys
import math
import json
import time
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
from target_detector import TargetDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'llm_nav_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
nav_state = {
    'agent': None,
    'detector': None,
    'goal': None,
    'goal_description': '',
    'iteration_count': 0,
    'log_dir': None,
    'latest_state': None  # Store latest visualization state
}


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
    nav_state['log_dir'] = setup_log_directory(goal, goal_description or '')
    
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
        'action_type': 'nav_step' | 'visual_servo' | 'turn_around' | 'none',
        'goal_pose': {
            'frame_id': 'odom',
            'pose': {
                'position': {'x': float, 'y': float, 'z': float},
                'orientation': {'x': float, 'y': float, 'z': float, 'w': float}
            }
        } | None,
        'detected': bool,
        'detection_score': float,
        'iteration': int,
        'finished': bool,
        'timing': {
            'total_server_time': float,
            'detection_time': float,
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
    t_detection_start = time.time()
    detection_vis_path = os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_detection.jpg')
    detection_result = nav_state['detector'].detect_from_cv_image(bgr, detection_vis_path)
    t_detection_end = time.time()
    detection_time = t_detection_end - t_detection_start
    
    detected = detection_result.get('detected', False)
    score = detection_result.get('score', 0.0)
    
    print(f"[Server] Detection: detected={detected}, score={score:.3f}, time={detection_time:.3f}s")
    
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
            if target_distance <= 5.0:
                print("[Server] Target detected within 5m -> VISUAL SERVO")
                yaw0 = mat_to_yaw(T_odom_base)
                
                dx = tx - bx
                dy = ty - by
                norm = math.hypot(dx, dy)
                if norm < 1e-9:
                    norm = 1e-9
                ux, uy = dx / norm, dy / norm
                
                d0 = 0.6
                x_new = tx - d0 * ux
                y_new = ty - d0 * uy
                yaw_new = math.atan2(ty - y_new, tx - x_new) + math.radians(15)
                
                # Build goal pose
                goal_pose = build_goal_pose(x_new, y_new, bz, yaw_new)
                
                # For visual_servo, we emit early since we don't have LLM response yet
                log_name = os.path.basename(nav_state['log_dir'])
                viz_state_early = {
                    'iteration': iteration,
                    'goal': nav_state['goal'],
                    'goal_description': nav_state['goal_description'],
                    'detected': True,
                    'detection_score': float(score),
                    'action_type': 'visual_servo',
                    'target_distance': float(target_distance),
                    'images': {},
                    'llm_response': 'Target detected within range - executing visual servo'
                }
                
                if os.path.exists(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_detection.jpg')):
                    viz_state_early['images']['detection'] = f'/logs/{log_name}/iter_{iteration:04d}_detection.jpg'
                
                nav_state['latest_state'] = viz_state_early
                socketio.emit('navigation_update', viz_state_early)
                
                # Calculate total server time
                t_server_end = time.time()
                total_server_time = t_server_end - t_server_start
                
                timing_info = {
                    'total_server_time': float(total_server_time),
                    'detection_time': float(detection_time),
                    'vlm_projection_time': 0.0,  # Visual servo doesn't use VLM
                    'vlm_inference_time': 0.0
                }
                
                # Save timing information for visual servo
                timing_path = os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_timing.json')
                timing_data = {
                    'iteration': iteration,
                    'total_server_time': float(total_server_time),
                    'detection_time': float(detection_time),
                    'vlm_projection_time': 0.0,
                    'vlm_inference_time': 0.0,
                    'action_type': 'visual_servo'
                }
                with open(timing_path, 'w') as f:
                    json.dump(timing_data, f, indent=2)
                
                print(f"[Server] Timing - Total: {total_server_time:.3f}s, Detection: {detection_time:.3f}s")
                
                # Return goal pose for nav2 NavigateToPose
                return jsonify({
                    'action_type': 'visual_servo',
                    'goal_pose': {
                        'frame_id': 'odom',
                        'pose': goal_pose
                    },
                    'detected': True,
                    'detection_score': score,
                    'target_distance': target_distance,
                    'iteration': iteration,
                    'finished': True,
                    'timing': timing_info
                })
            else:
                print(f"[Server] Target detected but too far ({target_distance:.3f}m > 1.5m) -> LLM NAV")
    
    # Step 3: LLM Navigation Step
    print("[Server] Calling NavAgent for decision...")
    response, rgb_vis, action, nav_timing = nav_state['agent']._nav(
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
    
    # Save timing information (will be updated at the end with total time and action type)
    timing_data = {
        'iteration': iteration,
        'detection_time': float(detection_time),
        'vlm_projection_time': float(nav_timing.get('projection_time', 0.0)),
        'vlm_inference_time': float(nav_timing.get('vlm_inference_time', 0.0)),
        'detected': detected,
        'detection_score': float(score)
    }
    timing_path = os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_timing.json')
    with open(timing_path, 'w') as f:
        json.dump(timing_data, f, indent=2)
    
    # Prepare visualization state for Socket.IO
    log_name = os.path.basename(nav_state['log_dir'])
    viz_state = {
        'iteration': iteration,
        'goal': nav_state['goal'],
        'goal_description': nav_state['goal_description'],
        'detected': detected,
        'detection_score': float(score),
        'action_type': None,
        'images': {},
        'llm_response': response if response else ''
    }
    
    # Add image paths
    if os.path.exists(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_detection.jpg')):
        viz_state['images']['detection'] = f'/logs/{log_name}/iter_{iteration:04d}_detection.jpg'
    if os.path.exists(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_rgb_vis.jpg')):
        viz_state['images']['rgb_vis'] = f'/logs/{log_name}/iter_{iteration:04d}_rgb_vis.jpg'
    if os.path.exists(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_explored_map.jpg')):
        viz_state['images']['explored_map'] = f'/logs/{log_name}/iter_{iteration:04d}_explored_map.jpg'
    
    if action is None:
        viz_state['action_type'] = 'none'
        nav_state['latest_state'] = viz_state
        socketio.emit('navigation_update', viz_state)
        
        # Calculate total server time
        t_server_end = time.time()
        total_server_time = t_server_end - t_server_start
        
        timing_info = {
            'total_server_time': float(total_server_time),
            'detection_time': float(detection_time),
            'vlm_projection_time': float(nav_timing.get('projection_time', 0.0)),
            'vlm_inference_time': float(nav_timing.get('vlm_inference_time', 0.0))
        }
        
        # Update timing file with total server time and action type
        timing_path = os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_timing.json')
        if os.path.exists(timing_path):
            with open(timing_path, 'r') as f:
                timing_data = json.load(f)
            timing_data['total_server_time'] = float(total_server_time)
            timing_data['action_type'] = 'none'
            with open(timing_path, 'w') as f:
                json.dump(timing_data, f, indent=2)
        
        print(f"[Server] Timing - Total: {total_server_time:.3f}s, Detection: {detection_time:.3f}s, "
              f"VLM Projection: {timing_info['vlm_projection_time']:.3f}s, "
              f"VLM Inference: {timing_info['vlm_inference_time']:.3f}s")
        
        return jsonify({
            'action_type': 'none',
            'goal_pose': None,
            'detected': detected,
            'detection_score': score,
            'iteration': iteration,
            'finished': False,
            'timing': timing_info
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
        # Turn 180 degrees at current position
        yaw_goal = yaw0 + math.pi
        goal_pose = build_goal_pose(x0, y0, z0, yaw_goal)
        action_type = 'turn_around'

    else:
        # Normal navigation step
        print(f"[Server] Action is NAV STEP: r={r:.2f}, theta={theta:.2f}")
        
        dis = min(0.4, r)
        # Compute goal in odom frame
        x_b = dis * math.cos(theta)
        y_b = dis * math.sin(theta)
        p_base = np.array([x_b, y_b, 0.0, 1.0], dtype=np.float64)
        p_goal_odom = T_odom_base @ p_base
        xg = float(p_goal_odom[0])
        yg = float(p_goal_odom[1])
        yawg = yaw0 + float(theta)
        
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
        'detection_time': float(detection_time),
        'vlm_projection_time': float(nav_timing.get('projection_time', 0.0)),
        'vlm_inference_time': float(nav_timing.get('vlm_inference_time', 0.0))
    }
    
    # Update timing file with total server time and action type
    timing_path = os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_timing.json')
    if os.path.exists(timing_path):
        with open(timing_path, 'r') as f:
            timing_data = json.load(f)
        timing_data['total_server_time'] = float(total_server_time)
        timing_data['action_type'] = action_type
        with open(timing_path, 'w') as f:
            json.dump(timing_data, f, indent=2)
    
    print(f"[Server] Timing - Total: {total_server_time:.3f}s, Detection: {detection_time:.3f}s, "
          f"VLM Projection: {timing_info['vlm_projection_time']:.3f}s, "
          f"VLM Inference: {timing_info['vlm_inference_time']:.3f}s")
    
    # Return goal pose for nav2 NavigateToPose
    return jsonify({
        'action_type': action_type,
        'goal_pose': {
            'frame_id': 'odom',
            'pose': goal_pose
        },
        'detected': detected,
        'detection_score': score,
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
    explored_map_img = f'{iter_prefix}_explored_map.jpg'
    response_txt = f'{iter_prefix}_response.txt'
    
    # Check which files exist
    result = {
        'iteration': iteration,
        'log_name': log_name,
        'goal': goal,
        'goal_description': goal_description,
        'images': {},
        'llm_response': ''
    }
    
    if os.path.exists(os.path.join(log_dir, detection_img)):
        result['images']['detection'] = f'/logs/{log_name}/{detection_img}'
    
    if os.path.exists(os.path.join(log_dir, rgb_vis_img)):
        result['images']['rgb_vis'] = f'/logs/{log_name}/{rgb_vis_img}'
    
    if os.path.exists(os.path.join(log_dir, explored_map_img)):
        result['images']['explored_map'] = f'/logs/{log_name}/{explored_map_img}'
    
    response_path = os.path.join(log_dir, response_txt)
    if os.path.exists(response_path):
        with open(response_path, 'r', encoding='utf-8') as f:
            result['llm_response'] = f.read()
    
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
    args = parser.parse_args()
    
    print("=" * 60)
    print("LLM Navigation Server (Flask + SocketIO)")
    print("=" * 60)
    print(f"Starting server on {args.host}:{args.port}")
    print("\nEndpoints:")
    print("  GET  /                    - Visualization Dashboard")
    print("  POST /navigation_reset    - Initialize navigation")
    print("  POST /navigation_step     - Execute navigation step")
    print("  GET  /health              - Health check")
    print("  GET  /api/list_logs       - List available logs")
    print("  GET  /api/get_iteration   - Get specific iteration data")
    print("=" * 60)
    print(f"\n🌐 Open dashboard: http://{args.host}:{args.port}/")
    print("=" * 60)
    
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)

