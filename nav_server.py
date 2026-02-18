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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'llm_nav_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
nav_state = {
    'agent': None,
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
        "confidence_threshold": float,  # optional, default 0.5
        "keyframe_mode": bool,  # optional, enable keyframe mode
        "keyframe_angle_range": float  # optional, angle range for non-keyframes (degrees)
    }
    """
    global nav_state
    
    data = request.get_json()
    goal = data.get('goal')
    goal_description = data.get('goal_description', '')
    confidence_threshold = data.get('confidence_threshold', 0.5)
    keyframe_mode = data.get('keyframe_mode', False)
    keyframe_angle_range = data.get('keyframe_angle_range', 25.0)
    
    if not goal:
        return jsonify({'error': 'goal is required'}), 400
    
    # Initialize components
    nav_state['goal'] = goal
    nav_state['goal_description'] = goal_description or ''
    nav_state['iteration_count'] = 0
    nav_state['log_dir'] = setup_log_directory(goal, goal_description or '')
    
    # Build agent config with keyframe mode settings
    agent_cfg = {
        'keyframe_mode': keyframe_mode,
        'keyframe_angle_range': keyframe_angle_range,

        'vlm_model': '/data/sea_disk0/liujy/models/Qwen/Qwen3-VL-8B-Instruct/',
        'vlm_api_key': 'EMPTY',
        'vlm_base_url': 'http://10.15.89.71:34134/v1/',
    }
    
    # Initialize NavAgent (or reset if already exists)
    if nav_state['agent'] is None:
        nav_state['agent'] = NavAgent(cfg=agent_cfg)
        nav_state['agent'].reset(goal=goal, goal_description=goal_description or '')
    else:
        # Update agent config and reset with new goal information
        nav_state['agent'].cfg['keyframe_mode'] = keyframe_mode
        nav_state['agent'].cfg['keyframe_angle_range'] = keyframe_angle_range
        nav_state['agent'].reset(goal=goal, goal_description=goal_description or '')
    
    print(f"[Server] Navigation reset: goal='{goal}', description='{goal_description}', log_dir={nav_state['log_dir']}")
    print(f"[Server] Keyframe mode: {keyframe_mode}, angle_range: {keyframe_angle_range}°")
    print(f"[Server] NavAgent reset with goal information, VLM initialized with full task briefing")
    
    return jsonify({
        'status': 'success',
        'goal': goal,
        'goal_description': goal_description,
        'log_dir': nav_state['log_dir'],
        'keyframe_mode': keyframe_mode,
        'keyframe_angle_range': keyframe_angle_range
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
    
    # Save prompt for debugging (if available in timing info)
    if nav_timing.get('prompt'):
        with open(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_prompt.txt'), 'w', encoding='utf-8') as f:
            f.write(nav_timing['prompt'])
    
    # Save voxel maps
    voxel_map = nav_state['agent'].voxel_map
    explored_map = nav_state['agent'].explored_map
    if voxel_map is not None:
        cv2.imwrite(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_voxel_map.jpg'), voxel_map)
    if explored_map is not None:
        cv2.imwrite(os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_explored_map.jpg'), explored_map)
    
    # Get keyframe history summary
    history_summary = nav_state['agent'].get_history_summary()
    
    # Save timing information (will be updated at the end with total time and action type)
    timing_data = {
        'iteration': iteration,
        'vlm_projection_time': float(nav_timing.get('projection_time', 0.0)),
        'vlm_inference_time': float(nav_timing.get('vlm_inference_time', 0.0)),
        'is_keyframe': history_summary.get('keyframe_mode', False) and getattr(nav_state['agent'], 'is_keyframe', True),
        'keyframe_mode_enabled': history_summary.get('keyframe_mode', False)
    }
    timing_path = os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_timing.json')
    with open(timing_path, 'w') as f:
        json.dump(timing_data, f, indent=2)
    
    # Save keyframe history summary
    if history_summary.get('keyframe_mode', False):
        history_path = os.path.join(nav_state['log_dir'], f'iter_{iteration:04d}_keyframe_history.json')
        with open(history_path, 'w') as f:
            json.dump(history_summary, f, indent=2)
    
    # Prepare visualization state for Socket.IO
    log_name = os.path.basename(nav_state['log_dir'])
    
    # Build history split into current cycle and historical keyframes
    current_cycle = []
    historical_keyframes = []
    
    if history_summary.get('keyframe_mode', False):
        full_history = history_summary.get('full_history', [])
        
        # Process all keyframes except the last one (which is current cycle)
        for kf in full_history[:-1]:
            kf_iter = kf['iter']
            hist_img_path = f'/logs/{log_name}/iter_{kf_iter:04d}_rgb_vis.jpg'
            hist_img_full_path = os.path.join(nav_state['log_dir'], f'iter_{kf_iter:04d}_rgb_vis.jpg')
            if os.path.exists(hist_img_full_path):
                historical_keyframes.append({
                    'iteration': kf_iter,
                    'action': kf['action_number'],
                    'image_path': hist_img_path,
                    'is_keyframe': True
                })
        
        # Process current cycle (last keyframe + its non-keyframes)
        if len(full_history) > 0:
            last_kf = full_history[-1]
            kf_iter = last_kf['iter']
            hist_img_path = f'/logs/{log_name}/iter_{kf_iter:04d}_rgb_vis.jpg'
            hist_img_full_path = os.path.join(nav_state['log_dir'], f'iter_{kf_iter:04d}_rgb_vis.jpg')
            if os.path.exists(hist_img_full_path):
                current_cycle.append({
                    'iteration': kf_iter,
                    'action': last_kf['action_number'],
                    'image_path': hist_img_path,
                    'is_keyframe': True
                })
            
            # Add non-keyframes of current cycle
            for nkf in last_kf.get('non_keyframes', []):
                nkf_iter = nkf['iter']
                hist_img_path = f'/logs/{log_name}/iter_{nkf_iter:04d}_rgb_vis.jpg'
                hist_img_full_path = os.path.join(nav_state['log_dir'], f'iter_{nkf_iter:04d}_rgb_vis.jpg')
                if os.path.exists(hist_img_full_path):
                    current_cycle.append({
                        'iteration': nkf_iter,
                        'action': nkf['action_number'],
                        'image_path': hist_img_path,
                        'is_keyframe': False
                    })
    else:
        # Fallback: non-keyframe mode, show last 4 iterations as current cycle
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
                            timing_data = json.load(f)
                            action_type_str = timing_data.get('action_type', None)
                    except:
                        pass
                
                current_cycle.append({
                    'iteration': prev_iter,
                    'action': action_type_str,
                    'image_path': hist_img_path,
                    'is_keyframe': False
                })
    
    viz_state = {
        'iteration': iteration,
        'goal': nav_state['goal'],
        'goal_description': nav_state['goal_description'],
        'action_type': None,
        'images': {},
        'llm_response': response if response else '',
        'current_cycle': current_cycle,
        'historical_keyframes': historical_keyframes
    }
    
    # Add image paths
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
        # Forward movement action: (r, theta)
        r, theta = float(action[0]), float(action[1])
        print(f"[Server] Action is NAV STEP: r={r:.2f}, theta={theta:.2f}")
        
        dis = min(0.4, r)
        # Compute goal in odom frame
        x_b = dis * math.cos(theta)
        y_b = dis * math.sin(theta)
        p_base = np.array([x_b, y_b, 0.0, 1.0], dtype=np.float64)
        p_goal_odom = T_odom_base @ p_base
        xg = float(p_goal_odom[0])
        yg = float(p_goal_odom[1])
        # Keep current orientation for forward movement (no rotation)
        yawg = yaw0
        
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
        'vlm_inference_time': float(nav_timing.get('vlm_inference_time', 0.0))
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


@app.route('/check_stop', methods=['POST'])
def check_stop():
    """
    轻量级stop检测 - 只判断是否到达目标
    供stop detection线程调用，2Hz频率
    
    Request:
        - Files: 'rgb' (JPEG)
    
    Returns JSON: {
        'should_stop': bool,
        'raw_response': str
    }
    """
    global nav_state
    
    if nav_state['goal'] is None or nav_state['agent'] is None:
        return jsonify({'error': 'Navigation not initialized. Call /navigation_reset first.'}), 400
    
    try:
        # 加载RGB
        rgb_file = request.files['rgb']
        rgb_pil = Image.open(rgb_file.stream).convert('RGB')
        rgb = np.asarray(rgb_pil)
    except Exception as e:
        return jsonify({'error': f'Failed to load image: {str(e)}'}), 400
    
    goal = nav_state['goal']
    goal_desc = nav_state['goal_description']
    
    prompt = f"""
    You are a robot navigation stop classifier.

    Goal: {goal}

    Answer 'yes' ONLY when you are confident we should stop now.

    There are two valid stop situations:

    1) Direct stop (APPROACHABLE goal):
    - The goal object is clearly visible (not tiny/far/ambiguous), AND
    - It appears near (If you think you are less than 1 meter to it.)

    2) Furniture stop (SURFACE goal, e.g., on a table/shelf/counter or inside cabinet):
    - The goal object is clearly visible, AND
    - You can see the supporting furniture (tabletop / shelf / counter / cabinet) indicating the robot cannot drive right up to the object, AND
    - You look close enough to the furniture edge (near the table/counter), even if the goal is not centered.

    If unsure, if the goal is far, answer 'no'.
    Answer ONLY 'yes' or 'no'.
    """
    
    try:
        action_vlm = nav_state['agent'].actionVLM
        
        # call_chat接口: call_chat(history: int, images: list[np.array], text_prompt: str)
        # history=0表示不使用历史对话
        response = action_vlm.call_chat(
            history=0,  # 不使用历史
            images=[rgb],  # 单张图片的列表
            text_prompt=prompt
        )
        
        # 解析yes/no
        response_lower = response.lower().strip()
        # 判断逻辑：包含yes且不包含no（避免"no, not yet"这种情况）
        should_stop = 'yes' in response_lower and 'no' not in response_lower
        
        print(f"[StopCheck] Goal: {goal}, VLM response: '{response}' -> should_stop={should_stop}")
        
        return jsonify({
            'should_stop': should_stop,
            'raw_response': response
        })
    
    except Exception as e:
        print(f"[StopCheck] VLM call failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'should_stop': False,
            'error': str(e)
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
        'llm_response': '',
        'current_cycle': [],
        'historical_keyframes': []
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
    
    # Load keyframe history if available
    history_path = os.path.join(log_dir, f'{iter_prefix}_keyframe_history.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history_summary = json.load(f)
                full_history = history_summary.get('full_history', [])
                
                # Process all keyframes except the last one (which is current cycle)
                for kf in full_history[:-1]:
                    kf_iter = kf['iter']
                    hist_img_path = f'/logs/{log_name}/iter_{kf_iter:04d}_rgb_vis.jpg'
                    hist_img_full_path = os.path.join(log_dir, f'iter_{kf_iter:04d}_rgb_vis.jpg')
                    if os.path.exists(hist_img_full_path):
                        result['historical_keyframes'].append({
                            'iteration': kf_iter,
                            'action': kf['action_number'],
                            'image_path': hist_img_path,
                            'is_keyframe': True
                        })
                
                # Process current cycle (last keyframe + its non-keyframes)
                if len(full_history) > 0:
                    last_kf = full_history[-1]
                    kf_iter = last_kf['iter']
                    hist_img_path = f'/logs/{log_name}/iter_{kf_iter:04d}_rgb_vis.jpg'
                    hist_img_full_path = os.path.join(log_dir, f'iter_{kf_iter:04d}_rgb_vis.jpg')
                    if os.path.exists(hist_img_full_path):
                        result['current_cycle'].append({
                            'iteration': kf_iter,
                            'action': last_kf['action_number'],
                            'image_path': hist_img_path,
                            'is_keyframe': True
                        })
                    
                    # Add non-keyframes of current cycle
                    for nkf in last_kf.get('non_keyframes', []):
                        nkf_iter = nkf['iter']
                        hist_img_path = f'/logs/{log_name}/iter_{nkf_iter:04d}_rgb_vis.jpg'
                        hist_img_full_path = os.path.join(log_dir, f'iter_{nkf_iter:04d}_rgb_vis.jpg')
                        if os.path.exists(hist_img_full_path):
                            result['current_cycle'].append({
                                'iteration': nkf_iter,
                                'action': nkf['action_number'],
                                'image_path': hist_img_path,
                                'is_keyframe': False
                            })
        except Exception as e:
            print(f"Error loading keyframe history: {e}")
    else:
        # Fallback: non-keyframe mode, show last 4 iterations as current cycle
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
                    'image_path': hist_img_path,
                    'is_keyframe': False
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
    print(f"\nOpen dashboard: http://{args.host}:{args.port}/")
    print("=" * 60)
    
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)

