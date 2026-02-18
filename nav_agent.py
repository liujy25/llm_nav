#!/usr/bin/env python3
import math
import time
import numpy as np
import cv2
import ast
from collections import deque

from vlm import OpenAIVLM
from utils import (
    GREEN, RED, BLACK, WHITE, GREY,
    point_to_pixel, agent_frame_to_image_coords,
    unproject_2d, local_to_global_matrix,
    find_intersections, depth_to_height, put_text_on_image
)


class NavAgent:
    # input obs format: rgb
    explored_color = GREY
    unexplored_color = GREEN

    map_size = 5000
    explore_threshold = 3
    voxel_ray_size = 60
    e_i_scaling = 0.8

    def __init__(self, cfg=None):
        self.cfg = cfg or {}

        # defaults
        self.cfg.setdefault('num_theta', 40)
        self.cfg.setdefault('image_edge_threshold', 0.04)
        self.cfg.setdefault('clip_dist', 2.0)
        self.cfg.setdefault('turn_angle_deg', 30.0)  # Turn left/right angle in degrees
        
        # Navigability configuration (obstacle height range)
        self.cfg.setdefault('obstacle_height_min', 0.15)  # Minimum obstacle height (m)
        self.cfg.setdefault('obstacle_height_max', 2.0)   # Maximum obstacle height (m)
        
        # Prompt mode: 'reasoning' or 'action_only'
        self.cfg.setdefault('prompt_mode', 'reasoning')  # 'reasoning' or 'action_only'
        
        # Keyframe mode configuration
        self.cfg.setdefault('keyframe_mode', False)  # Enable keyframe-based navigation
        self.cfg.setdefault('keyframe_angle_range', 25.0)  # Angle range for non-keyframes (degrees)
        
        # VLM configuration defaults
        self.cfg.setdefault('vlm_model', '/data/sea_disk0/liujy/models/Qwen/Qwen3-VL-8B-Instruct/')
        self.cfg.setdefault('vlm_api_key', 'EMPTY')
        self.cfg.setdefault('vlm_base_url', 'http://10.15.89.71:34134/v1/')
        self.cfg.setdefault('vlm_timeout', 10)

        self.turn_angle_deg = float(self.cfg['turn_angle_deg'])
        self.clip_dist = self.cfg['clip_dist']
        self.prompt_mode = self.cfg['prompt_mode']
        
        # Initialize goal attributes (will be set properly in reset())
        self.goal = None
        self.goal_description = ''
        
        self._initialize_vlms()
        self.reset()

    def _initialize_vlms(self):
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You are only allowed to search for the goal object in the room you are in now. You cannot go to other rooms."
            "You cannot move through doors. "
        )
        self.actionVLM = OpenAIVLM(
            model=self.cfg['vlm_model'],
            system_instruction=system_instruction,
            api_key=self.cfg['vlm_api_key'],
            base_url=self.cfg['vlm_base_url'],
            timeout=self.cfg['vlm_timeout']
        )

    def _construct_keyframe_prompt(self, num_actions: int, iter: int):
        """
        构建关键帧prompt：5个问题 + 统一输出格式
        
        关键帧是战略决策点，需要VLM进行深度推理：
        1. 回顾历史探索
        2. 分析当前观测
        3. 制定战略方向
        4. 选择最佳动作
        5. 解释决策理由
        
        Parameters
        ----------
        num_actions : int
            当前可用的MOVE动作数量
        iter : int
            当前迭代编号
            
        Returns
        -------
        str
            关键帧prompt文本
        """
        num_keyframes = len(self.keyframe_history)
        
        prompt = f"""--- Iteration {iter} (KEYFRAME #{num_keyframes + 1}) ---

CRITICAL DECISION POINT: This is a keyframe where you need to make a strategic decision.

Current observation: {num_actions} available MOVE waypoints shown (numbered 1..{num_actions}).
Additionally, there are always two TURN actions:
- Action -1: TURN LEFT by {self.turn_angle_deg:.0f} degrees in place
- Action -2: TURN RIGHT by {self.turn_angle_deg:.0f} degrees in place

================================================================================

Please think step by step and answer the following questions. 
In your response, you should: first, give your answer of all the questions, 
then provide your final decision in this exact format: {{'action': <action_number>}}.

Question 1: Historical Review
Review the {num_keyframes} previous keyframes in your conversation history:
- What areas have you explored at each keyframe?
- What strategic decisions did you make?
- Are there any unexplored areas you should prioritize?
Provide a brief summary of your exploration history and strategy.

Question 2: Current Observation
What do you see in the current image?
- Navigable areas (floor, corridors, open spaces)
- Obstacles (walls, furniture, objects)
- Potential target objects related to "{self.goal}"
- Unexplored directions

Question 3: Strategic Reasoning
Based on your historical review (Q1) and current observation (Q2):
- Should you continue your previous strategy or change direction?
- Which areas are most promising for finding "{self.goal}"?
- How does this decision fit into your overall exploration plan?

Question 4: Action Selection
Which action number achieves your strategic goal best?
Consider all available actions (1 to {num_actions}, -1, -2).

Question 5: Action Justification
Why did you choose this specific action?
- How does it align with your strategy from Q3?
- What do you expect to discover or achieve?
"""
        
        return prompt

    def _construct_nonkeyframe_prompt(self, num_actions: int, iter: int):
        """
        构建非关键帧prompt：回顾提示 + 直接输出action
        
        非关键帧是执行模式，继续执行父关键帧的战略决策。
        VLM只需要快速选择动作，无需深度推理。
        
        Parameters
        ----------
        num_actions : int
            当前可用的MOVE动作数量
        iter : int
            当前迭代编号
            
        Returns
        -------
        str
            非关键帧prompt文本
        """
        num_nonkeyframes = len(self.current_cycle_nonkeyframes)
        parent_kf_iter = self.keyframe_history[self.current_keyframe_idx]['iter'] if self.current_keyframe_idx >= 0 else 0
        
        prompt = f"""--- Iteration {iter} (NON-KEYFRAME, step {num_nonkeyframes + 1} in current cycle) ---

EXECUTION MODE: Continue following the strategy from keyframe #{self.current_keyframe_idx + 1} (iter {parent_kf_iter}).

Current observation: {num_actions} available MOVE waypoints shown (numbered 1..{num_actions}).
Additionally, there are always two TURN actions:
- Action -1: TURN LEFT by {self.turn_angle_deg:.0f} degrees in place
- Action -2: TURN RIGHT by {self.turn_angle_deg:.0f} degrees in place

================================================================================

CONTEXT REVIEW:
Review your conversation history (parent keyframe + {num_nonkeyframes} non-keyframes):
- What was the strategic decision at the parent keyframe?
- What actions have you taken since then?
- Are you still on track with the plan?

Based on the parent keyframe's strategy and your current observation, select the best action to continue the plan.

================================================================================

Provide your action selection in this exact format:
{{'action': <action_number>}}
"""
        
        return prompt

    def _encode_image_to_base64(self, image):
        """
        将numpy图片编码为base64 URL格式
        
        Parameters
        ----------
        image : np.ndarray
            RGB图片数组
            
        Returns
        -------
        str
            base64编码的图片URL
        """
        import base64
        from io import BytesIO
        from PIL import Image
        
        # 转换numpy数组为PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 编码为base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"

    def _build_vlm_history_for_keyframe(self):
        """
        为关键帧构建VLM历史：所有历史关键帧的对话
        
        关键帧需要看到所有历史关键帧的决策，以便：
        1. 避免重复探索已访问区域
        2. 保持长期探索策略的连贯性
        3. 基于历史经验做出更好的战略决策
        
        Returns
        -------
        list
            VLM消息历史，格式为[{"role": "user", "content": [...]}, {"role": "assistant", "content": "..."}, ...]
        """
        history = []
        
        for kf in self.keyframe_history:
            # User message（关键帧的标注图片 + prompt）
            history.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._encode_image_to_base64(kf['rgb_vis'])
                        }
                    },
                    {
                        "type": "text",
                        "text": kf['prompt']
                    }
                ]
            })
            
            # Assistant response
            history.append({
                "role": "assistant",
                "content": kf['response']
            })
        
        return history

    def _build_vlm_history_for_nonkeyframe(self):
        """
        为非关键帧构建VLM历史：当前关键帧 + 该周期内的所有非关键帧
        
        非关键帧只需要看到：
        1. 父关键帧的战略决策（作为执行依据）
        2. 该周期内已执行的非关键帧（了解执行进度）
        
        这样可以在保持上下文连贯性的同时，避免token消耗过大。
        
        Returns
        -------
        list
            VLM消息历史
        """
        history = []
        
        # 1. 当前所属的关键帧
        if self.current_keyframe_idx >= 0:
            kf = self.keyframe_history[self.current_keyframe_idx]
            
            history.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._encode_image_to_base64(kf['rgb_vis'])
                        }
                    },
                    {
                        "type": "text",
                        "text": kf['prompt']
                    }
                ]
            })
            
            history.append({
                "role": "assistant",
                "content": kf['response']
            })
        
        # 2. 该周期内的所有非关键帧
        for nkf in self.current_cycle_nonkeyframes:
            history.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._encode_image_to_base64(nkf['rgb_vis'])
                        }
                    },
                    {
                        "type": "text",
                        "text": nkf['prompt']
                    }
                ]
            })
            
            history.append({
                "role": "assistant",
                "content": nkf['response']
            })
        
        return history

    def step(self, obs: dict):
        if self.step_ndx == 0:
            self.init_pos = obs['base_to_odom_matrix'][:3, 3]

        agent_action, metadata = self._choose_action(obs)
        metadata['step_metadata'].update(self.cfg)

        self.step_ndx += 1
        return agent_action, metadata

    def reset(self, goal: str = None, goal_description: str = ''):
        """
        Reset the agent state and initialize VLM with goal information.
        
        Parameters
        ----------
        goal : str, optional
            Navigation goal (e.g., "chair", "door")
        goal_description : str, optional
            Additional description about the goal location
        """
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.scale = 100
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        
        # Store goal information as instance attributes
        self.goal = goal
        self.goal_description = goal_description
        
        # Keyframe mode state
        self.current_waypoint_P_global = None  # (x, y, z) in odom frame
        self.is_keyframe = True  # First frame is always keyframe
        
        # Memory管理：分层历史结构
        self.keyframe_history = []  # 所有关键帧记录（用于构建关键帧VLM历史）
        self.current_cycle_nonkeyframes = []  # 当前关键帧周期内的非关键帧
        self.current_keyframe_idx = -1  # 当前所属的关键帧索引
        
        # Build initial prompt with goal information if provided
        if goal:
            initial_prompt = self._build_initial_prompt(goal, goal_description)
            self.actionVLM.reset(initial_prompt=initial_prompt)
            print(f"[NavAgent] Reset with goal='{goal}', VLM initialized with full task briefing")
        else:
            self.actionVLM.reset()
            print("[NavAgent] Reset without goal, VLM history cleared")

    def _build_initial_prompt(self, goal: str, goal_description: str = ''):
        """
        Build comprehensive initial prompt with full task briefing.
        This is sent once during reset and stored in VLM's history.
        
        Parameters
        ----------
        goal : str
            Navigation goal (e.g., "chair", "door")
        goal_description : str, optional
            Additional description about the goal location
            
        Returns
        -------
        str
            Full task briefing prompt
        """
        description_text = ""
        if goal_description and goal_description.strip():
            description_text = f"\nADDITIONAL INFORMATION: {goal_description.strip()}\n"
        
        initial_prompt = f"""=== NAVIGATION TASK BRIEFING ===

OBJECTIVE: Navigate to the nearest {goal.upper()} and get as close to it as possible.

ROBOT CAPABILITIES:
- You have an RGB camera for observation
- Omnidirectional (omni-wheel) base with small turning radius
- Can rotate in place efficiently

NAVIGATION ACTIONS:
There are two types of actions available to you:

1. MOVE ACTIONS (numbered 1, 2, 3, ...):
   - Numbered circles in the image show potential destination waypoints
   - Each waypoint is labeled with a POSITIVE number (1, 2, 3, ...) in a circle with red border
   - The number indicates which waypoint you would move to
   - These waypoints represent safe, reachable positions in different directions
   - Selecting a MOVE action will navigate the robot to that waypoint

2. ROTATION ACTIONS (always available):
   - Action -1: TURN LEFT by {self.turn_angle_deg:.0f}° in place (shown on left side of image)
   - Action -2: TURN RIGHT by {self.turn_angle_deg:.0f}° in place (shown on right side of image)
   - These actions rotate the robot without changing position
{description_text}
DECISION STRATEGY:
1. Use prior knowledge about where items like {goal.upper()} are typically located in rooms
2. PRIORITIZE forward movement (MOVE actions) over rotation when possible
3. Use rotation actions (TURN LEFT/RIGHT) only when you need to adjust your viewing angle
4. Choose safe directions considering the robot's body size
5. To check a place (desk, open area, etc.), move NEAR it first, then turn to check (avoid direct collision)
6. Don't stay in one place too long; move to next area after checking
7. For far destinations, explore nearby first (use actions at image edges to reveal new areas)
8. Avoid paths near walls or obstacles

ROTATION USAGE GUIDELINES:
- Prefer MOVE actions for exploration (they cover more ground)
- Use TURN LEFT/RIGHT when:
  * You need to look around at your current position
  * You want to check behind or beside you
  * You need to align with a target before moving forward
- Avoid excessive rotation; make progress through movement

CONSTRAINTS:
- You CANNOT go through closed doors
- You CANNOT go up or down stairs
- Stay in the current room

Remember these instructions throughout the navigation episode. I will show you observations with iteration IDs, and you should apply these rules consistently to make navigation decisions.
"""
        return initial_prompt
    
    def get_history_summary(self):
        """
        Get a summary of navigation history for debugging/logging.
        Attaches current non-keyframe buffer to the last keyframe before returning.
        
        Returns
        -------
        dict
            Dictionary containing history statistics and full history
        """
        if not self.cfg['keyframe_mode']:
            return {'keyframe_mode': False}
        
        # Attach current non-keyframe buffer to the last keyframe
        if len(self.keyframe_history) > 0 and len(self.current_cycle_nonkeyframes) > 0:
            self.keyframe_history[-1]['non_keyframes'] = self.current_cycle_nonkeyframes.copy()
        
        # Build full history for serialization (without rgb_vis to save space)
        full_history = []
        for kf in self.keyframe_history:
            kf_record = {
                'iter': kf['iter'],
                'is_keyframe': True,
                'action': kf['action'],
                'action_number': kf['action_number'],
                'response': kf['response'],
                'timestamp': kf.get('timestamp'),
                'non_keyframes': []
            }
            # Add non-keyframes for this keyframe
            for nkf in kf.get('non_keyframes', []):
                nkf_record = {
                    'iter': nkf['iter'],
                    'is_keyframe': False,
                    'action': nkf['action'],
                    'action_number': nkf['action_number'],
                    'response': nkf['response'],
                    'timestamp': nkf.get('timestamp')
                }
                kf_record['non_keyframes'].append(nkf_record)
            full_history.append(kf_record)
        
        return {
            'keyframe_mode': True,
            'total_keyframes': len(self.keyframe_history),
            'nonkeyframe_buffer_size': len(self.current_cycle_nonkeyframes),
            'current_waypoint_P': self.current_waypoint_P_global.tolist() if self.current_waypoint_P_global is not None else None,
            'keyframe_iters': [kf['iter'] for kf in self.keyframe_history],
            'nonkeyframe_iters': [nkf['iter'] for nkf in self.current_cycle_nonkeyframes],
            'full_history': full_history
        }

    def cal_fov(self, intrinsics: np.ndarray, W: int):
        fx = intrinsics[0, 0]
        return 2 * np.arctan(W / (2 * fx)) * 180 / np.pi
    
    def _transform_waypoint_to_current_base(self, P_global, T_odom_base_current):
        """
        Transform waypoint P from global (odom) coordinates to current base frame.
        
        Parameters
        ----------
        P_global : np.ndarray
            Waypoint position in odom frame (x, y, z)
        T_odom_base_current : np.ndarray
            Current base pose (4x4 matrix, base->odom)
            
        Returns
        -------
        r : float
            Distance from current base to P
        theta : float
            Angle from current base to P (radians)
        """
        # Transform P from odom to current base frame
        # T_odom_base: base->odom, so we need odom->base = inv(T_odom_base)
        T_base_odom = np.linalg.inv(T_odom_base_current)
        
        # P in homogeneous coordinates
        P_homo = np.array([P_global[0], P_global[1], P_global[2], 1.0], dtype=np.float64)
        
        # Transform to base frame
        P_base = T_base_odom @ P_homo
        
        # Calculate r and theta in base frame (x-forward, y-left)
        r = float(np.sqrt(P_base[0]**2 + P_base[1]**2))
        theta = float(np.arctan2(P_base[1], P_base[0]))
        
        return r, theta
    
    def _can_project_waypoint(self, P_global, obs):
        """
        Check if waypoint P can be projected to current frame.
        
        Parameters
        ----------
        P_global : np.ndarray
            Waypoint position in odom frame (x, y, z)
        obs : dict
            Current observation with intrinsic, extrinsic, base_to_odom_matrix
            
        Returns
        -------
        can_project : bool
            Whether P can be projected to image
        pixel_coords : tuple or None
            (u, v) pixel coordinates if projectable, None otherwise
        """
        if P_global is None:
            return False, None
        
        # Transform P to current base frame
        r, theta = self._transform_waypoint_to_current_base(P_global, obs['base_to_odom_matrix'])
        
        # Try to project to image
        K = obs['intrinsic']
        T_cam_odom = obs['extrinsic']
        T_odom_base = obs['base_to_odom_matrix']
        T_cam_base = T_cam_odom @ T_odom_base
        
        # Check if can project (using existing method)
        pixel_coords = self._can_project(r, theta, obs['rgb'].shape[:2], K, T_cam_base)
        
        if pixel_coords is not None:
            return True, pixel_coords
        else:
            return False, None
    
    def _determine_frame_type(self, obs):
        """
        Determine if current frame is keyframe or non-keyframe.
        
        Parameters
        ----------
        obs : dict
            Current observation
            
        Returns
        -------
        is_keyframe : bool
            True if keyframe, False if non-keyframe
        angle_range : tuple or None
            (theta_min, theta_max) for non-keyframe, None for keyframe
        clip_dist_override : float or None
            Distance override for non-keyframe, None for keyframe
        r_P : float or None
            Distance to waypoint P in current base frame (for non-keyframe)
        theta_P : float or None
            Angle to waypoint P in current base frame (for non-keyframe)
        """
        if not self.cfg['keyframe_mode']:
            # Keyframe mode disabled, always treat as keyframe
            return True, None, None, None, None
        
        # First frame is always keyframe
        if self.current_waypoint_P_global is None:
            return True, None, None, None, None
        
        # Check if P can be projected to current frame
        can_project, _ = self._can_project_waypoint(self.current_waypoint_P_global, obs)
        
        if can_project:
            # Non-keyframe: P is visible
            r_P, theta_P = self._transform_waypoint_to_current_base(
                self.current_waypoint_P_global, obs['base_to_odom_matrix']
            )
            
            # Calculate angle range: P ± keyframe_angle_range
            angle_range_deg = self.cfg['keyframe_angle_range']
            angle_range_rad = np.deg2rad(angle_range_deg)
            theta_min = theta_P - angle_range_rad
            theta_max = theta_P + angle_range_rad
            
            return False, (theta_min, theta_max), r_P, r_P, theta_P
        else:
            # Keyframe: P is not visible
            return True, None, None, None, None

    def _global_to_grid(self, position: np.ndarray):
        dx = position[0] - self.init_pos[0]
        dy = position[1] - self.init_pos[1]
        H, W = self.voxel_map.shape[:2]
        x = int(W // 2 + dx * self.scale)
        y = int(H // 2 + dy * self.scale)
        return (x, y)

    def _get_navigability_mask(self, depth_image: np.ndarray, intrinsics: np.ndarray, T_cam_world: np.ndarray):
        """
        Compute navigability mask based on obstacle height range.
        
        Logic: A pixel is navigable if there are NO points in the obstacle height range [min, max].
        - If height is in [obstacle_height_min, obstacle_height_max]: obstacle (not navigable)
        - Otherwise: navigable (either too low like ground, or too high like ceiling)
        
        Parameters
        ----------
        depth_image : np.ndarray
            Depth image in meters
        intrinsics : np.ndarray
            Camera intrinsic matrix
        T_cam_world : np.ndarray
            Transform from world to camera (odom->cam)
            
        Returns
        -------
        navigability_mask : np.ndarray
            Boolean mask where True = navigable, False = obstacle
        """
        height_map = depth_to_height(depth_image, intrinsics, T_cam_world)
        
        # Get obstacle height range from config
        h_min = self.cfg['obstacle_height_min']
        h_max = self.cfg['obstacle_height_max']
        
        # Navigable if height is OUTSIDE the obstacle range
        # (either below h_min like ground, or above h_max like ceiling)
        navigability_mask = (height_map < h_min) | (height_map > h_max)
        
        # Also mark invalid depth as not navigable
        navigability_mask = navigability_mask & np.isfinite(height_map)
        
        return navigability_mask.astype(bool)

    def _get_radial_distance(
        self,
        start_pxl: tuple,
        theta_i: float,
        navigability_mask: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: np.ndarray,
        T_cam_base: np.ndarray,
        clip_dist: float = None,
    ):
        H, W = navigability_mask.shape
        
        # Use provided clip_dist or fall back to self.clip_dist
        if clip_dist is None:
            clip_dist = self.clip_dist

        agent_point_base = np.array([clip_dist * np.cos(theta_i), clip_dist * np.sin(theta_i), 0.0], dtype=np.float64)
        end_pxl = point_to_pixel(agent_point_base, intrinsics, T_cam_base)
        if end_pxl is None:
            return None, None
        end_pxl = end_pxl[0]
        if start_pxl is None:
            return None, None

        x0, y0 = start_pxl
        x1, y1 = int(round(end_pxl[0])), int(round(end_pxl[1]))
        if y1 < 0 or y1 >= H:
            return None, None

        intersections = find_intersections(x0, y0, x1, y1, W, H)
        if intersections is None:
            return None, None

        (xa, ya), (xb, yb) = intersections
        num_points = max(abs(xb - xa), abs(yb - ya)) + 1
        x_coords = np.linspace(xa, xb, num_points)
        y_coords = np.linspace(ya, yb, num_points)

        # 如果起点就不可通行，返回 0
        if not navigability_mask[int(np.clip(y_coords[0], 0, H - 1)), int(np.clip(x_coords[0], 0, W - 1))]:
            return 0.0, theta_i

        out = (int(x_coords[-1]), int(y_coords[-1]))
        stop_i = None
        for i in range(num_points - 4):
            y = int(y_coords[i])
            x = int(x_coords[i])
            if sum([navigability_mask[int(y_coords[j]), int(x_coords[j])] for j in range(i, i + 4)]) <= 2:
                out = (x, y)
                stop_i = i
                break

        if stop_i is not None and stop_i < 5:
            return 0.0, theta_i

        # 用 depth 取距离
        u = int(np.clip(out[0], 0, W - 1))
        v = int(np.clip(out[1], 0, H - 1))
        d = float(depth_image[v, u])
        if not np.isfinite(d) or d <= 1e-6:
            return None, None

        cam_coords = unproject_2d(u, v, d, intrinsics)  # camera frame
        base_coords = local_to_global_matrix(np.linalg.inv(T_cam_base), cam_coords)  # cam->base
        r_i = float(np.linalg.norm(base_coords[:2]))
        return r_i, theta_i

    def _update_voxel(self, r: float, theta: float, T_odom_base: np.ndarray, clip_dist: float, clip_frac: float):
        agent_coords = self._global_to_grid(T_odom_base[:3, 3])

        unclipped = max(r - 0.5, 0.0)
        local_coords = np.array([unclipped * np.cos(theta), unclipped * np.sin(theta), 0.0], dtype=np.float64)
        global_coords = local_to_global_matrix(T_odom_base, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, self.voxel_ray_size)

        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.cos(theta), clipped * np.sin(theta), 0.0], dtype=np.float64)
        global_coords = local_to_global_matrix(T_odom_base, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _navigability(self, obs: dict, angle_range=None, clip_dist_override=None):
        """
        Compute navigable directions.
        
        Parameters
        ----------
        obs : dict
            Observation dictionary
        angle_range : tuple, optional
            (theta_min, theta_max) to limit angle search range (in radians)
            If None, use full sensor range
        clip_dist_override : float, optional
            Override self.clip_dist for ray casting
            If None, use self.clip_dist
            
        Returns
        -------
        a_initial : list
            List of (r, theta) tuples
        """
        rgb_image = obs['rgb'].copy()
        depth_image = obs['depth']
        K = obs['intrinsic']
        T_cam_odom = obs['extrinsic']
        T_odom_base = obs['base_to_odom_matrix']

        # cam<-base
        T_cam_base = T_cam_odom @ T_odom_base

        self.fov = self.cal_fov(K, rgb_image.shape[1])
        if self.init_pos is None:
            self.init_pos = T_odom_base[:3, 3].copy()

        navigability_mask = self._get_navigability_mask(depth_image, K, T_cam_odom)

        # Determine angle range
        if angle_range is not None:
            # Use specified range
            theta_min, theta_max = angle_range
            all_thetas = np.linspace(theta_min, theta_max, int(self.cfg['num_theta']))
        else:
            # Use full sensor range
            sensor_range = np.deg2rad(self.fov / 2) * 1.5
            all_thetas = np.linspace(-sensor_range, sensor_range, int(self.cfg['num_theta']))

        start = agent_frame_to_image_coords([0.0, 0.0, 0.0], K, T_cam_base)
        
        # Determine clip_dist to use
        clip_dist = clip_dist_override if clip_dist_override is not None else self.clip_dist

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(
                start, theta_i, navigability_mask, depth_image, K, T_cam_base, clip_dist
            )
            if r_i is not None:
                self._update_voxel(r_i, theta_i, T_odom_base, clip_dist=clip_dist, clip_frac=0.66)
                a_initial.append((r_i, theta_i))

        return a_initial

    def _action_proposer(self, a_initial: list, T_odom_base: np.ndarray):
        min_angle = self.fov / 360
        explore_bias = 4

        explore = explore_bias > 0
        unique = {}
        for mag, theta in a_initial:
            unique.setdefault(theta, []).append(mag)

        arrowData = []
        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        for theta, mags in unique.items():
            mag = min(mags)
            cart = [self.e_i_scaling * mag * np.cos(theta), self.e_i_scaling * mag * np.sin(theta), 0.0]
            global_coords = local_to_global_matrix(T_odom_base, cart)
            grid_coords = self._global_to_grid(global_coords)
            xg, yg = grid_coords
            xg = np.clip(xg, 2, topdown_map.shape[1] - 3)
            yg = np.clip(yg, 2, topdown_map.shape[0] - 3)
            score = (
                np.sum(np.all((topdown_map[yg-2:yg+2, xg] == self.explored_color), axis=-1)) +
                np.sum(np.all(topdown_map[yg, xg-2:xg+2] == self.explored_color, axis=-1))
            )
            # 不再使用clip_frac，直接使用原始距离
            arrowData.append([mag, theta, score < 3])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        # 只过滤太近的点（< 0.5m）
        filter_thresh = 0.5
        filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))
        filtered.sort(key=lambda x: x[1])
        if not filtered:
            return []

        if explore:
            f = list(filter(lambda x: x[2], filtered))
            if len(f) > 0:
                longest = max(f, key=lambda x: x[0])
                longest_theta = longest[1]
                smallest_theta = longest[1]
                longest_ndx = f.index(longest)

                # 不再使用clip_mag限制
                out.append([longest[0], longest[1], longest[2]])
                thetas.add(longest[1])

                for i in range(longest_ndx + 1, len(f)):
                    if f[i][1] - longest_theta > (min_angle * 0.9):
                        out.append([f[i][0], f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        longest_theta = f[i][1]

                for i in range(longest_ndx - 1, -1, -1):
                    if smallest_theta - f[i][1] > (min_angle * 0.9):
                        out.append([f[i][0], f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        smallest_theta = f[i][1]

                for r_i, theta_i, e_i in filtered:
                    if len(thetas) == 0:
                        out.append((r_i, theta_i, e_i))
                        thetas.add(theta_i)
                    else:
                        if theta_i not in thetas and min([abs(theta_i - t) for t in thetas]) > min_angle * explore_bias:
                            out.append((r_i, theta_i, e_i))
                            thetas.add(theta_i)

        if len(out) == 0:
            longest = max(filtered, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = filtered.index(longest)
            out.append([longest[0], longest[1], longest[2]])

            for i in range(longest_ndx + 1, len(filtered)):
                if filtered[i][1] - longest_theta > min_angle:
                    out.append([filtered[i][0], filtered[i][1], filtered[i][2]])
                    longest_theta = filtered[i][1]

            for i in range(longest_ndx - 1, -1, -1):
                if smallest_theta - filtered[i][1] > min_angle:
                    out.append([filtered[i][0], filtered[i][1], filtered[i][2]])
                    smallest_theta = filtered[i][1]

        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta, _ in out]

    def _projection(self, a_final: list, obs: dict, chosen_action: int = None):
        rgb = obs['rgb'].copy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        K = obs['intrinsic']
        T_cam_odom = obs['extrinsic']
        T_odom_base = obs['base_to_odom_matrix']
        T_cam_base = T_cam_odom @ T_odom_base

        projected = self._project_onto_image(
            a_final=a_final,
            rgb_image=bgr,
            intrinsics=K,
            T_cam_base=T_cam_base,
            chosen_action=chosen_action,
        )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return projected, rgb

    def _can_project(self, r_i: float, theta_i: float, rgb_shape_hw, intrinsics, T_cam_base):
        H, W = rgb_shape_hw
        p_base = np.array([r_i * np.cos(theta_i), r_i * np.sin(theta_i), 0.0], dtype=np.float64)
        uv = point_to_pixel(p_base, intrinsics, T_cam_base)
        if uv is None:
            return None
        u, v = uv[0]
        u = int(round(u))
        v = int(round(v))

        thr = float(self.cfg['image_edge_threshold'])
        if (thr * W <= u <= (1 - thr) * W) and (thr * H <= v <= (1 - thr) * H):
            return (u, v)
        return None

    def _project_onto_image(self, a_final, rgb_image, intrinsics, T_cam_base, chosen_action=None):
        scale_factor = rgb_image.shape[0] / 1080.0
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_color = BLACK
        circle_color = WHITE

        projected = {}

        start_px = agent_frame_to_image_coords([0.0, 0.0, 0.0], intrinsics, T_cam_base)
        if start_px is None:
            start_px = (rgb_image.shape[1] // 2, rgb_image.shape[0] // 2)

        # ---------- 画waypoint圆圈 1..N（不画箭头）----------
        projected_count = 0
        skipped_count = 0
        skipped_reasons = []
        
        for (r_i, theta_i) in a_final:
            # 先检查原始距离是否可以投影（用于过滤）
            end_px = self._can_project(r_i, theta_i, rgb_image.shape[:2], intrinsics, T_cam_base)
            if end_px is None:
                skipped_count += 1
                skipped_reasons.append(f"r={r_i:.2f}, theta={np.degrees(theta_i):.1f}° - original distance not projectable")
                continue
            
            # Waypoint位置：最大距离的4/5处（80%）
            r_waypoint = r_i * 0.8
            
            # 投影waypoint到图像（不做边缘检测，因为已经用原始距离检查过了）
            H, W = rgb_image.shape[:2]
            p_base = np.array([r_waypoint * np.cos(theta_i), r_waypoint * np.sin(theta_i), 0.0], dtype=np.float64)
            uv = point_to_pixel(p_base, intrinsics, T_cam_base)
            if uv is None:
                skipped_count += 1
                skipped_reasons.append(f"r={r_i:.2f}, theta={np.degrees(theta_i):.1f}° - waypoint projection failed")
                continue
            waypoint_px = (int(round(uv[0][0])), int(round(uv[0][1])))
            
            # 确保waypoint在图像范围内
            if not (0 <= waypoint_px[0] < W and 0 <= waypoint_px[1] < H):
                skipped_count += 1
                skipped_reasons.append(f"r={r_i:.2f}, theta={np.degrees(theta_i):.1f}° - waypoint out of bounds: ({waypoint_px[0]}, {waypoint_px[1]})")
                continue
            
            projected_count += 1
            action_name = len(projected) + 1
            projected[(r_i, theta_i)] = action_name
            print(f"[DEBUG] Projected action {action_name}: r={r_i:.2f}m, waypoint_r={r_waypoint:.2f}m, theta={np.degrees(theta_i):.1f}°, px=({waypoint_px[0]}, {waypoint_px[1]})")

            # 只画圆圈（不画箭头）
            text = str(action_name)
            text_size = 2.4 * scale_factor
            text_thickness = max(1, math.ceil(3 * scale_factor))
            (tw, th), _ = cv2.getTextSize(text, font, text_size, text_thickness)

            circle_center = waypoint_px
            circle_radius = max(tw, th) // 2 + max(1, math.ceil(15 * scale_factor))

            if chosen_action is not None and action_name == chosen_action:
                cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)

            cv2.circle(rgb_image, circle_center, circle_radius, RED, max(1, math.ceil(2 * scale_factor)))

            text_position = (circle_center[0] - tw // 2, circle_center[1] + th // 2)
            cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)

        if len(a_final) > 0:
            print(f"[DEBUG] _project_onto_image: {len(a_final)} actions from _action_proposer, {projected_count} projected, {skipped_count} skipped by _can_project")

        # ---------- 画左转按钮 (Action -1) ----------
        # Add turn actions to projected dict (they are always available)
        projected[-1] = -1  # Turn left
        projected[-2] = -2  # Turn right
        
        text = '-1'
        text_size = 2.4 * scale_factor  # Same size as move actions
        text_thickness = max(1, math.ceil(3 * scale_factor))
        (tw, th), _ = cv2.getTextSize(text, font, text_size, text_thickness)

        circle_center_left = (math.ceil(0.05 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2))
        circle_radius = max(tw, th) // 2 + max(1, math.ceil(15 * scale_factor))

        if chosen_action == -1:
            cv2.circle(rgb_image, circle_center_left, circle_radius, GREEN, -1)
        else:
            cv2.circle(rgb_image, circle_center_left, circle_radius, circle_color, -1)

        cv2.circle(rgb_image, circle_center_left, circle_radius, RED, max(1, math.ceil(2 * scale_factor)))

        text_position_left = (circle_center_left[0] - tw // 2, circle_center_left[1] + th // 2)
        cv2.putText(rgb_image, text, text_position_left, font, text_size, text_color, text_thickness)
        
        # 添加 "TURN LEFT" 标签（放在圆圈下方）
        label_text = 'TURN LEFT'
        label_size = 1.8 * scale_factor
        label_thickness = max(1, math.ceil(3 * scale_factor))  # Thicker text
        cv2.putText(
            rgb_image, label_text,
            (text_position_left[0] // 2, text_position_left[1] + math.ceil(80 * scale_factor)),
            font, label_size, RED, label_thickness
        )

        # ---------- 画右转按钮 (Action -2) ----------
        text = '-2'
        (tw, th), _ = cv2.getTextSize(text, font, text_size, text_thickness)

        # Move right button further from edge to avoid text overflow
        circle_center_right = (math.ceil(0.90 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2))
        circle_radius = max(tw, th) // 2 + max(1, math.ceil(15 * scale_factor))

        if chosen_action == -2:
            cv2.circle(rgb_image, circle_center_right, circle_radius, GREEN, -1)
        else:
            cv2.circle(rgb_image, circle_center_right, circle_radius, circle_color, -1)

        cv2.circle(rgb_image, circle_center_right, circle_radius, RED, max(1, math.ceil(2 * scale_factor)))

        text_position_right = (circle_center_right[0] - tw // 2, circle_center_right[1] + th // 2)
        cv2.putText(rgb_image, text, text_position_right, font, text_size, text_color, text_thickness)
        
        # 添加 "TURN RIGHT" 标签（放在圆圈下方，确保不超出边界）
        label_text = 'TURN RIGHT'
        label_size = 1.8 * scale_factor
        label_thickness = max(1, math.ceil(3 * scale_factor))  # Thicker text
        (label_w, label_h), _ = cv2.getTextSize(label_text, font, label_size, label_thickness)
        # Right align with the circle, with some padding from right edge
        label_x = min(circle_center_right[0] - label_w // 2, rgb_image.shape[1] - label_w - 5)
        cv2.putText(
            rgb_image, label_text,
            (label_x, text_position_right[1] + math.ceil(80 * scale_factor)),
            font, label_size, RED, label_thickness
        )

        return projected

    def _nav(self, obs: dict, goal: str, iter: int, goal_description: str = ""):
        # Determine frame type (keyframe or non-keyframe)
        is_keyframe, angle_range, clip_dist_override, r_P, theta_P = self._determine_frame_type(obs)
        
        # Store frame type for prompt generation
        self.is_keyframe = is_keyframe
        
        print(f"[NavAgent] Frame type: {'KEYFRAME' if is_keyframe else 'NON-KEYFRAME'}")
        if not is_keyframe:
            print(f"[NavAgent] Non-keyframe: P at r={r_P:.2f}m, theta={np.degrees(theta_P):.1f}°, "
                  f"angle_range={np.degrees(angle_range[0]):.1f}° to {np.degrees(angle_range[1]):.1f}°, "
                  f"clip_dist={clip_dist_override:.2f}m")
        
        # Compute navigability with appropriate parameters
        a_initial = self._navigability(obs, angle_range=angle_range, clip_dist_override=clip_dist_override)
        a_final = self._action_proposer(a_initial, obs['base_to_odom_matrix'])

        # Start timing for projection (image annotation)
        t_projection_start = time.time()
        
        # Generate visualization without highlighting (for initial display)
        a_final_projected, rgb_vis = self._projection(a_final, obs)

        # Only send current image (VLM manages history internally)
        images = [rgb_vis]

        # 根据帧类型构建prompt和历史
        if is_keyframe:
            prompt = self._construct_keyframe_prompt(
                num_actions=len(a_final_projected),
                iter=iter
            )
            vlm_history = self._build_vlm_history_for_keyframe()
            print(f"[NavAgent] Keyframe {len(self.keyframe_history) + 1}: VLM sees {len(self.keyframe_history)} historical keyframes")
        else:
            prompt = self._construct_nonkeyframe_prompt(
                num_actions=len(a_final_projected),
                iter=iter
            )
            vlm_history = self._build_vlm_history_for_nonkeyframe()
            print(f"[NavAgent] Non-keyframe: VLM sees 1 parent keyframe + {len(self.current_cycle_nonkeyframes)} non-keyframes")
        
        t_projection_end = time.time()
        projection_time = t_projection_end - t_projection_start

        # Start timing for VLM inference
        t_vlm_start = time.time()
        
        # 调用VLM（使用自定义历史，由NavAgent控制Memory）
        response = self.actionVLM.call_chat_with_custom_history(
            custom_history=vlm_history,
            images=images,
            text_prompt=prompt
        )
        
        t_vlm_end = time.time()
        vlm_inference_time = t_vlm_end - t_vlm_start
        
        frame_type = "KEYFRAME" if is_keyframe else "NON-KEYFRAME"
        print(f'[NavAgent] Frame type: {frame_type}')
        print(f'[NavAgent] Timing - Projection: {projection_time:.3f}s, VLM inference: {vlm_inference_time:.3f}s')
        print(f'[NavAgent] Prompt length: {len(prompt)} chars')
        print(f'Response: {response}')

        rev = {v: k for k, v in a_final_projected.items()}
        try:
            response_dict = self._eval_response(response)
            action_number = int(response_dict['action'])
            
            # Re-generate visualization with chosen action highlighted in GREEN
            _, rgb_vis_final = self._projection(
                a_final,
                obs,
                chosen_action=action_number
            )
            
            # Prepare timing information (include prompt for debugging)
            timing_info = {
                'projection_time': float(projection_time),
                'vlm_inference_time': float(vlm_inference_time),
                'prompt': prompt,  # Include prompt for server to save
                'is_keyframe': is_keyframe
            }
            
            # Handle rotation actions
            if action_number == -1:
                # Turn left: return positive angle (counter-clockwise)
                return response, rgb_vis_final, ('turn', self.turn_angle_deg), timing_info
            elif action_number == -2:
                # Turn right: return negative angle (clockwise)
                return response, rgb_vis_final, ('turn', -self.turn_angle_deg), timing_info
            else:
                # Forward movement action
                action = rev.get(action_number)
                
                # History management for keyframe mode
                if self.cfg['keyframe_mode']:
                    # Create frame record（保存prompt用于构建VLM历史）
                    frame_record = {
                        'iter': iter,
                        'is_keyframe': is_keyframe,
                        'action': action,
                        'action_number': action_number,
                        'response': response,
                        'prompt': prompt,  # 保存prompt用于重建VLM对话历史
                        'rgb_vis': rgb_vis_final.copy() if rgb_vis_final is not None else None,
                        'timestamp': obs.get('timestamp', None)
                    }
                    
                    if is_keyframe:
                        # 关键帧：保存waypoint P并开始新周期
                        if action is not None:
                            r, theta = action
                            # Convert (r, theta) in current base frame to global (odom) coordinates
                            T_odom_base = obs['base_to_odom_matrix']
                            # Point in base frame
                            p_base = np.array([r * np.cos(theta), r * np.sin(theta), 0.0, 1.0], dtype=np.float64)
                            # Transform to odom frame
                            p_odom = T_odom_base @ p_base
                            self.current_waypoint_P_global = p_odom[:3]
                            print(f"[NavAgent] Keyframe: Saved waypoint P at global position: {self.current_waypoint_P_global}")
                        
                        # 将当前关键帧添加到历史
                        self.keyframe_history.append(frame_record)
                        self.current_keyframe_idx = len(self.keyframe_history) - 1
                        
                        # 清空当前周期的非关键帧列表（开始新周期）
                        self.current_cycle_nonkeyframes = []
                        
                        print(f"[NavAgent] Memory: Added keyframe #{self.current_keyframe_idx + 1} (total keyframes: {len(self.keyframe_history)})")
                    else:
                        # 非关键帧：添加到当前周期
                        self.current_cycle_nonkeyframes.append(frame_record)
                        print(f"[NavAgent] Memory: Added non-keyframe to current cycle (cycle size: {len(self.current_cycle_nonkeyframes)})")
                
                return response, rgb_vis_final, action, timing_info
        except (IndexError, KeyError, TypeError, ValueError) as e:
            print(f'Error parsing response {e}')
            return None, None, None, {'projection_time': 0.0, 'vlm_inference_time': 0.0}

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        try:
            eval_resp = ast.literal_eval(response[response.rindex('{'):response.rindex('}') + 1])
            if isinstance(eval_resp, dict):
                return eval_resp
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            return {}


if __name__ == "__main__":
    import pickle
    data = pickle.load(open('/home/liujy/nav_ws/src/llm_nav/captured_data/test_data.pkl', 'rb'))
    agent = NavAgent()
    agent._nav(data, 'banana', iter=1)
