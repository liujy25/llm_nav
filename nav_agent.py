#!/usr/bin/env python3
import math
import time
import json
import os
import numpy as np
import cv2
import ast
from collections import deque

from vlm import OpenAIVLM
from obstacle_map import ObstacleMap
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

        # Load prompts from JSON file
        prompts_path = os.path.join(os.path.dirname(__file__), 'prompts.json')
        with open(prompts_path, 'r', encoding='utf-8') as f:
            self.prompts = json.load(f)

        # defaults
        self.cfg.setdefault('num_theta', 40)
        self.cfg.setdefault('image_edge_threshold', 0.04)
        self.cfg.setdefault('clip_dist', 5.0)
        self.cfg.setdefault('turn_angle_deg', 30.0)  # Turn left/right angle in degrees
        
        # Navigability configuration (obstacle height range)
        self.cfg.setdefault('obstacle_height_min', 0.15)  # Minimum obstacle height (m)
        self.cfg.setdefault('obstacle_height_max', 2.0)   # Maximum obstacle height (m)
        
        # Prompt mode: 'reasoning' or 'action_only'
        self.cfg.setdefault('prompt_mode', 'reasoning')  # 'reasoning' or 'action_only'
        
        # VLM configuration defaults
        self.cfg.setdefault('vlm_model', '/data/sea_disk0/liujy/models/Qwen/Qwen3-VL-8B-Instruct/')
        self.cfg.setdefault('vlm_api_key', 'EMPTY')
        self.cfg.setdefault('vlm_base_url', 'http://10.15.89.71:34134/v1/')
        self.cfg.setdefault('vlm_timeout', 10)
        self.cfg.setdefault('vlm_history', 3)  # Number of conversation rounds to keep in context
        
        # BEV map configuration
        self.cfg.setdefault('enable_bev_map', True)  # Enable BEV map visualization
        self.cfg.setdefault('bev_map_size', 2000)  # BEV map size (200x200 for 20m x 20m)
        self.cfg.setdefault('bev_pixels_per_meter', 100)  # 10 pixels per meter

        self.turn_angle_deg = float(self.cfg['turn_angle_deg'])
        self.clip_dist = self.cfg['clip_dist']
        self.prompt_mode = self.cfg['prompt_mode']
        
        # Initialize goal attributes (will be set properly in reset())
        self.goal = None
        self.goal_description = ''

        # Initialize ObstacleMap (always create for navigation logic)
        # enable_bev_map only controls whether to send BEV to VLM
        self.obstacle_map = ObstacleMap(
            min_height=self.cfg['obstacle_height_min'],
            max_height=self.cfg['obstacle_height_max'],
            agent_radius=0.3,
            area_thresh=0.3,
            size=self.cfg['bev_map_size'],
            pixels_per_meter=self.cfg['bev_pixels_per_meter'],
        )
        self.bev_scale = self.cfg['bev_pixels_per_meter']

        self._initialize_vlms()
        self.reset()

    def _initialize_vlms(self):
        # Use system instruction from prompts.json
        system_instruction = self.prompts['system_instruction']

        self.actionVLM = OpenAIVLM(
            model=self.cfg['vlm_model'],
            system_instruction=system_instruction,
            api_key=self.cfg['vlm_api_key'],
            base_url=self.cfg['vlm_base_url'],
            timeout=self.cfg['vlm_timeout']
        )

    def _format_full_messages_as_text(self, current_prompt: str, num_images: int = 2):
        """
        Format the complete messages structure (as sent to VLM) into readable text.
        Images are replaced with placeholders like [IMAGE: RGB], [IMAGE: BEV].

        Parameters
        ----------
        current_prompt : str
            The current iteration's text prompt
        num_images : int
            Number of images in current message (default: 2 for RGB + BEV)

        Returns
        -------
        str
            Formatted text representation of all messages
        """
        lines = []
        lines.append("=" * 80)
        lines.append("COMPLETE CONVERSATION CONTEXT")
        lines.append("=" * 80)
        lines.append("")

        # 1. System instruction
        if self.actionVLM.system_instruction:
            lines.append("[SYSTEM INSTRUCTION]")
            lines.append("-" * 80)
            lines.append(self.actionVLM.system_instruction)
            lines.append("")

        # 2. Initial prompt
        if self.actionVLM.initial_prompt:
            lines.append("[INITIAL PROMPT - Task Briefing]")
            lines.append("-" * 80)
            lines.append(self.actionVLM.initial_prompt)
            lines.append("")
            lines.append("[ASSISTANT ACKNOWLEDGMENT]")
            lines.append("-" * 80)
            lines.append("Understood. I will follow these instructions for navigation.")
            lines.append("")

        # 3. Recent conversation history
        # Apply the same history limit logic as in vlm.py
        history_limit = self.cfg['vlm_history']

        if history_limit == 0:
            # When history=0, don't include any conversation history
            history = []
        elif len(self.actionVLM.history) > 2 * history_limit:
            history = self.actionVLM.history[-2 * history_limit:]
        else:
            history = self.actionVLM.history

        if len(history) > 0:
            lines.append(f"[CONVERSATION HISTORY - Last {history_limit} rounds]")
            lines.append("-" * 80)

            for i, msg in enumerate(history):
                role = msg['role'].upper()
                lines.append(f"[{role}]")

                if isinstance(msg['content'], str):
                    lines.append(msg['content'])
                elif isinstance(msg['content'], list):
                    for item in msg['content']:
                        if item['type'] == 'text':
                            lines.append(item['text'])
                        elif item['type'] == 'image_url':
                            lines.append("[IMAGE]")

                lines.append("")

        # 4. Current message
        lines.append("[CURRENT MESSAGE]")
        lines.append("-" * 80)
        lines.append(current_prompt)

        # Add image placeholders
        image_labels = ["[IMAGE: RGB]", "[IMAGE: BEV]"]
        for i in range(num_images):
            label = image_labels[i] if i < len(image_labels) else f"[IMAGE {i+1}]"
            lines.append(label)

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _construct_prompt(self, num_actions: int, iter: int, num_turn_actions: int = 2, available_turns: list = None):
        """
        构建统一的prompt：利用BEV地图+RGB做决策

        BEV地图已经提供了全局信息（探索区域、未探索区域、障碍物），
        不需要关键帧机制来维护长期记忆。每帧都能看到完整地图，直接做决策。

        Parameters
        ----------
        num_actions : int
            当前可用的MOVE动作数量
        iter : int
            当前迭代编号
        num_turn_actions : int
            当前可用的TURN动作数量 (0, 1, or 2)
        available_turns : list
            可用的转向动作列表 (e.g., [-1], [-2], or [-1, -2])

        Returns
        -------
        str
            prompt文本
        """
        if available_turns is None:
            available_turns = [-1, -2]

        # Use prompt template from JSON (choose based on enable_bev_map)
        if self.cfg['enable_bev_map']:
            template_lines = self.prompts['iteration_prompt_template'].copy()
        else:
            template_lines = self.prompts['iteration_prompt_template_no_bev'].copy()

        # Handle the case when no move waypoints are available
        if num_actions == 0:
            # Replace lines that mention waypoint counts
            for i, line in enumerate(template_lines):
                if "MOVE waypoints visible" in line:
                    template_lines[i] = "RGB Image (First): No MOVE waypoints available (only TURN actions)"
                elif line.strip().startswith("- MOVE:"):
                    template_lines[i] = "- MOVE: None available"

        # Build recent actions summary
        if len(self.action_history) > 0:
            action_strs = []
            for act in self.action_history[-5:]:  # Last 5 actions
                if act == -1:
                    action_strs.append(f"TURN LEFT ({self.turn_angle_deg}°)")
                elif act == -2:
                    action_strs.append(f"TURN RIGHT ({self.turn_angle_deg}°)")
                else:
                    action_strs.append(f"MOVE {act}")

            # Count consecutive turns in same direction
            consecutive_left = 0
            consecutive_right = 0
            for act in reversed(self.action_history):
                if act == -1:
                    consecutive_left += 1
                    if consecutive_right > 0:
                        break
                elif act == -2:
                    consecutive_right += 1
                    if consecutive_left > 0:
                        break
                else:
                    break

            recent_actions_text = f"Recent actions: {' -> '.join(action_strs)}"
            if consecutive_left > 0:
                total_angle = consecutive_left * self.turn_angle_deg
                recent_actions_text += f"\n(Turned left {total_angle}° total in last {consecutive_left} steps)"
            elif consecutive_right > 0:
                total_angle = consecutive_right * self.turn_angle_deg
                recent_actions_text += f"\n(Turned right {total_angle}° total in last {consecutive_right} steps)"
        else:
            recent_actions_text = "Recent actions: None (first iteration)"

        # Build turn actions text based on available turns
        if num_turn_actions == 0:
            turn_actions_text = "- TURN: None available"
        elif num_turn_actions == 1:
            if -1 in available_turns:
                turn_actions_text = f"- TURN: -1 (left {self.turn_angle_deg}°) only"
            else:
                turn_actions_text = f"- TURN: -2 (right {self.turn_angle_deg}°) only"
        else:  # num_turn_actions == 2
            turn_actions_text = f"- TURN: -1 (left {self.turn_angle_deg}°) or -2 (right {self.turn_angle_deg}°)"

        # Add current robot position information (trajectory point number)
        if self.current_trajectory_index is not None:
            current_position_text = f"Current robot position: Trajectory point {self.current_trajectory_index} (marked with red circle in BEV map)"
        else:
            current_position_text = "Current robot position: Starting position (no trajectory point yet)"

        prompt = '\n'.join(template_lines).format(
            iter=iter,
            num_actions=num_actions if num_actions > 0 else 0,
            turn_angle=self.turn_angle_deg,
            goal=self.goal,
            recent_actions=recent_actions_text,
            turn_actions=turn_actions_text,
            current_position=current_position_text
        )

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
        
        # Initialize trajectory history for visualization
        self.trajectory_history = []  # List of (x, y) positions in odom frame
        self.current_trajectory_index = None  # Index of current trajectory point (1-based, None if no trajectory yet)

        # Initialize action history for rotation strategy
        self.action_history = []  # List of recent actions (integers: -2, -1, 1, 2, ...)
        self.max_action_history = 5  # Keep last 5 actions
        self.last_turn_direction = None  # Track last turn direction: -1 (left), -2 (right), or None
        
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

        # Use prompt template from JSON (choose based on enable_bev_map)
        if self.cfg['enable_bev_map']:
            template_lines = self.prompts['initial_prompt_template']
        else:
            template_lines = self.prompts['initial_prompt_template_no_bev']

        initial_prompt = '\n'.join(template_lines).format(
            goal=goal.upper(),
            turn_angle=self.turn_angle_deg,
            description_text=description_text
        )

        return initial_prompt
    
    def get_history_summary(self):
        """
        Get a summary of navigation history for debugging/logging.
        
        Returns
        -------
        dict
            Dictionary containing basic history info
        """
        return {'keyframe_mode': False}

    def cal_fov(self, intrinsics: np.ndarray, W: int):
        fx = intrinsics[0, 0]
        return 2 * np.arctan(W / (2 * fx)) * 180 / np.pi

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

        # Check if depth is too small (obstacle very close)
        # If depth < 0.5m, treat as obstacle and return 0
        if d < 0.5:
            return 0.0, theta_i

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

    def _navigability_from_bev(self, obs: dict, angle_range=None, clip_dist_override=None):
        """
        Compute navigable directions using BEV obstacle map (more accurate).

        Parameters
        ----------
        obs : dict
            Observation dictionary
        angle_range : tuple, optional
            (theta_min, theta_max) to limit angle search range (in radians)
            If None, use full 360 degrees
        clip_dist_override : float, optional
            Override self.clip_dist for ray casting
            If None, use self.clip_dist

        Returns
        -------
        a_initial : list
            List of (r, theta) tuples
        """
        if self.obstacle_map is None:
            # Fallback to RGB-based method if BEV map is disabled
            return self._navigability(obs, angle_range, clip_dist_override)

        T_odom_base = obs['base_to_odom_matrix']

        if self.init_pos is None:
            self.init_pos = T_odom_base[:3, 3].copy()

        # Calculate FOV for _action_proposer (needed even for BEV-based method)
        if 'intrinsic' in obs and 'rgb' in obs:
            K = obs['intrinsic']
            rgb_image = obs['rgb']
            self.fov = self.cal_fov(K, rgb_image.shape[1])
        elif not hasattr(self, 'fov'):
            # Default FOV if not available
            self.fov = 90.0

        # Get robot position in odom frame
        robot_pos = T_odom_base[:3, 3]
        robot_yaw = np.arctan2(T_odom_base[1, 0], T_odom_base[0, 0])

        # Convert to pixel coordinates
        robot_px = self.obstacle_map._xy_to_px(robot_pos[:2].reshape(1, 2))[0]

        # Determine angle range
        clip_dist = clip_dist_override if clip_dist_override is not None else self.clip_dist

        if angle_range is not None:
            theta_min, theta_max = angle_range
            all_thetas = np.linspace(theta_min, theta_max, int(self.cfg['num_theta']))
        else:
            # Use forward-facing FOV range (strictly within camera FOV)
            sensor_range = np.deg2rad(self.fov / 2)
            all_thetas = np.linspace(-sensor_range, sensor_range, int(self.cfg['num_theta']))

        print(f"[_navigability_from_bev] FOV: {self.fov:.1f}°, sensor_range: {np.rad2deg(sensor_range):.1f}°")
        print(f"[_navigability_from_bev] Angle range: [{np.rad2deg(all_thetas[0]):.1f}°, {np.rad2deg(all_thetas[-1]):.1f}°]")
        print(f"[_navigability_from_bev] Robot yaw: {np.rad2deg(robot_yaw):.1f}°")

        a_initial = []

        for theta_i in all_thetas:
            # Compute absolute angle in odom frame
            absolute_theta = robot_yaw + theta_i

            # Raycast in BEV map
            r_i = self._raycast_bev(
                robot_px,
                robot_pos[:2],
                absolute_theta,
                clip_dist
            )

            if r_i is not None and r_i > 0.5:  # Minimum distance threshold
                # Note: _update_voxel is deprecated when using ObstacleMap
                # ObstacleMap.explored_area is updated automatically in update_map()
                # self._update_voxel(r_i, theta_i, T_odom_base, clip_dist=clip_dist, clip_frac=0.66)
                a_initial.append((r_i, theta_i))

        print(f"[_navigability_from_bev] Generated {len(a_initial)} waypoints")

        return a_initial

    def _raycast_bev(self, start_px, start_pos, angle, max_dist):
        """
        Raycast in BEV map to find maximum navigable distance.

        Parameters
        ----------
        start_px : np.ndarray
            Starting pixel position [px_x, px_y]
        start_pos : np.ndarray
            Starting position in odom frame [x, y]
        angle : float
            Ray direction in odom frame (radians)
        max_dist : float
            Maximum ray distance (meters)

        Returns
        -------
        float or None
            Maximum navigable distance, or None if no valid distance found
        """
        # Calculate end position
        end_x = start_pos[0] + max_dist * np.cos(angle)
        end_y = start_pos[1] + max_dist * np.sin(angle)
        end_px = self.obstacle_map._xy_to_px(np.array([[end_x, end_y]]))[0]

        # Bresenham line algorithm to get all pixels along the ray
        x0, y0 = int(start_px[0]), int(start_px[1])
        x1, y1 = int(end_px[0]), int(end_px[1])

        # Get line pixels
        line_pixels = self._bresenham_line(x0, y0, x1, y1)

        # Find first obstacle
        map_size = self.obstacle_map.size
        last_valid_px = None

        # Skip first few pixels near robot (robot itself may be detected as obstacle)
        skip_pixels = 5  # Skip ~0.05m around robot

        for i, (px_x, px_y) in enumerate(line_pixels):
            # Check bounds
            if not (0 <= px_x < map_size and 0 <= px_y < map_size):
                break

            # Skip pixels near robot position
            if i < skip_pixels:
                last_valid_px = (px_x, px_y)
                continue

            # Check if navigable AND explored
            # A region must be both navigable (based on robot radius) and explored
            # to be considered reachable. Unexplored regions (gray in BEV) should not
            # generate waypoints as we don't know if they're actually reachable.
            is_navigable = self.obstacle_map._navigable_map[px_y, px_x]
            is_explored = self.obstacle_map.explored_area[px_y, px_x]

            if not (is_navigable and is_explored):
                # Hit obstacle or unexplored area, return distance to last valid point
                break

            last_valid_px = (px_x, px_y)

        if last_valid_px is None:
            return None

        # Convert back to world coordinates and compute distance
        last_valid_pos = self.obstacle_map._px_to_xy(np.array([last_valid_px]))[0]
        distance = np.linalg.norm(last_valid_pos - start_pos)

        return distance

    def _bresenham_line(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm to get all pixels along a line.

        Parameters
        ----------
        x0, y0 : int
            Start pixel coordinates
        x1, y1 : int
            End pixel coordinates

        Returns
        -------
        list
            List of (x, y) pixel coordinates along the line
        """
        pixels = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            pixels.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return pixels

    def _navigability(self, obs: dict, angle_range=None, clip_dist_override=None):
        """
        Compute navigable directions (RGB-based, legacy method).

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

        # Get base frame origin in odom frame for projection
        base_origin_in_odom = T_odom_base[:3, 3]
        start = agent_frame_to_image_coords(base_origin_in_odom, K, T_cam_odom)

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

        # Use ObstacleMap's explored_area instead of voxel_map/explored_map
        use_obstacle_map = self.obstacle_map is not None

        for theta, mags in unique.items():
            mag = min(mags)
            cart = [self.e_i_scaling * mag * np.cos(theta), self.e_i_scaling * mag * np.sin(theta), 0.0]
            global_coords = local_to_global_matrix(T_odom_base, cart)

            # Calculate exploration score using ObstacleMap if available
            if use_obstacle_map:
                # Convert to ObstacleMap pixel coordinates
                xy = global_coords[:2].reshape(1, 2)
                px = self.obstacle_map._xy_to_px(xy)[0]
                px_x, px_y = int(px[0]), int(px[1])

                # Check if within bounds
                if 0 <= px_x < self.obstacle_map.size and 0 <= px_y < self.obstacle_map.size:
                    # Count explored pixels in surrounding area (similar to old logic)
                    x_min = max(0, px_x - 2)
                    x_max = min(self.obstacle_map.size, px_x + 3)
                    y_min = max(0, px_y - 2)
                    y_max = min(self.obstacle_map.size, px_y + 3)

                    score = (
                        np.sum(self.obstacle_map.explored_area[px_y, x_min:x_max]) +
                        np.sum(self.obstacle_map.explored_area[y_min:y_max, px_x])
                    )
                else:
                    score = 0  # Out of bounds = unexplored
            else:
                # Fallback to old voxel_map method if ObstacleMap is disabled
                grid_coords = self._global_to_grid(global_coords)
                xg, yg = grid_coords
                xg = np.clip(xg, 2, self.voxel_map.shape[1] - 3)
                yg = np.clip(yg, 2, self.voxel_map.shape[0] - 3)

                topdown_map = self.voxel_map.copy()
                mask = np.all(self.explored_map == self.explored_color, axis=-1)
                topdown_map[mask] = self.explored_color

                score = (
                    np.sum(np.all((topdown_map[yg-2:yg+2, xg] == self.explored_color), axis=-1)) +
                    np.sum(np.all(topdown_map[yg, xg-2:xg+2] == self.explored_color, axis=-1))
                )

            # score < 3 means frontier (unexplored area)
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
            else:
                # No frontier waypoints found, fallback to explored area waypoints
                # This allows the robot to navigate even when surrounded by explored areas
                print("[_action_proposer] No frontier waypoints, using explored area waypoints as fallback")
                longest = max(filtered, key=lambda x: x[0])
                longest_theta = longest[1]
                smallest_theta = longest[1]
                longest_ndx = filtered.index(longest)

                out.append([longest[0], longest[1], longest[2]])
                thetas.add(longest[1])

                for i in range(longest_ndx + 1, len(filtered)):
                    if filtered[i][1] - longest_theta > min_angle:
                        out.append([filtered[i][0], filtered[i][1], filtered[i][2]])
                        thetas.add(filtered[i][1])
                        longest_theta = filtered[i][1]

                for i in range(longest_ndx - 1, -1, -1):
                    if smallest_theta - filtered[i][1] > min_angle:
                        out.append([filtered[i][0], filtered[i][1], filtered[i][2]])
                        thetas.add(filtered[i][1])
                        smallest_theta = filtered[i][1]

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

    def _projection(self, a_final: list, obs: dict, chosen_action: int = None, available_turns: list = None):
        """
        Project actions onto RGB image.

        Parameters
        ----------
        a_final : list
            List of (mag, theta) tuples for MOVE actions
        obs : dict
            Observation dictionary
        chosen_action : int, optional
            The chosen action to highlight
        available_turns : list, optional
            List of available turn actions (e.g., [-1], [-2], or [-1, -2])
            If None, both turns are available
        """
        if available_turns is None:
            available_turns = [-1, -2]

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
            available_turns=available_turns,
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

    def _project_onto_image(self, a_final, rgb_image, intrinsics, T_cam_base, chosen_action=None, available_turns=None):
        """
        Project actions onto RGB image.

        Parameters
        ----------
        available_turns : list, optional
            List of available turn actions (e.g., [-1], [-2], or [-1, -2])
            If None, both turns are available
        """
        if available_turns is None:
            available_turns = [-1, -2]

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
            text_size = 1.8 * scale_factor  # Reduced font size to fit two digits
            text_thickness = max(1, math.ceil(3 * scale_factor))
            (tw, th), _ = cv2.getTextSize(text, font, text_size, text_thickness)

            circle_center = waypoint_px
            circle_radius = max(1, math.ceil(40 * scale_factor))  # Fixed radius for all circles

            if chosen_action is not None and action_name == chosen_action:
                cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)

            cv2.circle(rgb_image, circle_center, circle_radius, RED, max(1, math.ceil(2 * scale_factor)))

            text_position = (circle_center[0] - tw // 2, circle_center[1] + th // 2)
            cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)

        if len(a_final) > 0:
            print(f"[DEBUG] _project_onto_image: {len(a_final)} actions from _action_proposer, {projected_count} projected, {skipped_count} skipped by _can_project")

        # ---------- Add turn actions based on available_turns ----------
        # Only add turn actions that are in available_turns list
        for turn_action in available_turns:
            projected[turn_action] = turn_action

        # ---------- Draw turn buttons ----------
        # Draw left turn button (Action -1) if available
        if -1 in available_turns:
            text = '-1'
            text_size = 1.8 * scale_factor
            text_thickness = max(1, math.ceil(3 * scale_factor))
            (tw, th), _ = cv2.getTextSize(text, font, text_size, text_thickness)

            circle_center_left = (math.ceil(0.05 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2))
            circle_radius = max(1, math.ceil(40 * scale_factor))

            if chosen_action == -1:
                cv2.circle(rgb_image, circle_center_left, circle_radius, GREEN, -1)
            else:
                cv2.circle(rgb_image, circle_center_left, circle_radius, circle_color, -1)

            cv2.circle(rgb_image, circle_center_left, circle_radius, RED, max(1, math.ceil(2 * scale_factor)))

            text_position_left = (circle_center_left[0] - tw // 2, circle_center_left[1] + th // 2)
            cv2.putText(rgb_image, text, text_position_left, font, text_size, text_color, text_thickness)

        # Draw right turn button (Action -2) if available
        if -2 in available_turns:
            text = '-2'
            text_size = 1.8 * scale_factor
            text_thickness = max(1, math.ceil(3 * scale_factor))
            (tw, th), _ = cv2.getTextSize(text, font, text_size, text_thickness)

            circle_center_right = (math.ceil(0.90 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2))
            circle_radius = max(1, math.ceil(40 * scale_factor))

            if chosen_action == -2:
                cv2.circle(rgb_image, circle_center_right, circle_radius, GREEN, -1)
            else:
                cv2.circle(rgb_image, circle_center_right, circle_radius, circle_color, -1)

            cv2.circle(rgb_image, circle_center_right, circle_radius, RED, max(1, math.ceil(2 * scale_factor)))

            text_position_right = (circle_center_right[0] - tw // 2, circle_center_right[1] + th // 2)
            cv2.putText(rgb_image, text, text_position_right, font, text_size, text_color, text_thickness)

        return projected

    def _generate_bev_with_waypoints(self, base_to_odom: np.ndarray, waypoints: list = None) -> np.ndarray:
        """
        生成标注了waypoints的BEV地图（使用ObstacleMap）
        
        Parameters
        ----------
        base_to_odom : np.ndarray
            机器人base到odom的变换矩阵 (4x4)
        waypoints : list, optional
            Waypoint列表，每个元素为(r, theta)元组，theta相对于机器人base坐标系
            
        Returns
        -------
        np.ndarray
            标注后的BEV图像（RGB）
        """
        if self.obstacle_map is None:
            # If BEV map is disabled, return a blank image
            return np.zeros((200, 200, 3), dtype=np.uint8)
        
        # 提取机器人位置和朝向
        current_pos = base_to_odom[:3, 3]
        
        # 从旋转矩阵提取yaw角
        robot_yaw = np.arctan2(base_to_odom[1, 0], base_to_odom[0, 0])
        
        # 使用obstacle_map的可视化作为基础（不显示frontiers，不显示机器人）
        # 我们会在放大后再绘制机器人和其他标记
        bev = self.obstacle_map.visualize(robot_pos=None, show_frontiers=False)
        
        # 提高可视化分辨率（先放大地图，再绘制标记）
        scale_factor = 2
        bev = cv2.resize(bev, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        
        # 获取机器人在像素坐标系中的位置（放大后）
        robot_px = self.obstacle_map._xy_to_px(np.array([[current_pos[0], current_pos[1]]]))
        if len(robot_px) > 0:
            robot_px_x, robot_px_y = int(robot_px[0, 0] * scale_factor), int(robot_px[0, 1] * scale_factor)
        else:
            robot_px_x, robot_px_y = None, None

        # ========== 第一层：绘制箭头和FOV射线（在轨迹点下层） ==========
        if robot_px_x is not None and robot_px_y is not None:
            if 0 <= robot_px_x < bev.shape[1] and 0 <= robot_px_y < bev.shape[0]:
                # 绘制朝向箭头（红色，指示 RGB 相机的视野方向）
                arrow_length = 60  # 箭头长度（像素），加长以更清晰显示朝向
                # 注意：ObstacleMap 的 Y 轴是翻转的（图像坐标系），所以 sin 要取负
                arrow_end_x = int(robot_px_x + arrow_length * np.cos(robot_yaw))
                arrow_end_y = int(robot_px_y - arrow_length * np.sin(robot_yaw))  # Y 轴翻转
                # 红色箭头 (BGR: 0, 0, 255)
                cv2.arrowedLine(bev, (robot_px_x, robot_px_y), (arrow_end_x, arrow_end_y),
                               (0, 0, 255), 3, tipLength=0.3)

                # 绘制 FOV 边界射线（两条红色射线）
                if hasattr(self, 'fov'):
                    fov_half = np.deg2rad(self.fov / 2)

                    # 左边界射线
                    left_angle = robot_yaw - fov_half
                    # 右边界射线
                    right_angle = robot_yaw + fov_half

                    # 对每条边界射线进行 raycast,找到实际终点
                    for boundary_angle in [left_angle, right_angle]:
                        # Raycast 找到射线终点 (遇到障碍物/未探索区域/边界)
                        max_ray_dist = 20.0  # 最大射线长度 (米)

                        # 在 odom 坐标系中计算射线
                        ray_end_x = current_pos[0] + max_ray_dist * np.cos(boundary_angle)
                        ray_end_y = current_pos[1] + max_ray_dist * np.sin(boundary_angle)
                        ray_end_px = self.obstacle_map._xy_to_px(np.array([[ray_end_x, ray_end_y]])) * scale_factor

                        if len(ray_end_px) > 0:
                            end_px_x = int(ray_end_px[0, 0])
                            end_px_y = int(ray_end_px[0, 1])

                            # 使用 Bresenham 算法获取射线上的所有像素
                            line_pixels = self._bresenham_line(robot_px_x, robot_px_y, end_px_x, end_px_y)

                            # 找到第一个障碍物或未探索区域
                            actual_end_px = (end_px_x, end_px_y)  # 默认终点

                            for px_x, px_y in line_pixels:
                                # 转换回原始地图坐标 (缩小)
                                orig_px_x = px_x // scale_factor
                                orig_px_y = px_y // scale_factor

                                # 检查边界
                                if not (0 <= orig_px_x < self.obstacle_map.size and 0 <= orig_px_y < self.obstacle_map.size):
                                    actual_end_px = (px_x, px_y)
                                    break

                                # 检查是否是障碍物
                                if self.obstacle_map._obstacle_map[orig_px_y, orig_px_x]:
                                    actual_end_px = (px_x, px_y)
                                    break

                                # 检查是否是未探索区域
                                if not self.obstacle_map.explored_area[orig_px_y, orig_px_x]:
                                    actual_end_px = (px_x, px_y)
                                    break

                            # 绘制射线 (红色,较细)
                            cv2.line(bev, (robot_px_x, robot_px_y), actual_end_px, (0, 0, 255), 2)

        # ========== 第二层：绘制历史轨迹（在箭头上层） ==========
        if len(self.trajectory_history) > 1:
            # 转换所有历史位置到像素坐标（放大后）
            trajectory_array = np.array(self.trajectory_history)
            trajectory_px = self.obstacle_map._xy_to_px(trajectory_array) * scale_factor

            # 绘制轨迹点（浅蓝色圆圈用于历史点，红色圆圈用于当前点）
            for i in range(len(trajectory_px)):
                pt = (int(trajectory_px[i, 0]), int(trajectory_px[i, 1]))
                if 0 <= pt[0] < bev.shape[1] and 0 <= pt[1] < bev.shape[0]:
                    # 判断是否是当前位置
                    is_current = (self.current_trajectory_index is not None and
                                  i + 1 == self.current_trajectory_index)

                    if is_current:
                        # 当前位置：红色填充 (BGR: 0, 0, 255)
                        cv2.circle(bev, pt, 12, (0, 0, 255), -1)
                    else:
                        # 历史位置：浅蓝色填充 (BGR: 230, 216, 173)
                        cv2.circle(bev, pt, 12, (230, 216, 173), -1)

                    # 黑色轮廓
                    cv2.circle(bev, pt, 12, (0, 0, 0), 2)
                    # 在圆圈内添加黑色序号（居中）
                    text = str(i + 1)
                    font_scale = 0.5
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    text_x = pt[0] - text_width // 2
                    text_y = pt[1] + text_height // 2
                    cv2.putText(bev, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # ========== 第三层：标注waypoints（在轨迹点上层） ==========
        if waypoints is not None and len(waypoints) > 0:
            # 收集所有轨迹点的像素坐标（用于避免遮挡）
            trajectory_positions = []
            if len(self.trajectory_history) > 0:
                trajectory_array = np.array(self.trajectory_history)
                trajectory_px_for_check = self.obstacle_map._xy_to_px(trajectory_array) * scale_factor
                trajectory_positions = [(int(px[0]), int(px[1])) for px in trajectory_px_for_check]

            # 收集已绘制的waypoint位置（用于避免waypoint之间相互遮挡）
            drawn_waypoint_positions = []

            for i, (r, theta) in enumerate(waypoints, start=1):
                # 计算waypoint在odom坐标系中的位置
                absolute_theta = robot_yaw + theta
                waypoint_x = current_pos[0] + r * np.cos(absolute_theta)
                waypoint_y = current_pos[1] + r * np.sin(absolute_theta)
                
                # 转换为像素坐标（放大后）
                waypoint_px = self.obstacle_map._xy_to_px(np.array([[waypoint_x, waypoint_y]])) * scale_factor
                if len(waypoint_px) > 0:
                    px_x, px_y = int(waypoint_px[0, 0]), int(waypoint_px[0, 1])

                    # 检查是否在图像范围内
                    if 0 <= px_x < bev.shape[1] and 0 <= px_y < bev.shape[0]:
                        # 检查是否与轨迹点或其他waypoints太近，如果是则偏移位置
                        circle_radius = 12
                        min_distance = circle_radius * 2  # 两个圆圈的直径

                        # 检查与轨迹点的距离
                        too_close = False
                        for traj_pos in trajectory_positions:
                            dist = np.sqrt((px_x - traj_pos[0])**2 + (px_y - traj_pos[1])**2)
                            if dist < min_distance:
                                too_close = True
                                break

                        # 检查与已绘制waypoints的距离
                        if not too_close:
                            for wp_pos in drawn_waypoint_positions:
                                dist = np.sqrt((px_x - wp_pos[0])**2 + (px_y - wp_pos[1])**2)
                                if dist < min_distance:
                                    too_close = True
                                    break

                        # 如果太近，寻找附近的空白位置
                        final_px_x, final_px_y = px_x, px_y
                        if too_close:
                            # 在原位置周围搜索空白位置
                            # 搜索半径从30到60像素，每次增加10像素
                            # 每个半径上尝试8个方向
                            found = False
                            for search_radius in range(30, 70, 10):
                                if found:
                                    break
                                for angle_idx in range(8):
                                    angle = angle_idx * np.pi / 4  # 0, 45, 90, 135, 180, 225, 270, 315度
                                    test_x = int(px_x + search_radius * np.cos(angle))
                                    test_y = int(px_y + search_radius * np.sin(angle))

                                    # 检查是否在图像范围内
                                    if not (0 <= test_x < bev.shape[1] and 0 <= test_y < bev.shape[0]):
                                        continue

                                    # 检查是否在白色区域（可导航区域）
                                    # 注意：bev现在是BGR格式
                                    if not (bev[test_y, test_x, 0] > 200 and
                                            bev[test_y, test_x, 1] > 200 and
                                            bev[test_y, test_x, 2] > 200):
                                        continue

                                    # 检查与轨迹点的距离
                                    valid = True
                                    for traj_pos in trajectory_positions:
                                        dist = np.sqrt((test_x - traj_pos[0])**2 + (test_y - traj_pos[1])**2)
                                        if dist < min_distance:
                                            valid = False
                                            break

                                    if not valid:
                                        continue

                                    # 检查与已绘制waypoints的距离
                                    for wp_pos in drawn_waypoint_positions:
                                        dist = np.sqrt((test_x - wp_pos[0])**2 + (test_y - wp_pos[1])**2)
                                        if dist < min_distance:
                                            valid = False
                                            break

                                    if valid:
                                        final_px_x, final_px_y = test_x, test_y
                                        found = True
                                        print(f'[NavAgent] Waypoint {i} offset from ({px_x}, {px_y}) to ({final_px_x}, {final_px_y}) to avoid occlusion')
                                        break

                        # 记录这个waypoint的位置
                        drawn_waypoint_positions.append((final_px_x, final_px_y))

                        # 浅绿色填充 (BGR: 144, 238, 144)，半径12能容纳两位数
                        cv2.circle(bev, (final_px_x, final_px_y), 12, (144, 238, 144), -1)
                        # 黑色轮廓
                        cv2.circle(bev, (final_px_x, final_px_y), 12, (0, 0, 0), 2)
                        # 在圆圈内添加黑色序号（居中）
                        text = str(i)
                        font_scale = 0.5
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        text_x = final_px_x - text_width // 2
                        text_y = final_px_y + text_height // 2
                        cv2.putText(bev, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # 转换BGR到RGB，确保VLM能正确解析颜色
        bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)
        
        return bev

    def _nav(self, obs: dict, goal: str, iter: int, goal_description: str = ""):
        # Record current position to trajectory history
        current_pos = obs['base_to_odom_matrix'][:3, 3]
        current_xy = (current_pos[0], current_pos[1])

        # Check if current position overlaps with any existing trajectory point
        merge_threshold = 0.2  # meters
        merged_index = None

        if len(self.trajectory_history) > 0:
            # Search all trajectory points for overlap
            for i, hist_pos in enumerate(self.trajectory_history):
                distance = np.sqrt((current_xy[0] - hist_pos[0])**2 + (current_xy[1] - hist_pos[1])**2)
                if distance <= merge_threshold:
                    # Found overlapping point
                    merged_index = i + 1  # 1-based index
                    # Align current position to this trajectory point
                    current_xy = hist_pos
                    print(f'[NavAgent] Current position overlaps with trajectory point {merged_index} (distance={distance:.3f}m)')
                    break

        if merged_index is not None:
            # Overlapping with existing point, don't add new point
            self.current_trajectory_index = merged_index
        else:
            # No overlap, add new trajectory point
            self.trajectory_history.append(current_xy)
            self.current_trajectory_index = len(self.trajectory_history)
            print(f'[NavAgent] Added trajectory point {self.current_trajectory_index}: {current_xy}')

        # Update ObstacleMap with current observation FIRST
        # This ensures we have the latest obstacle information for raycast
        if self.obstacle_map is not None:
            self.obstacle_map.update_map(
                depth=obs['depth'],
                intrinsic=obs['intrinsic'],
                extrinsic=obs['extrinsic'],
                base_to_odom=obs['base_to_odom_matrix']
            )

            # Update colored BEV map with RGB projection
            self.obstacle_map.update_colored_bev(
                rgb=obs['rgb'],
                depth=obs['depth'],
                intrinsic=obs['intrinsic'],
                extrinsic=obs['extrinsic'],
                base_to_odom=obs['base_to_odom_matrix']
            )

        # Compute navigability using BEV-based raycast (more accurate)
        # This uses the updated obstacle map to avoid waypoints behind obstacles
        a_initial = self._navigability_from_bev(obs)
        a_final = self._action_proposer(a_initial, obs['base_to_odom_matrix'])

        # Filter turn actions based on last turn direction BEFORE projection
        # Determine which turn actions are available
        if self.last_turn_direction is not None:
            # Only allow the same turn direction as last time
            available_turns = [self.last_turn_direction]
            print(f'[NavAgent] Restricting to same turn direction: {self.last_turn_direction} (last turn was {self.last_turn_direction})')
        else:
            # Both turn directions are available
            available_turns = [-1, -2]

        # Start timing for projection (image annotation)
        t_projection_start = time.time()

        # Generate visualization without highlighting (for initial display)
        a_final_projected, rgb_vis = self._projection(a_final, obs, available_turns=available_turns)

        # Generate BEV map with waypoints (always generate for consistency)
        projected_waypoints = [k for k in a_final_projected.keys() if isinstance(k, tuple)]
        bev_map = self._generate_bev_with_waypoints(obs['base_to_odom_matrix'], waypoints=projected_waypoints)

        # Decide whether to send BEV to VLM based on enable_bev_map
        if self.cfg['enable_bev_map']:
            images = [rgb_vis, bev_map]  # Send both RGB and BEV to VLM
        else:
            images = [rgb_vis]  # Only send RGB to VLM (but BEV is still generated for navigation)

        # Check if only one turn action is available (no MOVE actions)
        # If so, skip VLM and return that turn action directly
        num_move_actions = len([k for k in a_final_projected.keys() if isinstance(k, tuple)])
        num_turn_actions = len([k for k in a_final_projected.keys() if k in [-1, -2]])

        if num_move_actions == 0 and num_turn_actions == 1:
            single_turn = list(a_final_projected.keys())[0]
            print(f'[NavAgent] No MOVE actions, only one turn available: {single_turn}, skipping VLM call')

            # Record action to history
            self.action_history.append(single_turn)
            if len(self.action_history) > self.max_action_history:
                self.action_history.pop(0)

            # Prepare timing info
            t_projection_end = time.time()
            projection_time = t_projection_end - t_projection_start

            timing_info = {
                'projection_time': float(projection_time),
                'vlm_inference_time': 0.0,
                'prompt': f'[SKIPPED VLM] No MOVE actions, only one turn available: {single_turn}',
                'is_keyframe': False
            }

            # Handle turn action
            if single_turn == -1:
                self.last_turn_direction = -1
                return f'[AUTO] Turn left (only option)', rgb_vis, ('turn', self.turn_angle_deg), timing_info, bev_map
            else:  # single_turn == -2
                self.last_turn_direction = -2
                return f'[AUTO] Turn right (only option)', rgb_vis, ('turn', -self.turn_angle_deg), timing_info, bev_map

        # Count available turn actions for prompt
        available_turns = [k for k in a_final_projected.keys() if k in [-1, -2]]

        # 构建统一的prompt（不再区分关键帧/非关键帧）
        # Count only move waypoints (tuples), not turn actions (integers)
        num_move_actions = len([k for k in a_final_projected.keys() if isinstance(k, tuple)])
        prompt = self._construct_prompt(
            num_actions=num_move_actions,
            iter=iter,
            num_turn_actions=num_turn_actions,
            available_turns=available_turns
        )
        
        t_projection_end = time.time()
        projection_time = t_projection_end - t_projection_start

        # Start timing for VLM inference
        t_vlm_start = time.time()

        # 调用VLM（使用VLM自己的历史管理）
        try:
            response = self.actionVLM.call_chat(
                history=self.cfg['vlm_history'],
                images=images,
                text_prompt=prompt
            )
            print(f'[NavAgent] VLM Response: {response}')
        except Exception as e:
            print(f'[NavAgent] ERROR: VLM call failed: {e}')
            import traceback
            traceback.print_exc()
            # Return with error info but still save rgb_vis
            return None, rgb_vis, None, {
                'projection_time': float(projection_time),
                'vlm_inference_time': 0.0,
                'error': str(e)
            }

        t_vlm_end = time.time()
        vlm_inference_time = t_vlm_end - t_vlm_start

        print(f'[NavAgent] Timing - Projection: {projection_time:.3f}s, VLM inference: {vlm_inference_time:.3f}s')
        print(f'[NavAgent] Prompt length: {len(prompt)} chars')
        print(f'Response: {response}')

        rev = {v: k for k, v in a_final_projected.items()}
        try:
            response_dict = self._eval_response(response)
            action_number = int(response_dict['action'])

            # Record action to history
            self.action_history.append(action_number)
            if len(self.action_history) > self.max_action_history:
                self.action_history.pop(0)  # Keep only last N actions

            # Re-generate visualization with chosen action highlighted in GREEN
            _, rgb_vis_final = self._projection(
                a_final,
                obs,
                chosen_action=action_number,
                available_turns=available_turns
            )

            # Prepare timing information (include full conversation context for debugging)
            full_prompt = self._format_full_messages_as_text(prompt, num_images=len(images))
            timing_info = {
                'projection_time': float(projection_time),
                'vlm_inference_time': float(vlm_inference_time),
                'prompt': full_prompt,
                'is_keyframe': False  # No longer using keyframe mode
            }

            # Handle rotation actions
            if action_number == -1:
                # Turn left: update last turn direction
                self.last_turn_direction = -1
                return response, rgb_vis_final, ('turn', self.turn_angle_deg), timing_info, bev_map
            elif action_number == -2:
                # Turn right: update last turn direction
                self.last_turn_direction = -2
                return response, rgb_vis_final, ('turn', -self.turn_angle_deg), timing_info, bev_map
            else:
                # Forward movement action: reset last turn direction
                self.last_turn_direction = None
                action = rev.get(action_number)
                return response, rgb_vis_final, action, timing_info, bev_map
        except (IndexError, KeyError, TypeError, ValueError) as e:
            print(f'[NavAgent] ERROR: Failed to parse VLM response: {e}')
            print(f'[NavAgent] Raw response: {response}')
            print(f'[NavAgent] Available actions: {list(a_final_projected.keys())}')
            # Return response and rgb_vis even if parsing failed
            return response, rgb_vis, None, {
                'projection_time': float(projection_time),
                'vlm_inference_time': float(vlm_inference_time),
                'error': f'Parse error: {str(e)}'
            }, bev_map

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        if response is None or not response:
            return {}
        try:
            eval_resp = ast.literal_eval(response[response.rindex('{'):response.rindex('}') + 1])
            if isinstance(eval_resp, dict):
                return eval_resp
            else:
                raise ValueError
        except (ValueError, SyntaxError, AttributeError):
            return {}


if __name__ == "__main__":
    import pickle
    data = pickle.load(open('/home/liujy/nav_ws/src/llm_nav/captured_data/test_data.pkl', 'rb'))
    agent = NavAgent()
    agent._nav(data, 'banana', iter=1)
