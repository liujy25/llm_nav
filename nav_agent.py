#!/usr/bin/env python3
import math
import time
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
        
        # VLM configuration defaults
        self.cfg.setdefault('vlm_model', '/data/sea_disk0/liujy/models/Qwen/Qwen3-VL-8B-Instruct/')
        self.cfg.setdefault('vlm_api_key', 'EMPTY')
        self.cfg.setdefault('vlm_base_url', 'http://10.15.89.71:34134/v1/')
        self.cfg.setdefault('vlm_timeout', 10)
        
        # BEV map configuration
        self.cfg.setdefault('enable_bev_map', True)  # Enable BEV map visualization
        self.cfg.setdefault('bev_map_size', 200)  # BEV map size (200x200 for 20m x 20m)
        self.cfg.setdefault('bev_pixels_per_meter', 10)  # 10 pixels per meter

        self.turn_angle_deg = float(self.cfg['turn_angle_deg'])
        self.clip_dist = self.cfg['clip_dist']
        self.prompt_mode = self.cfg['prompt_mode']
        
        # Initialize goal attributes (will be set properly in reset())
        self.goal = None
        self.goal_description = ''
        
        # Initialize ObstacleMap for BEV visualization
        if self.cfg['enable_bev_map']:
            self.obstacle_map = ObstacleMap(
                min_height=self.cfg['obstacle_height_min'],
                max_height=self.cfg['obstacle_height_max'],
                agent_radius=0.3,
                area_thresh=0.3,
                size=self.cfg['bev_map_size'],
                pixels_per_meter=self.cfg['bev_pixels_per_meter'],
            )
            self.bev_scale = self.cfg['bev_pixels_per_meter']
        else:
            self.obstacle_map = None
            self.bev_scale = None
        
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

    def _construct_prompt(self, num_actions: int, iter: int):
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
            
        Returns
        -------
        str
            prompt文本
        """
        
        prompt = f"""--- Iteration {iter} ---

You are provided with TWO images:
1. RGB Image (First): Your current first-person view showing {num_actions} available MOVE waypoints (numbered 1..{num_actions})
2. BEV Map (Second): Bird's-eye view showing explored areas (gray) and unexplored areas (green), with your current position marked

Current observation: {num_actions} available MOVE waypoints shown (numbered 1..{num_actions}).
Additionally, there are always two TURN actions:
- Action -1: TURN LEFT by {self.turn_angle_deg:.0f} degrees in place
- Action -2: TURN RIGHT by {self.turn_angle_deg:.0f} degrees in place

================================================================================

Please analyze the situation and answer the following questions. 
In your response, you should: first, give your answer of all the questions, 
then provide your final decision in this exact format: {{'action': <action_number>}}.

Question 1: Current Observation
What do you see in the RGB image (first image) and BEV map (second image)?
- RGB: Navigable areas (floor, corridors, open spaces), obstacles (walls, furniture, objects)
- BEV: Explored vs unexplored regions, your current position and orientation
- Potential target objects related to "{self.goal}"
- Unexplored directions that might lead to the target

Question 2: Strategic Reasoning
Based on your observation and the BEV map:
- Which areas are most promising for finding "{self.goal}"?
- Should you explore new areas or search more carefully in explored regions?
- Which direction offers the best balance between exploration and target search?

Question 3: Action Selection
Which action number achieves your goal best?
Consider all available actions (1 to {num_actions}, -1, -2).

Question 4: Action Justification
Why did you choose this specific action?
- How does it help you find "{self.goal}"?
- What do you expect to discover or achieve?
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
        
        # 绘制历史轨迹（在放大后的地图上）
        if len(self.trajectory_history) > 1:
            # 转换所有历史位置到像素坐标（放大后）
            trajectory_array = np.array(self.trajectory_history)
            trajectory_px = self.obstacle_map._xy_to_px(trajectory_array) * scale_factor
            
            # 绘制轨迹线（黑色细线）
            for i in range(len(trajectory_px) - 1):
                pt1 = (int(trajectory_px[i, 0]), int(trajectory_px[i, 1]))
                pt2 = (int(trajectory_px[i+1, 0]), int(trajectory_px[i+1, 1]))
                # 检查点是否在图像范围内
                if (0 <= pt1[0] < bev.shape[1] and 0 <= pt1[1] < bev.shape[0] and
                    0 <= pt2[0] < bev.shape[1] and 0 <= pt2[1] < bev.shape[0]):
                    cv2.line(bev, pt1, pt2, (0, 0, 0), 2)  # 黑色线，放大后用2像素
            
            # 绘制轨迹点（浅蓝色圆圈，带黑色轮廓和序号）
            for i in range(len(trajectory_px)):
                pt = (int(trajectory_px[i, 0]), int(trajectory_px[i, 1]))
                if 0 <= pt[0] < bev.shape[1] and 0 <= pt[1] < bev.shape[0]:
                    # 浅蓝色填充 (BGR: 230, 216, 173)，半径12能容纳两位数
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
        
        # 绘制机器人位置（浅蓝色圆圈，和轨迹点一样，显示序号）
        if robot_px_x is not None and robot_px_y is not None:
            if 0 <= robot_px_x < bev.shape[1] and 0 <= robot_px_y < bev.shape[0]:
                # 浅蓝色填充 (BGR: 230, 216, 173)，和轨迹点一样
                cv2.circle(bev, (robot_px_x, robot_px_y), 12, (230, 216, 173), -1)
                # 黑色轮廓
                cv2.circle(bev, (robot_px_x, robot_px_y), 12, (0, 0, 0), 2)
                # 在圆圈内添加黑色序号（当前位置是最后一个轨迹点）
                text = str(len(self.trajectory_history))
                font_scale = 0.5
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = robot_px_x - text_width // 2
                text_y = robot_px_y + text_height // 2
                cv2.putText(bev, text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # 标注waypoints（带黑色轮廓和序号）
        if waypoints is not None and len(waypoints) > 0:
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
                        # 浅绿色填充 (BGR: 144, 238, 144)，半径12能容纳两位数
                        cv2.circle(bev, (px_x, px_y), 12, (144, 238, 144), -1)
                        # 黑色轮廓
                        cv2.circle(bev, (px_x, px_y), 12, (0, 0, 0), 2)
                        # 在圆圈内添加黑色序号（居中）
                        text = str(i)
                        font_scale = 0.5
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        text_x = px_x - text_width // 2
                        text_y = px_y + text_height // 2
                        cv2.putText(bev, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # 转换BGR到RGB，确保VLM能正确解析颜色
        bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)
        
        return bev

    def _nav(self, obs: dict, goal: str, iter: int, goal_description: str = ""):
        # Record current position to trajectory history
        current_pos = obs['base_to_odom_matrix'][:3, 3]
        self.trajectory_history.append((current_pos[0], current_pos[1]))
        
        # Compute navigability (no angle_range or clip_dist_override needed)
        a_initial = self._navigability(obs)
        a_final = self._action_proposer(a_initial, obs['base_to_odom_matrix'])

        # Update ObstacleMap with current observation
        if self.obstacle_map is not None:
            self.obstacle_map.update_map(
                depth=obs['depth'],
                intrinsic=obs['intrinsic'],
                extrinsic=obs['extrinsic'],
                base_to_odom=obs['base_to_odom_matrix']
            )

        # Start timing for projection (image annotation)
        t_projection_start = time.time()
        
        # Generate visualization without highlighting (for initial display)
        a_final_projected, rgb_vis = self._projection(a_final, obs)

        # Generate BEV map with waypoints
        if self.obstacle_map is not None:
            bev_map = self._generate_bev_with_waypoints(obs['base_to_odom_matrix'], waypoints=a_final)
            images = [rgb_vis, bev_map]  # Send both RGB and BEV to VLM
        else:
            images = [rgb_vis]  # Only RGB if BEV is disabled

        # 构建统一的prompt（不再区分关键帧/非关键帧）
        prompt = self._construct_prompt(
            num_actions=len(a_final_projected),
            iter=iter
        )
        
        t_projection_end = time.time()
        projection_time = t_projection_end - t_projection_start

        # Start timing for VLM inference
        t_vlm_start = time.time()
        
        # 调用VLM（使用VLM自己的历史管理）
        response = self.actionVLM.call_chat(
            images=images,
            text_prompt=prompt
        )
        
        t_vlm_end = time.time()
        vlm_inference_time = t_vlm_end - t_vlm_start
        
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
                'prompt': prompt,
                'is_keyframe': False  # No longer using keyframe mode
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
