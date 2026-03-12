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
        
        # Frontier occlusion filter (simple 3x3 depth check)
        self.cfg.setdefault('frontier_occlusion_filter', True)
        self.cfg.setdefault('frontier_occlusion_margin', 0.15)  # meters
        self.cfg.setdefault('frontier_projection_height', 0.1)  # meters above ground
        
        # Fallback: force backtracking after too many consecutive turns
        self.cfg.setdefault('fallback_turn_threshold', 12)
        self.cfg.setdefault('fallback_max_history_candidates', 8)
        self.cfg.setdefault('fallback_min_history_distance', 0.8)  # meters
        
        # RGB waypoint arrow length (pixels at 1080p; scaled by image height)
        self.cfg.setdefault('rgb_waypoint_arrow_length_px', 84)

        self.turn_angle_deg = float(self.cfg['turn_angle_deg'])
        self.clip_dist = self.cfg['clip_dist']
        self.prompt_mode = self.cfg['prompt_mode']
        
        # Initialize goal attributes (will be set properly in reset())
        self.goal = None
        self.goal_description = ''

        # Initialize ObstacleMap (always create for navigation logic)
        # This is a local map that gets updated each frame with current observation
        self.obstacle_map = ObstacleMap(
            min_height=self.cfg['obstacle_height_min'],
            max_height=self.cfg['obstacle_height_max'],
            agent_radius=0.18,
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

    def _format_full_messages_as_text(
        self,
        current_prompt: str,
        num_images: int = 2,
        image_labels: list = None
    ):
        """
        Format the complete messages structure (as sent to VLM) into readable text.
        Images are replaced with placeholders like [IMAGE: RGB], [IMAGE: BEV].

        Parameters
        ----------
        current_prompt : str
            The current iteration's complete text prompt (includes template + history + observation)
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

        # 3. Current message (includes iteration template + history + current observation)
        lines.append("[CURRENT MESSAGE]")
        lines.append("-" * 80)
        lines.append(current_prompt)

        # Add image placeholders
        lines.append("")
        if image_labels is None:
            image_labels = ["[IMAGE: RGB]", "[IMAGE: BEV]"]
        for i in range(num_images):
            label = image_labels[i] if i < len(image_labels) else f"[IMAGE {i+1}]"
            lines.append(label)

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _construct_current_observation(
        self,
        num_actions: int,
        iter: int,
        num_turn_actions: int = 2,
        available_turns: list = None,
        fallback_note: str = "",
    ):
        """
        Construct current iteration observation (position, available actions).
        This is specific to each iteration.

        Parameters
        ----------
        num_actions : int
            Number of available MOVE actions
        iter : int
            Current iteration number
        num_turn_actions : int
            Number of available TURN actions (0, 1, or 2)
        available_turns : list
            Available turn actions (e.g., [-1], [-2], or [-1, -2])

        Returns
        -------
        str
            Current observation text
        """
        if available_turns is None:
            available_turns = [-1, -2]

        # Add current robot position information
        if self.current_trajectory_index is not None:
            current_position_text = f"Current robot position: Trajectory point {self.current_trajectory_index} (marked with red circle in BEV map)"
        else:
            current_position_text = "Current robot position: Starting position (no trajectory point yet)"

        # Build turn actions text based on available turns
        if num_turn_actions == 0:
            turn_actions_text = "- TURN: None available"
        elif num_turn_actions == 1:
            if -1 in available_turns:
                turn_actions_text = f"- TURN: -1 (left {self.turn_angle_deg}°) only"
            else:
                turn_actions_text = f"- TURN: -2 (right {self.turn_angle_deg}°) only"
        else:
            turn_actions_text = f"- TURN: -1 (left {self.turn_angle_deg}°) or -2 (right {self.turn_angle_deg}°)"

        # Use prompt template from JSON
        if self.cfg['enable_bev_map']:
            template_lines = list(self.prompts['iteration_prompt_template'])
        else:
            template_lines = list(self.prompts['iteration_prompt_template_no_bev'])

        if fallback_note:
            inserted = False
            for idx, line in enumerate(template_lines):
                if "Respond exactly in this format" in line:
                    template_lines.insert(idx, "")
                    template_lines.insert(idx + 1, fallback_note)
                    inserted = True
                    break
            if not inserted:
                template_lines.append("")
                template_lines.append(fallback_note)

        prompt = '\n'.join(template_lines).format(
            iter=iter,
            num_actions=num_actions if num_actions > 0 else 0,
            turn_actions=turn_actions_text,
            current_position=current_position_text
        )

        return prompt

    def _build_custom_history_messages(self):
        """
        Build custom history messages in trajectory-point-based format.
        Format: "At trajectory point X, turned Y" or "Moved from point X to point Y"

        Returns
        -------
        list
            List of message dicts for VLM
        """
        messages = []
        history_limit = self.cfg['vlm_history']

        # Get recent history (last N records)
        recent_history = self.simplified_history[-history_limit:] if history_limit > 0 else []

        for record in recent_history:
            if record['action_type'] == 'turn':
                # TURN action
                direction = "right" if record['total_rotation'] > 0 else "left"
                angle = abs(record['total_rotation'])

                if record['iter_start'] == record['iter_end']:
                    # Single turn
                    iter_text = f"Iteration {record['iter_start']}:"
                    action_text = f"At trajectory point {record['trajectory_point']}, turned {direction} {angle:.1f}°"
                else:
                    # Multiple turns merged
                    iter_text = f"Iteration {record['iter_start']}-{record['iter_end']}:"
                    action_text = f"At trajectory point {record['trajectory_point']}, turned {direction} {angle:.1f}° (net rotation after {record['turn_count']} turns)"

                messages.append({"role": "user", "content": iter_text})
                messages.append({"role": "assistant", "content": action_text})

            elif record['action_type'] == 'move':
                # MOVE action
                if record['to_trajectory_point'] is not None:
                    # MOVE completed (we know the destination point)
                    iter_text = f"Iteration {record['iter_start']}:"
                    action_text = f"Moved from trajectory point {record['from_trajectory_point']} to trajectory point {record['to_trajectory_point']}"

                    messages.append({"role": "user", "content": iter_text})
                    messages.append({"role": "assistant", "content": action_text})
                # else: MOVE not yet completed, skip (shouldn't appear in history)

        return messages

    def _construct_prompt(
        self,
        num_actions: int,
        iter: int,
        num_turn_actions: int = 2,
        available_turns: list = None,
        fallback_note: str = "",
    ):
        """
        Construct prompt using simplified structure (just returns current observation).
        The full context (system + initial + template + history) is built elsewhere.

        Parameters
        ----------
        num_actions : int
            Number of available MOVE actions
        iter : int
            Current iteration number
        num_turn_actions : int
            Number of available TURN actions (0, 1, or 2)
        available_turns : list
            Available turn actions (e.g., [-1], [-2], or [-1, -2])

        Returns
        -------
        str
            Current observation prompt
        """
        return self._construct_current_observation(
            num_actions,
            iter,
            num_turn_actions,
            available_turns,
            fallback_note=fallback_note,
        )

    def _construct_fallback_observation(self, iter: int, candidate_traj_ids: list):
        """
        Construct fallback prompt for BEV-only backtracking.
        In fallback mode, the model must choose a historical trajectory point ID.
        """
        ids_sorted = sorted([int(x) for x in candidate_traj_ids])
        ids_text = ", ".join(str(x) for x in ids_sorted)
        current_position_text = (
            f"Current robot position: Trajectory point {self.current_trajectory_index} (red circle in BEV map)"
            if self.current_trajectory_index is not None
            else "Current robot position: Starting position"
        )

        template_lines = self.prompts.get("iteration_prompt_template_fallback_bev")
        if not template_lines:
            template_lines = [
                "--- Iteration {iter} (FALLBACK MODE) ---",
                "",
                "{current_position}",
                "",
                "Input for this round:",
                "- Only one image is provided: BEV map.",
                "- RGB image is NOT provided in fallback mode.",
                "",
                "Available actions:",
                "- MOVE: choose exactly one trajectory point ID from [{candidate_ids}]",
                "- TURN: None available",
                "",
                "Respond exactly in this format:",
                "**BEV Observation:** {{Describe current position, obstacles, and why the selected trajectory point is appropriate.}}",
                "**Action and Reason:** {{Chosen trajectory point ID and concise reason.}}",
                "**Decision:** {{'action': <trajectory_point_id>}}",
                "",
            ]

        return "\n".join(template_lines).format(
            iter=iter,
            current_position=current_position_text,
            candidate_ids=ids_text
        )

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
        self.trajectory_yaw_history = []  # List of yaw (radians) for each trajectory point
        self.current_trajectory_index = None  # Index of current trajectory point (1-based, None if no trajectory yet)

        # Initialize action history for rotation strategy
        self.action_history = []  # List of recent actions (integers: -2, -1, 1, 2, ...)
        self.max_action_history = 5  # Keep last 5 actions
        self.last_turn_direction = None  # Track last turn direction: -1 (left), -2 (right), or None
        self.consecutive_turn_count = 0  # Consecutive turn actions without a move
        self.fallback_active = False

        # Simplified history for VLM context (iter_id + obs + action)
        self.simplified_history = []  # Format: [{"iter": 1, "action": -2}, ...]
        self.iteration_counter = 0  # Track iteration number

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

    def _is_frontier_occluded_simple(
        self,
        frontier_xy_odom: np.ndarray,
        depth_image: np.ndarray,
        intrinsic: np.ndarray,
        T_cam_odom: np.ndarray,
    ) -> bool:
        """
        Simple occlusion check:
        1) Project frontier 3D point to image
        2) Compute mean depth in 3x3 window
        3) If observed depth is significantly closer than projected depth,
           treat the frontier as occluded.
        """
        z_frontier = float(self.cfg.get('frontier_projection_height', 0.1))
        p_odom = np.array(
            [float(frontier_xy_odom[0]), float(frontier_xy_odom[1]), z_frontier, 1.0],
            dtype=np.float64
        )

        p_cam = T_cam_odom.astype(np.float64) @ p_odom
        z_pred = float(p_cam[2])
        if z_pred <= 1e-6:
            # Behind camera / invalid projection depth -> don't filter by occlusion here
            return False

        uv = point_to_pixel(np.array([p_odom[:3]], dtype=np.float64), intrinsic, T_cam_odom)
        if uv is None or len(uv) == 0:
            return False

        u = int(round(float(uv[0][0])))
        v = int(round(float(uv[0][1])))

        H, W = depth_image.shape
        if not (0 <= u < W and 0 <= v < H):
            return False

        x0 = max(0, u - 1)
        x1 = min(W, u + 2)
        y0 = max(0, v - 1)
        y1 = min(H, v + 2)
        depth_patch = depth_image[y0:y1, x0:x1]

        valid_mask = np.isfinite(depth_patch) & (depth_patch > 1e-3)
        if not np.any(valid_mask):
            return False

        depth_mean = float(np.mean(depth_patch[valid_mask]))
        margin = float(self.cfg.get('frontier_occlusion_margin', 0.15))
        return (depth_mean + margin) < z_pred

    def _build_fallback_history_candidates(self, obs: dict):
        """
        Build MOVE candidates from historical trajectory points for fallback mode.
        Returns
        -------
        tuple[list, list]
            (candidate_actions, trajectory_point_indices)
            candidate_actions format: [(r, theta, normal_odom), ...]
            trajectory_point_indices: [point_idx_1based, ...]
        """
        T_odom_base = obs['base_to_odom_matrix']
        robot_xy = T_odom_base[:2, 3]
        robot_yaw = float(np.arctan2(T_odom_base[1, 0], T_odom_base[0, 0]))

        min_dist = float(self.cfg.get('fallback_min_history_distance', 0.8))
        max_candidates = int(self.cfg.get('fallback_max_history_candidates', 8))

        candidates = []
        seen_xy = set()
        for idx, hist_xy in enumerate(self.trajectory_history, start=1):
            if self.current_trajectory_index is not None and idx == self.current_trajectory_index:
                continue

            key = (round(float(hist_xy[0]), 2), round(float(hist_xy[1]), 2))
            if key in seen_xy:
                continue
            seen_xy.add(key)

            dx = float(hist_xy[0] - robot_xy[0])
            dy = float(hist_xy[1] - robot_xy[1])
            r = float(np.hypot(dx, dy))
            if r < min_dist:
                continue

            theta_odom = float(np.arctan2(dy, dx))
            theta_base = float(np.arctan2(np.sin(theta_odom - robot_yaw), np.cos(theta_odom - robot_yaw)))
            normal_odom = np.array([dx / r, dy / r], dtype=np.float64)
            candidates.append((idx, r, theta_base, normal_odom))

        if len(candidates) == 0:
            return [], []

        # Prefer farther historical points to break local spinning loops.
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:max_candidates]

        actions = [(r, theta, normal) for _, r, theta, normal in candidates]
        traj_ids = [idx for idx, _, _, _ in candidates]
        return actions, traj_ids

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
        Compute navigable directions using frontier detection from BEV obstacle map.

        Parameters
        ----------
        obs : dict
            Observation dictionary
        angle_range : tuple, optional
            (theta_min, theta_max) to limit angle search range (in radians)
            If None, use full 360 degrees (not used with frontier detection)
        clip_dist_override : float, optional
            Override self.clip_dist for distance filtering
            If None, use self.clip_dist

        Returns
        -------
        a_initial : list
            List of (r, theta, normal) tuples where:
            - r: distance from robot to frontier (meters)
            - theta: angle from robot to frontier (radians, in base frame)
            - normal: [nx, ny] unit normal in odom frame (for goal orientation)
        """
        if self.obstacle_map is None:
            # Fallback to RGB-based method if BEV map is disabled
            return self._navigability(obs, angle_range, clip_dist_override)

        T_odom_base = obs['base_to_odom_matrix']

        if self.init_pos is None:
            self.init_pos = T_odom_base[:3, 3].copy()

        # Calculate FOV for _action_proposer (needed for filtering)
        if 'intrinsic' in obs and 'rgb' in obs:
            K = obs['intrinsic']
            rgb_image = obs['rgb']
            self.fov = self.cal_fov(K, rgb_image.shape[1])
        elif not hasattr(self, 'fov'):
            # Default FOV if not available
            self.fov = 90.0

        # Get robot position and orientation in odom frame
        robot_pos_odom = T_odom_base[:3, 3]
        robot_yaw = np.arctan2(T_odom_base[1, 0], T_odom_base[0, 0])

        # Get frontiers from obstacle map
        frontiers_odom = self.obstacle_map.frontiers  # Nx2 array
        frontier_normals_px = self.obstacle_map._frontier_normals  # Nx2 array

        print(f"[_navigability_from_bev] FOV: {self.fov:.1f}°")
        print(f"[_navigability_from_bev] Robot yaw: {np.rad2deg(robot_yaw):.1f}°")
        print(f"[_navigability_from_bev] Detected {len(frontiers_odom)} frontiers")

        if len(frontiers_odom) == 0:
            print("[_navigability_from_bev] No frontiers detected")
            return []

        # Convert frontiers to (r, theta, normal) format
        a_initial = []

        filtered_too_close = 0
        filtered_occluded = 0
        use_occlusion_filter = bool(self.cfg.get('frontier_occlusion_filter', True))
        depth_image = obs.get('depth', None)
        intrinsic = obs.get('intrinsic', None)
        T_cam_odom = obs.get('extrinsic', None)

        for i, (frontier_xy, normal_px) in enumerate(zip(frontiers_odom, frontier_normals_px), start=1):
            # Compute relative position in odom frame
            dx = frontier_xy[0] - robot_pos_odom[0]
            dy = frontier_xy[1] - robot_pos_odom[1]

            # Distance
            r = np.sqrt(dx**2 + dy**2)

            # Skip if too close (to avoid collision)
            if r < 0.5:
                filtered_too_close += 1
                print(f"[_navigability_from_bev] Frontier {i}: r={r:.2f}m - FILTERED (too close < 0.5m)")
                continue

            # Simple occlusion filter based on 3x3 depth mean
            if use_occlusion_filter and depth_image is not None and intrinsic is not None and T_cam_odom is not None:
                if self._is_frontier_occluded_simple(frontier_xy, depth_image, intrinsic, T_cam_odom):
                    filtered_occluded += 1
                    print(f"[_navigability_from_bev] Frontier {i}: r={r:.2f}m - FILTERED (occluded by depth)")
                    continue

            # Angle in odom frame
            theta_odom = np.arctan2(dy, dx)

            # Convert to base frame (relative to robot's current yaw)
            theta_base = theta_odom - robot_yaw

            # Normalize angle to [-pi, pi]
            theta_base = np.arctan2(np.sin(theta_base), np.cos(theta_base))

            # Convert normal from pixel space to odom space
            # normal_px is [nx_px, ny_px] in BEV pixel coordinates
            # In _px_to_xy: X stays same, Y is negated
            # For pointing to unexplored: negate X
            normal_odom = np.array([-normal_px[0], normal_px[1]])

            # Normalize to unit vector
            normal_norm = np.linalg.norm(normal_odom)
            if normal_norm > 1e-6:
                normal_odom = normal_odom / normal_norm
            else:
                # Fallback: use direction from robot to frontier
                normal_odom = np.array([dx, dy]) / r

            a_initial.append((r, theta_base, normal_odom))

        if filtered_too_close > 0:
            print(f"[_navigability_from_bev] Filtered: {filtered_too_close} too close")
        if filtered_occluded > 0:
            print(f"[_navigability_from_bev] Filtered: {filtered_occluded} occluded")

        print(f"[_navigability_from_bev] Generated {len(a_initial)} waypoints from frontiers")

        return a_initial

    # _raycast_bev() method removed - replaced by frontier detection
    # The new frontier-based approach provides more stable and accurate waypoints

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
        normals = {}  # Store normals for each theta

        # Handle both (mag, theta) and (mag, theta, normal) formats
        for item in a_initial:
            if len(item) == 3:
                mag, theta, normal = item
                unique.setdefault(theta, []).append(mag)
                normals[theta] = normal  # Store normal for this theta
            else:
                mag, theta = item
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
        # Return with normals if available
        result = []
        for mag, theta, _ in out:
            if theta in normals:
                result.append((mag, theta, normals[theta]))
            else:
                result.append((mag, theta))
        return result

    def _projection(
        self,
        a_final: list,
        obs: dict,
        chosen_action: int = None,
        available_turns: list = None,
        force_include_unprojectable: bool = False,
    ):
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
            T_odom_base=T_odom_base,
            chosen_action=chosen_action,
            available_turns=available_turns,
            force_include_unprojectable=force_include_unprojectable,
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

    def _project_onto_image(
        self,
        a_final,
        rgb_image,
        intrinsics,
        T_cam_base,
        T_odom_base,
        chosen_action=None,
        available_turns=None,
        force_include_unprojectable: bool = False,
    ):
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
        fixed_arrow_len_px = max(
            1,
            int(round(float(self.cfg.get('rgb_waypoint_arrow_length_px', 42)) * scale_factor))
        )

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

        for waypoint in a_final:
            # Handle both (r, theta) and (r, theta, normal) formats
            # Normal might be numpy array or tuple
            if len(waypoint) == 3:
                r_i, theta_i, normal_odom = waypoint
                # Convert tuple back to numpy array if needed
                if isinstance(normal_odom, tuple):
                    normal_odom = np.array(normal_odom)
            else:
                r_i, theta_i = waypoint
                normal_odom = None

            # 先检查原始距离是否可以投影（用于过滤）
            end_px = self._can_project(r_i, theta_i, rgb_image.shape[:2], intrinsics, T_cam_base)
            if end_px is None:
                if not force_include_unprojectable:
                    skipped_count += 1
                    skipped_reasons.append(f"r={r_i:.2f}, theta={np.degrees(theta_i):.1f}° - original distance not projectable")
                    continue
            
            # Project the real waypoint position (no distance scaling)
            H, W = rgb_image.shape[:2]
            p_base = np.array([r_i * np.cos(theta_i), r_i * np.sin(theta_i), 0.0], dtype=np.float64)
            uv = point_to_pixel(p_base, intrinsics, T_cam_base)
            if uv is None:
                if not force_include_unprojectable:
                    skipped_count += 1
                    skipped_reasons.append(f"r={r_i:.2f}, theta={np.degrees(theta_i):.1f}° - waypoint projection failed")
                    continue
                waypoint_px = None
            else:
                waypoint_px = (int(round(uv[0][0])), int(round(uv[0][1])))
            
            # 确保waypoint在图像范围内
            if waypoint_px is not None and not (0 <= waypoint_px[0] < W and 0 <= waypoint_px[1] < H):
                if not force_include_unprojectable:
                    skipped_count += 1
                    skipped_reasons.append(f"r={r_i:.2f}, theta={np.degrees(theta_i):.1f}° - waypoint out of bounds: ({waypoint_px[0]}, {waypoint_px[1]})")
                    continue
                waypoint_px = None
            
            projected_count += 1
            action_name = len(projected) + 1
            # Store the complete waypoint tuple (with normal if available)
            # Convert normal to tuple for hashability
            if normal_odom is not None:
                projected[(r_i, theta_i, tuple(normal_odom))] = action_name
            else:
                projected[(r_i, theta_i)] = action_name
            if waypoint_px is None:
                print(f"[DEBUG] Projected action {action_name}: r={r_i:.2f}m, theta={np.degrees(theta_i):.1f}°, px=(not_in_rgb)")
            else:
                print(f"[DEBUG] Projected action {action_name}: r={r_i:.2f}m, theta={np.degrees(theta_i):.1f}°, px=({waypoint_px[0]}, {waypoint_px[1]})")

            if waypoint_px is None:
                continue

            # Draw annotation in order: arrow -> circle -> index text.
            text = str(action_name)
            text_size = 1.8 * scale_factor  # Reduced font size to fit two digits
            text_thickness = max(1, math.ceil(3 * scale_factor))
            (tw, th), _ = cv2.getTextSize(text, font, text_size, text_thickness)

            circle_center = waypoint_px
            circle_radius = max(1, math.ceil(40 * scale_factor))  # Fixed radius for all circles

            # Draw normal arrow first, then draw circle/text on top to keep numbers readable.
            if normal_odom is not None:
                # Draw the RGB arrow as the projection of the BEV arrow segment.
                # BEV uses: start_offset_px=12, end_offset_px=12+30 on a 2x-upscaled map.
                bev_scale_factor = 2.0
                start_offset_px_bev = 12.0
                end_offset_px_bev = 42.0
                start_offset_m = start_offset_px_bev / (bev_scale_factor * self.bev_scale)
                end_offset_m = end_offset_px_bev / (bev_scale_factor * self.bev_scale)

                # Waypoint position in base frame
                waypoint_x_base = r_i * np.cos(theta_i)
                waypoint_y_base = r_i * np.sin(theta_i)

                # Convert normal from odom frame to base frame
                # T_odom_base rotates base to odom, so we need inverse rotation for odom->base
                robot_yaw = np.arctan2(T_odom_base[1, 0], T_odom_base[0, 0])
                cos_yaw = np.cos(-robot_yaw)
                sin_yaw = np.sin(-robot_yaw)
                normal_base = np.array([
                    cos_yaw * normal_odom[0] - sin_yaw * normal_odom[1],
                    sin_yaw * normal_odom[0] + cos_yaw * normal_odom[1]
                ], dtype=np.float64)

                normal_norm = np.linalg.norm(normal_base)
                if normal_norm > 1e-6:
                    normal_base = normal_base / normal_norm

                    # Construct BEV-equivalent arrow start/end in base frame
                    point_start_3d = np.array([
                        waypoint_x_base + normal_base[0] * start_offset_m,
                        waypoint_y_base + normal_base[1] * start_offset_m,
                        0.1
                    ], dtype=np.float64)
                    point_end_3d = np.array([
                        waypoint_x_base + normal_base[0] * end_offset_m,
                        waypoint_y_base + normal_base[1] * end_offset_m,
                        0.1
                    ], dtype=np.float64)

                    # Project both points and draw arrow from waypoint circle center
                    uv_start = point_to_pixel(point_start_3d, intrinsics, T_cam_base)
                    uv_end = point_to_pixel(point_end_3d, intrinsics, T_cam_base)
                    if uv_start is not None and uv_end is not None:
                        u_start = int(round(uv_start[0][0]))
                        v_start = int(round(uv_start[0][1]))
                        u_end = int(round(uv_end[0][0]))
                        v_end = int(round(uv_end[0][1]))

                        if (0 <= u_start < W and 0 <= v_start < H and
                                0 <= u_end < W and 0 <= v_end < H):
                            du = u_end - u_start
                            dv = v_end - v_start
                            norm_uv = np.hypot(du, dv)
                            if norm_uv > 1e-6:
                                dir_u = du / norm_uv
                                dir_v = dv / norm_uv
                                arrow_end = (
                                    int(round(circle_center[0] + dir_u * fixed_arrow_len_px)),
                                    int(round(circle_center[1] + dir_v * fixed_arrow_len_px))
                                )

                                cv2.arrowedLine(
                                    rgb_image,
                                    circle_center,
                                    arrow_end,
                                    BLACK,
                                    5,
                                    tipLength=0.2
                                )
                                cv2.arrowedLine(
                                    rgb_image,
                                    circle_center,
                                    arrow_end,
                                    WHITE,
                                    3,
                                    tipLength=0.2
                                )

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

    def _generate_bev_with_waypoints(
        self,
        base_to_odom: np.ndarray,
        waypoints: list = None,
        fallback_mode: bool = False,
        show_waypoints: bool = True
    ) -> np.ndarray:
        """
        生成标注了waypoints的BEV地图（使用ObstacleMap）
        
        Parameters
        ----------
        base_to_odom : np.ndarray
            机器人base到odom的变换矩阵 (4x4)
        waypoints : list, optional
            Waypoint列表，每个元素为(r, theta)元组，theta相对于机器人base坐标系
        show_waypoints : bool, optional
            是否绘制waypoint层。fallback模式下应为False，只保留正常BEV+轨迹标注。
            
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
        if len(self.trajectory_history) >= 1:  # Changed from > 1 to >= 1 to show first position
            # 转换所有历史位置到像素坐标（放大后）
            trajectory_array = np.array(self.trajectory_history)
            trajectory_px = self.obstacle_map._xy_to_px(trajectory_array) * scale_factor
            history_point_color = (230, 216, 173)
            history_arrow_length_px = 30

            # 绘制轨迹点（浅蓝色圆圈用于历史点，红色圆圈用于当前点）
            for i in range(len(trajectory_px)):
                pt = (int(trajectory_px[i, 0]), int(trajectory_px[i, 1]))
                if 0 <= pt[0] < bev.shape[1] and 0 <= pt[1] < bev.shape[0]:
                    # 判断是否是当前位置
                    is_current = (self.current_trajectory_index is not None and
                                  i + 1 == self.current_trajectory_index)

                    # 历史轨迹点方向箭头：使用该轨迹点记录的真实yaw
                    # 先画箭头，再画圆圈和数字，避免遮挡序号
                    if (not is_current) and (i < len(self.trajectory_yaw_history)):
                        pt_yaw = float(self.trajectory_yaw_history[i])
                        arrow_end = (
                            int(round(pt[0] + np.cos(pt_yaw) * history_arrow_length_px)),
                            int(round(pt[1] - np.sin(pt_yaw) * history_arrow_length_px))
                        )

                        # 黑色描边 + 同色箭头，风格与frontier箭头一致
                        cv2.arrowedLine(bev, pt, arrow_end, (0, 0, 0), 4, tipLength=0.2)
                        cv2.arrowedLine(bev, pt, arrow_end, history_point_color, 2, tipLength=0.2)

                    if is_current:
                        # 当前位置：红色填充 (BGR: 0, 0, 255)
                        cv2.circle(bev, pt, 12, (0, 0, 255), -1)
                    else:
                        # 历史位置：浅蓝色填充 (BGR: 230, 216, 173)
                        cv2.circle(bev, pt, 12, history_point_color, -1)

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
        if show_waypoints and waypoints is not None and len(waypoints) > 0:
            # 收集所有轨迹点的像素坐标（用于避免遮挡）
            trajectory_positions = []
            if len(self.trajectory_history) > 0:
                trajectory_array = np.array(self.trajectory_history)
                trajectory_px_for_check = self.obstacle_map._xy_to_px(trajectory_array) * scale_factor
                trajectory_positions = [(int(px[0]), int(px[1])) for px in trajectory_px_for_check]

            # 收集已绘制的waypoint位置（用于避免waypoint之间相互遮挡）
            drawn_waypoint_positions = []

            for i, waypoint in enumerate(waypoints, start=1):
                # Handle both (r, theta) and (r, theta, normal) formats
                # Normal might be numpy array or tuple
                if len(waypoint) == 3:
                    r, theta, normal_odom = waypoint
                    # Convert tuple back to numpy array if needed
                    if isinstance(normal_odom, tuple):
                        normal_odom = np.array(normal_odom)
                else:
                    r, theta = waypoint
                    normal_odom = None

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

                        # 如果太近，寻找附近的空白位置（fallback模式下不偏移，保持历史点精确位置）
                        final_px_x, final_px_y = px_x, px_y
                        if too_close and (not fallback_mode):
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

                        # Draw normal arrow first, then draw circle/text on top.
                        if normal_odom is not None:
                            arrow_length_px = 30  # pixels at 2x scale
                            # Normal is already in odom frame, pointing to unexplored
                            # Convert to pixel direction (note Y flip in image coordinates)
                            normal_px_dir = np.array([normal_odom[0], -normal_odom[1]])
                            normal_norm = np.linalg.norm(normal_px_dir)
                            if normal_norm > 1e-6:
                                normal_px_dir = normal_px_dir / normal_norm

                                # Arrow from circle center to end point
                                start_px = np.array([final_px_x, final_px_y], dtype=int)
                                end_px = start_px + (normal_px_dir * arrow_length_px).astype(int)

                                # Draw outlined arrow for visibility; center will be covered by circle
                                cv2.arrowedLine(bev, tuple(start_px), tuple(end_px),
                                               (0, 0, 0), 4, tipLength=0.2)
                                cv2.arrowedLine(bev, tuple(start_px), tuple(end_px),
                                               (144, 238, 144), 2, tipLength=0.2)  # Light green arrow
                                print(f'[BEV] Drew arrow for waypoint {i} at ({final_px_x}, {final_px_y}), normal={normal_odom}')
                            else:
                                print(f'[BEV] Waypoint {i} has zero-length normal, skipping arrow')
                        else:
                            print(f'[BEV] Waypoint {i} has no normal vector, skipping arrow')

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
        
        bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)
        
        return bev

    def update_observation_only(self, obs: dict, is_keyframe: bool = True):
        """
        Update map state from current observation.

        When `is_keyframe` is True, also maintain trajectory keypoints with the
        same merge logic used by the full navigation loop.
        """
        if is_keyframe:
            current_pos = obs['base_to_odom_matrix'][:3, 3]
            current_xy = (float(current_pos[0]), float(current_pos[1]))
            current_yaw = float(np.arctan2(obs['base_to_odom_matrix'][1, 0], obs['base_to_odom_matrix'][0, 0]))

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
                # Refresh yaw for this trajectory point to reflect current orientation
                if merged_index - 1 < len(self.trajectory_yaw_history):
                    self.trajectory_yaw_history[merged_index - 1] = current_yaw
            else:
                # No overlap, add new point
                self.trajectory_history.append(current_xy)
                self.trajectory_yaw_history.append(current_yaw)
                self.current_trajectory_index = len(self.trajectory_history)
                print(f'[NavAgent] Added trajectory point {self.current_trajectory_index}: {current_xy}')

        # Always update BEV obstacle map every frame
        if self.obstacle_map is not None:
            self.obstacle_map.update_map(
                depth=obs['depth'],
                intrinsic=obs['intrinsic'],
                extrinsic=obs['extrinsic'],
                base_to_odom=obs['base_to_odom_matrix']
            )

    def _nav(
        self,
        obs: dict,
        goal: str,
        iter: int,
        goal_description: str = "",
        observation_already_updated: bool = False
    ):
        # Keep legacy behavior: full navigation loop updates keyframe trajectory and map.
        if not observation_already_updated:
            self.update_observation_only(obs, is_keyframe=True)

        # Compute navigation candidates.
        # Normal mode: frontier-based waypoints.
        # Fallback mode: historical trajectory points (after too many consecutive turns).
        fallback_turn_threshold = int(self.cfg.get('fallback_turn_threshold', 12))
        fallback_triggered = self.consecutive_turn_count >= fallback_turn_threshold
        fallback_mode = False
        fallback_traj_ids = []

        if fallback_triggered:
            fallback_actions, fallback_traj_ids = self._build_fallback_history_candidates(obs)
            if len(fallback_actions) > 0:
                fallback_mode = True
                self.fallback_active = True
                a_final = fallback_actions
                print(
                    f"[NavAgent] FALLBACK ACTIVE: consecutive turns={self.consecutive_turn_count}, "
                    f"history candidates={len(a_final)}, trajectory points={fallback_traj_ids}"
                )
            else:
                print(
                    f"[NavAgent] FALLBACK TRIGGERED but no valid history candidates "
                    f"(consecutive turns={self.consecutive_turn_count}), using normal frontier mode"
                )
                a_initial = self._navigability_from_bev(obs)
                a_final = self._action_proposer(a_initial, obs['base_to_odom_matrix'])
        else:
            a_initial = self._navigability_from_bev(obs)
            a_final = self._action_proposer(a_initial, obs['base_to_odom_matrix'])

        # Filter turn actions based on last turn direction BEFORE projection
        # Determine which turn actions are available
        if fallback_mode:
            # Force a MOVE choice in fallback mode.
            available_turns = []
        elif self.last_turn_direction is not None:
            # Only allow the same turn direction as last time
            available_turns = [self.last_turn_direction]
            print(f'[NavAgent] Restricting to same turn direction: {self.last_turn_direction} (last turn was {self.last_turn_direction})')
        else:
            # Both turn directions are available
            available_turns = [-1, -2]

        # Start timing for projection (image annotation)
        t_projection_start = time.time()

        if fallback_mode:
            # In fallback, action number must be trajectory point ID, and VLM only sees BEV.
            a_final_projected = {}
            for waypoint, traj_id in zip(a_final, fallback_traj_ids):
                if len(waypoint) == 3:
                    r_i, theta_i, normal_odom = waypoint
                    if isinstance(normal_odom, np.ndarray):
                        normal_key = tuple(float(x) for x in normal_odom.tolist())
                    else:
                        normal_key = tuple(float(x) for x in normal_odom)
                    waypoint_key = (float(r_i), float(theta_i), normal_key)
                else:
                    r_i, theta_i = waypoint
                    waypoint_key = (float(r_i), float(theta_i))
                a_final_projected[waypoint_key] = int(traj_id)

            rgb_vis = obs['rgb'].copy()
            # Do not overlay fallback waypoints; model should choose from blue trajectory points in BEV.
            bev_map = self._generate_bev_with_waypoints(
                obs['base_to_odom_matrix'],
                waypoints=None,
                fallback_mode=False,
                show_waypoints=False
            )
            images = [bev_map]
            image_labels = ["[IMAGE: BEV]"]
        else:
            # Generate visualization without highlighting (for initial display)
            a_final_projected, rgb_vis = self._projection(
                a_final,
                obs,
                available_turns=available_turns,
                force_include_unprojectable=False
            )

            # Generate BEV map with waypoints (always generate for consistency)
            projected_waypoints = [k for k in a_final_projected.keys() if isinstance(k, tuple)]
            bev_map = self._generate_bev_with_waypoints(
                obs['base_to_odom_matrix'],
                waypoints=projected_waypoints,
                fallback_mode=False,
                show_waypoints=True
            )

            # Decide whether to send BEV to VLM based on enable_bev_map
            if self.cfg['enable_bev_map']:
                images = [rgb_vis, bev_map]  # Send both RGB and BEV to VLM
                image_labels = ["[IMAGE: RGB]", "[IMAGE: BEV]"]
            else:
                images = [rgb_vis]  # Only send RGB to VLM (but BEV is still generated for navigation)
                image_labels = ["[IMAGE: RGB]"]

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
            self.consecutive_turn_count += 1

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

        # Trajectory point merge/add has been handled at the start of _nav.
        # Here we only resolve pending MOVE destination using current trajectory index.
        if len(self.simplified_history) > 0:
            last_record = self.simplified_history[-1]
            if last_record['action_type'] == 'move' and last_record['to_trajectory_point'] is None:
                from_point = last_record.get('from_trajectory_point')
                to_point = self.current_trajectory_index
                if to_point is not None and from_point is not None and to_point != from_point:
                    last_record['to_trajectory_point'] = to_point
                    print(f"[NavAgent] Updated MOVE record: from point {from_point} to point {to_point}")

        # Increment iteration counter
        self.iteration_counter += 1

        # Build current observation prompt
        if fallback_mode:
            current_observation = self._construct_fallback_observation(
                iter=self.iteration_counter,
                candidate_traj_ids=fallback_traj_ids
            )
        else:
            current_observation = self._construct_prompt(
                num_actions=num_move_actions,
                iter=self.iteration_counter,
                num_turn_actions=num_turn_actions,
                available_turns=available_turns,
                fallback_note=""
            )

        # Build custom history messages (simplified: iter + action only)
        custom_history = self._build_custom_history_messages()

        # Build complete text prompt (history + current_observation)
        prompt_parts = []

        if len(custom_history) > 0:
            prompt_parts.append("=" * 80)
            prompt_parts.append("CONVERSATION HISTORY")
            prompt_parts.append("=" * 80)
            for msg in custom_history:
                if msg['role'] == 'user':
                    prompt_parts.append(msg['content'])
                else:
                    prompt_parts.append(msg['content'])
            prompt_parts.append("")

        prompt_parts.append(current_observation)

        full_text_prompt = "\n".join(prompt_parts)

        t_projection_end = time.time()
        projection_time = t_projection_end - t_projection_start

        # Format full prompt for logging BEFORE calling VLM
        full_prompt = self._format_full_messages_as_text(
            full_text_prompt,
            num_images=len(images),
            image_labels=image_labels
        )

        # Start timing for VLM inference
        t_vlm_start = time.time()

        # Call VLM with custom history (simplified structure)
        try:
            response = self.actionVLM.call_chat_with_custom_history(
                custom_history=custom_history,
                images=images,
                text_prompt=full_text_prompt
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
        print(f'[NavAgent] Prompt length: {len(full_text_prompt)} chars')
        print(f'Response: {response}')

        rev = {v: k for k, v in a_final_projected.items()}
        try:
            response_dict = self._eval_response(response)
            action_number = int(response_dict['action'])

            # Record action to history
            self.action_history.append(action_number)
            if len(self.action_history) > self.max_action_history:
                self.action_history.pop(0)  # Keep only last N actions

            # Save to simplified history with trajectory point information
            if action_number in [-1, -2]:
                # TURN action
                rotation = self.turn_angle_deg if action_number == -2 else -self.turn_angle_deg
                self.consecutive_turn_count += 1

                # Check if we should merge with the last TURN record
                if (len(self.simplified_history) > 0 and
                    self.simplified_history[-1]['action_type'] == 'turn' and
                    self.simplified_history[-1]['trajectory_point'] == self.current_trajectory_index):
                    # Merge with previous TURN record
                    self.simplified_history[-1]['total_rotation'] += rotation
                    self.simplified_history[-1]['iter_end'] = self.iteration_counter
                    self.simplified_history[-1]['turn_count'] += 1
                    print(f"[NavAgent] Merged TURN: point {self.current_trajectory_index}, total rotation {self.simplified_history[-1]['total_rotation']:.1f}°, iterations {self.simplified_history[-1]['iter_start']}-{self.simplified_history[-1]['iter_end']}")
                else:
                    # Create new TURN record
                    self.simplified_history.append({
                        'iter_start': self.iteration_counter,
                        'iter_end': self.iteration_counter,
                        'trajectory_point': self.current_trajectory_index,
                        'action_type': 'turn',
                        'total_rotation': rotation,
                        'turn_count': 1
                    })
                    print(f"[NavAgent] New TURN record: point {self.current_trajectory_index}, rotation {rotation:.1f}°, iteration {self.iteration_counter}")
            else:
                # MOVE action
                self.consecutive_turn_count = 0
                self.fallback_active = False
                self.simplified_history.append({
                    'iter_start': self.iteration_counter,
                    'iter_end': self.iteration_counter,
                    'from_trajectory_point': self.current_trajectory_index,
                    'to_trajectory_point': None,  # Will be updated in next iteration
                    'action_type': 'move'
                })
                print(f"[NavAgent] New MOVE record: from point {self.current_trajectory_index}, iteration {self.iteration_counter}")

            # Re-generate visualization with chosen action highlighted in GREEN
            if fallback_mode:
                rgb_vis_final = rgb_vis
                print(f"[NavAgent] FALLBACK selected trajectory point ID: {action_number}")
            else:
                _, rgb_vis_final = self._projection(
                    a_final,
                    obs,
                    chosen_action=action_number,
                    available_turns=available_turns,
                    force_include_unprojectable=False
                )

            # Prepare timing information (include full conversation context for debugging)
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
