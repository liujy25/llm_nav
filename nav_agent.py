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
        self.cfg.setdefault('vlm_history_length', 3)  # VLM keeps recent N conversation rounds
        self.cfg.setdefault('turn_angle_deg', 30.0)  # Turn left/right angle in degrees

        self.turn_angle_deg = float(self.cfg['turn_angle_deg'])
        self.clip_dist = self.cfg['clip_dist']
        self.vlm_history_length = self.cfg['vlm_history_length']
        
        self._initialize_vlms()
        self.reset()

    def _initialize_vlms(self):
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You are only allowed to search for the goal object in the room you are in now. You cannot go to other rooms."
            "You cannot move through doors. "
        )
        self.actionVLM = OpenAIVLM(model="/data/sea_disk0/liujy/models/Qwen/Qwen3-VL-8B-Instruct/", system_instruction=system_instruction)

    def _construct_prompt(self, num_actions: int = 0, iter: int = 0):
        """
        Build short iteration prompt for current observation.
        The full task briefing is already in VLM's history from reset().
        
        Parameters
        ----------
        num_actions : int
            Number of available actions in current observation
        iter : int
            Current iteration number
            
        Returns
        -------
        str
            Short iteration prompt
        """        
        
        # REASONING MODE
        iteration_prompt = (
            f"--- Iteration {iter} ---\n"
            f"Current observation: {num_actions} available actions shown.\n"
            f"\n"
            f"Now answer the 4 questions from the task briefing:\n"
            f"1. What do you see?\n"
            f"2. Which direction should you go?\n"
            f"3. Which action number?\n"
            f"4. {{'action': <number>}}"
        )
        # ACTION ONLY MODE
        # iteration_prompt = (
        #     f"--- Iteration {iter} ---\n"
        #     f"Current observation: {num_actions} available MOVE actions shown (numbered 1..{num_actions}).\n"
        #     f"Additionally, there are always two TURN actions:\n"
        #     f"- Action -1: TURN LEFT by {self.turn_angle_deg:.0f} degrees in place\n"
        #     f"- Action -2: TURN RIGHT by {self.turn_angle_deg:.0f} degrees in place\n\n"
        #     f"Now only answer the selected action number: {{'action': <number>}}"
        # )
        return iteration_prompt

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
   - Red arrows in images show potential forward movement directions
   - Each action is labeled with a POSITIVE number in a white circle
   - The number indicates the destination waypoint you would move to
   - These actions move the robot forward to new positions

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

RESPONSE FORMAT:
For each observation, you MUST think about ALL the following questions in order:

1. What do you see? Any leads on finding the {goal.upper()}?
   (Answer: Describe what you observe in the current view)

2. Based on what you've seen before (I will show you your history), which direction should you go?
   (Answer: Explain your navigation strategy based on history and current observation)

3. Which action number achieves that direction best?
   (Answer: State the specific action number and why you chose it)

4. Final decision:
   {{'action': <action_number>}}

You MUST think about the answers to questions 1-3 before giving the final decision in step 4.
Remember these instructions throughout the navigation episode. I will show you observations with iteration IDs, and you should apply these rules consistently to make navigation decisions.
"""
        return initial_prompt

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
        thresh = 0.3
        height_map = depth_to_height(depth_image, intrinsics, T_cam_world)
        navigability_mask = np.abs(height_map - (0.0 - 0.04)) < thresh
        navigability_mask = navigability_mask.astype(bool)
        return navigability_mask

    def _get_radial_distance(
        self,
        start_pxl: tuple,
        theta_i: float,
        navigability_mask: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: np.ndarray,
        T_cam_base: np.ndarray,
    ):
        H, W = navigability_mask.shape

        agent_point_base = np.array([self.clip_dist * np.cos(theta_i), self.clip_dist * np.sin(theta_i), 0.0], dtype=np.float64)
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

    def _navigability(self, obs: dict):
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

        sensor_range = np.deg2rad(self.fov / 2) * 1.5
        all_thetas = np.linspace(-sensor_range, sensor_range, int(self.cfg['num_theta']))

        start = agent_frame_to_image_coords([0.0, 0.0, 0.0], K, T_cam_base)

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(
                start, theta_i, navigability_mask, depth_image, K, T_cam_base
            )
            if r_i is not None:
                # 你原逻辑：voxel里写 r/2，action里仍返回 r（后面再 /2）
                self._update_voxel(r_i, theta_i, T_odom_base, clip_dist=self.clip_dist, clip_frac=0.66)
                a_initial.append((r_i, theta_i))

        return a_initial

    def _action_proposer(self, a_initial: list, T_odom_base: np.ndarray):
        min_angle = self.fov / 360
        explore_bias = 4
        clip_frac = 0.66
        clip_mag = self.clip_dist

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
            arrowData.append([clip_frac * mag, theta, score < 3])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        # Adjust filter threshold relative to clip_dist
        # filter_thresh should be proportional to clip_dist to avoid filtering out all actions
        # Original: 0.75 is about 0.75/2.0 = 37.5% of typical clip_dist
        # Use a relative threshold: at least 30% of clip_dist after clip_frac
        filter_thresh = max(0.5, clip_frac * self.clip_dist * 0.3)  # At least 0.5m or 30% of clip_dist*clip_frac
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

                out.append([min(longest[0], clip_mag), longest[1], longest[2]])
                thetas.add(longest[1])

                for i in range(longest_ndx + 1, len(f)):
                    if f[i][1] - longest_theta > (min_angle * 0.9):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        longest_theta = f[i][1]

                for i in range(longest_ndx - 1, -1, -1):
                    if smallest_theta - f[i][1] > (min_angle * 0.9):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        smallest_theta = f[i][1]

                for r_i, theta_i, e_i in filtered:
                    if len(thetas) == 0:
                        out.append((min(r_i, clip_mag), theta_i, e_i))
                        thetas.add(theta_i)
                    else:
                        if theta_i not in thetas and min([abs(theta_i - t) for t in thetas]) > min_angle * explore_bias:
                            out.append((min(r_i, clip_mag), theta_i, e_i))
                            thetas.add(theta_i)

        if len(out) == 0:
            longest = max(filtered, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = filtered.index(longest)
            out.append([min(longest[0], clip_mag), longest[1], longest[2]])

            for i in range(longest_ndx + 1, len(filtered)):
                if filtered[i][1] - longest_theta > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    longest_theta = filtered[i][1]

            for i in range(longest_ndx - 1, -1, -1):
                if smallest_theta - filtered[i][1] > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
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

        # ---------- 先画动作箭头 1..N ----------
        projected_count = 0
        skipped_count = 0
        for (r_i, theta_i) in a_final:
            end_px = self._can_project(r_i, theta_i, rgb_image.shape[:2], intrinsics, T_cam_base)
            if end_px is None:
                skipped_count += 1
                continue
            projected_count += 1

            action_name = len(projected) + 1
            projected[(r_i, theta_i)] = action_name

            cv2.arrowedLine(
                rgb_image, tuple(start_px), tuple(end_px),
                RED, max(1, math.ceil(5 * scale_factor)), tipLength=0.0
            )

            text = str(action_name)
            text_size = 2.4 * scale_factor
            text_thickness = max(1, math.ceil(3 * scale_factor))
            (tw, th), _ = cv2.getTextSize(text, font, text_size, text_thickness)

            circle_center = (end_px[0], end_px[1])
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
        a_initial = self._navigability(obs)
        a_final = self._action_proposer(a_initial, obs['base_to_odom_matrix'])

        # Start timing for projection (image annotation)
        t_projection_start = time.time()
        
        # Generate visualization without highlighting (for initial display)
        a_final_projected, rgb_vis = self._projection(a_final, obs)

        # Only send current image (VLM manages history internally)
        images = [rgb_vis]

        # Construct short iteration prompt (full task briefing already in VLM history)
        prompt = self._construct_prompt(
            num_actions=len(a_final_projected), 
            iter=iter
        )
        
        t_projection_end = time.time()
        projection_time = t_projection_end - t_projection_start

        # Start timing for VLM inference
        t_vlm_start = time.time()
        
        # Call VLM with current image and short prompt (VLM manages conversation history)
        response = self.actionVLM.call_chat(self.vlm_history_length, images, prompt)
        
        t_vlm_end = time.time()
        vlm_inference_time = t_vlm_end - t_vlm_start
        
        print(f'[NavAgent] VLM history length: {self.vlm_history_length} rounds')
        print(f'[NavAgent] Timing - Projection: {projection_time:.3f}s, VLM inference: {vlm_inference_time:.3f}s')
        print(f'[NavAgent] Prompt length: {len(prompt)} chars (short iteration prompt)')
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
            
            # Note: No need to save to memory anymore - VLM manages its own history
            
            # Prepare timing information
            timing_info = {
                'projection_time': float(projection_time),
                'vlm_inference_time': float(vlm_inference_time)
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
