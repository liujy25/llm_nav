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
        
        # Navigability configuration (obstacle height range)
        self.cfg.setdefault('obstacle_height_min', 0.15)  # Minimum obstacle height (m)
        self.cfg.setdefault('obstacle_height_max', 2.0)   # Maximum obstacle height (m)
        
        # Function Call configuration
        self.cfg.setdefault('max_fc_iterations', 5)  # Maximum Function Call iterations
        
        # VLM configuration defaults
        self.cfg.setdefault('vlm_model', '/data/sea_disk0/liujy/models/Qwen/Qwen3-VL-8B-Instruct/')
        self.cfg.setdefault('vlm_api_key', 'EMPTY')
        self.cfg.setdefault('vlm_base_url', 'http://10.15.89.71:34134/v1/')
        self.cfg.setdefault('vlm_timeout', 10)

        self.clip_dist = self.cfg['clip_dist']
        
        # Initialize goal attributes (will be set properly in reset())
        self.goal = None
        self.goal_description = ''
        
        # Waypoint管理
        self.waypoint_registry = {}  # {wp_id: {'pos': np.array, 'rgb': np.array, 'iter': int, 'yaw': float}}
        self.next_wp_id = 1
        
        self._initialize_vlms()
        self.reset()

    def _initialize_vlms(self):
        system_instruction = (
            "You are a robot navigation assistant with RGB camera and BEV map. "
            "You can view historical waypoints and make navigation decisions. "
            "You cannot go through doors or to other rooms."
        )
        self.actionVLM = OpenAIVLM(
            model=self.cfg['vlm_model'],
            system_instruction=system_instruction,
            api_key=self.cfg['vlm_api_key'],
            base_url=self.cfg['vlm_base_url'],
            timeout=self.cfg['vlm_timeout']
        )

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
        self.step_ndx = 0
        self.init_pos = None
        
        # Store goal information as instance attributes
        self.goal = goal
        self.goal_description = goal_description
        
        # Reset waypoint registry
        self.waypoint_registry = {}
        self.next_wp_id = 1
        
        # Build initial prompt with goal information if provided
        if goal:
            initial_prompt = self._build_initial_prompt(goal, goal_description)
            self.actionVLM.reset(initial_prompt=initial_prompt)
            print(f"[NavAgent] Reset with goal='{goal}', VLM initialized with Function Call mode")
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
    
    def _generate_bev_with_waypoints(self, current_pos: np.ndarray) -> np.ndarray:
        """
        生成标注了历史waypoints的BEV地图
        
        Parameters
        ----------
        current_pos : np.ndarray
            当前机器人位置（odom坐标系）
            
        Returns
        -------
        np.ndarray
            标注后的BEV图像（RGB）
        """
        # 复制explored_map作为基础
        bev = self.explored_map.copy()
        
        # 标注当前位置（红色大圆）
        current_grid = self._global_to_grid(current_pos)
        cv2.circle(bev, current_grid, 30, RED, -1)
        cv2.putText(bev, "YOU", (current_grid[0]-20, current_grid[1]-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
        
        # 标注历史waypoints（蓝色圆+编号）
        for wp_id, wp_data in self.waypoint_registry.items():
            wp_pos = wp_data['pos']
            wp_grid = self._global_to_grid(wp_pos)
            
            # 画圆
            cv2.circle(bev, wp_grid, 20, (255, 0, 0), -1)  # 蓝色
            cv2.circle(bev, wp_grid, 20, WHITE, 2)  # 白色边框
            
            # 画编号
            text = str(wp_id)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(bev, text, (wp_grid[0]-tw//2, wp_grid[1]+th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        
        # 添加图例
        legend_y = 50
        cv2.putText(bev, "BEV Map with Waypoints", (20, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2)
        cv2.putText(bev, f"Total Waypoints: {len(self.waypoint_registry)}", (20, legend_y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        
        return bev
    
    def _annotate_frontiers(self, rgb: np.ndarray, a_final: list, K, T_cam_base):
        """
        在当前RGB图像上标注可选择的frontiers
        
        Parameters
        ----------
        rgb : np.ndarray
            当前RGB图像
        a_final : list
            候选动作列表 [(r, theta), ...]
        K : np.ndarray
            相机内参
        T_cam_base : np.ndarray
            cam->base变换矩阵
            
        Returns
        -------
        rgb_annotated : np.ndarray
            标注后的RGB图像
        frontier_map : dict
            frontier映射 {frontier_id: (r, theta)}
        """
        rgb_annotated = rgb.copy()
        frontier_map = {}
        
        # 使用字母标识frontiers（A, B, C...）
        for idx, (r, theta) in enumerate(a_final):
            if idx >= 26:  # 最多26个frontiers
                break
            
            label = chr(65 + idx)  # A-Z
            
            # 投影waypoint到图像（80%距离处）
            r_waypoint = r * 0.8
            p_base = np.array([r_waypoint * np.cos(theta), r_waypoint * np.sin(theta), 0.0])
            uv = point_to_pixel(p_base, K, T_cam_base)
            
            if uv is None:
                continue
            
            pixel_pos = (int(round(uv[0][0])), int(round(uv[0][1])))
            H, W = rgb.shape[:2]
            
            if not (0 <= pixel_pos[0] < W and 0 <= pixel_pos[1] < H):
                continue
            
            frontier_map[label] = (r, theta)
            
            # 画绿色圆圈+字母
            cv2.circle(rgb_annotated, pixel_pos, 25, GREEN, 3)
            cv2.putText(rgb_annotated, label, (pixel_pos[0]-10, pixel_pos[1]+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 3)
        
        # 添加标题
        cv2.putText(rgb_annotated, "Current View - Select Frontier", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        
        return rgb_annotated, frontier_map
    
    def _get_function_definitions(self) -> list:
        """返回Function定义"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_waypoint_rgb",
                    "description": "查看历史waypoint的第一人称RGB视角，帮助回忆该位置的环境",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "wp_id": {
                                "type": "integer",
                                "description": f"waypoint编号，可选范围：{list(self.waypoint_registry.keys()) if self.waypoint_registry else 'none'}"
                            }
                        },
                        "required": ["wp_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "go_to",
                    "description": "导航回到某个历史waypoint位置",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "wp_id": {
                                "type": "integer",
                                "description": "目标waypoint编号"
                            }
                        },
                        "required": ["wp_id"]
                    }
                }
            }
        ]
    
    def _handle_function_call(self, tool_name: str, args: dict) -> dict:
        """
        处理Function Call
        
        Parameters
        ----------
        tool_name : str
            函数名称
        args : dict
            函数参数
            
        Returns
        -------
        dict
            {'success': bool, 'result': any, 'error': str}
        """
        if tool_name == 'get_waypoint_rgb':
            wp_id = args.get('wp_id')
            
            if wp_id not in self.waypoint_registry:
                return {
                    'success': False,
                    'result': None,
                    'error': f"Waypoint {wp_id} not found. Available: {list(self.waypoint_registry.keys())}"
                }
            
            return {
                'success': True,
                'result': self.waypoint_registry[wp_id]['rgb'],
                'error': None
            }
        
        elif tool_name == 'go_to':
            wp_id = args.get('wp_id')
            
            if wp_id not in self.waypoint_registry:
                return {
                    'success': False,
                    'result': None,
                    'error': f"Waypoint {wp_id} not found"
                }
            
            wp_data = self.waypoint_registry[wp_id]
            return {
                'success': True,
                'result': {
                    'action_type': 'go_to_waypoint',
                    'target_pos': wp_data['pos'],
                    'target_yaw': wp_data['yaw']
                },
                'error': None
            }
        
        return {
            'success': False,
            'result': None,
            'error': f"Unknown function: {tool_name}"
        }
    
    def _register_waypoint(self, obs: dict, action: tuple):
        """
        注册新waypoint到registry
        
        Parameters
        ----------
        obs : dict
            当前观测
        action : tuple
            (r, theta) 动作
        """
        r, theta = action
        T_odom_base = obs['base_to_odom_matrix']
        
        # 计算waypoint全局位置
        p_base = np.array([r * np.cos(theta), r * np.sin(theta), 0.0, 1.0])
        p_odom = T_odom_base @ p_base
        
        # 计算yaw
        yaw = mat_to_yaw(T_odom_base)
        
        # 保存
        self.waypoint_registry[self.next_wp_id] = {
            'pos': p_odom[:3],
            'rgb': obs['rgb'].copy(),
            'iter': self.step_ndx,
            'yaw': yaw
        }
        
        print(f"[NavAgent] Registered waypoint {self.next_wp_id} at {p_odom[:3]}")
        self.next_wp_id += 1
    
    def _nav_with_function_call(self, obs: dict, goal: str, iter: int, goal_description: str = ""):
        """
        支持Function Call的导航决策
        
        Returns
        -------
        response : str
            VLM完整响应
        rgb_vis : np.ndarray
            可视化图像
        action : tuple/dict
            动作（可能是frontier或waypoint）
        timing_info : dict
            时序信息
        """
        t_start = time.time()
        
        # 1. 计算可导航性
        a_initial = self._navigability(obs)
        a_final = self._action_proposer(a_initial, obs['base_to_odom_matrix'])
        
        # 2. 生成BEV和标注frontiers
        current_pos = obs['base_to_odom_matrix'][:3, 3]
        bev_map = self._generate_bev_with_waypoints(current_pos)
        
        K = obs['intrinsic']
        T_cam_base = obs['extrinsic'] @ obs['base_to_odom_matrix']
        rgb_annotated, frontier_map = self._annotate_frontiers(obs['rgb'], a_final, K, T_cam_base)
        
        # 3. 构建prompt
        frontier_list = ', '.join(frontier_map.keys()) if frontier_map else 'none'
        wp_list = ', '.join(map(str, self.waypoint_registry.keys())) if self.waypoint_registry else 'none'
        
        prompt = f"""Iteration {iter}: Finding {goal}

Image 1: BEV map (YOU=red, waypoints={wp_list})
Image 2: Current view (frontiers={frontier_list})

Actions:
1. Select frontier: {{"action": "frontier", "id": "A"}}
2. View waypoint: call get_waypoint_rgb(wp_id)
3. Go to waypoint: call go_to(wp_id)

Decide based on BEV global info and current view."""
        
        # 4. Function Call循环
        tools = self._get_function_definitions()
        images = [bev_map, rgb_annotated]
        
        for fc_iter in range(self.cfg['max_fc_iterations']):
            print(f"[NavAgent] FC iteration {fc_iter+1}/{self.cfg['max_fc_iterations']}")
            
            response = self.actionVLM.call_with_tools(images, prompt, tools)
            
            if response['type'] == 'tool_call':
                # VLM调用函数
                for tool_call in response['tool_calls']:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    tool_id = tool_call['id']
                    
                    print(f"[NavAgent] VLM called: {tool_name}({tool_args})")
                    
                    # 执行函数
                    result = self._handle_function_call(tool_name, tool_args)
                    
                    if not result['success']:
                        # 添加错误结果
                        self.actionVLM.add_tool_result(
                            tool_id, tool_name, f"Error: {result['error']}"
                        )
                        prompt = f"Error occurred. Please try again or make direct decision."
                        images = []
                    else:
                        if tool_name == 'get_waypoint_rgb':
                            # 返回waypoint视角
                            self.actionVLM.add_tool_result(
                                tool_id, tool_name,
                                f"Showing waypoint {tool_args['wp_id']} view"
                            )
                            prompt = f"This is waypoint {tool_args['wp_id']}. Continue decision."
                            images = [result['result']]
                        
                        elif tool_name == 'go_to':
                            # 直接返回
                            t_end = time.time()
                            return (
                                response['content'],
                                rgb_annotated,
                                result['result'],
                                {'total_time': t_end - t_start, 'fc_iterations': fc_iter + 1}
                            )
            else:
                # 最终决策
                try:
                    decision = self._eval_response(response['content'])
                    if decision.get('action') == 'frontier':
                        frontier_id = decision['id']
                        if frontier_id in frontier_map:
                            action = frontier_map[frontier_id]
                            self._register_waypoint(obs, action)
                            
                            t_end = time.time()
                            return (
                                response['content'],
                                rgb_annotated,
                                action,
                                {'total_time': t_end - t_start, 'fc_iterations': fc_iter + 1}
                            )
                except Exception as e:
                    print(f"[NavAgent] Parse error: {e}")
                
                t_end = time.time()
                return (
                    None,
                    rgb_annotated,
                    None,
                    {'total_time': t_end - t_start, 'fc_iterations': fc_iter + 1}
                )
        
        # 超时
        t_end = time.time()
        return (
            None,
            rgb_annotated,
            None,
            {'total_time': t_end - t_start, 'fc_iterations': self.cfg['max_fc_iterations']}
        )
    
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

    def _nav(self, obs: dict, goal: str, iter: int, goal_description: str = ""):
        """主导航逻辑 - 使用Function Call模式"""
        return self._nav_with_function_call(obs, goal, iter, goal_description)

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
