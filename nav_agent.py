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


def mat_to_yaw(T_4x4: np.ndarray) -> float:
    """Extract yaw from 4x4 transformation matrix"""
    Rm = T_4x4[:3, :3].astype(np.float64)
    return math.atan2(Rm[1, 0], Rm[0, 0])


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
        
        # Function Call configuration
        self.cfg.setdefault('max_fc_iterations', 5)  # Maximum Function Call iterations
        
        # VLM configuration defaults
        self.cfg.setdefault('vlm_model', '/data/sea_disk0/liujy/models/Qwen/Qwen3-VL-8B-Instruct/')
        self.cfg.setdefault('vlm_api_key', 'EMPTY')
        self.cfg.setdefault('vlm_base_url', 'http://10.15.89.71:34134/v1/')
        self.cfg.setdefault('vlm_timeout', 10)
        
        # ObstacleMap configuration
        self.cfg.setdefault('agent_radius', 0.3)  # Robot radius in meters
        self.cfg.setdefault('frontier_area_thresh', 3.0)  # Minimum frontier area in m^2

        self.clip_dist = self.cfg['clip_dist']
        self.turn_angle_deg = float(self.cfg['turn_angle_deg'])
        
        # Initialize goal attributes (will be set properly in reset())
        self.goal = None
        self.goal_description = ''
        
        # Waypoint管理
        self.waypoint_registry = {}  # {wp_id: {'pos': np.array, 'rgb': np.array, 'iter': int, 'yaw': float}}
        self.next_wp_id = 1
        
        # Frontier管理（单个step内全局一致的ID）
        self.frontier_registry = {}  # {frontier_id: (x, y)}
        
        # Initialize ObstacleMap for frontier-based navigation
        self.obstacle_map = ObstacleMap(
            min_height=self.cfg['obstacle_height_min'],
            max_height=self.cfg['obstacle_height_max'],
            agent_radius=self.cfg['agent_radius'],
            area_thresh=self.cfg['frontier_area_thresh'],
            size=self.map_size,
            pixels_per_meter=self.scale,
        )
        
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

    def get_history_summary(self):
        """
        Get a summary of navigation history for debugging/logging.
        
        Returns
        -------
        dict
            Dictionary containing history statistics (simplified for Function Call mode)
        """
        return {
            'keyframe_mode': False,  # Function Call mode doesn't use keyframes
            'total_waypoints': len(self.waypoint_registry),
            'waypoint_ids': list(self.waypoint_registry.keys())
        }
    
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
        self.scale = 100
        self.step_ndx = 0
        self.init_pos = None
        
        # Store goal information as instance attributes
        self.goal = goal
        self.goal_description = goal_description
        
        # Reset waypoint registry
        self.waypoint_registry = {}
        self.next_wp_id = 1
        
        # Reset frontier registry
        self.frontier_registry = {}
        
        # Reset ObstacleMap
        self.obstacle_map.reset()
        
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
- RGB camera for first-person observation
- Bird's-eye view (BEV) map showing explored areas
- Omnidirectional base with efficient turning
- Can revisit previously explored locations
{description_text}
VISUAL INFORMATION YOU WILL RECEIVE:
1. BEV Map (Bird's-Eye View):
   - Red circle = YOUR current position
   - Blue circles with numbers = Historical waypoints you've visited
   - Green dots = Frontiers (boundaries between explored and unexplored areas)
   - Use this for global navigation planning

2. Current RGB View (First-Person Camera):
   - Green circles with numbers = Visible frontiers you can navigate to
   - Frontier IDs are consistent across all views
   - These are exploration targets leading to new areas

AVAILABLE ACTIONS:
1. SELECT FRONTIER: Navigate to a visible frontier to explore new area
   - Format: {{"action": "frontier", "id": <frontier_id>}}
   - Frontiers lead to unexplored areas
   - Prioritize frontiers that might lead toward the goal

2. VIEW WAYPOINT: Check RGB view from a historical waypoint
   - Use function: get_waypoint_rgb(wp_ids=[<id1>, <id2>, ...])
   - Helps recall what you saw at that location
   - Waypoint views also show visible frontiers with consistent IDs

3. GO TO WAYPOINT: Navigate back to a historical waypoint
   - Use function: go_to(wp_id=<id>)
   - Useful for strategic repositioning or revisiting areas

DECISION STRATEGY:
1. Use prior knowledge about where {goal.upper()} is typically located
2. PRIORITIZE exploring new frontiers over revisiting waypoints
3. Use BEV map to understand global layout and avoid redundant exploration
4. Check waypoint views if you need to recall specific areas
5. Consider frontier directions that align with likely goal locations
6. Balance between systematic exploration and goal-directed search

CONSTRAINTS:
- You CANNOT go through closed doors
- You CANNOT go up or down stairs
- Stay in the current room/floor

Remember these instructions throughout the navigation episode. Each iteration will show you updated BEV map and current view with frontier options.
"""
        return initial_prompt
    
    def _generate_bev_with_waypoints(self, current_pos: np.ndarray, frontiers: list = None) -> np.ndarray:
        """
        生成标注了历史waypoints和当前frontiers的BEV地图（使用ObstacleMap）
        
        Parameters
        ----------
        current_pos : np.ndarray
            当前机器人位置（odom坐标系）
        frontiers : list, optional
            当前所有frontier点的全局坐标列表 [(x, y), ...]
            
        Returns
        -------
        np.ndarray
            标注后的BEV图像（RGB）
        """
        # 使用obstacle_map的可视化作为基础
        bev = self.obstacle_map.visualize(robot_pos=current_pos[:2])
        
        # 先绘制frontiers（绿色小圆点）
        if frontiers is not None and len(frontiers) > 0:
            GREEN = (0, 255, 0)  # BGR格式
            for frontier_xy in frontiers:
                frontier_px = self.obstacle_map._xy_to_px(np.array([frontier_xy]))[0]
                cv2.circle(bev, tuple(frontier_px.astype(int)), 5, GREEN, -1)  # 绿色填充
                cv2.circle(bev, tuple(frontier_px.astype(int)), 5, WHITE, 1)  # 白色边框
        
        # 标注历史waypoints（蓝色圆+FOV扇形+编号）
        BLUE = (255, 0, 0)  # BGR格式
        for wp_id, wp_data in self.waypoint_registry.items():
            wp_pos = wp_data['pos']
            wp_yaw = wp_data['yaw']
            wp_px = self.obstacle_map._xy_to_px(wp_pos[:2].reshape(1, 2))[0]
            
            # 绘制FOV扇形
            fov_radius = 50  # 扇形半径（像素）
            fov_angle = self.fov if hasattr(self, 'fov') else 60  # 使用相机FOV（度）
            
            # 计算扇形的起始和结束角度
            # yaw是机器人朝向（弧度），OpenCV的角度是从x轴正方向逆时针
            # 需要转换：OpenCV角度 = 90 - yaw_degrees
            start_angle = 90 - np.rad2deg(wp_yaw) - fov_angle / 2
            end_angle = 90 - np.rad2deg(wp_yaw) + fov_angle / 2
            
            # 绘制半透明扇形
            overlay = bev.copy()
            cv2.ellipse(
                overlay,
                tuple(wp_px.astype(int)),
                (fov_radius, fov_radius),
                0,  # 旋转角度
                start_angle,
                end_angle,
                BLUE,
                -1  # 填充
            )
            # 混合原图和扇形（半透明效果）
            cv2.addWeighted(overlay, 0.3, bev, 0.7, 0, bev)
            
            # 绘制扇形边界线
            cv2.ellipse(
                bev,
                tuple(wp_px.astype(int)),
                (fov_radius, fov_radius),
                0,
                start_angle,
                end_angle,
                BLUE,
                2
            )
            
            # 画中心圆
            cv2.circle(bev, tuple(wp_px.astype(int)), 15, BLUE, -1)  # 蓝色填充
            cv2.circle(bev, tuple(wp_px.astype(int)), 15, WHITE, 2)  # 白色边框
            
            # 画编号
            text = str(wp_id)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(bev, text, (int(wp_px[0]-tw//2), int(wp_px[1]+th//2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        
        # 添加图例
        legend_y = 50
        cv2.putText(bev, "BEV Map with Waypoints", (20, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2)
        cv2.putText(bev, f"Total Waypoints: {len(self.waypoint_registry)}", (20, legend_y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        
        return bev
    
    def _get_visible_frontiers(self, obs: dict) -> list:
        """
        获取当前视野内可见的frontier点
        
        Parameters
        ----------
        obs : dict
            观测字典，包含intrinsic, extrinsic, base_to_odom_matrix等
            
        Returns
        -------
        list
            可见frontier列表，每个元素为(frontier_xy, pixel_uv)
            frontier_xy: 全局坐标(x, y)
            pixel_uv: 图像坐标(u, v)
        """
        # 获取所有frontier（全局坐标）
        all_frontiers = self.obstacle_map.frontiers
        
        if len(all_frontiers) == 0:
            return []
        
        # 获取相机参数
        K = obs['intrinsic']
        T_cam_odom = obs['extrinsic']
        T_odom_base = obs['base_to_odom_matrix']
        T_cam_base = T_cam_odom @ T_odom_base
        
        # 机器人位置和朝向
        robot_pos = T_odom_base[:3, 3]
        robot_yaw = np.arctan2(T_odom_base[1, 0], T_odom_base[0, 0])
        
        # 图像尺寸
        H, W = obs['rgb'].shape[:2]
        
        # FOV参数
        fx = K[0, 0]
        fov_horizontal = 2 * np.arctan(W / (2 * fx))
        max_dist = self.clip_dist
        
        visible_frontiers = []
        
        for frontier_xy in all_frontiers:
            frontier_3d = np.array([frontier_xy[0], frontier_xy[1], 0.0])
            
            # 1. 检查FOV角度
            vec_to_frontier = frontier_3d[:2] - robot_pos[:2]
            angle_to_frontier = np.arctan2(vec_to_frontier[1], vec_to_frontier[0])
            angle_diff = np.abs(np.arctan2(
                np.sin(angle_to_frontier - robot_yaw),
                np.cos(angle_to_frontier - robot_yaw)
            ))
            
            if angle_diff > fov_horizontal / 2:
                continue
            
            # 2. 投影到图像
            # frontier在base坐标系中的位置
            frontier_homo = np.array([frontier_3d[0], frontier_3d[1], frontier_3d[2], 1.0])
            frontier_base = np.linalg.inv(T_odom_base) @ frontier_homo
            
            uv = point_to_pixel(frontier_base[:3], K, T_cam_base)
            
            if uv is None:
                continue
            
            pixel_pos = (int(round(uv[0][0])), int(round(uv[0][1])))
            
            # 3. 检查是否在图像范围内（留一些边距）
            margin = int(0.05 * min(W, H))
            if not (margin <= pixel_pos[0] < W - margin and margin <= pixel_pos[1] < H - margin):
                continue
            
            visible_frontiers.append((frontier_xy, pixel_pos))
        
        return visible_frontiers
    
    def _get_visible_frontiers_from_pose(self, pose: np.ndarray, yaw: float, 
                                         intrinsic: np.ndarray, extrinsic: np.ndarray,
                                         rgb_shape: tuple, all_frontiers: list) -> list:
        """
        从指定位姿计算可见的frontiers
        
        Parameters
        ----------
        pose : np.ndarray
            位置 (x, y, z) in odom frame
        yaw : float
            朝向角度（弧度）
        intrinsic : np.ndarray
            相机内参矩阵
        extrinsic : np.ndarray
            相机外参矩阵（cam to odom）
        rgb_shape : tuple
            RGB图像尺寸 (H, W, C)
        all_frontiers : list
            所有frontier点的全局坐标 [(x, y), ...]
            
        Returns
        -------
        list
            可见frontier列表，每个元素为(frontier_xy, pixel_uv)
        """
        if len(all_frontiers) == 0:
            return []
        
        # 构建base_to_odom矩阵
        T_odom_base = np.eye(4)
        T_odom_base[0, 0] = np.cos(yaw)
        T_odom_base[0, 1] = -np.sin(yaw)
        T_odom_base[1, 0] = np.sin(yaw)
        T_odom_base[1, 1] = np.cos(yaw)
        T_odom_base[:3, 3] = pose
        
        # 相机参数
        K = intrinsic
        T_cam_odom = extrinsic
        T_cam_base = T_cam_odom @ T_odom_base
        
        # 图像尺寸
        H, W = rgb_shape[:2]
        
        # FOV参数
        fx = K[0, 0]
        fov_horizontal = 2 * np.arctan(W / (2 * fx))
        
        visible_frontiers = []
        
        for frontier_xy in all_frontiers:
            frontier_3d = np.array([frontier_xy[0], frontier_xy[1], 0.0])
            
            # 1. 检查FOV角度
            vec_to_frontier = frontier_3d[:2] - pose[:2]
            angle_to_frontier = np.arctan2(vec_to_frontier[1], vec_to_frontier[0])
            angle_diff = np.abs(np.arctan2(
                np.sin(angle_to_frontier - yaw),
                np.cos(angle_to_frontier - yaw)
            ))
            
            if angle_diff > fov_horizontal / 2:
                continue
            
            # 2. 投影到图像
            frontier_homo = np.array([frontier_3d[0], frontier_3d[1], frontier_3d[2], 1.0])
            frontier_base = np.linalg.inv(T_odom_base) @ frontier_homo
            
            uv = point_to_pixel(frontier_base[:3], K, T_cam_base)
            
            if uv is None:
                continue
            
            pixel_pos = (int(round(uv[0][0])), int(round(uv[0][1])))
            
            # 3. 检查是否在图像范围内（留一些边距）
            margin = int(0.05 * min(W, H))
            if not (margin <= pixel_pos[0] < W - margin and margin <= pixel_pos[1] < H - margin):
                continue
            
            visible_frontiers.append((frontier_xy, pixel_pos))
        
        return visible_frontiers
    
    def _find_frontier_id(self, frontier_xy: tuple, threshold: float = 0.1) -> int:
        """
        通过坐标匹配找到frontier的全局ID
        
        Parameters
        ----------
        frontier_xy : tuple
            Frontier坐标 (x, y)
        threshold : float
            匹配距离阈值（米）
            
        Returns
        -------
        int or None
            匹配的frontier ID，如果未找到返回None
        """
        for fid, registered_xy in self.frontier_registry.items():
            dist = np.linalg.norm(np.array(frontier_xy) - np.array(registered_xy))
            if dist < threshold:
                return fid
        return None
    
    def _annotate_frontiers(self, rgb: np.ndarray, visible_frontiers: list, obs: dict):
        """
        在当前RGB图像上标注可见的frontiers（使用全局ID）
        
        Parameters
        ----------
        rgb : np.ndarray
            当前RGB图像
        visible_frontiers : list
            可见frontier列表 [(frontier_xy, pixel_uv), ...]
        obs : dict
            观测字典
            
        Returns
        -------
        rgb_annotated : np.ndarray
            标注后的RGB图像
        frontier_map : dict
            frontier映射 {frontier_id: frontier_xy}
        """
        rgb_annotated = rgb.copy()
        frontier_map = {}
        
        # 使用全局ID标识frontiers
        for frontier_xy, pixel_pos in visible_frontiers:
            # 找到对应的全局ID
            frontier_id = self._find_frontier_id(frontier_xy)
            
            if frontier_id is None:
                # 如果找不到ID（理论上不应该发生），跳过
                continue
            
            frontier_map[frontier_id] = frontier_xy
            
            # 画绿色圆圈+全局ID
            cv2.circle(rgb_annotated, pixel_pos, 25, GREEN, 3)
            text = str(frontier_id)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(rgb_annotated, text, (pixel_pos[0]-tw//2, pixel_pos[1]+th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)
        
        # 添加标题
        cv2.putText(rgb_annotated, f"Frontiers: {len(visible_frontiers)} visible", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        
        return rgb_annotated, frontier_map
    
    def _get_function_definitions(self) -> list:
        """返回Function定义"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_waypoint_rgb",
                    "description": "查看一个或多个历史waypoint的第一人称RGB视角，帮助回忆该位置的环境",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "wp_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": f"waypoint编号列表，可选范围：{list(self.waypoint_registry.keys()) if self.waypoint_registry else 'none'}"
                            }
                        },
                        "required": ["wp_ids"]
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
            wp_ids = args.get('wp_ids', [])
            
            if not wp_ids:
                return {
                    'success': False,
                    'result': None,
                    'error': "No waypoint IDs provided"
                }
            
            # 收集所有有效的waypoint RGB图像（带frontier标注）
            rgb_images = []
            invalid_ids = []
            
            for wp_id in wp_ids:
                if wp_id not in self.waypoint_registry:
                    invalid_ids.append(wp_id)
                    continue
                
                wp_data = self.waypoint_registry[wp_id]
                rgb = wp_data['rgb'].copy()
                
                # 计算从该waypoint能看到的frontiers
                visible_frontiers = self._get_visible_frontiers_from_pose(
                    pose=wp_data['pos'],
                    yaw=wp_data['yaw'],
                    intrinsic=wp_data['intrinsic'],
                    extrinsic=wp_data['extrinsic'],
                    rgb_shape=rgb.shape,
                    all_frontiers=list(self.frontier_registry.values())
                )
                
                # 标注frontiers（使用全局ID）
                # 创建一个临时obs字典用于_annotate_frontiers
                temp_obs = {'rgb': rgb}
                rgb_annotated, _ = self._annotate_frontiers(rgb, visible_frontiers, temp_obs)
                
                rgb_images.append(rgb_annotated)
            
            if invalid_ids:
                return {
                    'success': False,
                    'result': None,
                    'error': f"Waypoints {invalid_ids} not found. Available: {list(self.waypoint_registry.keys())}"
                }
            
            return {
                'success': True,
                'result': rgb_images,  # 返回标注后的图像列表
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
        
        # 保存（包括相机内参和外参）
        self.waypoint_registry[self.next_wp_id] = {
            'pos': p_odom[:3],
            'rgb': obs['rgb'].copy(),
            'iter': self.step_ndx,
            'yaw': yaw,
            'intrinsic': obs['intrinsic'].copy(),  # 保存相机内参
            'extrinsic': obs['extrinsic'].copy(),  # 保存相机外参
        }
        
        print(f"[NavAgent] Registered waypoint {self.next_wp_id} at {p_odom[:3]}")
        self.next_wp_id += 1
    
    def _nav_with_function_call(self, obs: dict, goal: str, iter: int, goal_description: str = ""):
        """
        支持Function Call的导航决策（使用frontier-based方法）
        
        Returns
        -------
        response : str
            VLM完整响应
        rgb_vis : np.ndarray
            可视化图像（标注后的RGB）
        action : tuple/dict
            动作（可能是frontier或waypoint）
        timing_info : dict
            时序信息，包含：
            - total_time: 总时间
            - fc_iterations: Function Call迭代次数
            - bev_map: BEV地图
            - rgb_original: 原始RGB图像
            - prompt: 初始prompt
            - conversation: 完整对话历史（不含图片）
        """
        t_start = time.time()
        
        # 1. 更新ObstacleMap
        self._navigability(obs)
        
        # 2. 获取可见的frontiers
        visible_frontiers = self._get_visible_frontiers(obs)
        
        # 3. 生成BEV和标注frontiers
        current_pos = obs['base_to_odom_matrix'][:3, 3]
        all_frontiers = self.obstacle_map.frontiers  # 获取所有frontier
        bev_map = self._generate_bev_with_waypoints(current_pos, all_frontiers)
        
        rgb_annotated, frontier_map = self._annotate_frontiers(obs['rgb'], visible_frontiers, obs)
        
        # 4. 构建prompt
        frontier_list = ', '.join(map(str, frontier_map.keys())) if frontier_map else 'none'
        wp_list = ', '.join(map(str, self.waypoint_registry.keys())) if self.waypoint_registry else 'none'
        
        prompt = f"""Iteration {iter}: Finding {goal}

IMAGE DESCRIPTIONS:
- Image 1 (BEV Map): Bird's-eye view showing explored area
  * Red circle = YOUR current position
  * Blue circles with numbers = Historical waypoints (IDs: {wp_list})
  * Green dots = Frontiers (unexplored boundaries)
  * Use this for global navigation planning

- Image 2 (Current RGB View): First-person camera view
  * Green circles with numbers = Visible frontiers (IDs: {frontier_list})
  * These are exploration targets you can navigate to
  * Frontier IDs are consistent across all views

AVAILABLE ACTIONS:
1. SELECT FRONTIER: Navigate to a visible frontier to explore new area
   Format: {{"action": "frontier", "id": <frontier_id>}}
   Example: {{"action": "frontier", "id": 1}}

2. VIEW WAYPOINT: Check RGB view from a historical waypoint (shows frontiers visible from there)
   Use function: get_waypoint_rgb(wp_ids=[<id1>, <id2>, ...])
   This helps you recall what you saw at that location

3. GO TO WAYPOINT: Navigate back to a historical waypoint
   Use function: go_to(wp_id=<id>)
   Useful for revisiting areas or strategic repositioning

DECISION STRATEGY:
- Prioritize exploring new frontiers over revisiting waypoints
- Use BEV map to understand global layout and avoid redundant exploration
- Check waypoint views if you need to recall specific areas
- Frontier IDs remain consistent across current view and waypoint views

Make your decision now."""
        
        # 保存初始prompt（用于log）
        initial_prompt = prompt
        
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
                            # 返回waypoint视角（支持多个）
                            wp_ids = tool_args['wp_ids']
                            self.actionVLM.add_tool_result(
                                tool_id, tool_name,
                                f"Showing waypoint {wp_ids} views"
                            )
                            prompt = f"These are waypoints {wp_ids}. Continue decision."
                            images = result['result']  # 已经是列表
                        
                        elif tool_name == 'go_to':
                            # 直接返回
                            t_end = time.time()
                            timing_info = {
                                'total_time': t_end - t_start,
                                'fc_iterations': fc_iter + 1,
                                'bev_map': bev_map,
                                'rgb_original': obs['rgb'],
                                'prompt': initial_prompt,
                                'conversation': self.actionVLM.get_conversation_history()
                            }
                            return (
                                response['content'],
                                rgb_annotated,
                                result['result'],
                                timing_info
                            )
            else:
                # 最终决策
                try:
                    decision = self._eval_response(response['content'])
                    if decision.get('action') == 'frontier':
                        frontier_id = decision['id']
                        if frontier_id in frontier_map:
                            # frontier_map[frontier_id] 是全局坐标 (x, y)
                            frontier_xy = frontier_map[frontier_id]
                            
                            # 转换为极坐标动作 (r, theta)
                            robot_pos = obs['base_to_odom_matrix'][:3, 3]
                            vec = frontier_xy - robot_pos[:2]
                            r = np.linalg.norm(vec)
                            
                            # 计算相对于机器人朝向的角度
                            robot_yaw = np.arctan2(obs['base_to_odom_matrix'][1, 0],
                                                   obs['base_to_odom_matrix'][0, 0])
                            frontier_angle = np.arctan2(vec[1], vec[0])
                            theta = frontier_angle - robot_yaw
                            
                            # 归一化到[-pi, pi]
                            theta = np.arctan2(np.sin(theta), np.cos(theta))
                            
                            action = (r, theta)
                            self._register_waypoint(obs, action)
                            
                            t_end = time.time()
                            timing_info = {
                                'total_time': t_end - t_start,
                                'fc_iterations': fc_iter + 1,
                                'bev_map': bev_map,
                                'rgb_original': obs['rgb'],
                                'prompt': initial_prompt,
                                'conversation': self.actionVLM.get_conversation_history()
                            }
                            return (
                                response['content'],
                                rgb_annotated,
                                action,
                                timing_info
                            )
                except Exception as e:
                    print(f"[NavAgent] Parse error: {e}")
                
                t_end = time.time()
                timing_info = {
                    'total_time': t_end - t_start,
                    'fc_iterations': fc_iter + 1,
                    'bev_map': bev_map,
                    'rgb_original': obs['rgb'],
                    'prompt': initial_prompt,
                    'conversation': self.actionVLM.get_conversation_history()
                }
                return (
                    None,
                    rgb_annotated,
                    None,
                    timing_info
                )
        
        # 超时
        t_end = time.time()
        timing_info = {
            'total_time': t_end - t_start,
            'fc_iterations': self.cfg['max_fc_iterations'],
            'bev_map': bev_map,
            'rgb_original': obs['rgb'],
            'prompt': prompt,
            'conversation': self.actionVLM.get_conversation_history()
        }
        return (
            None,
            rgb_annotated,
            None,
            timing_info
        )
    
    def cal_fov(self, intrinsics: np.ndarray, W: int):
        fx = intrinsics[0, 0]
        return 2 * np.arctan(W / (2 * fx)) * 180 / np.pi

    def _update_frontier_registry(self):
        """
        更新frontier注册表，为当前step的所有frontier分配简单的递增ID
        每个step重新分配ID，只需要在当前step内保持一致
        """
        current_frontiers = self.obstacle_map.frontiers  # [(x, y), ...]
        
        # 清空并重新分配ID（每个step独立）
        self.frontier_registry = {}
        
        # 为当前所有frontier分配ID（从1开始）
        for idx, frontier_xy in enumerate(current_frontiers):
            self.frontier_registry[idx + 1] = frontier_xy
    
    def _navigability(self, obs: dict):
        """
        Update ObstacleMap with current observation.
        
        Parameters
        ----------
        obs : dict
            Observation dictionary
        """
        depth_image = obs['depth']
        K = obs['intrinsic']
        T_cam_odom = obs['extrinsic']
        T_odom_base = obs['base_to_odom_matrix']

        if self.init_pos is None:
            self.init_pos = T_odom_base[:3, 3].copy()

        # Update ObstacleMap with current observation
        self.obstacle_map.update_map(
            depth=depth_image,
            intrinsic=K,
            extrinsic=T_cam_odom,
            base_to_odom=T_odom_base,
            min_depth=0.2,
            max_depth=self.clip_dist,
        )
        
        # Update frontier registry with persistent IDs
        self._update_frontier_registry()

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
