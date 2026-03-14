#!/usr/bin/env python3
"""
Habitat-Lab Client for LLM Navigation Server
Uses VelocityAction for flexible control with real-time visualization
"""

import argparse
import json
import math
import os
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, Callable, List

import cv2
import numpy as np
import requests
from PIL import Image

import habitat
from habitat.config.default_structured_configs import VelocityControlActionConfig
from habitat.core.env import Env

try:
    from habitat.utils.visualizations import maps
except ImportError:
    # Fallback if maps module not available
    maps = None


class SimpleNavigationController:
    """
    Navigation controller with two modes:
    1. PointNav mode: Uses pre-trained PointNav policy (like VLFM)
    2. Teleport mode: Directly sets agent state (for debugging)
    """

    def __init__(
        self,
        env: Env,
        use_pointnav: bool = True,
        pointnav_policy_path: str = "data/pointnav_weights.pth",
        stop_radius: float = 0.9,
        max_steps: int = 500
    ):
        """
        Args:
            env: Habitat environment
            use_pointnav: If True, use PointNav policy; if False, use teleport
            pointnav_policy_path: Path to PointNav weights
            stop_radius: Distance threshold to stop (meters)
            max_steps: Maximum navigation steps
        """
        self.env = env
        self.use_pointnav = use_pointnav

        if use_pointnav:
            # Import and initialize PointNav controller
            try:
                from pointnav_controller import PointNavController
                self.pointnav_controller = PointNavController(
                    env=env,
                    pointnav_policy_path=pointnav_policy_path,
                    stop_radius=stop_radius,
                    max_steps=max_steps
                )
                print(f'[Controller] Using PointNav policy mode')
            except Exception as e:
                print(f'[Controller] Failed to load PointNav controller: {e}')
                print(f'[Controller] Falling back to teleport mode')
                self.use_pointnav = False

    def navigate_to_pose(
        self,
        goal_position: np.ndarray,
        goal_orientation: Optional[np.ndarray] = None,
        max_steps: int = 100,
        step_callback: Optional[Callable] = None
    ) -> Tuple[bool, list, Dict[str, Any]]:
        """
        Navigate to goal pose

        Args:
            goal_position: [x, y, z] target position in Habitat coordinates
            goal_orientation: [x, y, z, w] target quaternion in Habitat coordinates (optional)
            max_steps: maximum steps to reach goal
            step_callback: optional callback for each PointNav env step

        Returns:
            (success, observations, info)
        """
        if self.use_pointnav:
            # Use PointNav policy
            success, observations, info = self.pointnav_controller.navigate_to_position(
                goal_position,
                goal_orientation,
                verbose=True,
                max_steps_override=max_steps,
                step_callback=step_callback
            )
            print(f'[Controller] PointNav navigation: success={success}, steps={info["num_steps"]}')
            return success, observations, info
        else:
            # Teleport mode (original behavior)
            return self._teleport_to_pose(goal_position, goal_orientation)

    def _teleport_to_pose(
        self,
        goal_position: np.ndarray,
        goal_orientation: Optional[np.ndarray] = None
    ) -> Tuple[bool, list, Dict[str, Any]]:
        """
        Teleport to goal pose (original behavior)

        Args:
            goal_position: [x, y, z] target position in Habitat coordinates
            goal_orientation: [x, y, z, w] target quaternion in Habitat coordinates (optional)

        Returns:
            (success, observations, info)
        """
        from habitat_sim.utils.common import quat_from_coeffs

        # Get current state
        agent_state = self.env.sim.get_agent_state()

        # Set new position
        agent_state.position = goal_position

        # Set new orientation if provided
        if goal_orientation is not None:
            # goal_orientation is [x, y, z, w] format
            agent_state.rotation = quat_from_coeffs(goal_orientation)
            print(f'[Controller] Teleporting to pose: position={goal_position}, orientation={goal_orientation}')
        else:
            print(f'[Controller] Teleporting to position: {goal_position}')

        # Apply the new state
        self.env.sim.set_agent_state(agent_state.position, agent_state.rotation)

        # Get observation after teleporting
        obs = self.env.sim.get_sensor_observations()

        info = {
            "num_steps": 1,
            "actions": [],
            "final_distance": -1.0,
            "success": True,
            "interrupted": False,
            "interrupt_reason": None,
            "step_budget": 1
        }
        return True, [obs], info


class HabitatLLMNavClient:
    """Client that connects Habitat-Lab with your LLM navigation server"""

    def __init__(
        self,
        env: Env,
        server_url: str,
        goal_text: str,
        camera_height: float = 1.5,
        max_pointnav_steps: int = 500,
        visualize: bool = True,
        save_video: bool = False,
        output_dir: str = "./output",
        save_data: bool = False,
        episode_id: Optional[int] = None,
        use_pointnav: bool = True,
        pointnav_policy_path: str = "data/pointnav_weights.pth"
    ):
        self.env = env
        self.server_url = server_url.rstrip('/')
        self.goal_text = goal_text
        self.camera_height = camera_height
        self.max_pointnav_steps = int(max_pointnav_steps)
        self.visualize = visualize
        self.save_video = save_video
        self.output_dir = output_dir
        self.save_data = save_data
        self.episode_id = episode_id  # Specific episode to run

        self.iteration_count = 0  # Keyframe index for visualization compatibility
        self.keyframe_count = 0
        self.pointnav_step_total = 0
        self.server_request_count = 0
        self.trajectory = []
        self.video_frames = []
        self.episode_metrics = None  # Store episode evaluation metrics

        # Create navigation controller
        # Set use_pointnav=True to use PointNav policy (like VLFM)
        # Set use_pointnav=False to use teleport mode (for debugging)
        self.nav_controller = SimpleNavigationController(
            env,
            use_pointnav=use_pointnav,
            pointnav_policy_path=pointnav_policy_path,
            stop_radius=0.5,
            max_steps=self.max_pointnav_steps
        )

        # Camera intrinsics
        self.intrinsic = self._compute_camera_intrinsic()

        # Ground height offset will be set after first reset
        self.ground_height_offset = None

        if save_video or save_data:
            os.makedirs(output_dir, exist_ok=True)

        # Create data directory if saving data
        if save_data:
            self.data_dir = os.path.join(output_dir, "navigation_data")
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"[Client] Will save navigation data to: {self.data_dir}")

        print(f"[Client] Initialized with server: {server_url}")
        if goal_text:
            print(f"[Client] Goal: {goal_text}")
        else:
            print(f"[Client] Goal: Will use episode's goal")
        print(f"[Client] Visualization: {'Enabled' if visualize else 'Disabled'}")
        print(f"[Client] Camera intrinsic:\n{self.intrinsic}")

    def _compute_camera_intrinsic(self) -> np.ndarray:
        """Compute camera intrinsic matrix from habitat config"""
        # Get sensor config from habitat config
        sim_config = self.env.sim.habitat_config

        # Get the first agent (agents is a dict with string keys like "main_agent")
        agent_names = list(sim_config.agents.keys())
        if not agent_names:
            print("[Client] Warning: No agents found in config, using defaults")
            width = 512
            height = 512
            hfov = 79
        else:
            agent_name = agent_names[0]
            agent_config = sim_config.agents[agent_name]

            # Prefer depth sensor intrinsics (must match uploaded depth image),
            # fall back to RGB sensor when needed.
            selected_sensor = None
            fallback_rgb_sensor = None
            if hasattr(agent_config, 'sim_sensors'):
                for sensor_name in agent_config.sim_sensors:
                    sensor_spec = agent_config.sim_sensors[sensor_name]
                    lname = sensor_name.lower()
                    if 'depth' in lname and selected_sensor is None:
                        selected_sensor = sensor_spec
                    if 'rgb' in lname and fallback_rgb_sensor is None:
                        fallback_rgb_sensor = sensor_spec

            if selected_sensor is None:
                selected_sensor = fallback_rgb_sensor

            if selected_sensor is None:
                # Fallback: use default values
                print("[Client] Warning: Could not find sensor config, using defaults")
                width = 512
                height = 512
                hfov = 79
            else:
                width = selected_sensor.width
                height = selected_sensor.height
                hfov = selected_sensor.hfov

        # Compute focal length from HFOV
        focal_length = (width / 2.0) / np.tan(np.radians(hfov / 2.0))

        # Construct intrinsic matrix
        intrinsic = np.array([
            [focal_length, 0, width / 2.0],
            [0, focal_length, height / 2.0],
            [0, 0, 1]
        ], dtype=np.float32)

        return intrinsic

    def _get_transform_matrices(self, obs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute transformation matrices from habitat observation

        Coordinate systems:
        - Habitat: X=right, Y=up, Z=back (forward=-Z)
        - ROS: X=forward, Y=left, Z=up
        - Camera (optical): X=right, Y=down, Z=forward

        Returns:
            T_cam_odom: Camera to odom transform (4x4)
            T_odom_base: Odom to base transform (4x4)
        """
        from scipy.spatial.transform import Rotation as R

        # Get agent state in Habitat coordinates
        agent_state = self.env.sim.get_agent_state()
        pos_habitat = agent_state.position  # [x, y, z] in Habitat
        rot_habitat = agent_state.rotation  # quaternion in Habitat

        # Apply ground height offset to make ROS Z=0 at ground level
        # (Habitat Y coordinate corresponds to height)
        pos_habitat_adjusted = pos_habitat.copy()
        pos_habitat_adjusted[1] -= self.ground_height_offset

        # Convert quaternion to rotation matrix (Habitat frame)
        quat_xyzw = [rot_habitat.x, rot_habitat.y, rot_habitat.z, rot_habitat.w]
        R_habitat = R.from_quat(quat_xyzw).as_matrix()

        # Coordinate transformation from Habitat to ROS
        # Habitat: X=right, Y=up, Z=back (forward=-Z)
        # ROS: X=forward, Y=left, Z=up
        # Mapping: ROS_X = -Habitat_Z, ROS_Y = -Habitat_X, ROS_Z = Habitat_Y
        C_habitat_to_ros = np.array([
            [ 0,  0, -1],  # ROS X = -Habitat Z (forward)
            [-1,  0,  0],  # ROS Y = -Habitat X (left)
            [ 0,  1,  0]   # ROS Z = Habitat Y (up)
        ], dtype=np.float32)

        # Convert position from Habitat to ROS
        pos_ros = C_habitat_to_ros @ pos_habitat_adjusted

        # Convert rotation from Habitat to ROS
        R_ros = C_habitat_to_ros @ R_habitat @ C_habitat_to_ros.T

        # T_odom_base: base pose in odom frame (ROS coordinates)
        T_odom_base = np.eye(4, dtype=np.float32)
        T_odom_base[:3, :3] = R_ros.astype(np.float32)
        T_odom_base[:3, 3] = pos_ros.astype(np.float32)

        # Camera position: prefer true sensor world pose from simulator state
        # (same convention as uploaded depth), fall back to config-derived pose.
        sim_config = self.env.sim.habitat_config
        agent_names = list(sim_config.agents.keys())
        depth_sensor_name = None

        if agent_names and hasattr(sim_config.agents[agent_names[0]], 'sim_sensors'):
            agent_config = sim_config.agents[agent_names[0]]
            # Find depth sensor name (fallback to RGB if depth missing)
            rgb_sensor_name = None
            for sensor_name in agent_config.sim_sensors:
                lname = sensor_name.lower()
                if 'depth' in lname and depth_sensor_name is None:
                    depth_sensor_name = sensor_name
                if 'rgb' in lname and rgb_sensor_name is None:
                    rgb_sensor_name = sensor_name
            if depth_sensor_name is None:
                depth_sensor_name = rgb_sensor_name

        cam_pos_habitat = None
        R_cam_habitat = None
        sensor_states = getattr(agent_state, "sensor_states", None)
        if depth_sensor_name is not None and sensor_states is not None:
            sensor_state = sensor_states.get(depth_sensor_name, None)
            if sensor_state is not None:
                cam_pos_habitat = np.array(sensor_state.position, dtype=np.float32)
                sensor_rot = sensor_state.rotation
                quat_xyzw_sensor = [
                    sensor_rot.x, sensor_rot.y, sensor_rot.z, sensor_rot.w
                ]
                R_cam_habitat = R.from_quat(quat_xyzw_sensor).as_matrix()

        if cam_pos_habitat is None or R_cam_habitat is None:
            cam_offset_habitat = np.array([0.0, 1.2, 0.0], dtype=np.float32)
            cam_orientation_habitat = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            if (
                depth_sensor_name is not None
                and agent_names
                and hasattr(sim_config.agents[agent_names[0]], 'sim_sensors')
            ):
                sensor_cfg = sim_config.agents[agent_names[0]].sim_sensors.get(depth_sensor_name)
                if sensor_cfg is not None:
                    if hasattr(sensor_cfg, 'position'):
                        cam_offset_habitat = np.array(sensor_cfg.position, dtype=np.float32)
                    if hasattr(sensor_cfg, 'orientation'):
                        cam_orientation_habitat = np.array(sensor_cfg.orientation, dtype=np.float32)

            cam_offset_world = R_habitat @ cam_offset_habitat
            cam_pos_habitat = pos_habitat + cam_offset_world

            from scipy.spatial.transform import Rotation as R_scipy
            R_sensor_local = R_scipy.from_euler('xyz', cam_orientation_habitat).as_matrix()
            R_cam_habitat = R_habitat @ R_sensor_local

        # Apply ground height offset
        cam_pos_habitat_adjusted = cam_pos_habitat.copy()
        cam_pos_habitat_adjusted[1] -= self.ground_height_offset

        # Convert camera pose to ROS
        cam_pos_ros = C_habitat_to_ros @ cam_pos_habitat_adjusted
        R_cam_ros = C_habitat_to_ros @ R_cam_habitat @ C_habitat_to_ros.T

        # T_odom_cam: camera pose in odom frame (ROS coordinates)
        T_odom_cam_ros = np.eye(4, dtype=np.float32)
        T_odom_cam_ros[:3, :3] = R_cam_ros.astype(np.float32)
        T_odom_cam_ros[:3, 3] = cam_pos_ros.astype(np.float32)

        # Convert from ROS to optical convention for camera
        # ROS: X=forward, Y=left, Z=up
        # Optical: X=right, Y=down, Z=forward
        # Mapping: Optical_X = -ROS_Y, Optical_Y = -ROS_Z, Optical_Z = ROS_X
        C_ros_to_optical = np.array([
            [ 0, -1,  0, 0],  # Optical X = -ROS Y (right)
            [ 0,  0, -1, 0],  # Optical Y = -ROS Z (down)
            [ 1,  0,  0, 0],  # Optical Z = ROS X (forward)
            [ 0,  0,  0, 1]
        ], dtype=np.float32)

        # When transforming points FROM optical TO ROS, we need the inverse (transpose)
        C_optical_to_ros = C_ros_to_optical.T
        T_odom_cam_optical = T_odom_cam_ros @ C_optical_to_ros
        T_cam_odom = np.linalg.inv(T_odom_cam_optical)

        return T_cam_odom.astype(np.float32), T_odom_base


    def _visualize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Visualize depth image with colormap

        Args:
            depth: Depth image in meters (H, W)

        Returns:
            RGB visualization (H, W, 3) uint8
        """
        # Normalize depth to [0, 1] for visualization
        depth_vis = depth.copy()

        # Clip to reasonable range (0-10m)
        depth_vis = np.clip(depth_vis, 0, 10.0)

        # Normalize to [0, 255]
        depth_vis = (depth_vis / 10.0 * 255).astype(np.uint8)

        # Apply colormap (turbo is good for depth)
        import cv2
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

        # Convert BGR to RGB
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        return depth_colored


    def _extract_images(self, obs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract RGB and Depth images from habitat observation"""
        # Find RGB and depth keys
        rgb_key = None
        depth_key = None

        for key in obs.keys():
            if 'rgb' in key.lower():
                rgb_key = key
            elif 'depth' in key.lower():
                depth_key = key

        if rgb_key is None or depth_key is None:
            raise ValueError(f"RGB or Depth not found in observation keys: {list(obs.keys())}")

        rgb = obs[rgb_key]
        depth = obs[depth_key]

        # Convert RGB to uint8
        if rgb.dtype == np.float32 or rgb.dtype == np.float64:
            rgb = (rgb * 255).astype(np.uint8)

        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]  # Remove alpha

        # Process depth
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        depth = depth.astype(np.float32)

        # Depth is directly in meters (normalize_depth=False in config)
        # Handle invalid values
        depth[~np.isfinite(depth)] = 0.0

        return rgb, depth

    def visualize_observation(self, obs: Dict, info_text: str = ""):
        """
        Visualize observation with RGB, depth, and top-down map

        Args:
            obs: Habitat observation
            info_text: Additional text to display
        """
        if maps is None:
            print("[Viz] Visualization not available - maps module not found")
            return None

        rgb, depth = self._extract_images(obs)

        # Normalize depth for visualization
        depth_vis = depth.copy()
        depth_vis[depth_vis == 0] = np.nan
        depth_vis = (depth_vis - np.nanmin(depth_vis)) / (np.nanmax(depth_vis) - np.nanmin(depth_vis) + 1e-8)
        depth_vis = np.nan_to_num(depth_vis, 0)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Get top-down map if available
        try:
            top_down_map = maps.get_topdown_map_from_sim(
                self.env.sim,
                map_resolution=1024,
                draw_border=True,
            )

            # Draw agent position
            agent_state = self.env.sim.get_agent_state()
            agent_position = agent_state.position
            map_agent_pos = maps.to_grid(
                agent_position[2],
                agent_position[0],
                top_down_map.shape[0:2],
                sim=self.env.sim,
            )

            # Draw agent as a circle
            top_down_map = cv2.circle(
                top_down_map,
                (map_agent_pos[1], map_agent_pos[0]),
                radius=10,
                color=(0, 255, 0),
                thickness=-1,
            )

            # Draw trajectory
            for traj_point in self.trajectory[-20:]:  # Last 20 points
                pos = traj_point['final_position']
                map_pos = maps.to_grid(
                    pos[2],
                    pos[0],
                    top_down_map.shape[0:2],
                    sim=self.env.sim,
                )
                top_down_map = cv2.circle(
                    top_down_map,
                    (map_pos[1], map_pos[0]),
                    radius=3,
                    color=(255, 0, 0),
                    thickness=-1,
                )

            # Ensure top_down_map is 3-channel
            if top_down_map.ndim == 2:
                top_down_map = cv2.cvtColor(top_down_map, cv2.COLOR_GRAY2BGR)

            # Resize to match RGB height
            h_ratio = rgb.shape[0] / top_down_map.shape[0]
            new_width = int(top_down_map.shape[1] * h_ratio)
            top_down_map = cv2.resize(top_down_map, (new_width, rgb.shape[0]))
        except Exception as e:
            print(f"[Viz] Failed to generate top-down map: {e}")
            top_down_map = np.zeros_like(rgb)

        # Resize depth to match RGB
        depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]))

        # Combine visualizations horizontally
        vis = np.hstack([rgb, depth_vis, top_down_map])

        # Add text overlay
        if info_text:
            cv2.putText(
                vis,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Add counters
        cv2.putText(
            vis,
            f"Keyframe: {self.keyframe_count} | PointNav steps: {self.pointnav_step_total}",
            (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Display
        cv2.imshow("Habitat LLM Navigation", vis)
        cv2.waitKey(1)

        # Save frame for video
        if self.save_video:
            self.video_frames.append(vis)

        return vis

    def reset_navigation_server(self):
        """Initialize navigation on server"""
        url = f'{self.server_url}/navigation_reset'
        data = {
            'goal': self.goal_text,
            'goal_description': '',
            'confidence_threshold': 0.7
        }

        print(f'[Client] Resetting navigation on server: {url}')
        try:
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            print(f'[Client] Server reset: {result}')
            return result
        except Exception as e:
            print(f'[Client] Failed to reset navigation: {e}')
            raise

    def request_navigation_step(self, obs: Dict, is_keyframe: bool = True) -> Dict:
        """Send observation to server and get navigation decision"""
        url = f'{self.server_url}/navigation_step'
        self.server_request_count += 1

        # Extract images
        rgb, depth = self._extract_images(obs)

        # Prepare RGB image (JPEG)
        rgb_pil = Image.fromarray(rgb, mode='RGB')
        rgb_buffer = BytesIO()
        rgb_pil.save(rgb_buffer, format='JPEG', quality=95)
        rgb_buffer.seek(0)

        # Prepare Depth image (PNG 16-bit in mm)
        depth_mm = (depth * 1000.0).astype(np.uint16)
        depth_pil = Image.fromarray(depth_mm, mode='I;16')
        depth_buffer = BytesIO()
        depth_pil.save(depth_buffer, format='PNG')
        depth_buffer.seek(0)

        # Get transformation matrices
        T_cam_odom, T_odom_base = self._get_transform_matrices(obs)

        # Prepare form data
        files = {
            'rgb': ('rgb.jpg', rgb_buffer, 'image/jpeg'),
            'depth': ('depth.png', depth_buffer, 'image/png')
        }

        data = {
            'intrinsic': json.dumps(self.intrinsic.flatten().tolist()),
            'T_cam_odom': json.dumps(T_cam_odom.flatten().tolist()),
            'T_odom_base': json.dumps(T_odom_base.flatten().tolist()),
            'is_keyframe': 'true' if is_keyframe else 'false'
        }

        # Save data if enabled
        if self.save_data:
            iter_dir = os.path.join(self.data_dir, f"req_{self.server_request_count:05d}")
            os.makedirs(iter_dir, exist_ok=True)

            # Save images
            rgb_pil.save(os.path.join(iter_dir, "rgb.jpg"), quality=95)
            depth_pil.save(os.path.join(iter_dir, "depth.png"))

            # Save depth visualization (colormap)
            depth_vis = self._visualize_depth(depth)
            depth_vis_pil = Image.fromarray(depth_vis)
            depth_vis_pil.save(os.path.join(iter_dir, "depth_vis.jpg"), quality=95)

            # Save metadata
            metadata = {
                'request_index': self.server_request_count,
                'keyframe_index': self.keyframe_count,
                'pointnav_step_total': self.pointnav_step_total,
                'is_keyframe': bool(is_keyframe),
                'goal': self.goal_text,
                'intrinsic': self.intrinsic.tolist(),
                'T_cam_odom': T_cam_odom.tolist(),
                'T_odom_base': T_odom_base.tolist()
            }
            with open(os.path.join(iter_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

        try:
            response = requests.post(url, files=files, data=data, timeout=600)
            response.raise_for_status()
            result = response.json()
            return result
        except Exception as e:
            print(f'[Client] Server request failed: {e}')
            raise

    def execute_goal_pose(
        self,
        goal_pose_dict: Dict,
        max_steps: int,
        step_callback: Optional[Callable] = None
    ) -> Tuple[bool, list, Dict[str, Any]]:
        """
        Execute navigation to goal pose using velocity control

        Args:
            goal_pose_dict: {"pose": {"position": {"x", "y", "z"}, "orientation": {...}}}
            max_steps: step budget for this local navigation
            step_callback: optional callback called after each PointNav micro-step

        Returns:
            (success, observations, info)
        """
        pose = goal_pose_dict['pose']
        goal_position_ros = np.array([
            pose['position']['x'],
            pose['position']['y'],
            pose['position']['z']
        ], dtype=np.float32)

        # Extract orientation if available
        goal_orientation_habitat = None
        if 'orientation' in pose:
            from scipy.spatial.transform import Rotation as R
            # Server sends quaternion in ROS frame (x, y, z, w)
            quat_ros = [
                pose['orientation']['x'],
                pose['orientation']['y'],
                pose['orientation']['z'],
                pose['orientation']['w']
            ]

            # Convert ROS orientation to Habitat orientation
            # ROS to Habitat rotation transformation
            R_ros = R.from_quat(quat_ros)

            # Apply coordinate frame transformation
            # ROS: X=forward, Y=left, Z=up
            # Habitat: X=right, Y=up, Z=back
            C_ros_to_habitat = R.from_matrix(np.array([
                [ 0, -1,  0],
                [ 0,  0,  1],
                [-1,  0,  0]
            ]))

            R_habitat = C_ros_to_habitat * R_ros * C_ros_to_habitat.inv()
            goal_orientation_habitat = R_habitat.as_quat()  # [x, y, z, w]

            print(f'[Client] Goal orientation (ROS quat): {quat_ros}')
            print(f'[Client] Goal orientation (Habitat quat): {goal_orientation_habitat}')

        print(f'[Client] Goal pose from server (ROS): {goal_position_ros}')

        # Transform from ROS to Habitat coordinates
        # ROS: X=forward, Y=left, Z=up
        # Habitat: X=right, Y=up, Z=back (forward=-Z)
        C_ros_to_habitat = np.array([
            [ 0, -1,  0],  # Habitat X = -ROS Y
            [ 0,  0,  1],  # Habitat Y = ROS Z
            [-1,  0,  0]   # Habitat Z = -ROS X
        ])

        goal_position_habitat = C_ros_to_habitat @ goal_position_ros

        # Add ground height offset (server uses Z=0 at ground, Habitat uses Y=ground_height_offset)
        goal_position_habitat[1] += self.ground_height_offset

        print(f'[Client] Goal pose in Habitat: {goal_position_habitat}')

        # Use navigation controller
        success, observations, nav_info = self.nav_controller.navigate_to_pose(
            goal_position_habitat,
            goal_orientation_habitat,
            max_steps=max_steps,
            step_callback=step_callback
        )

        # Record final position
        agent_state = self.env.sim.get_agent_state()
        self.trajectory.append({
            'keyframe': self.keyframe_count,
            'pointnav_step_total': self.pointnav_step_total,
            'goal_position_ros': goal_position_ros.tolist(),
            'goal_position_habitat': goal_position_habitat.tolist(),
            'final_position': agent_state.position.tolist(),
            'success': success
        })

        return success, observations, nav_info

    def execute_discrete_action(
        self,
        action_type: str
    ) -> Tuple[bool, list, Dict[str, Any]]:
        """
        Execute one direct Habitat action without PointNav policy.

        Supported action_type:
          - turn_left
          - turn_right
          - move_forward
          - stop
        """
        action_map = {
            "stop": 0,
            "move_forward": 1,
            "turn_left": 2,
            "turn_right": 3,
        }
        if action_type not in action_map:
            raise ValueError(f"Unsupported direct action_type: {action_type}")

        action_id = action_map[action_type]
        obs = self.env.step(action_id)
        info = {
            "num_steps": 1,
            "actions": [action_id],
            "final_distance": -1.0,
            "success": True,
            "interrupted": False,
            "interrupt_reason": None,
            "step_budget": 1,
            "mode": "direct_action",
            "action_type": action_type,
        }
        print(f"[Client] Direct action executed: {action_type} (id={action_id})")
        return True, [obs], info

    def execute_action_sequence(
        self,
        action_ids: List[int],
        step_callback: Optional[Callable[[Dict, int, int], None]] = None
    ) -> Tuple[bool, list, Dict[str, Any]]:
        """
        Execute a full open-loop Habitat action-id sequence.

        Args:
            action_ids: list of Habitat action IDs (0/1/2/3)
            step_callback: optional callback(step_obs, step_idx, action_id)
        """
        observations = []
        executed_actions = []
        for idx, aid in enumerate(action_ids):
            action_id = int(aid)
            obs = self.env.step(action_id)
            observations.append(obs)
            executed_actions.append(action_id)
            if step_callback is not None:
                try:
                    step_callback(obs, idx, action_id)
                except Exception as e:
                    print(f"[Client] Action-sequence callback failed (continuing): {e}")
        info = {
            "num_steps": len(executed_actions),
            "actions": executed_actions,
            "success": True,
            "mode": "planned_actions_open_loop",
        }
        print(f"[Client] Open-loop action sequence executed: {len(executed_actions)} steps")
        return True, observations, info

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _quat_dict_to_yaw(q: Dict[str, float]) -> float:
        return math.atan2(
            2.0 * (q['w'] * q['z'] + q['x'] * q['y']),
            1.0 - 2.0 * (q['y'] * q['y'] + q['z'] * q['z'])
        )

    def _get_current_yaw_ros(self) -> float:
        T_cam_odom, T_odom_base = self._get_transform_matrices({})
        _ = T_cam_odom
        return float(math.atan2(T_odom_base[1, 0], T_odom_base[0, 0]))

    def align_to_goal_orientation(
        self,
        goal_pose_dict: Dict[str, Any],
        step_callback: Optional[Callable[[Dict, int, int], None]] = None,
        max_turn_steps: int = 24
    ) -> Tuple[bool, list, Dict[str, Any]]:
        """
        Align current heading to goal_pose orientation using only turn actions.
        """
        pose = goal_pose_dict.get('pose', {})
        orientation = pose.get('orientation', None)
        if orientation is None:
            return True, [], {"num_steps": 0, "actions": [], "mode": "orientation_align_none"}
        if self.env.episode_over:
            print("[Client] Skip orientation align: episode is already over")
            return True, [], {
                "num_steps": 0,
                "actions": [],
                "success": True,
                "mode": "orientation_align_skipped_episode_over",
            }

        target_yaw = float(self._quat_dict_to_yaw(orientation))
        turn_step = math.radians(30.0)
        observations = []
        actions = []

        for i in range(max_turn_steps):
            if self.env.episode_over:
                print("[Client] Stop orientation align: episode became over")
                return True, observations, {
                    "num_steps": len(actions),
                    "actions": actions,
                    "success": True,
                    "mode": "orientation_align_truncated_episode_over",
                }
            current_yaw = self._get_current_yaw_ros()
            yaw_diff = self._wrap_angle(target_yaw - current_yaw)
            if abs(yaw_diff) <= turn_step / 2.0:
                break
            action_id = 2 if yaw_diff > 0 else 3
            obs = self.env.step(action_id)
            observations.append(obs)
            actions.append(action_id)
            if step_callback is not None:
                try:
                    step_callback(obs, i, action_id)
                except Exception as e:
                    print(f"[Client] Orientation-align callback failed (continuing): {e}")

        final_yaw = self._get_current_yaw_ros()
        final_diff = abs(self._wrap_angle(target_yaw - final_yaw))
        success = final_diff <= turn_step / 2.0
        print(
            f"[Client] Orientation align: steps={len(actions)}, "
            f"target={math.degrees(target_yaw):.1f}°, final={math.degrees(final_yaw):.1f}°, "
            f"diff={math.degrees(final_diff):.1f}°, success={success}"
        )
        return success, observations, {
            "num_steps": len(actions),
            "actions": actions,
            "success": success,
            "mode": "orientation_align",
        }

    def run_navigation(self) -> bool:
        """Main navigation loop"""
        # Reset environment to load an episode
        if self.episode_id is not None:
            # Load specific episode
            print(f"[Client] Loading episode {self.episode_id}...")
            obs = self.env.reset()
            # Keep resetting until we get the desired episode
            attempts = 0
            while self.env.current_episode.episode_id != str(self.episode_id):
                obs = self.env.reset()
                attempts += 1
                if attempts > 1000:
                    raise ValueError(f"Could not find episode {self.episode_id} after {attempts} attempts")
            print(f"[Client] Loaded episode: {self.env.current_episode.episode_id}")
            print(f"[Client] Scene: {self.env.current_episode.scene_id}")
        else:
            # Random episode
            obs = self.env.reset()
            print(f"[Client] Loaded random episode: {self.env.current_episode.episode_id}")
            print(f"[Client] Scene: {self.env.current_episode.scene_id}")

        # If goal_text is None or empty, read from episode
        if not self.goal_text or self.goal_text == "":
            episode = self.env.current_episode
            if hasattr(episode, 'goals') and episode.goals:
                goal = episode.goals[0]
                if hasattr(goal, 'object_category'):
                    self.goal_text = goal.object_category
                    print(f"[Client] Using goal from episode: {self.goal_text}")
                else:
                    raise ValueError("Episode goal does not have object_category")
            else:
                raise ValueError("Episode does not have goals defined")

        # Reset server with the determined goal
        self.reset_navigation_server()

        # Set ground height offset after reset (when agent is at starting position)
        if self.ground_height_offset is None:
            agent_state = self.env.sim.get_agent_state()
            self.ground_height_offset = agent_state.position[1]  # Habitat Y coordinate
            print(f"[Client] Ground height offset set to: {self.ground_height_offset:.3f}m (Habitat Y)")

        self.iteration_count = 0
        self.keyframe_count = 0
        self.pointnav_step_total = 0
        self.server_request_count = 0
        self.trajectory = []
        self.video_frames = []

        print(f"\n[Client] Starting navigation loop")
        print(f"[Client] Goal: {self.goal_text}")
        print(f"[Client] Max PointNav steps: {self.max_pointnav_steps}")

        overall_success = False
        stop_called = False

        def counted_steps_from_nav_info(nav_info: Dict[str, Any]) -> int:
            """
            Count step budget excluding STOP actions (action_id == 0).
            Fallback to num_steps when action list is unavailable.
            """
            actions = nav_info.get("actions", None)
            if isinstance(actions, list) and len(actions) > 0:
                count = 0
                for a in actions:
                    try:
                        if int(a) != 0:
                            count += 1
                    except Exception:
                        count += 1
                return int(count)
            return int(nav_info.get("num_steps", 0))

        def finalize_episode_with_stop():
            nonlocal obs, overall_success, stop_called
            if stop_called:
                return

            stop_called = True
            print('[Client] Calling STOP action for episode evaluation')
            try:
                obs = self.env.step(0)  # 0 = STOP action
            except Exception as e:
                print(f'[Client] STOP action failed: {e}')
                return

            if self.env.episode_over:
                metrics = self.env.get_metrics()
                print('\n[Client] ===== EPISODE EVALUATION =====')
                print(f"  Success: {metrics.get('success', False)}")
                print(f"  SPL: {metrics.get('spl', 0.0):.4f}")
                print(f"  Distance to goal: {metrics.get('distance_to_goal', -1):.2f}m")
                overall_success = bool(metrics.get('success', overall_success))
                self.episode_metrics = metrics

        while self.pointnav_step_total < self.max_pointnav_steps:
            self.keyframe_count += 1
            self.iteration_count = self.keyframe_count  # for compatibility with existing visualization code

            print(f"\n[Client] === KEYFRAME {self.keyframe_count} ===")
            print(f"[Client] PointNav steps used: {self.pointnav_step_total}/{self.max_pointnav_steps}")

            try:
                # Visualize current observation
                if self.visualize:
                    self.visualize_observation(obs, f"Goal: {self.goal_text}")

                # 1) Keyframe request: ask server for next navigation goal
                keyframe_result = self.request_navigation_step(obs, is_keyframe=True)
                action_type = keyframe_result.get('action_type', 'none')
                keyframe_finished = bool(keyframe_result.get('finished', False))
                goal_pose = keyframe_result.get('goal_pose')
                planned_actions_raw = keyframe_result.get('planned_actions', [])
                planned_actions: List[int] = []
                if isinstance(planned_actions_raw, list):
                    for a in planned_actions_raw:
                        try:
                            ai = int(a)
                        except Exception:
                            continue
                        if ai in (0, 1, 2, 3):
                            planned_actions.append(ai)
                print(f'[Client] Keyframe response: action={action_type}, finished={keyframe_finished}')

                if keyframe_finished and goal_pose is None:
                    print('[Client] finished=True without goal_pose; stopping for episode evaluation')
                    overall_success = True
                    finalize_episode_with_stop()
                    break

                if action_type == 'none' and goal_pose is None:
                    print('[Client] Server returned no actionable keyframe, stopping')
                    break

                if goal_pose is None:
                    print('[Client] No goal_pose in keyframe response, stopping')
                    break

                remaining_steps = self.max_pointnav_steps - self.pointnav_step_total
                if remaining_steps <= 0:
                    print('[Client] PointNav step budget exhausted before executing keyframe goal')
                    break

                # 2) Non-keyframe requests on each PointNav micro-step
                detected_finish_in_non_keyframe = False
                finished_goal_pose_from_non_keyframe = None

                def non_keyframe_callback(step_obs, step_idx, rho, theta, action_id):
                    nonlocal detected_finish_in_non_keyframe, finished_goal_pose_from_non_keyframe

                    # Count micro-steps but exclude STOP.
                    if int(action_id) != 0:
                        self.pointnav_step_total += 1

                    try:
                        non_keyframe_result = self.request_navigation_step(step_obs, is_keyframe=False)
                    except Exception as e:
                        print(f'[Client] Non-keyframe server request failed (continuing): {e}')
                        return False

                    # In current server design, non-keyframe can only return goal_pose when finished=True.
                    if bool(non_keyframe_result.get('finished', False)):
                        detected_finish_in_non_keyframe = True
                        finished_goal_pose_from_non_keyframe = non_keyframe_result.get('goal_pose', None)
                        print('[Client] Non-keyframe update reported finished=True, interrupting PointNav')
                        return True

                    return False

                def open_loop_non_keyframe_update(step_obs, step_idx, action_id):
                    # Open-loop mode: always execute full planned sequence, no interrupt.
                    if int(action_id) != 0:
                        self.pointnav_step_total += 1
                    try:
                        _ = self.request_navigation_step(step_obs, is_keyframe=False)
                    except Exception as e:
                        print(f'[Client] Open-loop non-keyframe update failed (continuing): {e}')

                # Rotation/primitive keyframes: execute direct Habitat action.
                # Goal-pose keyframes: execute local PointNav.
                # If finished=True with goal_pose, prioritize navigating to that
                # final goal pose before issuing STOP for evaluation.
                used_planned_actions = False
                if len(planned_actions) > 0 and goal_pose is not None:
                    used_planned_actions = True
                    required_steps = sum(1 for a in planned_actions if int(a) != 0)
                    print(
                        f"[Client] Executing open-loop planned actions from server: "
                        f"{len(planned_actions)} actions ({required_steps} budgeted steps)"
                    )
                    if required_steps > remaining_steps:
                        print(
                            f"[Client] Not enough step budget for full planned action sequence: "
                            f"required={required_steps}, remaining={remaining_steps}"
                        )
                        break

                    seq_success, seq_obs, seq_info = self.execute_action_sequence(
                        planned_actions,
                        step_callback=open_loop_non_keyframe_update
                    )
                    if self.env.episode_over:
                        print('[Client] Skipping orientation align: episode is over')
                        align_success, align_obs, align_info = True, [], {
                            "num_steps": 0,
                            "actions": [],
                            "success": True,
                            "mode": "orientation_align_skipped_episode_over",
                        }
                    else:
                        align_success, align_obs, align_info = self.align_to_goal_orientation(
                            goal_pose,
                            step_callback=open_loop_non_keyframe_update
                        )
                    success = bool(seq_success and align_success)
                    step_observations = list(seq_obs) + list(align_obs)
                    nav_info = {
                        "num_steps": int(seq_info.get("num_steps", 0)) + int(align_info.get("num_steps", 0)),
                        "actions": list(seq_info.get("actions", [])) + list(align_info.get("actions", [])),
                        "success": success,
                        "interrupted": False,
                        "mode": "planned_actions_open_loop",
                    }
                elif (
                    action_type in {"turn_left", "turn_right", "move_forward", "stop"}
                    and not (keyframe_finished and goal_pose is not None)
                ):
                    success, step_observations, nav_info = self.execute_discrete_action(action_type)
                else:
                    # Execute keyframe goal using local PointNav; each micro-step triggers non-keyframe update.
                    success, step_observations, nav_info = self.execute_goal_pose(
                        goal_pose,
                        max_steps=remaining_steps,
                        step_callback=non_keyframe_callback if self.nav_controller.use_pointnav else None
                    )
                    # Always refine final heading via discrete turn actions.
                    if self.env.episode_over:
                        print('[Client] Skipping orientation align: episode is over')
                        align_success, align_obs, align_info = True, [], {
                            "num_steps": 0,
                            "actions": [],
                            "success": True,
                            "mode": "orientation_align_skipped_episode_over",
                        }
                    else:
                        align_success, align_obs, align_info = self.align_to_goal_orientation(
                            goal_pose,
                            step_callback=open_loop_non_keyframe_update if self.nav_controller.use_pointnav else None
                        )
                    success = bool(success and align_success)
                    step_observations = list(step_observations) + list(align_obs)
                    nav_info = {
                        "num_steps": int(nav_info.get("num_steps", 0)) + int(align_info.get("num_steps", 0)),
                        "actions": list(nav_info.get("actions", [])) + list(align_info.get("actions", [])),
                        "success": success,
                        "interrupted": bool(nav_info.get("interrupted", False)),
                        "mode": f"{nav_info.get('mode', 'pointnav')}_with_align",
                    }

                # Direct-action mode or teleport mode has no micro-step callback.
                if used_planned_actions:
                    # Already counted per-step in open_loop_non_keyframe_update.
                    pass
                elif action_type in {"turn_left", "turn_right", "move_forward", "stop"}:
                    self.pointnav_step_total += counted_steps_from_nav_info(nav_info)
                elif not self.nav_controller.use_pointnav:
                    self.pointnav_step_total += counted_steps_from_nav_info(nav_info)

                print(
                    f"[Client] PointNav result: success={success}, "
                    f"steps_this_goal={nav_info.get('num_steps', 0)}, "
                    f"total_steps={self.pointnav_step_total}/{self.max_pointnav_steps}, "
                    f"interrupted={nav_info.get('interrupted', False)}"
                )

                # Visualize during navigation
                if self.visualize and step_observations:
                    for step_obs in step_observations[::5]:  # Show every 5th frame
                        self.visualize_observation(step_obs, "PointNav micro-steps (non-keyframe updates)")

                # Use latest observation for next keyframe
                if step_observations:
                    obs = step_observations[-1]
                    if not isinstance(obs, dict):
                        print(f'[Client] Invalid observation type: {type(obs)}, resetting environment')
                        obs = self.env.reset()

                if detected_finish_in_non_keyframe:
                    if finished_goal_pose_from_non_keyframe is None:
                        print('[Client] finished=True on non-keyframe but no goal_pose returned, stopping')
                        break

                    remaining_final_steps = self.max_pointnav_steps - self.pointnav_step_total
                    if remaining_final_steps <= 0:
                        print('[Client] Step budget exhausted before final finished-goal navigation')
                        break

                    def final_approach_non_keyframe_callback(step_obs, step_idx, rho, theta, action_id):
                        # Keep visual/map updates during final approach; do not interrupt.
                        if int(action_id) != 0:
                            self.pointnav_step_total += 1
                        try:
                            _ = self.request_navigation_step(step_obs, is_keyframe=False)
                        except Exception as e:
                            print(f'[Client] Final-approach non-keyframe update failed (continuing): {e}')
                        return False

                    print('[Client] Navigating to finished goal_pose from non-keyframe response...')
                    final_success, final_obs, final_nav_info = self.execute_goal_pose(
                        finished_goal_pose_from_non_keyframe,
                        max_steps=remaining_final_steps,
                        step_callback=final_approach_non_keyframe_callback if self.nav_controller.use_pointnav else None
                    )
                    if not self.nav_controller.use_pointnav:
                        self.pointnav_step_total += counted_steps_from_nav_info(final_nav_info)

                    if self.visualize and final_obs:
                        for step_obs in final_obs[::5]:
                            self.visualize_observation(step_obs, "Final approach to finished goal_pose")

                    if final_obs:
                        obs = final_obs[-1]

                    if final_success:
                        overall_success = True
                    else:
                        overall_success = False
                        print('[Client] Failed to reach finished goal_pose; sending STOP for evaluation anyway')
                    finalize_episode_with_stop()
                    break

                # Keyframe response may already indicate final approach task
                if keyframe_finished:
                    if not success:
                        print('[Client] keyframe finished=true but local navigation failed; sending STOP for evaluation')
                    overall_success = bool(success)
                    finalize_episode_with_stop()
                    break

                # Normal progress: reached this keyframe goal, request next keyframe.
                if success:
                    print('[Client] Reached keyframe navigation goal; requesting next keyframe')
                    continue

                # Failed to reach current goal (excluding finished interruption)
                print('[Client] Failed to reach keyframe navigation goal, stopping')
                break

            except Exception as e:
                print(f'[Client] Navigation step failed: {e}')
                import traceback
                traceback.print_exc()
                break

        # Save video if enabled
        if self.save_video and self.video_frames:
            self._save_video()

        # Close visualization window
        if self.visualize:
            cv2.destroyAllWindows()

        print(
            f"\n[Client] Navigation completed after {self.keyframe_count} keyframes, "
            f"{self.pointnav_step_total} PointNav micro-steps"
        )
        return overall_success

    def _save_video(self):
        """Save recorded frames as video"""
        if not self.video_frames:
            return

        video_path = os.path.join(self.output_dir, "navigation.mp4")
        height, width = self.video_frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

        for frame in self.video_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()
        print(f"[Client] Video saved to: {video_path}")


def create_habitat_env(config_path: str, scene_path: Optional[str] = None) -> Env:
    """Create habitat environment with necessary configuration"""

    # Load base config
    config = habitat.get_config(config_path)

    with habitat.config.read_write(config):
        # Set scene if provided
        if scene_path:
            config.habitat.dataset.data_path = scene_path

        # Configure sensors
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 512
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 512
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = 512
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = 512

        # Add VelocityAction
        config.habitat.task.actions.velocity_control = VelocityControlActionConfig(
            type="VelocityAction",
            lin_vel_range=[0.0, 0.25],
            ang_vel_range=[-10.0, 10.0],
            min_abs_lin_speed=0.0,  # Allow zero linear velocity for pure rotation
            min_abs_ang_speed=0.0,  # Allow zero angular velocity for straight motion
            time_step=1.0
        )

        # Set max episode steps
        config.habitat.environment.max_episode_steps = 1000

    # Create environment
    env = habitat.Env(config=config)

    return env


def main():
    parser = argparse.ArgumentParser(description="Habitat-Lab LLM Navigation Client")
    parser.add_argument("--config", type=str,
                       default="benchmark/nav/pointnav/pointnav_hm3d.yaml",
                       help="Path to habitat config")
    parser.add_argument("--scene", type=str, default=None,
                       help="Path to scene dataset (optional)")
    parser.add_argument("--goal", type=str, default=None,
                       help="Goal object/location (if not specified, use episode's goal)")
    parser.add_argument("--server", type=str, default="http://10.19.126.158:1874",
                       help="LLM navigation server URL")
    parser.add_argument("--max_pointnav_steps", type=int, default=500,
                       help="Maximum PointNav micro-steps (hard budget)")
    parser.add_argument("--max_iterations", type=int, default=None,
                       help="DEPRECATED alias of --max_pointnav_steps")
    parser.add_argument("--camera_height", type=float, default=1.5,
                       help="Camera height above base")
    parser.add_argument("--output", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--no_viz", action="store_true",
                       help="Disable visualization")
    parser.add_argument("--save_video", action="store_true",
                       help="Save navigation video")
    parser.add_argument("--save_data", action="store_true",
                       help="Save navigation data for offline testing")
    parser.add_argument("--episode_id", type=int, default=None,
                       help="Specific episode ID to run (default: random)")
    parser.add_argument("--use_pointnav", action="store_true",
                       help="Use PointNav policy for navigation (like VLFM)")
    parser.add_argument("--pointnav_policy", type=str, default="data/pointnav_weights.pth",
                       help="Path to PointNav policy weights")

    args = parser.parse_args()

    # Backward compatibility: --max_iterations maps to pointnav step budget.
    if args.max_iterations is not None:
        print("[Client] Warning: --max_iterations is deprecated; use --max_pointnav_steps")
        args.max_pointnav_steps = int(args.max_iterations)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create environment
    print("[Client] Creating Habitat environment...")
    env = create_habitat_env(args.config, args.scene)

    # Create client
    client = HabitatLLMNavClient(
        env=env,
        server_url=args.server,
        goal_text=args.goal,
        camera_height=args.camera_height,
        max_pointnav_steps=args.max_pointnav_steps,
        visualize=not args.no_viz,
        save_video=args.save_video,
        output_dir=args.output,
        save_data=args.save_data,
        episode_id=args.episode_id,
        use_pointnav=args.use_pointnav,
        pointnav_policy_path=args.pointnav_policy
    )

    # Run navigation
    success = client.run_navigation()

    # Save trajectory and evaluation metrics
    trajectory_path = os.path.join(args.output, "trajectory.json")
    with open(trajectory_path, 'w') as f:
        result_data = {
            'goal': args.goal,
            'success': success,
            'iterations': client.keyframe_count,
            'keyframes': client.keyframe_count,
            'pointnav_steps': client.pointnav_step_total,
            'trajectory': client.trajectory
        }

        # Add episode evaluation metrics if available
        if client.episode_metrics:
            result_data['evaluation'] = {
                'success': client.episode_metrics.get('success', False),
                'spl': client.episode_metrics.get('spl', 0.0),
                'distance_to_goal': client.episode_metrics.get('distance_to_goal', -1),
                'soft_spl': client.episode_metrics.get('soft_spl', 0.0),
                'num_steps': client.episode_metrics.get('num_steps', -1)
            }

        json.dump(result_data, f, indent=2)

    print(f"\n[Client] Trajectory saved to: {trajectory_path}")
    print(f"[Client] Success: {success}")
    print(f"[Client] Total keyframes: {client.keyframe_count}")
    print(f"[Client] Total PointNav steps: {client.pointnav_step_total}")

    # Print evaluation summary if available
    if client.episode_metrics:
        print(f"\n[Client] ===== FINAL EVALUATION =====")
        print(f"  Habitat Success: {client.episode_metrics.get('success', False)}")
        print(f"  SPL: {client.episode_metrics.get('spl', 0.0):.4f}")
        print(f"  Soft SPL: {client.episode_metrics.get('soft_spl', 0.0):.4f}")
        print(f"  Distance to goal: {client.episode_metrics.get('distance_to_goal', -1):.2f}m")
        print(f"  Number of steps: {client.episode_metrics.get('num_steps', -1)}")

    env.close()


if __name__ == "__main__":
    main()
