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
from typing import Dict, Any, Optional, Tuple

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
    """Simple navigation controller to reach goal poses"""

    def __init__(self, env: Env, linear_speed: float = 0.25, angular_speed: float = 10.0):
        self.env = env
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.dist_threshold = 0.3  # meters
        self.angle_threshold = 0.1  # radians

    def navigate_to_pose(
        self,
        goal_position: np.ndarray,
        goal_orientation: Optional[np.ndarray] = None,
        max_steps: int = 100
    ) -> Tuple[bool, list]:
        """
        Navigate to goal pose by directly setting agent state

        Args:
            goal_position: [x, y, z] target position in Habitat coordinates
            goal_orientation: [x, y, z, w] target quaternion in Habitat coordinates (optional)
            max_steps: maximum steps to reach goal (not used in direct mode)

        Returns:
            (success, observations): whether reached goal and list of observations
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
            print(f'[Controller] Setting pose: position={goal_position}, orientation={goal_orientation}')
        else:
            print(f'[Controller] Setting position: {goal_position}')

        # Apply the new state
        self.env.sim.set_agent_state(agent_state.position, agent_state.rotation)

        # Get observation after teleporting
        obs = self.env.sim.get_sensor_observations()

        return True, [obs]


class HabitatLLMNavClient:
    """Client that connects Habitat-Lab with your LLM navigation server"""

    def __init__(
        self,
        env: Env,
        server_url: str,
        goal_text: str,
        camera_height: float = 1.5,
        max_iterations: int = 50,
        visualize: bool = True,
        save_video: bool = False,
        output_dir: str = "./output",
        save_data: bool = False,
        episode_id: Optional[int] = None
    ):
        self.env = env
        self.server_url = server_url.rstrip('/')
        self.goal_text = goal_text
        self.camera_height = camera_height
        self.max_iterations = max_iterations
        self.visualize = visualize
        self.save_video = save_video
        self.output_dir = output_dir
        self.save_data = save_data
        self.episode_id = episode_id  # Specific episode to run

        self.iteration_count = 0
        self.trajectory = []
        self.video_frames = []
        self.episode_metrics = None  # Store episode evaluation metrics

        # Create navigation controller
        self.nav_controller = SimpleNavigationController(env)

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

            # Find RGB sensor from agent sim_sensors
            rgb_sensor = None
            if hasattr(agent_config, 'sim_sensors'):
                for sensor_name in agent_config.sim_sensors:
                    sensor_spec = agent_config.sim_sensors[sensor_name]
                    if 'rgb' in sensor_name.lower():
                        rgb_sensor = sensor_spec
                        break

            if rgb_sensor is None:
                # Fallback: use default values
                print("[Client] Warning: Could not find RGB sensor config, using defaults")
                width = 512
                height = 512
                hfov = 79
            else:
                width = rgb_sensor.width
                height = rgb_sensor.height
                hfov = rgb_sensor.hfov

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

        # Camera position: Get actual camera position and orientation from sensor
        # The sensor position in config is relative to agent base in Habitat frame
        # We need to transform it to world frame
        sim_config = self.env.sim.habitat_config
        agent_names = list(sim_config.agents.keys())
        cam_offset_habitat = np.array([0.0, 1.2, 0.0])  # Default
        cam_orientation_habitat = np.array([0.0, 0.0, 0.0])  # Default: no rotation

        if agent_names and hasattr(sim_config.agents[agent_names[0]], 'sim_sensors'):
            agent_config = sim_config.agents[agent_names[0]]
            # Find RGB sensor to get camera position and orientation
            for sensor_name in agent_config.sim_sensors:
                if 'rgb' in sensor_name.lower():
                    sensor = agent_config.sim_sensors[sensor_name]
                    if hasattr(sensor, 'position'):
                        cam_offset_habitat = np.array(sensor.position, dtype=np.float32)
                    if hasattr(sensor, 'orientation'):
                        cam_orientation_habitat = np.array(sensor.orientation, dtype=np.float32)
                    break

        # Camera position: offset is in agent's local frame, need to rotate to world frame
        # Transform offset from agent local frame to world frame using agent rotation
        cam_offset_world = R_habitat @ cam_offset_habitat
        cam_pos_habitat = pos_habitat + cam_offset_world

        # Apply ground height offset
        cam_pos_habitat_adjusted = cam_pos_habitat.copy()
        cam_pos_habitat_adjusted[1] -= self.ground_height_offset

        # Get camera rotation: agent rotation + sensor orientation
        # Sensor orientation is euler angles (pitch, yaw, roll) in Habitat frame
        from scipy.spatial.transform import Rotation as R_scipy
        R_sensor_local = R_scipy.from_euler('xyz', cam_orientation_habitat).as_matrix()
        R_cam_habitat = R_habitat @ R_sensor_local

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

        # Add iteration counter
        cv2.putText(
            vis,
            f"Iteration: {self.iteration_count}",
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

    def request_navigation_step(self, obs: Dict) -> Dict:
        """Send observation to server and get navigation decision"""
        url = f'{self.server_url}/navigation_step'

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
            'T_odom_base': json.dumps(T_odom_base.flatten().tolist())
        }

        # Save data if enabled
        if self.save_data:
            iter_dir = os.path.join(self.data_dir, f"iter_{self.iteration_count:03d}")
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
                'iteration': self.iteration_count,
                'goal': self.goal_text,
                'intrinsic': self.intrinsic.tolist(),
                'T_cam_odom': T_cam_odom.tolist(),
                'T_odom_base': T_odom_base.tolist()
            }
            with open(os.path.join(iter_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f'[Client] Saved data to: {iter_dir}')

        print('[Client] Sending observation to server...')
        try:
            response = requests.post(url, files=files, data=data, timeout=600)
            response.raise_for_status()
            result = response.json()
            return result
        except Exception as e:
            print(f'[Client] Server request failed: {e}')
            raise

    def execute_goal_pose(self, goal_pose_dict: Dict) -> Tuple[bool, list]:
        """
        Execute navigation to goal pose using velocity control

        Args:
            goal_pose_dict: {"pose": {"position": {"x", "y", "z"}, "orientation": {...}}}

        Returns:
            (success, observations)
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
        success, observations = self.nav_controller.navigate_to_pose(
            goal_position_habitat,
            goal_orientation_habitat
        )

        # Record final position
        agent_state = self.env.sim.get_agent_state()
        self.trajectory.append({
            'iteration': self.iteration_count,
            'goal_position_ros': goal_position_ros.tolist(),
            'goal_position_habitat': goal_position_habitat.tolist(),
            'final_position': agent_state.position.tolist(),
            'success': success
        })

        return success, observations

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
        self.trajectory = []
        self.video_frames = []

        print(f"\n[Client] Starting navigation loop")
        print(f"[Client] Goal: {self.goal_text}")
        print(f"[Client] Max iterations: {self.max_iterations}")

        overall_success = False

        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            print(f"\n[Client] === ITERATION {self.iteration_count} ===")

            try:
                # Visualize current observation
                if self.visualize:
                    self.visualize_observation(obs, f"Goal: {self.goal_text}")

                # 1. Get navigation decision from server
                result = self.request_navigation_step(obs)

                action_type = result.get('action_type', 'none')
                print(f'[Client] Server response: action={action_type}')

                if action_type == 'none':
                    print('[Client] Server returned no action, stopping')
                    break

                # 2. Execute goal pose
                if 'goal_pose' in result:
                    is_final_goal = result.get('finished', False)
                    success, step_observations = self.execute_goal_pose(result['goal_pose'])

                    # Visualize during navigation
                    if self.visualize and step_observations:
                        for step_obs in step_observations[::5]:  # Show every 5th frame
                            self.visualize_observation(step_obs, f"Navigating to goal...")

                    # Use last observation for next iteration
                    if step_observations:
                        obs = step_observations[-1]

                        # Validate observation is a dictionary
                        if not isinstance(obs, dict):
                            print(f'[Client] Invalid observation type: {type(obs)}, resetting environment')
                            obs = self.env.reset()

                    if success:
                        print('[Client] Reached goal!')
                        overall_success = True

                        # Check if this is the final goal
                        if is_final_goal:
                            print('[Client] Server indicated task finished, calling STOP action')

                            # Call STOP action to properly end the episode
                            obs = self.env.step(0)  # 0 = STOP action

                            # Check if episode is done (Habitat v0.2.x API)
                            done = self.env.episode_over

                            if done:
                                # Get evaluation metrics
                                metrics = self.env.get_metrics()

                                print('\n[Client] ===== EPISODE EVALUATION =====')
                                print(f"  Success: {metrics.get('success', False)}")
                                print(f"  SPL: {metrics.get('spl', 0.0):.4f}")
                                print(f"  Distance to goal: {metrics.get('distance_to_goal', -1):.2f}m")

                                # Update overall success based on Habitat's evaluation
                                overall_success = metrics.get('success', False)

                                # Store evaluation metrics
                                self.episode_metrics = metrics

                            break
                else:
                    print('[Client] No goal_pose in server response')
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

        print(f"\n[Client] Navigation completed after {self.iteration_count} iterations")
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
    parser.add_argument("--max_iterations", type=int, default=500,
                       help="Maximum navigation iterations")
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

    args = parser.parse_args()

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
        max_iterations=args.max_iterations,
        visualize=not args.no_viz,
        save_video=args.save_video,
        output_dir=args.output,
        save_data=args.save_data,
        episode_id=args.episode_id
    )

    # Run navigation
    success = client.run_navigation()

    # Save trajectory and evaluation metrics
    trajectory_path = os.path.join(args.output, "trajectory.json")
    with open(trajectory_path, 'w') as f:
        result_data = {
            'goal': args.goal,
            'success': success,
            'iterations': client.iteration_count,
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
    print(f"[Client] Total iterations: {client.iteration_count}")

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
