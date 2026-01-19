#!/usr/bin/python3
"""
ROS2 Client for LLM Navigation Server
机器人端负责数据采集、HTTP请求、执行Path
"""
import os
import sys
import time
import math
import json
from typing import Optional, Dict, Any
from io import BytesIO

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration

import numpy as np
import cv2
import requests
from PIL import Image
from cv_bridge import CvBridge

from sensor_msgs.msg import Image as ImageMsg, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose

from tf2_ros import Buffer, TransformListener, TransformException
from scipy.spatial.transform import Rotation as R


def transform_to_matrix(transform) -> np.ndarray:
    """Convert TransformStamped to 4x4 homogeneous transformation matrix"""
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    quat = np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64)
    rot_matrix = R.from_quat(quat).as_matrix()

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rot_matrix.astype(np.float32)
    matrix[:3, 3] = np.array([translation.x, translation.y, translation.z], dtype=np.float32)
    return matrix


def dict_to_quaternion(q_dict: Dict[str, float]) -> Quaternion:
    """Convert quaternion dict to ROS Quaternion message"""
    q = Quaternion()
    q.x = q_dict['x']
    q.y = q_dict['y']
    q.z = q_dict['z']
    q.w = q_dict['w']
    return q


class LLMNavClient(Node):
    """
    ROS2 Client for LLM Navigation
    - Subscribe to sensors (RGB, Depth, TF)
    - Send HTTP requests to server
    - Execute returned goal pose with NavigateToPose action (using Nav2 path planning)
    - Measure and report timing (network transfer, server processing, detection, VLM)
    """
    
    def __init__(self, goal: str, goal_description: str = "", server_url: str = "http://10.19.126.158:1874"):
        super().__init__('llm_nav_client')
        
        self.goal = goal
        self.goal_description = goal_description
        self.server_url = server_url.rstrip('/')
        self.iteration_count = 0
        
        self.get_logger().info(f'Goal: {self.goal}')
        if self.goal_description:
            self.get_logger().info(f'Goal Description: {self.goal_description}')
        self.get_logger().info(f'Server URL: {self.server_url}')
        
        # Frame names
        self.camera_frame = 'camera_head_left_link'
        self.base_frame = 'base_link'
        self.odom_frame = 'odom'
        
        # ROS helpers
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publishers for joint control
        pub_qos = QoSProfile(depth=10)
        self.torso_joint_state_pub = self.create_publisher(
            JointState, '/motion_target/target_joint_state_torso', pub_qos
        )
        self.right_joint_state_pub = self.create_publisher(
            JointState, '/motion_target/target_joint_state_arm_right', pub_qos
        )
        self.left_joint_state_pub = self.create_publisher(
            JointState, '/motion_target/target_joint_state_arm_left', pub_qos
        )
        
        # Nav2 action client
        self.navigate_to_pose_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        
        # Sensor data buffers
        from threading import Lock
        self._lock = Lock()
        self._latest_depth_info: Optional[CameraInfo] = None
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_rgb_stamp: Optional[Time] = None
        self._latest_depth_stamp: Optional[Time] = None
        
        # Subscriptions
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        self.depth_info_sub = self.create_subscription(
            CameraInfo, '/hdas/camera_head/depth/camera_info',
            self.depth_info_callback, sensor_qos
        )
        self.rgb_sub = self.create_subscription(
            ImageMsg, '/hdas/camera_head/rgb/image_rect_color',
            self.rgb_callback, sensor_qos
        )
        self.depth_sub = self.create_subscription(
            ImageMsg, '/hdas/camera_head/depth/depth_registered',
            self.depth_callback, sensor_qos
        )
        
        self.get_logger().info('Client node initialized successfully.')
    
    # ========== Callbacks ==========
    def depth_info_callback(self, msg: CameraInfo):
        with self._lock:
            self._latest_depth_info = msg
    
    def rgb_callback(self, msg: ImageMsg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            stamp = Time.from_msg(msg.header.stamp)
        except Exception as e:
            self.get_logger().warn(f'RGB callback failed: {e}')
            return
        
        with self._lock:
            self._latest_rgb = rgb
            self._latest_rgb_stamp = stamp
    
    def depth_callback(self, msg: ImageMsg):
        try:
            if msg.encoding == '32FC1':
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1').astype(np.float32)
            elif msg.encoding == '16UC1':
                depth_u16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                depth = depth_u16.astype(np.float32) * 0.001
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough').astype(np.float32)
            
            depth[~np.isfinite(depth)] = 0.0
            stamp = Time.from_msg(msg.header.stamp)
        except Exception as e:
            self.get_logger().warn(f'Depth callback failed: {e}')
            return
        
        with self._lock:
            self._latest_depth = depth
            self._latest_depth_stamp = stamp
    
    # ========== Utilities ==========
    def sleep_with_spin(self, duration_sec: float, step_sec: float = 0.05):
        """Sleep while spinning ROS"""
        t_end = time.time() + float(duration_sec)
        while rclpy.ok() and time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=step_sec)
    
    def wait_first_obs(self, timeout_sec: float = 10.0):
        """Wait until first observation is available"""
        start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            with self._lock:
                ok = (self._latest_depth_info is not None and
                      self._latest_rgb is not None and
                      self._latest_depth is not None)
            if ok:
                return
            if (time.time() - start) > timeout_sec:
                raise RuntimeError('Timeout waiting for first observation')
    
    def get_obs_snapshot(self) -> Dict[str, Any]:
        """Get current observation snapshot with TF transforms"""
        with self._lock:
            if self._latest_depth_info is None or self._latest_rgb is None or self._latest_depth is None:
                raise RuntimeError('Observation not ready')
            rgb = np.copy(self._latest_rgb)
            depth = np.copy(self._latest_depth)
            depth_info = self._latest_depth_info
        
        K = np.array(depth_info.k, dtype=np.float32).reshape(3, 3)
        
        try:
            t = rclpy.time.Time()  # latest
            odom_to_cam = self.tf_buffer.lookup_transform(
                self.camera_frame, self.odom_frame, t, timeout=Duration(seconds=1.0)
            )
            base_to_odom = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, t, timeout=Duration(seconds=1.0)
            )
        except TransformException as e:
            raise RuntimeError(f'TF lookup failed: {e}')
        
        T_cam_odom = transform_to_matrix(odom_to_cam)
        T_odom_base = transform_to_matrix(base_to_odom)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'intrinsic': K,
            'T_cam_odom': T_cam_odom,
            'T_odom_base': T_odom_base
        }
    
    # ========== Robot Control ==========
    def pre_nav(self):
        """Set robot to navigation pose"""
        torso = JointState()
        torso.position = [1.18, -2.10, -0.9, -0.1]
        self.torso_joint_state_pub.publish(torso)
        self.sleep_with_spin(1.0)
        
        right = JointState()
        right.position = [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
        self.right_joint_state_pub.publish(right)
        self.sleep_with_spin(1.0)

        left = JointState()
        left.position = [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
        self.left_joint_state_pub.publish(left)
        self.sleep_with_spin(1.0)
    
    def pre_manip(self):
        """Set robot to manipulation pose"""
        torso = JointState()
        torso.position = [0.0, 0.0, -0.5, 0.0]
        self.torso_joint_state_pub.publish(torso)
        
        right = JointState()
        right.position = [0.0, -1.57, 0.0, -1.57, -1.57, 0.0, 0.0]
        self.right_joint_state_pub.publish(right)
        
        left = JointState()
        left.position = [0.0, 1.57, 0.0, -1.57, 1.57, 0.0, 0.0]
        self.left_joint_state_pub.publish(left)
        
        self.sleep_with_spin(1.0)
    
    # ========== Path Execution ==========
    def navigate_to_pose(self, pose_stamped: PoseStamped):
        """Navigate to a goal pose using Nav2 NavigateToPose action"""
        self.get_logger().info('Waiting for NavigateToPose action server...')
        if not self.navigate_to_pose_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('NavigateToPose action server not available')
        
        goal = NavigateToPose.Goal()
        goal.pose = pose_stamped
        goal.behavior_tree = ""
        
        self.get_logger().info(f'Sending NavigateToPose to ({pose_stamped.pose.position.x:.2f}, '
                              f'{pose_stamped.pose.position.y:.2f})')
        send_goal_future = self.navigate_to_pose_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            raise RuntimeError('NavigateToPose goal rejected')
        
        self.get_logger().info('Goal accepted, navigating...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        wrapped = result_future.result()
        if wrapped.status != 4:  # SUCCEEDED
            raise RuntimeError(f'NavigateToPose failed with status: {wrapped.status}')
        
        self.get_logger().info('NavigateToPose completed!')
    
    def dict_to_pose_stamped(self, pose_dict: dict, frame_id: str = 'odom') -> PoseStamped:
        """Convert pose dict to ROS PoseStamped message"""
        ps = PoseStamped()
        now = self.get_clock().now().to_msg()
        ps.header.frame_id = frame_id
        ps.header.stamp = now
        ps.pose.position.x = pose_dict['position']['x']
        ps.pose.position.y = pose_dict['position']['y']
        ps.pose.position.z = pose_dict['position']['z']
        ps.pose.orientation = dict_to_quaternion(pose_dict['orientation'])
        return ps
    
    # ========== Server Communication ==========
    def reset_navigation(self):
        """Call server to reset navigation"""
        url = f'{self.server_url}/navigation_reset'
        data = {
            'goal': self.goal,
            'goal_description': self.goal_description,
            'confidence_threshold': 0.5
        }
        
        self.get_logger().info(f'Resetting navigation on server: {url}')
        try:
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            self.get_logger().info(f'Server reset: {result}')
            return result
        except Exception as e:
            self.get_logger().error(f'Failed to reset navigation: {e}')
            raise
    
    def request_navigation_step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Send observation to server and get navigation decision"""
        url = f'{self.server_url}/navigation_step'
        
        # Prepare RGB image (JPEG)
        rgb = obs['rgb']
        rgb_pil = Image.fromarray(rgb)
        rgb_buffer = BytesIO()
        rgb_pil.save(rgb_buffer, format='JPEG', quality=95)
        rgb_buffer.seek(0)
        
        # Prepare Depth image (PNG 16-bit)
        depth = obs['depth']
        depth_mm = (depth * 1000.0).astype(np.uint16)
        depth_pil = Image.fromarray(depth_mm, mode='I;16')
        depth_buffer = BytesIO()
        depth_pil.save(depth_buffer, format='PNG')
        depth_buffer.seek(0)
        
        # Prepare form data
        files = {
            'rgb': ('rgb.jpg', rgb_buffer, 'image/jpeg'),
            'depth': ('depth.png', depth_buffer, 'image/png')
        }
        
        data = {
            'intrinsic': json.dumps(obs['intrinsic'].flatten().tolist()),
            'T_cam_odom': json.dumps(obs['T_cam_odom'].flatten().tolist()),
            'T_odom_base': json.dumps(obs['T_odom_base'].flatten().tolist())
        }
        
        self.get_logger().info('Sending observation to server...')
        try:
            response = requests.post(url, files=files, data=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result
        except Exception as e:
            self.get_logger().error(f'Server request failed: {e}')
            raise
    
    # ========== Main Loop ==========
    def run(self):
        """Main navigation loop"""
        
        # Wait for sensors
        self.get_logger().info('Waiting for first observation...')
        self.wait_first_obs(timeout_sec=10.0)
        self.get_logger().info('First observation ready.')
        
        # Reset server
        self.reset_navigation()
        
        # Main loop
        while rclpy.ok():
            self.iteration_count += 1
            self.get_logger().info(f'\n=== ITERATION {self.iteration_count} ===')
            
            # Small delay
            self.sleep_with_spin(2.0)
            
            # Get observation
            try:
                obs = self.get_obs_snapshot()
            except Exception as e:
                self.get_logger().error(f'Failed to get observation: {e}')
                continue
            
            # Request decision from server with timing
            try:
                t_request_start = time.time()
                result = self.request_navigation_step(obs)
                t_request_end = time.time()
                total_request_time = t_request_end - t_request_start
            except Exception as e:
                self.get_logger().error(f'Server request failed: {e}')
                continue
            
            # Parse result and timing information
            action_type = result.get('action_type', 'none')
            goal_pose_data = result.get('goal_pose')
            detected = result.get('detected', False)
            score = result.get('detection_score', 0.0)
            finished = result.get('finished', False)
            timing_info = result.get('timing', {})
            
            # Calculate network transfer time
            server_time = timing_info.get('total_server_time', 0.0)
            network_time = total_request_time - server_time
            
            self.get_logger().info(f'Server response: action={action_type}, detected={detected}, score={score:.3f}')
            self.get_logger().info(f'Timing - Total: {total_request_time:.3f}s, Server: {server_time:.3f}s, '
                                  f'Network: {network_time:.3f}s ({network_time/total_request_time*100:.1f}%)')
            self.get_logger().info(f'  └─ Detection: {timing_info.get("detection_time", 0.0):.3f}s, '
                                  f'VLM Projection: {timing_info.get("vlm_projection_time", 0.0):.3f}s, '
                                  f'VLM Inference: {timing_info.get("vlm_inference_time", 0.0):.3f}s')
            
            # Handle no action
            if action_type == 'none' or goal_pose_data is None:
                self.get_logger().warn('No action from server, continuing...')
                continue
            
            # Extract goal pose
            frame_id = goal_pose_data.get('frame_id', 'odom')
            pose_dict = goal_pose_data.get('pose')
            
            if pose_dict is None:
                self.get_logger().warn('No pose in goal_pose data, continuing...')
                continue
            
            # Convert to PoseStamped
            goal_pose_stamped = self.dict_to_pose_stamped(pose_dict, frame_id=frame_id)
            
            # Handle visual servo (target reached)
            if action_type == 'visual_servo':
                self.get_logger().info('Visual servo triggered -> navigating to final position')
                try:
                    self.navigate_to_pose(goal_pose_stamped)
                    self.get_logger().info('Visual servo completed!')
                    # self.pre_manip()
                    self.get_logger().info('Navigation finished!')
                    break
                except Exception as e:
                    self.get_logger().error(f'Failed to execute visual servo: {e}')
                    continue
            
            # Handle turn around
            if action_type == 'turn_around':
                self.get_logger().info('Executing TURN AROUND (180 degrees)')
                try:
                    self.navigate_to_pose(goal_pose_stamped)
                    self.get_logger().info('Turn around completed.')
                except Exception as e:
                    self.get_logger().error(f'Failed to execute turn around: {e}')
                continue
            
            # Handle normal navigation step
            if action_type == 'nav_step':
                self.get_logger().info(f'Executing NAV STEP to ({pose_dict["position"]["x"]:.2f}, '
                                      f'{pose_dict["position"]["y"]:.2f})')
                try:
                    self.navigate_to_pose(goal_pose_stamped)
                    self.get_logger().info('Nav step completed.')
                except Exception as e:
                    self.get_logger().error(f'Failed to execute nav step: {e}')
                continue
        
        self.get_logger().info('Run finished.')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Navigation ROS2 Client')
    parser.add_argument('--goal', type=str, default=None, help='Navigation goal')
    parser.add_argument('--goal-description', type=str, default=None, 
                       help='Optional description of the goal location (e.g., "it may be on the green table in the corner of the room")')
    parser.add_argument('--server', type=str, default='http://10.19.126.158:1874', 
                       help='Server URL')
    args = parser.parse_args()
    
    print("=" * 60)
    print("LLM Navigation Client (ROS2)")
    print("=" * 60)
    
    if args.goal is None:
        goal = input("Please enter the navigation goal: ").strip()
    else:
        goal = args.goal
    
    if args.goal_description is None:
        goal_description = input("Please enter goal description (optional, press Enter to skip): ").strip()
    else:
        goal_description = args.goal_description
    
    rclpy.init()
    node = LLMNavClient(goal=goal, goal_description=goal_description, server_url=args.server)
    
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

