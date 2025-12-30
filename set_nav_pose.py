#!/usr/bin/env python3 
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState


class NavPoseSetter(Node):    
    def __init__(self):
        super().__init__('nav_pose_setter')
        
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
        
        # Wait for publishers to be ready
        self.get_logger().info('Waiting for publishers to be ready...')
        time.sleep(0.5)
        
        self.get_logger().info('Nav pose setter initialized.')
    
    def sleep_with_spin(self, duration_sec: float, step_sec: float = 0.05):
        """Sleep while spinning ROS"""
        t_end = time.time() + float(duration_sec)
        while rclpy.ok() and time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=step_sec)
    
    def set_nav_pose(self):
        """Set robot to navigation pose"""        
        # Set torso
        torso = JointState()
        torso.position = [1.18, -2.10, -0.9, -0.1]
        self.torso_joint_state_pub.publish(torso)
        self.sleep_with_spin(1.0)
        
        # Set right arm
        right = JointState()
        right.position = [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
        self.right_joint_state_pub.publish(right)
        self.sleep_with_spin(1.0)
        
        # Set left arm
        left = JointState()
        left.position = [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
        self.left_joint_state_pub.publish(left)
        self.sleep_with_spin(1.0)
        

def main():
    rclpy.init()
    
    node = NavPoseSetter()
    
    try:
        node.set_nav_pose()
        # Wait a bit more to ensure commands are sent
        node.sleep_with_spin(0.5)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    except Exception as e:
        node.get_logger().error(f'Error: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

