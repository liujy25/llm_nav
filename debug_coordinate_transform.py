#!/usr/bin/env python3
"""
调试脚本：验证坐标变换和BEV地图的一致性
"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def extract_yaw(tf_matrix: np.ndarray) -> float:
    """从4x4变换矩阵提取yaw角"""
    return np.arctan2(tf_matrix[1, 0], tf_matrix[0, 0])

def visualize_coordinate_systems():
    """可视化不同坐标系的关系"""
    
    # 假设机器人在原点，朝向+X方向（yaw=0）
    print("=" * 80)
    print("场景1: 机器人在原点，朝向+X方向（标准ROS odom坐标系）")
    print("=" * 80)
    
    # ROS标准odom坐标系：X前，Y左，Z上
    base_to_odom_x_forward = np.eye(4)
    yaw = extract_yaw(base_to_odom_x_forward)
    print(f"base_to_odom (朝向+X):\n{base_to_odom_x_forward[:3, :3]}")
    print(f"yaw = {np.rad2deg(yaw):.1f}度")
    print(f"机器人位置: (0, 0)")
    print(f"机器人朝向: +X方向（东）")
    
    # BEV地图约定：图像右侧=+X，图像上方=+Y
    # 如果机器人朝向+X，在BEV地图上应该朝右
    print(f"\n在BEV地图上：")
    print(f"  - 机器人应该在地图中心")
    print(f"  - 机器人应该朝向右侧（+X方向）")
    print(f"  - FOV扇形应该指向右侧")
    
    # 相机坐标系：X右，Y下，Z前
    # 如果相机朝向+X（和base一致），相机的Z轴应该指向+X
    print(f"\n相机坐标系（标准OpenCV）：X右，Y下，Z前")
    print(f"  - 如果相机朝向+X，相机Z轴 = odom +X轴")
    
    print("\n" + "=" * 80)
    print("场景2: 机器人在原点，朝向+Y方向（向左转90度）")
    print("=" * 80)
    
    # 朝向+Y方向（逆时针旋转90度）
    base_to_odom_y_forward = np.eye(4)
    base_to_odom_y_forward[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
    yaw = extract_yaw(base_to_odom_y_forward)
    print(f"base_to_odom (朝向+Y):\n{base_to_odom_y_forward[:3, :3]}")
    print(f"yaw = {np.rad2deg(yaw):.1f}度")
    print(f"机器人位置: (0, 0)")
    print(f"机器人朝向: +Y方向（北）")
    
    print(f"\n在BEV地图上：")
    print(f"  - 机器人应该在地图中心")
    print(f"  - 机器人应该朝向上方（+Y方向）")
    print(f"  - FOV扇形应该指向上方")
    
    print("\n" + "=" * 80)
    print("分析reveal_fog_of_war的角度约定")
    print("=" * 80)
    
    # reveal_fog_of_war使用的角度约定
    # current_angle参数：从+X轴逆时针测量的角度
    # 如果机器人朝向+X（yaw=0），current_angle应该是0
    # 如果机器人朝向+Y（yaw=90度），current_angle应该是90度
    
    print("reveal_fog_of_war的current_angle参数：")
    print("  - 0度 = 朝向图像右侧（+X方向）")
    print("  - 90度 = 朝向图像上方（+Y方向）")
    print("  - 180度 = 朝向图像左侧（-X方向）")
    print("  - 270度 = 朝向图像下方（-Y方向）")
    
    print("\n当前代码使用: current_angle = -agent_yaw")
    print("  - 如果agent_yaw=0（朝向+X），current_angle=0 ✓ 正确")
    print("  - 如果agent_yaw=90（朝向+Y），current_angle=-90 ✗ 错误！应该是90")
    
    print("\n" + "=" * 80)
    print("可能的问题：")
    print("=" * 80)
    print("1. reveal_fog_of_war的角度约定可能不是标准的数学约定")
    print("2. BEV地图的Y轴是翻转的（图像坐标系），可能需要调整角度")
    print("3. 相机外参可能有90度的偏差")
    
    print("\n" + "=" * 80)
    print("测试点云变换")
    print("=" * 80)
    
    # 假设相机在机器人前方，朝向+X
    # 相机看到的点在相机坐标系中是(0, 0, 1)（前方1米）
    # 应该变换到odom坐标系的(1, 0, 0)（+X方向1米）
    
    # 相机外参：odom->cam
    # 如果相机和base重合，T_cam_odom应该是单位矩阵
    T_cam_odom = np.eye(4)
    T_odom_cam = np.linalg.inv(T_cam_odom)
    
    point_camera = np.array([[0, 0, 1]])  # 相机前方1米
    point_odom = (T_odom_cam @ np.hstack([point_camera, [[1]]]).T).T[:, :3]
    
    print(f"相机坐标系中的点: {point_camera[0]}")
    print(f"odom坐标系中的点: {point_odom[0]}")
    print(f"期望: (1, 0, z) 或 (0, 1, z)，取决于相机朝向")
    
    # 如果结果是(0, 0, 1)，说明相机Z轴对应odom Z轴，这是错误的
    # 如果结果是(1, 0, 0)，说明相机Z轴对应odom X轴，这是正确的（相机朝向+X）
    # 如果结果是(0, 1, 0)，说明相机Z轴对应odom Y轴，这说明相机朝向+Y（90度偏差）
    
    if np.allclose(point_odom[0, :2], [1, 0], atol=0.1):
        print("✓ 相机朝向+X方向（正确）")
    elif np.allclose(point_odom[0, :2], [0, 1], atol=0.1):
        print("✗ 相机朝向+Y方向（90度偏差！）")
    elif np.allclose(point_odom[0, :2], [-1, 0], atol=0.1):
        print("✗ 相机朝向-X方向（180度偏差！）")
    elif np.allclose(point_odom[0, :2], [0, -1], atol=0.1):
        print("✗ 相机朝向-Y方向（270度偏差！）")
    else:
        print(f"? 未知朝向: {point_odom[0, :2]}")

if __name__ == "__main__":
    visualize_coordinate_systems()
    
    print("\n" + "=" * 80)
    print("建议的修复方案：")
    print("=" * 80)
    print("1. 检查ROS TF树，确认camera_frame相对于base_frame的朝向")
    print("2. 如果相机有90度偏差，需要在T_cam_odom中添加旋转修正")
    print("3. 或者修改reveal_fog_of_war的角度参数：current_angle = -agent_yaw + 90")
    print("4. 添加调试日志，打印实际的变换矩阵和点云坐标")
