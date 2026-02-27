#!/usr/bin/env python3
"""
简化的离线测试脚本：使用Open3D可视化点云
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Error: Open3D not installed. Please install it: pip install open3d")
    sys.exit(1)

from obstacle_map import ObstacleMap
from geometry_utils import get_point_cloud, transform_points


def load_snapshot(snapshot_dir: str) -> dict:
    """加载保存的观测数据"""
    rgb_path = os.path.join(snapshot_dir, 'rgb.png')
    rgb = np.array(Image.open(rgb_path))
    
    depth_path = os.path.join(snapshot_dir, 'depth.png')
    depth_mm = np.array(Image.open(depth_path))
    depth = depth_mm.astype(np.float32) / 1000.0
    
    params_path = os.path.join(snapshot_dir, 'params.npz')
    params = np.load(params_path)
    
    return {
        'rgb': rgb,
        'depth': depth,
        'intrinsic': params['intrinsic'],
        'extrinsic': params['T_cam_odom'],
        'base_to_odom_matrix': params['T_odom_base']
    }


def visualize_point_cloud_open3d(obs: dict):
    """使用Open3D可视化点云（先相机坐标系，再odom坐标系）"""
    
    # 生成点云
    depth = obs['depth']
    intrinsic = obs['intrinsic']
    extrinsic = obs['extrinsic']
    base_to_odom = obs['base_to_odom_matrix']
    
    robot_pos = base_to_odom[:3, 3]
    robot_yaw = np.arctan2(base_to_odom[1, 0], base_to_odom[0, 0])
    
    mask = (depth > 0.1) & (depth < 10.0) & np.isfinite(depth)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    point_cloud_camera = get_point_cloud(depth, mask, fx, fy, cx, cy)
    
    T_odom_cam = np.linalg.inv(extrinsic)
    point_cloud_odom = transform_points(T_odom_cam, point_cloud_camera)
    
    print(f"\nGenerated {len(point_cloud_camera)} points")
    print(f"Camera frame - Z range: [{point_cloud_camera[:, 2].min():.3f}, {point_cloud_camera[:, 2].max():.3f}]")
    print(f"Odom frame - Z range: [{point_cloud_odom[:, 2].min():.3f}, {point_cloud_odom[:, 2].max():.3f}]")
    
    # ========== 窗口1：相机坐标系 ==========
    print("\n[1/2] Showing Camera Frame...")
    
    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(point_cloud_camera)
    
    z_cam = point_cloud_camera[:, 2]
    z_cam_norm = (z_cam - z_cam.min()) / (z_cam.max() - z_cam.min())
    colors_cam = plt.cm.plasma(z_cam_norm)[:, :3]
    pcd_cam.colors = o3d.utility.Vector3dVector(colors_cam)
    
    coord_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries(
        [pcd_cam, coord_cam],
        window_name="[1/2] Camera Frame (OpenCV: X=right, Y=down, Z=forward)",
        width=1600,
        height=1200
    )
    
    # ========== 窗口2：odom坐标系 ==========
    print("\n[2/2] Showing Odom Frame...")
    
    pcd_odom = o3d.geometry.PointCloud()
    pcd_odom.points = o3d.utility.Vector3dVector(point_cloud_odom)
    
    z_odom = point_cloud_odom[:, 2]
    z_odom_norm = (z_odom - z_odom.min()) / (z_odom.max() - z_odom.min())
    colors_odom = plt.cm.viridis(z_odom_norm)[:, :3]
    pcd_odom.colors = o3d.utility.Vector3dVector(colors_odom)
    
    coord_odom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    
    robot_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    robot_sphere.translate(robot_pos)
    robot_sphere.paint_uniform_color([1.0, 0.0, 1.0])
    
    # 高度平面
    plane_size = max(
        point_cloud_odom[:, 0].max() - point_cloud_odom[:, 0].min(),
        point_cloud_odom[:, 1].max() - point_cloud_odom[:, 1].min()
    ) * 1.2
    
    def create_plane(z, color):
        plane = o3d.geometry.TriangleMesh.create_box(plane_size, plane_size, 0.01)
        plane.translate([-plane_size/2, -plane_size/2, z - 0.005])
        plane.paint_uniform_color(color)
        return plane
    
    plane_m01 = create_plane(-0.1, [1, 0, 0])
    plane_0 = create_plane(0.0, [0, 1, 0])
    plane_p01 = create_plane(0.1, [0, 0, 1])
    
    o3d.visualization.draw_geometries(
        [pcd_odom, coord_odom, robot_sphere, plane_m01, plane_0, plane_p01],
        window_name="[2/2] Odom Frame (ROS: X=forward, Y=left, Z=up)",
        width=1600,
        height=1200
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_offline_data_simple.py <snapshot_dir>")
        sys.exit(1)
    
    snapshot_dir = sys.argv[1]
    if not os.path.exists(snapshot_dir):
        print(f"Error: {snapshot_dir} not found")
        sys.exit(1)
    
    print(f"Loading snapshot from: {snapshot_dir}")
    obs = load_snapshot(snapshot_dir)
    
    visualize_point_cloud_open3d(obs)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
