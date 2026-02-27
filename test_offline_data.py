#!/usr/bin/env python3
"""
离线测试脚本：使用保存的数据测试obstacle_map
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not installed, falling back to matplotlib")

from obstacle_map import ObstacleMap
from geometry_utils import get_point_cloud, transform_points
from utils import point_to_pixel


def load_snapshot(snapshot_dir: str) -> dict:
    """
    加载保存的观测数据
    
    Parameters
    ----------
    snapshot_dir : str
        保存数据的目录路径
        
    Returns
    -------
    dict
        包含rgb, depth, intrinsic, T_cam_odom, T_odom_base的字典
    """
    # 加载RGB图像
    rgb_path = os.path.join(snapshot_dir, 'rgb.png')
    rgb = np.array(Image.open(rgb_path))
    print(f"Loaded RGB: {rgb.shape}, dtype={rgb.dtype}")
    
    # 加载Depth图像（16位PNG，单位毫米）
    depth_path = os.path.join(snapshot_dir, 'depth.png')
    depth_mm = np.array(Image.open(depth_path))
    depth = depth_mm.astype(np.float32) / 1000.0  # 转换为米
    print(f"Loaded Depth: {depth.shape}, dtype={depth.dtype}, range=[{depth.min():.3f}, {depth.max():.3f}]")
    
    # 加载参数
    params_path = os.path.join(snapshot_dir, 'params.npz')
    params = np.load(params_path)
    
    intrinsic = params['intrinsic']
    T_cam_odom = params['T_cam_odom']
    T_odom_base = params['T_odom_base']
    
    print(f"\nCamera Intrinsic:\n{intrinsic}")
    print(f"\nT_cam_odom (odom->cam):\n{T_cam_odom}")
    print(f"\nT_odom_base (base->odom):\n{T_odom_base}")
    
    # 提取机器人位姿
    base_pos = T_odom_base[:3, 3]
    base_yaw = np.arctan2(T_odom_base[1, 0], T_odom_base[0, 0])
    print(f"\nRobot position: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
    print(f"Robot yaw: {base_yaw:.3f} rad ({np.rad2deg(base_yaw):.1f} deg)")
    
    return {
        'rgb': rgb,
        'depth': depth,
        'intrinsic': intrinsic,
        'extrinsic': T_cam_odom,
        'base_to_odom_matrix': T_odom_base
    }


def visualize_point_cloud_open3d(obs: dict):
    """
    使用Open3D可视化点云（同时显示相机坐标系和odom坐标系）
    
    Parameters
    ----------
    obs : dict
        观测数据
    """
    if not HAS_OPEN3D:
        print("Open3D not available, skipping visualization")
        return
    
    print("\n" + "=" * 80)
    print("Visualizing Point Cloud with Open3D")
    print("=" * 80)
    
    # 1. 生成点云
    depth = obs['depth']
    intrinsic = obs['intrinsic']
    extrinsic = obs['extrinsic']
    base_to_odom = obs['base_to_odom_matrix']
    
    robot_pos = base_to_odom[:3, 3]
    robot_rot = base_to_odom[:3, :3]
    robot_yaw = np.arctan2(robot_rot[1, 0], robot_rot[0, 0])
    
    # 生成相机坐标系点云
    mask = (depth > 0.1) & (depth < 10.0) & np.isfinite(depth)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    point_cloud_camera = get_point_cloud(depth, mask, fx, fy)
    
    # 转换到odom坐标系
    T_cam_odom = extrinsic
    T_odom_cam = np.linalg.inv(T_cam_odom)
    point_cloud_odom = transform_points(point_cloud_camera, T_odom_cam)
    
    print(f"Generated {len(point_cloud_odom)} points")
    print(f"Camera frame - Z range: [{point_cloud_camera[:, 2].min():.3f}, {point_cloud_camera[:, 2].max():.3f}]")
    print(f"Odom frame - X range: [{point_cloud_odom[:, 0].min():.3f}, {point_cloud_odom[:, 0].max():.3f}]")
    print(f"Odom frame - Y range: [{point_cloud_odom[:, 1].min():.3f}, {point_cloud_odom[:, 1].max():.3f}]")
    print(f"Odom frame - Z range: [{point_cloud_odom[:, 2].min():.3f}, {point_cloud_odom[:, 2].max():.3f}]")
    
    # 2. 创建相机坐标系点云（蓝色）
    pcd_camera = o3d.geometry.PointCloud()
    pcd_camera.points = o3d.utility.Vector3dVector(point_cloud_camera)
    pcd_camera.paint_uniform_color([0.3, 0.3, 1.0])  # 蓝色
    
    # 将相机点云移到相机位置（在odom坐标系中）
    camera_pos = T_odom_cam[:3, 3]
    camera_rot = T_odom_cam[:3, :3]
    pcd_camera.rotate(camera_rot, center=[0, 0, 0])
    pcd_camera.translate(camera_pos)
    
    # 3. 创建odom坐标系点云（根据高度着色）
    pcd_odom = o3d.geometry.PointCloud()
    pcd_odom.points = o3d.utility.Vector3dVector(point_cloud_odom)
    
    # 根据Z值着色
    z_values = point_cloud_odom[:, 2]
    z_normalized = (z_values - z_values.min()) / (z_values.max() - z_values.min())
    colors = plt.cm.viridis(z_normalized)[:, :3]
    pcd_odom.colors = o3d.utility.Vector3dVector(colors)
    
    # 4. 创建odom坐标系（原点）
    axis_length = 2.0
    coord_frame_odom = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_length, origin=[0, 0, 0]
    )
    
    # 5. 创建相机坐标系（在相机位置）
    coord_frame_camera = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_length * 0.5, origin=[0, 0, 0]
    )
    coord_frame_camera.rotate(camera_rot, center=[0, 0, 0])
    coord_frame_camera.translate(camera_pos)
    
    # 6. 创建机器人位置标记（球体）
    robot_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    robot_sphere.translate(robot_pos)
    robot_sphere.paint_uniform_color([1.0, 0.0, 1.0])  # 紫色
    
    # 7. 创建机器人朝向箭头
    arrow_length = 1.0
    arrow_end = robot_pos + np.array([
        arrow_length * np.cos(robot_yaw),
        arrow_length * np.sin(robot_yaw),
        0
    ])
    
    # 创建箭头（用圆柱体和圆锥体）
    arrow_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.02, height=arrow_length * 0.8)
    arrow_cone = o3d.geometry.TriangleMesh.create_cone(radius=0.05, height=arrow_length * 0.2)
    
    # 旋转和平移箭头
    arrow_direction = arrow_end - robot_pos
    arrow_direction = arrow_direction / np.linalg.norm(arrow_direction)
    
    # 计算旋转矩阵（从Z轴旋转到箭头方向）
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, arrow_direction)
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, arrow_direction), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    else:
        R = np.eye(3)
    
    arrow_cylinder.rotate(R, center=[0, 0, 0])
    arrow_cylinder.translate(robot_pos + arrow_direction * arrow_length * 0.4)
    arrow_cylinder.paint_uniform_color([1.0, 0.0, 1.0])
    
    arrow_cone.rotate(R, center=[0, 0, 0])
    arrow_cone.translate(robot_pos + arrow_direction * arrow_length * 0.9)
    arrow_cone.paint_uniform_color([1.0, 0.0, 1.0])
    
    # 8. 创建高度平面
    plane_size = max(
        point_cloud_odom[:, 0].max() - point_cloud_odom[:, 0].min(),
        point_cloud_odom[:, 1].max() - point_cloud_odom[:, 1].min()
    ) * 1.2
    
    def create_plane(z_height, color):
        """创建一个水平平面"""
        plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=0.01)
        plane.translate([-plane_size/2, -plane_size/2, z_height - 0.005])
        plane.paint_uniform_color(color)
        plane.compute_vertex_normals()
        return plane
    
    plane_minus_01 = create_plane(-0.1, [1.0, 0.0, 0.0])  # 红色
    plane_0 = create_plane(0.0, [0.0, 1.0, 0.0])  # 绿色
    plane_plus_01 = create_plane(0.1, [0.0, 0.0, 1.0])  # 蓝色
    
    # 9. 可视化
    print("\nLaunching Open3D visualizer...")
    print("Legend:")
    print("  - Blue point cloud: Camera frame (transformed to odom for display)")
    print("  - Colored point cloud: Odom frame (colored by height)")
    print("  - Large RGB axes: Odom coordinate frame at origin")
    print("  - Small RGB axes: Camera coordinate frame")
    print("  - Purple sphere + arrow: Robot position and orientation")
    print("  - Red/Green/Blue planes: z=-0.1/0/+0.1 meters")
    print("\nControls:")
    print("  - Mouse left: Rotate")
    print("  - Mouse right: Pan")
    print("  - Mouse wheel: Zoom")
    print("  - Press 'Q' or close window to exit")
    
    geometries = [
        pcd_camera,  # 相机坐标系点云（蓝色）
        pcd_odom,    # odom坐标系点云（按高度着色）
        coord_frame_odom,    # odom坐标系
        coord_frame_camera,  # 相机坐标系
        robot_sphere,
        arrow_cylinder,
        arrow_cone,
        plane_minus_01,
        plane_0,
        plane_plus_01
    ]
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud: Camera Frame (Blue) vs Odom Frame (Colored)",
        width=1600,
        height=1200,
        left=50,
        top=50
    )


def visualize_point_cloud_with_planes(obs: dict, save_dir: str = None):
    """
    可视化点云并显示z=-0.1, z=0, z=0.1三个平面
    
    Parameters
    ----------
    obs : dict
        观测数据
    save_dir : str, optional
        保存可视化结果的目录
    """
    print("\n" + "=" * 80)
    print("Visualizing Point Cloud with Height Planes")
    print("=" * 80)
    
    # 1. 生成点云（相机坐标系）
    depth = obs['depth']
    intrinsic = obs['intrinsic']
    
    mask = (depth > 0.5) & (depth < 5.0) & np.isfinite(depth)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    point_cloud_camera = get_point_cloud(depth, mask, fx, fy)
    
    print(f"Generated {len(point_cloud_camera)} points in camera frame")
    
    # 2. 变换到odom坐标系
    T_cam_odom = obs['extrinsic']
    T_odom_cam = np.linalg.inv(T_cam_odom)
    point_cloud_odom = transform_points(T_odom_cam, point_cloud_camera)
    
    print(f"Transformed to odom frame")
    print(f"  X range: [{point_cloud_odom[:, 0].min():.3f}, {point_cloud_odom[:, 0].max():.3f}]")
    print(f"  Y range: [{point_cloud_odom[:, 1].min():.3f}, {point_cloud_odom[:, 1].max():.3f}]")
    print(f"  Z range: [{point_cloud_odom[:, 2].min():.3f}, {point_cloud_odom[:, 2].max():.3f}]")
    
    # 3. 提取机器人位置
    robot_pos = obs['base_to_odom_matrix'][:3, 3]
    robot_yaw = np.arctan2(obs['base_to_odom_matrix'][1, 0], obs['base_to_odom_matrix'][0, 0])
    
    # 4. 创建3D可视化 - 单独窗口显示完整点云
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 对点云进行下采样以提高显示速度
    downsample_factor = max(1, len(point_cloud_odom) // 20000)
    pc_downsampled = point_cloud_odom[::downsample_factor]
    
    # 根据Z值着色
    z_colors = pc_downsampled[:, 2]
    scatter = ax.scatter(
        pc_downsampled[:, 0],
        pc_downsampled[:, 1],
        pc_downsampled[:, 2],
        c=z_colors,
        cmap='viridis',
        s=1,
        alpha=0.6
    )
    
    # 绘制odom坐标系原点和坐标轴（长一点）
    axis_length = 2.0  # 2米长的坐标轴
    origin = np.array([0, 0, 0])
    
    # X轴（红色）
    ax.quiver(origin[0], origin[1], origin[2],
              axis_length, 0, 0,
              color='red', arrow_length_ratio=0.1, linewidth=3, label='X (forward)')
    
    # Y轴（绿色）
    ax.quiver(origin[0], origin[1], origin[2],
              0, axis_length, 0,
              color='green', arrow_length_ratio=0.1, linewidth=3, label='Y (left)')
    
    # Z轴（蓝色）
    ax.quiver(origin[0], origin[1], origin[2],
              0, 0, axis_length,
              color='blue', arrow_length_ratio=0.1, linewidth=3, label='Z (up)')
    
    # 绘制三个高度平面
    x_range = [point_cloud_odom[:, 0].min(), point_cloud_odom[:, 0].max()]
    y_range = [point_cloud_odom[:, 1].min(), point_cloud_odom[:, 1].max()]
    xx, yy = np.meshgrid(
        np.linspace(x_range[0], x_range[1], 10),
        np.linspace(y_range[0], y_range[1], 10)
    )
    
    # z = -0.1 平面（红色半透明）
    zz1 = np.ones_like(xx) * -0.1
    ax.plot_surface(xx, yy, zz1, alpha=0.2, color='red')
    
    # z = 0 平面（绿色半透明）
    zz2 = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz2, alpha=0.2, color='green')
    
    # z = 0.1 平面（蓝色半透明）
    zz3 = np.ones_like(xx) * 0.1
    ax.plot_surface(xx, yy, zz3, alpha=0.2, color='blue')
    
    # 绘制机器人位置和朝向
    ax.scatter([robot_pos[0]], [robot_pos[1]], [robot_pos[2]],
                c='magenta', s=200, marker='o', label='Robot', edgecolors='black', linewidths=2)
    
    # 绘制机器人朝向箭头
    arrow_length = 1.0
    ax.quiver(robot_pos[0], robot_pos[1], robot_pos[2],
               arrow_length * np.cos(robot_yaw),
               arrow_length * np.sin(robot_yaw),
               0,
               color='magenta', arrow_length_ratio=0.2, linewidth=3)
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('X (m) - Forward', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m) - Left', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m) - Up', fontsize=12, fontweight='bold')
    ax.set_title('Point Cloud in Odom Frame with Coordinate Axes', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, label='Z (m)', shrink=0.6, pad=0.1)
    cbar.set_label('Height (m)', fontsize=11)
    
    # 添加图例
    ax.legend(loc='upper left', fontsize=10)
    
    # 设置相等的坐标轴比例
    max_range = np.array([
        point_cloud_odom[:, 0].max() - point_cloud_odom[:, 0].min(),
        point_cloud_odom[:, 1].max() - point_cloud_odom[:, 1].min(),
        point_cloud_odom[:, 2].max() - point_cloud_odom[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (point_cloud_odom[:, 0].max() + point_cloud_odom[:, 0].min()) * 0.5
    mid_y = (point_cloud_odom[:, 1].max() + point_cloud_odom[:, 1].min()) * 0.5
    mid_z = (point_cloud_odom[:, 2].max() + point_cloud_odom[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 保存图像
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, 'point_cloud_visualization.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {fig_path}")
    
    # 显示图像
    try:
        plt.show()
    except:
        print("\nCannot display plot (no display available)")
    
    # 5. 统计各平面附近的点数
    print("\n" + "=" * 80)
    print("Point Statistics Near Height Planes")
    print("=" * 80)
    
    tolerance = 0.05  # ±5cm
    for z_plane in [-0.1, 0.0, 0.1]:
        mask = np.abs(point_cloud_odom[:, 2] - z_plane) < tolerance
        count = mask.sum()
        percentage = 100.0 * count / len(point_cloud_odom)
        print(f"z = {z_plane:+.1f} ± {tolerance:.2f}m: {count:6d} points ({percentage:5.2f}%)")
    
    # 6. 分析地面点
    print("\n" + "=" * 80)
    print("Ground Plane Analysis")
    print("=" * 80)
    
    # 假设地面在z=0附近
    ground_mask = np.abs(point_cloud_odom[:, 2]) < 0.1
    ground_points = point_cloud_odom[ground_mask]
    
    if len(ground_points) > 0:
        print(f"Ground points (|z| < 0.1m): {len(ground_points)}")
        print(f"  X range: [{ground_points[:, 0].min():.3f}, {ground_points[:, 0].max():.3f}]")
        print(f"  Y range: [{ground_points[:, 1].min():.3f}, {ground_points[:, 1].max():.3f}]")
        print(f"  Z mean: {ground_points[:, 2].mean():.3f} ± {ground_points[:, 2].std():.3f}")
    else:
        print("No ground points found")


def verify_transform(obs: dict):
    """
    验证坐标变换是否正确
    
    Parameters
    ----------
    obs : dict
        观测数据
    """
    print("\n" + "=" * 80)
    print("Verifying Coordinate Transform")
    print("=" * 80)
    
    T_cam_odom = obs['extrinsic']
    T_odom_base = obs['base_to_odom_matrix']
    
    # 1. 检查变换矩阵的正交性
    R_cam_odom = T_cam_odom[:3, :3]
    det = np.linalg.det(R_cam_odom)
    orthogonality = np.linalg.norm(R_cam_odom @ R_cam_odom.T - np.eye(3))
    
    print(f"\nT_cam_odom rotation matrix:")
    print(f"  Determinant: {det:.6f} (should be ±1)")
    print(f"  Orthogonality error: {orthogonality:.6e} (should be ~0)")
    
    # 2. 分析相机朝向
    camera_x = R_cam_odom[:, 0]  # 相机X轴（右）在odom中的方向
    camera_y = R_cam_odom[:, 1]  # 相机Y轴（下）在odom中的方向
    camera_z = R_cam_odom[:, 2]  # 相机Z轴（前）在odom中的方向
    
    print(f"\nCamera axes in odom frame:")
    print(f"  Camera X (right): {camera_x}")
    print(f"  Camera Y (down):  {camera_y}")
    print(f"  Camera Z (front): {camera_z}")
    
    # 3. 计算相机的俯仰角和偏航角
    # 相机Z轴在XY平面的投影
    camera_z_xy = np.array([camera_z[0], camera_z[1], 0])
    camera_z_xy_norm = np.linalg.norm(camera_z_xy)
    
    if camera_z_xy_norm > 0.01:
        pitch = np.arctan2(-camera_z[2], camera_z_xy_norm)
        yaw = np.arctan2(camera_z[1], camera_z[0])
        print(f"\nCamera orientation:")
        print(f"  Pitch: {np.rad2deg(pitch):.1f}° (negative = looking down)")
        print(f"  Yaw: {np.rad2deg(yaw):.1f}° (0°=+X, 90°=+Y, -90°=-Y)")
    
    # 4. 检查地面法向量
    # 如果变换正确，相机Y轴（下）应该大致指向odom的-Z方向
    print(f"\nExpected: Camera Y (down) should point to odom -Z")
    print(f"Actual: Camera Y = {camera_y}")
    print(f"Angle with -Z: {np.rad2deg(np.arccos(-camera_y[2])):.1f}°")
    
    # 5. 测试一个已知点的变换
    # 相机坐标系中的一个点：(0, 0, 1) = 前方1米
    point_cam = np.array([0, 0, 1, 1])
    T_odom_cam = np.linalg.inv(T_cam_odom)
    point_odom = T_odom_cam @ point_cam
    
    print(f"\nTest point transform:")
    print(f"  Camera frame: (0, 0, 1) = 1m in front of camera")
    print(f"  Odom frame: ({point_odom[0]:.3f}, {point_odom[1]:.3f}, {point_odom[2]:.3f})")
    
    # 6. 检查是否需要交换轴或改变符号
    print(f"\n" + "=" * 80)
    print("Diagnosis:")
    print("=" * 80)
    
    # 检查相机Y轴是否指向下方
    if camera_y[2] < -0.5:
        print("✓ Camera Y axis points downward (correct)")
    elif camera_y[2] > 0.5:
        print("✗ Camera Y axis points upward (WRONG! Should point down)")
        print("  Possible fix: Flip Y axis or check camera frame definition")
    else:
        print("? Camera Y axis is not clearly up or down")
    
    # 检查相机Z轴是否大致水平
    if abs(camera_z[2]) < 0.5:
        print("✓ Camera Z axis is roughly horizontal (correct)")
    else:
        print(f"? Camera Z axis has significant vertical component: {camera_z[2]:.3f}")
        print(f"  This means camera is tilted {np.rad2deg(np.arcsin(-camera_z[2])):.1f}° from horizontal")


def test_obstacle_map(obs: dict, save_dir: str = None):
    """
    测试ObstacleMap
    
    Parameters
    ----------
    obs : dict
        观测数据
    save_dir : str, optional
        保存可视化结果的目录
    """
    print("\n" + "=" * 80)
    print("Testing ObstacleMap")
    print("=" * 80)
    
    # 创建ObstacleMap
    obstacle_map = ObstacleMap(
        min_height=0.40,
        max_height=1.5,
        agent_radius=0.3,
        area_thresh=0.5,
        size=400,
        pixels_per_meter=10,
    )
    
    # 更新地图
    obstacle_map.update_map(
        depth=obs['depth'],
        intrinsic=obs['intrinsic'],
        extrinsic=obs['extrinsic'],
        base_to_odom=obs['base_to_odom_matrix'],
        min_depth=0.4,
        max_depth=4.0,
    )
    
    # 生成可视化
    robot_pos = obs['base_to_odom_matrix'][:2, 3]
    bev_vis = obstacle_map.visualize(robot_pos=robot_pos)
    
    # 显示结果
    print(f"\nFrontiers detected: {len(obstacle_map.frontiers)}")
    
    # 创建带frontier标记的RGB图像
    rgb_with_frontiers = obs['rgb'].copy()
    
    if len(obstacle_map.frontiers) > 0:
        print("Frontier coordinates (odom frame):")
        for i, frontier in enumerate(obstacle_map.frontiers):
            print(f"  {i+1}: ({frontier[0]:.3f}, {frontier[1]:.3f})")
        
        # 投影frontier到图像并打印像素坐标
        print("\nProjecting frontiers to image:")
        H, W = obs['depth'].shape
        T_cam_odom = obs['extrinsic']
        K = obs['intrinsic']
        
        for i, frontier_xy in enumerate(obstacle_map.frontiers):
            # frontier在odom坐标系中的3D位置（假设高度为-0.1m，即地面附近）
            frontier_3d_odom = np.array([frontier_xy[0], frontier_xy[1], -0.10])
            
            # 投影到图像
            uv = point_to_pixel(frontier_3d_odom, K, T_cam_odom)
            
            if uv is None or len(uv) == 0:
                print(f"  Frontier {i+1} at ({frontier_xy[0]:.3f}, {frontier_xy[1]:.3f}): projection failed (behind camera)")
                continue
            
            pixel_u, pixel_v = int(round(uv[0][0])), int(round(uv[0][1]))
            
            # 检查是否在图像范围内
            if 0 <= pixel_u < W and 0 <= pixel_v < H:
                print(f"  Frontier {i+1} at ({frontier_xy[0]:.3f}, {frontier_xy[1]:.3f}): pixel ({pixel_u}, {pixel_v}) - IN IMAGE")
                
                # 在RGB图像上绘制frontier标记
                # 绘制十字标记
                cross_size = 20
                color = (0, 255, 0)  # 绿色
                thickness = 2
                cv2.line(rgb_with_frontiers,
                        (pixel_u - cross_size, pixel_v),
                        (pixel_u + cross_size, pixel_v),
                        color, thickness)
                cv2.line(rgb_with_frontiers,
                        (pixel_u, pixel_v - cross_size),
                        (pixel_u, pixel_v + cross_size),
                        color, thickness)
                
                # 绘制圆圈
                cv2.circle(rgb_with_frontiers, (pixel_u, pixel_v), 15, color, thickness)
                
                # 添加编号文字
                cv2.putText(rgb_with_frontiers,
                           f"F{i+1}",
                           (pixel_u + 20, pixel_v - 20),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.8,
                           color,
                           2)
            else:
                print(f"  Frontier {i+1} at ({frontier_xy[0]:.3f}, {frontier_xy[1]:.3f}): pixel ({pixel_u}, {pixel_v}) - OUT OF BOUNDS")
    
    # 保存可视化
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存BEV地图
        bev_path = os.path.join(save_dir, 'bev_map.png')
        cv2.imwrite(bev_path, cv2.cvtColor(bev_vis, cv2.COLOR_RGB2BGR))
        print(f"\nSaved BEV map to {bev_path}")
        
        # 保存原始RGB图像
        rgb_path = os.path.join(save_dir, 'rgb.png')
        cv2.imwrite(rgb_path, cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR))
        print(f"Saved RGB to {rgb_path}")
        
        # 保存带frontier标记的RGB图像
        rgb_frontiers_path = os.path.join(save_dir, 'rgb_with_frontiers.png')
        cv2.imwrite(rgb_frontiers_path, cv2.cvtColor(rgb_with_frontiers, cv2.COLOR_RGB2BGR))
        print(f"Saved RGB with frontiers to {rgb_frontiers_path}")
    
    # 显示可视化（可选）
    try:
        cv2.imshow('BEV Map', cv2.cvtColor(bev_vis, cv2.COLOR_RGB2BGR))
        cv2.imshow('RGB (Original)', cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR))
        cv2.imshow('RGB with Frontiers', cv2.cvtColor(rgb_with_frontiers, cv2.COLOR_RGB2BGR))
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("\nCannot display windows (no display available)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_offline_data.py <snapshot_dir> [output_dir] [--visualize-pointcloud]")
        print("\nExample:")
        print("  python test_offline_data.py test_data/snapshot_20260226_170000")
        print("  python test_offline_data.py test_data/snapshot_20260226_170000 test_results")
        print("  python test_offline_data.py test_data/snapshot_20260226_170000 test_results --visualize-pointcloud")
        sys.exit(1)
    
    snapshot_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    visualize_pc = '--visualize-pointcloud' in sys.argv
    
    if not os.path.exists(snapshot_dir):
        print(f"Error: Snapshot directory not found: {snapshot_dir}")
        sys.exit(1)
    
    print(f"Loading snapshot from: {snapshot_dir}")
    obs = load_snapshot(snapshot_dir)
    
    # 验证坐标变换
    verify_transform(obs)
    
    # 可视化点云（如果指定）
    if visualize_pc:
        if HAS_OPEN3D:
            print(f"\nVisualizing point cloud with Open3D...")
            visualize_point_cloud_open3d(obs)
        else:
            print(f"\nVisualizing point cloud with matplotlib...")
            visualize_point_cloud_with_planes(obs, save_dir=output_dir)
    
    # 测试ObstacleMap
    print(f"\nTesting ObstacleMap...")
    test_obstacle_map(obs, save_dir=output_dir)
    
    print("\nTest completed!")


if __name__ == '__main__':
    main()
