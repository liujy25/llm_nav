#!/usr/bin/env python3
"""
离线测试脚本：使用保存的数据测试obstacle_map
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image

from obstacle_map import ObstacleMap


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
        min_height=0.2,
        max_height=1.5,
        agent_radius=0.3,
        area_thresh=1.0,
        size=200,
        pixels_per_meter=10,
    )
    
    # 更新地图
    obstacle_map.update_map(
        depth=obs['depth'],
        intrinsic=obs['intrinsic'],
        extrinsic=obs['extrinsic'],
        base_to_odom=obs['base_to_odom_matrix'],
        min_depth=0.5,
        max_depth=5.0,
    )
    
    # 生成可视化
    robot_pos = obs['base_to_odom_matrix'][:2, 3]
    bev_vis = obstacle_map.visualize(robot_pos=robot_pos)
    
    # 显示结果
    print(f"\nFrontiers detected: {len(obstacle_map.frontiers)}")
    if len(obstacle_map.frontiers) > 0:
        print("Frontier coordinates (odom frame):")
        for i, frontier in enumerate(obstacle_map.frontiers):
            print(f"  {i+1}: ({frontier[0]:.3f}, {frontier[1]:.3f})")
    
    # 保存可视化
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存BEV地图
        bev_path = os.path.join(save_dir, 'bev_map.png')
        cv2.imwrite(bev_path, cv2.cvtColor(bev_vis, cv2.COLOR_RGB2BGR))
        print(f"\nSaved BEV map to {bev_path}")
        
        # 保存RGB图像（用于对比）
        rgb_path = os.path.join(save_dir, 'rgb.png')
        cv2.imwrite(rgb_path, cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR))
        print(f"Saved RGB to {rgb_path}")
    
    # 显示可视化（可选）
    try:
        cv2.imshow('BEV Map', cv2.cvtColor(bev_vis, cv2.COLOR_RGB2BGR))
        cv2.imshow('RGB', cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR))
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("\nCannot display windows (no display available)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_offline_data.py <snapshot_dir> [output_dir]")
        print("\nExample:")
        print("  python test_offline_data.py test_data/snapshot_20260226_170000")
        print("  python test_offline_data.py test_data/snapshot_20260226_170000 test_results")
        sys.exit(1)
    
    snapshot_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(snapshot_dir):
        print(f"Error: Snapshot directory not found: {snapshot_dir}")
        sys.exit(1)
    
    print(f"Loading snapshot from: {snapshot_dir}")
    obs = load_snapshot(snapshot_dir)
    
    print(f"\nTesting with loaded data...")
    test_obstacle_map(obs, save_dir=output_dir)
    
    print("\nTest completed!")


if __name__ == '__main__':
    main()
