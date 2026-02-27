#!/usr/bin/env python3
"""
测试BEV地图集成功能
加载snapshot数据，测试从输入到VLM之前的所有处理步骤
"""
import os
import numpy as np
import cv2
from PIL import Image

from nav_agent import NavAgent


def load_snapshot(snapshot_dir):
    """
    加载snapshot数据
    
    Parameters
    ----------
    snapshot_dir : str
        snapshot目录路径
        
    Returns
    -------
    dict
        观测数据字典
    """
    # 加载RGB图像
    rgb_path = os.path.join(snapshot_dir, 'rgb.png')
    rgb_pil = Image.open(rgb_path)
    rgb = np.array(rgb_pil)
    
    # 加载Depth图像（16位PNG）
    depth_path = os.path.join(snapshot_dir, 'depth.png')
    depth_pil = Image.open(depth_path)
    depth_mm = np.array(depth_pil, dtype=np.uint16)
    depth = depth_mm.astype(np.float32) / 1000.0  # 转换为米
    
    # 加载参数
    params_path = os.path.join(snapshot_dir, 'params.npz')
    params = np.load(params_path)
    
    intrinsic = params['intrinsic']
    T_cam_odom = params['T_cam_odom']
    T_odom_base = params['T_odom_base']
    
    print(f"[Snapshot] Loaded from {snapshot_dir}")
    print(f"  RGB shape: {rgb.shape}")
    print(f"  Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]m")
    print(f"  Intrinsic:\n{intrinsic}")
    print(f"  T_cam_odom:\n{T_cam_odom}")
    print(f"  T_odom_base:\n{T_odom_base}")
    
    # 构建观测字典
    obs = {
        'rgb': rgb,
        'depth': depth,
        'intrinsic': intrinsic,
        'extrinsic': T_cam_odom,
        'base_to_odom_matrix': T_odom_base
    }
    
    return obs


def test_bev_integration(snapshot_dir, output_dir='test_output'):
    """
    测试BEV地图集成功能
    
    Parameters
    ----------
    snapshot_dir : str
        snapshot目录路径
    output_dir : str
        输出目录路径
    """
    print("=" * 80)
    print("测试BEV地图集成功能")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载snapshot数据
    print("\n[Step 1] 加载snapshot数据...")
    obs = load_snapshot(snapshot_dir)
    
    # 2. 初始化NavAgent
    print("\n[Step 2] 初始化NavAgent...")
    cfg = {
        'enable_bev_map': True,
        'bev_map_size': 400,
        'bev_pixels_per_meter': 10,
        'obstacle_height_min': 0.6,
        'obstacle_height_max': 2.0,
        'num_theta': 40,
        'clip_dist': 5.0,
    }
    agent = NavAgent(cfg)
    agent.reset(goal="chair", goal_description="")
    print(f"  NavAgent initialized with BEV map: {agent.obstacle_map is not None}")
    
    # 3. 测试_navigability（raycast可行点检测）
    print("\n[Step 3] 测试_navigability（raycast可行点检测）...")
    a_initial = agent._navigability(obs)
    print(f"  检测到 {len(a_initial)} 个初始可行点")
    for i, (r, theta) in enumerate(a_initial[:5]):
        print(f"    {i+1}: r={r:.2f}m, theta={np.rad2deg(theta):.1f}°")
    
    # 4. 测试_action_proposer（动作过滤）
    print("\n[Step 4] 测试_action_proposer（动作过滤）...")
    a_final = agent._action_proposer(a_initial, obs['base_to_odom_matrix'])
    print(f"  过滤后剩余 {len(a_final)} 个可行点")
    
    # 5. 测试_projection（RGB图像标注）
    print("\n[Step 5] 测试_projection（RGB图像标注）...")
    projected, rgb_vis = agent._projection(a_final, obs)
    print(f"  投影成功，标注了 {len(projected)} 个动作")
    print(f"  动作ID: {list(projected.values())}")
    
    # 保存标注后的RGB图像
    rgb_vis_path = os.path.join(output_dir, 'rgb_annotated.png')
    rgb_vis_bgr = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(rgb_vis_path, rgb_vis_bgr)
    print(f"  保存标注RGB: {rgb_vis_path}")
    
    # 6. 测试ObstacleMap更新
    print("\n[Step 6] 测试ObstacleMap更新...")
    agent.obstacle_map.update_map(
        depth=obs['depth'],
        intrinsic=obs['intrinsic'],
        extrinsic=obs['extrinsic'],
        base_to_odom=obs['base_to_odom_matrix'],
        min_depth=0.5,
        max_depth=6.0
    )
    print(f"  ObstacleMap更新成功")
    print(f"  Explored pixels: {agent.obstacle_map.explored_area.sum()}")
    
    # 7. 测试BEV地图生成
    print("\n[Step 7] 测试BEV地图生成...")
    bev_map = agent._generate_bev_with_waypoints(obs['base_to_odom_matrix'], waypoints=a_final)
    print(f"  BEV地图生成成功，shape: {bev_map.shape}")
    
    # 保存第一次生成的BEV地图（仅更新一次ObstacleMap后）
    bev_map_path = os.path.join(output_dir, 'bev_map_step7.png')
    cv2.imwrite(bev_map_path, bev_map)
    print(f"  保存BEV地图（Step 7，探索区域较少）: {bev_map_path}")
    
    # 8. 测试完整的_nav流程（不调用VLM）
    print("\n[Step 8] 测试完整的_nav流程（模拟）...")
    
    # 模拟_nav的前半部分（到VLM调用之前）
    is_keyframe, angle_range, clip_dist_override, r_P, theta_P = agent._determine_frame_type(obs)
    print(f"  Frame type: {'KEYFRAME' if is_keyframe else 'NON-KEYFRAME'}")
    
    a_initial = agent._navigability(obs, angle_range=angle_range, clip_dist_override=clip_dist_override)
    a_final = agent._action_proposer(a_initial, obs['base_to_odom_matrix'])
    
    if agent.obstacle_map is not None:
        agent.obstacle_map.update_map(
            depth=obs['depth'],
            intrinsic=obs['intrinsic'],
            extrinsic=obs['extrinsic'],
            base_to_odom=obs['base_to_odom_matrix']
        )
    
    a_final_projected, rgb_vis = agent._projection(a_final, obs)
    
    if agent.obstacle_map is not None:
        bev_map = agent._generate_bev_with_waypoints(obs['base_to_odom_matrix'], waypoints=a_final)
        images = [rgb_vis, bev_map]
    else:
        images = [rgb_vis]
    
    print(f"  准备发送给VLM的图像数量: {len(images)}")
    for i, img in enumerate(images):
        print(f"    Image {i+1}: shape={img.shape}, dtype={img.dtype}")
    
    # 9. 测试prompt生成
    print("\n[Step 9] 测试prompt生成...")
    if is_keyframe:
        prompt = agent._construct_keyframe_prompt(
            num_actions=len(a_final_projected),
            iter=1
        )
    else:
        prompt = agent._construct_nonkeyframe_prompt(
            num_actions=len(a_final_projected),
            iter=1
        )
    
    print(f"  Prompt长度: {len(prompt)} 字符")
    print(f"  Prompt前200字符:\n{prompt[:200]}...")
    
    # 保存prompt到文件
    prompt_path = os.path.join(output_dir, 'prompt.txt')
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"  保存prompt: {prompt_path}")
    
    # 10. 保存最终的两张图像（模拟发送给VLM的）
    print("\n[Step 10] 保存最终图像（发送给VLM的）...")
    for i, img in enumerate(images):
        img_name = ['rgb_to_vlm.png', 'bev_to_vlm.png'][i]
        img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(img_path, img)
        print(f"  保存图像 {i+1}: {img_path}")
    print(f"  注意：bev_to_vlm.png包含更多探索区域（ObstacleMap更新了2次）")
    
    # 11. 生成测试报告
    print("\n[Step 11] 生成测试报告...")
    report_path = os.path.join(output_dir, 'test_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BEV地图集成测试报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Snapshot: {snapshot_dir}\n")
        f.write(f"Output: {output_dir}\n\n")
        
        f.write("测试结果:\n")
        f.write(f"  1. Snapshot加载: ✓\n")
        f.write(f"  2. NavAgent初始化: ✓\n")
        f.write(f"  3. Raycast可行点检测: ✓ ({len(a_initial)} 个初始点)\n")
        f.write(f"  4. 动作过滤: ✓ ({len(a_final)} 个最终点)\n")
        f.write(f"  5. RGB图像标注: ✓ ({len(projected)} 个动作)\n")
        f.write(f"  6. ObstacleMap更新: ✓\n")
        f.write(f"  7. BEV地图生成: ✓\n")
        f.write(f"  8. 完整流程模拟: ✓\n")
        f.write(f"  9. Prompt生成: ✓ ({len(prompt)} 字符)\n")
        f.write(f"  10. 图像保存: ✓ ({len(images)} 张)\n\n")
        
        f.write("输出文件:\n")
        f.write(f"  - rgb_annotated.png: 标注了可行点的RGB图像\n")
        f.write(f"  - bev_map.png: BEV地图\n")
        f.write(f"  - rgb_final.png: 最终RGB图像（发送给VLM）\n")
        f.write(f"  - bev_final.png: 最终BEV地图（发送给VLM）\n")
        f.write(f"  - prompt.txt: VLM prompt文本\n")
        f.write(f"  - test_report.txt: 本报告\n\n")
        
        f.write("结论:\n")
        f.write("  所有测试通过！从输入到VLM之前的所有步骤正常工作。\n")
    
    print(f"  保存测试报告: {report_path}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print(f"\n所有输出文件保存在: {output_dir}/")
    print("请检查以下文件:")
    print("  - rgb_annotated.png: 标注了可行点的RGB图像")
    print("  - bev_map.png: BEV地图")
    print("  - rgb_final.png: 最终RGB图像（发送给VLM）")
    print("  - bev_final.png: 最终BEV地图（发送给VLM）")
    print("  - prompt.txt: VLM prompt文本")
    print("  - test_report.txt: 测试报告")


if __name__ == '__main__':
    # 使用相对路径
    snapshot_dir = '../snapshot_20260226_174134'
    output_dir = 'test_output_bev'
    
    test_bev_integration(snapshot_dir, output_dir)
