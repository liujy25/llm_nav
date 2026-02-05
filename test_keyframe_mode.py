#!/usr/bin/env python3
"""
测试关键帧模式的脚本
演示如何使用关键帧模式进行导航
"""
import pickle
import numpy as np
import cv2
from nav_agent import NavAgent

def test_keyframe_mode():
    """测试关键帧模式"""
    
    # 加载测试数据
    print("加载测试数据...")
    data_path = './test_data.pkl'
    try:
        data = pickle.load(open(data_path, 'rb'))
        print(f"✓ 测试数据加载成功")
    except FileNotFoundError:
        print(f"✗ 测试数据未找到: {data_path}")
        print("请提供有效的测试数据文件")
        return
    
    # 配置关键帧模式
    cfg = {
        'keyframe_mode': True,  # 启用关键帧模式
        'keyframe_angle_range': 25.0,  # 非关键帧的角度范围（度）
        'clip_dist': 5.0,  # 关键帧的最大搜索距离
        'turn_angle_deg': 30.0,
        'num_theta': 40,
        'vlm_history_length': 3,
        'vlm_model': 'qwen3-vl-8b-instruct',
        'vlm_api_key': 'sk-83be1f30087144a2a901f8f060ccf543',
        'vlm_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'vlm_timeout': 10
    }
    
    print("\n" + "="*60)
    print("初始化NavAgent（关键帧模式）")
    print("="*60)
    print(f"关键帧模式: {cfg['keyframe_mode']}")
    print(f"非关键帧角度范围: ±{cfg['keyframe_angle_range']}°")
    print(f"关键帧搜索距离: {cfg['clip_dist']}m")
    
    # 初始化agent
    agent = NavAgent(cfg=cfg)
    agent.reset(goal='keyboard', goal_description='')
    
    print("\n" + "="*60)
    print("模拟连续帧测试")
    print("="*60)
    
    # 模拟第1帧（关键帧）
    print("\n--- 第1帧（应该是关键帧）---")
    response1, rgb_vis1, action1, timing1 = agent._nav(data, 'keyboard', iter=1)
    
    if action1 is not None:
        print(f"✓ 动作: {action1}")
        print(f"✓ VLM响应: {response1[:100]}...")
        
        # 显示history状态
        history_summary = agent.get_history_summary()
        print(f"\nHistory状态:")
        print(f"  - 关键帧数量: {history_summary['total_keyframes']}")
        print(f"  - 非关键帧buffer: {history_summary['nonkeyframe_buffer_size']}")
        print(f"  - 当前waypoint P: {history_summary['current_waypoint_P']}")
        
        # 保存可视化
        if rgb_vis1 is not None:
            cv2.imwrite('./keyframe_test_frame1.jpg', cv2.cvtColor(rgb_vis1, cv2.COLOR_RGB2BGR))
            print(f"✓ 可视化已保存: /tmp/keyframe_test_frame1.jpg")
    else:
        print("✗ 第1帧失败")
        return
    
    # 模拟第2帧（应该是非关键帧，如果P可投影）
    print("\n--- 第2帧（测试是否为非关键帧）---")
    
    # 稍微修改base位置（模拟机器人移动了一点）
    data_frame2 = data.copy()
    T_odom_base_2 = data['base_to_odom_matrix'].copy()
    # 向前移动0.2m
    T_odom_base_2[0, 3] += 0.2
    data_frame2['base_to_odom_matrix'] = T_odom_base_2
    
    response2, rgb_vis2, action2, timing2 = agent._nav(data_frame2, 'keyboard', iter=2)
    
    if action2 is not None:
        print(f"✓ 动作: {action2}")
        print(f"✓ VLM响应: {response2[:100]}...")
        
        # 显示history状态
        history_summary = agent.get_history_summary()
        print(f"\nHistory状态:")
        print(f"  - 关键帧数量: {history_summary['total_keyframes']}")
        print(f"  - 非关键帧buffer: {history_summary['nonkeyframe_buffer_size']}")
        print(f"  - 关键帧迭代: {history_summary['keyframe_iters']}")
        print(f"  - 非关键帧迭代: {history_summary['nonkeyframe_iters']}")
        
        # 保存可视化
        if rgb_vis2 is not None:
            cv2.imwrite('./keyframe_test_frame2.jpg', cv2.cvtColor(rgb_vis2, cv2.COLOR_RGB2BGR))
            print(f"✓ 可视化已保存: /tmp/keyframe_test_frame2.jpg")
    else:
        print("✗ 第2帧失败")
    
    # 模拟第3帧（继续测试）
    print("\n--- 第3帧 ---")
    
    data_frame3 = data.copy()
    T_odom_base_3 = data['base_to_odom_matrix'].copy()
    # 再向前移动0.2m
    T_odom_base_3[0, 3] += 0.4
    data_frame3['base_to_odom_matrix'] = T_odom_base_3
    
    response3, rgb_vis3, action3, timing3 = agent._nav(data_frame3, 'keyboard', iter=3)
    
    if action3 is not None:
        print(f"✓ 动作: {action3}")
        
        # 显示最终history状态
        history_summary = agent.get_history_summary()
        print(f"\n最终History状态:")
        print(f"  - 关键帧数量: {history_summary['total_keyframes']}")
        print(f"  - 非关键帧buffer: {history_summary['nonkeyframe_buffer_size']}")
        print(f"  - 关键帧迭代: {history_summary['keyframe_iters']}")
        print(f"  - 非关键帧迭代: {history_summary['nonkeyframe_iters']}")
        
        # 保存可视化
        if rgb_vis3 is not None:
            cv2.imwrite('./keyframe_test_frame3.jpg', cv2.cvtColor(rgb_vis3, cv2.COLOR_RGB2BGR))
            print(f"✓ 可视化已保存: /tmp/keyframe_test_frame3.jpg")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print("\n生成的文件:")
    print("  - /tmp/keyframe_test_frame1.jpg")
    print("  - /tmp/keyframe_test_frame2.jpg")
    print("  - /tmp/keyframe_test_frame3.jpg")
    print("\n预期行为:")
    print("  - 第1帧: 关键帧（全视野，reasoning模式）")
    print("  - 第2帧: 如果P可投影 → 非关键帧（限制角度范围，action-only模式）")
    print("  - 第3帧: 如果P仍可投影 → 非关键帧；否则 → 新关键帧")


def test_keyframe_disabled():
    """测试关键帧模式禁用时的行为"""
    
    print("\n" + "="*60)
    print("测试关键帧模式禁用")
    print("="*60)
    
    cfg = {
        'keyframe_mode': False,  # 禁用关键帧模式
        'clip_dist': 2.0,
        'turn_angle_deg': 30.0,
        'num_theta': 40,
        'vlm_history_length': 3,
        'vlm_model': 'qwen3-vl-8b-instruct',
        'vlm_api_key': 'sk-83be1f30087144a2a901f8f060ccf543',
        'vlm_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'vlm_timeout': 10
    }
    
    agent = NavAgent(cfg=cfg)
    agent.reset(goal='keyboard')
    
    history_summary = agent.get_history_summary()
    print(f"关键帧模式: {history_summary['keyframe_mode']}")
    print("✓ 关键帧模式已禁用，所有帧都按原来的方式处理")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--disabled':
        test_keyframe_disabled()
    else:
        test_keyframe_mode()
