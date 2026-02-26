"""
Function Call相关的新方法
将这些方法添加到nav_agent.py的NavAgent类中
"""

def _generate_bev_with_waypoints(self, current_pos: np.ndarray) -> np.ndarray:
    """
    生成标注了历史waypoints的BEV地图
    
    Parameters
    ----------
    current_pos : np.ndarray
        当前机器人位置（odom坐标系）
        
    Returns
    -------
    np.ndarray
        标注后的BEV图像（RGB）
    """
    # 复制explored_map作为基础
    bev = self.explored_map.copy()
    
    # 标注当前位置（红色大圆）
    current_grid = self._global_to_grid(current_pos)
    cv2.circle(bev, current_grid, 30, RED, -1)
    cv2.putText(bev, "YOU", (current_grid[0]-20, current_grid[1]-35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
    
    # 标注历史waypoints（蓝色圆+编号）
    for wp_id, wp_data in self.waypoint_registry.items():
        wp_pos = wp_data['pos']
        wp_grid = self._global_to_grid(wp_pos)
        
        # 画圆
        cv2.circle(bev, wp_grid, 20, (255, 0, 0), -1)  # 蓝色
        cv2.circle(bev, wp_grid, 20, WHITE, 2)  # 白色边框
        
        # 画编号
        text = str(wp_id)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(bev, text, (wp_grid[0]-tw//2, wp_grid[1]+th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    
    # 添加图例
    legend_y = 50
    cv2.putText(bev, "BEV Map with Waypoints", (20, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2)
    cv2.putText(bev, f"Total Waypoints: {len(self.waypoint_registry)}", (20, legend_y+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    
    return bev


def _annotate_frontiers(self, rgb: np.ndarray, a_final: list, K, T_cam_base):
    """
    在当前RGB图像上标注可选择的frontiers
    
    Parameters
    ----------
    rgb : np.ndarray
        当前RGB图像
    a_final : list
        候选动作列表 [(r, theta), ...]
    K : np.ndarray
        相机内参
    T_cam_base : np.ndarray
        cam->base变换矩阵
        
    Returns
    -------
    rgb_annotated : np.ndarray
        标注后的RGB图像
    frontier_map : dict
        frontier映射 {frontier_id: (r, theta)}
    """
    from utils import point_to_pixel
    
    rgb_annotated = rgb.copy()
    frontier_map = {}
    
    # 使用字母标识frontiers（A, B, C...）
    for idx, (r, theta) in enumerate(a_final):
        if idx >= 26:  # 最多26个frontiers
            break
        
        label = chr(65 + idx)  # A-Z
        
        # 投影waypoint到图像（80%距离处）
        r_waypoint = r * 0.8
        p_base = np.array([r_waypoint * np.cos(theta), r_waypoint * np.sin(theta), 0.0])
        uv = point_to_pixel(p_base, K, T_cam_base)
        
        if uv is None:
            continue
        
        pixel_pos = (int(round(uv[0][0])), int(round(uv[0][1])))
        H, W = rgb.shape[:2]
        
        if not (0 <= pixel_pos[0] < W and 0 <= pixel_pos[1] < H):
            continue
        
        frontier_map[label] = (r, theta)
        
        # 画绿色圆圈+字母
        cv2.circle(rgb_annotated, pixel_pos, 25, GREEN, 3)
        cv2.putText(rgb_annotated, label, (pixel_pos[0]-10, pixel_pos[1]+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 3)
    
    # 添加标题
    cv2.putText(rgb_annotated, "Current View - Select Frontier", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    
    return rgb_annotated, frontier_map


def _get_function_definitions(self) -> list:
    """返回Function定义"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_waypoint_rgb",
                "description": "查看历史waypoint的第一人称RGB视角，帮助回忆该位置的环境",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wp_id": {
                            "type": "integer",
                            "description": f"waypoint编号，可选范围：{list(self.waypoint_registry.keys()) if self.waypoint_registry else 'none'}"
                        }
                    },
                    "required": ["wp_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "go_to",
                "description": "导航回到某个历史waypoint位置",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wp_id": {
                            "type": "integer",
                            "description": "目标waypoint编号"
                        }
                    },
                    "required": ["wp_id"]
                }
            }
        }
    ]


def _handle_function_call(self, tool_name: str, args: dict) -> dict:
    """
    处理Function Call
    
    Parameters
    ----------
    tool_name : str
        函数名称
    args : dict
        函数参数
        
    Returns
    -------
    dict
        {'success': bool, 'result': any, 'error': str}
    """
    if tool_name == 'get_waypoint_rgb':
        wp_id = args.get('wp_id')
        
        if wp_id not in self.waypoint_registry:
            return {
                'success': False,
                'result': None,
                'error': f"Waypoint {wp_id} not found. Available: {list(self.waypoint_registry.keys())}"
            }
        
        return {
            'success': True,
            'result': self.waypoint_registry[wp_id]['rgb'],
            'error': None
        }
    
    elif tool_name == 'go_to':
        wp_id = args.get('wp_id')
        
        if wp_id not in self.waypoint_registry:
            return {
                'success': False,
                'result': None,
                'error': f"Waypoint {wp_id} not found"
            }
        
        wp_data = self.waypoint_registry[wp_id]
        return {
            'success': True,
            'result': {
                'action_type': 'go_to_waypoint',
                'target_pos': wp_data['pos'],
                'target_yaw': wp_data['yaw']
            },
            'error': None
        }
    
    return {
        'success': False,
        'result': None,
        'error': f"Unknown function: {tool_name}"
    }


def _register_waypoint(self, obs: dict, action: tuple):
    """
    注册新waypoint到registry
    
    Parameters
    ----------
    obs : dict
        当前观测
    action : tuple
        (r, theta) 动作
    """
    r, theta = action
    T_odom_base = obs['base_to_odom_matrix']
    
    # 计算waypoint全局位置
    p_base = np.array([r * np.cos(theta), r * np.sin(theta), 0.0, 1.0])
    p_odom = T_odom_base @ p_base
    
    # 计算yaw
    from nav_agent import mat_to_yaw
    yaw = mat_to_yaw(T_odom_base)
    
    # 保存
    self.waypoint_registry[self.next_wp_id] = {
        'pos': p_odom[:3],
        'rgb': obs['rgb'].copy(),
        'iter': self.step_ndx,
        'yaw': yaw
    }
    
    print(f"[NavAgent] Registered waypoint {self.next_wp_id} at {p_odom[:3]}")
    self.next_wp_id += 1


def _nav_with_function_call(self, obs: dict, goal: str, iter: int, goal_description: str = ""):
    """
    支持Function Call的导航决策（完全替换原_nav方法）
    
    Returns
    -------
    response : str
        VLM完整响应
    rgb_vis : np.ndarray
        可视化图像
    action : tuple/dict
        动作（可能是frontier或waypoint）
    timing_info : dict
        时序信息
    """
    t_start = time.time()
    
    # 1. 计算可导航性
    a_initial = self._navigability(obs)
    a_final = self._action_proposer(a_initial, obs['base_to_odom_matrix'])
    
    # 2. 生成BEV和标注frontiers
    current_pos = obs['base_to_odom_matrix'][:3, 3]
    bev_map = self._generate_bev_with_waypoints(current_pos)
    
    K = obs['intrinsic']
    T_cam_base = obs['extrinsic'] @ obs['base_to_odom_matrix']
    rgb_annotated, frontier_map = self._annotate_frontiers(obs['rgb'], a_final, K, T_cam_base)
    
    # 3. 构建prompt
    frontier_list = ', '.join(frontier_map.keys()) if frontier_map else 'none'
    wp_list = ', '.join(map(str, self.waypoint_registry.keys())) if self.waypoint_registry else 'none'
    
    prompt = f"""Iteration {iter}: Finding {goal}

Image 1: BEV map (YOU=red, waypoints={wp_list})
Image 2: Current view (frontiers={frontier_list})

Actions:
1. Select frontier: {{"action": "frontier", "id": "A"}}
2. View waypoint: call get_waypoint_rgb(wp_id)
3. Go to waypoint: call go_to(wp_id)

Decide based on BEV global info and current view."""
    
    # 4. Function Call循环
    tools = self._get_function_definitions()
    images = [bev_map, rgb_annotated]
    
    for fc_iter in range(self.cfg['max_fc_iterations']):
        print(f"[NavAgent] FC iteration {fc_iter+1}/{self.cfg['max_fc_iterations']}")
        
        response = self.actionVLM.call_with_tools(images, prompt, tools)
        
        if response['type'] == 'tool_call':
            # VLM调用函数
            for tool_call in response['tool_calls']:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id = tool_call['id']
                
                print(f"[NavAgent] VLM called: {tool_name}({tool_args})")
                
                # 执行函数
                result = self._handle_function_call(tool_name, tool_args)
                
                if not result['success']:
                    # 添加错误结果
                    self.actionVLM.add_tool_result(
                        tool_id, tool_name, f"Error: {result['error']}"
                    )
                    prompt = f"Error occurred. Please try again or make direct decision."
                    images = []
                else:
                    if tool_name == 'get_waypoint_rgb':
                        # 返回waypoint视角
                        self.actionVLM.add_tool_result(
                            tool_id, tool_name,
                            f"Showing waypoint {tool_args['wp_id']} view"
                        )
                        prompt = f"This is waypoint {tool_args['wp_id']}. Continue decision."
                        images = [result['result']]
                    
                    elif tool_name == 'go_to':
                        # 直接返回
                        t_end = time.time()
                        return (
                            response['content'],
                            rgb_annotated,
                            result['result'],
                            {'total_time': t_end - t_start, 'fc_iterations': fc_iter + 1}
                        )
        else:
            # 最终决策
            try:
                decision = self._eval_response(response['content'])
                if decision.get('action') == 'frontier':
                    frontier_id = decision['id']
                    if frontier_id in frontier_map:
                        action = frontier_map[frontier_id]
                        self._register_waypoint(obs, action)
                        
                        t_end = time.time()
                        return (
                            response['content'],
                            rgb_annotated,
                            action,
                            {'total_time': t_end - t_start, 'fc_iterations': fc_iter + 1}
                        )
            except Exception as e:
                print(f"[NavAgent] Parse error: {e}")
            
            t_end = time.time()
            return (
                None,
                rgb_annotated,
                None,
                {'total_time': t_end - t_start, 'fc_iterations': fc_iter + 1}
            )
    
    # 超时
    t_end = time.time()
    return (
        None,
        rgb_annotated,
        None,
        {'total_time': t_end - t_start, 'fc_iterations': self.cfg['max_fc_iterations']}
    )
