# Function Call 导航功能实现指南

## 已完成的工作 ✅

### 1. vlm.py
- ✅ 添加了 `call_with_tools()` 方法
- ✅ 添加了 `add_tool_result()` 方法
- ✅ 修改了 `__init__` 使用 `conversation_history`
- ✅ 修改了 `reset()` 方法

### 2. nav_agent.py (部分完成)
- ✅ 修改了 `__init__` 添加waypoint管理
- ✅ 简化了 `_initialize_vlms()`
- ✅ 修改了 `reset()` 方法

### 3. 新方法文件
- ✅ 创建了 `nav_agent_fc_methods.py` 包含所有新方法

## 剩余工作清单

### Step 1: 删除nav_agent.py中不需要的方法

删除以下方法（约600行代码）：
- `_construct_keyframe_prompt()` (行80-150)
- `_construct_nonkeyframe_prompt()` (行152-199)
- `_encode_image_to_base64()` (行201-228)
- `_build_vlm_history_for_keyframe()` (行230-270)
- `_build_vlm_history_for_nonkeyframe()` (行272-337)
- `_determine_frame_type()` (行589-637)
- `get_history_summary()` (行451-511)
- `_transform_waypoint_to_current_base()` (行517-549)
- `_can_project_waypoint()` (行551-587)

### Step 2: 将nav_agent_fc_methods.py中的方法添加到nav_agent.py

在 `_build_initial_prompt()` 方法之后（约450行），添加以下方法：

1. `_generate_bev_with_waypoints()`
2. `_annotate_frontiers()`
3. `_get_function_definitions()`
4. `_handle_function_call()`
5. `_register_waypoint()`
6. `_nav_with_function_call()`

### Step 3: 修改 `_nav()` 方法

将原来的 `_nav()` 方法（约1091-1229行）完全替换为：

```python
def _nav(self, obs: dict, goal: str, iter: int, goal_description: str = ""):
    """主导航逻辑 - 使用Function Call模式"""
    return self._nav_with_function_call(obs, goal, iter, goal_description)
```

### Step 4: 修改 nav_server.py

#### 4.1 简化 `/navigation_reset` 端点

删除关键帧相关配置：
```python
agent_cfg = {
    'max_fc_iterations': data.get('max_fc_iterations', 5),
    'vlm_model': '/data/sea_disk0/liujy/models/Qwen/Qwen3-VL-8B-Instruct/',
    'vlm_api_key': 'EMPTY',
    'vlm_base_url': 'http://10.15.89.71:34134/v1/',
}
```

#### 4.2 修改 `/navigation_step` 端点

在action解析部分添加 `go_to_waypoint` 处理：

```python
# 在现有的action处理之前添加
if isinstance(action, dict) and action.get('action_type') == 'go_to_waypoint':
    target_pos = action['target_pos']
    target_yaw = action['target_yaw']
    
    goal_pose = build_goal_pose(
        target_pos[0], target_pos[1], target_pos[2], target_yaw
    )
    
    return jsonify({
        'action_type': 'go_to_waypoint',
        'goal_pose': {
            'frame_id': 'odom',
            'pose': goal_pose
        },
        'iteration': iteration,
        'finished': False,
        'timing': timing_info
    })
```

删除关键帧历史相关的可视化代码（约308-378行）。

### Step 5: 修改 llm_nav_client.py

#### 5.1 简化 `__init__`

删除 keyframe_mode 相关参数。

#### 5.2 简化 `reset_navigation`

删除 keyframe 配置。

#### 5.3 修改 `run` 方法

在action处理部分添加：

```python
if action_type == 'go_to_waypoint':
    self.get_logger().info('Executing GO TO WAYPOINT')
    try:
        self.navigate_to_pose(goal_pose_stamped)
        self.get_logger().info('Waypoint navigation completed.')
    except NavigationStoppedException:
        self.get_logger().info('Waypoint navigation stopped')
        break
    except Exception as e:
        self.get_logger().error(f'Failed to navigate to waypoint: {e}')
    continue
```

## 快速实现方案

如果你想快速完成，可以：

1. **备份现有文件**
   ```bash
   cp nav_agent.py nav_agent.py.backup
   cp nav_server.py nav_server.py.backup
   cp llm_nav_client.py llm_nav_client.py.backup
   ```

2. **手动编辑nav_agent.py**
   - 删除上述列出的不需要的方法
   - 从 `nav_agent_fc_methods.py` 复制所有方法到合适位置
   - 修改 `_nav()` 方法调用 `_nav_with_function_call()`

3. **修改nav_server.py和llm_nav_client.py**
   按照上述说明进行修改

4. **测试**
   ```bash
   # 启动服务器
   python nav_server.py --port 1874 --host 10.19.126.158
   
   # 启动客户端
   python llm_nav_client.py --goal "chair" --server http://10.19.126.158:1874
   ```

## 注意事项

1. 确保所有import语句正确（特别是 `mat_to_yaw` 函数）
2. 检查 `point_to_pixel` 等工具函数的导入
3. 测试时注意观察VLM的Function Call行为
4. 检查waypoint注册和BEV地图生成是否正常

## 预期效果

实现完成后，系统将：
1. 在每次迭代时生成BEV地图（显示历史waypoints）
2. 在当前视角标注frontiers（A, B, C...）
3. VLM可以主动调用 `get_waypoint_rgb()` 查看历史位置
4. VLM可以调用 `go_to()` 返回历史waypoint
5. 所有决策过程可追踪（通过Function Call日志）
