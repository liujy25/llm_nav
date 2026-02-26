# VLM导航系统重构进度文档

## 项目概述

将原有的基于射线投射的导航系统重构为VLFM风格的frontier-based导航系统。

---

## 已完成的功能

### 1. 完全替换为VLFM的frontier-based系统

#### 新增文件
- **`obstacle_map.py`**: 改编自VLFM的`ObstacleMap`类
  - 维护3D点云（odom坐标系）
  - 检测frontiers（探索/未探索边界）
  - 使用`frontier_exploration`库进行frontier检测
  
- **`geometry_utils.py`**: 坐标变换辅助函数
  - 点云变换
  - 坐标系转换工具

#### 修改的核心逻辑
- **移除了**：`voxel_map`、`explored_map`、射线投射逻辑
- **新增了**：`ObstacleMap`集成，frontier检测流程
- **简化了**：`_navigability()`方法，现在只负责更新ObstacleMap

#### 地图更新流程
```
深度图 → 点云（camera frame）→ 变换到odom frame → 高度过滤 → 障碍物地图 → frontier检测
```

#### 坐标系适配
- 使用`base_to_odom`矩阵（机器人base在odom frame中的位姿）
- 点云变换：`T_odom_cam = base_to_odom @ extrinsic`
- 保持了相机点云变换的准确性

---

### 2. Waypoint可视化增强

在BEV地图上，每个waypoint现在显示：
- **FOV扇形区域**（半透明蓝色）
  - 扇形半径：50像素
  - FOV角度：基于`self.fov`（从相机内参计算）
  - 方向：基于waypoint的`yaw`朝向
  - 半透明效果：30%扇形 + 70%原图
- **扇形边界线**（蓝色实线）
- **中心圆圈**（蓝色填充+白色边框）
- **编号**（白色文字，显示waypoint ID）

#### 实现位置
- [`_generate_bev_with_waypoints()`](nav_agent.py:231-327)

#### 角度转换
```python
start_angle = 90 - np.rad2deg(wp_yaw) - fov_angle / 2
end_angle = 90 - np.rad2deg(wp_yaw) + fov_angle / 2
```

---

### 3. 批量查看waypoint

#### 功能增强
- `get_waypoint_rgb()`现在接受`wp_ids`参数（列表）
- 一次Function Call可查看多个waypoint
- 减少VLM的迭代次数，提高决策效率

#### 实现位置
- [`_handle_function_call()`](nav_agent_fc_methods.py)中的`get_waypoint_rgb`处理逻辑

---

### 4. BEV地图显示frontiers

#### 可视化
- 在BEV地图上用**绿色小圆点**标注所有当前frontiers
- 同时显示历史waypoints（带FOV扇形）和当前frontiers

#### 实现位置
- [`_generate_bev_with_waypoints()`](nav_agent.py:231-327)接受`frontiers`参数
- 在[`_nav_with_function_call()`](nav_agent.py:580-583)中传递`self.obstacle_map.frontiers`

#### 绘制逻辑
```python
# 先绘制frontiers（绿色小圆点）
if frontiers is not None and len(frontiers) > 0:
    GREEN = (0, 255, 0)  # BGR格式
    for frontier_xy in frontiers:
        frontier_px = self.obstacle_map._xy_to_px(np.array([frontier_xy]))[0]
        cv2.circle(bev, tuple(frontier_px.astype(int)), 5, GREEN, -1)  # 绿色填充
        cv2.circle(bev, tuple(frontier_px.astype(int)), 5, WHITE, 1)  # 白色边框
```

---

### 5. Frontier选择流程

新的导航决策流程：
```
全局frontiers → FOV过滤 → 图像投影过滤 → 标注可见frontiers → VLM选择 → 转换为(r, theta)动作
```

#### 关键方法
- [`_get_visible_frontiers()`](nav_agent.py:329-395): 过滤当前视野内可见的frontiers
  - FOV检查（基于相机内参）
  - 图像投影检查（确保在图像范围内）
  - **无距离过滤**（与原系统不同）

---

### 6. Frontier全局ID管理（单step内一致性）✅

> **重要设计决策**：Frontier ID只需在**单个推理step内**保持一致，不需要跨时间步持久化。这大大简化了实现。

#### 实现的方法

**[`_update_frontier_registry()`](nav_agent.py:877-926)**
- 在每次[`_navigability()`](nav_agent.py:928-956)后自动调用
- **每个step重新分配简单递增ID**（1, 2, 3...）
- 不维护跨时间步的ID持久性
- 实现简单高效

**[`_find_frontier_id()`](nav_agent.py:489-511)**
- 通过坐标匹配找到frontier的全局ID
- 使用距离阈值（默认0.1米）进行匹配
- 返回匹配的frontier ID或None

#### 修改的方法

**[`_annotate_frontiers()`](nav_agent.py:513-558)**
- 使用全局ID标注frontiers（而非临时编号）
- 通过`_find_frontier_id()`匹配每个可见frontier的全局ID
- 在RGB图像上显示全局一致的frontier ID
- 绿色圆圈 + 白色数字

**[`_register_waypoint()`](nav_agent.py:691-709)**
- 保存相机内参（`intrinsic`）
- 保存相机外参（`extrinsic`）
- 为后续从waypoint视角计算可见frontiers提供必要参数

**[`_get_visible_frontiers_from_pose()`](nav_agent.py:407-487)**
- 从任意位姿计算可见的frontiers
- 接受位置、朝向、相机参数作为输入
- 返回可见frontier列表（带像素坐标）
- 用于waypoint视角的frontier检测

**[`_handle_function_call()`](nav_agent.py:700-748)中的`get_waypoint_rgb`处理**
- 计算从waypoint视角能看到的frontiers
- 调用`_get_visible_frontiers_from_pose()`获取可见frontiers
- 使用`_annotate_frontiers()`标注frontiers（带全局ID）
- 返回标注后的RGB图像

#### 工作流程

```
每个推理step开始
    ↓
地图更新 → 更新frontier注册表 → 分配简单递增ID (1, 2, 3...)
    ↓
当前视角 → 获取可见frontiers → 匹配全局ID → 标注RGB
    ↓
Waypoint视角 → 计算可见frontiers → 匹配全局ID → 标注RGB
    ↓
BEV地图 → 显示所有frontiers（绿色圆点）
    ↓
VLM决策（所有视角中frontier ID一致）
```

#### 关键特性

1. **单step内ID一致性**：同一推理step中，当前RGB、waypoint RGB、BEV地图显示相同的frontier ID
2. **简单递增ID**：每个step的frontier ID从1开始递增，易于理解
3. **多视角支持**：VLM可以在不同视角中看到一致标注的frontiers
4. **无跨时间步持久性**：不需要维护复杂的frontier跟踪逻辑

---

### 7. Web可视化更新 ✅

#### 删除的内容
- "Current Cycle"历史记录部分
- "Historical Keyframes"历史记录部分
- 相关的CSS样式和JavaScript函数

#### 新增的内容
- **BEV Map显示区域**（与RGB Visualization和LLM Response并列）
- 3列布局：RGB Visualization | BEV Map | LLM Response

#### 修改的文件
- [`templates/dashboard.html`](templates/dashboard.html)
  - 删除了`.history-section`和`.history-container`的CSS
  - 更新了`clearDisplay()`函数
  - 更新了`updateDashboard()`函数以处理BEV map数据
  - 删除了`updateHistory()`和`createHistoryItem()`函数

#### 新的布局结构
```html
<div class="dashboard-grid">
    <div class="dashboard-section">
        <h2>RGB Visualization</h2>
        <img id="rgbVis" />
    </div>
    <div class="dashboard-section">
        <h2>BEV Map</h2>
        <img id="bevMap" />
    </div>
    <div class="dashboard-section">
        <h2>LLM Response</h2>
        <pre id="llmResponse"></pre>
    </div>
</div>
```

---

### 8. VLM Prompt优化 ✅

#### 增强的描述内容

**图像说明**
- 详细说明了BEV Map的含义（红色=当前位置，蓝色=waypoints，绿色=frontiers）
- 详细说明了Current RGB View的含义（绿色圆圈+数字=frontiers）
- 强调了frontier ID在所有视角中的一致性

**动作说明**
- 明确了3种可用动作的格式和用途
- 提供了具体的JSON格式示例
- 说明了每种动作的使用场景

**决策策略**
- 优先探索新frontiers
- 使用BEV地图理解全局布局
- 必要时查看waypoint视角

#### 实现位置
- [`_nav_with_function_call()`](nav_agent.py:757-795)中的prompt构建

#### Prompt结构
```
Iteration {iter}: Finding {goal}

IMAGE DESCRIPTIONS:
- Image 1 (BEV Map): ...
- Image 2 (Current RGB View): ...

AVAILABLE ACTIONS:
1. SELECT FRONTIER: ...
2. VIEW WAYPOINT: ...
3. GO TO WAYPOINT: ...

DECISION STRATEGY:
- ...
```

---

### 9. 清理遗留代码 ✅

#### 已删除的keyframe相关代码
- `/api/reset`端点中的keyframe参数
- `timing_data`中的`is_keyframe`和`keyframe_mode_enabled`字段
- `history_summary`的保存逻辑
- `current_cycle`和`historical_keyframes`的构建逻辑

#### 修改的文件
- [`nav_server.py`](nav_server.py)
  - 简化了`/api/reset`端点
  - 清理了`/api/step`中的keyframe历史保存逻辑
  - 删除了`viz_state`中的cycle/keyframes字段

---

### 10. Log保存重构 ✅

#### 新的保存内容
每个iteration保存以下文件：
1. **`iter_XXXX_rgb_original.jpg`**: 原始RGB观测（未标注）
2. **`iter_XXXX_rgb_annotated.jpg`**: 标注后的RGB（带frontier ID）
3. **`iter_XXXX_bev_map.jpg`**: 传递给VLM的BEV地图
4. **`iter_XXXX_conversation.txt`**: VLM完整对话历史（纯文本，不含图片）
5. **`iter_XXXX_timing.json`**: 时序和元数据
   - `total_time`: 总耗时
   - `fc_iterations`: Function Call迭代次数

#### 删除的保存内容
- `voxel_map.jpg`（已废弃）
- `explored_map.jpg`（已废弃）
- `response.txt`（替换为conversation.txt）
- `prompt.txt`（已包含在conversation中）

#### 实现位置
- [`_nav_with_function_call()`](nav_agent.py:730-910)返回值增强
  - 在`timing_info`中添加`bev_map`, `rgb_original`, `prompt`, `conversation`
- [`nav_server.py`](nav_server.py:239-285)的`/api/step`端点
  - 保存原始RGB、标注RGB、BEV地图
  - 保存VLM对话历史
  - 更新timing.json结构

---

## 待完成的任务

暂无待完成任务。所有核心功能已实现。

---

## 关键技术点

### Frontier匹配策略
- **单step内匹配**：使用空间距离阈值（0.1米）匹配frontier
- **简单递增ID**：每个step从1开始分配ID
- **无持久化需求**：不需要跨时间步跟踪frontier

### 坐标系转换
- **odom frame**: 全局坐标系
- **base frame**: 机器人本体坐标系
- **camera frame**: 相机坐标系

变换关系：
```
T_odom_cam = T_odom_base @ T_base_cam
T_cam_odom = inv(T_odom_cam)
```

### FOV计算
```python
fx = K[0, 0]
fov_horizontal = 2 * arctan(W / (2 * fx))
```

---

## 依赖要求

- `frontier_exploration`库
  - `frontier_exploration.frontier_detection.detect_frontier_waypoints`
  - `frontier_exploration.utils.fog_of_war.reveal_fog_of_war`

---

## 测试建议

### 1. Frontier ID一致性测试
- 在同一个step中，检查当前RGB、waypoint RGB、BEV地图中的frontier ID是否一致
- 验证VLM能否正确引用frontier ID

### 2. 多视角测试
- 测试waypoint RGB是否正确显示该视角的可见frontiers
- 验证frontier标注的准确性

### 3. Web可视化测试
- 验证BEV地图是否正确显示
- 检查RGB和BEV地图的同步更新
- 确认LLM响应正确显示

### 4. 边界情况测试
- 没有frontier时的处理
- waypoint没有可见frontier时的处理
- frontier数量变化时的处理

---

## 文件清单

### 新增文件
- `obstacle_map.py` (11KB)
- `geometry_utils.py` (3.8KB)

### 修改文件
- `nav_agent.py` (44KB+)
  - `__init__`: 添加ObstacleMap和frontier_registry
  - `reset()`: 重置frontier_registry
  - `_navigability()`: 简化为只更新ObstacleMap，调用`_update_frontier_registry()`
  - `_update_frontier_registry()`: 每个step重新分配frontier ID
  - `_find_frontier_id()`: 通过坐标匹配找到frontier ID
  - `_generate_bev_with_waypoints()`: 添加frontiers可视化
  - `_get_visible_frontiers()`: 过滤可见frontiers
  - `_get_visible_frontiers_from_pose()`: 从任意位姿计算可见frontiers
  - `_annotate_frontiers()`: 使用全局ID标注frontiers
  - `_register_waypoint()`: 保存相机内参和外参
  - `_nav_with_function_call()`: 使用frontier逻辑，优化prompt

- `nav_agent_fc_methods.py`
  - `get_waypoint_rgb`: 支持批量查看，添加frontier标注

- `templates/dashboard.html`
  - 删除history sections
  - 添加BEV map显示
  - 更新JavaScript函数

---

## 下一步行动

1. **清理遗留代码**（可选）
   - 清理`nav_server.py`中的keyframe相关代码
   - 简化数据结构

2. **系统测试**
   - 运行完整的导航任务
   - 验证frontier ID一致性
   - 测试Web可视化

3. **性能优化**（如需要）
   - 优化frontier检测性能
   - 减少图像处理开销

---

## 设计决策记录

### Frontier ID持久性策略
**决策**：Frontier ID只在单个推理step内保持一致，不跨时间步持久化

**理由**：
1. VLM只需要在当前决策中识别frontiers
2. 跨时间步的frontier跟踪复杂且容易出错
3. 简单递增ID易于理解和调试
4. 满足多视角一致性需求

**影响**：
- 简化了实现逻辑
- 提高了系统鲁棒性
- 不影响VLM的决策能力

---

## 2026-02-26 更新记录

### 11. Log保存重构完成 ✅

#### 新增功能
**[`vlm.py`](vlm.py:236-304)**: 添加`get_conversation_history()`方法
- 返回纯文本格式的完整对话历史
- 包含系统指令、初始prompt和所有对话轮次
- 自动过滤图片内容，只保留文本
- 正确处理assistant content的list和string两种格式

**[`nav_agent.py`](nav_agent.py:807-920)**: 修复prompt保存
- 在Function Call循环前保存`initial_prompt`
- 所有`timing_info`中使用初始prompt而非被修改的prompt

**[`nav_server.py`](nav_server.py:285-318,636-673)**: 更新显示逻辑
- 实时显示和历史查看都读取`conversation.txt`
- 修复图像路径：使用`rgb_annotated.jpg`和`bev_map.jpg`
- web端显示完整VLM对话历史

#### Bug修复
1. **[`nav_agent.py`](nav_agent.py:62)**: 修复初始化顺序错误
   - 在`__init__`中提前定义`self.scale`
   - 避免在初始化`ObstacleMap`时访问未定义的属性

2. **[`obstacle_map.py`](obstacle_map.py:168)**: 修复类型错误
   - 将`current_point`从tuple转换为numpy数组

3. **[`vlm.py`](vlm.py:285-295)**: 修复content类型错误
   - 正确处理assistant content的list和string格式

4. **[`nav_server.py`](nav_server.py:437)**: 修复机器人朝向
   - 机器人到达frontier后朝向移动方向（`yawg = yaw0 + theta`）

5. **[`nav_server.py`](nav_server.py:493-512)**: 禁用VLM stop check
   - 直接返回False，不调用VLM
   - 减少VLM调用次数

#### Web端优化
**[`templates/dashboard.html`](templates/dashboard.html:196-428)**: 修改为两列布局
- 左列：RGB Visualization和BEV Map垂直排列
- 右列：LLM Response占满整列
- 使用CSS Grid实现响应式布局

#### 调试信息添加
**[`obstacle_map.py`](obstacle_map.py:112-189)**: 地图更新调试
- 打印点云生成统计
- 打印障碍物检测统计
- 打印探索区域统计
- 打印frontier检测数量

**[`nav_agent.py`](nav_agent.py:349-415)**: Frontier可见性调试
- 打印总frontier数量
- 打印机器人位置和FOV信息
- 打印每个frontier的检查结果（FOV、投影、边界）
- 打印最终可见frontier数量

**[`obstacle_map.py`](obstacle_map.py:342-352)**: 可视化调试
- 打印机器人位置的坐标转换
- 检查机器人位置是否在地图范围内
- 警告超出边界的情况

---

### 12. 地图配置优化 ✅

#### 配置变更
**[`nav_agent.py`](nav_agent.py:30,62)**: 调整地图参数
- **Voxel size**: 1cm → 10cm
- **Pixels per meter**: 100 → 10
- **地图尺寸**: 5000x5000 pixels (50m x 50m) → 200x200 pixels (20m x 20m)
- **实际覆盖范围**: ±25m → ±10m

#### 优势
1. **降低内存占用**：从25M像素降至40K像素（减少99.8%）
2. **提高处理速度**：更小的地图尺寸加快frontier检测
3. **适合室内导航**：20m x 20m足够覆盖单个房间或楼层
4. **保持精度**：10cm分辨率对于机器人导航足够精确

---

### 保存的Log文件

每个iteration保存以下文件：
1. **`iter_XXXX_rgb_original.jpg`**: 原始RGB观测（未标注）
2. **`iter_XXXX_rgb_annotated.jpg`**: 标注后的RGB（带frontier ID）
3. **`iter_XXXX_bev_map.jpg`**: 传递给VLM的BEV地图
4. **`iter_XXXX_conversation.txt`**: VLM完整对话历史（纯文本，不含图片）
5. **`iter_XXXX_timing.json`**: 时序和元数据
   - `total_time`: 总耗时
   - `fc_iterations`: Function Call迭代次数

---

### 已知问题和待解决

1. **Frontier可见性问题**：
   - 地图更新正常，检测到frontier
   - 但frontier可能不在相机FOV内，导致VLM看不到
   - 需要通过调试信息确认具体原因

2. **VLM超时问题**：
   - 阿里云Qwen3.5-plus推理时间较长
   - 建议在agent_cfg中设置`'vlm_timeout': 120`

3. **ROS 2共享内存问题**：
   - `RTPS_TRANSPORError`
   - 解决方法：`rm -rf /dev/shm/fastrtps_*`
