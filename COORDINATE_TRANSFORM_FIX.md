# 坐标变换错位问题分析与修复

## 问题描述

第一帧中出现以下异常：
1. **Frontier位置错误**：生成的frontier在机器人前方（RGB视野中），但输出坐标是(-3, -4)，完全在机器人后方
2. **BEV地图障碍物错位**：可视化的障碍物完全在视野外，大概和视野偏差了90度

## 问题分析

### 1. 点云变换（✓ 正确）

从保存的点云文件分析：
```
点云总数: 440352
X范围: [0.718, 5.444]  # 全部为正值，说明在机器人前方
Y范围: [-4.674, 3.657]  # 左右分布
Z范围: [-0.116, 1.401]  # 高度范围合理
```

**结论**：点云从相机坐标系到odom坐标系的变换是正确的。

### 2. BEV地图坐标约定（✓ 正确）

从`_xy_to_px`函数分析：
```python
px[:, 0] = size/2 + x * scale  # 图像X = odom X
px[:, 1] = size/2 - y * scale  # 图像Y = -odom Y
```

坐标约定：
- 图像右侧 = odom +X方向（机器人前方）
- 图像上方 = odom +Y方向（机器人左侧）
- 图像下方 = odom -Y方向（机器人右侧）
- 图像左侧 = odom -X方向（机器人后方）

**结论**：BEV地图的坐标约定是正确的。

### 3. FOV角度传递（✗ 错误）

**问题根源**：`reveal_fog_of_war`函数的角度约定与我们的不一致。

在`frontier_exploration/utils/fog_of_war.py`第66行：
```python
angle_cv2 = np.rad2deg(wrap_heading(-current_angle + np.pi / 2))
```

这个函数内部会对输入角度做变换：`-current_angle + π/2`

**角度约定对比**：

| 机器人朝向 | agent_yaw | reveal_fog_of_war期望 | 原代码传入 | 结果 |
|-----------|-----------|---------------------|-----------|------|
| +X方向（前） | 0° | 0° | -0° = 0° | ✓ 正确 |
| +Y方向（左） | 90° | 90° | -90° = -90° | ✗ 错误 |
| -X方向（后） | 180° | 180° | -180° = -180° | ✗ 错误 |
| -Y方向（右） | 270° | 270° | -270° = -270° | ✗ 错误 |

**原代码**（obstacle_map.py 第199行）：
```python
current_angle=-agent_yaw,  # 错误：使用了负号
```

## 修复方案

### 修复1：FOV角度传递

**文件**：`obstacle_map.py` 第199行

**修改前**：
```python
new_explored = reveal_fog_of_war(
    top_down_map=self._navigable_map.astype(np.uint8),
    current_fog_of_war_mask=np.zeros_like(self._obstacle_map, dtype=np.uint8),
    current_point=np.array([agent_pixel[1], agent_pixel[0]], dtype=np.int32),
    current_angle=-agent_yaw,  # 错误
    fov=fov_deg,
    max_line_len=int(max_depth * self.pixels_per_meter),
)
```

**修改后**：
```python
# reveal_fog_of_war内部会做变换: angle_cv2 = -current_angle + π/2
# 我们的agent_yaw: 0度=+X方向, 90度=+Y方向
# reveal_fog_of_war期望: 0度=+Y方向, 90度=+X方向
# 因此需要传入: current_angle = agent_yaw (不需要负号)
new_explored = reveal_fog_of_war(
    top_down_map=self._navigable_map.astype(np.uint8),
    current_fog_of_war_mask=np.zeros_like(self._obstacle_map, dtype=np.uint8),
    current_point=np.array([agent_pixel[1], agent_pixel[0]], dtype=np.int32),
    current_angle=agent_yaw,  # 修复：去掉负号
    fov=fov_deg,
    max_line_len=int(max_depth * self.pixels_per_meter),
)
```

### 修复2：添加调试日志

为了验证修复效果，添加了详细的调试日志：

1. **变换矩阵日志**（第134-149行）：
   - 打印T_cam_odom和T_odom_cam矩阵
   - 打印base_to_odom矩阵
   - 打印样本点的变换结果

2. **机器人位姿日志**（第191-200行）：
   - 打印base_to_odom位置和旋转矩阵
   - 打印agent_yaw角度（弧度和度数）
   - 打印agent_pixel坐标
   - 打印FOV角度

3. **点云投影日志**（第181-188行）：
   - 打印XY点的范围
   - 打印像素点的范围

## 验证步骤

1. 运行导航程序，查看调试日志
2. 检查第一帧的输出：
   - 机器人位姿是否正确
   - 点云XY范围是否合理（X应该为正，表示前方）
   - 像素点范围是否合理（应该在地图右侧）
   - Frontier坐标是否在机器人前方
3. 查看BEV地图可视化：
   - 障碍物是否在机器人前方（地图右侧）
   - FOV扇形是否指向正确方向
   - Frontier是否在合理位置

## 可能的额外问题

如果修复后仍有问题，可能的原因：

1. **相机外参本身有误**：
   - 检查ROS TF树，确认camera_frame相对于base_frame的变换
   - 使用`ros2 run tf2_tools view_frames`生成TF树图
   - 检查相机是否真的朝向+X方向

2. **坐标系定义不一致**：
   - 确认odom坐标系的定义（X前Y左Z上 vs X右Y前Z上）
   - 确认base坐标系的定义
   - 确认camera坐标系的定义（OpenCV标准 vs ROS标准）

3. **BEV地图方向**：
   - 如果机器人在实际环境中朝向不是+X，需要调整初始朝向
   - 检查episode_origin的设置是否正确

## 总结

主要问题是`reveal_fog_of_war`函数的角度约定与我们的不一致，导致FOV扇形的朝向错误，进而影响explored area的计算和frontier的检测。

修复方法是去掉传入角度的负号，让角度约定保持一致。

添加的调试日志可以帮助验证修复效果，并在出现新问题时快速定位。
