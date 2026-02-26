# 数据保存和离线测试使用说明

## 功能说明

为了方便调试和测试，添加了数据保存和离线测试功能，可以保存机器人的观测数据（RGB、Depth、相机参数、TF变换），然后在不启动机器人的情况下进行测试。

## 1. 保存数据

在ROS2 client端调用`save_observation_snapshot`方法保存当前观测数据。

### 方法1：在代码中调用

在[`llm_nav_client.py`](llm_nav_client.py:251)中添加了`save_observation_snapshot`方法，可以在任何地方调用：

```python
# 在LLMNavClient中调用
snapshot_dir = self.save_observation_snapshot(
    save_dir='test_data',  # 保存目录
    prefix='snapshot'       # 文件名前缀
)
```

### 方法2：添加服务调用

可以在需要保存数据的地方（例如导航开始时）调用：

```python
# 例如在run_navigation方法开始时
def run_navigation(self, goal_description: str):
    self.get_logger().info(f'Starting navigation: {goal_description}')
    
    # 保存第一帧数据用于测试
    snapshot_dir = self.save_observation_snapshot(
        save_dir='test_data',
        prefix='nav_start'
    )
    self.get_logger().info(f'Saved snapshot to {snapshot_dir}')
    
    # 继续正常的导航流程
    ...
```

### 保存的数据结构

```
test_data/
└── snapshot_20260226_170000/
    ├── rgb.png              # RGB图像
    ├── depth.png            # Depth图像（16位PNG，单位毫米）
    ├── params.npz           # 相机参数和TF变换（NumPy格式）
    └── info.txt             # 可读的文本信息
```

## 2. 离线测试

使用保存的数据进行离线测试，无需启动机器人。

### 基本用法

```bash
# 加载数据并测试
conda run -n model_server python test_offline_data.py test_data/snapshot_20260226_170000

# 保存测试结果到指定目录
conda run -n model_server python test_offline_data.py test_data/snapshot_20260226_170000 test_results
```

### 测试脚本功能

[`test_offline_data.py`](test_offline_data.py:1)脚本会：

1. 加载保存的RGB、Depth、相机参数和TF变换
2. 创建ObstacleMap并更新地图
3. 检测frontiers
4. 生成BEV地图可视化
5. 保存结果（如果指定了输出目录）
6. 显示可视化窗口（如果有显示器）

### 输出示例

```
Loading snapshot from: test_data/snapshot_20260226_170000
Loaded RGB: (540, 960, 3), dtype=uint8
Loaded Depth: (540, 960), dtype=float32, range=[0.500, 5.000]

Camera Intrinsic:
[[fx  0  cx]
 [ 0 fy  cy]
 [ 0  0   1]]

Robot position: [-0.249, -0.341, -0.220]
Robot yaw: -0.001 rad (-0.1 deg)

================================================================================
Testing ObstacleMap
================================================================================
[ObstacleMap] Episode origin set: [-0.249 -0.341 -0.220]
[ObstacleMap] Depth shape: (540, 960), valid points: 440380
...
[ObstacleMap] Detected 5 frontiers

Frontiers detected: 5
Frontier coordinates (odom frame):
  1: (2.500, -1.200)
  2: (3.100, 0.800)
  3: (1.800, 2.300)
  4: (4.200, -0.500)
  5: (2.900, 1.600)

Saved BEV map to test_results/bev_map.png
Saved RGB to test_results/rgb.png

Test completed!
```

## 3. 修改和扩展

### 自定义ObstacleMap参数

在[`test_offline_data.py`](test_offline_data.py:67)的`test_obstacle_map`函数中修改参数：

```python
obstacle_map = ObstacleMap(
    min_height=0.2,      # 最小障碍物高度
    max_height=1.5,      # 最大障碍物高度
    agent_radius=0.3,    # 机器人半径
    area_thresh=1.0,     # 最小frontier面积
    size=200,            # 地图大小（像素）
    pixels_per_meter=10, # 分辨率
)
```

### 添加更多测试

可以在`test_obstacle_map`函数中添加更多测试逻辑：

```python
def test_obstacle_map(obs: dict, save_dir: str = None):
    # ... 现有代码 ...
    
    # 测试不同的参数
    print("\n测试不同的agent_radius...")
    for radius in [0.2, 0.3, 0.4]:
        obstacle_map = ObstacleMap(agent_radius=radius, ...)
        obstacle_map.update_map(...)
        print(f"  radius={radius}: {len(obstacle_map.frontiers)} frontiers")
    
    # 测试不同的高度范围
    print("\n测试不同的高度范围...")
    for min_h, max_h in [(0.1, 1.0), (0.2, 1.5), (0.3, 2.0)]:
        obstacle_map = ObstacleMap(min_height=min_h, max_height=max_h, ...)
        obstacle_map.update_map(...)
        print(f"  height=[{min_h}, {max_h}]: {len(obstacle_map.frontiers)} frontiers")
```

## 4. 调试技巧

### 查看保存的数据

```python
import numpy as np
from PIL import Image

# 加载参数
params = np.load('test_data/snapshot_20260226_170000/params.npz')
print("Keys:", params.files)
print("Intrinsic:", params['intrinsic'])
print("T_cam_odom:", params['T_cam_odom'])

# 查看RGB
rgb = Image.open('test_data/snapshot_20260226_170000/rgb.png')
rgb.show()

# 查看Depth
depth_mm = np.array(Image.open('test_data/snapshot_20260226_170000/depth.png'))
depth_m = depth_mm / 1000.0
print(f"Depth range: [{depth_m.min():.3f}, {depth_m.max():.3f}] meters")
```

### 对比不同版本的结果

保存多个版本的测试结果，对比修改前后的差异：

```bash
# 修改前
python test_offline_data.py test_data/snapshot_20260226_170000 results_before

# 修改后
python test_offline_data.py test_data/snapshot_20260226_170000 results_after

# 对比
diff results_before/info.txt results_after/info.txt
```

## 5. 常见问题

### Q: 如何在导航过程中自动保存数据？

A: 在`run_navigation`方法中添加保存逻辑，例如每次迭代保存一次：

```python
for iteration in range(max_iterations):
    # 保存当前帧数据
    snapshot_dir = self.save_observation_snapshot(
        save_dir='test_data',
        prefix=f'iter_{iteration:04d}'
    )
    
    # 继续导航
    ...
```

### Q: 保存的数据占用空间大吗？

A: 每个snapshot大约2-5MB（取决于图像分辨率）：
- RGB: ~500KB (PNG压缩)
- Depth: ~1MB (16位PNG)
- Params: ~1KB (NumPy)

### Q: 如何批量测试多个snapshot？

A: 创建一个批量测试脚本：

```python
import os
import glob

snapshot_dirs = glob.glob('test_data/snapshot_*')
for snapshot_dir in sorted(snapshot_dirs):
    print(f"\nTesting {snapshot_dir}...")
    obs = load_snapshot(snapshot_dir)
    test_obstacle_map(obs, save_dir=f'results/{os.path.basename(snapshot_dir)}')
```

## 总结

通过数据保存和离线测试功能，你可以：
1. 快速迭代和调试，无需每次都启动机器人
2. 保存关键帧数据用于回归测试
3. 对比不同参数和算法的效果
4. 分享数据给其他开发者进行调试
