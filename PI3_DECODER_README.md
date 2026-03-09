# PI3Decoder Implementation

## 概述

PI3Decoder 使用 Pi3 模型从预测的视频中重建 3D 轨迹，用于局部导航规划。

## 实现文件

- `pi3_decoder_impl.py` - Pi3Decoder 类实现
- `test_pi3_decoder.py` - 单元测试脚本
- `nav_server.py` - 集成到导航服务器（已修改）

## 使用方法

### 1. 准备 Pi3 模型权重

下载或准备 Pi3 模型的 checkpoint 文件（.pt 或 .safetensors 格式）。

### 2. 启动导航服务器

使用 `--pi3-checkpoint` 参数指定模型路径：

```bash
conda activate model_server
cd /home/liujy/mobile_manipulation/model_server/nav

python nav_server.py \
    --host 10.19.126.158 \
    --port 1874 \
    --pi3-checkpoint /path/to/pi3_checkpoint.pt
```

如果不提供 `--pi3-checkpoint` 参数，服务器会使用 placeholder 实现（回退到单点路径）。

### 3. 运行单元测试

```bash
conda activate model_server
cd /home/liujy/mobile_manipulation/model_server/nav

python test_pi3_decoder.py
```

## 实现细节

### Pi3Decoder 类

**初始化参数：**
- `checkpoint_path`: Pi3 checkpoint 文件路径
- `device`: 'cuda' 或 'cpu'
- `dtype`: 'bfloat16' 或 'float16'
- `pos_type`: 位置编码类型（默认 'rope100'）
- `decoder_size`: 解码器大小（'small', 'base', 'large'）

**主要方法：**
- `decode(data: PI3DecodeInput) -> np.ndarray`: 从视频预测中解码 3D 轨迹

### 数据流

1. **输入**: PI3DecodeInput
   - `video_prediction.frames`: (T, H, W, 3) RGB uint8
   - `intrinsic`: (3, 3) 相机内参
   - `extrinsic`: (4, 4) odom->cam 变换
   - `base_to_odom`: (4, 4) base->odom 变换

2. **预处理**:
   - 视频帧归一化到 [0, 1]
   - 重排维度: (T, H, W, 3) -> (1, T, 3, H, W)

3. **推理**:
   - 使用 Pi3 模型直接推理（不使用 Pi3XVO 包装器）
   - 输出 camera_poses: (1, T, 4, 4)

4. **后处理**:
   - 提取轨迹: camera_poses[0, :, :3, 3] -> (T, 3)
   - 坐标变换: camera frame -> odom frame
   - 使用 `transform_points()` 应用逆变换

5. **输出**: (N, 3) 轨迹在 odom 坐标系

### 坐标系变换

```
camera_frame -> odom_frame
```

- `extrinsic` 是 odom->cam 变换
- 需要逆变换: `T_cam_to_odom = inv(extrinsic)`
- 应用到轨迹点: `traj_odom = transform_points(T_cam_to_odom, traj_cam)`

## 错误处理

- 模型加载失败: 打印警告，使用 placeholder 实现
- 推理失败: 异常传播到 LocalPlanningPipeline，触发回退机制
- 回退行为: 返回单点路径（当前机器人位置）

## 与 LocalPlanningPipeline 集成

```python
from pi3_decoder_impl import Pi3Decoder
from local_planning_pipeline import LocalPlanningPipeline

# 创建 decoder
pi3_decoder = Pi3Decoder(
    checkpoint_path='/path/to/checkpoint.pt',
    device='cuda',
    dtype='bfloat16'
)

# 传递给 pipeline
pipeline = LocalPlanningPipeline(
    agent=nav_agent,
    pi3_decoder=pi3_decoder
)

# 使用
result = pipeline.plan(obs, goal, iteration)
# result['status'] 会是 'ok' 而不是 'interface_not_implemented'
# result['traj_3d'] 包含重建的 3D 轨迹
# result['path_2d'] 包含压缩的 2D 路径点
```

## 验证

运行单元测试验证：
- 坐标变换逻辑正确
- 视频预处理格式正确
- 数据流完整

使用实际 checkpoint 测试：
- 模型加载成功
- 推理运行无错误
- 输出轨迹合理（范围、方向）

## 注意事项

1. **模型选择**: 使用 Pi3（RGB-only）而不是 Pi3X（multimodal），避免对 rays 输出的依赖
2. **懒加载**: 模型在第一次调用 `decode()` 时加载，避免拖慢服务器启动
3. **内存管理**: 推理使用 torch.no_grad() 和 autocast 减少内存占用
4. **路径依赖**: Pi3 模块路径自动添加到 sys.path
