# 视频生成 + PI3 集成说明

本文档说明如何把你自己的视频生成模型（如 Wan2.6）和 PI3 重建模型接入当前导航框架。

## 1. 当前接口位置

核心文件：`local_planning_pipeline.py`

已预留 2 个接口：

- `VideoGenerator.generate(data: VideoGenerationInput) -> VideoPrediction`
- `PI3Decoder.decode(data: PI3DecodeInput) -> np.ndarray`

默认实现为：

- `PlaceholderVideoGenerator`
- `PlaceholderPI3Decoder`

默认行为：抛出 `NotImplementedError`，pipeline 自动降级为原地单点路径（不会崩）。

---

## 2. 输入输出协议

### 2.1 VideoGenerator

#### 输入：`VideoGenerationInput`

- `rgb`: `np.ndarray`，形状 `(H, W, 3)`，RGB
- `depth`: `np.ndarray`，形状 `(H, W)`，深度图
- `intrinsic`: `np.ndarray`，`3x3` 相机内参
- `extrinsic`: `np.ndarray`，`4x4` 外参（odom->cam）
- `base_to_odom`: `np.ndarray`，`4x4` 机器人 base 在 odom 下位姿
- `subgoal`: `dict`，LLM 输出的局部语言目标
- `goal`: `str`，全局导航目标
- `iteration`: `int`，当前迭代

#### 输出：`VideoPrediction`

- `frames`: `np.ndarray`，形状 `(T, H, W, 3)`，建议 `uint8` RGB
- `fps`: `float`
- `metadata`: `dict`，可放模型名、采样参数等

### 2.2 PI3Decoder

#### 输入：`PI3DecodeInput`

- `video_prediction`: `VideoPrediction`
- `intrinsic`: `np.ndarray`，`3x3`
- `extrinsic`: `np.ndarray`，`4x4`
- `base_to_odom`: `np.ndarray`，`4x4`
- `subgoal`: `dict`

#### 输出：`traj_3d`

- 类型：`np.ndarray`
- 形状：`(N, 3)`
- 坐标系：**odom**（非常重要）
- 单位：米

pipeline 会自动调用 `compress_traj3d_to_2d()` 转成 `path_2d`。

---

## 3. 接入步骤

### 步骤 A：实现 VideoGenerator

示例（伪代码）：

```python
from local_planning_pipeline import VideoGenerator, VideoGenerationInput, VideoPrediction

class Wan26Generator(VideoGenerator):
    def __init__(self, endpoint: str, timeout_s: float = 10.0):
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def generate(self, data: VideoGenerationInput) -> VideoPrediction:
        # 1) 打包 rgb/depth/subgoal
        # 2) 调用 Wan2.6 服务
        # 3) 解码返回视频帧
        return VideoPrediction(frames=frames, fps=8.0, metadata={"model": "wan2.6"})
```

### 步骤 B：实现 PI3Decoder

```python
from local_planning_pipeline import PI3Decoder, PI3DecodeInput
import numpy as np

class PI3ServiceDecoder(PI3Decoder):
    def __init__(self, endpoint: str, timeout_s: float = 10.0):
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def decode(self, data: PI3DecodeInput) -> np.ndarray:
        # 1) 上传预测视频 + 相机参数 + subgoal
        # 2) 获得局部轨迹
        # 3) 转换到 odom 坐标系并返回 (N,3)
        return traj_3d_odom
```

### 步骤 C：在 server reset 时注入实现

在 `nav_server.py` 的 `navigation_reset()` 中替换：

```python
nav_state['pipeline'] = LocalPlanningPipeline(agent=nav_state['agent'])
```

为：

```python
nav_state['pipeline'] = LocalPlanningPipeline(
    agent=nav_state['agent'],
    video_generator=Wan26Generator(endpoint='http://xxx'),
    pi3_decoder=PI3ServiceDecoder(endpoint='http://yyy'),
)
```

---

## 4. 运行时状态与日志

`/navigation_step` 返回：

- `pipeline_status`: `ok | interface_not_implemented | pipeline_error`
- `pipeline_error`: 错误信息（可为空）
- `path_2d`: 压缩后的 2D 路径
- `subgoal`: 当前局部语言规划

日志目录会保存：

- `iter_xxxx_subgoal.json`
- `iter_xxxx_path2d.json`
- `iter_xxxx_traj3d.npy`
- `iter_xxxx_future_video.npy`（仅当视频生成成功）

---

## 5. 集成注意事项

1. **坐标系统一**：PI3 输出务必转换到 odom 坐标系。
2. **时间同步**：视频帧率与轨迹点频率建议有明确映射。
3. **异常处理**：接口失败时建议抛异常，pipeline 会自动降级，保证服务不中断。
4. **性能建议**：把视频生成与 PI3 解码控制在单步预算内，超时应快速失败并回退。
5. **可观测性**：在 `metadata` 中写入模型版本、参数，方便回放定位问题。

