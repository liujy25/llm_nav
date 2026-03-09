# NavDreamer 中 π³ 与 MoGe-2 对齐并恢复尺度的复现流程

本文档只整理 **NavDreamer 论文中“使用 π³ 和 MoGe-2 对齐并恢复深度/尺度”** 的部分，不包含视频生成、VLM 打分和低层控制器执行。

---

## 1. 目标

输入：

- 一段已经被选中的导航视频 `V*`

输出：

- 一组带真实物理尺度的 waypoint 序列 `{W_t}`

核心任务分成两部分：

1. 使用 **π³** 从图像序列中恢复局部几何和归一化 waypoint
2. 使用 **MoGe-2** 提供米制深度参考，估计全局尺度因子 `S`，把 π³ 的 waypoint 缩放到真实物理空间

---

## 2. 总体思路

NavDreamer 的做法不是直接信任 π³ 输出的绝对尺度，而是：

- 先让 π³ 给出 pointmap 和归一化轨迹
- 再让 MoGe-2 对同样的图像给出 metric depth
- 对两者逐像素比较，估计一个全局尺度因子
- 最后把 π³ 输出的 waypoint 统一乘上这个尺度因子

可以概括成一句话：

> π³ 负责恢复“轨迹形状和方向”，MoGe-2 负责提供“真实米制标尺”。

---

## 3. 详细复现流程

### Step 1. 从最优视频中抽取图像序列

从最终选中的导航视频 `V*` 中按固定时间间隔抽帧，得到有序图像序列：

\[
\{I_t\}_{t=1}^N
\]

这里要做的事情：

- 统一视频帧率
- 固定抽帧间隔
- 保证图像时序不被打乱
- 不要额外做随机 crop 或打乱顺序

这一步的目的，是给 π³ 和 MoGe-2 构造同一组时序输入。

---

### Step 2. 运行 π³，获取 pointmap 和归一化 waypoint

把图像序列 `{I_t}` 输入 π³，得到两类输出：

#### 2.1 每帧的局部 pointmap

记作：

\[
X_t \in \mathbb{R}^{H \times W \times 3}
\]

其中每个像素对应一个 3D 点。

#### 2.2 归一化 waypoint 序列

记作：

\[
w_t \in \mathbb{R}^3
\]

如果 π³ 还输出 yaw，也一并保留。

这一步不要急着把 waypoint 当成“米制轨迹”，因为论文默认 π³ 的几何存在绝对尺度歧义，尤其在户外更明显。

---

### Step 3. 从 pointmap 中提取 π³ 的预测深度图

从每帧 pointmap `X_t` 中提取前向深度，构造：

\[
Z_t^{pred}(u,v)
\]

也就是每个像素对应的预测深度值。

这里要特别注意：

- 必须明确 π³ pointmap 的坐标系定义
- 提取的是“沿相机前向轴”的深度
- 不能误把横向或竖向坐标当成深度

最终得到：

\[
Z_t^{pred} \in \mathbb{R}^{H \times W}
\]

它表示 π³ 侧的深度图，但这个深度通常不具备可靠的绝对米制意义。

---

### Step 4. 运行 MoGe-2，获取 metric depth 参考图

对同样的图像序列 `{I_t}` 运行 MoGe-2，得到每一帧的参考深度图：

\[
D_t^{ref}
\]

要求：

- 输出必须是 metric depth
- 分辨率要和 `Z_t^{pred}` 一致，或插值到同一分辨率
- 像素位置必须严格对齐

至此，你会得到两套逐像素深度：

- `Z_t^{pred}`：来自 π³
- `D_t^{ref}`：来自 MoGe-2

---

### Step 5. 构造有效掩码，只保留可靠像素

论文没有对全部像素直接做比值，而是先定义一个有效掩码 `M_t`：

\[
M_t = \{(u,v)\mid \tau_{min} < D_t^{ref}(u,v) < \tau_{max}, \; Z_t^{pred}(u,v) > 0\}
\]

其中可靠深度范围通常取：

\[
\tau \in [0.5, 30] \text{ meters}
\]

也就是说，每个像素必须同时满足：

1. MoGe-2 深度在可信范围内
2. π³ 深度为正且有效

这一步的目的是过滤掉不可靠区域，例如：

- 天空
- 极远区域
- 无效深度
- 极端异常值

---

### Step 6. 计算逐像素尺度比

在每一帧的有效区域 `M_t` 中，对每个像素计算尺度比：

\[
s_t(u,v)=\frac{D_t^{ref}(u,v)}{Z_t^{pred}(u,v)}
\]

直觉上，这就是在问：

- MoGe-2 认为这个像素的真实深度是多少米
- π³ 认为这个像素深度是多少内部单位
- 两者之比暗示 π³ 的全局尺度偏差有多大

这一阶段不会立刻对每一帧单独缩放，而是先收集所有像素给出的局部尺度证据。

---

### Step 7. 跨帧做中位数共识，求全局尺度因子

把所有帧、所有有效像素上的尺度比合并起来，取中位数，得到全局尺度因子 `S`：

\[
S = \mathrm{median}
\left(
\bigcup_{t=1}^{N}
\left(
\frac{D_t^{ref}(u,v)}{Z_t^{pred}(u,v)}
\right)_{(u,v)\in M_t}
\right)
\]

这里选择中位数而不是均值，是为了增强鲁棒性，减小以下因素的影响：

- depth noise
- 局部几何错配
- 远处区域误差
- 异常值污染

这个步骤的核心假设是：

> π³ 恢复出来的几何形状大体是对的，主要问题是整体尺度偏大或偏小。

因此，一个全局标量 `S` 就可以对整个轨迹做统一尺度校正。

---

### Step 8. 用全局尺度因子缩放 waypoint

得到 `S` 之后，对 π³ 输出的归一化 waypoint 做统一放缩：

\[
W_t = S \cdot w_t
\]

得到的 `{W_t}` 就是带真实物理尺度的 waypoint 序列。

这时轨迹从“归一化几何空间”被映射到了“米制物理空间”，可以送给后续低层规划器。

---

## 4. 每一步建议保存的中间结果

为了便于调试，建议把每个阶段的关键中间量都保存下来。

### 输入阶段

保存：

- `V*`
- 抽帧后的图像序列 `{I_t}`

### π³ 阶段

保存：

- 每帧 pointmap `X_t`
- 每帧预测深度 `Z_t^{pred}`
- 归一化 waypoint `w_t`
- yaw（如果有）

### MoGe-2 阶段

保存：

- 每帧参考深度 `D_t^{ref}`

### 对齐阶段

保存：

- 每帧 valid mask `M_t`
- 每帧尺度比图 `s_t(u,v)`
- 所有有效尺度比拼接得到的一维数组
- 全局尺度因子 `S`

### 输出阶段

保存：

- 米制 waypoint `{W_t}`

---

## 5. 实现时的注意事项

### 5.1 两套深度图必须严格像素对齐

`Z_t^{pred}` 和 `D_t^{ref}` 的比值是逐像素计算的，因此两者必须：

- 对应同一帧
- 对应同一分辨率
- 对应同一像素坐标

如果中间做过 resize 或 crop，必须映射回同一坐标系。

---

### 5.2 必须确认 π³ 的深度轴定义

不能直接假设 pointmap 的第三维就是深度，必须确认：

- π³ 使用的相机坐标定义
- 哪个轴表示相机前向深度

否则 `Z_t^{pred}` 会提取错误。

---

### 5.3 不要对整张图直接求比值

必须使用 valid mask 过滤：

- 天空
- 极远区域
- 非法深度
- 噪声点

否则即使用中位数，也可能被大量无效区域干扰。

---

### 5.4 论文使用的是全局单一尺度因子

论文做法不是：

- 每帧一个尺度因子
- 每个 waypoint 一个尺度因子
- 分段尺度拟合

而是整个序列共享一个全局 `S`。

如果你要复现论文本身，就不要改成 frame-wise scale。

---

### 5.5 这一步只能恢复尺度，不能修正轨迹拓扑错误

尺度对齐只能解决：

- 走多远

不能解决：

- 往哪转错了
- 轨迹几何形状错了
- pointmap 本身结构出错了

也就是说，它只能把“方向大致正确但长度不准”的轨迹拉回米制空间，不能修复更高层的规划错误。

---

## 6. 伪代码

```python
# input: selected navigation video V_star

frames = sample_video_at_fixed_intervals(V_star)   # {I_t}

# Step 1: run pi3
pi3_outputs = run_pi3(frames)
pointmaps = pi3_outputs["pointmaps"]               # {X_t}
waypoints_norm = pi3_outputs["waypoints"]          # {w_t}

# Step 2: extract predicted depth from pointmaps
depth_pred = [extract_forward_depth(X_t) for X_t in pointmaps]   # {Z_t_pred}

# Step 3: run MoGe-2
depth_ref = run_moge2(frames)                      # {D_t_ref}

# Step 4: compute valid ratios
all_ratios = []
for t in range(len(frames)):
    valid_mask = (
        (depth_ref[t] > tau_min) &
        (depth_ref[t] < tau_max) &
        (depth_pred[t] > 0)
    )
    ratios_t = depth_ref[t][valid_mask] / depth_pred[t][valid_mask]
    all_ratios.append(ratios_t)

# Step 5: global scale estimation
all_ratios = concatenate(all_ratios)
S = median(all_ratios)

# Step 6: metric waypoint recovery
waypoints_metric = [S * w for w in waypoints_norm]   # {W_t}