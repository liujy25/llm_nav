# Project: LLM_nav

## 项目简介

基于 VLM (Vision-Language Model) 的移动机器人目标导航系统。机器人在未知环境中，给定语义目标（如 "chair"、"door"），通过 RGB-D 感知、BEV 障碍物地图、Frontier 探索和 VLM 决策，自主导航到目标物体附近。支持 Habitat 仿真环境评测和真实机器人部署。项目中的代码需要运行在model_server这个conda环境下。

## 系统架构

采用 Client-Server 架构，通过 Flask HTTP API 通信：

- **Server 端** (`habitat_server.py` / `nav_server.py`): 接收 RGB-D 观测，运行 NavAgent 推理，返回导航动作（目标位姿或转向指令）
- **Client 端** (`habitat_client.py` / `habitat_client_batch.py`): 在 Habitat 仿真器中运行环境，采集观测发送给 Server，执行返回的动作。Client 使用 PointNav 策略 (`pointnav_controller.py`) 执行局部路径跟踪
- **NavAgent** (`nav_agent.py`): 核心决策模块，整合感知、地图、VLM 推理
- **ObstacleMap** (`obstacle_map.py`): BEV 障碍物/已探索区域地图，基于 VLFM 改写

## 核心算法流程

每个导航 step 的处理流程：

1. **地图更新** (`update_observation_only`): 将 RGB-D 深度图反投影为 3D 点云，按高度过滤障碍物，投影到 BEV 栅格地图；通过 fog-of-war raycast 更新已探索区域
2. **Frontier 检测** (`_navigability_from_bev`): 在 BEV 地图上检测已探索与未探索区域的边界（frontier），作为候选导航目标点；对 frontier 做遮挡过滤（depth-based occlusion check）
3. **动作提议** (`_action_proposer`): 将 frontier 点按角度聚类、去重，生成稀疏的候选 MOVE 动作集合；同时提供 TURN LEFT/RIGHT 动作
4. **可视化标注** (`_projection` + `_generate_bev_with_waypoints`): 将候选动作投影到 RGB 图像上标注编号圆圈和箭头；生成带 waypoint 标注的 BEV 俯视图
5. **VLM 决策** (`_nav`): 将标注后的 RGB + BEV 图像和结构化 prompt 发送给 VLM（Qwen3-VL / GPT），VLM 输出场景理解、空间分析、动作比较，最终选择一个动作
6. **动作解析与执行**: 解析 VLM 返回的 `{'action': N}`，MOVE 动作转换为 odom 坐标系下的目标位姿返回给 Client；TURN 动作返回转向角度

**Fallback 机制**: 连续转向超过阈值时，切换到全局 frontier 选择模式，强制选择一个远处的 frontier 打破局部循环。

**Detic 目标检测** (`detic_detector.py`): 使用 Detic 开放词汇检测器在 RGB 中检测目标物体，检测到后通过 VLM 二次验证，确认后触发导航完成。

## 核心文件说明

| 文件 | 职责 |
|------|------|
| `nav_agent.py` | 核心决策 Agent：地图更新、frontier 计算、动作提议、VLM 调用、动作解析 |
| `obstacle_map.py` | BEV 栅格地图：障碍物检测、已探索区域维护、frontier 检测、fog-of-war |
| `habitat_server.py` | Habitat 评测用 Flask Server，处理 `/navigation_reset` 和 `/navigation_step` |
| `nav_server.py` | 真实机器人用 Flask Server，接口与 habitat_server 类似 |
| `habitat_client.py` | Habitat 仿真 Client，单 episode 运行 |
| `habitat_client_batch.py` | Habitat 批量评测 Client |
| `vlm.py` | VLM 封装（OpenAI API 兼容），支持多轮对话和自定义历史 |
| `detic_detector.py` | Detic 开放词汇目标检测器封装 |
| `pointnav_controller.py` | 预训练 PointNav 策略，用于局部路径跟踪 |
| `prompts.json` | VLM 的 system instruction、initial prompt、iteration prompt 模板 |
| `geometry_utils.py` | 点云生成、坐标变换、FOV 过滤等几何工具 |
| `utils.py` | 图像编码、坐标投影、深度转高度等通用工具 |
| `frontier_exploration/` | Frontier 检测算法（基于 VLFM），包含 frontier 提取、fog-of-war、bresenham 等 |

## Communication and Coding Guidelines

### Communication Rules

1. **No Emojis or Symbols**: Do not use emojis, memes, or emotional/status symbols (like ✓, ✅, 🎉, etc.) in communication
2. **Ask When Uncertain**: If not confident about the current task, ask follow-up questions or request relevant documentation, code, or examples
3. **Discuss Before Executing**: ALWAYS discuss the modification plan with the user before making any code changes
4. **Think Before Responding**: MUST use `<think>...</think>` tags before every response to analyze the problem thoroughly (up to 24576 tokens)
5. **Use Chinese**

### Coding Rules

1. **No Unnecessary Symbols**: Do not add checkmarks, emojis, or decorative symbols in code or comments
2. **Clean Code**: Keep code minimal and focused on functionality
