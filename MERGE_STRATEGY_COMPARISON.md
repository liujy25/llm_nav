# 融合策略对比分析

## 方案对比

### 方案1：将主分支raycast搬到当前分支（原计划）
**操作**：主分支 → 当前分支（waypoint）

**需要做的事**：
1. 复制5个raycast相关方法（`_get_navigability_mask`, `_get_radial_distance`, `_navigability`, `_action_proposer`, `_projection`）
2. 修改`_nav()`方法，替换frontier检测逻辑
3. 调整prompt，从frontier改为navigable points
4. 调整Function Call处理逻辑
5. 保留ObstacleMap基础设施（但不用frontier功能）

**优点**：
- 保留当前分支的Function Call框架
- 保留waypoint注册机制
- 保留BEV地图可视化

**缺点**：
- 需要修改较多代码
- ObstacleMap的frontier功能变成冗余
- 两套系统混合，复杂度高

---

### 方案2：将当前分支BEV标注搬到主分支（新方案）✅ 推荐
**操作**：当前分支（waypoint） → 主分支（main）

**需要做的事**：
1. 复制`ObstacleMap`类到主分支
2. 复制`_generate_bev_with_waypoints()`方法
3. 在主分支的`_nav()`中添加BEV地图生成
4. 修改prompt，添加BEV地图说明
5. 可选：添加waypoint注册机制

**优点**：
- ✅ **更简单**：主分支的raycast逻辑已经成熟稳定，只需添加BEV可视化
- ✅ **功能增强**：给主分支增加全局视野（BEV地图）
- ✅ **保持主分支简洁**：不引入复杂的Function Call框架
- ✅ **易于维护**：主分支代码量更少，逻辑更清晰
- ✅ **向后兼容**：不破坏主分支现有功能

**缺点**：
- 需要引入ObstacleMap依赖
- 增加一些计算开销（维护地图）

---

## 详细对比

| 维度 | 方案1（raycast→waypoint） | 方案2（BEV→main）✅ |
|------|--------------------------|-------------------|
| **代码修改量** | 大（5个方法+逻辑重构） | 小（1个类+1个方法） |
| **复杂度** | 高（两套系统混合） | 低（功能叠加） |
| **风险** | 高（可能破坏FC框架） | 低（只是增强可视化） |
| **维护性** | 差（代码冗余） | 好（职责清晰） |
| **功能完整性** | 失去全局视野 | 保留全局视野 |
| **实施时间** | 2-3小时 | 1小时 |

---

## 推荐方案：方案2（BEV→main）

### 核心思路

**主分支现状**：
- 使用raycast生成可行点（RGB图像上标注）
- 只能看到当前FOV内的可行点
- 没有全局地图

**改进目标**：
- 保留raycast可行点检测（RGB标注）
- 添加BEV地图显示全局探索状态
- VLM同时看到：RGB图像（当前视野）+ BEV地图（全局视野）

### 实施步骤

#### 步骤1：复制ObstacleMap到主分支
```bash
git checkout main
cp obstacle_map.py obstacle_map.py  # 从waypoint分支复制
```

#### 步骤2：修改主分支的`__init__`
```python
from obstacle_map import ObstacleMap

def __init__(self, cfg=None):
    # ... 现有代码 ...
    
    # 添加ObstacleMap
    self.obstacle_map = ObstacleMap(
        min_height=self.cfg['obstacle_height_min'],
        max_height=self.cfg['obstacle_height_max'],
        agent_radius=0.3,
        area_thresh=0.3,
        size=200,  # 20m x 20m
        pixels_per_meter=10,
    )
    self.scale = 10
```

#### 步骤3：复制BEV生成方法
从waypoint分支复制`_generate_bev_with_waypoints()`到主分支

#### 步骤4：修改`_nav()`方法
```python
def _nav(self, obs: dict, goal: str, iter: int, goal_description: str = ""):
    # 1. 更新ObstacleMap
    self.obstacle_map.update(obs)
    
    # 2. 生成可行点（原有逻辑）
    a_initial = self._navigability(obs)
    a_final = self._action_proposer(a_initial, obs['base_to_odom_matrix'])
    projected, annotated_rgb = self._projection(a_final, obs)
    
    # 3. 生成BEV地图（新增）
    current_pos = obs['base_to_odom_matrix'][:2, 3]
    bev_map = self._generate_bev_with_waypoints(current_pos)
    
    # 4. 发送给VLM（两张图）
    response = self.actionVLM.chat(
        prompt=prompt,
        images=[annotated_rgb, bev_map]  # RGB + BEV
    )
```

#### 步骤5：更新prompt
```python
prompt = f"""
You have two views:
1. RGB Image: Shows current camera view with navigable waypoints (numbered circles)
2. BEV Map: Shows explored area (gray) and unexplored area (green)
   - Red triangle: your current position and heading
   - Blue circles: historical waypoints

Choose a waypoint number to navigate to, or choose -1 (turn left) / -2 (turn right).
"""
```

---

## 预期效果

### 融合后的主分支功能

**输入**：
- RGB图像：标注了可行点（1, 2, 3, ...）和转向按钮（-1, -2）
- BEV地图：显示全局探索状态、当前位置、历史轨迹

**VLM决策**：
- 看RGB图像：选择当前可达的waypoint
- 看BEV地图：了解全局探索进度，避免重复探索
- 结合两者：做出更智能的导航决策

**优势**：
1. **局部精确**：raycast提供当前FOV内的精确可行点
2. **全局感知**：BEV地图提供全局探索状态
3. **简单高效**：不引入复杂的Function Call框架
4. **易于调试**：两个视图独立，便于分析问题

---

## 实施建议

### 立即执行：方案2

**理由**：
1. 更简单、风险更低
2. 功能更完整（局部+全局）
3. 保持主分支的简洁性
4. 易于后续维护

**预计工作量**：1小时

**步骤**：
1. 切换到main分支
2. 从waypoint分支复制`obstacle_map.py`
3. 复制`_generate_bev_with_waypoints()`方法
4. 修改`__init__`和`_nav()`
5. 更新prompt
6. 测试验证

---

## 后续优化方向

融合完成后，可以考虑：

1. **动态视图切换**：根据情况选择显示frontier或waypoint
2. **多尺度BEV**：提供不同缩放级别的地图
3. **语义标注**：在BEV上标注房间、门等语义信息
4. **轨迹预测**：在BEV上显示规划路径

---

## 结论

**推荐方案2**：将当前分支的BEV标注功能搬到主分支

这是一个**增量式改进**，而不是**重构式替换**，因此：
- 风险更低
- 实施更快
- 效果更好
- 维护更容易
