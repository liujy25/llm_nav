# IAmGoodNavigator -- Codebase Research

## 0. Flowchart (Draw.io)

- Draw.io source file: `navigation_algorithm_flow.xml`
- Full path: `/home/liujy/mobile_manipulation/model_server/nav/navigation_algorithm_flow.xml`
- Open method: draw.io -> **File** -> **Open From** -> **Device** -> select this file.
- Flow scope: server startup + `/navigation_reset` + `/navigation_step` main algorithm, including:
  - Detic multi-class goal pool + active goal selection
  - Detection-blocklist (IoU>=0.7) skip logic
  - VLM secondary verification branch
  - `finished=True` direct approach branch
  - Normal NavAgent navigation branch / Habitat non-keyframe short path

## 1. Project Overview

This project implements a **VLM-guided frontier-based object navigation system** for mobile robots. The robot uses a Vision-Language Model (VLM) to choose navigation actions (move to frontier or turn) based on RGB images and a Bird's-Eye-View (BEV) obstacle map. The system follows a **client-server architecture**: the client runs on the robot (or in Habitat simulation), collects sensor data, and sends it to the server; the server runs VLM inference and returns goal poses.

The system has two deployment targets:
1. **Real robot** (ROS2-based): `llm_nav_client.py` + `nav_server.py`
2. **Habitat simulation**: `habitat_client.py` + `habitat_server.py`

---

## 2. Architecture

```
+------------------+       HTTP/REST        +-------------------+
|     Client       | <--------------------> |      Server       |
| (Robot/Habitat)  |   RGB, Depth, TF       |  (Flask + NavAgent)|
|                  |   <-- goal_pose         |                   |
+------------------+                        +-------------------+
       |                                           |
       v                                           v
  Nav2/PointNav                              NavAgent
  (path execution)                         /     |     \
                                     ObstacleMap  VLM   DeticDetector
                                     (BEV Map)  (GPT/Qwen) (Goal Detection)
```

### Core Components

| File | Role |
|------|------|
| `nav_agent.py` (~2100 lines) | Core navigation agent: frontier detection, action proposal, VLM prompting, BEV visualization |
| `obstacle_map.py` (~1100 lines) | BEV obstacle map: depth-to-obstacle projection, explored area tracking, frontier detection |
| `nav_server.py` | Flask server for real robot deployment |
| `habitat_server.py` | Flask server for Habitat simulation deployment |
| `llm_nav_client.py` | ROS2 client for real robot |
| `habitat_client.py` | Habitat simulation client with PointNav policy |
| `vlm.py` | OpenAI-compatible VLM wrapper (supports vLLM, OpenAI API) |
| `detic_detector.py` | Detic object detector for goal recognition |
| `prompts.json` | VLM prompt templates |
| `pointnav_controller.py` | Pre-trained PointNav policy for low-level navigation in Habitat |
| `habitat_pathplanner.py` | A* path planner (ApexNav style) |
| `habitat_nav_agent.py` | Standalone Habitat navigation agent (obstacle map + A* planning) |
| `habitat_agent.py` | Simplified Habitat agent (similar to habitat_nav_agent) |
| `geometry_utils.py` | Point cloud, coordinate transform utilities |
| `utils.py` | Image encoding, projection, drawing utilities |
| `transform_utils.py` | Transformation matrix utilities (pose_inv, etc.) |

---

## 3. Navigation Pipeline (per iteration)

### 3.1 Server-side (`nav_server.py` / `habitat_server.py`)

1. **`/navigation_reset` (POST)**: Initialize session
   - Create `NavAgent` with config (VLM model, BEV map params)
   - Initialize `DeticDetector` for goal object detection
   - Set goal text and description

2. **`/navigation_step` (POST)**: Per-step decision
   - Receive: RGB image, depth image, camera intrinsic (3x3), T_cam_odom (4x4), T_odom_base (4x4)
   - **Step 1: Detic Detection** -- Check if goal object is visible in RGB
     - If detected: compute 3D position via mask + depth unprojection, calculate approach pose (0.8m from object), return `finished=True`
   - **Step 2: VLM Navigation** (if no detection)
     - Call `NavAgent._nav()` which:
       1. Updates obstacle map from depth observation
       2. Detects frontiers (explored/unexplored boundaries)
       3. Proposes actions (frontiers + turn actions)
       4. Projects actions onto RGB image (numbered circles)
       5. Generates BEV map with annotated waypoints
       6. Calls VLM with RGB + BEV images + prompt
       7. Parses VLM response to get action number
     - Convert action to goal pose in odom frame
   - Return: `action_type` (nav_step/turn_left/turn_right/none), `goal_pose`, `finished`, `timing`

### 3.2 Client-side

**Real Robot (`llm_nav_client.py`)**:
- ROS2 node subscribing to RGB, depth, TF
- Sends observations to server via HTTP
- Executes returned goal pose via Nav2 `NavigateToPose` action
- Runs a stop detection thread (2Hz) checking if goal is reached

**Habitat (`habitat_client.py`)**:
- Runs in Habitat simulator
- Uses `PointNavController` (pre-trained RL policy) or teleport for low-level navigation
- Sends server observations in same format as real robot
- Supports evaluation metrics (SPL, success rate, etc.)

---

## 4. NavAgent Deep Dive (`nav_agent.py`)

### 4.1 Configuration Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_bev_map` | True | Use BEV obstacle map |
| `bev_map_size` | 2000 | Map size in pixels |
| `bev_pixels_per_meter` | 100 | Map resolution |
| `clip_dist` | 5.0 | Maximum action distance |
| `turn_angle_deg` | 30.0 | Turn angle per action |
| `vlm_history` | 3 | Conversation rounds to keep |
| `vlm_timeout` | 10 | VLM request timeout (seconds) |
| `obstacle_height_min` | 0.15 | Min obstacle height (m) |
| `obstacle_height_max` | 2.0 | Max obstacle height (m) |
| `fallback_turn_threshold` | 12 | Consecutive turns before fallback |

### 4.2 State Management

- **`trajectory_history`**: List of (x, y) odom positions for past keyframes
- **`trajectory_yaw_history`**: Corresponding yaw angles
- **`current_trajectory_index`**: 1-based index of current trajectory point
- **`action_history`**: Last N actions (for rotation strategy)
- **`consecutive_turn_count`**: Consecutive turns without move (triggers fallback)
- **`simplified_history`**: Compact history for VLM context (TURN records with merged rotations, MOVE records with from/to trajectory points)
- **`last_turn_direction`**: Restricts to same turn direction once started turning

### 4.3 Frontier-based Navigation

1. **`_navigability_from_bev()`**: Gets frontiers from ObstacleMap, converts to (r, theta, normal) in base frame
   - Filters: too close (<0.5m), occluded by depth
2. **`_action_proposer()`**: Selects representative waypoints from frontier candidates
   - Prioritizes frontier (unexplored) waypoints over explored ones
   - Enforces minimum angular separation between waypoints
3. **`_projection()`**: Projects waypoints onto RGB image as numbered circles with arrows
4. **`_generate_bev_with_waypoints()`**: Draws annotated BEV map with:
   - White: explored navigable space
   - Gray: unexplored space
   - Black: obstacles/walls
   - Light green circles: frontier waypoints
   - Light blue circles: trajectory history points
   - Red circle: current robot position
   - Red arrow: facing direction
   - Red boundary rays: camera FOV

### 4.4 Fallback Mode

When `consecutive_turn_count >= fallback_turn_threshold` (default 12):
- Switch to "fallback mode": present historical trajectory points as candidates
- VLM sees only BEV map (no RGB) and must choose a trajectory point ID to backtrack to
- This breaks local spinning loops

### 4.5 VLM Interaction

Messages structure:
1. **System instruction**: "You are an embodied robotic navigation assistant..."
2. **Initial prompt**: Task description, action definitions, observation format, output format
3. **Custom history**: Simplified action records (turn/move summaries)
4. **Current prompt**: Iteration number, available actions, format instructions

VLM output format:
```
**Action and Reason:** {Chosen action and reason}
**Decision:** {'action': <number>}
```

---

## 5. ObstacleMap Deep Dive (`obstacle_map.py`)

### 5.1 Map Representation

- 3 binary maps of size `(size x size)`:
  - `_obstacle_map`: True where obstacles detected
  - `_navigable_map`: Inverse of dilated obstacle map (accounts for robot radius)
  - `explored_area`: True where area has been observed

### 5.2 Update Pipeline (`update_map()`)

1. Generate point cloud from depth in camera frame
2. Transform to odom frame
3. Filter by height range [min_height, max_height] -> obstacle cloud
4. Create FOV mask (ellipse sector)
5. **Incremental** obstacle update: only ADD new obstacles, never remove
6. Project obstacles to 2D pixels, mark in obstacle map
7. Dilate obstacles by robot radius -> navigable map
8. Update explored area:
   a. Mark depth-visible areas
   b. Raycast (fog-of-war) to fill blind spots
   c. Union, restrict to navigable areas
   d. Keep only connected component around robot

### 5.3 Frontier Detection (`_update_frontiers()`)

1. Global frontier detection (from `frontier_exploration` library)
2. FOV filtering (expanded FOV angle)
3. Obstacle proximity filtering
4. **Reachability filtering** (A* pathfinding via `route_through_array`)
5. Move frontiers inward into explored area (0.25m)
6. Compute normals (gradient of explored area)

### 5.4 Coordinate System

- **Odom frame**: X=forward, Y=left, Z=up
- **Pixel frame**: Origin at map center, X=right, Y=down (Y-axis flipped)
- `_xy_to_px()`: odom -> pixel, `_px_to_xy()`: pixel -> odom
- Episode origin set on first observation

---

## 6. VLM Wrapper (`vlm.py`)

### OpenAIVLM

- Wraps OpenAI-compatible API (supports both OpenAI official and local vLLM/Qwen)
- Key methods:
  - `call_chat()`: Standard conversation with rolling history window
  - `call_chat_with_custom_history()`: NavAgent controls the history directly
  - `reset()`: Clear history, set initial prompt
- Conversation structure: system -> initial_prompt -> history -> current_message
- Auto-detects OpenAI official API (no `extra_body` params)
- Parameters: temperature=0.7, top_p=0.8, presence_penalty=1.5, max_tokens=32768

---

## 7. Detic Detector (`detic_detector.py`)

- Wraps Detic (Detectron2-based) open-vocabulary object detector
- `set_goal()`: Sets target object class using CLIP text embeddings
- `detect()`: Runs detection, returns (bbox, confidence, mask) sorted by confidence
- `get_3d_position()`: Mask + depth unprojection -> 3D position in base frame
  - Uses near-surface depth filtering (p30 + 0.25m window) to reduce background leakage

---

## 8. Habitat Integration

### habitat_client.py (HabitatLLMNavClient)

- Main evaluation loop for Habitat ObjectNav benchmark
- Creates Habitat environment from config
- Per-iteration: get observation -> send to server -> receive goal pose -> navigate via PointNav
- Supports:
  - PointNav policy (pre-trained RL) for low-level navigation
  - Teleport mode for debugging
  - Video recording, data saving, metric evaluation

### habitat_server.py

- Same as nav_server.py but configured for Habitat
- Uses local Qwen VLM by default (vs. OpenAI for real robot)
- Includes `ApexNavPyPlanner` for A* path planning (used by HabitatNavAgent)

### pointnav_controller.py

- Loads Habitat-baselines PointNav ResNet policy
- Computes (rho, theta) polar coordinates to goal
- Outputs discrete actions: STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
- Supports goal orientation alignment after reaching position

### habitat_pathplanner.py (ApexNavPyPlanner)

- A* search with 12-directional movement (30-degree steps)
- Continuous world-space collision checking
- `select_local_target()`: Select intermediate waypoint on path
- `decide_next_action()`: Convert local target to discrete action

---

## 9. Prompt Templates (`prompts.json`)

4 prompt templates:
1. **`initial_prompt_template`**: With BEV map -- describes both RGB and BEV observations
2. **`initial_prompt_template_no_bev`**: RGB only
3. **`iteration_prompt_template`**: Per-step prompt (with BEV)
4. **`iteration_prompt_template_no_bev`**: Per-step prompt (RGB only)
5. **`iteration_prompt_template_fallback_bev`**: Fallback mode (BEV only, choose trajectory point)

Key design: Actions are numbered integers. MOVE actions are 1..N (frontier indices). TURN LEFT is -1, TURN RIGHT is -2.

---

## 10. Key Design Decisions

1. **Frontier-based exploration**: Instead of pure VLM reasoning about where to go, the system pre-computes frontier waypoints and lets the VLM choose among them. This constrains VLM output to valid actions.

2. **Dual detection**: Detic detector runs first for direct goal recognition; VLM handles exploration when goal is not visible.

3. **Incremental obstacle map**: Obstacles are only added, never removed. This prevents "phantom passageways" from noisy depth readings.

4. **Reachability filtering**: Frontiers are verified reachable via A* before presenting to VLM.

5. **Fallback mechanism**: After too many consecutive turns (spinning), the agent backtracks to a historical trajectory point.

6. **Turn direction locking**: Once the agent starts turning in a direction, it must continue that direction until a MOVE action resets it. This prevents left-right oscillation.

7. **Simplified history**: VLM context uses compact action summaries ("turned left 60 degrees", "moved from point 3 to point 5") rather than full conversation history. This saves token budget.

8. **Client-server separation**: Navigation intelligence runs on a GPU server; the robot only handles sensor collection and path execution. This allows using large VLMs without on-robot GPU requirements.

---

## 11. Data Flow Summary

```
Client                                    Server
  |                                         |
  | --- /navigation_reset (goal) ---------> |
  |                                         | Create NavAgent, DeticDetector
  | <---- {status: ok} ------------------- |
  |                                         |
  | --- /navigation_step (rgb,depth,tf) --> |
  |                                         | 1. Detic: check if goal visible
  |                                         | 2. If not: NavAgent._nav()
  |                                         |    a. Update obstacle map
  |                                         |    b. Detect frontiers
  |                                         |    c. Propose actions
  |                                         |    d. Project onto RGB
  |                                         |    e. Generate BEV map
  |                                         |    f. Call VLM
  |                                         |    g. Parse response
  |                                         | 3. Convert to goal pose
  | <---- {action_type, goal_pose} -------- |
  |                                         |
  | Execute via Nav2/PointNav               |
  | (repeat until finished=True)            |
```

---

## 12. Supported VLM Backends

- **OpenAI API**: GPT-5.2 (used for real robot nav_server.py)
- **Local vLLM**: Qwen3.5-27B-GPTQ-Int4 (used for Habitat habitat_server.py)
- Both accessed via OpenAI-compatible API, auto-detected by checking `base_url`

---

## 13. frontier_exploration Submodule

Located in `frontier_exploration/`, provides:
- `frontier_detection.py`: `detect_frontier_waypoints()` -- finds boundary between explored and unexplored regions
- `fog_of_war.py`: `reveal_fog_of_war()` -- raycast-based exploration marking
- `utils/`: Bresenham line, path utilities, inflection sensor
- Based on Habitat ObjectNav frontier exploration research

---

## 14. File Size and Complexity

| File | Lines | Complexity |
|------|-------|------------|
| nav_agent.py | ~2100 | High -- core navigation logic, VLM integration, visualization |
| obstacle_map.py | ~1100 | High -- depth processing, frontier detection, map management |
| habitat_client.py | ~1600 | Medium -- Habitat environment setup, evaluation loop |
| habitat_server.py | ~1200 | Medium -- same as nav_server + Habitat-specific configs |
| nav_server.py | ~890 | Medium -- Flask endpoints, pose conversion |
| llm_nav_client.py | ~690 | Medium -- ROS2 integration, Nav2 action client |
| pointnav_controller.py | ~540 | Medium -- RL policy loading, action selection |
| habitat_pathplanner.py | ~300 | Medium -- A* search, action planning |
| vlm.py | ~390 | Low -- API wrapper |
| detic_detector.py | ~230 | Low -- detection wrapper |
| habitat_nav_agent.py | ~475 | Low -- standalone agent combining obstacle_map + planner |
| geometry_utils.py | ~170 | Low -- utility functions |
| utils.py | ~265 | Low -- utility functions |
