import argparse
import json
import os
import sys
import math
from io import BytesIO
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
import cv2
import requests
from PIL import Image
from isaacsim import SimulationApp

# Import evaluation functions from demo.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------
# 1. EVALUATION FUNCTIONS (from demo.py)
# ---------------------------------------------------------

def calc_ndtw(pred_traj, gt_traj, threshold=3.0):
    if len(pred_traj) == 0 or len(gt_traj) == 0:
        return 0.0
    pred_xy = np.array(pred_traj)
    gt_xy = np.array(gt_traj)
    N, M = len(pred_xy), len(gt_xy)
    dtw = np.full((N + 1, M + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            dist = np.linalg.norm(pred_xy[i - 1] - gt_xy[j - 1])
            dtw[i, j] = dist + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    dtw_distance = dtw[N, M]
    ndtw = np.exp(-dtw_distance / (threshold * len(gt_xy)))
    return float(ndtw)

def calculate_trajectory_length(coords):
    if len(coords) < 2:
        return 0.0
    total_length = 0.0
    for i in range(1, len(coords)):
        dist = math.sqrt((coords[i][0] - coords[i-1][0])**2 +
                        (coords[i][1] - coords[i-1][1])**2)
        total_length += dist
    return total_length

def evaluate_single_episode(csv_file, episode_data):
    if not os.path.exists(csv_file):
        print("Error: CSV file not found.")
        return None, None

    df = pd.read_csv(csv_file)
    coords = df[['pos_x', 'pos_y']].values.tolist()

    filtered_coords = [coords[0]]
    for point in coords[1:]:
        if point != filtered_coords[-1]:
            filtered_coords.append(point)

    goal = episode_data['goals']['position']
    goal_x, goal_y = goal[0], goal[1]

    reference_path = episode_data['reference_path']
    gt_coords = [[pos[0], pos[1]] for pos in reference_path]

    last_x, last_y = filtered_coords[-1]
    dist_last = math.sqrt((goal_x - last_x)**2 + (goal_y - last_y)**2)
    sr = 1 if dist_last <= 3 else 0

    osr = 0
    for idx, (x, y) in enumerate(filtered_coords):
        if math.sqrt((goal_x - x)**2 + (goal_y - y)**2) <= 3:
            osr = 1
            break

    ndtw_score = calc_ndtw(filtered_coords, gt_coords, threshold=3.0)
    tl = calculate_trajectory_length(filtered_coords)
    tl_gt = calculate_trajectory_length(gt_coords)

    if tl_gt > 0 and tl > 0:
        spl = sr * (tl_gt / max(tl, tl_gt))
    else:
        spl = 0.0

    result_str = (
        f"--- Episode Evaluation ---\n"
        f"SR: {sr}, OSR: {osr}, SPL: {spl:.3f}, nDTW: {ndtw_score:.3f}, TL: {tl:.2f}m"
    )
    print(f"\n{result_str}\n")

    metrics = {
        "SR": sr,
        "OSR": osr,
        "SPL": spl,
        "nDTW": ndtw_score,
        "TL": tl,
        "Goal Dist": dist_last
    }
    return result_str, metrics


# ---------------------------------------------------------
# 2. ISAAC SIM SETUP
# ---------------------------------------------------------

config = {
    "launch_config": {
        "renderer": "RayTracedLighting",
        "headless": False,
    },
    "resolution": [512, 512],
    "writer": "BasicWriter",
}

parser = argparse.ArgumentParser()
parser.add_argument("--index", default=0, help="0-56")
parser.add_argument("--task", default="fine", help="fine | coarse")
parser.add_argument("--headless", default=False, action="store_true")
parser.add_argument("--render", default=False, action="store_true")
parser.add_argument("--record", default=True, action="store_true")
parser.add_argument("--camera_height", default=1.5, type=float, help="Camera height from ground")
parser.add_argument("--server", type=str, default='http://10.19.126.158:1874', help='LLM Nav Server URL')
parser.add_argument("--max_iterations", type=int, default=50, help='Maximum navigation iterations')
args, unknown_args = parser.parse_known_args()

config["launch_config"]["headless"] = args.headless
simulation_app = SimulationApp(config["launch_config"])


def rotation_from_direction(direction, up_vector=np.array([0, 0, 1])):
    from scipy.spatial.transform import Rotation as R
    direction = np.array(direction, dtype=np.float64)
    forward = direction / np.linalg.norm(direction)
    right = np.cross(up_vector, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm
    up = np.cross(forward, right)
    rot_mat = np.column_stack((forward, right, up))
    quat = R.from_quat(rot_mat).as_quat()
    return np.array([quat[3], quat[0], quat[1], quat[2]])


# ---------------------------------------------------------
# 3. LLM NAVIGATION CONTROLLER
# ---------------------------------------------------------

class LLMNavigationController:
    def __init__(self, world, scene_data, server_url, goal_text):
        from isaacsim.sensors.camera import Camera
        from scipy.spatial.transform import Rotation as R

        self.world = world
        self.server_url = server_url.rstrip('/')
        self.goal_text = goal_text

        self.camera_height = args.camera_height
        self.resolution = 512
        self.camera_path = "/World/FloatingCamera"

        # Fixed camera parameters (from manual testing in Isaac Sim)
        self.camera_position = np.array([2.3, -1.6, 1.5], dtype=np.float32)
        # Camera looks at +Y direction with 20° downward tilt
        # Note: Euler pitch=20° gives 20° below horizontal in ROS convention
        self.camera_rotation_xyz = np.array([0.0, 20.0, 90.0])

        # Convert camera rotation (XYZ Euler angles) to quaternion using official Isaac Sim utils
        import isaacsim.core.utils.numpy.rotations as rot_utils
        camera_orientation = rot_utils.euler_angles_to_quats(
            self.camera_rotation_xyz,
            degrees=True,
            extrinsic=True  # Use extrinsic rotation (default)
        )

        # Focal length in mm
        self.focal_length_mm = 18.0

        # Create Camera object (will be initialized after world.reset())
        self.camera = Camera(
            prim_path=self.camera_path,
            resolution=(self.resolution, self.resolution),
            position=self.camera_position,
            orientation=camera_orientation
        )

        # Navigation state
        # Start position: on ground, directly below initial camera position
        self.start_position = self.camera_position.copy()
        self.start_position[2] = 0.0  # Base is on ground

        # Initial look_direction will be calculated from camera's actual pose after initialization
        self.current_position = self.start_position.copy()
        self.look_direction = np.array([1.0, 0.0, 0.0])  # Temporary, will be updated in reset()

        print(f"Initial robot base position: {self.start_position}")

        self.move_speed = 0.25
        self.turn_speed = 15.0

        self.mission_complete = False
        self.iteration_count = 0

        # Camera intrinsics (will be set after initialization)
        self.intrinsic = None

        print(f"Camera created at position: {self.camera_position}")
        print(f"Camera rotation (XYZ degrees): {self.camera_rotation_xyz}")
        print(f"Resolution: {self.resolution}x{self.resolution}")
        print("Note: Camera will be initialized after world.reset()")

    def initialize_camera(self):
        """Initialize camera after world.reset() - must be called explicitly"""
        # Initialize the camera (creates render product)
        self.camera.initialize()

        self.camera.set_focal_length(self.focal_length_mm / 10.0)
        self.camera.set_clipping_range(0.01, 1000.0)

        # Add distance_to_image_plane annotator once during initialization
        self.camera.add_distance_to_image_plane_to_frame()

        # Get intrinsics from Isaac Sim Camera
        intrinsic_tensor = self.camera.get_intrinsics_matrix()

        # Convert to numpy array
        if hasattr(intrinsic_tensor, 'cpu'):
            self.intrinsic = intrinsic_tensor.cpu().numpy().astype(np.float32)
        elif hasattr(intrinsic_tensor, 'numpy'):
            self.intrinsic = np.array(intrinsic_tensor, dtype=np.float32)
        else:
            self.intrinsic = np.array(intrinsic_tensor, dtype=np.float32)

        print(f"\nCamera initialized!")
        print(f"Focal length: {self.focal_length_mm}mm")
        print(f"Intrinsic matrix from Isaac Sim Camera:")
        print(self.intrinsic)

    def reset(self):
        from scipy.spatial.transform import Rotation as R

        self.current_position = self.start_position.copy()

        # Get actual camera pose from Isaac Sim
        camera_pos, camera_quat = self.camera.get_world_pose()  # quat is [w, x, y, z]

        # Convert quaternion to rotation matrix
        quat_xyzw = [camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]]  # [w,x,y,z] -> [x,y,z,w]
        R_cam = R.from_quat(quat_xyzw)

        # Camera looks at -Z direction in its own frame
        camera_forward = R_cam.apply([0, 0, -1])

        # Project to ground (XY plane) and normalize
        camera_forward_ground = np.array([camera_forward[0], camera_forward[1], 0.0])
        if np.linalg.norm(camera_forward_ground) > 1e-6:
            self.look_direction = camera_forward_ground / np.linalg.norm(camera_forward_ground)
        else:
            # If camera is looking straight down, default to +X
            self.look_direction = np.array([1.0, 0.0, 0.0])

        self.mission_complete = False
        self.iteration_count = 0

        # Update camera to follow robot at initial position
        self._update_camera_follow_robot()

        print('=' * 10, "reset", "=" * 10)
        print(f"Robot base position: {self.current_position}")
        print(f"Robot look direction: {self.look_direction}")

    def _update_camera_follow_robot(self):
        """Update camera position and orientation to follow robot"""
        import isaacsim.core.utils.numpy.rotations as rot_utils

        # Camera position: above robot base at camera_height
        camera_pos = self.current_position.copy()
        camera_pos[2] = self.camera_height

        # Camera orientation using XYZ Euler angles: [0, pitch, yaw]
        # pitch=20 gives 20° below horizontal in ROS convention
        yaw = math.atan2(self.look_direction[1], self.look_direction[0])
        yaw_deg = math.degrees(yaw)
        camera_rotation_xyz = np.array([0.0, 20.0, yaw_deg])

        # Convert to quaternion [w, x, y, z]
        camera_orientation = rot_utils.euler_angles_to_quats(
            camera_rotation_xyz,
            degrees=True,
            extrinsic=True
        )

        # Use Camera API to set pose
        self.camera.set_world_pose(
            position=camera_pos,
            orientation=camera_orientation
        )

    def get_camera_transform(self):
        return {
            "pos_x": self.current_position[0],
            "pos_y": self.current_position[1],
            "pos_z": self.current_position[2],
            "look_x": self.look_direction[0],
            "look_y": self.look_direction[1],
            "look_z": self.look_direction[2],
        }

    def get_transform_matrices(self):
        """Construct transformation matrices for server request"""
        from scipy.spatial.transform import Rotation as R

        # Base pose in odom frame
        # Base position is current_position (on ground, z=0)
        base_position = self.current_position.copy()
        base_position[2] = 0.0  # Ensure on ground

        # Base orientation from look_direction
        # Base X-axis points in look_direction (projected to ground)
        yaw = math.atan2(self.look_direction[1], self.look_direction[0])
        base_quat = R.from_euler('z', yaw).as_quat()  # [x, y, z, w]
        base_rot = R.from_quat(base_quat).as_matrix()

        T_odom_base = np.eye(4, dtype=np.float32)
        T_odom_base[:3, :3] = base_rot.astype(np.float32)
        T_odom_base[:3, 3] = base_position.astype(np.float32)

        # Camera pose in odom frame - get directly from Camera API
        # Isaac Sim uses ROS convention: X=forward, Y=left, Z=up
        camera_pos, camera_quat = self.camera.get_world_pose()  # quat is [w, x, y, z]

        # Convert quaternion to rotation matrix
        quat_xyzw = [camera_quat[1], camera_quat[2], camera_quat[3], camera_quat[0]]  # [w,x,y,z] -> [x,y,z,w]
        R_cam = R.from_quat(quat_xyzw)

        T_odom_cam_ros = np.eye(4, dtype=np.float32)
        T_odom_cam_ros[:3, :3] = R_cam.as_matrix().astype(np.float32)
        T_odom_cam_ros[:3, 3] = camera_pos.astype(np.float32)

        # Convert from ROS convention to optical convention
        # ROS: X=forward, Y=left, Z=up
        # Optical: X=right, Y=down, Z=forward
        # Transformation: X_opt = -Y_ros, Y_opt = -Z_ros, Z_opt = X_ros

        # T_optical_to_ros: transforms points from optical frame to ROS frame
        T_optical_to_ros = np.array([
            [ 0,  0,  1, 0],
            [-1,  0,  0, 0],
            [ 0, -1,  0, 0],
            [ 0,  0,  0, 1]
        ], dtype=np.float32)

        # T_ros_to_optical: transforms points from ROS frame to optical frame
        T_ros_to_optical = np.array([
            [ 0, -1,  0, 0],
            [ 0,  0, -1, 0],
            [ 1,  0,  0, 0],
            [ 0,  0,  0, 1]
        ], dtype=np.float32)

        # Transform to optical convention
        # To go from optical camera frame to odom: first optical→ROS, then ROS→odom
        T_odom_cam_optical = T_odom_cam_ros @ T_optical_to_ros
        # To go from odom to optical camera frame: first odom→ROS, then ROS→optical
        T_cam_odom_optical = T_ros_to_optical @ np.linalg.inv(T_odom_cam_ros)

        return T_cam_odom_optical.astype(np.float32), T_odom_base

    def capture_images(self):
        """Capture RGB and Depth images using Camera class"""
        # Annotator was already added during initialization
        # Just get the data directly
        rgb = self.camera.get_rgb()
        depth = self.camera.get_depth()

        if depth is None:
            raise RuntimeError("Failed to get depth data")

        # Convert to numpy and ensure correct format
        rgb = np.array(rgb, dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]  # Remove alpha if present

        depth = np.array(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]  # Take first channel if multi-channel
        depth[~np.isfinite(depth)] = 0.0

        return rgb, depth

    def save_snapshot(self, output_dir, iteration):
        """Save observation snapshot for offline testing (compatible with test_bev_integration.py)"""
        # Create snapshot directory for this iteration
        snapshot_dir = os.path.join(output_dir, f"snapshot_{iteration:04d}")
        os.makedirs(snapshot_dir, exist_ok=True)

        # Capture images
        rgb, depth = self.capture_images()

        # Get transformation matrices
        T_cam_odom, T_odom_base = self.get_transform_matrices()

        # Save RGB image as PNG
        rgb_path = os.path.join(snapshot_dir, "rgb.png")
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(rgb_path, rgb_bgr)

        # Save Depth image as 16-bit PNG (in mm)
        depth_path = os.path.join(snapshot_dir, "depth.png")
        depth_mm = (depth * 1000.0).astype(np.uint16)
        cv2.imwrite(depth_path, depth_mm)

        # Save parameters as npz file
        params_path = os.path.join(snapshot_dir, "params.npz")
        np.savez(
            params_path,
            intrinsic=self.intrinsic,
            T_cam_odom=T_cam_odom,
            T_odom_base=T_odom_base
        )

        # Extract base pose for printing
        base_pos = T_odom_base[:3, 3]
        base_yaw = math.atan2(T_odom_base[1, 0], T_odom_base[0, 0])

        print(f"[Snapshot] Saved iteration {iteration} to {snapshot_dir}")
        print(f"  - rgb.png")
        print(f"  - depth.png")
        print(f"  - params.npz")
        print(f"  Base pose (T_odom_base):")
        print(f"    [{T_odom_base[0,0]:.6f}, {T_odom_base[0,1]:.6f}, {T_odom_base[0,2]:.6f}, {T_odom_base[0,3]:.6f}]")
        print(f"    [{T_odom_base[1,0]:.6f}, {T_odom_base[1,1]:.6f}, {T_odom_base[1,2]:.6f}, {T_odom_base[1,3]:.6f}]")
        print(f"    [{T_odom_base[2,0]:.6f}, {T_odom_base[2,1]:.6f}, {T_odom_base[2,2]:.6f}, {T_odom_base[2,3]:.6f}]")
        print(f"    [{T_odom_base[3,0]:.6f}, {T_odom_base[3,1]:.6f}, {T_odom_base[3,2]:.6f}, {T_odom_base[3,3]:.6f}]")

        return rgb, depth, snapshot_dir

    def reset_navigation_server(self):
        """Initialize navigation on server"""
        url = f'{self.server_url}/navigation_reset'
        data = {
            'goal': self.goal_text,
            'goal_description': '',
            'confidence_threshold': 0.5
        }

        print(f'[Client] Resetting navigation on server: {url}')
        try:
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            print(f'[Client] Server reset: {result}')
            return result
        except Exception as e:
            print(f'[Client] Failed to reset navigation: {e}')
            raise

    def request_navigation_step(self, rgb, depth):
        """Send observation to server and get navigation decision"""
        url = f'{self.server_url}/navigation_step'

        # Prepare RGB image (JPEG)
        rgb_pil = Image.fromarray(rgb, mode='RGB')
        rgb_buffer = BytesIO()
        rgb_pil.save(rgb_buffer, format='JPEG', quality=95)
        rgb_buffer.seek(0)

        # Prepare Depth image (PNG 16-bit)
        depth_mm = (depth * 1000.0).astype(np.uint16)
        depth_pil = Image.fromarray(depth_mm, mode='I;16')
        depth_buffer = BytesIO()
        depth_pil.save(depth_buffer, format='PNG')
        depth_buffer.seek(0)

        # Get transformation matrices
        T_cam_odom, T_odom_base = self.get_transform_matrices()

        # Prepare form data
        files = {
            'rgb': ('rgb.jpg', rgb_buffer, 'image/jpeg'),
            'depth': ('depth.png', depth_buffer, 'image/png')
        }

        data = {
            'intrinsic': json.dumps(self.intrinsic.flatten().tolist()),
            'T_cam_odom': json.dumps(T_cam_odom.flatten().tolist()),
            'T_odom_base': json.dumps(T_odom_base.flatten().tolist())
        }

        print('[Client] Sending observation to server...')
        try:
            response = requests.post(url, files=files, data=data, timeout=600)
            response.raise_for_status()
            result = response.json()
            return result
        except Exception as e:
            print(f'[Client] Server request failed: {e}')
            raise

    def execute_action(self, action, step_size):
        """Execute navigation action"""
        from scipy.spatial.transform import Rotation as R

        if action is None:
            return

        # Check action type
        if isinstance(action, tuple) and len(action) == 2:
            if action[0] == 'turn':
                # Rotation action: ('turn', angle_in_degrees)
                angle_deg = float(action[1])
                angle_rad = math.radians(angle_deg)

                print(f'[Client] Executing TURN: {angle_deg:.1f} degrees')

                # Rotate look direction
                r = R.from_euler('z', angle_rad, degrees=False)
                self.look_direction = r.apply(self.look_direction)
            else:
                # Forward movement: (r, theta)
                r_dist, theta = float(action[0]), float(action[1])
                print(f'[Client] Executing NAV STEP: r={r_dist:.2f}, theta={theta:.2f}')

                # Use the distance from server directly (no additional limit)
                dis = r_dist

                # Compute movement in base frame
                x_b = dis * math.cos(theta)
                y_b = dis * math.sin(theta)

                # Transform to world frame
                yaw = math.atan2(self.look_direction[1], self.look_direction[0])
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)

                dx = x_b * cos_yaw - y_b * sin_yaw
                dy = x_b * sin_yaw + y_b * cos_yaw

                self.current_position[0] += dx
                self.current_position[1] += dy

        # Keep base at ground level
        self.current_position[2] = 0.0

        # Update camera to follow robot
        self._update_camera_follow_robot()

    def run_navigation(self):
        """Main navigation loop"""
        # Reset server
        self.reset_navigation_server()

        return True  # Ready to navigate


# ---------------------------------------------------------
# 4. MAIN SIMULATION LOOP
# ---------------------------------------------------------

reset_needed = False
first_step = True

def run(scene_data, output_dir=None, episode_id=None, goal_text=""):
    from pxr import Sdf
    from isaacsim.core.api import World
    from isaacsim.core.utils.prims import define_prim
    import omni.replicator.core as rep

    my_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 200.0, rendering_dt=8.0 / 200.0)
    my_world.scene.add_default_ground_plane(z_position=0, name="default_ground_plane", prim_path="/World/defaultGroundPlane")

    if "usd_path" in scene_data:
        prim = define_prim("/World/Ground", "Xform")
        asset_path = scene_data["usd_path"]
        prim.GetReferences().AddReference(asset_path, "/Root")

    nav_controller = LLMNavigationController(
        world=my_world,
        scene_data=scene_data,
        server_url=args.server,
        goal_text=goal_text
    )

    stage = simulation_app.context.get_stage()
    dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)

    my_world.reset()

    # Initialize camera AFTER world.reset()
    nav_controller.initialize_camera()

    nav_controller.reset()

    global reset_needed
    def on_physics_step(step_size):
        global first_step, reset_needed
        if first_step:
            nav_controller.reset()
            first_step = False
        elif reset_needed:
            my_world.reset(True)
            reset_needed = False
            first_step = True

    my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

    camera_transforms = []
    frame = 0
    nav_started = False
    wait_frames = 0  # Wait counter for synchronous execution

    print(f"\n[Client] Starting LLM Navigation (Synchronous Mode)")
    print(f"[Client] Goal: {goal_text}")
    print(f"[Client] Max iterations: {args.max_iterations}")

    while simulation_app.is_running():
        my_world.step(render=True)
        if my_world.is_stopped() and not reset_needed:
            reset_needed = True

        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                nav_controller.reset()
                frame = 0
                reset_needed = False
                camera_transforms = []
                nav_started = False
                wait_frames = 0

            # Start navigation on first frame
            if not nav_started:
                nav_controller.run_navigation()
                nav_started = True
                wait_frames = 20  # Wait for camera annotator to be ready
                print('[Client] Waiting for camera annotator to initialize...')
                continue

            # Record trajectory
            if args.record and (frame % 50 == 0):
                transform = nav_controller.get_camera_transform()
                transform["frame"] = frame
                camera_transforms.append(transform)

            # If waiting for simulation to stabilize, just continue
            if wait_frames > 0:
                wait_frames -= 1
                frame += 1
                continue

            # Check max iterations
            if nav_controller.iteration_count >= args.max_iterations:
                print(f"\n[Client] Reached max iterations ({args.max_iterations})")
                break

            # Synchronous navigation step: Perceive -> Navigate -> Execute
            try:
                nav_controller.iteration_count += 1
                print(f"\n[Client] === ITERATION {nav_controller.iteration_count} ===")

                # 1. PERCEIVE: Capture images
                snapshot_dir = os.path.join(output_dir, f"snapshots_episode_{episode_id}")
                rgb, depth, _ = nav_controller.save_snapshot(snapshot_dir, nav_controller.iteration_count)

                # 2. NAVIGATE: Get decision from server
                result = nav_controller.request_navigation_step(rgb, depth)

                action_type = result.get('action_type', 'none')
                print(f'[Client] Server response: action={action_type}')

                # 3. UPDATE ORIENTATION: Point toward goal
                if action_type == 'nav_step' and result.get('goal_pose'):
                    goal_pose = result['goal_pose']['pose']
                    goal_x = goal_pose['position']['x']
                    goal_y = goal_pose['position']['y']

                    dx = goal_x - nav_controller.current_position[0]
                    dy = goal_y - nav_controller.current_position[1]
                    dist = math.sqrt(dx*dx + dy*dy)

                    if dist > 1e-6:
                        nav_controller.look_direction[0] = dx / dist
                        nav_controller.look_direction[1] = dy / dist
                        nav_controller.look_direction[2] = 0.0
                        print(f'[Client] Updated look_direction toward goal: ({dx/dist:.3f}, {dy/dist:.3f})')
                        print(f'[Client] Distance to goal: {dist:.3f}m')

                # 4. EXECUTE: Perform action
                if action_type != 'none':
                    # Calculate action based on goal_pose and action_type
                    if action_type == 'nav_step' and result.get('goal_pose'):
                        # Forward movement: move toward goal_pose (already oriented toward it)
                        goal_pose = result['goal_pose']['pose']
                        goal_x = goal_pose['position']['x']
                        goal_y = goal_pose['position']['y']

                        dx = goal_x - nav_controller.current_position[0]
                        dy = goal_y - nav_controller.current_position[1]
                        dist = math.sqrt(dx*dx + dy*dy)

                        # Action in robot frame: move forward (theta=0) by distance
                        action = (dist, 0.0)

                    elif action_type in ['turn_left', 'turn_right'] and result.get('goal_pose'):
                        # Rotation action: extract angle from goal_pose orientation
                        goal_pose = result['goal_pose']['pose']
                        goal_orientation = goal_pose['orientation']

                        # Convert goal quaternion to yaw
                        q = goal_orientation
                        goal_yaw = math.atan2(
                            2.0 * (q['w'] * q['z'] + q['x'] * q['y']),
                            1.0 - 2.0 * (q['y'] * q['y'] + q['z'] * q['z'])
                        )

                        # Current yaw from look_direction
                        current_yaw = math.atan2(nav_controller.look_direction[1], nav_controller.look_direction[0])

                        # Calculate angle difference
                        angle_diff = goal_yaw - current_yaw
                        # Normalize to [-pi, pi]
                        while angle_diff > math.pi:
                            angle_diff -= 2 * math.pi
                        while angle_diff < -math.pi:
                            angle_diff += 2 * math.pi

                        angle_deg = math.degrees(angle_diff)
                        action = ('turn', angle_deg)
                        print(f'[Client] Rotation action: {angle_deg:.1f} degrees')

                    else:
                        # Unknown action type, skip
                        print(f'[Client] Unknown action type: {action_type}, skipping')
                        action = None

                    if action is not None:
                        nav_controller.execute_action(action, 0.04)
                        print(f'[Client] Action executed, waiting for simulation to stabilize...')

                # 5. WAIT: Let simulation stabilize before next iteration
                wait_frames = 10  # Wait 10 frames (~0.4 seconds at 25 FPS)

            except Exception as e:
                print(f'[Client] Navigation step failed: {e}')
                import traceback
                traceback.print_exc()
                break

            if args.headless and frame > 2000:
                break
            frame += 1

    if args.record:
        print("save camera_transforms", len(camera_transforms))
        df = pd.DataFrame(camera_transforms)
        csv_save_path = os.path.join(output_dir, f"{episode_id}.csv")
        df.to_csv(csv_save_path, index=False)
        return csv_save_path, True

    return None, True


if __name__ == '__main__':
    work_dir = os.getcwd()

    task = args.task
    print(task)

    with open(f"{task}_grained_demo.json", 'r') as f:
        data = json.load(f)

    cur_episode = data[int(args.index)]
    scene_id = cur_episode['scan']
    loc = cur_episode['start_position']
    ori = cur_episode['start_rotation']
    # instruction_text = cur_episode['instruction']['instruction_text']
    instruction_text = "sofa"

    if task == "coarse":
        instruction_text = instruction_text["natural"]

    usd_path = os.path.join(work_dir, scene_id, f'{scene_id}.usda')

    loc[-1] = 1.5
    scene_data = {
        "usd_path": usd_path,
        "start_location": loc,
        "start_orientation": ori,
    }

    output_dir = work_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Instruction: {instruction_text}")
    print(f"Server URL: {args.server}")

    # Run simulation with LLM navigation
    saved_csv_path, success = run(scene_data, output_dir, cur_episode['episode_id'], goal_text=instruction_text)

    if not success:
        sys.exit(0)

    # Evaluate performance
    if saved_csv_path and os.path.exists(saved_csv_path):
        print("\nEvaluating Performance...")
        _, metrics = evaluate_single_episode(saved_csv_path, cur_episode)

        if metrics:
            print("\n" + "="*50)
            print("FINAL RESULTS:")
            print(f"  SR:   {metrics['SR']}")
            print(f"  OSR:  {metrics['OSR']}")
            print(f"  SPL:  {metrics['SPL']:.3f}")
            print(f"  nDTW: {metrics['nDTW']:.3f}")
            print(f"  TL:   {metrics['TL']:.2f}m")
            print(f"  Goal Distance: {metrics['Goal Dist']:.2f}m")
            print("="*50)

    print(f"Done.")
