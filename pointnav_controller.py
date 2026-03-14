#!/usr/bin/env python3
"""
PointNav-based Navigation Controller
Uses a pre-trained PointNav policy to navigate to goal positions
Similar to VLFM's approach
"""

import numpy as np
import torch
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from habitat.core.env import Env


class PointNavController:
    """Navigation controller using pre-trained PointNav policy"""

    def __init__(
        self,
        env: Env,
        pointnav_policy_path: str = "data/pointnav_weights.pth",
        depth_image_shape: Tuple[int, int] = (224, 224),
        stop_radius: float = 0.9,
        max_steps: int = 500,
        device: str = "cuda"
    ):
        """
        Args:
            env: Habitat environment
            pointnav_policy_path: Path to pre-trained PointNav policy weights
            depth_image_shape: Input depth image size for policy
            stop_radius: Distance threshold to stop (meters)
            max_steps: Maximum steps for navigation
            device: Device to run policy on
        """
        self.env = env
        self.stop_radius = stop_radius
        self.max_steps = max_steps
        self.device = torch.device(device)
        self.depth_image_shape = depth_image_shape
        self.depth_min, self.depth_max, self.depth_is_normalized = (
            self._resolve_depth_normalization_params()
        )

        # Load PointNav policy
        self.policy = self._load_pointnav_policy(pointnav_policy_path)
        self.policy.to(self.device)
        self.policy.eval()

        # Policy state
        self.reset_policy_state()

        print(f"[PointNavController] Initialized with policy: {pointnav_policy_path}")
        print(f"[PointNavController] Stop radius: {stop_radius}m")
        print(f"[PointNavController] Max steps: {max_steps}")
        print(
            "[PointNavController] Depth normalization: "
            f"min={self.depth_min:.3f}, max={self.depth_max:.3f}, "
            f"already_normalized={self.depth_is_normalized}"
        )

    def _resolve_depth_normalization_params(self) -> Tuple[float, float, bool]:
        """Infer depth normalization settings from Habitat sensor config."""
        depth_min = 0.5
        depth_max = 5.0
        depth_is_normalized = False

        try:
            sim_cfg = self.env.sim.habitat_config
            agent_names = list(sim_cfg.agents.keys())
            if agent_names:
                agent_cfg = sim_cfg.agents[agent_names[0]]
                if hasattr(agent_cfg, "sim_sensors"):
                    for sensor_name in agent_cfg.sim_sensors:
                        if "depth" not in sensor_name.lower():
                            continue
                        sensor_cfg = agent_cfg.sim_sensors[sensor_name]
                        if hasattr(sensor_cfg, "min_depth"):
                            depth_min = float(sensor_cfg.min_depth)
                        if hasattr(sensor_cfg, "max_depth"):
                            depth_max = float(sensor_cfg.max_depth)
                        if hasattr(sensor_cfg, "normalize_depth"):
                            depth_is_normalized = bool(sensor_cfg.normalize_depth)
                        break
        except Exception:
            # Fall back to VLFM defaults.
            pass

        if depth_max <= depth_min:
            depth_min, depth_max = 0.0, 1.0
        return depth_min, depth_max, depth_is_normalized

    def _load_pointnav_policy(self, ckpt_path: str):
        """Load pre-trained PointNav policy"""
        from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
        from gym import spaces
        from gym.spaces import Dict as SpaceDict, Discrete
        from omegaconf import OmegaConf, open_dict

        # Define observation and action spaces
        obs_space = SpaceDict({
            "depth": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.depth_image_shape[0], self.depth_image_shape[1], 1),
                dtype=np.float32
            ),
            "pointgoal_with_gps_compass": spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32
            ),
        })
        action_space = Discrete(4)  # STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT

        # Some VLFM checkpoints serialize config objects from `vlfm.*`.
        # Try to resolve a local sibling checkout automatically.
        try:
            import vlfm  # noqa: F401
        except ImportError:
            repo_root = Path(__file__).resolve().parent
            candidates = [
                repo_root.parent / "vlfm",
                Path.home() / "navigation" / "vlfm",
            ]
            for cand in candidates:
                if (cand / "vlfm").is_dir():
                    sys.path.insert(0, str(cand))
                    break

        # Load checkpoint.
        # PyTorch 2.6 changed torch.load default to weights_only=True, which can
        # fail for Habitat checkpoints that serialize config objects.
        try:
            ckpt_dict = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            # Backward compatibility for older PyTorch versions without the
            # `weights_only` argument.
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")

        # Compatibility: some older checkpoints store a single
        # `habitat_baselines.rl.policy` object, while newer Habitat expects
        # `habitat_baselines.rl.policy.<agent_name>`.
        config = OmegaConf.create(
            OmegaConf.to_container(ckpt_dict["config"], resolve=False)
        )
        try:
            agent_name = config.habitat.simulator.agents_order[0]
            policy_cfg = config.habitat_baselines.rl.policy
            has_agent_policy = False
            try:
                has_agent_policy = agent_name in policy_cfg
            except Exception:
                has_agent_policy = False

            if not has_agent_policy and "name" in policy_cfg:
                policy_cfg_container = OmegaConf.to_container(
                    policy_cfg, resolve=False
                )
                with open_dict(config.habitat_baselines.rl):
                    config.habitat_baselines.rl.policy = {
                        agent_name: policy_cfg_container
                    }
        except Exception:
            # Keep original config if compatibility rewrite is not applicable.
            pass

        # Create policy
        policy = PointNavResNetPolicy.from_config(
            config,
            obs_space,
            action_space
        )

        # Load weights
        policy.load_state_dict(ckpt_dict["state_dict"])

        return policy

    def reset_policy_state(self):
        """Reset policy hidden state and previous action"""
        self.rnn_hidden_states = torch.zeros(
            1,  # num_envs
            self.policy.net.num_recurrent_layers,
            512,  # hidden_size
            device=self.device,
            dtype=torch.float32
        )
        self.prev_actions = torch.zeros(
            1,  # num_envs
            1,  # action_dim for discrete actions
            device=self.device,
            dtype=torch.long
        )
        self.not_done_masks = torch.zeros(
            1, 1,
            device=self.device,
            dtype=torch.bool
        )

    def _compute_rho_theta(
        self,
        current_position: np.ndarray,
        current_heading: float,
        goal_position: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute polar coordinates (rho, theta) from current pose to goal

        Args:
            current_position: [x, y] in meters
            current_heading: yaw in radians
            goal_position: [x, y] in meters

        Returns:
            rho: distance to goal (meters)
            theta: angle to goal (radians, positive=left)
        """
        # Rotation matrix to transform to robot's local frame
        cos_h = np.cos(-current_heading)
        sin_h = np.sin(-current_heading)
        rotation_matrix = np.array([
            [cos_h, -sin_h],
            [sin_h, cos_h]
        ])

        # Goal in robot's local frame
        local_goal = goal_position - current_position
        local_goal = rotation_matrix @ local_goal

        # Convert to polar coordinates
        rho = np.linalg.norm(local_goal)
        theta = np.arctan2(local_goal[1], local_goal[0])

        return float(rho), float(theta)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def _yaw_from_xyzw(quat_xyzw: np.ndarray) -> float:
        """Extract yaw (rotation around Y) from quaternion [x, y, z, w]."""
        x, y, z, w = quat_xyzw
        return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y**2 + z**2)))

    def _current_yaw(self) -> float:
        """Read current agent yaw from simulator state."""
        quat = self.env.sim.get_agent_state().rotation
        return float(
            np.arctan2(
                2.0 * (quat.w * quat.y + quat.x * quat.z),
                1.0 - 2.0 * (quat.y**2 + quat.z**2),
            )
        )

    def _resize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Resize depth image to policy input size"""
        import cv2

        # Ensure depth is 2D
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        # Resize using area interpolation (best for downsampling)
        depth_resized = cv2.resize(
            depth,
            (self.depth_image_shape[1], self.depth_image_shape[0]),
            interpolation=cv2.INTER_AREA
        )

        # Add channel dimension
        depth_resized = depth_resized[:, :, np.newaxis]

        return depth_resized

    def _extract_observation(
        self,
        obs: Dict,
        goal_position: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """
        Extract and prepare observation for PointNav policy

        Args:
            obs: Habitat observation dict
            goal_position: [x, y] goal position in Habitat coordinates

        Returns:
            Policy observation dict with depth and pointgoal
        """
        # Find depth sensor
        depth_key = None
        for key in obs.keys():
            if 'depth' in key.lower():
                depth_key = key
                break

        if depth_key is None:
            raise ValueError(f"Depth sensor not found in observation keys: {list(obs.keys())}")

        depth = obs[depth_key]

        # Process depth
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = depth.astype(np.float32)

        # Handle invalid values
        depth[~np.isfinite(depth)] = self.depth_max

        # Normalize to [0, 1] before feeding policy.
        if self.depth_is_normalized:
            depth = np.clip(depth, 0.0, 1.0)
        else:
            depth = np.clip(depth, self.depth_min, self.depth_max)
            depth = (depth - self.depth_min) / (self.depth_max - self.depth_min)

        # Resize depth
        depth_resized = self._resize_depth(depth)

        # Get agent state
        agent_state = self.env.sim.get_agent_state()
        current_position = agent_state.position[[0, 2]]  # [x, z] in Habitat

        # Get heading from quaternion
        quat = agent_state.rotation
        # Convert quaternion to yaw (rotation around Y axis in Habitat)
        # yaw = atan2(2*(w*y + x*z), 1 - 2*(y^2 + z^2))
        yaw = np.arctan2(
            2.0 * (quat.w * quat.y + quat.x * quat.z),
            1.0 - 2.0 * (quat.y**2 + quat.z**2)
        )

        # Compute rho and theta
        rho, theta = self._compute_rho_theta(
            current_position,
            yaw,
            goal_position
        )

        # Prepare tensors
        depth_tensor = torch.from_numpy(depth_resized).to(
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)  # Add batch dimension

        pointgoal_tensor = torch.tensor(
            [[rho, theta]],
            device=self.device,
            dtype=torch.float32
        )

        policy_obs = {
            "depth": depth_tensor,
            "pointgoal_with_gps_compass": pointgoal_tensor
        }

        return policy_obs, rho, theta

    def navigate_to_position(
        self,
        goal_position: np.ndarray,
        goal_orientation: Optional[np.ndarray] = None,
        verbose: bool = True,
        max_steps_override: Optional[int] = None,
        step_callback=None
    ) -> Tuple[bool, list, Dict[str, Any]]:
        """
        Navigate to goal position using PointNav policy

        Args:
            goal_position: [x, y, z] target position in Habitat coordinates
            goal_orientation: [x, y, z, w] target quaternion (optional, for future use)
            verbose: Print navigation progress
            max_steps_override: Optional max step budget for this call
            step_callback: Optional callback called after each env.step(action).
                Signature: callback(obs, step_idx, rho, theta, action_id) -> bool
                Return True to interrupt navigation early.

        Returns:
            success: Whether reached goal
            observations: List of observations during navigation
            info: Navigation statistics
        """
        # Reset policy state
        self.reset_policy_state()

        # Extract 2D goal position (x, z in Habitat)
        goal_2d = goal_position[[0, 2]]
        goal_yaw = None
        if goal_orientation is not None:
            goal_yaw = self._yaw_from_xyzw(np.asarray(goal_orientation, dtype=np.float32))

        observations = []
        actions_taken = []
        interrupted = False
        interrupt_reason = None

        max_steps = self.max_steps if max_steps_override is None else int(max_steps_override)
        if max_steps < 0:
            max_steps = 0

        if verbose:
            print(f"[PointNavController] Navigating to: {goal_position}")
            print(f"[PointNavController] Goal 2D (x, z): {goal_2d}")
            print(f"[PointNavController] Step budget: {max_steps}")

        if max_steps == 0:
            info = {
                "num_steps": 0,
                "actions": [],
                "final_distance": -1.0,
                "success": False,
                "interrupted": True,
                "interrupt_reason": "step_budget_exhausted",
                "step_budget": 0,
            }
            return False, observations, info

        for step in range(max_steps):
            # Get current observation
            obs = self.env.sim.get_sensor_observations()
            observations.append(obs)

            # Prepare policy observation
            policy_obs, rho, theta = self._extract_observation(obs, goal_2d)

            if verbose and step % 10 == 0:
                print(f"  Step {step}: rho={rho:.2f}m, theta={np.rad2deg(theta):.1f}°")

            # Check if reached goal
            if rho < self.stop_radius:
                if verbose:
                    print(f"[PointNavController] Reached goal! Final distance: {rho:.2f}m")

                # Do NOT emit STOP here; in Habitat tasks STOP may end the episode.
                # For iterative keyframe navigation, treat "within stop_radius" as
                # local success and optionally align heading.
                if goal_yaw is not None:
                    yaw_tol = np.deg2rad(30.0)
                    if verbose:
                        print("[PointNavController] Aligning to goal orientation...")

                    while len(actions_taken) < max_steps:
                        cur_yaw = self._current_yaw()
                        yaw_err = self._normalize_angle(goal_yaw - cur_yaw)
                        if abs(yaw_err) <= yaw_tol:
                            if verbose:
                                print(
                                    "[PointNavController] Orientation aligned "
                                    f"(error={np.rad2deg(yaw_err):.1f}°)"
                                )
                            break

                        # 2: TURN_LEFT, 3: TURN_RIGHT
                        action_id = 2 if yaw_err > 0 else 3
                        obs = self.env.step(action_id)
                        observations.append(obs)
                        actions_taken.append(action_id)

                        if step_callback is not None:
                            should_interrupt = bool(
                                step_callback(
                                    obs, len(actions_taken), rho, theta, action_id
                                )
                            )
                            if should_interrupt:
                                interrupted = True
                                interrupt_reason = "step_callback_interrupt"
                                break

                        if self.env.episode_over:
                            interrupted = True
                            interrupt_reason = "episode_over"
                            break

                success = not interrupted
                break

            # Get action from policy
            with torch.no_grad():
                policy_output = self.policy.act(
                    policy_obs,
                    self.rnn_hidden_states,
                    self.prev_actions,
                    self.not_done_masks,
                    deterministic=True
                )

                action = policy_output.actions
                self.rnn_hidden_states = policy_output.rnn_hidden_states

            # Execute action in environment
            action_id = action.item()
            actions_taken.append(action_id)

            obs = self.env.step(action_id)
            observations.append(obs)

            if step_callback is not None:
                should_interrupt = bool(step_callback(obs, len(actions_taken), rho, theta, action_id))
                if should_interrupt:
                    interrupted = True
                    interrupt_reason = "step_callback_interrupt"
                    success = False
                    break

            # Update policy state
            self.prev_actions = action
            self.not_done_masks = torch.ones_like(self.not_done_masks)

            # Check if episode ended
            if self.env.episode_over:
                if verbose:
                    print(f"[PointNavController] Episode ended at step {step}")
                success = False
                break
        else:
            # Max steps reached
            if verbose:
                print(f"[PointNavController] Max steps ({max_steps}) reached")
            success = False

        # Navigation statistics
        info = {
            "num_steps": len(actions_taken),
            "actions": actions_taken,
            "final_distance": rho if 'rho' in locals() else -1,
            "success": success,
            "interrupted": interrupted,
            "interrupt_reason": interrupt_reason,
            "step_budget": int(max_steps)
        }

        return success, observations, info
