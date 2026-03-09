#!/usr/bin/env python3
"""Local planning pipeline with pluggable interfaces.

Flow:
1) LLM generates a local language subgoal.
2) Video generator predicts a future clip conditioned on RGBD + subgoal.
3) PI3 decoder reconstructs local 3D trajectory from predicted clip.
4) Trajectory is compressed into 2D waypoints for execution.

This module intentionally leaves video generation and PI3 decoding as interfaces
so concrete model integration can be implemented externally.
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from geometry_utils import compress_traj3d_to_2d


@dataclass
class VideoGenerationInput:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    base_to_odom: np.ndarray
    subgoal: dict[str, Any]
    goal: str
    iteration: int


@dataclass
class VideoPrediction:
    """Container for generated future clip and metadata."""

    frames: np.ndarray  # (T, H, W, 3), RGB uint8 preferred
    fps: float
    metadata: dict[str, Any]


@dataclass
class PI3DecodeInput:
    video_prediction: VideoPrediction
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    base_to_odom: np.ndarray


class VideoGenerator(Protocol):
    """Interface for future video generation model (e.g., Wan2.6)."""

    def generate(self, data: VideoGenerationInput) -> VideoPrediction:
        raise NotImplementedError


class PI3Decoder(Protocol):
    """Interface for PI3-style trajectory reconstruction model."""

    def decode(self, data: PI3DecodeInput) -> np.ndarray:
        """Return local trajectory with shape (N, 3) in odom frame."""
        raise NotImplementedError


class PlaceholderVideoGenerator:
    """Placeholder implementation to be replaced by real model integration."""

    def generate(self, data: VideoGenerationInput) -> VideoPrediction:
        raise NotImplementedError(
            "PlaceholderVideoGenerator is not implemented. "
            "Please provide a concrete VideoGenerator (e.g., Wan2.6 service/client)."
        )


class PlaceholderPI3Decoder:
    """Placeholder implementation to be replaced by real PI3 integration."""

    def decode(self, data: PI3DecodeInput) -> np.ndarray:
        raise NotImplementedError(
            "PlaceholderPI3Decoder is not implemented. "
            "Please provide a concrete PI3Decoder implementation."
        )


class LocalPlanningPipeline:
    def __init__(self, agent, video_generator: VideoGenerator | None = None, pi3_decoder: PI3Decoder | None = None):
        self.agent = agent
        self.video_generator = video_generator or PlaceholderVideoGenerator()
        self.pi3_decoder = pi3_decoder or PlaceholderPI3Decoder()

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        if not text:
            return {}
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
        except Exception:
            pass
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                parsed = ast.literal_eval(text[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            pass
        return {}

    def generate_subgoal(self, obs: dict, goal: str, iteration: int, goal_description: str = '') -> tuple[dict[str, Any], str, np.ndarray | None]:
        bev_map = None
        if self.agent.obstacle_map is not None:
            self.agent.obstacle_map.update_map(
                depth=obs['depth'], intrinsic=obs['intrinsic'], extrinsic=obs['extrinsic'], base_to_odom=obs['base_to_odom_matrix']
            )
            bev_map = self.agent._generate_bev_with_waypoints(obs['base_to_odom_matrix'], waypoints=[])

        prompt = (
            f"Iteration {iteration}. Goal: {goal}. "
            f"Goal description: {goal_description or 'N/A'}. "
            "Based on RGBD and BEV context, output ONE local subgoal in JSON: "
            "{\"intent\": one of [go_straight, follow_corridor, turn_left, turn_right, look_into_room], "
            "\"instruction\": short Chinese imperative, \"horizon_m\": float, \"confidence\": 0~1}."
        )
        images = [obs['rgb']] + ([bev_map] if bev_map is not None else [])
        raw = self.agent.actionVLM.call_chat(history=self.agent.cfg.get('vlm_history', 3), images=images, text_prompt=prompt)
        parsed = self._parse_json_response(raw)
        if 'intent' not in parsed:
            parsed = {
                'intent': 'go_straight',
                'instruction': '沿当前方向直行并保持避障',
                'horizon_m': 2.0,
                'confidence': 0.3,
            }
        return parsed, raw, bev_map

    def _fallback_path_from_pose(self, base_to_odom: np.ndarray) -> list[dict[str, float]]:
        """Fallback path when model interfaces are not wired yet."""
        x = float(base_to_odom[0, 3])
        y = float(base_to_odom[1, 3])
        yaw = float(np.arctan2(base_to_odom[1, 0], base_to_odom[0, 0]))
        return [{'x': x, 'y': y, 'yaw': yaw}]

    def plan(self, obs: dict, goal: str, iteration: int, goal_description: str = '') -> dict[str, Any]:
        t0 = time.time()
        subgoal, subgoal_raw, bev_map = self.generate_subgoal(obs, goal, iteration, goal_description)
        t1 = time.time()

        video_prediction = None
        traj_3d = np.zeros((0, 3), dtype=np.float64)
        path_2d: list[dict[str, float]]
        status = 'ok'
        interface_error = None

        try:
            v_in = VideoGenerationInput(
                rgb=obs['rgb'],
                depth=obs['depth'],
                intrinsic=obs['intrinsic'],
                extrinsic=obs['extrinsic'],
                base_to_odom=obs['base_to_odom_matrix'],
                subgoal=subgoal,
                goal=goal,
                iteration=iteration,
            )
            video_prediction = self.video_generator.generate(v_in)
            t2 = time.time()

            d_in = PI3DecodeInput(
                video_prediction=video_prediction,
                intrinsic=obs['intrinsic'],
                extrinsic=obs['extrinsic'],
                base_to_odom=obs['base_to_odom_matrix'],
            )
            traj_3d = self.pi3_decoder.decode(d_in)
            path_2d = compress_traj3d_to_2d(traj_3d)
            t3 = time.time()
        except NotImplementedError as e:
            status = 'interface_not_implemented'
            interface_error = str(e)
            path_2d = self._fallback_path_from_pose(obs['base_to_odom_matrix'])
            t2 = t1
            t3 = t2
        except Exception as e:
            status = 'pipeline_error'
            interface_error = str(e)
            path_2d = self._fallback_path_from_pose(obs['base_to_odom_matrix'])
            t2 = t1
            t3 = t2

        # Regenerate BEV map with trajectory visualization
        if self.agent.obstacle_map is not None and len(traj_3d) > 0:
            bev_map = self.agent._generate_bev_with_waypoints(
                obs['base_to_odom_matrix'],
                waypoints=[],
                traj_3d=traj_3d
            )

        return {
            'status': status,
            'error': interface_error,
            'subgoal': subgoal,
            'subgoal_raw': subgoal_raw,
            'video_prediction': video_prediction,
            'traj_3d': traj_3d,
            'path_2d': path_2d,
            'bev_map': bev_map,
            'timing': {
                'subgoal_time': float(t1 - t0),
                'video_gen_time': float(t2 - t1),
                'traj_decode_time': float(t3 - t2),
            }
        }
