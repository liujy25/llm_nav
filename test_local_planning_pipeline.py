import numpy as np

from geometry_utils import compress_traj3d_to_2d
from local_planning_pipeline import (
    LocalPlanningPipeline,
    PlaceholderPI3Decoder,
    PlaceholderVideoGenerator,
)


def test_compress_traj3d_to_2d_returns_sparse_path():
    traj = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.4, 0.0, 0.0],
        [0.8, 0.2, 0.0],
    ])
    path = compress_traj3d_to_2d(traj, min_waypoint_dist=0.25)
    assert len(path) >= 2
    assert all({'x', 'y', 'yaw'} <= set(w.keys()) for w in path)


def test_placeholder_interfaces_raise_not_implemented():
    with np.testing.assert_raises(NotImplementedError):
        PlaceholderVideoGenerator().generate(None)
    with np.testing.assert_raises(NotImplementedError):
        PlaceholderPI3Decoder().decode(None)


def test_pipeline_fallback_when_interfaces_not_implemented():
    class DummyVLM:
        def call_chat(self, history, images, text_prompt):
            return '{"intent": "go_straight", "instruction": "直行", "horizon_m": 2.0, "confidence": 0.9}'

    class DummyAgent:
        obstacle_map = None
        actionVLM = DummyVLM()
        cfg = {'vlm_history': 1}

    obs = {
        'rgb': np.zeros((8, 8, 3), dtype=np.uint8),
        'depth': np.ones((8, 8), dtype=np.float32),
        'intrinsic': np.eye(3),
        'extrinsic': np.eye(4),
        'base_to_odom_matrix': np.eye(4),
    }

    result = LocalPlanningPipeline(DummyAgent()).plan(obs, goal='chair', iteration=1)
    assert result['status'] == 'interface_not_implemented'
    assert len(result['path_2d']) == 1
