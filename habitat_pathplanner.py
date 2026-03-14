#!/usr/bin/env python3
"""
Minimal Python reimplementation of the ApexNav "path -> discrete action" logic.

Inputs:
- 2D occupancy map (0 free, 1 occupied)
- start/goal in world coordinates (meters)
- current yaw (radians)

Outputs:
- A* path in world coordinates
- next discrete action: MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / STOP
"""

from __future__ import annotations

import argparse
import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


Action = str
STOP: Action = "STOP"
MOVE_FORWARD: Action = "MOVE_FORWARD"
TURN_LEFT: Action = "TURN_LEFT"
TURN_RIGHT: Action = "TURN_RIGHT"


@dataclass
class PlannerConfig:
    map_resolution: float = 0.05
    astar_resolution: float = 0.1
    step_length: float = 0.25
    turn_angle: float = math.pi / 6.0
    local_distance: float = 0.80
    action_distance: float = 0.25
    yaw_threshold_scale: float = 1.9


class ApexNavPyPlanner:
    def __init__(self, occmap: np.ndarray, origin_xy: Tuple[float, float], cfg: PlannerConfig):
        """
        Args:
            occmap: HxW uint8/bool map, 0=free, non-zero=occupied.
            origin_xy: World (x, y) for grid index (0, 0) corner.
            cfg: Planner hyper-parameters.
        """
        if occmap.ndim != 2:
            raise ValueError(f"occmap must be HxW, got shape={occmap.shape}")

        self.occmap = (occmap > 0).astype(np.uint8)
        self.h, self.w = self.occmap.shape
        self.origin = np.array(origin_xy, dtype=np.float64)
        self.cfg = cfg
        self.astar_dirs = self._make_12_dirs(cfg.step_length)

    @staticmethod
    def _make_12_dirs(step_length: float) -> np.ndarray:
        dirs = []
        for i in range(12):
            a = i * (math.pi / 6.0)
            dirs.append([step_length * math.cos(a), step_length * math.sin(a)])
        return np.asarray(dirs, dtype=np.float64)

    def world_to_grid(self, p: np.ndarray, resolution: Optional[float] = None) -> Tuple[int, int]:
        res = self.cfg.astar_resolution if resolution is None else float(resolution)
        idx_f = np.floor((p - self.origin) / res).astype(np.int64)
        return int(idx_f[0]), int(idx_f[1])

    def grid_to_world(self, idx: Tuple[int, int], resolution: Optional[float] = None) -> np.ndarray:
        res = self.cfg.astar_resolution if resolution is None else float(resolution)
        return (np.array(idx, dtype=np.float64) + 0.5) * res + self.origin

    def _is_in_bounds_world(self, p: np.ndarray) -> bool:
        gx, gy = self.world_to_grid(p, resolution=self.cfg.map_resolution)
        return 0 <= gx < self.w and 0 <= gy < self.h

    def _is_occupied_world(self, p: np.ndarray) -> bool:
        gx, gy = self.world_to_grid(p, resolution=self.cfg.map_resolution)
        if gx < 0 or gx >= self.w or gy < 0 or gy >= self.h:
            return True
        return bool(self.occmap[gy, gx])

    def _is_safe_world(self, p: np.ndarray) -> bool:
        if not self._is_in_bounds_world(p):
            return False
        return not self._is_occupied_world(p)

    @staticmethod
    def _diag_heuristic(a: np.ndarray, b: np.ndarray) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        tie = 1.0 + 1e-6 * (dx + dy)
        return tie * (math.sqrt(2.0) * min(dx, dy) + abs(dx - dy))

    def astar_search(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        success_dist: float = 0.25,
        max_expansions: int = 100000,
    ) -> List[np.ndarray]:
        """
        Habitat/ApexNav-like A*: 12 headings, fixed step, continuous world checks.
        """
        start_xy = np.asarray(start_xy, dtype=np.float64)
        goal_xy = np.asarray(goal_xy, dtype=np.float64)
        if not self._is_safe_world(start_xy) or not self._is_safe_world(goal_xy):
            return []

        start_idx = self.world_to_grid(start_xy)
        goal_idx = self.world_to_grid(goal_xy)

        open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
        g_score: Dict[Tuple[int, int], float] = {start_idx: 0.0}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        pos_of: Dict[Tuple[int, int], np.ndarray] = {start_idx: start_xy.copy()}
        closed = set()
        tie = 0

        f0 = self._diag_heuristic(start_xy, goal_xy)
        heapq.heappush(open_heap, (f0, tie, start_idx))

        expansions = 0
        while open_heap and expansions < max_expansions:
            _, _, cur_idx = heapq.heappop(open_heap)
            if cur_idx in closed:
                continue
            closed.add(cur_idx)
            expansions += 1
            cur_pos = pos_of[cur_idx]

            reach_end = abs(cur_idx[0] - goal_idx[0]) <= 1 and abs(cur_idx[1] - goal_idx[1]) <= 1
            if np.linalg.norm(cur_pos - goal_xy) < success_dist:
                reach_end = True
            if reach_end:
                return self._reconstruct_path(parent, pos_of, cur_idx, goal_xy)

            for step in self.astar_dirs:
                nbr_pos = cur_pos + step

                if np.linalg.norm(nbr_pos - start_xy) > 0.25:
                    if not self._is_safe_world(nbr_pos):
                        continue
                    # Segment safety check every 2.5cm.
                    seg = step
                    seg_len = np.linalg.norm(seg)
                    if seg_len < 1e-9:
                        continue
                    seg_dir = seg / seg_len
                    safe = True
                    d = 0.025
                    while d < seg_len:
                        ck = cur_pos + d * seg_dir
                        if not self._is_safe_world(ck):
                            safe = False
                            break
                        d += 0.025
                    if not safe:
                        continue

                nbr_idx = self.world_to_grid(nbr_pos)
                if nbr_idx in closed:
                    continue

                cand_g = g_score[cur_idx] + float(np.linalg.norm(step))
                old_g = g_score.get(nbr_idx, float("inf"))
                if cand_g >= old_g:
                    continue

                g_score[nbr_idx] = cand_g
                parent[nbr_idx] = cur_idx
                pos_of[nbr_idx] = nbr_pos
                tie += 1
                f = cand_g + self._diag_heuristic(nbr_pos, goal_xy)
                heapq.heappush(open_heap, (f, tie, nbr_idx))

        return []

    @staticmethod
    def _reconstruct_path(
        parent: Dict[Tuple[int, int], Tuple[int, int]],
        pos_of: Dict[Tuple[int, int], np.ndarray],
        end_idx: Tuple[int, int],
        goal_xy: np.ndarray,
    ) -> List[np.ndarray]:
        nodes: List[np.ndarray] = [goal_xy.copy(), pos_of[end_idx].copy()]
        cur = end_idx
        while cur in parent:
            cur = parent[cur]
            nodes.append(pos_of[cur].copy())
        nodes.reverse()
        return nodes

    def select_local_target(self, current_pos: np.ndarray, path: List[np.ndarray]) -> np.ndarray:
        """
        Equivalent to ApexNav selectLocalTarget:
        choose a point ~local_distance ahead on path.
        """
        if not path:
            return current_pos.copy()
        if len(path) == 1:
            return path[0].copy()

        cp = np.asarray(current_pos, dtype=np.float64)
        target = path[-1].copy()

        start_idx = 0
        min_dist = float("inf")
        for i in range(len(path) - 1):
            d = float(np.linalg.norm(path[i] - cp))
            if d < min_dist:
                min_dist = d
                start_idx = i + 1

        if start_idx >= len(path):
            return target

        length = float(np.linalg.norm(path[start_idx] - cp))
        for i in range(start_idx + 1, len(path)):
            length += float(np.linalg.norm(path[i] - path[i - 1]))
            if length > self.cfg.local_distance and np.linalg.norm(cp - path[i - 1]) > 0.30:
                target = path[i - 1].copy()
                break
        return target

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def decide_next_action(self, current_yaw: float, target_yaw: float) -> Action:
        yaw_diff = self._wrap_angle(target_yaw - current_yaw)
        thr = self.cfg.turn_angle / self.cfg.yaw_threshold_scale
        if abs(yaw_diff) > thr:
            return TURN_LEFT if yaw_diff > 0 else TURN_RIGHT
        return MOVE_FORWARD

    def plan_action_once(
        self,
        current_pos: np.ndarray,
        current_yaw: float,
        goal_pos: np.ndarray,
    ) -> Tuple[Action, List[np.ndarray]]:
        """
        One closed-loop planning tick:
        goal -> path -> local target -> one discrete action.
        """
        cp = np.asarray(current_pos, dtype=np.float64)
        gp = np.asarray(goal_pos, dtype=np.float64)

        if np.linalg.norm(cp - gp) <= self.cfg.action_distance:
            return STOP, [cp.copy()]

        path = self.astar_search(cp, gp, success_dist=self.cfg.action_distance)
        if not path:
            return STOP, []

        local_target = self.select_local_target(cp, path)
        target_yaw = math.atan2(local_target[1] - cp[1], local_target[0] - cp[0])
        action = self.decide_next_action(current_yaw, target_yaw)
        return action, path


def _make_demo_map(size: int = 160) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.uint8)
    m[40:120, 78:82] = 1
    m[78:82, 40:120] = 1
    m[75:85, 75:85] = 0  # opening
    return m


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run with built-in toy map.")
    args = parser.parse_args()

    if not args.demo:
        print("Use --demo to run a minimal example.")
        return

    occ = _make_demo_map()
    cfg = PlannerConfig(map_resolution=0.05, astar_resolution=0.1, step_length=0.25)
    planner = ApexNavPyPlanner(occ, origin_xy=(0.0, 0.0), cfg=cfg)

    current_pos = np.array([1.0, 1.0], dtype=np.float64)
    goal_pos = np.array([6.0, 6.0], dtype=np.float64)
    current_yaw = 0.0

    action, path = planner.plan_action_once(current_pos, current_yaw, goal_pos)
    print(f"action={action}, path_len={len(path)}")
    if path:
        print(f"first={path[0]}, last={path[-1]}")


if __name__ == "__main__":
    _main()
