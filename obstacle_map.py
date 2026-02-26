#!/usr/bin/env python3
"""
Obstacle map for frontier-based navigation.
Adapted from VLFM to work with existing nav_agent coordinate system.
"""

import cv2
import numpy as np
from typing import Optional

from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

from geometry_utils import (
    extract_yaw,
    get_point_cloud,
    transform_points,
    filter_points_by_height
)


class ObstacleMap:
    """
    Maintains obstacle and explored area maps for frontier-based navigation.
    Adapted to work with nav_agent's coordinate system (odom frame).
    """
    
    def __init__(
        self,
        min_height: float = 0.15,
        max_height: float = 2.0,
        agent_radius: float = 0.3,
        area_thresh: float = 3.0,  # square meters
        size: int = 5000,
        pixels_per_meter: int = 100,  # Match nav_agent's scale
    ):
        """
        Parameters
        ----------
        min_height : float
            Minimum obstacle height in meters
        max_height : float
            Maximum obstacle height in meters
        agent_radius : float
            Robot radius in meters for inflation
        area_thresh : float
            Minimum frontier area in square meters
        size : int
            Map size in pixels
        pixels_per_meter : int
            Map resolution (pixels per meter)
        """
        self.size = size
        self.pixels_per_meter = pixels_per_meter
        self._min_height = min_height
        self._max_height = max_height
        self._area_thresh_in_pixels = area_thresh * (pixels_per_meter ** 2)
        
        # Initialize maps
        self._obstacle_map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)
        self.explored_area = np.zeros((size, size), dtype=bool)
        
        # Compute inflation kernel for robot radius
        kernel_size = int(pixels_per_meter * agent_radius * 2)
        kernel_size = kernel_size + (kernel_size % 2 == 0)  # Make odd
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Frontier storage
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])  # In odom coordinates
        
        # Episode origin (will be set on first update)
        self._episode_origin = None
    
    def reset(self) -> None:
        """Reset all maps."""
        self._obstacle_map.fill(0)
        self._navigable_map.fill(0)
        self.explored_area.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])
        self._episode_origin = None
    
    def update_map(
        self,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        base_to_odom: np.ndarray,
        min_depth: float = 0.5,
        max_depth: float = 5.0,
    ) -> None:
        """
        Update obstacle map and explored area from depth observation.
        
        Parameters
        ----------
        depth : np.ndarray
            Depth image in meters, shape (H, W)
        intrinsic : np.ndarray
            Camera intrinsic matrix (3x3)
        extrinsic : np.ndarray
            Camera extrinsic matrix (4x4), odom->cam
        base_to_odom : np.ndarray
            Base to odom transformation (4x4)
        min_depth : float
            Minimum valid depth
        max_depth : float
            Maximum valid depth
        """
        # Set episode origin on first update
        if self._episode_origin is None:
            self._episode_origin = base_to_odom[:3, 3].copy()
        
        # 1. Generate point cloud in camera frame
        mask = (depth > min_depth) & (depth < max_depth) & np.isfinite(depth)
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        point_cloud_camera = get_point_cloud(depth, mask, fx, fy)
        
        if len(point_cloud_camera) == 0:
            return
        
        # 2. Transform to odom frame
        # cam->odom = base->odom @ cam->base
        T_cam_odom = extrinsic  # This is odom->cam, need to invert
        T_odom_cam = np.linalg.inv(T_cam_odom)
        point_cloud_odom = transform_points(T_odom_cam, point_cloud_camera)
        
        # 3. Filter by height (z coordinate in odom frame)
        obstacle_cloud = filter_points_by_height(
            point_cloud_odom, self._min_height, self._max_height
        )
        
        # 4. Project to 2D map
        if len(obstacle_cloud) > 0:
            xy_points = obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)
            
            # Mark obstacles
            valid_mask = (
                (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < self.size) &
                (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < self.size)
            )
            valid_pixels = pixel_points[valid_mask]
            self._obstacle_map[valid_pixels[:, 1], valid_pixels[:, 0]] = True
            
            # Update navigable map (inverse of dilated obstacles)
            self._navigable_map = ~cv2.dilate(
                self._obstacle_map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1
            ).astype(bool)
        
        # 5. Update explored area using FOV
        agent_pixel = self._xy_to_px(base_to_odom[:2, 3].reshape(1, 2))[0]
        agent_yaw = extract_yaw(base_to_odom)
        
        # Calculate FOV from intrinsics
        H, W = depth.shape
        fov_horizontal = 2 * np.arctan(W / (2 * fx))
        fov_deg = np.rad2deg(fov_horizontal)
        
        new_explored = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._obstacle_map, dtype=np.uint8),
            current_point=(agent_pixel[1], agent_pixel[0]),  # (row, col)
            current_angle=-agent_yaw,
            fov=fov_deg,
            max_line_len=int(max_depth * self.pixels_per_meter),
        )
        
        # Dilate slightly and merge
        new_explored = cv2.dilate(new_explored, np.ones((3, 3), np.uint8), iterations=1)
        self.explored_area[new_explored > 0] = True
        self.explored_area[~self._navigable_map] = False
        
        # Keep only the connected component containing the agent
        self._filter_explored_area(agent_pixel)
        
        # 6. Detect frontiers
        self._update_frontiers()
    
    def _filter_explored_area(self, agent_pixel: np.ndarray) -> None:
        """Keep only the explored area connected to the agent."""
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        
        if len(contours) > 1:
            # Find contour containing agent
            best_idx = 0
            min_dist = np.inf
            
            for idx, cnt in enumerate(contours):
                dist = cv2.pointPolygonTest(
                    cnt, tuple(agent_pixel.astype(int)), True
                )
                if dist >= 0:
                    best_idx = idx
                    break
                elif abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_idx = idx
            
            # Keep only the best contour
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, contours, best_idx, 1, -1)
            self.explored_area = new_area.astype(bool)
    
    def _update_frontiers(self) -> None:
        """Detect frontier waypoints."""
        # Dilate explored area slightly to prevent small gaps
        explored_dilated = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1
        )
        
        # Detect frontiers
        frontiers_px = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            explored_dilated,
            self._area_thresh_in_pixels,
        )
        
        self._frontiers_px = frontiers_px
        
        # Convert to odom coordinates
        if len(frontiers_px) > 0:
            self.frontiers = self._px_to_xy(frontiers_px)
        else:
            self.frontiers = np.array([])
    
    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        """
        Convert odom coordinates to pixel coordinates.
        
        Parameters
        ----------
        points : np.ndarray
            Nx2 array of (x, y) in odom frame
            
        Returns
        -------
        np.ndarray
            Nx2 array of (px_x, px_y) pixel coordinates
        """
        if self._episode_origin is None:
            origin = np.array([0.0, 0.0])
        else:
            origin = self._episode_origin[:2]
        
        # Relative to origin
        rel_points = points - origin
        
        # Convert to pixels (y-axis flipped for image coordinates)
        px = np.zeros_like(points, dtype=int)
        px[:, 0] = (self.size // 2 + rel_points[:, 0] * self.pixels_per_meter).astype(int)
        px[:, 1] = (self.size // 2 - rel_points[:, 1] * self.pixels_per_meter).astype(int)
        
        return px
    
    def _px_to_xy(self, px: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates to odom coordinates.
        
        Parameters
        ----------
        px : np.ndarray
            Nx2 array of (px_x, px_y) pixel coordinates
            
        Returns
        -------
        np.ndarray
            Nx2 array of (x, y) in odom frame
        """
        if self._episode_origin is None:
            origin = np.array([0.0, 0.0])
        else:
            origin = self._episode_origin[:2]
        
        # Convert from pixels
        rel_points = np.zeros_like(px, dtype=float)
        rel_points[:, 0] = (px[:, 0] - self.size // 2) / self.pixels_per_meter
        rel_points[:, 1] = -(px[:, 1] - self.size // 2) / self.pixels_per_meter
        
        # Add origin
        points = rel_points + origin
        
        return points
    
    def visualize(self, robot_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate visualization of the map.
        
        Parameters
        ----------
        robot_pos : np.ndarray, optional
            Current robot position (x, y) in odom frame
            
        Returns
        -------
        np.ndarray
            RGB visualization image
        """
        vis_img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 255
        
        # Draw explored area in light green
        vis_img[self.explored_area] = (200, 255, 200)
        
        # Draw unnavigable areas in gray
        vis_img[~self._navigable_map] = (100, 100, 100)
        
        # Draw obstacles in black
        vis_img[self._obstacle_map] = (0, 0, 0)
        
        # Draw frontiers in blue
        for frontier_px in self._frontiers_px:
            cv2.circle(vis_img, tuple(frontier_px.astype(int)), 5, (200, 0, 0), 2)
        
        # Draw robot position in red
        if robot_pos is not None:
            robot_px = self._xy_to_px(robot_pos.reshape(1, 2))[0]
            cv2.circle(vis_img, tuple(robot_px.astype(int)), 10, (0, 0, 255), -1)
        
        return vis_img
