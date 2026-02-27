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
            print(f"[ObstacleMap] Episode origin set: {self._episode_origin}")
        
        # 1. Generate point cloud in camera frame (OpenCV optical frame)
        mask = (depth > min_depth) & (depth < max_depth) & np.isfinite(depth)
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        point_cloud_optical = get_point_cloud(depth, mask, fx, fy, cx, cy)
        
        print(f"[ObstacleMap] Depth shape: {depth.shape}, valid points: {mask.sum()}, point cloud size: {len(point_cloud_optical)}")
        
        if len(point_cloud_optical) == 0:
            print("[ObstacleMap] WARNING: No valid points in point cloud!")
            return
        
        # 2. Convert from optical frame to camera_link frame
        # OpenCV optical: X=right, Y=down, Z=forward
        # ROS camera_link: X=forward, Y=left, Z=up
        # Transformation: [X_link, Y_link, Z_link] = [Z_optical, -X_optical, -Y_optical]
        point_cloud_camera = np.zeros_like(point_cloud_optical)
        point_cloud_camera[:, 0] = point_cloud_optical[:, 0]   # X_link = Z_optical (forward)
        point_cloud_camera[:, 1] = point_cloud_optical[:, 1]  # Y_link = -X_optical (left)
        point_cloud_camera[:, 2] = point_cloud_optical[:, 2]  # Z_link = -Y_optical (up)
        
        print(f"[ObstacleMap] Converted from optical frame to camera_link frame")
        
        # 3. Transform to odom frame
        # cam->odom = base->odom @ cam->base
        T_cam_odom = extrinsic  # This is odom->cam_link, need to invert
        T_odom_cam = np.linalg.inv(T_cam_odom)
        
        # Debug: print transformation matrices
        print(f"[ObstacleMap] T_cam_odom (odom->cam):\n{T_cam_odom}")
        print(f"[ObstacleMap] T_odom_cam (cam->odom):\n{T_odom_cam}")
        print(f"[ObstacleMap] base_to_odom:\n{base_to_odom}")
        
        # Sample a few camera points for debugging
        if len(point_cloud_camera) > 0:
            sample_idx = len(point_cloud_camera) // 2
            sample_cam = point_cloud_camera[sample_idx]
            print(f"[ObstacleMap] Sample point in camera frame: {sample_cam}")
        
        point_cloud_odom = transform_points(T_odom_cam, point_cloud_camera)
        
        # Print the transformed sample point
        if len(point_cloud_odom) > 0:
            sample_odom = point_cloud_odom[sample_idx]
            print(f"[ObstacleMap] Sample point in odom frame: {sample_odom}")
        
        # Print z range for debugging
        if len(point_cloud_odom) > 0:
            z_min = point_cloud_odom[:, 2].min()
            z_max = point_cloud_odom[:, 2].max()
            print(f"[ObstacleMap] Point cloud odom z range: [{z_min:.3f}, {z_max:.3f}]")
        
        # Save point cloud to PCD file for debugging
        self._save_point_cloud_pcd(point_cloud_odom, "point_cloud_odom.pcd")
        
        # 3. Filter by height (z coordinate in odom frame)
        obstacle_cloud = filter_points_by_height(
            point_cloud_odom, self._min_height, self._max_height
        )
        
        # Print z range for filtered cloud
        if len(obstacle_cloud) > 0:
            z_min = obstacle_cloud[:, 2].min()
            z_max = obstacle_cloud[:, 2].max()
            print(f"[ObstacleMap] Obstacle cloud z range: [{z_min:.3f}, {z_max:.3f}]")
        
        # Save filtered obstacle cloud
        if len(obstacle_cloud) > 0:
            self._save_point_cloud_pcd(obstacle_cloud, "obstacle_cloud_odom.pcd")
        
        # 4. Project to 2D map
        if len(obstacle_cloud) > 0:
            xy_points = obstacle_cloud[:, :2]
            
            # Debug: print XY distribution
            print(f"[ObstacleMap] XY points - X range: [{xy_points[:, 0].min():.3f}, {xy_points[:, 0].max():.3f}]")
            print(f"[ObstacleMap] XY points - Y range: [{xy_points[:, 1].min():.3f}, {xy_points[:, 1].max():.3f}]")
            
            pixel_points = self._xy_to_px(xy_points)
            
            # Debug: print pixel distribution
            print(f"[ObstacleMap] Pixel points - X range: [{pixel_points[:, 0].min()}, {pixel_points[:, 0].max()}]")
            print(f"[ObstacleMap] Pixel points - Y range: [{pixel_points[:, 1].min()}, {pixel_points[:, 1].max()}]")
            
            # Mark obstacles
            valid_mask = (
                (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < self.size) &
                (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < self.size)
            )
            valid_pixels = pixel_points[valid_mask]
            self._obstacle_map[valid_pixels[:, 1], valid_pixels[:, 0]] = True
            
            print(f"[ObstacleMap] Obstacle cloud: {len(obstacle_cloud)} points, valid pixels: {len(valid_pixels)}")
            print(f"[ObstacleMap] Obstacle map coverage: {self._obstacle_map.sum()} / {self._obstacle_map.size} pixels")
            
            # Update navigable map (inverse of dilated obstacles)
            self._navigable_map = ~cv2.dilate(
                self._obstacle_map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1
            ).astype(bool)
            
            print(f"[ObstacleMap] Navigable map coverage: {self._navigable_map.sum()} / {self._navigable_map.size} pixels")
        else:
            print("[ObstacleMap] WARNING: No obstacle points after height filtering!")
        
        # 5. Update explored area using FOV
        agent_pixel = self._xy_to_px(base_to_odom[:2, 3].reshape(1, 2))[0]
        agent_yaw = extract_yaw(base_to_odom)
        
        # Debug: print transformation matrices
        print(f"[ObstacleMap] base_to_odom position: {base_to_odom[:3, 3]}")
        print(f"[ObstacleMap] base_to_odom rotation:\n{base_to_odom[:3, :3]}")
        print(f"[ObstacleMap] agent_yaw (radians): {agent_yaw:.3f}, (degrees): {np.rad2deg(agent_yaw):.1f}")
        print(f"[ObstacleMap] agent_pixel: {agent_pixel}")
        
        # Debug: check if agent is in navigable area
        if 0 <= agent_pixel[0] < self.size and 0 <= agent_pixel[1] < self.size:
            is_navigable = self._navigable_map[agent_pixel[1], agent_pixel[0]]
            print(f"[ObstacleMap] Agent is in navigable area: {is_navigable}")
        else:
            print(f"[ObstacleMap] WARNING: Agent pixel {agent_pixel} is outside map bounds!")
        
        # Calculate FOV from intrinsics
        H, W = depth.shape
        fov_horizontal = 2 * np.arctan(W / (2 * fx))
        fov_deg = np.rad2deg(fov_horizontal)
        print(f"[ObstacleMap] FOV: {fov_deg:.1f} degrees")
        
        # reveal_fog_of_war内部会做变换: angle_cv2 = np.rad2deg(-current_angle + π/2)
        # 我们的agent_yaw: 0度=+X方向（图像右侧）, 90度=+Y方向（图像上方）
        # reveal_fog_of_war的angle_cv2: 0度=图像右侧, 90度=图像上方（OpenCV椭圆约定）
        #
        # 推导：如果agent_yaw=0（朝向+X），我们希望angle_cv2=0（图像右侧）
        #      angle_cv2 = -current_angle + 90度
        #      0 = -current_angle + 90
        #      current_angle = 90度 = π/2
        # 因此：current_angle = π/2 - agent_yaw
        current_angle_input = np.pi / 2 - agent_yaw
        print(f"[ObstacleMap] Calling reveal_fog_of_war with angle={current_angle_input:.3f} rad ({np.rad2deg(current_angle_input):.1f} deg)")
        new_explored = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=self.explored_area.astype(np.uint8),
            current_point=np.array([agent_pixel[1], agent_pixel[0]], dtype=np.int32),  # (row, col)
            current_angle=current_angle_input,  # 修复：使用π/2 - agent_yaw
            fov=fov_deg,
            max_line_len=int(max_depth * self.pixels_per_meter),
        )
        print(f"[ObstacleMap] reveal_fog_of_war returned {new_explored.sum()} explored pixels")
        
        # Dilate slightly and merge
        new_explored = cv2.dilate(new_explored, np.ones((3, 3), np.uint8), iterations=1)
        self.explored_area[new_explored > 0] = True
        self.explored_area[~self._navigable_map] = False
        
        print(f"[ObstacleMap] Explored area: {self.explored_area.sum()} / {self.explored_area.size} pixels")
        
        # Keep only the connected component containing the agent
        self._filter_explored_area(agent_pixel)
        
        print(f"[ObstacleMap] Explored area after filtering: {self.explored_area.sum()} pixels")
        
        # 6. Detect frontiers
        self._update_frontiers()
        
        print(f"[ObstacleMap] Detected {len(self.frontiers)} frontiers")
    
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
    
    def _save_point_cloud_pcd(self, points: np.ndarray, filename: str) -> None:
        """
        Save point cloud to PCD file for debugging.
        
        Parameters
        ----------
        points : np.ndarray
            Nx3 array of (x, y, z) points
        filename : str
            Output filename
        """
        if len(points) == 0:
            print(f"[ObstacleMap] WARNING: Cannot save empty point cloud to {filename}")
            return
        
        try:
            with open(filename, 'w') as f:
                # Write PCD header
                f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                f.write("VERSION 0.7\n")
                f.write("FIELDS x y z\n")
                f.write("SIZE 4 4 4\n")
                f.write("TYPE F F F\n")
                f.write("COUNT 1 1 1\n")
                f.write(f"WIDTH {len(points)}\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {len(points)}\n")
                f.write("DATA ascii\n")
                
                # Write points
                for point in points:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
            
            print(f"[ObstacleMap] Saved {len(points)} points to {filename}")
        except Exception as e:
            print(f"[ObstacleMap] ERROR: Failed to save PCD file {filename}: {e}")
    
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
    
    def visualize(self, robot_pos: Optional[np.ndarray] = None, show_frontiers: bool = True) -> np.ndarray:
        """
        Generate visualization of the map.
        
        Parameters
        ----------
        robot_pos : np.ndarray, optional
            Current robot position (x, y) in odom frame
        show_frontiers : bool, optional
            Whether to show frontier points (default: True)
            
        Returns
        -------
        np.ndarray
            RGB visualization image
        """
        # Initialize with gray (unexplored area)
        vis_img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 128
        
        # Draw explored area in white
        vis_img[self.explored_area] = (255, 255, 255)
        
        # Draw obstacles in black
        vis_img[self._obstacle_map] = (0, 0, 0)
        
        # Draw frontiers in blue (RGB format) - only if show_frontiers is True
        if show_frontiers:
            for frontier_px in self._frontiers_px:
                cv2.circle(vis_img, tuple(frontier_px.astype(int)), 5, (0, 0, 200), 2)
        
        # Draw robot position in red (RGB format)
        if robot_pos is not None:
            robot_px = self._xy_to_px(robot_pos.reshape(1, 2))[0]
            print(f"[ObstacleMap.visualize] Robot pos (odom): {robot_pos}, pixel: {robot_px}, map size: {self.size}")
            
            # Check if robot is within map bounds
            if 0 <= robot_px[0] < self.size and 0 <= robot_px[1] < self.size:
                cv2.circle(vis_img, tuple(robot_px.astype(int)), 10, (255, 0, 0), -1)
                print(f"[ObstacleMap.visualize] Robot drawn at pixel {robot_px}")
            else:
                print(f"[ObstacleMap.visualize] WARNING: Robot position {robot_px} is outside map bounds [0, {self.size}]")
        
        return vis_img
