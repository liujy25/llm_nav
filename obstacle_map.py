#!/usr/bin/env python3
"""
Obstacle map for frontier-based navigation.
Adapted from VLFM to work with existing nav_agent coordinate system.
"""

import cv2
import numpy as np
from typing import Optional
from skimage.graph import route_through_array

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
        self._frontier_normals = np.array([])  # Normal directions (Nx2) for visualization

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
        max_depth: float = 10.0,
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
        
        # 2. Point cloud is already in camera_link frame from get_point_cloud
        # No conversion needed - get_point_cloud returns camera_link coordinates
        # Camera_link frame: X=forward, Y=left, Z=up
        point_cloud_camera = np.zeros_like(point_cloud_optical)
        point_cloud_camera[:, 0] = point_cloud_optical[:, 0]   # X (forward)
        point_cloud_camera[:, 1] = point_cloud_optical[:, 1]   # Y (left)
        point_cloud_camera[:, 2] = point_cloud_optical[:, 2]   # Z (up)
        
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
        # self._save_point_cloud_pcd(point_cloud_odom, "point_cloud_odom.pcd")
        
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
        # if len(obstacle_cloud) > 0:
        #     self._save_point_cloud_pcd(obstacle_cloud, "obstacle_cloud_odom.pcd")

        # 4. Create FOV mask BEFORE updating obstacle map (needed for incremental update)
        robot_pixel = self._xy_to_px(base_to_odom[:2, 3].reshape(1, 2))[0]
        robot_pos_odom = base_to_odom[:2, 3]
        robot_yaw = extract_yaw(base_to_odom)
        H, W = depth.shape
        fx = intrinsic[0, 0]
        fov_horizontal = 2 * np.arctan(W / (2 * fx))
        max_depth_px = int(max_depth * self.pixels_per_meter)

        current_fov_mask = self._create_fov_mask(
            robot_pos_odom=robot_pos_odom,
            robot_yaw=robot_yaw,
            fov_rad=fov_horizontal,
            max_depth=max_depth
        )

        # 5. Incremental update of obstacle map
        # IMPORTANT: Once a voxel is marked as obstacle (black), it stays obstacle
        # Only clear non-obstacle areas in FOV for re-observation
        # This prevents obstacles from being erased by later observations
        print(f"[ObstacleMap] Updating obstacles in FOV (preserving existing obstacles)")
        # Do NOT clear obstacles: self._obstacle_map[current_fov_mask] = False
        # Instead, we only ADD new obstacles, never remove them

        # 6. Project to 2D map and mark new obstacles
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

            # Force robot position to be navigable (robot itself may be detected as obstacle)
            robot_pixel = self._xy_to_px(base_to_odom[:2, 3].reshape(1, 2))[0]
            if 0 <= robot_pixel[0] < self.size and 0 <= robot_pixel[1] < self.size:
                # Mark a small area around robot as navigable (5x5 pixels = ~0.05m radius)
                x_min = max(0, robot_pixel[0] - 2)
                x_max = min(self.size, robot_pixel[0] + 3)
                y_min = max(0, robot_pixel[1] - 2)
                y_max = min(self.size, robot_pixel[1] + 3)
                self._navigable_map[y_min:y_max, x_min:x_max] = True
                print(f"[ObstacleMap] Forced robot position {robot_pixel} to be navigable")
        else:
            print("[ObstacleMap] WARNING: No obstacle points after height filtering!")

        # 7. Update explored area: combine depth observation and raycast
        # Step 7a: Mark depth-visible areas
        self._mark_explored_from_depth(point_cloud_odom)
        depth_explored_count = self.explored_area.sum()
        print(f"[ObstacleMap] Depth explored area: {depth_explored_count} pixels")

        # Step 7b: Add raycast in FOV to fill blind spots (especially near robot)
        # FOV mask and robot info already computed above
        # IMPORTANT: reveal_fog_of_war expects 1=navigable, 0=obstacle.
        # Use the inflated navigable map to reduce "light leaks" through thin wall gaps.
        raycast_topdown = self._navigable_map.astype(np.uint8)

        # DEBUG: Check obstacle map statistics
        obstacle_count = self._obstacle_map.sum()
        obstacle_in_fov = (self._obstacle_map & current_fov_mask).sum()
        print(f"[ObstacleMap] DEBUG: Total obstacles: {obstacle_count}, obstacles in FOV: {obstacle_in_fov}")

        # DEBUG: Save obstacle map for visualization
        if obstacle_in_fov > 0:
            # Check if obstacles form contours
            obstacles_in_cone_uint8 = (self._obstacle_map & current_fov_mask).astype(np.uint8)
            test_contours, _ = cv2.findContours(obstacles_in_cone_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[ObstacleMap] DEBUG: Found {len(test_contours)} obstacle contours in FOV")
            if len(test_contours) > 0:
                total_contour_pixels = sum([cv2.contourArea(cnt) for cnt in test_contours])
                print(f"[ObstacleMap] DEBUG: Total contour area: {total_contour_pixels} pixels")

        raycast_explored = reveal_fog_of_war(
            top_down_map=raycast_topdown,
            current_fog_of_war_mask=np.zeros_like(raycast_topdown, dtype=np.uint8),
            current_point=robot_pixel,
            current_angle=robot_yaw + np.pi / 2,  # Original angle convention
            fov=np.rad2deg(fov_horizontal),  # Convert to degrees
            max_line_len=max_depth_px
        )

        # DEBUG: Check raycast results
        raycast_count = raycast_explored.sum()
        print(f"[ObstacleMap] DEBUG: Raycast marked {raycast_count} pixels as explored")

        # Step 7c: Take union of depth and raycast
        unique_raycast = np.unique(raycast_explored)
        print(f"[ObstacleMap] Raycast unique values: {unique_raycast}")

        # DEBUG: Check where raycast added explored pixels
        before_raycast = self.explored_area.sum()
        combined_explored = self.explored_area | raycast_explored.astype(bool)
        raycast_added = combined_explored.sum() - depth_explored_count
        print(f"[ObstacleMap] Raycast added: {raycast_added} pixels")

        # DEBUG: Check if raycast marked areas behind obstacles
        # Areas behind obstacles should NOT be marked as explored
        raycast_only = raycast_explored.astype(bool) & ~self.explored_area
        raycast_behind_obstacles = raycast_only & self._obstacle_map
        if raycast_behind_obstacles.sum() > 0:
            print(f"[ObstacleMap] WARNING: Raycast marked {raycast_behind_obstacles.sum()} pixels INSIDE obstacles!")

        # Check if raycast marked non-navigable areas
        raycast_non_navigable = raycast_only & ~self._navigable_map
        if raycast_non_navigable.sum() > 0:
            print(f"[ObstacleMap] WARNING: Raycast marked {raycast_non_navigable.sum()} non-navigable pixels!")

        # Step 7d: Dilate to fill small gaps
        # IMPORTANT: Restrict to navigable areas BEFORE dilation to prevent crossing obstacles
        # If we dilate first, dilation will cross obstacles and mark areas behind walls
        combined_in_navigable = combined_explored & self._navigable_map

        # DEBUG: Check navigable_map vs obstacle_map relationship
        navigable_overlaps_obstacle = self._navigable_map & self._obstacle_map
        if navigable_overlaps_obstacle.sum() > 0:
            print(f"[ObstacleMap] ERROR: navigable_map overlaps with obstacle_map by {navigable_overlaps_obstacle.sum()} pixels!")

        # DEBUG: Check if combined_in_navigable contains obstacles
        combined_in_obstacles = combined_in_navigable & self._obstacle_map
        if combined_in_obstacles.sum() > 0:
            print(f"[ObstacleMap] ERROR: combined_in_navigable contains {combined_in_obstacles.sum()} obstacle pixels!")

        kernel = np.ones((3, 3), np.uint8)  # Reduced from (5,5) to (3,3)

        # DEBUG: Check before dilation
        before_dilate = combined_in_navigable.sum()
        print(f"[ObstacleMap] DEBUG: Before dilation: {before_dilate} pixels in navigable area")

        # explored_dilated = cv2.dilate(
        #     combined_in_navigable.astype(np.uint8),
        #     kernel,
        #     iterations=1
        # ).astype(bool)

        explored_dilated = combined_in_navigable

        # Hard clamp after dilation so explored area never crosses obstacles/walls.
        explored_dilated = explored_dilated & self._navigable_map

        unique_dilated = np.unique(explored_dilated.astype(np.uint8))
        print(f"[ObstacleMap] Dilated unique values: {unique_dilated}")

        # DEBUG: Check dilation effect
        after_dilate = explored_dilated.sum()
        dilate_added = after_dilate - before_dilate
        print(f"[ObstacleMap] DEBUG: Dilation added {dilate_added} pixels")

        # Check if dilation crossed obstacles (should be much less now)
        dilated_only = explored_dilated & ~combined_in_navigable
        dilated_into_obstacles = dilated_only & self._obstacle_map
        if dilated_into_obstacles.sum() > 0:
            print(f"[ObstacleMap] WARNING: Dilation extended {dilated_into_obstacles.sum()} pixels INTO obstacles!")

        # Check if dilation crossed into non-navigable areas (should be minimal now)
        dilated_non_navigable = dilated_only & ~self._navigable_map
        if dilated_non_navigable.sum() > 0:
            print(f"[ObstacleMap] WARNING: Dilation extended {dilated_non_navigable.sum()} pixels into non-navigable areas!")
            # This should be ZERO now since we dilate within navigable_map
            print(f"[ObstacleMap] ERROR: This should not happen! Dilation should stay within navigable_map!")

        # Step 7e: Clip dilated areas to current FOV
        explored_clipped = explored_dilated & current_fov_mask
        clipped_count = explored_dilated.sum() - explored_clipped.sum()
        if clipped_count > 0:
            print(f"[ObstacleMap] Clipped {clipped_count} dilated pixels outside FOV")

        # Step 7f: Incremental update - merge current FOV observations with global map
        # Only keep explored pixels in navigable areas
        old_explored_count = self.explored_area.sum()

        # DEBUG: Check what will be added
        new_explored_in_fov = explored_clipped & self._navigable_map
        new_explored_count = new_explored_in_fov.sum()
        print(f"[ObstacleMap] DEBUG: Will add {new_explored_count} new explored pixels in FOV")

        # Check if new explored areas overlap with obstacles
        new_explored_in_obstacles = new_explored_in_fov & self._obstacle_map
        if new_explored_in_obstacles.sum() > 0:
            print(f"[ObstacleMap] ERROR: Trying to mark {new_explored_in_obstacles.sum()} obstacle pixels as explored!")

        self.explored_area = self.explored_area | (explored_clipped & self._navigable_map)

        # Keep only the connected explored component around the robot to avoid
        # isolated white regions that appear behind walls.
        self._filter_explored_area(robot_pixel)

        final_added = self.explored_area.sum() - old_explored_count
        print(f"[ObstacleMap] DEBUG: Actually added {final_added} explored pixels to global map")
        print(f"[ObstacleMap] Final explored area: {self.explored_area.sum()} / {self.explored_area.size} pixels")

        # 8. Detect frontiers (with FOV filtering)
        self._update_frontiers(
            robot_pos_odom=robot_pos_odom,
            robot_yaw=robot_yaw,
            fov_rad=fov_horizontal,
            fov_max_depth=max_depth,
        )

        print(f"[ObstacleMap] Detected {len(self.frontiers)} frontiers")

    def _mark_explored_from_depth(self, point_cloud_odom: np.ndarray) -> None:
        """
        Mark all areas visible in depth observation as explored.

        This directly marks pixels that appear in the depth image, rather than
        using raycast approximation. Areas behind obstacles are correctly marked
        as explored if they are visible in the depth image.

        Parameters
        ----------
        point_cloud_odom : np.ndarray
            Nx3 array of points in odom frame from depth camera
        """
        if len(point_cloud_odom) == 0:
            print("[ObstacleMap] No points in point cloud, skipping explored area update")
            return

        # Project all depth points to 2D map coordinates
        xy_points = point_cloud_odom[:, :2]
        pixel_points = self._xy_to_px(xy_points)

        # Filter valid pixels (within map bounds)
        valid_mask = (
            (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] < self.size) &
            (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] < self.size)
        )
        valid_pixels = pixel_points[valid_mask]

        if len(valid_pixels) == 0:
            print("[ObstacleMap] No valid pixels after filtering, skipping explored area update")
            return

        # Mark these pixels as explored
        self.explored_area[valid_pixels[:, 1], valid_pixels[:, 0]] = True

        print(f"[ObstacleMap] Marked {len(valid_pixels)} pixels as explored from depth observation")

    def _create_fov_mask(
        self,
        robot_pos_odom: np.ndarray,
        robot_yaw: float,
        fov_rad: float,
        max_depth: float
    ) -> np.ndarray:
        """
        Create a binary mask for the robot's field of view (sector shape).

        Parameters
        ----------
        robot_pos_odom : np.ndarray
            Robot position in odom frame [x, y]
        robot_yaw : float
            Robot yaw angle in radians (odom frame convention)
        fov_rad : float
            Field of view in radians (horizontal FOV)
        max_depth : float
            Maximum depth range in meters

        Returns
        -------
        np.ndarray
            Binary mask of shape (size, size) where True = within FOV
        """
        # Create empty mask
        fov_mask = np.zeros((self.size, self.size), dtype=np.uint8)

        # Convert robot position to pixel coordinates
        robot_px = self._xy_to_px(robot_pos_odom.reshape(1, 2))[0]
        robot_px_x, robot_px_y = int(robot_px[0]), int(robot_px[1])

        # Check if robot is within map bounds
        if not (0 <= robot_px_x < self.size and 0 <= robot_px_y < self.size):
            print(f"[ObstacleMap] WARNING: Robot position {robot_px} outside map bounds, returning empty FOV mask")
            return fov_mask.astype(bool)

        # Convert angle to OpenCV convention
        # In odom frame: X=forward, Y=left, yaw=0 means facing +X (right in image)
        # In image frame: X=right, Y=down
        # cv2.ellipse angles: 0° = right (+X), 90° = down (+Y), measured clockwise
        #
        # Robot yaw in odom frame needs to be converted to image frame:
        # - Image Y-axis is flipped (down is positive)
        # - So robot_yaw needs to be negated for Y-axis flip
        # - Then convert to degrees
        #
        # For cv2.ellipse:
        # - startAngle and endAngle are measured clockwise from the positive X-axis (right)
        # - We need to calculate the center direction and FOV boundaries

        # Convert robot yaw to image frame angle (clockwise from right/+X)
        # In odom: yaw=0 is +X (right), yaw=90° is +Y (up in odom, but down in image due to flip)
        # In image: angle=0 is right, angle=90 is down
        # So: image_angle = -robot_yaw (in degrees), but we need to handle the Y-flip
        # Actually: image_angle = -robot_yaw * 180/π (negate due to Y-flip)
        center_angle_deg = -np.rad2deg(robot_yaw)

        # Calculate FOV boundaries
        fov_deg = np.rad2deg(fov_rad)
        start_angle = center_angle_deg - fov_deg / 2
        end_angle = center_angle_deg + fov_deg / 2

        # Calculate axes for ellipse (max_depth in pixels)
        max_depth_px = int(max_depth * self.pixels_per_meter)

        # Draw filled ellipse sector
        cv2.ellipse(
            fov_mask,
            center=(robot_px_x, robot_px_y),
            axes=(max_depth_px, max_depth_px),
            angle=0,  # Rotation of ellipse axes (0 for circle)
            startAngle=start_angle,
            endAngle=end_angle,
            color=1,
            thickness=-1  # Filled
        )

        print(f"[ObstacleMap] Created FOV mask: center=({robot_px_x}, {robot_px_y}), "
              f"radius={max_depth_px}px, center_angle={center_angle_deg:.1f}°, fov={fov_deg:.1f}°, "
              f"range=[{start_angle:.1f}°, {end_angle:.1f}°]")

        return fov_mask.astype(bool)

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
                    cnt, (int(agent_pixel[0]), int(agent_pixel[1])), True
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
    
    def _update_frontiers(
        self,
        robot_pos_odom: np.ndarray,
        robot_yaw: float,
        fov_rad: float,
        fov_expansion_deg: float = 10.0,
        fov_max_depth: float = 20.0,
    ) -> None:
        """
        Detect frontier waypoints based on global map, then filter by FOV.

        Pipeline:
        1. Global frontier detection (avoid local optimum)
        2. FOV filtering (keep only frontiers in expanded FOV)
        3. Obstacle proximity filtering
        4. Normal computation
        5. Inward movement

        Parameters
        ----------
        robot_pos_odom : np.ndarray
            Robot position in odom frame [x, y]
        robot_yaw : float
            Robot yaw angle in radians
        fov_rad : float
            Camera FOV in radians
        fov_expansion_deg : float
            Expand FOV by this many degrees for frontier selection (default: 10)
        fov_max_depth : float
            Distance limit in meters for frontier FOV filtering
        """
        # Step 1: Detect frontiers based on global map
        # Use explored_area directly without dilation
        frontiers_px = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            self.explored_area.astype(np.uint8),
            self._area_thresh_in_pixels,
        )

        print(f"[ObstacleMap] Global frontiers detected: {len(frontiers_px)}")

        # Step 2: Filter by expanded FOV
        if len(frontiers_px) > 0:
            # Create expanded FOV mask (camera FOV + expansion)
            fov_expanded_rad = fov_rad + np.deg2rad(fov_expansion_deg)
            fov_mask_expanded = self._create_fov_mask(
                robot_pos_odom=robot_pos_odom,
                robot_yaw=robot_yaw,
                fov_rad=fov_expanded_rad,
                max_depth=fov_max_depth
            )

            # Filter frontiers by FOV
            valid_in_fov = []
            for fp in frontiers_px:
                px_x, px_y = int(fp[0]), int(fp[1])
                if 0 <= px_x < self.size and 0 <= px_y < self.size:
                    if fov_mask_expanded[px_y, px_x]:
                        valid_in_fov.append(fp)

            frontiers_px = np.array(valid_in_fov) if valid_in_fov else np.array([])
            print(f"[ObstacleMap] Frontiers after FOV filtering (FOV + {fov_expansion_deg}°): {len(frontiers_px)}")

        # Step 3: Filter out frontiers that are too close to obstacles
        if len(frontiers_px) > 0:
            frontiers_px = self._filter_frontiers_near_obstacles(frontiers_px)

        # Step 4: Filter out unreachable frontiers using A* pathfinding
        if len(frontiers_px) > 0:
            robot_px = self._xy_to_px(robot_pos_odom.reshape(1, 2))[0]
            frontiers_px = self._filter_unreachable_frontiers(frontiers_px, robot_px)

        # Step 5 & 6: Move remaining frontiers inward and compute normals
        if len(frontiers_px) > 0:
            # Save original frontiers before moving (for visualization)
            self._frontiers_px_original = frontiers_px.copy()

            frontiers_px, normals = self._move_frontiers_inward(frontiers_px, self.explored_area)
            self._frontier_normals = normals
        else:
            self._frontiers_px_original = np.array([])
            self._frontier_normals = np.array([])

        self._frontiers_px = frontiers_px

        # Convert to odom coordinates
        if len(frontiers_px) > 0:
            self.frontiers = self._px_to_xy(frontiers_px)
        else:
            self.frontiers = np.array([])

        print(f"[ObstacleMap] Final frontiers: {len(self.frontiers)}")

    def get_global_frontiers(
        self,
        robot_pos_odom: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect all global frontiers without FOV or A* reachability filtering.
        Used by fallback mode to provide frontier candidates from all directions.

        Pipeline: global detection -> obstacle proximity filter -> inward move + normal

        Parameters
        ----------
        robot_pos_odom : np.ndarray
            Robot position in odom frame [x, y]

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (frontiers_odom Nx2, normals Nx2) in odom coordinates
        """
        # Step 1: Global frontier detection
        frontiers_px = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            self.explored_area.astype(np.uint8),
            self._area_thresh_in_pixels,
        )
        print(f"[ObstacleMap] Global frontiers (fallback): {len(frontiers_px)}")

        if len(frontiers_px) == 0:
            return np.array([]), np.array([])

        # Step 2: Obstacle proximity filter (skip FOV and A* filters)
        frontiers_px = self._filter_frontiers_near_obstacles(frontiers_px)
        print(f"[ObstacleMap] After obstacle filter (fallback): {len(frontiers_px)}")

        if len(frontiers_px) == 0:
            return np.array([]), np.array([])

        # Step 3: Inward move + normal computation
        frontiers_px, normals = self._move_frontiers_inward(frontiers_px, self.explored_area)

        if len(frontiers_px) == 0:
            return np.array([]), np.array([])

        # Convert to odom coordinates
        frontiers_odom = self._px_to_xy(frontiers_px)
        print(f"[ObstacleMap] Final global frontiers (fallback): {len(frontiers_odom)}")

        return frontiers_odom, normals

    def _move_frontiers_inward(
        self,
        frontiers_px: np.ndarray,
        explored_area: np.ndarray,
        move_distance: float = 0.25  # meters
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Move frontiers inward into explored area along the normal direction.

        Parameters
        ----------
        frontiers_px : np.ndarray
            Nx2 array of frontier pixel coordinates
        explored_area : np.ndarray
            Binary map of explored area (used to compute gradient)
        move_distance : float
            Distance in meters to move inward

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (adjusted frontier pixel coordinates, normal directions Nx2)
        """
        if len(frontiers_px) == 0:
            return frontiers_px, np.array([])

        # Convert move distance to pixels
        move_distance_px = move_distance * self.pixels_per_meter

        # Compute gradient of explored area (normal direction)
        # Gradient points from 0 (unexplored) to 1 (explored)
        gradient_x = cv2.Sobel(explored_area.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(explored_area.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)

        adjusted_frontiers = []
        normals = []
        moved_count = 0

        for frontier_px in frontiers_px:
            fx, fy = int(frontier_px[0]), int(frontier_px[1])

            # Check bounds
            if not (0 <= fx < self.size and 0 <= fy < self.size):
                adjusted_frontiers.append(frontier_px)
                normals.append([0.0, 0.0])  # Zero normal for invalid frontier
                continue

            # Get gradient at frontier position
            gx = gradient_x[fy, fx]
            gy = gradient_y[fy, fx]

            # Compute normal magnitude
            norm = np.sqrt(gx**2 + gy**2)

            if norm < 1e-6:
                # No clear gradient, keep original position
                adjusted_frontiers.append(frontier_px)
                normals.append([0.0, 0.0])  # Zero normal
                continue

            # Normalize gradient to get unit normal
            normal_x = gx / norm
            normal_y = gy / norm

            # Save the normal direction
            normals.append([normal_x, normal_y])

            # Move along normal direction
            new_fx = fx + normal_x * move_distance_px
            new_fy = fy + normal_y * move_distance_px

            # Check if new position is valid
            new_fx_int = int(np.round(new_fx))
            new_fy_int = int(np.round(new_fy))

            # Validate new position
            if (0 <= new_fx_int < self.size and 0 <= new_fy_int < self.size and
                self._navigable_map[new_fy_int, new_fx_int]):
                # New position is valid
                adjusted_frontiers.append([new_fx, new_fy])
                moved_count += 1
            else:
                # Keep original position if new position is invalid
                adjusted_frontiers.append(frontier_px)

        if moved_count > 0:
            print(f"[ObstacleMap] Moved {moved_count}/{len(frontiers_px)} frontiers inward by {move_distance}m")

        return np.array(adjusted_frontiers), np.array(normals)

    def _filter_frontiers_near_obstacles(
        self,
        frontiers_px: np.ndarray,
        obstacle_check_radius: float = 0.0  # meters
    ) -> np.ndarray:
        """
        Filter out frontiers that have obstacles nearby.

        Parameters
        ----------
        frontiers_px : np.ndarray
            Nx2 array of frontier pixel coordinates
        obstacle_check_radius : float
            Radius in meters to check for obstacles around each frontier

        Returns
        -------
        np.ndarray
            Filtered frontier pixel coordinates
        """
        if len(frontiers_px) == 0:
            return frontiers_px

        # Convert radius to pixels
        check_radius_px = int(obstacle_check_radius * self.pixels_per_meter)

        # Create a mask for checking obstacles
        # Use the obstacle map (not navigable map, to check actual obstacles)
        obstacle_map = ~self._navigable_map

        valid_frontiers = []
        filtered_count = 0

        for frontier_px in frontiers_px:
            fx, fy = int(frontier_px[0]), int(frontier_px[1])

            # Define region to check
            x_min = max(0, fx - check_radius_px)
            x_max = min(self.size, fx + check_radius_px + 1)
            y_min = max(0, fy - check_radius_px)
            y_max = min(self.size, fy + check_radius_px + 1)

            # Check if there are obstacles in the region
            region = obstacle_map[y_min:y_max, x_min:x_max]
            has_obstacles = region.sum() > 0

            if not has_obstacles:
                valid_frontiers.append(frontier_px)
            else:
                filtered_count += 1

        if filtered_count > 0:
            print(f"[ObstacleMap] Filtered {filtered_count} frontiers near obstacles (radius={obstacle_check_radius}m)")

        return np.array(valid_frontiers) if len(valid_frontiers) > 0 else np.array([])

    def _filter_unreachable_frontiers(
        self,
        frontiers_px: np.ndarray,
        robot_px: np.ndarray
    ) -> np.ndarray:
        """
        Filter out frontiers that are not reachable from robot position.
        Uses A* pathfinding on explored + navigable areas.

        Parameters
        ----------
        frontiers_px : np.ndarray
            Nx2 array of frontier pixel coordinates
        robot_px : np.ndarray
            Robot position in pixel coordinates [px_x, px_y]

        Returns
        -------
        np.ndarray
            Filtered frontier pixel coordinates (only reachable ones)
        """
        if len(frontiers_px) == 0:
            return frontiers_px

        # Create cost map for pathfinding
        # Traversable areas: explored AND navigable (white areas in BEV)
        traversable = self.explored_area & self._navigable_map

        # Create cost map: 1 for traversable, very high cost for non-traversable
        # route_through_array finds minimum cost path
        cost_map = np.where(traversable, 1.0, 1e10)

        robot_px_int = (int(robot_px[1]), int(robot_px[0]))  # (row, col) for route_through_array

        reachable_frontiers = []
        unreachable_count = 0

        for frontier_px in frontiers_px:
            frontier_px_int = (int(frontier_px[1]), int(frontier_px[0]))  # (row, col)

            # Check bounds
            if not (0 <= frontier_px_int[0] < self.size and 0 <= frontier_px_int[1] < self.size):
                unreachable_count += 1
                continue

            if not (0 <= robot_px_int[0] < self.size and 0 <= robot_px_int[1] < self.size):
                unreachable_count += 1
                continue

            try:
                # Use route_through_array to find path
                # Returns (indices, weight) where indices is list of (row, col) tuples
                path, cost = route_through_array(
                    cost_map,
                    robot_px_int,
                    frontier_px_int,
                    fully_connected=True  # Allow diagonal movement
                )

                # If cost is reasonable (not hitting the high cost barrier), it's reachable
                if cost < 1e9:  # Much less than 1e10 barrier
                    reachable_frontiers.append(frontier_px)
                else:
                    unreachable_count += 1

            except Exception as e:
                # If pathfinding fails, consider unreachable
                print(f"[ObstacleMap] Pathfinding failed for frontier {frontier_px}: {e}")
                unreachable_count += 1

        if unreachable_count > 0:
            print(f"[ObstacleMap] Filtered {unreachable_count} unreachable frontiers (A* pathfinding)")

        return np.array(reachable_frontiers) if len(reachable_frontiers) > 0 else np.array([])

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
    
    def visualize(self, robot_pos: Optional[np.ndarray] = None, show_frontiers: bool = True, show_normals: bool = False) -> np.ndarray:
        """
        Generate visualization of the map.

        Parameters
        ----------
        robot_pos : np.ndarray, optional
            Current robot position (x, y) in odom frame
        show_frontiers : bool, optional
            Whether to show frontier points (default: True)
        show_normals : bool, optional
            Whether to show frontier normal arrows (default: False)

        Returns
        -------
        np.ndarray
            RGB visualization image
        """
        # Initialize with gray (unexplored area)
        vis_img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 128

        # Draw explored area in white
        vis_img[self.explored_area] = (255, 255, 255)

        # Draw inflated obstacles in black (non-navigable areas)
        vis_img[~self._navigable_map] = (0, 0, 0)

        # Draw frontiers in blue (RGB format) - only if show_frontiers is True
        if show_frontiers and len(self._frontiers_px) > 0:
            for frontier_px in self._frontiers_px:
                cv2.circle(vis_img, tuple(frontier_px.astype(int)), 5, (0, 0, 200), 2)

        # Draw frontier normals as arrows
        if show_normals and len(self._frontiers_px) > 0 and len(self._frontier_normals) > 0:
            print(f"[ObstacleMap.visualize] Drawing normals for {len(self._frontiers_px)} frontiers")

            arrow_length_px = 30  # pixels
            arrows_drawn = 0

            for frontier_px, normal in zip(self._frontiers_px, self._frontier_normals):
                fx, fy = int(frontier_px[0]), int(frontier_px[1])

                if not (0 <= fx < self.size and 0 <= fy < self.size):
                    print(f"[ObstacleMap.visualize] Frontier ({fx}, {fy}) out of bounds")
                    continue

                # Check if normal is valid (non-zero)
                normal_x, normal_y = normal[0], normal[1]
                norm = np.sqrt(normal_x**2 + normal_y**2)

                if norm < 1e-6:
                    print(f"[ObstacleMap.visualize] Frontier ({fx}, {fy}) has zero normal")
                    continue

                # Arrow end point (use negative normal to point towards unexplored area)
                end_x = int(fx - normal_x * arrow_length_px)
                end_y = int(fy - normal_y * arrow_length_px)

                # Draw arrow (yellow color in RGB)
                cv2.arrowedLine(vis_img, (fx, fy), (end_x, end_y),
                               (255, 255, 0), 2, tipLength=0.3)
                arrows_drawn += 1
                print(f"[ObstacleMap.visualize] Drew arrow at ({fx}, {fy}) -> ({end_x}, {end_y}), normal=({normal_x:.2f}, {normal_y:.2f})")

            print(f"[ObstacleMap.visualize] Drew {arrows_drawn}/{len(self._frontiers_px)} arrows")

        # Draw robot position in orange-red (RGB format) with black outline
        if robot_pos is not None:
            robot_px = self._xy_to_px(robot_pos.reshape(1, 2))[0]
            print(f"[ObstacleMap.visualize] Robot pos (odom): {robot_pos}, pixel: {robot_px}, map size: {self.size}")

            # Check if robot is within map bounds
            if 0 <= robot_px[0] < self.size and 0 <= robot_px[1] < self.size:
                # Draw orange-red filled circle (BGR: 0, 69, 255)
                cv2.circle(vis_img, tuple(robot_px.astype(int)), 4, (0, 69, 255), -1)
                # Draw black outline
                cv2.circle(vis_img, tuple(robot_px.astype(int)), 4, (0, 0, 0), 1)
                print(f"[ObstacleMap.visualize] Robot drawn at pixel {robot_px}")
            else:
                print(f"[ObstacleMap.visualize] WARNING: Robot position {robot_px} outside map bounds [0, {self.size}]")

        return vis_img

    def visualize_frontiers(self, robot_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate visualization showing all detected frontiers.

        Parameters
        ----------
        robot_pos : np.ndarray, optional
            Current robot position (x, y) in odom frame

        Returns
        -------
        np.ndarray
            RGB visualization image showing frontiers
        """
        # Initialize with gray (unexplored area)
        vis_img = np.ones((self.size, self.size, 3), dtype=np.uint8) * 128

        # Draw explored area in white
        vis_img[self.explored_area] = (255, 255, 255)

        # Draw obstacles in black
        vis_img[~self._navigable_map] = (0, 0, 0)

        # Draw all frontiers in bright cyan with numbers
        if len(self._frontiers_px) > 0:
            for i, frontier_px in enumerate(self._frontiers_px, start=1):
                fx, fy = int(frontier_px[0]), int(frontier_px[1])

                # Draw cyan circle
                cv2.circle(vis_img, (fx, fy), 8, (0, 255, 255), -1)  # Cyan filled
                cv2.circle(vis_img, (fx, fy), 8, (0, 0, 0), 2)  # Black outline

                # Draw number
                cv2.putText(vis_img, str(i), (fx-6, fy+6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Draw frontier normals as arrows
        if len(self._frontiers_px) > 0 and len(self._frontier_normals) > 0:
            arrow_length_px = 40
            for frontier_px, normal in zip(self._frontiers_px, self._frontier_normals):
                fx, fy = int(frontier_px[0]), int(frontier_px[1])

                # Check if normal is valid
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    # Arrow pointing towards unexplored (negative normal direction)
                    end_x = int(fx - normal[0] * arrow_length_px)
                    end_y = int(fy - normal[1] * arrow_length_px)

                    # Draw yellow arrow
                    cv2.arrowedLine(vis_img, (fx, fy), (end_x, end_y),
                                   (255, 255, 0), 2, tipLength=0.3)

        # Draw robot position in red
        if robot_pos is not None:
            robot_px = self._xy_to_px(robot_pos.reshape(1, 2))[0]
            if 0 <= robot_px[0] < self.size and 0 <= robot_px[1] < self.size:
                cv2.circle(vis_img, tuple(robot_px.astype(int)), 6, (255, 0, 0), -1)  # Red filled
                cv2.circle(vis_img, tuple(robot_px.astype(int)), 6, (0, 0, 0), 2)  # Black outline

        return vis_img
