#!/usr/bin/env python3
"""
Geometry utility functions adapted from VLFM.
"""

import numpy as np


def compress_traj3d_to_2d(traj_3d: np.ndarray, min_waypoint_dist: float = 0.35) -> list[dict]:
    """
    Compress a local 3D trajectory into sparse 2D waypoints with heading.

    Parameters
    ----------
    traj_3d : np.ndarray
        Trajectory points with shape (N, 3) in odom frame.
    min_waypoint_dist : float
        Minimum Euclidean distance (meters) between consecutive kept waypoints.

    Returns
    -------
    list[dict]
        List of waypoints formatted as
        [{'x': float, 'y': float, 'yaw': float}, ...].
    """
    if traj_3d is None or len(traj_3d) == 0:
        return []

    traj = np.asarray(traj_3d, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError("traj_3d must have shape (N, 3) or (N, >=2)")

    keep_indices = [0]
    last_xy = traj[0, :2]
    for i in range(1, traj.shape[0]):
        cur_xy = traj[i, :2]
        if np.linalg.norm(cur_xy - last_xy) >= min_waypoint_dist:
            keep_indices.append(i)
            last_xy = cur_xy

    if keep_indices[-1] != traj.shape[0] - 1:
        keep_indices.append(traj.shape[0] - 1)

    sparse = traj[keep_indices, :2]
    waypoints = []
    for i in range(sparse.shape[0]):
        if i < sparse.shape[0] - 1:
            delta = sparse[i + 1] - sparse[i]
        elif sparse.shape[0] > 1:
            delta = sparse[i] - sparse[i - 1]
        else:
            delta = np.array([1.0, 0.0], dtype=np.float64)
        yaw = float(np.arctan2(delta[1], delta[0]))
        waypoints.append({'x': float(sparse[i, 0]), 'y': float(sparse[i, 1]), 'yaw': yaw})
    return waypoints


def extract_yaw(tf_matrix: np.ndarray) -> float:
    """
    Extract yaw angle from 4x4 transformation matrix.
    
    Parameters
    ----------
    tf_matrix : np.ndarray
        4x4 transformation matrix
        
    Returns
    -------
    float
        Yaw angle in radians
    """
    return np.arctan2(tf_matrix[1, 0], tf_matrix[0, 0])


def get_point_cloud(depth: np.ndarray, mask: np.ndarray, fx: float, fy: float, cx: float = None, cy: float = None) -> np.ndarray:
    """
    Convert depth image to 3D point cloud in camera frame.
    
    Parameters
    ----------
    depth : np.ndarray
        Depth image in meters, shape (H, W)
    mask : np.ndarray
        Boolean mask indicating valid pixels, shape (H, W)
    fx : float
        Focal length in x direction (pixels)
    fy : float
        Focal length in y direction (pixels)
    cx : float, optional
        Principal point x coordinate (pixels). If None, uses image center.
    cy : float, optional
        Principal point y coordinate (pixels). If None, uses image center.
        
    Returns
    -------
    np.ndarray
        Point cloud in camera frame, shape (N, 3)
    """
    H, W = depth.shape
    
    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Apply mask
    u = u[mask]
    v = v[mask]
    z = depth[mask]
    
    # Use provided principal point or default to image center
    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0
    
    # Unproject to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into Nx3 array
    points = np.stack([x, y, z], axis=1)
    
    return points


def transform_points(tf_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Transform 3D points using a 4x4 transformation matrix.
    
    Parameters
    ----------
    tf_matrix : np.ndarray
        4x4 transformation matrix
    points : np.ndarray
        Nx3 array of 3D points
        
    Returns
    -------
    np.ndarray
        Transformed Nx3 array of 3D points
    """
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_homo = np.hstack([points, ones])
    
    # Apply transformation
    transformed = (tf_matrix @ points_homo.T).T
    
    # Convert back to 3D
    return transformed[:, :3]


def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    """
    Filter 3D points by height (z coordinate).
    
    Parameters
    ----------
    points : np.ndarray
        Nx3 array of 3D points
    min_height : float
        Minimum height threshold
    max_height : float
        Maximum height threshold
         Returns
    -------
    np.ndarray
        Filtered points
    """
    mask = (points[:, 2] >= min_height) & (points[:, 2] <= max_height)
    return points[mask]


def within_fov_cone(
    camera_pos: np.ndarray,
    camera_yaw: float,
    fov: float,
    max_dist: float,
    points: np.ndarray
) -> np.ndarray:
    """
    Filter points that are within the camera's FOV cone.
    
    Parameters
    ----------
    camera_pos : np.ndarray
        Camera position (x, y, z) or (x, y)
    camera_yaw : float
        Camera yaw angle in radians
    fov : float
        Field of view in radians
    max_dist : float
        Maximum distance from camera
    points : np.ndarray
        Nx3 or Nx2 array of points
        
    Returns
    -------
    np.ndarray
        Boolean mask of points within FOV
    """
    # Use only x, y coordinates
    camera_xy = camera_pos[:2]
    points_xy = points[:, :2]
    
    # Vector from camera to each point
    vectors = points_xy - camera_xy
    distances = np.linalg.norm(vectors, axis=1)
    
    # Filter by distance
    dist_mask = distances <= max_dist
    
    # Calculate angles
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle_diff = np.abs(np.arctan2(np.sin(angles - camera_yaw), np.cos(angles - camera_yaw)))
    
    # Filter by FOV
    fov_mask = angle_diff <= fov / 2
    
    return dist_mask & fov_mask
