#!/usr/bin/env python3
"""Utilities for saving and visualizing point clouds."""

import numpy as np
from pathlib import Path


def save_ply(filename: str, points: np.ndarray, colors: np.ndarray = None, confidence: np.ndarray = None):
    """
    Save point cloud to PLY format.

    Parameters
    ----------
    filename : str
        Output PLY file path
    points : np.ndarray
        Point coordinates with shape (N, 3)
    colors : np.ndarray, optional
        RGB colors with shape (N, 3), values in [0, 255]
    confidence : np.ndarray, optional
        Confidence scores with shape (N,), values in [0, 1]
    """
    N = len(points)

    # Prepare header
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    if confidence is not None:
        header.append("property float confidence")

    header.append("end_header")

    # Write file
    with open(filename, 'w') as f:
        # Write header
        f.write('\n'.join(header) + '\n')

        # Write points
        for i in range(N):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"

            if colors is not None:
                line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"

            if confidence is not None:
                line += f" {confidence[i]:.6f}"

            f.write(line + '\n')

    print(f"Saved point cloud with {N} points to: {filename}")


def filter_points_by_confidence(points: np.ndarray, colors: np.ndarray, confidence: np.ndarray,
                                 threshold: float = 0.5) -> tuple:
    """
    Filter points by confidence threshold.

    Parameters
    ----------
    points : np.ndarray
        Point coordinates with shape (N, 3)
    colors : np.ndarray
        RGB colors with shape (N, 3)
    confidence : np.ndarray
        Confidence scores with shape (N,)
    threshold : float
        Confidence threshold, default 0.5

    Returns
    -------
    tuple
        Filtered (points, colors, coce)
    """
    mask = confidence >= threshold
    return points[mask], colors[mask], confidence[mask]


def downsample_points(points: np.ndarray, colors: np.ndarray, confidence: np.ndarray,
                      voxel_size: float = 0.05) -> tuple:
    """
    Downsample point cloud using voxel grid.

    Parameters
    ----------
    points : np.ndarray
        Point coordinates with shape (N, 3)
    colors : np.ndarray
        RGB colors with shape (N, 3)
    confidence : np.ndarray
        Confidence scores with shape (N,)
    voxel_size : float
        Voxel size for downsampling, default 0.05m

    Returns
    -------
    tuple
        Downsampled (points, colors, confidence)
    """
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Create unique voxel keys
    voxel_keys = {}
    for i in range(len(points)):
        key = tuple(voxel_indices[i])
        if key not in voxel_keys:
            voxel_keys[key] = []
        voxel_keys[key].append(i)

    # Average points in each voxel
    downsampled_points = []
    downsampled_colors = []
    downsampled_conf = []

    for indices in voxel_keys.values():
        downsampled_points.append(points[indices].mean(axis=0))
        downsampled_colors.append(colors[indices].mean(axis=0))
        downsampled_conf.append(confidence[indices].mean())

    return (
        np.array(downsampled_points),
        np.array(downsampled_colors),
        np.array(downsampled_conf)
    )
