import base64
import io
import logging
import os
import cv2
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import transform_utils as T  # 你自己的 T.pose_inv 等

# BGR
GREEN = (0, 255, 0)
RED   = (0, 0, 255) 
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY  = (200, 200, 200)


def _as_points_N3(x) -> np.ndarray:
    """Ensure array is (N,3)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape == (3,):
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"Expected (3,) or (N,3), got {x.shape}")
    return x


def point_to_pixel(pt, intrinsics, extrinsics):
    """
    Project 3D points -> pixels using pinhole model.

    pt         : (3,) or (N,3) points in WORLD/ODOM frame
    intrinsics : (3,3) K
    extrinsics : (4,4) T_cam_world (WORLD/ODOM -> CAMERA)

    Return:
      (N,2) pixels for valid points (z>0), OR None if all invalid.
    """
    pt = _as_points_N3(pt)
    K = np.asarray(intrinsics, dtype=np.float64)
    T_cam_w = np.asarray(extrinsics, dtype=np.float64)

    homo = np.hstack((pt, np.ones((pt.shape[0], 1), dtype=np.float64)))  # (N,4)
    cam = (T_cam_w @ homo.T)[:3, :]  # (3,N)

    z = cam[2, :]
    valid = z > 1e-6
    if not np.any(valid):
        return None

    pix = (K @ cam)  # (3,N)
    pix[:, valid] /= z[valid]
    return pix[:2, valid].T  # (Nv,2)


def agent_frame_to_image_coords(point, intrinsics, extrinsics):
    """
    Wrapper: project ONE point (3,) and return (u,v) int.
    point is in WORLD/ODOM frame (same as extrinsics expects).
    """
    uv = point_to_pixel(point, intrinsics, extrinsics)
    if uv is None or uv.shape[0] == 0:
        return None
    u, v = uv[0]
    return (int(round(u)), int(round(v)))


def unproject_2d(x_pixel, y_pixel, depth, intrinsics):
    """
    Back-project pixel -> camera 3D point (in camera frame).
    """
    K = np.asarray(intrinsics, dtype=np.float64)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    z = float(depth)
    x = (float(x_pixel) - cx) * z / fx
    y = (float(y_pixel) - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def local_to_global_matrix(transform_matrix, local_point):
    """
    transform_matrix: (4,4) local->global
    local_point: (3,) or (N,3)
    """
    p = _as_points_N3(local_point)
    Tm = np.asarray(transform_matrix, dtype=np.float64)
    homo = np.hstack([p, np.ones((p.shape[0], 1), dtype=np.float64)])
    out = (Tm @ homo.T).T[:, :3]
    return out[0] if local_point is not None and np.asarray(local_point).shape == (3,) else out


def global_to_local_matrix(transform_matrix, global_point):
    inv_transform = T.pose_inv(transform_matrix)
    return local_to_global_matrix(inv_transform, global_point)


def pixel_to_3d_points(depth_image, intrinsics, extrinsics):
    """
    depth_image: (H,W) depth in meters
    intrinsics : K
    extrinsics : T_cam_world (world->cam)

    Return:
      world_points: (H,W,3) in WORLD frame
    """
    H, W = depth_image.shape
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    K = np.asarray(intrinsics, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if torch.is_tensor(depth_image):
        depth_image = depth_image.detach().cpu().numpy()
    z = depth_image.astype(np.float64)

    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    cam = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    cam_h = np.hstack((cam, np.ones((cam.shape[0], 1), dtype=np.float64)))  # (N,4)

    # extrinsics is world->cam, so cam->world is inv(extrinsics)
    T_w_cam = T.pose_inv(np.asarray(extrinsics, dtype=np.float64))
    world_h = (T_w_cam @ cam_h.T).T  # (N,4)
    world = world_h[:, :3] / world_h[:, 3:4]
    return world.reshape(H, W, 3)


def depth_to_height(depth_image, intrinsics, extrinsics):
    world_points = pixel_to_3d_points(depth_image, intrinsics, extrinsics)
    return world_points[:, :, 2]


def find_intersections(x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int):
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    else:
        m = None
        b = None

    intersections = []
    if m is not None and m != 0:
        x_at_yh = int((img_height - b) / m)
        if 0 <= x_at_yh <= img_width:
            intersections.append((x_at_yh, img_height - 1))

    if m is not None:
        y_at_x0 = int(b)
        if 0 <= y_at_x0 <= img_height:
            intersections.append((0, y_at_x0))

    if m is not None:
        y_at_xw = int(m * img_width + b)
        if 0 <= y_at_xw <= img_height:
            intersections.append((img_width - 1, y_at_xw))

    if m is not None and m != 0:
        x_at_y0 = int(-b / m)
        if 0 <= x_at_y0 <= img_width:
            intersections.append((x_at_y0, 0))

    if m is None:
        intersections.append((x1, img_height - 1))
        intersections.append((x1, 0))

    if len(intersections) == 2:
        return intersections
    return None


def put_text_on_image(image, text, location, font=cv2.FONT_HERSHEY_SIMPLEX, text_size=2.7,
                      bg_color=(255, 255, 255), text_color=(0, 0, 0),
                      text_thickness=3, highlight=True):
    scale_factor = image.shape[0] / 1080
    adjusted_thickness = math.ceil(scale_factor * text_thickness)
    adjusted_size = scale_factor * text_size

    assert location in ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_center', 'center']

    img_height, img_width = image.shape[:2]
    ts, _ = cv2.getTextSize(text, font, adjusted_size, adjusted_thickness)

    offset = math.ceil(10 * scale_factor)
    if location == 'top_left':
        text_x, text_y = offset, ts[1] + offset
    elif location == 'top_right':
        text_x, text_y = img_width - ts[0] - offset, ts[1] + offset
    elif location == 'bottom_left':
        text_x, text_y = offset, img_height - offset
    elif location == 'bottom_right':
        text_x, text_y = img_width - ts[0] - offset, img_height - offset
    elif location == 'top_center':
        text_x, text_y = (img_width - ts[0]) // 2, ts[1] + offset
    else:
        text_x, text_y = (img_width - ts[0]) // 2, (img_height + ts[1]) // 2

    if highlight:
        cv2.rectangle(image,
                      (text_x - offset // 2, text_y - ts[1] - offset),
                      (text_x + ts[0] + offset // 2, text_y + offset),
                      bg_color, -1)

    cv2.putText(image, text, (text_x, text_y), font, adjusted_size, text_color, adjusted_thickness)
    return image

def encode_image_b64(image: Image.Image, format="PNG") -> str:
    """
    Convert a PIL Image to Base64 string.

    Parameters:
        image (Image.Image): The PIL image to convert.

    Returns:
        image_b64: The image in base64 as a string.
    """
    with io.BytesIO() as output:
        image.save(output, format=format)
        return base64.b64encode(output.getvalue()).decode(('utf-8'))


def append_mime_tag(image_b64, format="png"):
    return f"data:image/{format};base64,{image_b64}"

def resize_image_if_needed(image: Image.Image, max_dimension: int) -> Image.Image:
    """
    Resize a PIL image to ensure that its largest dimension does not exceed max_size.

    Parameters:
        image (Image.Image): The PIL image to resize.
        max_size (int): The maximum size for the largest dimension.

    Returns:
        Image.Image: The resized image.
    """
    width, height = image.size

    # Check if the image has a palette and convert it to true color mode
    if image.mode == "P":
        if "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")

    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        image = image.resize((new_width, new_height), Image.LANCZOS)

    return image