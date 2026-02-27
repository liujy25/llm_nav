#!/usr/bin/env python3
"""
ROS2节点：接收RGBD和TF，发布odom坐标系下的点云
用于验证坐标变换是否正确
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener, TransformException
from scipy.spatial.transform import Rotation as R

import struct


def transform_to_matrix(transform) -> np.ndarray:
    """将TransformStamped转换为4x4齐次变换矩阵"""
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    quat = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
    rot_matrix = R.from_quat(quat).as_matrix()
    
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rot_matrix
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def get_point_cloud(depth: np.ndarray, mask: np.ndarray, fx: float, fy: float, 
                    cx: float, cy: float) -> np.ndarray:
    """从深度图生成点云（OpenCV光学坐标系）"""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    u = u[mask]
    v = v[mask]
    z = depth[mask]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return np.stack([x, y, z], axis=1)


def transform_points(tf_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """使用4x4变换矩阵变换3D点"""
    ones = np.ones((points.shape[0], 1))
    points_homo = np.hstack([points, ones])
    transformed = (tf_matrix @ points_homo.T).T
    return transformed[:, :3]


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
    """体素下采样点云
    
    Args:
        points: Nx3点云数组
        voxel_size: 体素大小（米），默认0.1m即10cm
    
    Returns:
        下采样后的点云
    """
    if len(points) == 0:
        return points
    
    # 计算每个点所属的体素索引
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # 使用字典存储每个体素中的点
    voxel_dict = {}
    for i, voxel_idx in enumerate(voxel_indices):
        key = tuple(voxel_idx)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(points[i])
    
    # 对每个体素取中心点
    downsampled = []
    for voxel_points in voxel_dict.values():
        center = np.mean(voxel_points, axis=0)
        downsampled.append(center)
    
    return np.array(downsampled, dtype=np.float32)


def create_pointcloud2_msg(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """创建PointCloud2消息"""
    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    
    msg.height = 1
    msg.width = len(points)
    
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    
    msg.is_bigendian = False
    msg.point_step = 12  # 3 * 4 bytes
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    
    # 打包点云数据
    buffer = []
    for point in points:
        buffer.append(struct.pack('fff', point[0], point[1], point[2]))
    
    msg.data = b''.join(buffer)
    
    return msg


class OdomPointCloudPublisher(Node):
    """发布odom坐标系下的点云"""
    
    def __init__(self):
        super().__init__('odom_pointcloud_publisher')
        
        # 参数（与llm_nav_client保持一致）
        self.declare_parameter('camera_frame', 'camera_head_left_link')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('depth_topic', '/hdas/camera_head/depth/depth_registered')
        self.declare_parameter('camera_info_topic', '/hdas/camera_head/depth/camera_info')
        self.declare_parameter('min_depth', 0.5)
        self.declare_parameter('max_depth', 5.0)
        
        self.camera_frame = self.get_parameter('camera_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        
        # QoS配置
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 订阅
        self.depth_sub = self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self.depth_callback,
            qos
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback,
            qos
        )
        
        # 发布
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/pointcloud_odom',
            10
        )
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 状态
        self.camera_info = None
        self.bridge = CvBridge()
        
        self.get_logger().info(f'Odom PointCloud Publisher started')
        self.get_logger().info(f'  Camera frame: {self.camera_frame}')
        self.get_logger().info(f'  Odom frame: {self.odom_frame}')
        self.get_logger().info(f'  Depth range: [{self.min_depth}, {self.max_depth}]')
        self.get_logger().info(f'  Use optical->link transform: {self.use_optical_transform}')
    
    def camera_info_callback(self, msg: CameraInfo):
        """接收相机内参"""
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info(f'Received camera info: fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}')
    
    def depth_callback(self, msg: Image):
        """接收深度图并发布点云"""
        if self.camera_info is None:
            self.get_logger().warn('Waiting for camera info...', throttle_duration_sec=2.0)
            return
        
        try:
            # 1. 转换深度图
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # 调试：检查深度图编码和范围
            self.get_logger().info(
                f'Depth image: encoding={msg.encoding}, shape={depth.shape}, '
                f'dtype={depth.dtype}, range=[{depth.min()}, {depth.max()}]',
                throttle_duration_sec=5.0
            )
            
            # 根据编码决定是否需要转换
            if msg.encoding == '16UC1':
                depth = depth.astype(np.float32) / 1000.0  # mm -> m
            elif msg.encoding == '32FC1':
                depth = depth.astype(np.float32)  # 已经是米
            else:
                self.get_logger().warn(f'Unknown depth encoding: {msg.encoding}')
                depth = depth.astype(np.float32) / 1000.0
            
            # 2. 生成点云（光学坐标系）
            mask = (depth > self.min_depth) & (depth < self.max_depth) & np.isfinite(depth)
            
            self.get_logger().info(
                f'Depth after conversion: range=[{depth.min():.3f}, {depth.max():.3f}], '
                f'valid pixels: {mask.sum()}/{mask.size}',
                throttle_duration_sec=5.0
            )
            
            K = np.array(self.camera_info.k).reshape(3, 3)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            pc_optical = get_point_cloud(depth, mask, fx, fy, cx, cy)
            
            if len(pc_optical) == 0:
                self.get_logger().warn(
                    f'No valid points! Depth range: [{self.min_depth}, {self.max_depth}]',
                    throttle_duration_sec=2.0
                )
                return
            
            # 3. 转换坐标系（如果需要）
            if self.use_optical_transform:
                # OpenCV optical -> ROS camera_link
                # [X_link, Y_link, Z_link] = [Z_optical, -X_optical, -Y_optical]
                pc_camera = np.zeros_like(pc_optical)
                pc_camera[:, 0] = pc_optical[:, 2]   # forward
                pc_camera[:, 1] = -pc_optical[:, 0]  # left
                pc_camera[:, 2] = -pc_optical[:, 1]  # up
            else:
                pc_camera = pc_optical
            
            # 4. 查询TF变换
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.odom_frame,
                    self.camera_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.1)
                )
            except TransformException as e:
                self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=2.0)
                return
            
            # 5. 变换到odom坐标系
            T_odom_cam = transform_to_matrix(transform)
            pc_odom = transform_points(T_odom_cam, pc_camera)
            
            # 6. 体素下采样（10cm）
            pc_odom_downsampled = voxel_downsample(pc_odom, voxel_size=0.1)
            
            self.get_logger().info(
                f'Downsampled: {len(pc_odom)} -> {len(pc_odom_downsampled)} points',
                throttle_duration_sec=1.0
            )
            
            # 7. 发布点云
            pc_msg = create_pointcloud2_msg(pc_odom_downsampled, self.odom_frame, msg.header.stamp)
            self.pointcloud_pub.publish(pc_msg)
            
            # 统计信息
            z_min, z_max = pc_odom_downsampled[:, 2].min(), pc_odom_downsampled[:, 2].max()
            self.get_logger().info(
                f'Published {len(pc_odom_downsampled)} points, Z range: [{z_min:.3f}, {z_max:.3f}]',
                throttle_duration_sec=1.0
            )
            
        except Exception as e:
            self.get_logger().error(f'Error processing depth: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = OdomPointCloudPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
