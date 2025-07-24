#!/usr/bin/env python3

import math
import pyproj
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import NavSatFix, Imu
# from roship_msgs.msg import ProjectionInfo
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R


class TransformPoseNode(Node):
    def __init__(self):
        super().__init__('gps2utm')

        self.publishers_ = {}
        self.subscribers_ = {}
        # self.proj_publishers = {}
        self.imu_publishers = {}
        self.last_gps = None

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        self.topics = self.declare_parameter('topics', []).value
        self.position_topic = self.declare_parameter('position_topic', '').value
        self.projected_frame = self.declare_parameter('projected_frame', 'utm_local').value
        self.map_projection = self.declare_parameter('map_projection', 'utm').value  # 'utm' or 'ecef'

        self.declare_parameter('utm_zone', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('origin_lon', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('origin_lat', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('origin_z', 0.0)

        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.create_timer(1.0, self.update_subscribers)

    def params_ready(self):
        return all([
            self.has_parameter('utm_zone'),
            self.has_parameter('origin_lon'),
            self.has_parameter('origin_lat'),
            self.has_parameter('origin_z'),
            self.position_topic != ''
        ])

    def compute_projection(self):
        self.origin_z = self.get_parameter('origin_z').value
        origin_lon = self.get_parameter('origin_lon').value
        origin_lat = self.get_parameter('origin_lat').value

        if self.map_projection == 'utm':
            self.utm_zone = self.get_parameter('utm_zone').value
            self.proj = pyproj.Proj(proj='utm', zone=self.utm_zone, ellps='WGS84', preserve_units=True)
            self.xoff, self.yoff = self.proj(origin_lon, origin_lat)

            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = f"utm_{self.utm_zone}"
            tf_msg.child_frame_id = self.projected_frame
            tf_msg.transform.translation.x = self.xoff
            tf_msg.transform.translation.y = self.yoff
            tf_msg.transform.translation.z = self.origin_z
            tf_msg.transform.rotation.w = 1.0
            self.tf_static_broadcaster.sendTransform(tf_msg)

        elif self.map_projection == 'ecef':
            self.proj = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
            self.geodetic = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
            self.xoff, self.yoff, self.zoff = pyproj.transform(self.geodetic, self.proj, origin_lon, origin_lat, self.origin_z)
        else:
            raise ValueError(f"Unsupported projection: {self.map_projection}")

    def update_subscribers(self):
        if not self.params_ready():
            self.get_logger().info("Waiting for valid parameters")
            return

        if not hasattr(self, 'proj'):
            self.compute_projection()

        if 'pos_sub' not in self.subscribers_:
            self.subscribers_['pos_sub'] = self.create_subscription(
                NavSatFix, self.position_topic, self.position_navsat_callback, self.sensor_qos)

        topic_list = self.get_topic_names_and_types()
        for topic_name, types in topic_list:
            msg_type = types[0]

            if topic_name in self.subscribers_:
                continue

            if topic_name.startswith('/utm/'):
                continue

            if self.topics and topic_name not in self.topics:
                continue

            if msg_type == 'sensor_msgs/msg/NavSatFix':
                self.get_logger().info(f"Subscribing to NavSatFix topic: {topic_name}")
                self.subscribers_[topic_name] = self.create_subscription(
                    NavSatFix, topic_name, lambda msg, t=topic_name: self.gps_callback(msg, t), self.sensor_qos)
                self.publishers_[topic_name] = self.create_publisher(PoseWithCovarianceStamped, f"/utm{topic_name}", 10)
                # self.proj_publishers[topic_name] = self.create_publisher(ProjectionInfo, f"/utm{topic_name}/proj", 10)

            elif msg_type == 'sensor_msgs/msg/Imu':
                self.get_logger().info(f"Subscribing to IMU topic: {topic_name}")
                self.subscribers_[topic_name] = self.create_subscription(
                    Imu, topic_name, lambda msg, t=topic_name: self.imu_callback(msg, t), self.sensor_qos)
                self.imu_publishers[topic_name] = self.create_publisher(Imu, f"/utm{topic_name}", 10)
                # self.proj_publishers[topic_name] = self.create_publisher(ProjectionInfo, f"/utm{topic_name}/proj", 10)

    def position_navsat_callback(self, msg):
        self.last_gps = msg

    def compute_convergence_angle(self, lon, lat):
        utm_center = self.utm_zone * 6 - 180 - 3
        return math.radians((lon - utm_center) * math.sin(math.radians(lat)))

    def imu_callback(self, msg, topic):
        if self.last_gps is None:
            return

        time_diff = (msg.header.stamp.sec - self.last_gps.header.stamp.sec)
        if time_diff > 5:
            self.get_logger().warn("IMU and GPS timestamps differ by more than 5s")
            return

        try:
            convergence_angle = self.compute_convergence_angle(
                self.last_gps.longitude, self.last_gps.latitude)
        except Exception as e:
            self.get_logger().warn(f"Convergence angle computation failed: {e}")
            return

        quat = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]
        r = R.from_quat(quat)
        roll, pitch, yaw = r.as_euler('xyz')
        yaw = (yaw - convergence_angle) % (2 * math.pi)
        new_quat = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()

        msg.orientation.x = new_quat[0]
        msg.orientation.y = new_quat[1]
        msg.orientation.z = new_quat[2]
        msg.orientation.w = new_quat[3]
        self.imu_publishers[topic].publish(msg)

    def gps_callback(self, msg, topic):
        try:
            if self.map_projection == 'utm':
                x, y = self.proj(msg.longitude, msg.latitude)
                x -= self.xoff
                y -= self.yoff
                z = msg.altitude - self.origin_z
            elif self.map_projection == 'ecef':
                lon, lat, alt = msg.longitude, msg.latitude, msg.altitude
                x, y, z = pyproj.transform(self.geodetic, self.proj, lon, lat, alt)
                x -= self.xoff
                y -= self.yoff
                z -= self.zoff
            else:
                self.get_logger().error("Unsupported projection type")
                return
        except RuntimeError as e:
            self.get_logger().warn(f"Projection failed: {e}")
            return

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = msg.header
        pose_msg.header.frame_id = self.projected_frame
        pose_msg.pose.pose.position.x = x
        pose_msg.pose.pose.position.y = y
        pose_msg.pose.pose.position.z = z
        pose_msg.pose.covariance[0] = msg.position_covariance[0]
        pose_msg.pose.covariance[7] = msg.position_covariance[4]
        pose_msg.pose.covariance[14] = msg.position_covariance[8]

        self.publishers_[topic].publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TransformPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
