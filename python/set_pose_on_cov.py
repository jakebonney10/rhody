#!/usr/bin/env python3
"""
set_pose_on_cov.py

ROS2 node that sets robot_localization EKF pose via service when
odometry positional covariance (x or y) exceeds a threshold.

Behavior:
- Subscribes to odometry topic (default: /rhody/nav/odometry/filtered)
- Monitors pose covariance and computes stddev for x and y
- When stddev_x or stddev_y > threshold (default 10.0 m), waits for a
  good pose message (xy covariance < 1.0 m) from the specified source topic
- Calls /rhody/nav/set_pose service with xy from source and z from odometry
- Optionally sets pose immediately on node start (default: True)

Parameters:
- odom_topic: Topic to monitor for covariance (default: /rhody/nav/odometry/filtered)
- xy_source: Topic for xy pose when resetting (default: /utm/rhody/nav/sensors/subsonus_usbl/fix)
- threshold_m: Covariance threshold in meters (default: 10.0)
- good_pose_threshold_m: Maximum xy covariance for accepting pose (default: 1.0)
- set_on_start: Whether to set pose on node start (default: True)
- set_pose_service: Service name for SetPose (default: /rhody/nav/set_pose)
- service_timeout_s: Service call timeout (default: 5.0)

Run:
    python3 src/rhody/python/set_pose_on_cov.py
    # or if installed as entry point:
    ros2 run rhody set_pose_on_cov
"""

from __future__ import annotations
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

# Try import of SetPose service from robot_localization
try:
    from robot_localization.srv import SetPose
except ImportError:
    SetPose = None


class SetPoseOnCovNode(Node):
    def __init__(self):
        super().__init__('set_pose_on_cov')

        # Declare parameters
        self.declare_parameter('odom_topic', '/rhody/nav/odometry/filtered')
        self.declare_parameter('xy_source', '/utm/rhody/nav/sensors/subsonus_usbl/fix')
        self.declare_parameter('threshold_m', 2.0)
        self.declare_parameter('good_pose_threshold_m', 0.25)
        self.declare_parameter('set_on_start', True)
        self.declare_parameter('set_pose_service', '/rhody/nav/set_pose')
        self.declare_parameter('service_timeout_s', 5.0)

        # Read parameters
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.xy_source_topic = self.get_parameter('xy_source').get_parameter_value().string_value
        self.threshold_m = self.get_parameter('threshold_m').get_parameter_value().double_value
        self.good_pose_threshold_m = self.get_parameter('good_pose_threshold_m').get_parameter_value().double_value
        self.set_on_start = self.get_parameter('set_on_start').get_parameter_value().bool_value
        self.service_name = self.get_parameter('set_pose_service').get_parameter_value().string_value
        self.service_timeout_s = self.get_parameter('service_timeout_s').get_parameter_value().double_value

        self.get_logger().info(f"Parameters:")
        self.get_logger().info(f"  odom_topic: {self.odom_topic}")
        self.get_logger().info(f"  xy_source: {self.xy_source_topic}")
        self.get_logger().info(f"  threshold_m: {self.threshold_m}")
        self.get_logger().info(f"  good_pose_threshold_m: {self.good_pose_threshold_m}")
        self.get_logger().info(f"  set_on_start: {self.set_on_start}")
        self.get_logger().info(f"  set_pose_service: {self.service_name}")

        # State variables
        self.latest_xy_pose: Optional[PoseWithCovarianceStamped] = None
        self.latest_odom: Optional[Odometry] = None
        self.waiting_for_good_pose = False
        self.initial_set_done = False

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, 10)
        self.xy_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, self.xy_source_topic, self._xy_pose_callback, 10)

        # Service client
        if SetPose is not None:
            self.client = self.create_client(SetPose, self.service_name)
        else:
            self.get_logger().error(
                'robot_localization.srv.SetPose could not be imported; service client disabled')
            self.client = None

        # Timer for initial set
        if self.set_on_start:
            self.get_logger().info('Waiting 10.0s for source topics before initial set_pose')
            self.create_timer(10.0, self._initial_set_callback)

    def _xy_pose_callback(self, msg: PoseWithCovarianceStamped):
        """Cache latest XY pose and potentially trigger set_pose if waiting for good pose."""
        self.latest_xy_pose = msg

        # If we're waiting for a good pose, check if this one qualifies
        if self.waiting_for_good_pose:
            xy_stddev = self._get_xy_stddev_from_pose_cov(msg.pose.covariance)
            if xy_stddev is not None and xy_stddev < self.good_pose_threshold_m:
                self.get_logger().info(
                    f'Received good pose (xy_stddev={xy_stddev:.3f}m < {self.good_pose_threshold_m}m); calling set_pose')
                self.waiting_for_good_pose = False
                self._call_set_pose(reason=f'good_pose_received (stddev={xy_stddev:.3f}m)')

    def _odom_callback(self, msg: Odometry):
        """Monitor odometry covariance and trigger waiting state if threshold exceeded."""
        self.latest_odom = msg

        # Extract x and y covariance (diagonal elements 0 and 7)
        cov = msg.pose.covariance
        try:
            var_x = float(cov[0])
            var_y = float(cov[7])
        except (IndexError, TypeError, ValueError):
            self.get_logger().warning('Odometry covariance malformed; skipping')
            return

        std_x = math.sqrt(max(0.0, var_x))
        std_y = math.sqrt(max(0.0, var_y))

        # Check if either x or y stddev exceeds threshold
        if (std_x > self.threshold_m or std_y > self.threshold_m) and not self.waiting_for_good_pose:
            self.get_logger().warning(
                f'Covariance threshold exceeded: std_x={std_x:.2f}m, std_y={std_y:.2f}m '
                f'(threshold={self.threshold_m}m); waiting for good pose')
            self.waiting_for_good_pose = True

    def _initial_set_callback(self):
        """One-shot timer callback to set pose on node start."""
        if not self.initial_set_done:
            self.initial_set_done = True
            if self.latest_xy_pose is None or self.latest_odom is None:
                self.get_logger().warning(
                    'No source messages received for initial set_pose; skipping')
                return

            # Check if xy pose is good enough
            xy_stddev = self._get_xy_stddev_from_pose_cov(self.latest_xy_pose.pose.covariance)
            if xy_stddev is not None and xy_stddev < self.good_pose_threshold_m:
                self.get_logger().info(f'Initial set_pose (xy_stddev={xy_stddev:.3f}m)')
                self._call_set_pose(reason='initial_set_on_start')
            else:
                self.get_logger().warning(
                    f'Initial xy pose covariance too high (stddev={xy_stddev}m); skipping initial set')

    def _get_xy_stddev_from_pose_cov(self, covariance) -> Optional[float]:
        """Extract max of x and y standard deviation from pose covariance."""
        try:
            var_x = float(covariance[0])
            var_y = float(covariance[7])
            std_x = math.sqrt(max(0.0, var_x))
            std_y = math.sqrt(max(0.0, var_y))
            return max(std_x, std_y)
        except (IndexError, TypeError, ValueError):
            return None

    def _call_set_pose(self, reason: str = ''):
        """Call the SetPose service with xy from latest_xy_pose and z from latest_odom."""
        if self.latest_xy_pose is None or self.latest_odom is None:
            self.get_logger().warning('No source messages available; cannot call set_pose')
            return

        if self.client is None:
            self.get_logger().error('Service client not available; cannot call set_pose')
            return

        # Build the pose message: xy from xy_source, z from odometry
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = self.latest_xy_pose.header
        pose_msg.header.frame_id = self.latest_xy_pose.header.frame_id
        
        # Position: xy from source, z from odometry
        pose_msg.pose.pose.position.x = self.latest_xy_pose.pose.pose.position.x
        pose_msg.pose.pose.position.y = self.latest_xy_pose.pose.pose.position.y
        pose_msg.pose.pose.position.z = self.latest_odom.pose.pose.position.z
        
        # Orientation from xy source
        pose_msg.pose.pose.orientation = self.latest_xy_pose.pose.pose.orientation
        
        # Covariance: use xy source for xy, odometry for z
        pose_msg.pose.covariance = list(self.latest_xy_pose.pose.covariance)
        try:
            # Override z covariance (index 14) with odometry value
            pose_msg.pose.covariance[14] = self.latest_odom.pose.covariance[14]
        except (IndexError, TypeError):
            pass

        # Wait for service
        if not self.client.wait_for_service(timeout_sec=self.service_timeout_s):
            self.get_logger().error(
                f'SetPose service {self.service_name} not available after {self.service_timeout_s}s')
            return

        # Build and send request
        try:
            req = SetPose.Request()
            req.pose = pose_msg
            
            self.get_logger().info(
                f'Calling SetPose service ({reason}): '
                f'x={pose_msg.pose.pose.position.x:.2f}, '
                f'y={pose_msg.pose.pose.position.y:.2f}, '
                f'z={pose_msg.pose.pose.position.z:.2f}')
            
            future = self.client.call_async(req)
            
            # Add callback to log result
            future.add_done_callback(self._service_response_callback)
            
        except Exception as e:
            self.get_logger().error(f'Exception while calling SetPose service: {e}')

    def _service_response_callback(self, future):
        """Callback for service response."""
        try:
            response = future.result()
            self.get_logger().info(f'SetPose service call succeeded')
        except Exception as e:
            self.get_logger().error(f'SetPose service call failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SetPoseOnCovNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
