#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure, Temperature
from geometry_msgs.msg import PoseWithCovarianceStamped

# -----------------------------
# Constants (saltwater equation)
# -----------------------------
C1 = 9.72659
C2 = -2.2512e-5
C3 = 2.279e-10
C4 = -1.82e-15
GAMMA = 2.184e-6

# -----------------------------
# Constants (freshwater equation)
# -----------------------------
DEFAULT_TEMP_C = 25.0
DEFAULT_DENSITY = 997.0  # kg/m³ at ~25°C
GRAVITY = 9.80665        # m/s²

class Pressure2DepthNode(Node):
    def __init__(self):
        super().__init__('pressure2depth_node')

        # Declare and get parameters
        self.declare_parameter('pressure_topic', '/pressure')
        self.declare_parameter('depth_topic', '/depth')
        self.declare_parameter('ref_lat', 45.0)
        self.declare_parameter('atmospheric_pressure', 101325.0)
        self.declare_parameter('relative_frame', 'base_link')
        self.declare_parameter('cov_offset', 0.1)
        self.declare_parameter('freshwater', False)
        self.declare_parameter('temperature_topic', '')

        # Parameters
        self.pressure_topic = self.get_parameter('pressure_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.atmospheric_pressure = self.get_parameter('atmospheric_pressure').get_parameter_value().double_value
        self.relative_frame = self.get_parameter('relative_frame').get_parameter_value().string_value
        self.cov_offset = self.get_parameter('cov_offset').get_parameter_value().double_value
        self.freshwater = self.get_parameter('freshwater').get_parameter_value().bool_value
        self.temperature_topic = self.get_parameter('temperature_topic').get_parameter_value().string_value

        # Initialize last known temperature
        self.temperature_c = DEFAULT_TEMP_C

        # Subscriptions
        self.subscription = self.create_subscription(
            FluidPressure,
            self.pressure_topic,
            self.pressure_callback,
            10
        )

        if self.temperature_topic:
            self.temp_sub = self.create_subscription(
                Temperature,
                self.temperature_topic,
                self.temperature_callback,
                10
            )
            self.get_logger().info(f"Temperature topic subscribed: {self.temperature_topic}")
        else:
            self.temp_sub = None

        # Publisher
        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            self.depth_topic,
            10
        )

        self.get_logger().info(f"Pressure2Depth node started. Freshwater mode: {self.freshwater}")

    def temperature_callback(self, msg):
        self.temperature_c = msg.temperature

    def calc_density_freshwater(self, temp_c):
        """
        Empirical polynomial for freshwater density as a function of temperature (°C)
        Based on Tanaka et al., 2001 for 0–40°C range
        """
        T = temp_c
        return (999.842594 +
                6.793952e-2 * T -
                9.09529e-3 * T**2 +
                1.001685e-4 * T**3 -
                1.120083e-6 * T**4 +
                6.536332e-9 * T**5)

    def convert_pressure(self, pressure_pa):
        if self.freshwater:
            rho = self.calc_density_freshwater(self.temperature_c)
            return (pressure_pa - self.atmospheric_pressure) / (rho * GRAVITY)
        else:
            dp = pressure_pa / 10000.0  # Pa → dbar
            return (C1 + dp * (C2 + dp * (C3 + dp * C4))) * dp * (1 - GAMMA * dp)

    def pressure_callback(self, msg):
        pressure_pa = msg.fluid_pressure
        depth = self.convert_pressure(pressure_pa)

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.relative_frame
        pose_msg.pose.pose.position.z = depth

        pose_msg.pose.covariance = [0.0] * 36
        pose_msg.pose.covariance[8] = self.cov_offset ** 2

        self.publisher.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Pressure2DepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
