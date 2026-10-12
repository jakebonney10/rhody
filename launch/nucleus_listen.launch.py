#!/usr/bin/env python3

"""
Launch file for the Nortek Nucleus 1000 DVL driver in listen-only (streaming)
mode, under the rhody/nav/sensors namespace (alongside sprintnav and adnav).

Connects to the Nucleus read-only streaming port (9002) and only parses and
publishes incoming data. Device configuration and start/stop are handled
externally (e.g. by the Nortek software on the command port 9000) — this node
never writes to the socket, so it publishes nothing until some other client has
configured and started the device.

Topics land at /rhody/nav/sensors/nucleus_node/*, e.g. imu_common,
bottom_lock_velocity_common, altimeter_common, depth_odometry_common.

Note the driver has no reconnection logic: auto_connect is a single attempt at
startup, and a link that drops later is not detected. The systemd unit
(rhody-nucleus.service) works around the first case by refusing to start until
the port is reachable; see the README for the second.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('rhody'),
        'config',
        'nucleus_listen.yaml'
    )

    nucleus_node = Node(
        package='nucleus_driver_ros2',
        executable='nucleus_node',
        name='nucleus_node',
        namespace='rhody/nav/sensors',
        output='screen',
        parameters=[config]
    )

    return LaunchDescription([
        nucleus_node
    ])
