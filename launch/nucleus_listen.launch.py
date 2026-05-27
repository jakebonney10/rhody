#!/usr/bin/env python3

"""
Launch file for the Nortek Nucleus DVL driver in listen-only (streaming) mode.

Connects to the Nucleus read-only streaming port (9002) and only parses and
publishes incoming data. Device configuration and start/stop are handled
externally (e.g. by the Nortek software on the command port 9000).
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
        namespace='rhody',
        output='screen',
        parameters=[config]
    )

    return LaunchDescription([
        nucleus_node
    ])
