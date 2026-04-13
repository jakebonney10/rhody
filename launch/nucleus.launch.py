#!/usr/bin/env python3

"""
Launch file for Nortek Nucleus DVL driver in the rhody namespace.
Connects via TCP to the Nucleus at 192.168.2.201.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('rhody'),
        'config',
        'nucleus.yaml'
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
