#!/usr/bin/env python3

"""
Launch file for the Sonardyne SPRINT-Nav Mini INS/DVL driver under the
rhody/nav/sensors namespace. Connects via TCP to the SPRINT-Nav at 192.168.2.206.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('rhody'),
        'config',
        'sprintnav.yaml'
    )

    sprintnav_node = Node(
        package='sprintnav_driver',
        executable='sprintnav',
        name='sprintnav',
        namespace='rhody/nav/sensors',
        output='screen',
        parameters=[config]
    )

    return LaunchDescription([
        sprintnav_node
    ])
