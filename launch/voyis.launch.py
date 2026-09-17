#!/usr/bin/env python3

"""
Launch file for the Voyis Discovery stereo driver in the rhody namespace.

Runs the driver under the 'rhody/perception/sensors/voyis' namespace, so its
topics resolve under /rhody/perception/sensors/voyis (e.g.
/rhody/perception/sensors/voyis/stereo/points). This parallels the nav-side
convention (rhody/nav/sensors/<device>). Data is stamped in the
voyis_left_optical / voyis_right_optical frames (config/voyis.yaml), which the
vehicle URDF publishes — run rhody_description.launch.py (or any launch that
starts robot_state_publisher) alongside so the point cloud attaches to the
vehicle through TF.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('rhody'),
        'config',
        'voyis.yaml'
    )

    serial_id_arg = DeclareLaunchArgument(
        'serial_id',
        default_value='',
        description='Camera serial ID to connect to (empty = first detected)'
    )

    voyis_node = Node(
        package='voyis_discovery_driver',
        executable='voyis_discovery',
        name='voyis_discovery',
        namespace='rhody/perception/sensors/voyis',
        output='screen',
        parameters=[
            config,
            {'serial_id': LaunchConfiguration('serial_id')},
        ]
    )

    return LaunchDescription([
        serial_id_arg,
        voyis_node,
    ])
