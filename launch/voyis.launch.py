#!/usr/bin/env python3

"""
Launch file for the Voyis Discovery stereo driver in the rhody namespace.

Runs the driver under the 'rhody' namespace, so its topics resolve under /rhody
(e.g. /rhody/stereo/points, /rhody/stereo/disparity). Also broadcasts a static
transform map -> stereo_left_optical so the point cloud (published in the
stereo_left_optical frame) is visible in RViz with Fixed Frame = map.
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
        namespace='rhody',
        output='screen',
        parameters=[
            config,
            {'serial_id': LaunchConfiguration('serial_id')},
        ]
    )

    # Static transform so RViz (Fixed Frame = map) can place the cloud, which the
    # driver stamps in the stereo_left_optical frame. Left un-namespaced so it
    # broadcasts on the global /tf_static topic.
    # Args: x y z yaw pitch roll parent_frame child_frame
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='voyis_static_tf',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'stereo_left_optical'],
    )

    return LaunchDescription([
        serial_id_arg,
        voyis_node,
        static_tf,
    ])
