#!/usr/bin/env python3

"""
Launch file for the Tritech Gemini 1200ik multibeam imaging sonar driver under
the rhody/perception/sensors namespace. The driver discovers the head over the
Gemini SDK on the 192.168.2.x subnet.

The node comes up idle; call the start_sonar service to begin pinging, e.g.:
  ros2 service call \
    /rhody/perception/sensors/gemini/start_sonar \
    gemini_sonar_driver_interfaces/srv/StartSonar "{enable_logging: false, log_directory: ''}"

Parameters are read from rhody/config/gemini.yaml (a copy of the driver default,
editable here without touching the upstream gemini_sonar_driver package).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('rhody'),
        'config',
        'gemini.yaml'
    )

    gemini_node = Node(
        package='gemini_sonar_driver',
        executable='gemini_sonar_node',
        name='gemini_sonar_driver',
        namespace='rhody/perception/sensors',
        output='screen',
        parameters=[config],
        emulate_tty=True
    )

    return LaunchDescription([
        gemini_node
    ])
