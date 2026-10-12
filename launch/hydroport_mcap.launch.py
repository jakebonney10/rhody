#!/usr/bin/env python3

"""
Launch file for the hydroport MCAP bag recorder under the rhody/data namespace.

Records every discovered topic into a split .mcap bag. The node is remotely
operated: it comes up *idle* and records nothing until an operator latches true
on the control topic.

  # start recording
  ros2 topic pub --once --qos-durability transient_local --qos-reliability reliable \
    /rhody/data/logging_on std_msgs/msg/Bool "{data: true}"

  # stop (closes and flushes the current bag)
  ros2 topic pub --once --qos-durability transient_local --qos-reliability reliable \
    /rhody/data/logging_on std_msgs/msg/Bool "{data: false}"

State is echoed on /rhody/data/bag_recorder/logging_on (also transient_local),
and status — recording / not recording / settings error, topic count, path — is
published at 1 Hz on the global /diagnostics topic alongside the other rhody
drivers.

The executable is hydroport's `bag_recorder`, not its `mcap_logger`: both write
mcap, but bag_recorder wraps rosbag2_transport::Recorder (the class behind
`ros2 bag record`), which adapts each subscription to the QoS the publisher
actually offers. mcap_logger subscribes with a fixed KeepLast(10) RELIABLE
profile, so a BEST_EFFORT publisher — the Gemini and Voyis image streams — is
QoS-incompatible and silently records nothing.

Parameters are read from rhody/config/hydroport_mcap.yaml, except the output
directory, which comes from $HOME here so it follows the service user.
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
        'hydroport_mcap.yaml'
    )

    # Expanded here rather than in the YAML so the path tracks whichever user
    # the service runs as. The node creates the directory on first record and
    # reports a diagnostic ERROR (without dying) if it cannot.
    log_dir_arg = DeclareLaunchArgument(
        'log_dir',
        default_value=os.path.join(os.path.expanduser('~'), 'data', 'rhody_mcap'),
        description='Directory the .mcap bags are written to'
    )

    bag_recorder_node = Node(
        package='hydroport',
        executable='bag_recorder',
        name='bag_recorder',
        namespace='rhody/data',
        output='screen',
        parameters=[
            config,
            {'logging.directory': LaunchConfiguration('log_dir')},
        ],
        emulate_tty=True
    )

    return LaunchDescription([
        log_dir_arg,
        bag_recorder_node
    ])
