"""Rhody 2 + Voyis stereo + Isaac ROS Visual SLAM, live off the camera.

    ros2 launch rhody voyis_slam_live.launch.py                  # driver already running
    ros2 launch rhody voyis_slam_live.launch.py driver:=true     # start it too

Live front-end for voyis_pipeline.launch.py. Same pipeline as
voyis_slam_demo.launch.py; the only differences are that nothing replays a bag,
and:

  use_sim_time  false. There is no /clock live, and nodes left on sim time wait
                for one forever.
  prefix        /rhody/stereo. voyis.launch.py runs the driver in the 'rhody'
                namespace and config/voyis.yaml gives relative topic names, so
                the topics land under /rhody. Pass prefix:= if you run the
                driver somewhere else.

Streaming is started externally by the pilot/Voyis application -- the driver
only registers callbacks, so bring this up once frames are flowing.

    image_width / image_height MUST match what the camera is actually streaming.
    ImageFormatConverterNode is given them explicitly, and they default to the
    2816x2816 native sensor rather than the 704 the demo bags were downscaled to.

The SGM and voxel settings in config/nvblox_voyis.yaml were derived at 704
(f*B = 80.42 px*m). At 2816 that is 321.7, so the defaults do not carry over:
max_disparity 128 puts the near clip at 2.5 m instead of 0.63 m, and one pixel
of disparity is worth 4x less depth error. Re-tune before trusting a live map --
see the "Sizing the voxels" section of README.md.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    pkg_share = get_package_share_directory('rhody')

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'voyis.launch.py')),
            launch_arguments={'serial_id': arg('serial_id')}.items(),
            condition=IfCondition(LaunchConfiguration('driver')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'voyis_pipeline.launch.py')),
            launch_arguments={
                'prefix': arg('prefix'),
                'use_sim_time': 'false',
                'nvblox': arg('nvblox'),
                'color': arg('color'),
                'rviz': arg('rviz'),
                'foxglove': arg('foxglove'),
                'image_width': arg('image_width'),
                'image_height': arg('image_height'),
            }.items(),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('driver', default_value='false',
                              description='also start voyis.launch.py'),
        DeclareLaunchArgument('serial_id', default_value='',
                              description='camera serial; empty = first detected'),
        DeclareLaunchArgument('prefix', default_value='/rhody/stereo',
                              description='where the driver publishes'),
        DeclareLaunchArgument('nvblox', default_value='false'),
        DeclareLaunchArgument('color', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('foxglove', default_value='true'),
        DeclareLaunchArgument('image_width', default_value='2816',
                              description='must match what the camera streams'),
        DeclareLaunchArgument('image_height', default_value='2816'),
        OpaqueFunction(function=launch_setup),
    ])
