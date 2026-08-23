"""Rhody 2 + Voyis stereo + Isaac ROS Visual SLAM, replayed from a survey bag.

    ros2 launch rhody voyis_slam_demo.launch.py bag:=/path/to/demo_bag

Bag replay front-end for voyis_pipeline.launch.py, which holds the actual
pipeline. This file is only the survey source:

    bag play ──> stereo/{left,right}/image_raw/compressed ──> voyis_pipeline

The URDF owns every transform below base_link. A bag that also publishes them
fights robot_state_publisher for the same edges, so this launch diverts the
bag's /tf_static to /bag_tf_static -- building with --no-tf-static is no longer
required, and survey bags that carry it now replay correctly.

Because cuVSLAM publishes odom -> base_link, the whole Rhody model flies through
the map on the SLAM solution, which is the point of the visual.

    nvblox:=true   dense TSDF reconstruction from the same stereo pair
    color:=true    photo-texture the mesh; rgb8 bags only (demo_wreck_*)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    Shutdown,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

PREFIX = '/stereo'


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    pkg_share = get_package_share_directory('rhody')

    # `bag:=~/bags/foo` is not tilde-expanded by the shell (the ~ is not at the
    # start of the word), so it arrives here literally and ros2 bag play fails
    # with "Bag path does not exist".
    bag = os.path.expanduser(arg('bag'))
    if not os.path.isdir(bag):
        raise RuntimeError(f"bag path does not exist: {bag}")

    # Divert the bag's /tf_static into a dead topic so robot_state_publisher owns
    # every edge below base_link unopposed. Bags built with --no-tf-static do not
    # carry it, but the survey bags do, and voyis_images_to_bag.py defaults
    # --base-to-left to identity: the bag then wins the race and
    # base_link -> voyis_left_optical resolves to identity, dropping the optical
    # frame rotation entirely. That points the camera along base_link +z (up)
    # instead of forward, so cuVSLAM solves in a rotated frame and nvblox
    # reconstructs the seabed above the vehicle.
    play = ['ros2', 'bag', 'play', bag, '--clock', '--rate', arg('rate'),
            '--remap', '/tf_static:=/bag_tf_static']
    if arg('loop').lower() in ('true', '1', 'yes'):
        play.append('--loop')

    return [
        ExecuteProcess(cmd=play, output='screen', on_exit=[Shutdown()]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'voyis_pipeline.launch.py')),
            launch_arguments={
                'prefix': PREFIX,
                'use_sim_time': 'true',
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
        DeclareLaunchArgument('bag', description='survey or demo bag to replay'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('loop', default_value='true',
                              description='loop the bag -- handy for a live demo, but it '
                                          'restarts the clock and nvblox drops its map'),
        DeclareLaunchArgument('nvblox', default_value='false',
                              description='also build a dense TSDF reconstruction'),
        DeclareLaunchArgument('color', default_value='false',
                              description='colour the nvblox mesh; rgb8 bags only'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('foxglove', default_value='true',
                              description='serve Foxglove on ws://localhost:8765'),
        DeclareLaunchArgument('image_width', default_value='704',
                              description='must match the bag; 704 for --scale 0.25'),
        DeclareLaunchArgument('image_height', default_value='704'),
        OpaqueFunction(function=launch_setup),
    ])
