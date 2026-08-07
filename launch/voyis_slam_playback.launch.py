"""Replay a capture bag recorded from voyis_slam_demo.launch.py.

    ros2 launch rhody voyis_slam_playback.launch.py bag:=$HOME/bags/demo_capture

A capture bag (`ros2 bag record -a -s mcap` taken while the demo ran) already
contains /tf, /tf_static, /robot_description, the stereo imagery and every
cuVSLAM output. So this launch deliberately starts NO pipeline: no cuVSLAM, no
robot_state_publisher, no image decoding. Running those again would republish
the same TF edges and fight the recording.

It also does not pass --clock: the capture already contains a /clock topic, and
adding a second publisher makes sim time jitter. Consumers run with
use_sim_time so they follow the recorded clock.
"""

import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    def flag(name):
        return arg(name).lower() in ('true', '1', 'yes')

    pkg_share = get_package_share_directory('rhody')
    bag = os.path.expanduser(arg('bag'))
    if not os.path.isdir(bag):
        raise RuntimeError(f"bag path does not exist: {bag}")

    # gemini_sonar_driver's vendored libavcodec shadows the system ffmpeg and
    # breaks every OpenCV consumer, RViz's compressed image display included.
    def without_vendored_ffmpeg(path):
        return ':'.join(
            e for e in path.split(':')
            if e and not glob.glob(os.path.join(e, 'libavcodec.so*'))
        )

    clean_ld = without_vendored_ffmpeg(os.environ.get('LD_LIBRARY_PATH', ''))

    play = ['ros2', 'bag', 'play', bag, '--rate', arg('rate')]
    if flag('loop'):
        play.append('--loop')
    if arg('start_offset'):
        play += ['--start-offset', arg('start_offset')]

    actions = [ExecuteProcess(cmd=play, output='screen', on_exit=[Shutdown()])]

    actions.append(Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'voyis_slam_demo.rviz')],
        parameters=[{'use_sim_time': True}],
        additional_env={'LD_LIBRARY_PATH': clean_ld},
        condition=IfCondition(LaunchConfiguration('rviz')),
    ))

    actions.append(Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': 8765,
            'use_sim_time': True,
            'asset_uri_allowlist': [
                '^package://(?!.*\\.\\.).*\\.(dae|fbx|glb|gltf|jpeg|jpg|mtl|obj|png|stl|tif|tiff|urdf|webp|xacro)$'
            ],
        }],
        condition=IfCondition(LaunchConfiguration('foxglove')),
    ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('bag', description='capture bag recorded from the demo'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('loop', default_value='true'),
        DeclareLaunchArgument('start_offset', default_value=''),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('foxglove', default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])
