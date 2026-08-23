"""Voyis stereo perception pipeline: cuVSLAM, optional nvblox, optional RViz.

The shared core behind voyis_slam_demo.launch.py (bag replay) and
voyis_slam_live.launch.py (live driver). Everything that is not "where do the
images come from" lives here, so the two front-ends differ only in that.

    robot_state_publisher (rhody2.urdf.xacro) ──> base_link -> voyis_*_optical
    <prefix>/{left,right}/image_raw/compressed ──> compressed_to_mono.py
                                                        │ image_mono
                                                        ├─> cuVSLAM ─> odom -> base_link
                                                        └─> voyis_nvblox.launch.py

Not launched on its own -- it assumes something is publishing
<prefix>/{left,right}/image_raw/compressed and the matching camera_info.

The two settings that differ live vs. replay and are easy to get wrong:

  use_sim_time  true for a bag (which publishes /clock), false live. Getting
                this wrong live means every node blocks forever waiting for a
                clock that never comes.
  prefix        the bag builder writes /stereo; the live driver runs in the
                'rhody' namespace (config/voyis.yaml uses relative names), so
                its topics land under /rhody/stereo.
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from isaac_env import isaac_env, opencv_env, rviz_env  # noqa: E402


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    def flag(name):
        return arg(name).lower() in ('true', '1', 'yes')

    pkg_share = get_package_share_directory('rhody')
    prefix = arg('prefix').rstrip('/')
    sim_time = flag('use_sim_time')
    actions = []

    actions.append(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(
                Command(['xacro ', os.path.join(pkg_share, 'urdf', 'rhody2.urdf.xacro')]),
                value_type=str),
            'use_sim_time': sim_time,
        }],
        condition=IfCondition(LaunchConfiguration('robot_state_publisher')),
    ))

    # Decode straight to mono8 rather than `image_transport republish`, which
    # would preserve rgb8 and put 1.49 MB raw frames on the wire. With
    # net.core.rmem_max at its 208 KB default those get shredded by fragmented
    # UDP reassembly and cuVSLAM sees ~0.75 Hz instead of 5 Hz. mono8 is 495 KB
    # and is all cuVSLAM uses; the colour compressed topic is untouched for
    # RViz and Foxglove.
    for eye in ('left', 'right'):
        actions.append(Node(
            package='rhody',
            executable='compressed_to_mono.py',
            name=f'to_mono_{eye}',
            remappings=[
                ('in/compressed', f'{prefix}/{eye}/image_raw/compressed'),
                ('out', f'{prefix}/{eye}/image_mono'),
            ],
            parameters=[{'use_sim_time': sim_time}],
            additional_env=opencv_env(),
        ))

    actions.append(Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam',
        name='visual_slam',
        output='screen',
        parameters=[{
            'use_sim_time': sim_time,
            'rectified_images': True,
            'enable_image_denoising': False,
            # Frame-to-frame delta cap, not a jitter tolerance: must exceed the
            # 200 ms survey period or every frame is rejected.
            'image_jitter_threshold_ms': float(arg('jitter_threshold_ms')),
            'enable_imu_fusion': False,
            'num_cameras': 2,
            'min_num_images': 2,
            'enable_localization_n_mapping': True,
            # The vis/* clouds have publishers but stay silent unless these are
            # on. They are what makes the demo read as SLAM rather than dead
            # reckoning: tracked landmarks accumulating over the wreck.
            'enable_slam_visualization': True,
            'enable_landmarks_view': True,
            'enable_observations_view': True,
            'publish_odom_to_base_tf': True,
            'publish_map_to_odom_tf': True,
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'camera_optical_frames': ['voyis_left_optical', 'voyis_right_optical'],
        }],
        remappings=[
            ('visual_slam/image_0', f'{prefix}/left/image_mono'),
            ('visual_slam/camera_info_0', f'{prefix}/left/camera_info'),
            ('visual_slam/image_1', f'{prefix}/right/image_mono'),
            ('visual_slam/camera_info_1', f'{prefix}/right/camera_info'),
        ],
        additional_env=isaac_env(),
    ))

    if flag('nvblox'):
        # Scoped, and not optional. IncludeLaunchDescription's launch_arguments
        # leak into the *parent* context, so the 'rviz': 'false' below would
        # otherwise overwrite this launch's own rviz configuration and silently
        # suppress the RViz node declared further down -- no error, RViz simply
        # never starts.
        actions.append(GroupAction([IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'voyis_nvblox.launch.py')),
            launch_arguments={
                'prefix': prefix,
                'use_sim_time': arg('use_sim_time'),
                'color': arg('color'),
                'image_width': arg('image_width'),
                'image_height': arg('image_height'),
                'rviz': 'false',   # the pipeline owns RViz, not the sub-launch
            }.items(),
        )], scoped=True, forwarding=False))

    rviz_config = arg('rviz_config') or os.path.join(
        pkg_share, 'rviz',
        'voyis_nvblox_demo.rviz' if flag('nvblox') else 'voyis_slam_demo.rviz')
    actions.append(Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': sim_time}],
        additional_env=rviz_env(),
        condition=IfCondition(LaunchConfiguration('rviz')),
    ))

    actions.append(Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': 8765,
            'use_sim_time': sim_time,
            # Foxglove renders the robot model from /robot_description, whose
            # URDF points at package://rhody/meshes/*.dae. Only a live bridge
            # can serve those assets -- an .mcap opened on its own cannot.
            'asset_uri_allowlist': [
                '^package://(?!.*\\.\\.).*\\.(dae|fbx|glb|gltf|jpeg|jpg|mtl|obj|png|stl|tif|tiff|urdf|webp|xacro)$'
            ],
        }],
        condition=IfCondition(LaunchConfiguration('foxglove')),
    ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('prefix', default_value='/stereo',
                              description='namespace of the stereo topics'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true for bag replay, false live'),
        DeclareLaunchArgument('nvblox', default_value='false'),
        DeclareLaunchArgument('color', default_value='false',
                              description='colour the nvblox mesh; rgb8 sources only'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value='',
                              description='override the rviz config chosen by nvblox:='),
        DeclareLaunchArgument('foxglove', default_value='false'),
        DeclareLaunchArgument('robot_state_publisher', default_value='true'),
        DeclareLaunchArgument('image_width', default_value='704'),
        DeclareLaunchArgument('image_height', default_value='704'),
        DeclareLaunchArgument('jitter_threshold_ms', default_value='400.0'),
        OpaqueFunction(function=launch_setup),
    ])
