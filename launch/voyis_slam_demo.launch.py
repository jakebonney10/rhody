"""Rhody 2 + Voyis stereo + Isaac ROS Visual SLAM, replayed from a survey bag.

    ros2 launch rhody voyis_slam_demo.launch.py bag:=/path/to/demo_bag

Demo/visualisation pipeline for a bag built by
`voyis_discovery_driver/scripts/voyis_images_to_bag.py --no-tf-static`:

    bag play ──> stereo/{left,right}/image_raw/compressed ──> RViz / Foxglove
                        │
                        └─ republish ─> image_rect ─> cuVSLAM ─> odom -> base_link TF
    robot_state_publisher (rhody2.urdf.xacro) ──> base_link -> voyis_*_optical + robot model

The bag must be built with --no-tf-static: the URDF owns every transform below
base_link, and a bag that also publishes them would fight robot_state_publisher
for the same edges of the TF tree.

Because cuVSLAM publishes odom -> base_link, the whole Rhody model flies through
the map on the SLAM solution, which is the point of the visual.
"""

import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PREFIX = '/stereo'


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    def flag(name):
        return arg(name).lower() in ('true', '1', 'yes')

    pkg_share = get_package_share_directory('rhody')
    actions = []

    # `bag:=~/bags/foo` is not tilde-expanded by the shell (the ~ is not at the
    # start of the word), so it arrives here literally and ros2 bag play fails
    # with "Bag path does not exist".
    bag = os.path.expanduser(arg('bag'))

    # The apt isaac_ros_gxf keeps its GXF extensions under
    # share/isaac_ros_gxf/gxf/lib/<component>/ but only puts <prefix>/lib on the
    # loader path, so visual_slam cannot find libgxf_serialization.so. Scope this
    # to the one process that needs it rather than mutating the whole launch.
    gxf_root = '/opt/ros/humble/share/isaac_ros_gxf/gxf/lib'
    gxf_dirs = sorted(d for d in glob.glob(f'{gxf_root}/*') if os.path.isdir(d))

    # gemini_sonar_driver vendors its SDK's ffmpeg (libavcodec.so.58,
    # libavutil.so.56) into install/gemini_sonar_driver/lib, which colcon puts on
    # the global LD_LIBRARY_PATH. Those shadow the system ffmpeg, so the system
    # libchromaprint that OpenCV pulls in fails with
    #   undefined symbol: av_rdft_calc, version LIBAVCODEC_58
    # and every OpenCV consumer in the workspace dies. Drop those directories for
    # the processes here that need OpenCV. The real fix belongs in
    # gemini_sonar_driver: install the vendored SDK libs into a private
    # subdirectory and reach them via the $ORIGIN RPATH instead of the search path.
    def without_vendored_ffmpeg(path):
        keep = []
        for entry in path.split(':'):
            if entry and not glob.glob(os.path.join(entry, 'libavcodec.so*')):
                keep.append(entry)
        return ':'.join(keep)

    clean_ld = without_vendored_ffmpeg(os.environ.get('LD_LIBRARY_PATH', ''))

    slam_env = {'LD_LIBRARY_PATH': ':'.join(gxf_dirs + [clean_ld])}

    # PYTHONNOUSERSITE additionally guards against a user-local numpy in ~/.local
    # shadowing the one apt's python3-opencv was built against.
    opencv_env = {'LD_LIBRARY_PATH': clean_ld, 'PYTHONNOUSERSITE': '1'}

    # --- the vehicle ------------------------------------------------------
    actions.append(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(
                Command(['xacro ', os.path.join(pkg_share, 'urdf', 'rhody2.urdf.xacro')]),
                value_type=str),
            'use_sim_time': True,
        }],
    ))

    # --- the survey -------------------------------------------------------
    if not os.path.isdir(bag):
        raise RuntimeError(f"bag path does not exist: {bag}")

    play = ['ros2', 'bag', 'play', bag, '--clock', '--rate', arg('rate')]
    if flag('loop'):
        play.append('--loop')
    actions.append(ExecuteProcess(cmd=play, output='screen', on_exit=[Shutdown()]))

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
                ('in/compressed', f'{PREFIX}/{eye}/image_raw/compressed'),
                ('out', f'{PREFIX}/{eye}/image_mono'),
            ],
            parameters=[{'use_sim_time': True}],
            additional_env=opencv_env,
        ))

    # --- SLAM -------------------------------------------------------------
    actions.append(Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam',
        name='visual_slam',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'rectified_images': True,
            'enable_image_denoising': False,
            # Frame-to-frame delta cap, not a jitter tolerance: must exceed the
            # 200 ms survey period or every frame is rejected.
            'image_jitter_threshold_ms': 400.0,
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
            ('visual_slam/image_0', f'{PREFIX}/left/image_mono'),
            ('visual_slam/camera_info_0', f'{PREFIX}/left/camera_info'),
            ('visual_slam/image_1', f'{PREFIX}/right/image_mono'),
            ('visual_slam/camera_info_1', f'{PREFIX}/right/camera_info'),
        ],
        additional_env=slam_env,
    ))

    # --- visualisation ----------------------------------------------------
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
            # Foxglove renders the robot model from /robot_description, which
            # robot_state_publisher offers transient-local.
            'asset_uri_allowlist': [
                '^package://(?!.*\\.\\.).*\\.(dae|fbx|glb|gltf|jpeg|jpg|mtl|obj|png|stl|tif|tiff|urdf|webp|xacro)$'
            ],
        }],
        condition=IfCondition(LaunchConfiguration('foxglove')),
    ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('bag', description='demo bag built with --no-tf-static'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('loop', default_value='true',
                              description='loop the bag — handy for a live demo'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('foxglove', default_value='true',
                              description='serve Foxglove on ws://localhost:8765'),
        OpaqueFunction(function=launch_setup),
    ])
