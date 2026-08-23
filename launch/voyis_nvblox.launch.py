"""Dense reconstruction from the Voyis stereo pair with Isaac ROS nvblox.

    ros2 launch rhody voyis_nvblox.launch.py

Assumes something else is already publishing the rectified stereo pair and the
TF chain -- voyis_slam_demo.launch.py (bag replay + cuVSLAM) or the live
voyis_discovery_driver. Nothing here touches the SLAM pipeline.

    /stereo/{left,right}/image_mono ──> ImageFormatConverterNode x2 (mono8 -> rgb8)
                                            │ /nvblox/{left,right}/image_rgb
                                            v
                                        DisparityNode (VPI SGM)
                                            │ disparity
                                            v
                                        DisparityToDepthNode
                                            │ depth (32FC1)
                                            v
                                          nvblox ──> /nvblox_node/mesh

All three live in one component container on purpose. A 32FC1 depth image is
1.98 MB at 704x704, and with net.core.rmem_max at its 212992 B default anything
that size gets shredded by fragmented UDP reassembly -- the same failure that
pushed the stereo pair to mono8 in the first place (see the throughput notes in
README.md). Composed, disparity and depth never reach the wire, and the only
inter-process traffic is the two 495 KB mono8 images that already flow for
cuVSLAM.

Note none of these carry use_intra_process_comms. All three are NITROS nodes,
and NITROS creates transient-local publishers internally, which rclcpp's
intra-process path rejects outright:

    Component constructor threw an exception:
    intraprocess communication allowed only with volatile durability

NITROS does its own zero-copy anyway -- inside a container it negotiates the
type adaptation and hands over CUDA buffer handles, which is strictly better
than the rclcpp intra-process path since the data never leaves the GPU.

The mono8 -> rgb8 conversion is not optional. NITROS lists nitros_image_mono8
among the formats DisparityNode will accept, but that is only the ROS-side type
adaptation table; the VPI wrapper underneath cannot map an 8-bit grayscale GXF
buffer to a VPI image format and kills the whole container on the first frame:

    ./gxf/gems/vpi/image_wrapper.cpp@31: Expression
    'vpi::VideoFormatToImageFormat(image_info.color_format)' failed with error
    'GXF_INVALID_DATA_FORMAT'

Doing the expansion here rather than in compressed_to_mono.py is deliberate:
mono8 stays on the wire at 495 KB and the 3x expansion happens on the GPU
inside the container, where it costs nothing in transport.

Export the finished map with:

    ros2 service call /nvblox_node/save_ply nvblox_msgs/srv/FilePath \
        "{file_path: /home/bonnaroo/bags/william_elgin_704.ply}"
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from isaac_env import isaac_env, rviz_env  # noqa: E402


def launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    def flag(name):
        return arg(name).lower() in ('true', '1', 'yes')

    pkg_share = get_package_share_directory('rhody')
    prefix = arg('prefix').rstrip('/')
    sim_time = flag('use_sim_time')

    config = arg('config') or os.path.join(pkg_share, 'config', 'nvblox_voyis.yaml')

    # max_disparity bounds the *near* field: 80.42 px*m / 128 px = 0.63 m at
    # 704x704. Anything closer than that is rejected, which is what keeps SGM
    # from painting the backscatter and particulate drifting in front of the
    # dome into the map as solid surface. It also halves SGM cost versus the
    # 256 default, and nothing on a survey is that close anyway.
    max_disparity = float(arg('max_disparity'))
    width, height = int(arg('image_width')), int(arg('image_height'))

    to_rgb = [ComposableNode(
        name=f'to_rgb_{eye}',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'use_sim_time': sim_time,
            'encoding_desired': 'rgb8',
            'image_width': width,
            'image_height': height,
        }],
        remappings=[
            ('image_raw', f'{prefix}/{eye}/image_mono'),
            ('image', f'/nvblox/{eye}/image_rgb'),
        ],
    ) for eye in ('left', 'right')]

    disparity = ComposableNode(
        name='disparity',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
        parameters=[{
            'use_sim_time': sim_time,
            'backend': 'CUDA',
            'max_disparity': max_disparity,
            'window_size': 7,
            'num_passes': 2,
        }],
        remappings=[
            ('left/image_rect', '/nvblox/left/image_rgb'),
            ('left/camera_info', f'{prefix}/left/camera_info'),
            ('right/image_rect', '/nvblox/right/image_rgb'),
            ('right/camera_info', f'{prefix}/right/camera_info'),
        ],
    )

    disparity_to_depth = ComposableNode(
        name='disparity_to_depth',
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
        parameters=[{'use_sim_time': sim_time}],
    )

    # Colour is only worth asking for on an rgb8 bag -- demo_wreck_pass and
    # demo_wreck_long. The william_elgin exports are mono8, where this would
    # integrate a grey colour layer for nothing.
    #
    # The decode is done here, in-container, rather than by another
    # compressed_to_mono.py process: raw rgb8 at 704x704 is 1.49 MB and does not
    # survive the default transport. Composed, only the bag's ~150 KB JPEG is on
    # the wire.
    colour = []
    colour_remaps = []
    if flag('color'):
        colour = [ComposableNode(
            name='to_color_left',
            package='rhody',
            plugin='rhody::CompressedDecoderNode',
            parameters=[{'use_sim_time': sim_time, 'encoding': 'rgb8'}],
            remappings=[
                ('in/compressed', f'{prefix}/left/image_raw/compressed'),
                ('out', '/nvblox/left/image_color'),
            ],
        )]
        colour_remaps = [
            ('camera_0/color/image', '/nvblox/left/image_color'),
            ('camera_0/color/camera_info', f'{prefix}/left/camera_info'),
        ]

    # The depth image inherits the disparity header, so its frame_id is
    # voyis_left_optical and its intrinsics are the left camera's -- which is
    # why the left camera_info feeds the SGM node, nvblox depth and, when
    # enabled, nvblox colour. Depth and colour are already registered to each
    # other because both are the left rectified view.
    nvblox = ComposableNode(
        name='nvblox_node',
        package='nvblox_ros',
        plugin='nvblox::NvbloxNode',
        parameters=[config, {
            'use_sim_time': sim_time,
            'voxel_size': float(arg('voxel_size')),
            'use_color': flag('color'),
        }],
        remappings=[
            ('camera_0/depth/image', 'depth'),
            ('camera_0/depth/camera_info', f'{prefix}/left/camera_info'),
        ] + colour_remaps,
    )

    rviz = [Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'voyis_nvblox_demo.rviz')],
        parameters=[{'use_sim_time': sim_time}],
        additional_env=rviz_env(),
        condition=IfCondition(LaunchConfiguration('rviz')),
    )]

    return rviz + [ComposableNodeContainer(
        name='nvblox_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=colour + to_rgb + [disparity, disparity_to_depth, nvblox],
        output='screen',
        additional_env=isaac_env(),
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('prefix', default_value='/stereo',
                              description='namespace of the stereo topics'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='only when running this launch standalone; '
                                          'voyis_pipeline.launch.py owns RViz otherwise'),
        DeclareLaunchArgument('config', default_value='',
                              description='override rhody/config/nvblox_voyis.yaml'),
        DeclareLaunchArgument('voxel_size', default_value='0.05',
                              description='TSDF voxel edge in metres'),
        DeclareLaunchArgument('max_disparity', default_value='128.0',
                              description='SGM search range; floors the near range at f*B/max'),
        DeclareLaunchArgument('color', default_value='false',
                              description='integrate a colour layer; rgb8 bags only'),
        DeclareLaunchArgument('image_width', default_value='704',
                              description='must match the bag; 704 for --scale 0.25'),
        DeclareLaunchArgument('image_height', default_value='704'),
        OpaqueFunction(function=launch_setup),
    ])
