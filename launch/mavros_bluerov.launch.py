#!/usr/bin/env python3

"""
Launch file for MAVROS connection to BlueROV
Publishes IMU and pressure sensor data from the BlueROV flight controller
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directory
    pkg_share = FindPackageShare(package='rhody').find('rhody')
    
    # Default parameters
    default_namespace = 'rhody'
    default_fcu_url = 'udp://:14550@'
    default_gcs_url = ''
    default_config_file = os.path.join(pkg_share, 'config/mavros_bluerov.yaml')
    
    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value=default_namespace,
        description='Namespace for the MAVROS node'
    )
    
    fcu_url_arg = DeclareLaunchArgument(
        'fcu_url',
        default_value=default_fcu_url,
        description='FCU connection URL (e.g., udp://:14550@, /dev/ttyACM0:57600)'
    )
    
    gcs_url_arg = DeclareLaunchArgument(
        'gcs_url',
        default_value=default_gcs_url,
        description='GCS link URL (optional)'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Path to MAVROS configuration file'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    # MAVROS node
    mavros_node = Node(
        package='mavros',
        executable='mavros_node',
        name='mavros',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'fcu_url': LaunchConfiguration('fcu_url'),
                'gcs_url': LaunchConfiguration('gcs_url'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }
        ],
        # Remap topics to be more intuitive/standard
        remappings=[
            ('/diagnostics', '/rhody/diagnostics'),
            ('mavros/data', 'mavros/imu/data'),
            ('mavros/data_raw', 'mavros/imu/data_raw'),
            ('mavros/mag', 'mavros/imu/mag'),
            ('mavros/temperature_imu', 'mavros/imu/temperature'),
            ('mavros/static_pressure', 'mavros/imu/static_pressure'),
            ('mavros/diff_pressure', 'mavros/imu/diff_pressure'),
            ('mavros/temperature_baro', 'mavros/imu/temperature_baro'),
        ]
    )
    
    # Pressure to depth conversion node (if you have the python script)
    pressure2depth_node = Node(
        package='rhody',
        executable='pressure2depth.py',
        name='pressure2depth',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[
            ('pressure', 'mavros/imu/static_pressure'),
            ('depth', 'depth'),
        ]
    )
    
    return LaunchDescription([
        namespace_arg,
        fcu_url_arg,
        gcs_url_arg,
        config_file_arg,
        use_sim_time_arg,
        mavros_node,
        pressure2depth_node,
    ])
