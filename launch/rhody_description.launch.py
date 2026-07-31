"""
Publishes the rhody URDF (robot_state_publisher) and optionally opens RViz.

Defaults to the global namespace so /robot_description and /tf are visible to
RViz and to the sensor drivers without extra remapping. Pass namespace:=rhody/nav
to match the ekf_global / navigation launch files.
"""

import os

import launch
import launch_ros
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare(package='rhody')

    default_model_path = PathJoinSubstitution([pkg_share, 'urdf', 'rhody.urdf.xacro'])
    default_rviz_config_path = PathJoinSubstitution([pkg_share, 'rviz', 'rhody_description.rviz'])

    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(
                Command(['xacro ', LaunchConfiguration('model')]),
                value_type=str,
            ),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )

    # All joints in rhody.urdf.xacro are fixed, so joint_state_publisher is only
    # needed if articulated joints are added later.
    joint_state_publisher_node = launch_ros.actions.Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=LaunchConfiguration('namespace'),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('joint_state_publisher')),
    )

    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='namespace', default_value='',
            description='Namespace for robot_state_publisher (also namespaces /tf)'),
        launch.actions.DeclareLaunchArgument(
            name='model', default_value=default_model_path,
            description='Path to the robot xacro/urdf file'),
        launch.actions.DeclareLaunchArgument(
            name='rvizconfig', default_value=default_rviz_config_path,
            description='Path to the rviz config file'),
        launch.actions.DeclareLaunchArgument(
            name='rviz', default_value='true',
            description='Launch RViz alongside robot_state_publisher'),
        launch.actions.DeclareLaunchArgument(
            name='joint_state_publisher', default_value='false',
            description='Launch joint_state_publisher (not needed for fixed joints)'),
        launch.actions.DeclareLaunchArgument(
            name='use_sim_time', default_value='false',
            description='Use /clock instead of wall time'),
        robot_state_publisher_node,
        joint_state_publisher_node,
        rviz_node,
    ])
