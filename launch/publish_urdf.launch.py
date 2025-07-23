import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'rhody'
    xacro_file_name = 'rhody.urdf.xacro'

    pkg_dir = get_package_share_directory(pkg_name)
    default_xacro_path = os.path.join(pkg_dir, 'urdf', xacro_file_name)

    # Declare a launch argument for the xacro file
    declare_xacro_file_path = DeclareLaunchArgument(
        'xacro_file',
        default_value=default_xacro_path,
        description='Path to the robot xacro file'
    )

    # Corrected xacro command — DO NOT put 'xacro ' in one string
    robot_description = Command([
        'xacro', ' ',
        LaunchConfiguration('xacro_file')
    ])

    # Node to publish the robot description
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        namespace='rhody',
        parameters=[{
            'robot_description': ParameterValue(robot_description, value_type=str)
        }]
    )

    return LaunchDescription([
        declare_xacro_file_path,
        robot_state_publisher_node
    ])
