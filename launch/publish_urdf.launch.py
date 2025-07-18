import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution

def generate_launch_description():
    # Define the package name
    pkg_name = 'rhody'
    # Define the xacro file name
    xacro_file_name = 'rhody.urdf.xacro'

    # Get the package directory
    pkg_dir = get_package_share_directory(pkg_name)

    # Define the path to the xacro file
    xacro_file_path = os.path.join(pkg_dir, 'urdf', xacro_file_name)

    # Declare launch arguments
    declare_xacro_file_path = DeclareLaunchArgument(
        'xacro_file', default_value=xacro_file_path,
        description='Full path to the robot xacro file')

    # Use xacro to convert xacro file to URDF
    robot_description = Command(
        ['xacro ', LaunchConfiguration('xacro_file')])

    # Node to publish URDF to /robot_description
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        namespace='rhody',
        parameters=[{'robot_description': robot_description}],
    )

    return LaunchDescription([
        declare_xacro_file_path,
        robot_state_publisher_node,
    ])
