import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
import os
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare(package='rhody').find('rhody')
    default_namespace = 'rhody/nav'
    default_model_path = os.path.join(pkg_share, 'urdf/rhody2.urdf.xacro')
    default_rviz_config_path = os.path.join(pkg_share, 'rviz/ekf_global.rviz')

    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=default_namespace,
        parameters=[{
            'robot_description': ParameterValue(
                Command(['xacro ', LaunchConfiguration('model')]),
                value_type=str
            )
        }, {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
    )

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='model', default_value=default_model_path, description='Absolute path to robot urdf file'),
        launch.actions.DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path, description='Absolute path to rviz config file'),
        launch.actions.DeclareLaunchArgument(name='use_sim_time', default_value='True', description='Flag to enable use_sim_time'),
        robot_state_publisher_node,
        rviz_node,
    ])