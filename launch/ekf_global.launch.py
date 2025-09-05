import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
import os
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare(package='rhody').find('rhody')
    default_namespace = 'rhody/nav'
    ekf_config = 'config/ekf_global.yaml'

    robot_localization_node = launch_ros.actions.Node(
         package='robot_localization',
         executable='ekf_node',
         name='ekf_global',
         namespace=default_namespace,
         output='screen',
         parameters=[os.path.join(pkg_share, ekf_config), {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='use_sim_time', default_value='True', description='Flag to enable use_sim_time'),
        robot_localization_node,
    ])