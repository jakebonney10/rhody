import launch
from launch.substitutions import Command, LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
import launch_ros
import os
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare(package='rhody').find('rhody')
    default_namespace = 'rhody/nav'
    ekf_config = 'config/ekf_global.yaml'
    default_model_path = os.path.join(pkg_share, 'urdf/rhody2.urdf')
    default_rviz_config_path = os.path.join(pkg_share, 'rviz/ekf_global.rviz')

    # Include gps2map launch file
    gps2map_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'gps2map.launch.xml')
        )
    )

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

    robot_localization_node = launch_ros.actions.Node(
         package='robot_localization',
         executable='ekf_node',
         name='ekf_global',
         namespace=default_namespace,
         output='screen',
         parameters=[os.path.join(pkg_share, ekf_config), {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    foxglove_bridge_node = launch_ros.actions.Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='model', default_value=default_model_path, description='Absolute path to robot urdf file'),
        launch.actions.DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path, description='Absolute path to rviz config file'),
        launch.actions.DeclareLaunchArgument(name='use_sim_time', default_value='True', description='Flag to enable use_sim_time'),
        gps2map_launch,
        robot_state_publisher_node,
        robot_localization_node,
        foxglove_bridge_node,
        rviz_node,

    ])