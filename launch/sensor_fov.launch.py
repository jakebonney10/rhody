"""Show the Rhody 2 sensor coverage volumes in RViz.

    ros2 launch rhody sensor_fov.launch.py

Thin wrapper over rhody_description.launch.py that pins the three things this
view needs together: the rhody2 description (Rhody 1 has no FOV block),
fov:=true, and the rhody_fov RViz config. Without the wrapper you have to pass
all three by hand and it is easy to get a model/config mismatch that silently
shows nothing.

For the offscreen figure instead of the live view, see
scripts/render_fov_figure.py.
"""

import launch
import launch_ros
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare(package='rhody')

    description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_share, 'launch',
                                  'rhody_description.launch.py']),
        ]),
        launch_arguments={
            'model': PathJoinSubstitution([pkg_share, 'urdf',
                                           'rhody2.urdf.xacro']),
            'rvizconfig': PathJoinSubstitution([pkg_share, 'rviz',
                                                'rhody_fov.rviz']),
            'fov': 'true',
            'rviz': LaunchConfiguration('rviz'),
            'namespace': LaunchConfiguration('namespace'),
        }.items(),
    )

    return launch.LaunchDescription([
        DeclareLaunchArgument(
            name='rviz', default_value='true',
            description='Open RViz (set false to only publish the description)'),
        DeclareLaunchArgument(
            name='namespace', default_value='',
            description='Namespace for robot_state_publisher (also namespaces /tf)'),
        description,
    ])
