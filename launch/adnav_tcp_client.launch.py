import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    config = os.path.join(
        get_package_share_directory('rhody'),
        'config',
        'adnav_tcp_client.yaml'
        )

    # name 'adnav' under namespace 'rhody/nav/sensors' -> node and topics publish
    # at /rhody/nav/sensors/adnav/* (the driver prefixes every topic with the
    # node name).
    node=Node(
        name = 'adnav',
        namespace = 'rhody/nav/sensors',
        package = 'adnav_driver',
        executable = 'adnav_driver',
        emulate_tty = True,
        output = 'screen',
        # arguments=['--ros-args', '--log-level', 'debug'],
        parameters = [config]
    )

    ld.add_action(node)
    return ld
