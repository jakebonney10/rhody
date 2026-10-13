#!/usr/bin/env python3

"""
Launch file for the inverted LBL (iLBL) vehicle position feed.

The iLBL topside sends the vehicle's position as NMEA GGA sentences over UDP.
Two roship nodes turn that into a NavSatFix:

  1. transport  roship_io/udp_connection   /rhody/connections/ilbl
                binds the UDP port and republishes each datagram as a
                RawPacket on ~/from_device.
  2. parser     roship_vessel/nmea_gga     /rhody/nav/sensors/ilbl
                subscribes to that RawPacket topic, keeps only GGA sentences
                (message_filter), and publishes sensor_msgs/NavSatFix on
                ~/fix, i.e. /rhody/nav/sensors/ilbl/fix.

Both nodes read config/ilbl.yaml. The iLBL's IP and port are placeholders in
that file until the topside is on the bench.

Both nodes carry on_exit=Shutdown(): if either dies (the usual case being the
transport failing to bind a held port and aborting) the whole launch exits, so
systemd's Restart=always actually fires instead of leaving a half-alive pair.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('rhody'),
        'config',
        'ilbl.yaml'
    )

    transport = Node(
        package='roship_io',
        executable='udp_connection',
        name='ilbl',
        namespace='rhody/connections',
        output='screen',
        parameters=[config],
        on_exit=Shutdown(),
    )

    parser = Node(
        package='roship_vessel',
        executable='nmea_gga',
        name='ilbl',
        namespace='rhody/nav/sensors',
        output='screen',
        parameters=[config],
        on_exit=Shutdown(),
    )

    return LaunchDescription([
        transport,
        parser,
    ])
