#!/bin/bash
cd "$(dirname "$0")"
source $HOME/ros/rhody_ws/install/setup.bash
ros2 launch rhody adnav_tcp_client.launch.py
