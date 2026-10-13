#!/bin/bash
cd "$(dirname "$0")"
source $HOME/ros/rhody_ws/install/setup.bash
exec ros2 launch rhody ilbl.launch.py
