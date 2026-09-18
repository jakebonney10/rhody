#!/bin/bash
cd "$(dirname "$0")"
source $HOME/ros/rhody_ws/install/setup.bash
ros2 run foxglove_bridge foxglove_bridge
