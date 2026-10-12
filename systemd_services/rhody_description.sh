#!/bin/bash
cd "$(dirname "$0")"
source $HOME/ros/rhody_ws/install/setup.bash

# model is passed explicitly: rhody_description.launch.py still defaults to
# rhody.urdf.xacro, which is the *Rhody 1* description. rhody2 is this vehicle.
# rviz:=false because this runs headless; fov stays off so the runtime link set
# is the real sensor set, not the visual coverage volumes.
SHARE="$(ros2 pkg prefix --share rhody)"
exec ros2 launch rhody rhody_description.launch.py \
    model:="$SHARE/urdf/rhody2.urdf.xacro" \
    rviz:=false
