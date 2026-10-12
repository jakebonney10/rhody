#!/bin/bash
cd "$(dirname "$0")"
source $HOME/ros/rhody_ws/install/setup.bash

# Output location and naming match the hydroport recorder: one bag directory
# per start, ~/data/rhody_mcap/rhody_<YYYY-MM-DDTHHMMSSZ> (UTC). Overridable
# (systemd Environment=) so the unit can be pointed elsewhere without editing
# this script.
BAG_DIR=${RHODY_BAG_DIR:-$HOME/data/rhody_mcap}
BAG_PREFIX=${RHODY_BAG_PREFIX:-rhody_}
BAG_CONFIG=${RHODY_BAG_CONFIG:-$(ros2 pkg prefix --share rhody)/config/rosbag_record.yaml}

if ! mkdir -p "$BAG_DIR"; then
    echo "Cannot create $BAG_DIR; exiting for systemd to retry."
    exit 1
fi

# rosbag2 refuses to open a bag directory that already exists, and a restart
# can land in the same second as the bag it replaces.
uri="$BAG_DIR/${BAG_PREFIX}$(date -u +%FT%H%M%SZ)"
base="$uri"
i=1
while [ -e "$uri" ]; do
    uri="${base}_$i"
    i=$((i + 1))
done

echo "Recording to $uri (config $BAG_CONFIG)"
# exec the recorder itself rather than going through `ros2 run`: systemd then
# signals the node directly, which is what closes the bag and writes
# metadata.yaml on stop.
exec "$(ros2 pkg prefix rosbag2_transport)/lib/rosbag2_transport/recorder" --ros-args \
    --params-file "$BAG_CONFIG" \
    -p storage.uri:="$uri" \
    -r __ns:=/rhody/data
