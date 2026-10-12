#!/bin/bash
cd "$(dirname "$0")"
source $HOME/ros/rhody_ws/install/setup.bash

# The Nucleus must already be streaming on its read-only port before the node is
# worth starting: the driver's auto_connect is one-shot, so a node started while
# the device is unreachable sits alive forever publishing nothing. Exit non-zero
# instead and let systemd (Restart=always, RestartSec) retry until the device
# appears. Keep these in sync with config/nucleus_listen.yaml.
# Overridable so the unit can be pointed elsewhere (systemd Environment=) without
# editing this script, and so the probe can be exercised against a local stub.
NUCLEUS_IP=${NUCLEUS_IP:-192.168.2.201}
NUCLEUS_PORT=${NUCLEUS_PORT:-9002}

if ! timeout 3 bash -c "echo > /dev/tcp/${NUCLEUS_IP}/${NUCLEUS_PORT}" 2>/dev/null; then
    echo "Nucleus ${NUCLEUS_IP}:${NUCLEUS_PORT} unreachable; exiting for systemd to retry."
    exit 1
fi

echo "Nucleus ${NUCLEUS_IP}:${NUCLEUS_PORT} reachable; starting listen-only driver."
exec ros2 launch rhody nucleus_listen.launch.py
