#!/usr/bin/env bash
# Restore the GLSL 150 shaders that nvblox_rviz_plugin needs.
#
# ros-humble-rviz-rendering 11.2.23 deleted ogre_media/materials/{glsl150,scripts150}
# (https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/issues/159). The nvblox
# RViz plugin still asks rviz_rendering for them -- it calls
# rviz_rendering::get_resource_directory(), registers those two directories as
# an Ogre resource group named "rviz_rendering_150", and looks up the geometry
# program "rviz/glsl150/box.geom" when it first draws a mesh block. With the
# directories gone RViz does not warn, it aborts:
#
#   ItemIdentityException: Unable to locate geometry program called rviz/glsl150/box.geom
#   terminate called after throwing an instance of 'Ogre::ItemIdentityException'
#
# It only fires once mesh geometry is actually on screen, so a session where the
# mesh is off-camera survives and one where it is in view dies, which makes it
# look intermittent rather than deterministic.
#
# This restores the two directories from the last release that shipped them,
# 11.2.22. It only adds files -- nothing existing is modified or removed, and
# RViz's own rendering is untouched because only the nvblox plugin registers the
# "rviz_rendering_150" group.
#
#     ./install_rviz_glsl150.sh          # install (prompts for sudo)
#     ./install_rviz_glsl150.sh --check  # report status only, no changes
#
# Re-run after any `apt upgrade` that touches ros-humble-rviz-rendering: a
# reinstall of the package does not restore these, and a newer version will not
# either -- they are gone for good upstream.

set -euo pipefail

RVIZ_TAG=11.2.22
DEST=/opt/ros/humble/share/rviz_rendering/ogre_media/materials
URL="https://github.com/ros2/rviz/archive/refs/tags/${RVIZ_TAG}.tar.gz"

if [[ -f "${DEST}/glsl150/box.geom" && -d "${DEST}/scripts150" ]]; then
    echo "already installed: ${DEST}/glsl150/box.geom"
    exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
    echo "MISSING: ${DEST}/glsl150 and/or ${DEST}/scripts150"
    echo "the nvblox RViz mesh display will crash RViz until these are restored"
    exit 1
fi

if [[ ! -d /opt/ros/humble/share/rviz_rendering ]]; then
    echo "rviz_rendering not found -- is ROS 2 Humble installed?" >&2
    exit 1
fi

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT

echo "fetching rviz ${RVIZ_TAG} ..."
curl -fsSL -o "${work}/rviz.tar.gz" "${URL}"

src="rviz-${RVIZ_TAG}/rviz_rendering/ogre_media/materials"
tar xzf "${work}/rviz.tar.gz" -C "${work}" "${src}/glsl150" "${src}/scripts150"

# Fail loudly rather than installing an empty directory if upstream ever moves
# these again.
test -f "${work}/${src}/glsl150/box.geom"

echo "installing into ${DEST} (sudo) ..."
sudo cp -rn "${work}/${src}/glsl150" "${work}/${src}/scripts150" "${DEST}/"

echo "done:"
ls "${DEST}/glsl150" "${DEST}/scripts150"
