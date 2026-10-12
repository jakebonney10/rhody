#!/bin/bash
cd "$(dirname "$0")"

# Rhody services run as user units under the current login. Copy the unit(s)
# and their launch wrappers into the user systemd directory.
DIR="$HOME/.config/systemd/user/"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    # The directory does not exist, so create it
    echo "Directory does not exist. Creating now..."
    mkdir -p "$DIR"
    echo "Directory created at $DIR"
else
    # The directory exists
    echo "Directory already exists at $DIR"
fi

# nucleus_driver_ros2 imports the plain-Python nucleus_driver library that ships
# alongside it in the submodule, but that library is never installed by colcon
# (it is a pip package, not an ament one) and this image has no pip and is PEP
# 668 externally managed. Put its src/ on the user's import path with a .pth
# file instead: no sudo, no pip, and it tracks the submodule checkout. Without
# this the node dies instantly with ModuleNotFoundError: nucleus_driver.
NUCLEUS_SRC="$(cd "$(dirname "$0")/../nucleus_driver/nucleus_driver/src" && pwd)"
USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())')"
if [ -d "$NUCLEUS_SRC" ]; then
    mkdir -p "$USER_SITE"
    echo "$NUCLEUS_SRC" > "$USER_SITE/nucleus_driver.pth"
    echo "nucleus_driver on import path via $USER_SITE/nucleus_driver.pth"
    python3 -c "import nucleus_driver" 2>/dev/null \
        && echo "  import nucleus_driver: OK" \
        || echo "  WARNING: import nucleus_driver still failing"
else
    echo "WARNING: $NUCLEUS_SRC not found - is the nucleus_driver submodule checked out?"
fi

cp systemd_services/* "$DIR"
chmod +x "$DIR"/*.sh

systemctl --user daemon-reload

systemctl --user enable rhody-sprintnavmini.service
systemctl --user restart rhody-sprintnavmini.service

systemctl --user enable rhody-adnav.service
systemctl --user restart rhody-adnav.service

systemctl --user enable rhody-mavros.service
systemctl --user restart rhody-mavros.service

# Vehicle TF tree. Nothing geometric works without this: no sensor frame
# resolves and recorded bags carry no extrinsics.
systemctl --user enable rhody-description.service
systemctl --user restart rhody-description.service

# Nortek Nucleus 1000 DVL, listen-only. The wrapper refuses to start while the
# device is unreachable, so expect this one to sit in auto-restart (activating)
# until the Nucleus is powered and streaming — that is normal, not a failure.
systemctl --user enable rhody-nucleus.service
systemctl --user restart rhody-nucleus.service

# MCAP bag recorder (stock rosbag2). Enabled so it is up on every boot, but it
# comes up paused and writes nothing until an operator calls
# /rhody/data/rosbag2_recorder/resume, so enabling it does not start filling
# the disk. It replaces the hydroport recorder, which is left installed but
# disabled: the two must not run together (both record every topic).
systemctl --user disable --now rhody-hydroport.service
systemctl --user enable rhody-rosbag.service
systemctl --user restart rhody-rosbag.service

# Allow user services to run at boot without an active login session (the
# robot boots headless). Only needs to succeed once.
sudo loginctl enable-linger "$USER"

# Kernel UDP socket buffers. Fast DDS sizes its sockets from the kernel default
# and the stock 208 KB overflowed on the recorder (2 MB sonar images and point
# clouds). See system/60-rhody-udp.conf for the numbers. Processes only pick the
# new default up when their sockets are created, so the restarts above matter.
sudo install -m 644 system/60-rhody-udp.conf /etc/sysctl.d/60-rhody-udp.conf
sudo sysctl -q -p /etc/sysctl.d/60-rhody-udp.conf

echo "Done. Verify with:"
echo "  systemctl --user status rhody-sprintnavmini.service"
echo "  journalctl --user -u rhody-sprintnavmini.service -f"
echo "  systemctl --user status rhody-adnav.service"
echo "  journalctl --user -u rhody-adnav.service -f"
echo "  systemctl --user status rhody-mavros.service"
echo "  journalctl --user -u rhody-mavros.service -f"
echo "  systemctl --user status rhody-rosbag.service"
echo "  journalctl --user -u rhody-rosbag.service -f"
echo "  systemctl --user status rhody-nucleus.service"
echo "  journalctl --user -u rhody-nucleus.service -f"
echo "  systemctl --user status rhody-description.service"
echo "  ros2 run tf2_tools view_frames   # confirm the tree is one connected piece"
