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

cp systemd_services/* "$DIR"
chmod +x "$DIR"/*.sh

systemctl --user daemon-reload

systemctl --user enable rhody-sprintnavmini.service
systemctl --user restart rhody-sprintnavmini.service

systemctl --user enable rhody-adnav.service
systemctl --user restart rhody-adnav.service

# Allow user services to run at boot without an active login session (the
# robot boots headless). Only needs to succeed once.
sudo loginctl enable-linger "$USER"

echo "Done. Verify with:"
echo "  systemctl --user status rhody-sprintnavmini.service"
echo "  journalctl --user -u rhody-sprintnavmini.service -f"
echo "  systemctl --user status rhody-adnav.service"
echo "  journalctl --user -u rhody-adnav.service -f"
