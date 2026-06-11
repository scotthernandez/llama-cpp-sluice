#!/bin/bash
# setup-lxc.sh: GPU Passthrough Helper for Proxmox LXC

CONTAINER_ID=$1

if [ -z "$CONTAINER_ID" ]; then
    echo "Usage: sudo ./setup-lxc.sh <LXC_ID>"
    exit 1
fi

echo "[SLUICE] Configuring GPU Passthrough for LXC $CONTAINER_ID..."

# 1. Get GIDs from host
RENDER_GID=$(getent group render | cut -d: -f3)
VIDEO_GID=$(getent group video | cut -d: -f3)

CONF_FILE="/etc/pve/lxc/${CONTAINER_ID}.conf"

if [ ! -f "$CONF_FILE" ]; then
    echo "[ERROR] Container config not found at $CONF_FILE"
    exit 1
fi

# 2. Append GPU nodes
cat <<EOF >> "$CONF_FILE"

# SLUICE: GPU Passthrough
lxc.cgroup2.devices.allow: c 195:* rwm
lxc.cgroup2.devices.allow: c 226:* rwm
lxc.cgroup2.devices.allow: c 234:* rwm
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
lxc.mount.entry: /dev/dri dev/dri none bind,optional,create=dir
EOF

echo "[SUCCESS] GPU nodes added to $CONF_FILE."
echo "Please restart the container and ensure the 'render' group inside has GID $RENDER_GID."
