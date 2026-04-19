#!/usr/bin/env bash
# Build, source, and launch the trajectory_viz RViz simulation.
# Run from anywhere inside the repo:
#   bash trajectory_viz/scripts/launch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PKG_DIR"

source /opt/ros/jazzy/setup.bash

echo "==> Building..."
colcon build 2>&1

echo "==> Sourcing install..."
source install/setup.bash

echo "==> Launching..."
ros2 launch trajectory_viz viz.launch.py
