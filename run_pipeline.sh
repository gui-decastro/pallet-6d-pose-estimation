#!/usr/bin/env bash
# Full pipeline: launches RViz2, then runs pose estimation.
#
# File-based (existing captured data):
#   ./run_pipeline.sh --rgb <path> --depth_bin <path> --intrinsics <path> --out_dir <path> [--pick farthest|largest]
#
# Live camera (demo mode):
#   ./run_pipeline.sh --live
#   C++ capture window opens. SPACE to capture, ESC to quit.
#   Pipeline triggers automatically after each capture.
#
# RViz2 opens immediately in both modes.

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAJ_DIR="$REPO_DIR/trajectory_viz"
PIPELINE_DIR="$REPO_DIR/real_world_code/real_world_code"
CAPTURE_DIR="$REPO_DIR/camera_capture/collected_data"
CAPTURE_BIN="$REPO_DIR/camera_capture/d400e_cpp/build/capture_d400e"
VENV_PYTHON="$REPO_DIR/venv/bin/python"

# ── 1. Source ROS2 + trajectory_viz ──────────────────────────────────────────
source /opt/ros/jazzy/setup.bash
source "$TRAJ_DIR/install/setup.bash"

# ── 2. Kill any stale instances, then launch RViz2 ───────────────────────────
pkill -f simulation_node 2>/dev/null || true
pkill -f rviz2            2>/dev/null || true
sleep 1

echo "==> Launching RViz2..."
ros2 launch trajectory_viz viz.launch.py &
RVIZ_PID=$!

# Kill RViz when this script exits (Ctrl-C or normal finish)
trap "kill $RVIZ_PID 2>/dev/null; exit" INT TERM EXIT

# Give RViz time to open before the pipeline starts printing
sleep 4

# ── 3. Run pose estimation pipeline ──────────────────────────────────────────
if [[ "$1" == "--live" ]]; then
    mkdir -p "$CAPTURE_DIR"

    # Watch for new intrinsics.json (last file written per capture) and trigger pipeline
    inotifywait -m -r --format '%w%f' -e close_write "$CAPTURE_DIR" 2>/dev/null \
    | while read filepath; do
        if [[ "$filepath" == *"_intrinsics.json" ]]; then
            base="${filepath%_intrinsics.json}"
            echo ""
            echo "==> New capture detected — running pipeline..."
            cd "$PIPELINE_DIR"
            "$VENV_PYTHON" main.py \
                --rgb        "${base}_rgb.png" \
                --depth_bin  "${base}_depth_meters.bin" \
                --intrinsics "${base}_intrinsics.json" \
                --out_dir    "$REPO_DIR/OUTPUT"
        fi
    done &
    WATCHER_PID=$!
    trap "kill $RVIZ_PID $WATCHER_PID 2>/dev/null; exit" INT TERM EXIT

    echo "==> Starting camera capture (SPACE=capture, ESC=quit)..."
    "$CAPTURE_BIN"

else
    cd "$PIPELINE_DIR"
    echo "==> Running pose estimation pipeline..."
    "$VENV_PYTHON" main.py "$@"
fi

# Keep RViz open until the user closes it or hits Ctrl-C
echo ""
echo "RViz2 is still open — press Ctrl-C to exit."
wait $RVIZ_PID
