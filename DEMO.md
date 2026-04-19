# Pallet 6D Pose Estimation — Demo Guide

End-to-end demo: live camera capture → YOLO + ICP pose estimation → RViz2 forklift trajectory.

---

## One-time setup

```bash
# Build trajectory_viz (only needed after code changes)
cd ~/repositories/pallet-6d-pose-estimation/trajectory_viz
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install

# Make script executable
chmod +x ~/repositories/pallet-6d-pose-estimation/run_pipeline.sh
```

---

## Demo — 3 terminals

### Terminal 1 — Main pipeline (start here)

```bash
cd ~/repositories/pallet-6d-pose-estimation
./run_pipeline.sh --live
```

**What opens:**
- RViz2 with the forklift simulation
- RGB camera window (live feed)
- Depth camera window (live feed)

**What it does:**
- Watches for new captures automatically
- Press `SPACE` in the camera window to capture and run the pipeline
- Press `ESC` to close the camera (RViz stays open)

---

### Terminal 2 — Ground truth pose

Run this after positioning the pallet at a known location.

```bash
source /opt/ros/jazzy/setup.bash
source ~/repositories/pallet-6d-pose-estimation/trajectory_viz/install/setup.bash
python3 ~/repositories/pallet-6d-pose-estimation/real_world_code/real_world_code/set_ground_truth.py \
    --x <meters> --y <meters> --yaw_deg <degrees>
```

Example:
```bash
python3 ~/repositories/pallet-6d-pose-estimation/real_world_code/real_world_code/set_ground_truth.py \
    --x 2.0 --y 0.1 --yaw_deg 5.0
```

The ground truth pallet (blue) appears in RViz immediately.

---

### Terminal 3 — Forklift animation control

```bash
source /opt/ros/jazzy/setup.bash
source ~/repositories/pallet-6d-pose-estimation/trajectory_viz/install/setup.bash
ros2 run trajectory_viz animation_control
```

| Key | Action |
|-----|--------|
| `r` | Run forklift animation once |
| `s` | Snap forklift back to start |
| `l` | Loop animation continuously |
| `e` | Snap to end pose |
| `q` | Quit |

---

## Full demo sequence

1. **Start Terminal 1** — wait for RViz and camera windows to open (~5 seconds)

2. **Start Terminal 3** — animation control ready in background

3. **Position the pallet** at a known location and measure its pose

4. **Terminal 2** — publish the ground truth pose. Ground truth pallet appears in RViz

5. **Press `SPACE`** in the camera window — pipeline runs automatically in Terminal 1

6. **Watch RViz** — estimated pallet pose appears, forklift trajectory updates

7. **Press `r`** in Terminal 3 — forklift animates toward the pallet

8. **Reset and repeat** — press `s` in Terminal 3 to send forklift back to start, move pallet to a new position, repeat from step 3

---

## Captured files

Each session saves to:
```
camera_capture/collected_data/session_YYYY-MM-DD_HH-MM-SS/
    frame_0000_rgb.png
    frame_0000_depth_meters.bin
    frame_0000_intrinsics.json
    frame_0000_depth.png
    frame_0000_depth_vis.png
```

Pipeline output goes to:
```
OUTPUT/
```

---

## Troubleshooting

**RViz shows wrong ground truth on launch**
Edit `trajectory_viz/config/pallet_poses.yaml` and relaunch — changes apply immediately (symlink install).

**Two simulation_node instances / flickering poses**
`run_pipeline.sh` kills stale instances automatically. If it persists: `pkill -f simulation_node && pkill -f rviz2`

**Pipeline doesn't trigger after SPACE**
Check `inotify-tools` is installed: `sudo apt install inotify-tools`

**Camera not detected**
Verify the camera adapter route is set: `ping 192.168.0.200`
If it fails: `ping -I enx000ec662e40e 192.168.0.200` — if this works, recheck the NetworkManager static route.

**No YOLO detection**
Lower `YOLO_CONF` in `real_world_code/real_world_code/main.py` (try `0.001`).

**ICP result looks wrong**
Check the Open3D visualizer window (appears after pipeline, closes manually).
Set `VISUALIZE_BEST = False` in `real_world_code/real_world_code/register_new_constraint_init.py` to skip it.
