# trajectory_viz

RViz2 visualization for validating the pallet 6D pose estimation pipeline. Displays a forklift at a known pose, the ground truth pallet pose, and — once the perception pipeline publishes an estimate — the estimated pallet pose and a generated approach trajectory. The forklift can then be animated along the trajectory manually or triggered by a ROS message.

## Prerequisites

- ROS2 Jazzy
- `rviz2` and `nav2` packages (standard ROS2 desktop install)

## Build and Launch

```bash
cd trajectory_viz
./scripts/launch.sh
```

This builds, sources, and launches RViz in one step. On subsequent runs from the same terminal you can also do it manually:

```bash
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
ros2 launch trajectory_viz viz.launch.py
```

## What You'll See

### At startup

| Element | Description |
|---|---|
| Orange forklift mesh | Forklift at the world origin (front axle = camera position) |
| Brown pallet mesh | Ground truth pallet pose (manually configured) |
| Blue floor tape lines | FOV reference lines at 0°, ±30° from the camera |
| TF axes | `world`, `forklift`, `pallet_gt` frames only |

### After estimated pose is received

| Element | Description |
|---|---|
| Blue transparent pallet | Estimated pallet pose from the perception pipeline |
| Green path line | Two-phase approach trajectory: arc-blend curve to approach pose, then straight insertion |
| TF axes | `pallet_est` and `pickup` frames added |

## Coordinate System

All poses are in the **world frame**, whose origin is the camera (fork face). The forklift front axle starts at `x = -CAMERA_X_OFFSET` so the camera sits at (0, 0).

- `+X` — forward (away from the forklift, toward the pallet)
- `+Y` — left
- `+Z` — up

Pallet `+X` faces the forklift. A pallet at `x=2.0` is 2 m directly in front of the camera.

## Pipeline Integration

The visualization is driven by two ROS topics published by the perception pipeline.

### Estimated pallet pose

**Preferred — plain x, y, yaw (degrees):**

```
Topic:  /est_pallet_pose_2d
Type:   geometry_msgs/msg/Pose2D
Fields:
  x      — metres forward from camera
  y      — metres lateral from camera (positive = left)
  theta  — yaw in DEGREES (positive = CCW when viewed from above)
```

Example:
```bash
ros2 topic pub --once /est_pallet_pose_2d geometry_msgs/msg/Pose2D \
  "{x: 2.0, y: 0.1, theta: 12.0}"
```

**Alternative — full PoseStamped with quaternion:**

```
Topic:  /est_pallet_pose_in
Type:   geometry_msgs/msg/PoseStamped
Frame:  world (camera origin)
```

### Ground truth pallet pose (optional override)

```
Topic:  /gt_pallet_pose_in
Type:   geometry_msgs/msg/PoseStamped
Frame:  world (camera origin)
```

If not published, the ground truth falls back to the values in `config/pallet_poses.yaml`.

### Animation trigger

Once the estimated pose is received and the trajectory is visible, trigger the forklift motion:

```
Topic:  /animation/run_once   — run trajectory once; republish to replay
Topic:  /animation/loop       — loop continuously
Topic:  /animation/goto_start — snap forklift to start pose (no animation)
Topic:  /animation/goto_end   — snap forklift to pickup pose (no animation)
Type:   std_msgs/msg/Empty
```

```bash
ros2 topic pub --once /animation/run_once std_msgs/msg/Empty '{}'
```

## Keyboard Controller

Launch in a second terminal for manual control:

```bash
source install/setup.bash
ros2 run trajectory_viz animation_control
```

| Key | Action |
|-----|--------|
| `r` | Run once |
| `l` | Continuous loop |
| `s` | Snap to start pose |
| `e` | Snap to end (pickup) pose |
| `q` / Ctrl-C | Quit |

## Configuration

### Pallet poses — runtime (no restart needed)

Edit `config/pallet_poses.yaml` then load it while the node is running:

```bash
ros2 param load /simulation_node config/pallet_poses.yaml
```

All 6 pose parameters update atomically and the trajectory recomputes immediately. You can also set individual parameters:

```bash
ros2 param set /simulation_node est_pallet_x 2.5
ros2 param set /simulation_node est_pallet_yaw_deg -5.0
```

### Motion profile and planner — static

Edit the class constants at the top of `trajectory_viz/simulation_node.py` (requires rebuild):

```python
ANIM_LINEAR_SPEED  = 0.8   # m/s   — top speed on straights
ANIM_MAX_ACCEL     = 0.25  # m/s²  — acceleration / deceleration limit
ANIM_MAX_TURN_RATE = 0.30  # rad/s — max heading-change rate while driving
ANIM_ANGULAR_SPEED = 0.20  # rad/s — in-place rotation speed
APPROACH_PAUSE     = 1.5   # s     — dwell at approach pose before insertion

PLANNER = 'arc_blend'   # recommended — cubic Bezier, blends turning into forward motion
# PLANNER = 'diff_drive' — rotate in place → straight → rotate in place
# PLANNER = 'dubins'     — car-like arc-straight-arc (set DUBINS_RADIUS)
```
