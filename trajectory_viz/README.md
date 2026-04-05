# trajectory_viz

RViz2 visualization for validating the pallet 6D pose estimation pipeline. Displays a forklift at a known pose, the estimated pallet pose, and a generated approach path. The forklift can be animated along the trajectory manually or driven by a ROS message from the broader pipeline.

## Prerequisites

- ROS2 Jazzy
- `rviz2` and `nav2` packages (standard ROS2 desktop install)

## Build

```bash
cd trajectory_viz
source /opt/ros/jazzy/setup.bash
colcon build
```

## Launch

```bash
source install/setup.bash
ros2 launch trajectory_viz viz.launch.py
```

## What You'll See

| Element | Description |
|---|---|
| Orange forklift mesh | Forklift at its current pose |
| Blue ghost forklift | Target pickup pose (where forks fully insert) |
| Brown pallet mesh + green arrow | Estimated pallet pose; arrow shows fork-entry axis |
| Green path line | Two-phase approach trajectory |
| Colored TF axes | `world`, `forklift`, `pallet`, and `pickup` frames |

## Animating the Forklift

The forklift starts stationary at the origin. Trigger animation from a second terminal using one of two modes.

### Mode 1 — Run Once

The forklift moves through the full trajectory and stops at the pickup pose. Publish again to replay from the start.

```bash
ros2 topic pub --once /animation/run_once std_msgs/msg/Empty '{}'
```

### Mode 2 — Continuous Loop

The forklift moves through the trajectory and immediately restarts from the origin, repeating indefinitely.

```bash
ros2 topic pub --once /animation/loop std_msgs/msg/Empty '{}'
```

You can switch modes at any time, including mid-animation. Publishing `/animation/loop` while in run-once mode (or vice versa) takes effect on the next restart.

### Triggering from the Pipeline

Both topics accept `std_msgs/msg/Empty`, so any ROS2 node in the pipeline can trigger animation by publishing to `/animation/run_once` or `/animation/loop` — no changes to this package needed.

## Planner Selection

Edit `PLANNER` in `trajectory_viz/simulation_node.py`:

| Value | Behaviour |
|---|---|
| `'diff_drive'` | Rotate in place → drive straight → rotate in place. Supports reverse. |
| `'dubins'` | Car-like arc-straight-arc path with minimum turn radius `DUBINS_RADIUS`. |

```python
PLANNER       = 'diff_drive'
DUBINS_RADIUS = 1.0   # metres; only used when PLANNER = 'dubins'
```

## Swapping in Real Poses

Edit the two pose dicts in `__init__` inside `trajectory_viz/simulation_node.py`:

```python
self.forklift_pose = {
    'x': 0.0, 'y': 0.0, 'z': 0.0,
    'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0
}
self.pallet_pose = {
    'x': 6.0, 'y': 2.0, 'z': 0.0,
    'qx': 0.0, 'qy': 0.0, 'qz': ..., 'qw': ...
}
```

The approach path, pickup pose, and all markers recompute automatically on the next build.
