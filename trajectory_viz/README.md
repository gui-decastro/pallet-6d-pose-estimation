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

The forklift starts stationary at the origin. There are two ways to control the animation.

### Option A — Keyboard controller (manual)

Launch the animation controller in a second terminal:

```bash
source install/setup.bash
ros2 run trajectory_viz animation_control
```

| Key | Action |
|-----|--------|
| `r` | Run once — move through the trajectory and stop at the pickup pose. Press again to replay. |
| `l` | Continuous loop — replay the trajectory indefinitely. |
| `q` / Ctrl-C | Quit. |

### Option B — ROS topic (manual or from pipeline)

Publish directly to the animation topics from any terminal or ROS2 node:

```bash
# Run once
ros2 topic pub --once /animation/run_once std_msgs/msg/Empty '{}'

# Continuous loop
ros2 topic pub --once /animation/loop std_msgs/msg/Empty '{}'
```

Both topics accept `std_msgs/msg/Empty`, so the pipeline can trigger animation without any changes to this package.

You can switch modes at any time, including mid-animation.

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
