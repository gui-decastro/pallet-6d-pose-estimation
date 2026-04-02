# trajectory_viz

RViz2 visualization for validating the pallet 6D pose estimation pipeline. Displays a forklift at a known pose, the estimated pallet pose, and a generated approach path.

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
| Orange box + yellow forks | Forklift at its known pose |
| Brown flat box + green arrow | Estimated pallet pose; arrow shows fork-entry axis |
| Green curved line | Two-phase approach path |
| Blue arrows on path | Per-waypoint forklift heading along the route |
| Colored TF axes | `world`, `forklift`, and `pallet` frames |

## Swapping in Real Poses

Edit the two pose dicts at the top of `trajectory_viz/simulation_node.py`:

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

The approach path and all markers recompute automatically on the next build.
