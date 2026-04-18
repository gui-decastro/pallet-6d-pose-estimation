"""
publish_to_viz.py
=================
Sends the estimated pallet pose to whichever visualization backend is
available, in priority order:

  1. ROS2 / RViz2 (trajectory_viz simulation node)
     Publishes to /est_pallet_pose_in and /animation/run_once.
     Requires rclpy and the simulation node to be running.

  2. Standalone matplotlib (viz_standalone.py)
     Launched as a background subprocess — the window stays open while the
     pipeline continues.  No ROS2 required; only needs matplotlib + numpy.

Convention (matches simulation_node._pose_from_xyyaw):
  x, y     — pallet position in the world/camera frame (metres)
               X = forward from fork face, Y = lateral
  yaw_rad  — raw ICP yaw (radians); the simulation node adds π internally
"""

import math
import subprocess
import sys
import time
from pathlib import Path

_VIZ_SCRIPT = str(Path(__file__).parent / "viz_standalone.py")


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 1: ROS2 / RViz2
# ─────────────────────────────────────────────────────────────────────────────

def _try_ros2(x: float, y: float, yaw_rad: float,
              wait_for_sub_sec: float = 2.0) -> bool:
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Empty
    except ImportError:
        return False

    try:
        if not rclpy.ok():
            rclpy.init()

        node = rclpy.create_node("pallet_pose_publisher_oneshot")
        pose_pub = node.create_publisher(PoseStamped, "/est_pallet_pose_in", 10)
        anim_pub = node.create_publisher(Empty,       "/animation/run_once",  10)

        msg = PoseStamped()
        msg.header.stamp    = node.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(yaw_rad / 2.0)
        msg.pose.orientation.w = math.cos(yaw_rad / 2.0)

        deadline = time.monotonic() + wait_for_sub_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if pose_pub.get_subscription_count() > 0:
                break

        pose_pub.publish(msg)
        anim_pub.publish(Empty())
        rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()

        print(f"[viz/ros2] Pose published  →  "
              f"x={x:.3f} m  y={y:.3f} m  yaw={math.degrees(yaw_rad):.1f}°")
        print("[viz/ros2] Animation triggered on /animation/run_once")
        return True

    except Exception as exc:
        print(f"[viz/ros2] Failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 2: standalone matplotlib (viz_standalone.py)
# ─────────────────────────────────────────────────────────────────────────────

def _try_standalone(x: float, y: float, yaw_rad: float) -> bool:
    if not Path(_VIZ_SCRIPT).exists():
        print(f"[viz/standalone] Script not found: {_VIZ_SCRIPT}")
        return False

    yaw_deg = math.degrees(yaw_rad)
    cmd = [
        sys.executable, _VIZ_SCRIPT,
        "--x",       str(x),
        "--y",       str(y),
        "--yaw_deg", f"{yaw_deg:.6f}",
    ]

    try:
        # Non-blocking — the window stays open while the pipeline continues.
        subprocess.Popen(cmd)
        print(f"[viz/standalone] Visualization launched  →  "
              f"x={x:.3f} m  y={y:.3f} m  yaw={yaw_deg:.1f}°")
        return True
    except Exception as exc:
        print(f"[viz/standalone] Failed to launch: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def publish_pose(x: float, y: float, yaw_rad: float) -> bool:
    """
    Send the estimated pallet pose to the best available visualization backend.

    Tries ROS2/RViz2 first; falls back to a standalone matplotlib window.

    Args:
        x:       pallet X in world/camera frame (metres, forward)
        y:       pallet Y in world/camera frame (metres, lateral)
        yaw_rad: pallet yaw (radians, rotation about Z axis)

    Returns True if at least one backend succeeded.
    """
    if _try_ros2(x, y, yaw_rad):
        return True

    print("[viz] rclpy unavailable — falling back to standalone matplotlib viz.")
    return _try_standalone(x, y, yaw_rad)
