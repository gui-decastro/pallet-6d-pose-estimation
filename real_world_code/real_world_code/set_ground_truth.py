"""
set_ground_truth.py
===================
Publish a known ground truth pallet pose to the trajectory_viz simulation.

Run from a terminal where ROS2 is sourced (i.e. after run_pipeline.sh has
started, open a second terminal and source ROS2 there):

    source /opt/ros/jazzy/setup.bash
    source ~/repositories/pallet-6d-pose-estimation/trajectory_viz/install/setup.bash
    python3 set_ground_truth.py --x 2.0 --y 0.1 --yaw_deg 5.0
"""

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped


def main():
    parser = argparse.ArgumentParser(description="Publish ground truth pallet pose to RViz")
    parser.add_argument("--x",       type=float, required=True,  help="X position (metres)")
    parser.add_argument("--y",       type=float, required=True,  help="Y position (metres)")
    parser.add_argument("--yaw_deg", type=float, required=True,  help="Yaw (degrees)")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("gt_pose_publisher")
    pub  = node.create_publisher(PoseStamped, "/gt_pallet_pose_in", 10)

    msg                     = PoseStamped()
    msg.header.frame_id     = "world"
    msg.pose.position.x     = args.x
    msg.pose.position.y     = args.y
    msg.pose.position.z     = 0.0
    yaw                     = math.radians(args.yaw_deg)
    msg.pose.orientation.z  = math.sin(yaw / 2.0)
    msg.pose.orientation.w  = math.cos(yaw / 2.0)

    # Wait up to 2 s for the simulation node to be ready
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if pub.get_subscription_count() > 0:
            break

    msg.header.stamp = node.get_clock().now().to_msg()
    pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()
    print(f"Ground truth published: x={args.x} m  y={args.y} m  yaw={args.yaw_deg}°")


if __name__ == "__main__":
    main()
