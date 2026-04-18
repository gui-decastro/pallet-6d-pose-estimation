"""
viz_standalone.py
=================
Standalone top-down matplotlib visualization for the pallet 6-DOF pose
estimation pipeline.  No ROS2 required.

Shows (top-down world view, origin = camera / fork face):
  - Camera FOV lines        (±30°, 4 m)
  - Estimated pallet pose   (brown rectangle + fork-entry arrow)
  - Forklift start position (orange rectangle)
  - Planned approach + insertion trajectory  (diff-drive planner)
  - Approach waypoint       (grey arrow + label)
  - Animated forklift travelling the full trajectory

Usage (called automatically by publish_to_viz.py, or run directly):
    python viz_standalone.py --x 2.1 --y -0.05 --yaw_deg 14.3

Geometry conventions match simulation_node.py:
    X = forward from fork face / camera
    Y = lateral (left positive)
    Yaw = rotation about Z (radians, CCW positive)
"""

import argparse
import math
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation


# ── Geometry constants (mirror simulation_node.py) ───────────────────────────
CAMERA_X_OFFSET = 0.335   # forklift world-origin is this far behind fork face
FORK_LENGTH      = 1.07
PALLET_LENGTH    = 0.762
PALLET_WIDTH     = 0.8128
BODY_LENGTH      = 2.830
BODY_WIDTH       = 1.230
APPROACH_DIST    = 1.0

# ── Animation settings ────────────────────────────────────────────────────────
ANIM_FPS         = 60      # target frame rate
STEP_PER_FRAME   = 3       # waypoints advanced per animation frame (speed-up)


# ─────────────────────────────────────────────────────────────────────────────
#  Diff-drive planner  (mirrors simulation_node.plan_diff_drive_path)
# ─────────────────────────────────────────────────────────────────────────────

def _angle_diff(a: float, b: float) -> float:
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi


def _plan_diff_drive(x0, y0, th0, x1, y1, th1,
                     step=0.05, rot_step_deg=2.0):
    rot_step = math.radians(rot_step_deg)
    dx, dy   = x1 - x0, y1 - y0
    dist     = math.hypot(dx, dy)
    th_fwd   = math.atan2(dy, dx)
    th_rev   = math.atan2(-dy, -dx)
    fwd_cost = abs(_angle_diff(th_fwd, th0)) + abs(_angle_diff(th1, th_fwd))
    rev_cost = abs(_angle_diff(th_rev, th0)) + abs(_angle_diff(th1, th_rev))
    th_drive = th_rev if rev_cost < fwd_cost else th_fwd

    pts = []

    delta1 = _angle_diff(th_drive, th0)
    n1 = max(2, int(abs(delta1) / rot_step))
    for i in range(n1):
        pts.append((x0, y0, th0 + (i / n1) * delta1))

    n2 = max(2, int(dist / step))
    for i in range(n2):
        f = i / n2
        pts.append((x0 + f * dx, y0 + f * dy, th_drive))

    delta3 = _angle_diff(th1, th_drive)
    n3 = max(2, int(abs(delta3) / rot_step))
    for i in range(n3 + 1):
        th = th_drive + (i / n3) * delta3 if n3 > 0 else th1
        pts.append((x1, y1, th))

    return pts


# ─────────────────────────────────────────────────────────────────────────────
#  Pose helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pickup_pose(px, py, pallet_yaw):
    """Forklift pose (x, y, th) when forks are fully inserted."""
    offset = FORK_LENGTH - PALLET_LENGTH / 2.0 + 0.10
    th = pallet_yaw + math.pi
    return (px + offset * math.cos(pallet_yaw),
            py + offset * math.sin(pallet_yaw),
            th)


def _approach_pose(pk_x, pk_y, pk_th):
    """APPROACH_DIST behind the pickup pose, already aligned."""
    return (pk_x - APPROACH_DIST * math.cos(pk_th),
            pk_y - APPROACH_DIST * math.sin(pk_th),
            pk_th)


def _build_waypoints(pallet_x, pallet_y, pallet_yaw):
    pk = _pickup_pose(pallet_x, pallet_y, pallet_yaw)
    ap = _approach_pose(*pk)

    seg1 = _plan_diff_drive(-CAMERA_X_OFFSET, 0.0, 0.0,
                             ap[0], ap[1], ap[2])
    n    = max(2, int(APPROACH_DIST / 0.05))
    seg2 = [(ap[0] + (i / n) * (pk[0] - ap[0]),
             ap[1] + (i / n) * (pk[1] - ap[1]),
             ap[2])
            for i in range(n + 1)]

    return seg1 + seg2, ap, pk


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rect_corners(cx, cy, th, length, width):
    """4 world-frame corners of a rectangle centred at (cx, cy), heading th."""
    local = np.array([[ length / 2,  width / 2],
                      [ length / 2, -width / 2],
                      [-length / 2, -width / 2],
                      [-length / 2,  width / 2]])
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th),  math.cos(th)]])
    return (R @ local.T).T + np.array([cx, cy])


def _draw_arrow(ax, x, y, th, length=0.5, color="white", lw=1.5, alpha=1.0):
    ax.annotate(
        "",
        xy=(x + length * math.cos(th), y + length * math.sin(th)),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="->", color=color,
                        lw=lw, mutation_scale=12),
        alpha=alpha,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main visualization
# ─────────────────────────────────────────────────────────────────────────────

def run_viz(pallet_x: float, pallet_y: float, pallet_yaw_rad: float) -> None:
    waypoints, approach, pickup = _build_waypoints(pallet_x, pallet_y, pallet_yaw_rad)
    traj_x = [p[0] for p in waypoints]
    traj_y = [p[1] for p in waypoints]

    # ── Figure / axes ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#12122a")
    ax.set_facecolor("#1a1a35")
    ax.set_aspect("equal")
    ax.set_xlabel("X  (m — forward from camera)", color="#aaaacc", fontsize=9)
    ax.set_ylabel("Y  (m — lateral)", color="#aaaacc", fontsize=9)
    ax.set_title(
        f"Pallet 6-DOF Pose Estimation\n"
        f"x={pallet_x:.3f} m   y={pallet_y:.3f} m   "
        f"yaw={math.degrees(pallet_yaw_rad):.1f}°",
        color="white", fontsize=10,
    )
    ax.tick_params(colors="#888899")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

    # ── Floor grid ────────────────────────────────────────────────────────
    all_x = traj_x + [pallet_x, -CAMERA_X_OFFSET - BODY_LENGTH]
    all_y = traj_y + [pallet_y,  BODY_WIDTH / 2, -BODY_WIDTH / 2]
    pad   = 1.5
    xl = (min(all_x) - pad, max(all_x) + pad)
    yl = (min(all_y) - pad, max(all_y) + pad)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)

    for gx in np.arange(math.floor(xl[0]), math.ceil(xl[1]) + 1):
        ax.axvline(gx, color="#22224a", lw=0.5, zorder=0)
    for gy in np.arange(math.floor(yl[0]), math.ceil(yl[1]) + 1):
        ax.axhline(gy, color="#22224a", lw=0.5, zorder=0)

    # ── Camera origin ─────────────────────────────────────────────────────
    ax.plot(0, 0, "o", color="#4488ff", ms=7, zorder=5, label="Camera / fork face")

    # ── FOV lines ─────────────────────────────────────────────────────────
    fov_len = min(4.0, xl[1] - 0.2)
    for deg in (0, -30, 30):
        rad = math.radians(deg)
        ax.plot([0, fov_len * math.cos(rad)],
                [0, fov_len * math.sin(rad)],
                color="#2255bb", lw=1.0, ls="--", alpha=0.6, zorder=1)

    # ── Planned trajectory (static dotted line) ───────────────────────────
    ax.plot(traj_x, traj_y, color="#445566", lw=1.0, ls=":",
            zorder=2, label="Planned path")

    # ── Approach waypoint ─────────────────────────────────────────────────
    _draw_arrow(ax, approach[0], approach[1], approach[2],
                length=0.5, color="#888899", lw=1.2, alpha=0.7)
    ax.text(approach[0], approach[1] - 0.25, "approach",
            color="#888899", fontsize=7, ha="center", va="top")

    # ── Pallet rectangle ──────────────────────────────────────────────────
    pc = _rect_corners(pallet_x, pallet_y, pallet_yaw_rad,
                       PALLET_LENGTH, PALLET_WIDTH)
    pallet_poly = plt.Polygon(pc, closed=True,
                              facecolor="#6b3a0f", edgecolor="#ffaa33",
                              lw=1.8, alpha=0.9, zorder=3,
                              label="Pallet (estimated)")
    ax.add_patch(pallet_poly)
    # Fork-entry direction arrow
    _draw_arrow(ax, pallet_x, pallet_y, pallet_yaw_rad + math.pi,
                length=0.55, color="#ffaa33", lw=2.0, alpha=0.9)

    # ── Ghost forklift at pickup pose ─────────────────────────────────────
    ghost_c = _rect_corners(pickup[0], pickup[1], pickup[2],
                            BODY_LENGTH, BODY_WIDTH)
    ghost_poly = plt.Polygon(ghost_c, closed=True,
                             facecolor="#225588", edgecolor="#4499cc",
                             lw=1.2, alpha=0.35, zorder=3,
                             label="Pickup pose (target)")
    ax.add_patch(ghost_poly)

    # ── Animated forklift body ────────────────────────────────────────────
    init_c = _rect_corners(-CAMERA_X_OFFSET, 0.0, 0.0, BODY_LENGTH, BODY_WIDTH)
    fork_poly = plt.Polygon(init_c, closed=True,
                            facecolor="#cc6600", edgecolor="#ffcc55",
                            lw=1.5, alpha=0.92, zorder=4,
                            label="Forklift")
    ax.add_patch(fork_poly)

    # Heading arrow (updated each frame)
    heading_ann = ax.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#ffee88",
                        lw=1.8, mutation_scale=14),
        zorder=5,
    )

    # Frame counter text
    frame_text = ax.text(
        xl[0] + 0.1, yl[1] - 0.1, "",
        color="#667788", fontsize=7, va="top", zorder=6,
    )

    ax.legend(
        facecolor="#12122a", edgecolor="#333355",
        labelcolor="white", loc="upper left", fontsize=8,
    )

    # ── Animation ─────────────────────────────────────────────────────────
    total_frames = math.ceil(len(waypoints) / STEP_PER_FRAME) + 30  # +30 pause at end

    def update(frame):
        idx = min(frame * STEP_PER_FRAME, len(waypoints) - 1)
        wx, wy, wth = waypoints[idx]

        corners = _rect_corners(wx, wy, wth, BODY_LENGTH, BODY_WIDTH)
        fork_poly.set_xy(corners)

        arrow_len = 0.7
        heading_ann.set_position((wx, wy))
        heading_ann.xy = (wx + arrow_len * math.cos(wth),
                          wy + arrow_len * math.sin(wth))

        pct = int(100 * idx / (len(waypoints) - 1))
        frame_text.set_text(f"progress: {pct}%")

        return fork_poly, heading_ann, frame_text

    anim = FuncAnimation(      # noqa: F841  (keep reference alive)
        fig, update,
        frames=total_frames,
        interval=int(1000 / ANIM_FPS),
        blit=True,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standalone pallet pose + forklift trajectory visualization"
    )
    parser.add_argument("--x",       type=float, required=True,
                        help="Pallet X in world/camera frame (metres, forward)")
    parser.add_argument("--y",       type=float, required=True,
                        help="Pallet Y in world/camera frame (metres, lateral)")
    parser.add_argument("--yaw_deg", type=float, required=True,
                        help="Pallet yaw (degrees, rotation about Z axis)")
    args = parser.parse_args()

    run_viz(args.x, args.y, math.radians(args.yaw_deg))


if __name__ == "__main__":
    main()
