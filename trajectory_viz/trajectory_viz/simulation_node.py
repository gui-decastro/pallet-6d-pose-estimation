import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose2D, Point
from std_msgs.msg import Empty
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
from rcl_interfaces.msg import SetParametersResult
import bisect
import math

# ======================================================================
#  Differential-drive path planner
#
#  A differential-drive vehicle can rotate in place (zero turn radius),
#  so the shortest feasible path is three segments:
#    1. Rotate in place from th0 to face the goal (x1, y1)
#    2. Drive straight from (x0, y0) to (x1, y1)
#    3. Rotate in place from the travel heading to th1
# ======================================================================

def _angle_diff(a, b):
    """Shortest signed angular difference a - b, result in (-π, π]."""
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi


def plan_diff_drive_path(x0, y0, th0, x1, y1, th1,
                         step=0.15, rot_step_deg=5.0):
    """
    3-segment differential-drive path with automatic forward/reverse selection.
    Chooses whichever direction (forward or reverse) requires less total rotation,
    producing the shortest overall path.

      rotate in place → drive forward or reverse → rotate in place
    """
    rot_step = math.radians(rot_step_deg)

    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    th_travel = math.atan2(dy, dx)
    th_reverse = math.atan2(-dy, -dx)

    # Total rotation cost for forward vs reverse
    fwd_rot = abs(_angle_diff(th_travel, th0)) + abs(_angle_diff(th1, th_travel))
    rev_rot = abs(_angle_diff(th_reverse, th0)) + abs(_angle_diff(th1, th_reverse))
    reverse = rev_rot < fwd_rot
    th_drive = th_reverse if reverse else th_travel

    points = []

    # ---- Segment 1: rotate in place at (x0, y0) ----------------------
    delta1 = _angle_diff(th_drive, th0)
    n1 = max(2, int(abs(delta1) / rot_step))
    for i in range(n1):
        points.append((x0, y0, th0 + (i / n1) * delta1))

    # ---- Segment 2: drive to (x1, y1) --------------------------------
    n2 = max(2, int(dist / step))
    for i in range(n2):
        frac = i / n2
        points.append((x0 + frac * dx, y0 + frac * dy, th_drive))

    # ---- Segment 3: rotate in place at (x1, y1) ----------------------
    delta3 = _angle_diff(th1, th_drive)
    n3 = max(2, int(abs(delta3) / rot_step))
    for i in range(n3 + 1):
        th = th_drive + (i / n3) * delta3 if n3 > 0 else th1
        points.append((x1, y1, th))

    return points


# ======================================================================
#  Arc-blend diff-drive planner
#
#  A real differential-drive vehicle can rotate and translate at the same
#  time.  This planner blends heading correction into forward motion using
#  a cubic Bezier curve whose tangents match the start and end headings:
#
#    • If the heading deviation is within a blend threshold (~50°), a
#      single Bezier arc handles the whole path smoothly.
#    • If the deviation is larger, a short in-place spin brings the
#      heading within the threshold first, then the Bezier takes over.
#      This avoids backward loops while still minimising spin time.
#
#  The final heading th1 is enforced exactly at the endpoint, so the
#  forklift always arrives aligned with the pallet axis.
# ======================================================================

def plan_arc_blend_path(x0, y0, th0, x1, y1, th1,
                        step=0.10, rot_step_deg=5.0):
    """Cubic-Bezier arc-blend path for a differential-drive vehicle."""
    rot_step = math.radians(rot_step_deg)

    dx, dy = x1 - x0, y1 - y0
    dist   = math.hypot(dx, dy)
    th_drive = math.atan2(dy, dx)

    # Fall back to classical spin-drive-spin for very short moves
    if dist < 0.25:
        return plan_diff_drive_path(x0, y0, th0, x1, y1, th1,
                                    step, rot_step_deg)

    BLEND_THRESHOLD = math.radians(50)
    d0 = _angle_diff(th_drive, th0)

    points = []
    xs, ys, ths = x0, y0, th0   # effective start after optional pre-spin

    # Optional pre-spin: rotate to within blend threshold of travel heading
    if abs(d0) > BLEND_THRESHOLD:
        target_th = th_drive - math.copysign(BLEND_THRESHOLD * 0.8, d0)
        delta = _angle_diff(target_th, th0)
        n = max(2, int(abs(delta) / rot_step))
        for i in range(n):
            points.append((x0, y0, th0 + (i / n) * delta))
        ths = target_th

    # Cubic Bezier: tangents are ths at start and th1 at end
    handle = dist * 0.28          # keep short — long handles cause S-curves on large heading changes
    p0b = (xs, ys)
    p1b = (xs + handle * math.cos(ths), ys + handle * math.sin(ths))
    p2b = (x1 - handle * math.cos(th1), y1 - handle * math.sin(th1))
    p3b = (x1, y1)

    n = max(10, int(dist / step))
    prev_x, prev_y, prev_th = xs, ys, ths
    for i in range(n + 1):
        u  = i / n
        iv = 1.0 - u
        bx = iv**3*p0b[0] + 3*iv**2*u*p1b[0] + 3*iv*u**2*p2b[0] + u**3*p3b[0]
        by = iv**3*p0b[1] + 3*iv**2*u*p1b[1] + 3*iv*u**2*p2b[1] + u**3*p3b[1]

        ddx, ddy = bx - prev_x, by - prev_y
        th = math.atan2(ddy, ddx) if (abs(ddx) > 1e-8 or abs(ddy) > 1e-8) else prev_th
        if i == n:
            th = th1          # enforce exact final heading
        points.append((bx, by, th))
        prev_x, prev_y, prev_th = bx, by, th

    return points


# ======================================================================
#  Dubins path planner (car-like, constant minimum turn radius)
#
#  Geometry conventions:
#    Right circle center: c_R = (x + r·sin(th), y − r·cos(th))
#    Left  circle center: c_L = (x − r·sin(th), y + r·cos(th))
#    Right circle position: p = c_R + r·(−sin(th),  cos(th))
#    Left  circle position: p = c_L + r·( sin(th), −cos(th))
#
#  Four path types (arc-straight-arc):
#    RSR — same-side external tangent, both arcs CW
#    LSL — same-side external tangent, both arcs CCW
#    RSL — cross tangent, first CW then CCW
#    LSR — cross tangent, first CCW then CW
# ======================================================================

def _mod2pi(x):
    return x % (2.0 * math.pi)


def _sample_right_arc(cx, cy, r, th_start, th_end, step):
    """Sample a CW arc on a right circle. Always travels CW (th decreasing)."""
    travel = _mod2pi(th_start - th_end)
    n = max(2, int(travel * r / step))
    pts = []
    for i in range(n):
        th = th_start - (i / n) * travel
        pts.append((cx - r * math.sin(th), cy + r * math.cos(th), th))
    return pts


def _sample_left_arc(cx, cy, r, th_start, th_end, step):
    """Sample a CCW arc on a left circle. Always travels CCW (th increasing)."""
    travel = _mod2pi(th_end - th_start)
    n = max(2, int(travel * r / step))
    pts = []
    for i in range(n):
        th = th_start + (i / n) * travel
        pts.append((cx + r * math.sin(th), cy - r * math.cos(th), th))
    return pts


def _sample_straight(x0, y0, x1, y1, th, step):
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(dist / step))
    return [(x0 + (i / n) * (x1 - x0), y0 + (i / n) * (y1 - y0), th)
            for i in range(n)]


def plan_dubins_path(x0, y0, th0, x1, y1, th1,
                     min_radius=3.0, step=0.15):
    """
    Plan a Dubins path for a car-like vehicle and return
    a list of (x, y, theta) world-frame waypoints.

    min_radius — minimum turning radius (meters)
    step       — sample spacing along the path (meters)
    """
    r = min_radius

    # Circle centres
    cR0 = (x0 + r * math.sin(th0), y0 - r * math.cos(th0))
    cL0 = (x0 - r * math.sin(th0), y0 + r * math.cos(th0))
    cR1 = (x1 + r * math.sin(th1), y1 - r * math.cos(th1))
    cL1 = (x1 - r * math.sin(th1), y1 + r * math.cos(th1))

    best_len = math.inf
    best = None

    # --- RSR ---
    dx, dy = cR1[0] - cR0[0], cR1[1] - cR0[1]
    D = math.hypot(dx, dy)
    if D > 1e-6:
        th_t = math.atan2(dy, dx)
        a0 = _mod2pi(th0 - th_t) * r
        a1 = _mod2pi(th_t - th1) * r
        total = a0 + D + a1
        if total < best_len:
            best_len = total
            best = ('RSR', cR0, th_t, cR1, th_t)

    # --- LSL ---
    dx, dy = cL1[0] - cL0[0], cL1[1] - cL0[1]
    D = math.hypot(dx, dy)
    if D > 1e-6:
        th_t = math.atan2(dy, dx)
        a0 = _mod2pi(th_t - th0) * r
        a1 = _mod2pi(th1 - th_t) * r
        total = a0 + D + a1
        if total < best_len:
            best_len = total
            best = ('LSL', cL0, th_t, cL1, th_t)

    # --- RSL ---
    dx, dy = cL1[0] - cR0[0], cL1[1] - cR0[1]
    D = math.hypot(dx, dy)
    if D >= 2 * r:
        th_c = math.atan2(dy, dx)
        th_t = th_c - math.asin(2 * r / D)
        straight = math.sqrt(D * D - 4 * r * r)
        a0 = _mod2pi(th0 - th_t) * r
        a1 = _mod2pi(th1 - th_t) * r
        total = a0 + straight + a1
        if total < best_len:
            best_len = total
            best = ('RSL', cR0, th_t, cL1, th_t)

    # --- LSR ---
    dx, dy = cR1[0] - cL0[0], cR1[1] - cL0[1]
    D = math.hypot(dx, dy)
    if D >= 2 * r:
        th_c = math.atan2(dy, dx)
        th_t = th_c + math.asin(2 * r / D)
        straight = math.sqrt(D * D - 4 * r * r)
        a0 = _mod2pi(th_t - th0) * r
        a1 = _mod2pi(th_t - th1) * r
        total = a0 + straight + a1
        if total < best_len:
            best_len = total
            best = ('LSR', cL0, th_t, cR1, th_t)

    if best is None:
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
        return [(x0 + i / n * (x1 - x0), y0 + i / n * (y1 - y0),
                 math.atan2(y1 - y0, x1 - x0)) for i in range(n + 1)]

    ptype, c0, th_depart, c1, th_arrive = best

    if ptype == 'RSR':
        p_dep = (c0[0] - r * math.sin(th_depart), c0[1] + r * math.cos(th_depart))
        p_arr = (c1[0] - r * math.sin(th_arrive), c1[1] + r * math.cos(th_arrive))
        return (_sample_right_arc(c0[0], c0[1], r, th0, th_depart, step) +
                _sample_straight(p_dep[0], p_dep[1], p_arr[0], p_arr[1], th_depart, step) +
                _sample_right_arc(c1[0], c1[1], r, th_arrive, th1, step))

    if ptype == 'LSL':
        p_dep = (c0[0] + r * math.sin(th_depart), c0[1] - r * math.cos(th_depart))
        p_arr = (c1[0] + r * math.sin(th_arrive), c1[1] - r * math.cos(th_arrive))
        return (_sample_left_arc(c0[0], c0[1], r, th0, th_depart, step) +
                _sample_straight(p_dep[0], p_dep[1], p_arr[0], p_arr[1], th_depart, step) +
                _sample_left_arc(c1[0], c1[1], r, th_arrive, th1, step))

    if ptype == 'RSL':
        p_dep = (c0[0] - r * math.sin(th_depart), c0[1] + r * math.cos(th_depart))
        p_arr = (c1[0] + r * math.sin(th_arrive), c1[1] - r * math.cos(th_arrive))
        return (_sample_right_arc(c0[0], c0[1], r, th0, th_depart, step) +
                _sample_straight(p_dep[0], p_dep[1], p_arr[0], p_arr[1], th_depart, step) +
                _sample_left_arc(c1[0], c1[1], r, th_arrive, th1, step))

    # LSR
    p_dep = (c0[0] + r * math.sin(th_depart), c0[1] - r * math.cos(th_depart))
    p_arr = (c1[0] - r * math.sin(th_arrive), c1[1] + r * math.cos(th_arrive))
    return (_sample_left_arc(c0[0], c0[1], r, th0, th_depart, step) +
            _sample_straight(p_dep[0], p_dep[1], p_arr[0], p_arr[1], th_depart, step) +
            _sample_right_arc(c1[0], c1[1], r, th_arrive, th1, step))


# ======================================================================
#  ROS2 simulation node
# ======================================================================

class SimulationNode(Node):

    # ---- Planner selection -------------------------------------------
    # 'arc_blend'  — cubic Bezier blending rotation into forward motion (recommended)
    # 'diff_drive' — rotate in place → straight → rotate in place (supports reverse)
    # 'dubins'     — car-like path with minimum turn radius (DUBINS_RADIUS, forward only)
    PLANNER       = 'arc_blend'
    DUBINS_RADIUS = 0.0    # meters; minimum turning radius for Dubins planner
    APPROACH_DIST  = 1.0   # meters; forklift stops here behind pallet entry before final insertion
    APPROACH_PAUSE = 1.5   # seconds; dwell at approach pose before insertion

    # ---- Animation speed limits ------------------------------------
    ANIM_LINEAR_SPEED  = 0.8   # m/s   — top speed on straight sections
    ANIM_MAX_ACCEL     = 0.25  # m/s²  — acceleration and deceleration limit
    ANIM_MAX_TURN_RATE = 0.30  # rad/s — max heading-change rate while driving
    ANIM_ANGULAR_SPEED = 0.20  # rad/s — in-place rotation speed

    # ---- Forklift geometry (meters) ----------------------------------
    BODY_LENGTH    = 2.830   # CAT DP70: Z extent in STL (forward/length)
    BODY_WIDTH     = 1.230   # X extent in STL (width)
    BODY_HEIGHT    = 1.711   # Y extent in STL (up/height)
    FORK_LENGTH    = 1.07    # 42 inches
    FORK_WIDTH     = 0.065   # must fit inside 71.8 mm pallet pocket
    FORK_THICKNESS = 0.07
    FORK_Y_OFFSET  = 0.369   # lateral distance from centre to each fork (from pallet mesh)
    # CAMERA_X_OFFSET = 0.610 # distance from forklift origin (front axle) to fork face (mast face)
    CAMERA_X_OFFSET  = 0.335  # distance from forklift origin (front axle) to fork face (mast face)
    CAMERA_Z_HEIGHT  = 1.0    # camera height above ground (meters)
    CAMERA_PITCH     = math.radians(45)   # pitch down angle (positive = looking down)

    # FRAMOS D400e (Intel RealSense D455) depth FOV
    CAMERA_FOV_H     = math.radians(87.0)   # horizontal full FOV
    CAMERA_FOV_V     = math.radians(58.0)   # vertical full FOV
    FRUSTUM_RANGE    = 3.0                  # how far to draw the frustum (meters)

    # ---- Pallet dimensions (meters) ----------------------------------
    # Frame origin: centre of the TOP surface.
    PALLET_LENGTH  = 0.762    # x — fork-entry depth  (30")
    PALLET_WIDTH   = 0.8128   # y                     (32")
    PALLET_HEIGHT  = 0.12065  # z                     (4.75")

    def __init__(self):
        super().__init__('simulation_node')

        # ---- Pallet poses (overridable via config/pallet_poses.yaml) --------
        # X/Y are in the CAMERA frame (fork face). Yaw in degrees for readability.
        # In production these are overridden by /est_pallet_pose_in and /gt_pallet_pose_in.
        self.declare_parameter('est_pallet_x',       1.95)
        self.declare_parameter('est_pallet_y',        0.1)
        self.declare_parameter('est_pallet_yaw_deg', 12.0)
        self.declare_parameter('gt_pallet_x',         2.0)
        self.declare_parameter('gt_pallet_y',         0.0)
        self.declare_parameter('gt_pallet_yaw_deg',  11.0)
        self.add_on_set_parameters_callback(self._on_pallet_pose_params)

        self.marker_pub          = self.create_publisher(MarkerArray, '/visualization_markers',   10)
        self.floor_pub           = self.create_publisher(MarkerArray, '/warehouse_floor',         10)
        self.est_pallet_pub      = self.create_publisher(MarkerArray, '/pallet_estimated',        10)
        self.gt_pallet_pub       = self.create_publisher(MarkerArray, '/pallet_ground_truth',     10)
        self.approach_marker_pub = self.create_publisher(MarkerArray, '/approach_marker',         10)
        self.tape_rays_pub       = self.create_publisher(MarkerArray, '/tape_rays',               10)
        self.frustum_pub         = self.create_publisher(MarkerArray, '/camera_frustum',          10)
        self.path_pub          = self.create_publisher(Path, '/trajectory', 10)
        self.forklift_pose_pub = self.create_publisher(PoseStamped, '/forklift_pose', 10)
        self.est_pallet_pose_pub   = self.create_publisher(PoseStamped, '/est_pallet_pose', 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.forklift_pose = {
            'x': -self.CAMERA_X_OFFSET, 'y': 0.0, 'z': 0.0,
            'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0,
        }
        self.est_pallet_pose = self._pose_from_xyyaw(
            self.get_parameter('est_pallet_x').value,
            self.get_parameter('est_pallet_y').value,
            math.radians(self.get_parameter('est_pallet_yaw_deg').value))
        self.ground_truth_pallet_pose = self._pose_from_xyyaw(
            self.get_parameter('gt_pallet_x').value,
            self.get_parameter('gt_pallet_y').value,
            math.radians(self.get_parameter('gt_pallet_yaw_deg').value))


        # Pose the forklift must reach for forks to be fully inserted
        self.pickup_pose   = self._compute_pickup_pose()
        # Approach pose: APPROACH_DIST behind pickup, already aligned with pallet
        self.approach_pose = self._compute_approach_pose()

        self._approach_waypoint_idx = 0
        self._path_waypoints  = self._build_path_waypoints()
        self._waypoint_times  = self._build_waypoint_times()

        self._anim_time  = self._waypoint_times[-1]  # stopped until triggered
        self._loop_mode  = False
        self._est_pose_received = False  # hide estimated pose + trajectory until pipeline sends one
        self._gt_pose_received  = False  # hide ground truth pallet until set_ground_truth.py sends one

        self.create_subscription(PoseStamped, '/est_pallet_pose_in',    self._on_est_pallet_pose,    10)
        self.create_subscription(Pose2D,      '/est_pallet_pose_2d',    self._on_est_pallet_pose_2d, 10)
        self.create_subscription(PoseStamped, '/gt_pallet_pose_in',  self._on_gt_pallet_pose,  10)
        self.create_subscription(Empty, '/animation/run_once',  self._on_run_once,  10)
        self.create_subscription(Empty, '/animation/loop',      self._on_loop,      10)
        self.create_subscription(Empty, '/animation/goto_start', self._on_goto_start, 10)
        self.create_subscription(Empty, '/animation/goto_end',   self._on_goto_end,   10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(
            'Simulation node started.\n'
            '  Run once:  ros2 topic pub --once /animation/run_once std_msgs/msg/Empty "{}"\n'
            '  Loop:      ros2 topic pub --once /animation/loop     std_msgs/msg/Empty "{}"'
        )

    # ------------------------------------------------------------------ #
    #  Animation trigger                                                   #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Pallet pose helpers                                                 #
    # ------------------------------------------------------------------ #

    def _on_pallet_pose_params(self, params):
        """React to runtime `ros2 param set` changes for pallet pose parameters."""
        EST_PARAMS = {'est_pallet_x', 'est_pallet_y', 'est_pallet_yaw_deg'}
        GT_PARAMS  = {'gt_pallet_x',  'gt_pallet_y',  'gt_pallet_yaw_deg'}
        changed = {p.name: p.value for p in params if p.name in EST_PARAMS | GT_PARAMS}
        if not changed:
            return SetParametersResult(successful=True)

        def _get(name):
            return changed[name] if name in changed else self.get_parameter(name).value

        if changed.keys() & EST_PARAMS:
            self.est_pallet_pose = self._pose_from_xyyaw(
                _get('est_pallet_x'),
                _get('est_pallet_y'),
                math.radians(_get('est_pallet_yaw_deg')))
            self._est_pose_received = True
            self.pickup_pose     = self._compute_pickup_pose()
            self.approach_pose   = self._compute_approach_pose()
            self._path_waypoints = self._build_path_waypoints()
            self._waypoint_times = self._build_waypoint_times()
            self._anim_time      = self._waypoint_times[-1]  # reset to stopped

        if changed.keys() & GT_PARAMS:
            self.ground_truth_pallet_pose = self._pose_from_xyyaw(
                _get('gt_pallet_x'),
                _get('gt_pallet_y'),
                math.radians(_get('gt_pallet_yaw_deg')))

        return SetParametersResult(successful=True)

    def _pose_from_xyyaw(self, x, y, yaw):
        """x, y are in the camera (fork face) frame; world origin IS the camera."""
        yaw_gt = yaw + math.pi  # +X faces forklift convention
        return {
            'x': x, 'y': y, 'z': self.PALLET_HEIGHT,
            'qx': 0.0, 'qy': 0.0,
            'qz': math.sin(yaw_gt / 2.0),
            'qw': math.cos(yaw_gt / 2.0),
        }

    def _on_est_pallet_pose(self, msg):
        q = msg.pose.orientation
        yaw = 2.0 * math.atan2(q.z, q.w)
        self.est_pallet_pose = self._pose_from_xyyaw(
            msg.pose.position.x, msg.pose.position.y, yaw)
        self._est_pose_received = True
        # Recompute trajectory to new estimated pallet pose
        self.pickup_pose     = self._compute_pickup_pose()
        self.approach_pose   = self._compute_approach_pose()
        self._path_waypoints = self._build_path_waypoints()
        self._waypoint_times = self._build_waypoint_times()
        self.get_logger().info('Estimated pallet pose received — trajectory ready.')


    def _on_est_pallet_pose_2d(self, msg):
        """Accept x, y (metres) and theta (degrees) — matches the manual constant format."""
        self.est_pallet_pose = self._pose_from_xyyaw(msg.x, msg.y, math.radians(msg.theta))
        self._est_pose_received = True
        self.pickup_pose     = self._compute_pickup_pose()
        self.approach_pose   = self._compute_approach_pose()
        self._path_waypoints = self._build_path_waypoints()
        self._waypoint_times = self._build_waypoint_times()
        self.get_logger().info('Estimated pallet pose received — trajectory ready.')

    def _on_gt_pallet_pose(self, msg):
        q = msg.pose.orientation
        yaw = 2.0 * math.atan2(q.z, q.w)
        self.ground_truth_pallet_pose = self._pose_from_xyyaw(
            msg.pose.position.x, msg.pose.position.y, yaw)
        self._gt_pose_received = True

    def _reset_forklift(self):
        self.forklift_pose['x']  = -self.CAMERA_X_OFFSET
        self.forklift_pose['y']  = 0.0
        self.forklift_pose['qz'] = 0.0
        self.forklift_pose['qw'] = 1.0
        self._anim_time = 0.0

    def _on_run_once(self, _msg):
        self._loop_mode = False
        self._reset_forklift()
        self.get_logger().info('Mode: run once.')

    def _on_loop(self, _msg):
        self._loop_mode = True
        self._reset_forklift()
        self.get_logger().info('Mode: continuous loop.')

    def _on_goto_start(self, _msg):
        self._loop_mode = False
        self._reset_forklift()
        self._anim_time = self._waypoint_times[-1]  # park at start, don't animate
        self.get_logger().info('Showing start pose.')

    def _on_goto_end(self, _msg):
        self._loop_mode = False
        x, y, th = self._path_waypoints[-1]
        self.forklift_pose['x']  = x
        self.forklift_pose['y']  = y
        self.forklift_pose['qz'] = math.sin(th / 2.0)
        self.forklift_pose['qw'] = math.cos(th / 2.0)
        self._anim_time = self._waypoint_times[-1]  # park at end, don't animate
        self.get_logger().info('Showing end pose.')

    # ------------------------------------------------------------------ #
    #  Timer                                                               #
    # ------------------------------------------------------------------ #

    def timer_callback(self):
        total = self._waypoint_times[-1]
        if self._anim_time >= total and self._loop_mode:
            self._reset_forklift()
        if self._anim_time < total:
            x, y, th = self._interpolate_pose(self._anim_time)
            self.forklift_pose['x']  = x
            self.forklift_pose['y']  = y
            self.forklift_pose['qz'] = math.sin(th / 2.0)
            self.forklift_pose['qw'] = math.cos(th / 2.0)
            self._anim_time += 0.1

        now = self.get_clock().now().to_msg()
        self._publish_transforms(now)
        self._publish_markers(now)
        self._publish_poses(now)
        if self._est_pose_received:
            self._publish_trajectory(now)
            self._publish_approach_marker(now)
        self._publish_floor(now)
        self._publish_tape_rays(now)
        self._publish_camera_frustum(now)

    # ------------------------------------------------------------------ #
    #  Velocity profile                                                    #
    # ------------------------------------------------------------------ #

    def _build_waypoint_times(self):
        """
        Trapezoidal velocity profile along the path arc length.

        For each waypoint, compute the maximum allowable speed based on the
        local curvature (v_max = ANIM_MAX_TURN_RATE / curvature).  Then run a
        forward pass (acceleration limit) and a backward pass (deceleration
        limit) to produce a physically smooth, continuous speed profile.
        Time for each segment is derived from the average speed at its two
        endpoints, so acceleration and deceleration are natural outputs of the
        physics rather than a post-process smoothing hack.
        """
        pts = self._path_waypoints
        n   = len(pts)

        # Per-segment arc length and heading change
        segs = []
        for i in range(1, n):
            x0, y0, th0 = pts[i - 1]
            x1, y1, th1 = pts[i]
            segs.append((math.hypot(x1 - x0, y1 - y0),
                         abs(_angle_diff(th1, th0))))

        # Max speed at each waypoint node from curvature constraint
        v_max = [self.ANIM_LINEAR_SPEED] * n
        v_max[0]  = 0.0   # start from rest
        v_max[-1] = 0.0   # end at rest
        idx = self._approach_waypoint_idx
        if 0 < idx < n:
            v_max[idx] = 0.0  # full stop at approach pose before insertion
        for i, (dist, dth) in enumerate(segs):
            if dist > 1e-4 and dth > 1e-4:
                v_curve = self.ANIM_MAX_TURN_RATE / (dth / dist)
                v_max[i]     = min(v_max[i],     v_curve)
                v_max[i + 1] = min(v_max[i + 1], v_curve)

        # Forward pass — acceleration limit
        v = list(v_max)
        for i in range(1, n):
            dist = segs[i - 1][0]
            if dist > 1e-4:
                v[i] = min(v[i], math.sqrt(max(0.0, v[i-1]**2 + 2*self.ANIM_MAX_ACCEL*dist)))

        # Backward pass — deceleration limit
        for i in range(n - 2, -1, -1):
            dist = segs[i][0]
            if dist > 1e-4:
                v[i] = min(v[i], math.sqrt(max(0.0, v[i+1]**2 + 2*self.ANIM_MAX_ACCEL*dist)))

        # Assign timestamps from average speed on each segment,
        # injecting APPROACH_PAUSE at the approach waypoint
        times = [0.0]
        for i, (dist, dth) in enumerate(segs):
            if dist > 1e-4:
                v_avg = (v[i] + v[i + 1]) / 2.0
                dt = dist / max(v_avg, 1e-3)
            else:
                dt = dth / self.ANIM_ANGULAR_SPEED if dth > 1e-4 else 0.0
            times.append(times[-1] + max(dt, 1e-4))
            # Insert dwell after the forklift reaches the approach pose
            if i + 1 == self._approach_waypoint_idx:
                times[-1] += self.APPROACH_PAUSE
        return times

    def _interpolate_pose(self, t):
        """Return (x, y, theta) at time t by linear interpolation between waypoints."""
        times = self._waypoint_times
        pts   = self._path_waypoints
        if t <= 0.0:
            return pts[0]
        if t >= times[-1]:
            return pts[-1]
        i = bisect.bisect_right(times, t) - 1
        i = max(0, min(i, len(pts) - 2))
        t0, t1 = times[i], times[i + 1]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        x0, y0, th0 = pts[i]
        x1, y1, th1 = pts[i + 1]
        return (
            x0 + alpha * (x1 - x0),
            y0 + alpha * (y1 - y0),
            th0 + alpha * _angle_diff(th1, th0),
        )

    # ------------------------------------------------------------------ #
    #  TF                                                                  #
    # ------------------------------------------------------------------ #

    def _publish_transforms(self, now):
        transforms = []

        frames = [('forklift', self.forklift_pose)]
        if self._gt_pose_received:
            frames.append(('pallet_gt', self.ground_truth_pallet_pose))
        if self._est_pose_received:
            frames.append(('pallet_est', self.est_pallet_pose))
        for child, pose in frames:
            tf = TransformStamped()
            tf.header.stamp    = now
            tf.header.frame_id = 'world'
            tf.child_frame_id  = child
            tf.transform.translation.x = pose['x']
            tf.transform.translation.y = pose['y']
            tf.transform.translation.z = pose['z']
            tf.transform.rotation.x = pose['qx']
            tf.transform.rotation.y = pose['qy']
            tf.transform.rotation.z = pose['qz']
            tf.transform.rotation.w = pose['qw']
            transforms.append(tf)

        # 'pickup' frame — only meaningful once estimated pose is known
        if self._est_pose_received:
            pp = self.pickup_pose
            tf = TransformStamped()
            tf.header.stamp    = now
            tf.header.frame_id = 'world'
            tf.child_frame_id  = 'pickup'
            tf.transform.translation.x = pp['x']
            tf.transform.translation.y = pp['y']
            tf.transform.translation.z = 0.0
            tf.transform.rotation.z = pp['qz']
            tf.transform.rotation.w = pp['qw']
            transforms.append(tf)

        self.tf_broadcaster.sendTransform(transforms)

    # ------------------------------------------------------------------ #
    #  Markers                                                             #
    # ------------------------------------------------------------------ #

    def _publish_markers(self, now):
        markers = MarkerArray()
        markers.markers += self._forklift_markers(now, frame='forklift', id_base=0, alpha=1.00)
        self.marker_pub.publish(markers)

        if self._est_pose_received:
            est = MarkerArray()
            est.markers += self._pallet_markers(now)
            self.est_pallet_pub.publish(est)

        if self._gt_pose_received:
            gt = MarkerArray()
            gt.markers += self._ground_truth_pallet_markers(now)
            self.gt_pallet_pub.publish(gt)

    # --- Forklift STL constants --------------------------------------- #
    # forklift.stl (CAT DP70): exported from FreeCAD in mm.
    #   STL axes: X=width(1230 mm), Y=up/height(1711 mm), Z=forward/length(2830 mm)
    #   FreeCAD origin: front drivetrain (front axle) at ground level
    #     → X=0 is lateral centre, Y=0 is ground, Z=0 is front axle
    #   ROS convention: X=forward, Z=up
    #   Rotation rpy=(π/2, 0, π/2) → q=(w=0.5, x=0.5, y=0.5, z=0.5)
    #   After rotation: STL-Z → ROS-X, STL-X → ROS-Y, STL-Y → ROS-Z
    #   No position offset needed: origin is already at ground-centre on the
    #   front axle, which is the reference point for fork-insertion geometry.
    # Uniform scale = 0.001 (mm → m)
    _STL_FK_Y_EXTENT = 1711.0   # mm; drives uniform scale (mm → m)

    # --- Forklift (solid) and pickup target (ghost) ------------------- #

    def _forklift_markers(self, now, frame, id_base, alpha):
        """
        Draws forklift mesh in `frame`.
        id_base offsets IDs so forklift and pickup markers don't collide.
        alpha < 0.5 → ghost/target style (blue); alpha ≥ 0.5 → solid (orange).
        """
        markers = []
        solid = alpha >= 0.5
        r, g, b = (1.0, 0.5, 0.0) if solid else (0.3, 0.6, 1.0)

        # Uniform scale: mm → m  (BODY_HEIGHT[m] / Y_extent[mm] = 0.001)
        s = self.BODY_HEIGHT / self._STL_FK_Y_EXTENT

        # Mesh
        m = self._base_marker(frame, id_base, now, Marker.MESH_RESOURCE)
        m.mesh_resource = 'package://trajectory_viz/meshes/forklift.stl'
        m.mesh_use_embedded_materials = False
        m.scale.x = s
        m.scale.y = s
        m.scale.z = s
        # rpy=(π/2, 0, π/2) → q=(0.5, 0.5, 0.5, 0.5)
        # Maps: STL-Z(forward) → ROS-X, STL-X(width) → ROS-Y, STL-Y(up) → ROS-Z
        m.pose.orientation.w = 0.5
        m.pose.orientation.x = 0.5
        m.pose.orientation.y = 0.5
        m.pose.orientation.z = 0.5
        # FreeCAD origin is front axle at ground → no position offset needed
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.0
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, alpha
        markers.append(m)

        return markers

    # --- Pallet ------------------------------------------------------- #

    # pallet.stl: exported from FreeCAD in meters; Z is up (matches ROS convention).
    #   Bounding box: X=[0, 0.762], Y=[0.019, 0.832], Z=[0, 0.121]
    #   Origin is at a corner on the ground — not centred.
    #   Scale = PALLET_DIM[m] / _STL_RANGE[m] ≈ 1.0 on all axes.
    #   No rotation needed (Z_stl=up already matches ROS Z=up).
    _STL_X_RANGE  = 0.762
    _STL_Y_RANGE  = 0.813
    _STL_Y_CENTER = 0.4255  # (0.019 + 0.832) / 2
    _STL_Z_RANGE  = 0.121

    def _pallet_markers(self, now):
        markers = []
        h  = self.PALLET_HEIGHT
        pl = self.PALLET_LENGTH
        pw = self.PALLET_WIDTH

        # Mesh — replaces the plain CUBE
        m = self._base_marker('pallet_est', 10, now, Marker.MESH_RESOURCE)
        m.mesh_resource = 'package://trajectory_viz/meshes/pallet.stl'
        m.mesh_use_embedded_materials = False
        m.scale.x = pl / self._STL_X_RANGE
        m.scale.y = pw / self._STL_Y_RANGE
        m.scale.z = h  / self._STL_Z_RANGE
        # 90° Z rotation: STL-X → pallet-(-Y), STL-Y → pallet-(+X)
        # Position offsets must account for the rotated axes:
        #   pallet-X centre ← STL-Y centre (_STL_Y_CENTER)
        #   pallet-Y centre ← STL-X centre (_STL_X_RANGE/2), sign flipped
        #   pallet-Z: STL Z=0 is ground, Z=range is top → shift so top is at z=0
        m.pose.position.x =  self._STL_Y_CENTER         * m.scale.y
        m.pose.position.y = -(self._STL_X_RANGE / 2.0) * m.scale.x
        m.pose.position.z = -h
        m.pose.orientation.w = 0.7071068
        m.pose.orientation.z = 0.7071068
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.7, 1.0, 0.40
        markers.append(m)

        # Label
        m = self._base_marker('pallet_est', 12, now, Marker.TEXT_VIEW_FACING)
        m.pose.position.z  = 0.45
        m.pose.orientation.w = 1.0
        m.scale.z = 0.35
        m.color.r, m.color.g, m.color.b, m.color.a = 0.4, 0.8, 1.0, 0.60
        m.text = 'estimated'
        markers.append(m)

        return markers

    def _ground_truth_pallet_markers(self, now):
        """Transparent pallet mesh showing the ground truth pose."""
        markers = []
        h  = self.PALLET_HEIGHT
        pl = self.PALLET_LENGTH
        pw = self.PALLET_WIDTH

        m = self._base_marker('pallet_gt', 10, now, Marker.MESH_RESOURCE)
        m.mesh_resource = 'package://trajectory_viz/meshes/pallet.stl'
        m.mesh_use_embedded_materials = False
        m.scale.x = pl / self._STL_X_RANGE
        m.scale.y = pw / self._STL_Y_RANGE
        m.scale.z = h  / self._STL_Z_RANGE
        m.pose.position.x =  self._STL_Y_CENTER        * m.scale.y
        m.pose.position.y = -(self._STL_X_RANGE / 2.0) * m.scale.x
        m.pose.position.z = -h
        m.pose.orientation.w = 0.7071068
        m.pose.orientation.z = 0.7071068
        m.color.r, m.color.g, m.color.b, m.color.a = 0.55, 0.27, 0.07, 1.00
        markers.append(m)

        m = self._base_marker('pallet_gt', 12, now, Marker.TEXT_VIEW_FACING)
        m.pose.position.z  = 0.45
        m.pose.orientation.w = 1.0
        m.scale.z = 0.35
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 1.0
        m.text = 'ground truth'
        markers.append(m)

        return markers

    def _publish_approach_marker(self, now):
        msg = MarkerArray()
        msg.markers += self._approach_markers(now)
        self.approach_marker_pub.publish(msg)

    def _approach_markers(self, now):
        """Arrow + label showing the approach pose in the world frame."""
        ap = self.approach_pose
        markers = []

        # Arrow pointing in the forklift heading direction at approach
        m = Marker()
        m.header.stamp    = now
        m.header.frame_id = 'world'
        m.ns              = 'approach'
        m.id              = 0
        m.type            = Marker.ARROW
        m.action          = Marker.ADD
        m.pose.position.x = ap['x']
        m.pose.position.y = ap['y']
        m.pose.position.z = 0.1
        m.pose.orientation.z = ap['qz']
        m.pose.orientation.w = ap['qw']
        m.scale.x = 0.8    # shaft length
        m.scale.y = 0.06   # shaft diameter
        m.scale.z = 0.10   # head diameter
        m.color.r, m.color.g, m.color.b, m.color.a = 0.8, 0.8, 0.8, 0.5
        markers.append(m)

        # Label
        m = Marker()
        m.header.stamp    = now
        m.header.frame_id = 'world'
        m.ns              = 'approach'
        m.id              = 1
        m.type            = Marker.TEXT_VIEW_FACING
        m.action          = Marker.ADD
        m.pose.position.x = ap['x']
        m.pose.position.y = ap['y']
        m.pose.position.z = 0.6
        m.pose.orientation.w = 1.0
        m.scale.z = 0.25
        m.color.r, m.color.g, m.color.b, m.color.a = 0.8, 0.8, 0.8, 0.5
        m.text = 'approach'
        markers.append(m)

        return markers

    # ------------------------------------------------------------------ #
    #  Warehouse floor                                                     #
    # ------------------------------------------------------------------ #

    def _publish_floor(self, now):
        msg = MarkerArray()
        msg.markers += self._floor_markers(now)
        self.floor_pub.publish(msg)

    def _floor_markers(self, now):
        markers = []

        def _wm(mid, mtype):
            m = Marker()
            m.header.stamp    = now
            m.header.frame_id = 'world'
            m.ns              = 'floor'
            m.id              = mid
            m.type            = mtype
            m.action          = Marker.ADD
            m.pose.orientation.w = 1.0
            return m

        # Concrete slab — 30 x 30 m x 0.3 m thick, top surface flush with z=0
        m = _wm(0, Marker.CUBE)
        m.pose.position.x =  5.0
        m.pose.position.y =  0.0
        m.pose.position.z = -0.15
        m.scale.x = 30.0
        m.scale.y = 30.0
        m.scale.z =  0.30
        m.color.r, m.color.g, m.color.b, m.color.a = 0.42, 0.42, 0.40, 1.0
        markers.append(m)

        return markers

    # ------------------------------------------------------------------ #
    #  30-degree reference tape rays (physical tape on warehouse floor)   #
    # ------------------------------------------------------------------ #

    def _publish_tape_rays(self, now):
        """Three tape strips on the floor: centre ray and ±30° rays.
        These represent physical tape placed in the real environment to
        mark ground-truth image capture positions — not the camera FOV."""
        length     = 3.0
        tape_width = 0.05
        tape_thick = 0.002   # essentially flush with z=0

        markers = []
        for mid, deg in enumerate((0, -30, 30)):
            rad = math.radians(deg)
            m = Marker()
            m.header.stamp    = now
            m.header.frame_id = 'world'
            m.ns              = 'tape_rays'
            m.id              = mid
            m.type            = Marker.CUBE
            m.action          = Marker.ADD
            # Centre the box halfway along the ray
            m.pose.position.x = (length / 2.0) * math.cos(rad)
            m.pose.position.y = (length / 2.0) * math.sin(rad)
            m.pose.position.z = tape_thick / 2.0
            # Rotate box so its long axis (X) points along the ray
            m.pose.orientation.z = math.sin(rad / 2.0)
            m.pose.orientation.w = math.cos(rad / 2.0)
            m.scale.x = length
            m.scale.y = tape_width
            m.scale.z = tape_thick
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.35, 0.9, 1.0
            markers.append(m)

        self.tape_rays_pub.publish(MarkerArray(markers=markers))

    # ------------------------------------------------------------------ #
    #  Camera frustum (FRAMOS D400e / RealSense D455, 87° H × 58° V)     #
    # ------------------------------------------------------------------ #

    def _publish_camera_frustum(self, now):
        """Draws the D455 depth camera frustum in the forklift TF frame.

        The camera is at (CAMERA_X_OFFSET, 0, CAMERA_Z_HEIGHT) in the
        forklift frame, pitched down by CAMERA_PITCH (positive = nose down).
        Rotation R_y(+pitch): x'=x·cosθ+z·sinθ, z'=-x·sinθ+z·cosθ.
        For positive θ the +X forward vector gains -Z (downward). ✓

        Each corner ray is clipped at z=0 so nothing is drawn below the floor.
        The fill quad is the camera's footprint on the ground plane.

        Two markers are published:
          id=0  LINE_LIST     — wireframe edges of the frustum pyramid
          id=1  TRIANGLE_LIST — semi-opaque ground footprint (what the camera sees)
        """
        half_h = self.CAMERA_FOV_H / 2.0
        half_v = self.CAMERA_FOV_V / 2.0
        r      = self.FRUSTUM_RANGE
        pitch  = self.CAMERA_PITCH

        # Camera apex in forklift frame
        ax, ay, az = self.CAMERA_X_OFFSET, 0.0, self.CAMERA_Z_HEIGHT

        # Half-extents of the far rectangle in camera space (camera looks +X)
        fy = r * math.tan(half_h)
        fz = r * math.tan(half_v)

        # Helper: apply R_y(+pitch) then translate to forklift frame
        cp, sp = math.cos(pitch), math.sin(pitch)
        def _fk(x, y, z):
            return Point(
                x=ax + x * cp + z * sp,
                y=ay + y,
                z=az - x * sp + z * cp,
            )

        apex = _fk(0.0, 0.0, 0.0)
        corners = [
            _fk(r,  fy,  fz),   # top-left
            _fk(r, -fy,  fz),   # top-right
            _fk(r, -fy, -fz),   # bottom-right
            _fk(r,  fy, -fz),   # bottom-left
        ]

        # Clip a ray from apex to pt at z=0 (floor) for the wireframe.
        # If pt is already above the floor it is returned unchanged;
        # otherwise the floor intersection is returned so nothing is drawn below ground.
        def _clip(pt):
            if pt.z >= 0.0:
                return pt
            dz = pt.z - apex.z   # always negative when pt is below floor
            t  = -apex.z / dz
            return Point(x=apex.x + t * (pt.x - apex.x),
                         y=apex.y + t * (pt.y - apex.y),
                         z=0.0)

        # Project a corner ray onto the floor (z=0) for the fill polygon.
        # Extends the ray past the far plane if needed — the fill always lies flat on the ground.
        # Returns None if the ray goes upward and never reaches the floor.
        def _project_to_ground(pt):
            dz = pt.z - apex.z
            if dz >= 0.0:
                return None  # ray goes up — no ground intersection
            t = -apex.z / dz
            return Point(x=apex.x + t * (pt.x - apex.x),
                         y=apex.y + t * (pt.y - apex.y),
                         z=0.0)

        clipped = [_clip(c) for c in corners]
        ground  = [_project_to_ground(c) for c in corners]

        # --- Wireframe: 4 apex→clipped-corner rays + 4 far-rectangle edges ---
        wire_pts = []
        for c in clipped:
            wire_pts += [apex, c]
        for i in range(4):
            wire_pts += [clipped[i], clipped[(i + 1) % 4]]

        m_wire = Marker()
        m_wire.header.stamp    = now
        m_wire.header.frame_id = 'forklift'
        m_wire.ns              = 'camera_frustum'
        m_wire.id              = 0
        m_wire.type            = Marker.LINE_LIST
        m_wire.action          = Marker.ADD
        m_wire.pose.orientation.w = 1.0
        m_wire.scale.x         = 0.02
        m_wire.color.r, m_wire.color.g, m_wire.color.b, m_wire.color.a = 0.1, 0.9, 0.55, 1.0
        m_wire.points          = wire_pts

        # --- Ground footprint fill: project frustum rays onto z=0 ---
        m_fill = Marker()
        m_fill.header.stamp    = now
        m_fill.header.frame_id = 'forklift'
        m_fill.ns              = 'camera_frustum'
        m_fill.id              = 1
        m_fill.type            = Marker.TRIANGLE_LIST
        m_fill.action          = Marker.ADD
        m_fill.pose.orientation.w = 1.0
        m_fill.scale.x = m_fill.scale.y = m_fill.scale.z = 1.0
        m_fill.color.r, m_fill.color.g, m_fill.color.b, m_fill.color.a = 0.1, 0.9, 0.55, 0.35
        if all(g is not None for g in ground):
            m_fill.points = [
                ground[0], ground[1], ground[2],
                ground[0], ground[2], ground[3],
                ground[2], ground[1], ground[0],
                ground[3], ground[2], ground[0],
            ]

        self.frustum_pub.publish(MarkerArray(markers=[m_wire, m_fill]))

    def _base_marker(self, frame, marker_id, now, mtype):
        m = Marker()
        m.header.stamp    = now
        m.header.frame_id = frame
        m.ns              = frame
        m.id              = marker_id
        m.type            = mtype
        m.action          = Marker.ADD
        return m

    # ------------------------------------------------------------------ #
    #  Poses                                                               #
    # ------------------------------------------------------------------ #

    def _publish_poses(self, now):
        for pub, pose in [
            (self.forklift_pose_pub, self.forklift_pose),
            (self.est_pallet_pose_pub,   self.est_pallet_pose),
        ]:
            msg = PoseStamped()
            msg.header.stamp    = now
            msg.header.frame_id = 'world'
            msg.pose.position.x = pose['x']
            msg.pose.position.y = pose['y']
            msg.pose.position.z = pose['z']
            msg.pose.orientation.x = pose['qx']
            msg.pose.orientation.y = pose['qy']
            msg.pose.orientation.z = pose['qz']
            msg.pose.orientation.w = pose['qw']
            pub.publish(msg)

    # ------------------------------------------------------------------ #
    #  Trajectory                                                          #
    # ------------------------------------------------------------------ #

    def _publish_trajectory(self, now):
        path = Path()
        path.header.stamp    = now
        path.header.frame_id = 'world'

        for x, y, th in self._path_waypoints:
            pose = PoseStamped()
            pose.header.stamp    = now
            pose.header.frame_id = 'world'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(th / 2.0)
            pose.pose.orientation.w = math.cos(th / 2.0)
            path.poses.append(pose)

        self.path_pub.publish(path)

    # ------------------------------------------------------------------ #
    #  Pickup pose                                                         #
    # ------------------------------------------------------------------ #

    def _compute_pickup_pose(self):
        """
        Compute the forklift world-frame pose at which the forks are
        fully inserted into the pallet.

        Geometry:
          - Forklift frame origin is the front axle at ground level.
          - Forks mount at the front axle and extend FORK_LENGTH forward.
          - Fork tips reach the far edge of the pallet (+PALLET_LENGTH/2
            in pallet frame) → full insertion.
          - fork_tip_reach = FORK_LENGTH from forklift origin (front axle).
          - Forklift origin is therefore (FORK_LENGTH - PALLET_LENGTH/2)
            meters behind the pallet centre, along pallet -X.
        """
        pp = self.est_pallet_pose
        pallet_yaw = 2.0 * math.atan2(pp['qz'], pp['qw'])

        fork_tip_reach = self.FORK_LENGTH
        offset         = fork_tip_reach - self.PALLET_LENGTH / 2.0 + 0.10  # 150 mm short of full insertion

        # Pallet +X faces forklift: forklift is on the +X side, heading in pallet -X direction
        th = pallet_yaw + math.pi
        x = pp['x'] + offset * math.cos(pallet_yaw)
        y = pp['y'] + offset * math.sin(pallet_yaw)

        return {
            'x':  x,
            'y':  y,
            'th': th,
            'qz': math.sin(th / 2.0),
            'qw': math.cos(th / 2.0),
        }

    def _compute_approach_pose(self):
        """
        Compute the approach pose: APPROACH_DIST meters behind the pickup pose
        along the pallet entry axis. The forklift is already aligned with the
        pallet yaw here, so the final segment to pickup is a straight drive.
        """
        pp = self.pickup_pose
        return {
            'x':  pp['x'] - self.APPROACH_DIST * math.cos(pp['th']),
            'y':  pp['y'] - self.APPROACH_DIST * math.sin(pp['th']),
            'th': pp['th'],
            'qz': pp['qz'],
            'qw': pp['qw'],
        }

    # ------------------------------------------------------------------ #
    #  Path planning                                                       #
    # ------------------------------------------------------------------ #

    def _build_path_waypoints(self):
        """
        Two-segment path: start → approach_pose → pickup_pose.

        Segment 1 (start → approach): uses selected planner.
        Segment 2 (approach → pickup): always a straight forward drive —
          the forklift is already aligned with the pallet axis at approach_pose.
        """
        fp  = self.forklift_pose
        th0 = 2.0 * math.atan2(fp['qz'], fp['qw'])

        ap  = self.approach_pose
        pp  = self.pickup_pose

        if self.PLANNER == 'arc_blend':
            seg1 = plan_arc_blend_path(
                fp['x'], fp['y'], th0,
                ap['x'], ap['y'], ap['th'],
            )
            label = 'Arc-blend'
        elif self.PLANNER == 'dubins':
            seg1 = plan_dubins_path(
                fp['x'], fp['y'], th0,
                ap['x'], ap['y'], ap['th'],
                min_radius=self.DUBINS_RADIUS,
            )
            label = f'Dubins (r={self.DUBINS_RADIUS} m)'
        else:
            seg1 = plan_diff_drive_path(
                fp['x'], fp['y'], th0,
                ap['x'], ap['y'], ap['th'],
            )
            label = 'Diff-drive'

        # Straight insertion run: approach → pickup (already aligned)
        n = max(2, int(self.APPROACH_DIST / 0.15))
        seg2 = [
            (ap['x'] + (i / n) * (pp['x'] - ap['x']),
             ap['y'] + (i / n) * (pp['y'] - ap['y']),
             ap['th'])
            for i in range(n + 1)
        ]

        waypoints = seg1 + seg2
        self._approach_waypoint_idx = len(seg1)
        self.get_logger().info(
            f'{label} path: {len(seg1)} + {len(seg2)} waypoints '
            f'(approach + insertion)'
        )
        return waypoints


def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
