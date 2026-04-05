import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Empty
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
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

    min_radius — minimum turning radius (metres)
    step       — sample spacing along the path (metres)
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
    # 'diff_drive' — rotate in place → straight → rotate in place (supports reverse)
    # 'dubins'     — car-like path with minimum turn radius (DUBINS_RADIUS, forward only)
    PLANNER       = 'diff_drive'
    DUBINS_RADIUS = 1.0    # metres; minimum turning radius for Dubins planner
    APPROACH_DIST = 1.5    # metres; forklift stops here behind pallet entry before final insertion

    # ---- Forklift geometry (metres) ----------------------------------
    BODY_LENGTH    = 2.830   # CAT DP70: Z extent in STL (forward/length)
    BODY_WIDTH     = 1.230   # X extent in STL (width)
    BODY_HEIGHT    = 1.711   # Y extent in STL (up/height)
    FORK_LENGTH    = 1.07    # 42 inches
    FORK_WIDTH     = 0.065   # must fit inside 71.8 mm pallet pocket
    FORK_THICKNESS = 0.07
    FORK_Y_OFFSET  = 0.369   # lateral distance from centre to each fork (from pallet mesh)
    FORK_FACE_OFFSET = 0.610 # distance from forklift origin (front axle) to fork face (mast face)

    # ---- Pallet dimensions (metres) ----------------------------------
    # Frame origin: centre of the TOP surface.
    PALLET_LENGTH  = 0.762    # x — fork-entry depth  (30")
    PALLET_WIDTH   = 0.8128   # y                     (32")
    PALLET_HEIGHT  = 0.12065  # z                     (4.75")


    def __init__(self):
        super().__init__('simulation_node')

        self.marker_pub        = self.create_publisher(MarkerArray, '/visualization_markers', 10)
        self.path_pub          = self.create_publisher(Path, '/trajectory', 10)
        self.forklift_pose_pub = self.create_publisher(PoseStamped, '/forklift_pose', 10)
        self.pallet_pose_pub   = self.create_publisher(PoseStamped, '/pallet_pose', 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        # --- Poses (replace with real estimates in production) -----------
        self.forklift_pose = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0,
        }
        yaw = math.radians(15.0)
        _cam_dist = 2.0   # metres from camera (fork face) to pallet
        self.pallet_pose = {
            'x': self.FORK_FACE_OFFSET + _cam_dist, 'y': 0.0, 'z': self.PALLET_HEIGHT,
            'qx': 0.0, 'qy': 0.0,
            'qz': math.sin(yaw / 2.0),
            'qw': math.cos(yaw / 2.0),
        }

        # Pose the forklift must reach for forks to be fully inserted
        self.pickup_pose   = self._compute_pickup_pose()
        # Approach pose: APPROACH_DIST behind pickup, already aligned with pallet
        self.approach_pose = self._compute_approach_pose()

        self._path_waypoints = self._build_path_waypoints()
        self._anim_idx = len(self._path_waypoints)  # stopped until triggered
        self._loop_mode = False

        self.create_subscription(Empty, '/animation/run_once', self._on_run_once, 10)
        self.create_subscription(Empty, '/animation/loop',     self._on_loop,     10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(
            'Simulation node started.\n'
            '  Run once:  ros2 topic pub --once /animation/run_once std_msgs/msg/Empty "{}"\n'
            '  Loop:      ros2 topic pub --once /animation/loop     std_msgs/msg/Empty "{}"'
        )

    # ------------------------------------------------------------------ #
    #  Animation trigger                                                   #
    # ------------------------------------------------------------------ #

    def _reset_forklift(self):
        self.forklift_pose['x']  = 0.0
        self.forklift_pose['y']  = 0.0
        self.forklift_pose['qz'] = 0.0
        self.forklift_pose['qw'] = 1.0
        self._anim_idx = 0

    def _on_run_once(self, _msg):
        self._loop_mode = False
        self._reset_forklift()
        self.get_logger().info('Mode: run once.')

    def _on_loop(self, _msg):
        self._loop_mode = True
        self._reset_forklift()
        self.get_logger().info('Mode: continuous loop.')

    # ------------------------------------------------------------------ #
    #  Timer                                                               #
    # ------------------------------------------------------------------ #

    def timer_callback(self):
        if self._anim_idx >= len(self._path_waypoints) and self._loop_mode:
            self._reset_forklift()
        if self._anim_idx < len(self._path_waypoints):
            x, y, th = self._path_waypoints[self._anim_idx]
            self.forklift_pose['x']  = x
            self.forklift_pose['y']  = y
            self.forklift_pose['qz'] = math.sin(th / 2.0)
            self.forklift_pose['qw'] = math.cos(th / 2.0)
            self._anim_idx += 1

        now = self.get_clock().now().to_msg()
        self._publish_transforms(now)
        self._publish_markers(now)
        self._publish_poses(now)
        self._publish_trajectory(now)

    # ------------------------------------------------------------------ #
    #  TF                                                                  #
    # ------------------------------------------------------------------ #

    def _publish_transforms(self, now):
        transforms = []

        for child, pose in [('forklift', self.forklift_pose),
                             ('pallet',   self.pallet_pose)]:
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

        # 'pickup' frame — where the forklift must be to lift the pallet
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
        markers.markers += self._forklift_markers(now, frame='forklift', id_base=0,  alpha=1.00)
        markers.markers += self._forklift_markers(now, frame='pickup',   id_base=20, alpha=0.30)
        markers.markers += self._pallet_markers(now)
        self.marker_pub.publish(markers)

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

        # Label
        m = self._base_marker(frame, id_base + 1, now, Marker.TEXT_VIEW_FACING)
        m.pose.position.z = self.BODY_HEIGHT + 0.4
        m.pose.orientation.w = 1.0
        m.scale.z = 0.4
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, alpha
        m.text = 'FORKLIFT' if solid else 'PICKUP TARGET'
        markers.append(m)

        return markers

    # --- Pallet ------------------------------------------------------- #

    # pallet.stl: exported from FreeCAD in metres; Z is up (matches ROS convention).
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
        m = self._base_marker('pallet', 10, now, Marker.MESH_RESOURCE)
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
        m.color.r, m.color.g, m.color.b, m.color.a = 0.55, 0.27, 0.07, 1.00
        markers.append(m)

        # Fork-entry arrow (just above top surface, spanning full depth)
        m = self._base_marker('pallet', 11, now, Marker.ARROW)
        m.pose.position.x  = -self.PALLET_LENGTH / 2.0
        m.pose.position.z  = 0.03
        m.pose.orientation.w = 1.0
        m.scale.x = self.PALLET_LENGTH   # shaft length
        m.scale.y = 0.08                  # shaft diameter
        m.scale.z = 0.12                  # head diameter
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.1, 0.9
        markers.append(m)

        # Label
        m = self._base_marker('pallet', 12, now, Marker.TEXT_VIEW_FACING)
        m.pose.position.z  = 0.45
        m.pose.orientation.w = 1.0
        m.scale.z = 0.4
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, 1.0
        m.text = 'PALLET (estimated)'
        markers.append(m)

        return markers

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
            (self.pallet_pose_pub,   self.pallet_pose),
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
            metres behind the pallet centre, along pallet -X.
        """
        pp = self.pallet_pose
        pallet_yaw = 2.0 * math.atan2(pp['qz'], pp['qw'])

        fork_tip_reach = self.FORK_LENGTH
        offset         = fork_tip_reach - self.PALLET_LENGTH / 2.0 + 0.10  # 150 mm short of full insertion

        x = pp['x'] - offset * math.cos(pallet_yaw)
        y = pp['y'] - offset * math.sin(pallet_yaw)

        return {
            'x':  x,
            'y':  y,
            'th': pallet_yaw,
            'qz': math.sin(pallet_yaw / 2.0),
            'qw': math.cos(pallet_yaw / 2.0),
        }

    def _compute_approach_pose(self):
        """
        Compute the approach pose: APPROACH_DIST metres behind the pickup pose
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

        if self.PLANNER == 'dubins':
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
