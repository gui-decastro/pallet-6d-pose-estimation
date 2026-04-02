import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster
import math

# ======================================================================
#  Dubins path solver
#
#  Finds the shortest path between two planar poses (x, y, heading)
#  for a vehicle that moves forward only with a minimum turning radius.
#  A Dubins path is at most 3 segments: Left-arc, Right-arc, Straight.
#  The 6 candidate types are LSL, RSR, LSR, RSL, RLR, LRL.
#
#  Reference: Dubins (1957); AndrewWalker/Dubins-Curves (C reference).
# ======================================================================

def _mod2pi(x):
    """Wrap x into [0, 2π)."""
    return x - 2.0 * math.pi * math.floor(x / (2.0 * math.pi))


def _dubins_LSL(alpha, beta, d):
    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta),  math.cos(beta)
    tmp0 = d + sa - sb
    p_sq = 2.0 + d**2 - 2.0 * math.cos(alpha - beta) + 2.0 * d * (sa - sb)
    if p_sq < 0.0:
        return None
    p    = math.sqrt(p_sq)
    tmp1 = math.atan2(cb - ca, tmp0)
    t    = _mod2pi(-alpha + tmp1)
    q    = _mod2pi(beta   - tmp1)
    return t, p, q


def _dubins_RSR(alpha, beta, d):
    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta),  math.cos(beta)
    tmp0 = d - sa + sb
    p_sq = 2.0 + d**2 - 2.0 * math.cos(alpha - beta) + 2.0 * d * (sb - sa)
    if p_sq < 0.0:
        return None
    p    = math.sqrt(p_sq)
    tmp1 = math.atan2(ca - cb, tmp0)
    t    = _mod2pi( alpha - tmp1)
    q    = _mod2pi(-beta  + tmp1)
    return t, p, q


def _dubins_LSR(alpha, beta, d):
    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta),  math.cos(beta)
    p_sq  = -2.0 + d**2 + 2.0 * math.cos(alpha - beta) + 2.0 * d * (sa + sb)
    if p_sq < 0.0:
        return None
    p    = math.sqrt(p_sq)
    tmp0 = d + sa + sb
    tmp2 = math.atan2(-ca - cb, tmp0 + p) - math.atan2(-2.0, p)
    t    = _mod2pi(-alpha + tmp2)
    q    = _mod2pi(-_mod2pi(beta) + tmp2)
    return t, p, q


def _dubins_RSL(alpha, beta, d):
    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta),  math.cos(beta)
    p_sq  = -2.0 + d**2 + 2.0 * math.cos(alpha - beta) - 2.0 * d * (sa + sb)
    if p_sq < 0.0:
        return None
    p    = math.sqrt(p_sq)
    tmp0 = d - sa - sb
    tmp2 = math.atan2(ca + cb, tmp0 + p) - math.atan2(2.0, p)
    t    = _mod2pi(alpha - tmp2)
    q    = _mod2pi(beta  - tmp2)
    return t, p, q


def _dubins_RLR(alpha, beta, d):
    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta),  math.cos(beta)
    tmp = (6.0 - d**2 + 2.0 * math.cos(alpha - beta) + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(2.0 * math.pi - math.acos(tmp))
    t = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + _mod2pi(p / 2.0))
    q = _mod2pi(alpha - beta - t + _mod2pi(p))
    return t, p, q


def _dubins_LRL(alpha, beta, d):
    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta),  math.cos(beta)
    tmp = (6.0 - d**2 + 2.0 * math.cos(alpha - beta) + 2.0 * d * (-sa + sb)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(2.0 * math.pi - math.acos(tmp))
    t = _mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
    q = _mod2pi(_mod2pi(beta) - alpha - t + _mod2pi(p))
    return t, p, q


_DUBINS_FNS = {
    'LSL': _dubins_LSL,
    'RSR': _dubins_RSR,
    'LSR': _dubins_LSR,
    'RSL': _dubins_RSL,
    'RLR': _dubins_RLR,
    'LRL': _dubins_LRL,
}


def dubins_shortest(x0, y0, th0, x1, y1, th1, R):
    """
    Return (path_type, t, p, q) for the shortest Dubins path from
    (x0, y0, th0) to (x1, y1, th1) with minimum turning radius R.

    t, p, q are normalised segment lengths:
      L/R arcs  → arc angle in radians  (actual arc length = angle * R)
      S straight → normalised length     (actual length = p * R)
    """
    dx, dy = x1 - x0, y1 - y0
    D      = math.hypot(dx, dy)
    d      = D / R
    phi    = math.atan2(dy, dx)
    alpha  = _mod2pi(th0 - phi)
    beta   = _mod2pi(th1 - phi)

    best = None
    for name, fn in _DUBINS_FNS.items():
        result = fn(alpha, beta, d)
        if result is None:
            continue
        t, p, q = result
        length = (t + p + q) * R
        if best is None or length < best[0]:
            best = (length, name, t, p, q)

    if best is None:
        return None
    _, path_type, t, p, q = best
    return path_type, t, p, q


def sample_dubins(x0, y0, th0, path_type, t_norm, p_norm, q_norm, R, step=0.15):
    """
    Sample a Dubins path at ~step-metre intervals.
    Returns a list of (x, y, theta) world-frame waypoints.

    For L/R segments: length is the arc angle (radians).
    For S segments:   length is the normalised straight (actual = p*R metres).
    """
    segments = [
        (path_type[0], t_norm),
        (path_type[1], p_norm),
        (path_type[2], q_norm),
    ]
    points = []
    x, y, th = x0, y0, th0

    for seg_type, length in segments:
        if length < 1e-9:
            continue

        if seg_type == 'S':
            actual = length * R
            n = max(2, int(actual / step))
            for i in range(n):
                s = (i / n) * actual
                points.append((x + s * math.cos(th),
                                y + s * math.sin(th),
                                th))
            x += actual * math.cos(th)
            y += actual * math.sin(th)

        elif seg_type == 'L':
            arc = length
            n = max(2, int(arc * R / step))
            for i in range(n):
                s = (i / n) * arc
                points.append((
                    x + R * (math.sin(th + s) - math.sin(th)),
                    y + R * (-math.cos(th + s) + math.cos(th)),
                    _mod2pi(th + s),
                ))
            x  += R * (math.sin(th + arc) - math.sin(th))
            y  += R * (-math.cos(th + arc) + math.cos(th))
            th  = _mod2pi(th + arc)

        elif seg_type == 'R':
            arc = length
            n = max(2, int(arc * R / step))
            for i in range(n):
                s = (i / n) * arc
                points.append((
                    x + R * (-math.sin(th - s) + math.sin(th)),
                    y + R * (math.cos(th - s) - math.cos(th)),
                    _mod2pi(th - s),
                ))
            x  += R * (-math.sin(th - arc) + math.sin(th))
            y  += R * (math.cos(th - arc) - math.cos(th))
            th  = _mod2pi(th - arc)

    points.append((x, y, th))
    return points


# ======================================================================
#  ROS2 simulation node
# ======================================================================

class SimulationNode(Node):

    # ---- Forklift geometry (metres) ----------------------------------
    BODY_LENGTH    = 2.5
    BODY_WIDTH     = 1.2
    BODY_HEIGHT    = 1.5
    FORK_LENGTH    = 1.3
    FORK_WIDTH     = 0.10
    FORK_THICKNESS = 0.07
    FORK_Y_OFFSET  = 0.28   # lateral distance from centre to each fork

    # ---- Pallet dimensions (metres) ----------------------------------
    # Frame origin: centre of the TOP surface.
    PALLET_LENGTH  = 0.762   # x — fork-entry depth
    PALLET_WIDTH   = 0.813   # y
    PALLET_HEIGHT  = 0.12    # z

    # ---- Vehicle kinematics ------------------------------------------
    # Minimum turning radius of the forklift (rear-wheel steered).
    # Typical warehouse counterbalanced forklift ≈ 2.0–3.0 m.
    MIN_TURN_RADIUS = 2.5    # metres

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
        self.pallet_pose = {
            'x': 6.0, 'y': 2.0, 'z': self.PALLET_HEIGHT,
            'qx': 0.0, 'qy': 0.0,
            'qz': math.sin(yaw / 2.0),
            'qw': math.cos(yaw / 2.0),
        }

        # Pose the forklift must reach for forks to be fully inserted
        self.pickup_pose = self._compute_pickup_pose()

        # Dubins path: forklift start → pickup pose
        self._path_waypoints = self._build_path_waypoints()

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('Simulation node started.')

    # ------------------------------------------------------------------ #
    #  Timer                                                               #
    # ------------------------------------------------------------------ #

    def timer_callback(self):
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
        markers.markers += self._forklift_markers(now, frame='forklift', id_base=0,  alpha=0.95)
        markers.markers += self._forklift_markers(now, frame='pickup',   id_base=20, alpha=0.30)
        markers.markers += self._pallet_markers(now)
        self.marker_pub.publish(markers)

    # --- Forklift (solid) and pickup target (ghost) ------------------- #

    def _forklift_markers(self, now, frame, id_base, alpha):
        """
        Draws forklift body + forks in `frame`.
        id_base offsets IDs so forklift and pickup markers don't collide.
        alpha < 0.5 → ghost/target style (blue); alpha ≥ 0.5 → solid (orange).
        """
        markers = []
        solid = alpha >= 0.5
        br, bg, bb = (1.0, 0.5, 0.0) if solid else (0.3, 0.6, 1.0)
        fr, fg, fb = (0.9, 0.9, 0.1) if solid else (0.5, 0.7, 1.0)

        # Body
        m = self._base_marker(frame, id_base, now, Marker.CUBE)
        m.pose.position.z  = self.BODY_HEIGHT / 2.0
        m.pose.orientation.w = 1.0
        m.scale.x = self.BODY_LENGTH
        m.scale.y = self.BODY_WIDTH
        m.scale.z = self.BODY_HEIGHT
        m.color.r, m.color.g, m.color.b, m.color.a = br, bg, bb, alpha
        markers.append(m)

        # Left and right forks
        for i, y_off in enumerate([+self.FORK_Y_OFFSET, -self.FORK_Y_OFFSET]):
            m = self._base_marker(frame, id_base + 1 + i, now, Marker.CUBE)
            m.pose.position.x = self.BODY_LENGTH / 2.0 + self.FORK_LENGTH / 2.0
            m.pose.position.y = y_off
            m.pose.position.z = self.FORK_THICKNESS / 2.0
            m.pose.orientation.w = 1.0
            m.scale.x = self.FORK_LENGTH
            m.scale.y = self.FORK_WIDTH
            m.scale.z = self.FORK_THICKNESS
            m.color.r, m.color.g, m.color.b, m.color.a = fr, fg, fb, alpha
            markers.append(m)

        # Label
        m = self._base_marker(frame, id_base + 3, now, Marker.TEXT_VIEW_FACING)
        m.pose.position.z  = self.BODY_HEIGHT + 0.4
        m.pose.orientation.w = 1.0
        m.scale.z = 0.4
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 1.0, alpha
        m.text = 'FORKLIFT' if solid else 'PICKUP TARGET'
        markers.append(m)

        return markers

    # --- Pallet ------------------------------------------------------- #

    def _pallet_markers(self, now):
        markers = []
        h = self.PALLET_HEIGHT

        # Body — origin is top surface, so centre is h/2 below
        m = self._base_marker('pallet', 10, now, Marker.CUBE)
        m.pose.position.z  = -h / 2.0
        m.pose.orientation.w = 1.0
        m.scale.x = self.PALLET_LENGTH
        m.scale.y = self.PALLET_WIDTH
        m.scale.z = h
        m.color.r, m.color.g, m.color.b, m.color.a = 0.55, 0.27, 0.07, 0.95
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
          - Forklift faces along the pallet's +X (fork-entry) axis.
          - Fork tips reach the far edge of the pallet (+PALLET_LENGTH/2
            in pallet frame) → full insertion.
          - fork_tip_reach = BODY_LENGTH/2 + FORK_LENGTH from forklift origin.
          - Forklift origin is therefore (fork_tip_reach - PALLET_LENGTH/2)
            metres behind the pallet centre, along pallet -X.
        """
        pp = self.pallet_pose
        pallet_yaw = 2.0 * math.atan2(pp['qz'], pp['qw'])

        fork_tip_reach = self.BODY_LENGTH / 2.0 + self.FORK_LENGTH
        offset         = fork_tip_reach - self.PALLET_LENGTH / 2.0

        x = pp['x'] - offset * math.cos(pallet_yaw)
        y = pp['y'] - offset * math.sin(pallet_yaw)

        return {
            'x':  x,
            'y':  y,
            'th': pallet_yaw,
            'qz': math.sin(pallet_yaw / 2.0),
            'qw': math.cos(pallet_yaw / 2.0),
        }

    # ------------------------------------------------------------------ #
    #  Dubins path planning                                                #
    # ------------------------------------------------------------------ #

    def _build_path_waypoints(self):
        """
        Plan the shortest kinematically-valid path from the forklift's
        current pose to the pickup pose using Dubins path planning.

        The vehicle constraint is MIN_TURN_RADIUS — no arc in the path
        will be tighter than this.
        """
        fp  = self.forklift_pose
        th0 = 2.0 * math.atan2(fp['qz'], fp['qw'])

        pp  = self.pickup_pose
        th1 = pp['th']

        result = dubins_shortest(
            fp['x'], fp['y'], th0,
            pp['x'], pp['y'], th1,
            self.MIN_TURN_RADIUS,
        )

        if result is None:
            self.get_logger().warn('Dubins solver returned no path — using straight line.')
            return [(fp['x'], fp['y'], th0), (pp['x'], pp['y'], th1)]

        path_type, t, p, q = result
        total_len = (t + p + q) * self.MIN_TURN_RADIUS
        self.get_logger().info(
            f'Dubins path: {path_type}  length: {total_len:.2f} m  '
            f'segments: {t*self.MIN_TURN_RADIUS:.2f} / '
            f'{p*self.MIN_TURN_RADIUS:.2f} / '
            f'{q*self.MIN_TURN_RADIUS:.2f} m'
        )

        return sample_dubins(
            fp['x'], fp['y'], th0,
            path_type, t, p, q,
            self.MIN_TURN_RADIUS,
        )


def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
