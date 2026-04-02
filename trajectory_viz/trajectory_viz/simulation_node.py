import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
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
    Plan a 3-segment path for a differential-drive robot and return
    a list of (x, y, theta) world-frame waypoints.

    step         — straight-line sample spacing (metres)
    rot_step_deg — heading sample spacing during in-place rotations (degrees)
    """
    points = []
    rot_step = math.radians(rot_step_deg)

    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    th_travel = math.atan2(dy, dx)

    # ---- Segment 1: rotate in place at (x0, y0) ----------------------
    delta1 = _angle_diff(th_travel, th0)
    n1 = max(2, int(abs(delta1) / rot_step))
    for i in range(n1):
        th = th0 + (i / n1) * delta1
        points.append((x0, y0, th))

    # ---- Segment 2: drive straight to (x1, y1) -----------------------
    n2 = max(2, int(dist / step))
    for i in range(n2):
        frac = i / n2
        points.append((x0 + frac * dx, y0 + frac * dy, th_travel))

    # ---- Segment 3: rotate in place at (x1, y1) ----------------------
    delta3 = _angle_diff(th1, th_travel)
    n3 = max(2, int(abs(delta3) / rot_step))
    for i in range(n3 + 1):
        th = th_travel + (i / n3) * delta3 if n3 > 0 else th1
        points.append((x1, y1, th))

    return points


# ======================================================================
#  ROS2 simulation node
# ======================================================================

class SimulationNode(Node):

    # ---- Forklift geometry (metres) ----------------------------------
    BODY_LENGTH    = 2.5
    BODY_WIDTH     = 1.2
    BODY_HEIGHT    = 1.5
    FORK_LENGTH    = 1.07 # 42 inches
    FORK_WIDTH     = 0.10
    FORK_THICKNESS = 0.07
    FORK_Y_OFFSET  = 0.28   # lateral distance from centre to each fork

    # ---- Pallet dimensions (metres) ----------------------------------
    # Frame origin: centre of the TOP surface.
    PALLET_LENGTH  = 0.762   # x — fork-entry depth
    PALLET_WIDTH   = 0.813   # y
    PALLET_HEIGHT  = 0.12    # z


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
    #  Path planning                                                       #
    # ------------------------------------------------------------------ #

    def _build_path_waypoints(self):
        """
        Plan a differential-drive path from the forklift's current pose
        to the pickup pose: rotate in place → drive straight → rotate in place.
        """
        fp  = self.forklift_pose
        th0 = 2.0 * math.atan2(fp['qz'], fp['qw'])

        pp  = self.pickup_pose
        th1 = pp['th']

        waypoints = plan_diff_drive_path(
            fp['x'], fp['y'], th0,
            pp['x'], pp['y'], th1,
        )
        dist = math.hypot(pp['x'] - fp['x'], pp['y'] - fp['y'])
        self.get_logger().info(
            f'Diff-drive path: {len(waypoints)} waypoints, '
            f'straight dist: {dist:.2f} m'
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
