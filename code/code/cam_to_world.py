import numpy as np


def rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,  0,   0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=np.float64)


def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0],
                     [sa,  ca, 0],
                     [ 0,   0, 1]], dtype=np.float64)


def cam_to_world(pts: np.ndarray) -> np.ndarray:
    """
    Transform points from camera frame to world frame.

    Camera: standard CV (X right, Y down, Z forward)
    World:  Z up, Y right, X into page (Blender convention)

    Steps:
      1. CV → Blender axis flip  (Y_cv = -Y_blender, Z_cv = -Z_blender)
      2. Rotate about X by +52.5°  (camera tilt down toward pallet)
      3. Rotate about Z by -90°    (horizontal axis alignment)
      4. Translate: camera origin is 987mm above world origin in world Z,
                    and 47.5mm offset in -Y world
    """
    R_flip = np.array([[1,  0,  0],
                       [0, -1,  0],
                       [0,  0, -1]], dtype=np.float64)

    R_tilt = rot_x(np.radians(52.5))
    R_yaw  = rot_z(np.radians(-90.0))

    R_cam_to_world = R_yaw @ R_tilt @ R_flip

    t_cam_in_world = np.array([0.0, -0.0475, 0.987], dtype=np.float64)

    return (R_cam_to_world @ pts.T).T + t_cam_in_world


def run(in_path: str, out_path: str):
    data = np.loadtxt(in_path)
    if data.ndim == 1:
        data = data[None, :]
    print(f"[load]      {len(data):,} points from '{in_path}'")

    data_out = data.copy()
    data_out[:, :3] = cam_to_world(data[:, :3])
    print(f"[transform] camera frame → world frame")

    np.savetxt(out_path, data_out, fmt="%.6f")
    print(f"[save]      {len(data_out):,} points saved to '{out_path}'")