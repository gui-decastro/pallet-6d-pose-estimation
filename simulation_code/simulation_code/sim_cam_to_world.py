import numpy as np
from pathlib import Path


def load_cam_to_world(txt_path: str) -> np.ndarray:
    """
    Loads a 4x4 camera-to-world matrix from a .txt file.
    Lines starting with '#' are treated as comments and skipped.
    Expected format (row-major 4x4):
        # any comment lines
        -0.00000059  0.74314487  -0.66913056  9.81021786
        -1.00000000  -0.00000170  -0.00000101  3.24600267
        -0.00000189  0.66913056  0.74314487  1.50000000
         0.00000000  0.00000000   0.00000000  1.00000000
    """
    mat = np.loadtxt(txt_path, comments='#', dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix in '{txt_path}', got shape {mat.shape}")
    return mat


def run(in_path: str, out_path: str, cam_to_world_txt: str):
    """
    Transform points from camera frame to world frame using
    a 4x4 homogeneous matrix loaded from a .txt file.

    in_path:          input .xyz file (camera frame)
    out_path:         output .xyz file (world frame)
    cam_to_world_txt: path to .txt file containing the 4x4 matrix
    """
    cam_to_world_mat = load_cam_to_world(cam_to_world_txt)

    data = np.loadtxt(in_path)
    if data.ndim == 1:
        data = data[None, :]
    print(f"[load]      {len(data):,} points from '{in_path}'")
    print(f"[matrix]    loaded cam-to-world from '{cam_to_world_txt}'")

    ones     = np.ones((len(data), 1), dtype=np.float64)
    pts_h    = np.hstack([data[:, :3], ones])
    data_out = data.copy()
    data_out[:, :3] = (cam_to_world_mat @ pts_h.T).T[:, :3]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, data_out, fmt="%.6f")
    print(f"[save]      {len(data_out):,} points -> '{out_path}'")
    print(f"[stats]     center: {data_out[:, :3].mean(axis=0)}")
    print(f"[stats]     min:    {data_out[:, :3].min(axis=0)}")
    print(f"[stats]     max:    {data_out[:, :3].max(axis=0)}")