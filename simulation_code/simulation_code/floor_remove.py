from pathlib import Path
import numpy as np
import open3d as o3d


# ── Config ────────────────────────────────────────────────────────────────────
FLOOR_TH         = 0.01
FLOOR_ITERS      = 3000
FLOOR_MAX_PLANES = 3


def run(
    in_path: str,
    out_path: str,
    pick: str = "farthest",
    camera_center: tuple = (0.0, 0.0, 0.0)
):
    """
    Removes the floor plane from an ASCII .xyz file by:
      1) Extracting up to FLOOR_MAX_PLANES planes via RANSAC iteratively
      2) Selecting which plane is the floor using a heuristic
      3) Removing that plane's inliers and saving remaining points

    pick:
      - "farthest" (default): plane whose inliers have largest mean distance to camera_center
      - "largest":            plane with most inliers
    """
    data = np.loadtxt(in_path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError("Input must have at least 3 columns: x y z")

    xyz     = data[:, :3]
    cam     = np.array(camera_center, dtype=float)
    pts_rem = xyz.copy()
    idx_rem = np.arange(len(xyz))
    planes  = []

    for k in range(FLOOR_MAX_PLANES):
        if len(pts_rem) < 50:
            break

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_rem)
        _, inliers = pcd.segment_plane(
            distance_threshold=FLOOR_TH,
            ransac_n=3,
            num_iterations=FLOOR_ITERS
        )
        inliers = np.array(inliers, dtype=int)
        if inliers.size < 50:
            break

        orig_idx   = idx_rem[inliers]
        inlier_pts = xyz[orig_idx]
        centroid   = inlier_pts.mean(axis=0)
        mean_range = np.linalg.norm(inlier_pts - cam, axis=1).mean()

        planes.append({
            "k":          k,
            "orig_idx":   orig_idx,
            "count":      int(inliers.size),
            "centroid":   centroid,
            "mean_range": float(mean_range),
        })

        keep_mask          = np.ones(len(pts_rem), dtype=bool)
        keep_mask[inliers] = False
        pts_rem = pts_rem[keep_mask]
        idx_rem = idx_rem[keep_mask]

    if not planes:
        raise RuntimeError("No planes found. Try increasing FLOOR_TH or FLOOR_ITERS.")

    if pick == "farthest":
        floor = max(planes, key=lambda p: p["mean_range"])
    elif pick == "largest":
        floor = max(planes, key=lambda p: p["count"])
    else:
        raise ValueError("pick must be 'farthest' or 'largest'")

    keep              = np.ones(len(xyz), dtype=bool)
    keep[floor["orig_idx"]] = False
    out = data[keep]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, out, fmt="%.6f")

    print("Planes found:")
    for p in planes:
        print(f"  plane#{p['k']}  inliers={p['count']}  mean_range={p['mean_range']:.3f}  centroid={p['centroid']}")
    print(f"Chosen floor: plane#{floor['k']} (pick='{pick}')")
    print(f"Removed: {len(floor['orig_idx'])} pts | Kept: {out.shape[0]} | Saved: {out_path}")