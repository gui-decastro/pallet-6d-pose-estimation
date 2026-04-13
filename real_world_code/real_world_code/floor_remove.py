import numpy as np
import open3d as o3d


def run(in_path: str, out_path: str, pick: str = "farthest",
        th: float = 0.01, iters: int = 3000, max_planes: int = 3,
        camera_center: tuple = (0.0, 0.0, 0.0)):
    """
    Removes the floor plane from an ASCII .xyz file by:
      1) extracting up to max_planes planes via RANSAC
      2) selecting which plane is 'floor' using a heuristic
      3) removing that plane's inliers and saving remaining points

    pick:
      - "farthest" (default): plane whose inliers have largest mean distance to camera_center
      - "largest": plane with most inliers
    """
    data = np.loadtxt(in_path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError("Input must have at least 3 columns: x y z")

    xyz = data[:, :3]
    cam = np.array(camera_center, dtype=float)

    pts_rem = xyz.copy()
    idx_rem = np.arange(len(xyz))

    planes = []
    for k in range(max_planes):
        if len(pts_rem) < 50:
            break

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_rem)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=th,
            ransac_n=3,
            num_iterations=iters
        )

        inliers = np.array(inliers, dtype=int)
        if inliers.size < 50:
            break

        orig_inlier_idx = idx_rem[inliers]
        inlier_pts = xyz[orig_inlier_idx]

        centroid   = inlier_pts.mean(axis=0)
        mean_range = np.linalg.norm(inlier_pts - cam, axis=1).mean()

        planes.append({
            "k":           k,
            "plane_model": plane_model,
            "orig_idx":    orig_inlier_idx,
            "count":       int(inliers.size),
            "centroid":    centroid,
            "mean_range":  float(mean_range),
        })

        keep_mask = np.ones(len(pts_rem), dtype=bool)
        keep_mask[inliers] = False
        pts_rem = pts_rem[keep_mask]
        idx_rem = idx_rem[keep_mask]

    if not planes:
        raise RuntimeError("No planes found. Try increasing th or iters.")

    if pick == "farthest":
        floor = max(planes, key=lambda p: p["mean_range"])
    elif pick == "largest":
        floor = max(planes, key=lambda p: p["count"])
    else:
        raise ValueError("pick must be 'farthest' or 'largest'")

    floor_idx = floor["orig_idx"]

    keep = np.ones(len(xyz), dtype=bool)
    keep[floor_idx] = False
    out = data[keep]

    np.savetxt(out_path, out, fmt="%.6f")

    print("Planes found:")
    for p in planes:
        print(f"  plane#{p['k']}  inliers={p['count']}  "
              f"mean_range={p['mean_range']:.3f}  centroid={p['centroid']}")
    print(f"Chosen floor plane: plane#{floor['k']} (pick='{pick}')")
    print(f"Removed floor points: {len(floor_idx)} | "
          f"Kept: {out.shape[0]} | Saved: {out_path}")