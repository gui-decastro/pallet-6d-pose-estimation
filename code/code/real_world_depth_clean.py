import numpy as np
import open3d as o3d


VOXEL_SIZE       = 0.003
SOR_NB_NEIGHBORS = 20
SOR_STD_RATIO    = 2.0
ROR_RADIUS       = 0.02
ROR_MIN_POINTS   = 3
NORMAL_RADIUS    = 0.02
NORMAL_MAX_NN    = 30


def load_xyz(path: str) -> o3d.geometry.PointCloud:
    pts = np.loadtxt(path)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    xyz = pts[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if pts.shape[1] >= 6:
        colours = pts[:, 3:6]
        if colours.max() > 1.0:
            colours = colours / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colours)

    print(f"[load]  {len(pcd.points):,} points loaded from '{path}'")

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    print(f"[bounds] min: {mins}")
    print(f"[bounds] max: {maxs}")
    print(f"[bounds] extent: {maxs - mins}")

    return pcd


def clean(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"[voxel] {len(pcd.points):,} points after voxel downsampling")

    if len(pcd.points) < 20:
        print("[warn] Too few points after voxel downsampling. Returning early.")
        return pcd

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
    print(f"[SOR]   {len(pcd.points):,} points after statistical outlier removal")

    if len(pcd.points) < 20:
        print("[warn] Too few points after SOR. Returning early.")
        return pcd

    pcd, _ = pcd.remove_radius_outlier(
        nb_points=ROR_MIN_POINTS, radius=ROR_RADIUS)
    print(f"[ROR]   {len(pcd.points):,} points after radius outlier removal")

    if len(pcd.points) < 20:
        print("[warn] Too few points left after ROR. Skipping normal estimation.")
        return pcd

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN))

    if len(pcd.points) >= 15:
        pcd.orient_normals_consistent_tangent_plane(k=15)
        print("[normals] estimated and oriented")
    else:
        print("[normals] estimated but skipped consistent orientation (too few points)")

    return pcd


def save_xyz(pcd: o3d.geometry.PointCloud, path: str) -> None:
    pts = np.asarray(pcd.points)
    data = pts

    if pcd.has_colors():
        cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        data = np.hstack([pts, cols])

    np.savetxt(path, data, fmt="%.6f")
    print(f"[save]  cleaned cloud saved to '{path}'")


def run(in_path: str, out_path: str):
    pcd = load_xyz(in_path)
    cleaned = clean(pcd)
    save_xyz(cleaned, out_path)