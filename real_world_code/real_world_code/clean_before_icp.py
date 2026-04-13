import numpy as np
import open3d as o3d


DBSCAN_EPS        = 0.03
DBSCAN_MIN_POINTS = 10


def load_xyz(path: str):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    print(f"[load]   {len(pcd.points):,} points from '{path}'")
    return pcd, data


def keep_largest_cluster(pcd, data):
    """
    DBSCAN: keep only the largest cluster = the pallet.
    All smaller blobs (residual floor patches, wall slivers, noise) are dropped.
    """
    labels = np.array(pcd.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        print_progress=False
    ))

    n_clusters  = labels.max() + 1
    noise_count = int(np.sum(labels == -1))
    print(f"[DBSCAN] {n_clusters} cluster(s) found, {noise_count} noise points")

    if n_clusters == 0:
        print("[DBSCAN] WARNING: no clusters found — returning full cloud unchanged")
        print("         → try increasing DBSCAN_EPS (e.g. 0.05)")
        return pcd, data

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    for lbl, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"         cluster {lbl}: {cnt:,} points")

    largest_label = unique[np.argmax(counts)]
    mask = labels == largest_label

    pcd_out  = pcd.select_by_index(np.where(mask)[0])
    data_out = data[mask]

    print(f"[DBSCAN] keeping cluster {largest_label} with {len(pcd_out.points):,} points")
    return pcd_out, data_out


def save_xyz(data: np.ndarray, path: str) -> None:
    np.savetxt(path, data, fmt="%.6f")
    print(f"[save]   {data.shape[0]:,} points saved to '{path}'")


def run(in_path: str, out_path: str):
    pcd, data = load_xyz(in_path)
    pcd, data = keep_largest_cluster(pcd, data)
    save_xyz(data, out_path)