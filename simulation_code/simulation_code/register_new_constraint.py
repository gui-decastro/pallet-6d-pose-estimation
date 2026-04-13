import numpy as np
import open3d as o3d
import random


# ── Config ────────────────────────────────────────────────────────────────────
VOXEL_SIZE         = 0.02
N_MESH_POINTS      = 12000
ICP_MAX_ITERS      = 30
ICP_DIST           = 0.03
CONVERGE_YAW_DEG   = 0.05
CONVERGE_T         = 1e-4
YAW_STEP_DEG       = 5.0
YAW_SCORE_DIST     = 0.06
ROLL_FIXED_DEG     = 0.0
PITCH_FIXED_DEG    = 0.0
CHOOSE_MIN_ABS_YAW = True
BASE_SEED          = 7
NUM_TRIALS         = 10
VISUALIZE_BEST     = True


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    try:
        o3d.utility.random.seed(seed)
    except Exception:
        pass


def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def wrap_to_180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)


def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float64)


def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)


def make_pcd(points):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(points)
    return p


def solve_yaw_and_translation(P, Q):
    if P.shape[0] < 3:
        return None, None
    Pxy, Qxy   = P[:, :2], Q[:, :2]
    p_bar, q_bar = Pxy.mean(axis=0), Qxy.mean(axis=0)
    Pc, Qc     = Pxy - p_bar, Qxy - q_bar
    a = np.sum(Pc[:, 0] * Qc[:, 0] + Pc[:, 1] * Qc[:, 1])
    b = np.sum(Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0])
    yaw  = np.arctan2(b, a)
    R    = rot_z(yaw)
    t_xy = q_bar - (R[:2, :2] @ p_bar)
    t_z  = np.mean(Q[:, 2] - P[:, 2])
    return yaw, np.array([t_xy[0], t_xy[1], t_z], dtype=np.float64)


def find_correspondences(src_transformed, tgt_points, kdtree, max_dist):
    src_idx, tgt_idx, dists = [], [], []
    max_dist2 = max_dist * max_dist
    for i, p in enumerate(src_transformed):
        k, idx, dist2 = kdtree.search_knn_vector_3d(p, 1)
        if k == 1 and dist2[0] <= max_dist2:
            src_idx.append(i)
            tgt_idx.append(idx[0])
            dists.append(np.sqrt(dist2[0]))
    if not src_idx:
        return None, None, None
    return (np.array(src_idx, dtype=np.int64),
            np.array(tgt_idx, dtype=np.int64),
            np.array(dists,   dtype=np.float64))


def score_yaw_candidates(src_pts_fixed, tgt_pcd_down):
    tgt_pts  = np.asarray(tgt_pcd_down.points)
    tgt_mean = tgt_pts.mean(axis=0)
    best     = {"yaw": None, "t": None, "fitness": -1.0, "rmse": 1e9}
    for yaw in np.deg2rad(np.arange(-180.0, 180.0, YAW_STEP_DEG)):
        Rz      = rot_z(yaw)
        src_rot = (Rz @ src_pts_fixed.T).T
        t       = tgt_mean - src_rot.mean(axis=0)
        src_try = make_pcd(src_rot + t)
        d       = np.asarray(src_try.compute_point_cloud_distance(tgt_pcd_down))
        inliers = d < YAW_SCORE_DIST
        fit     = float(np.sum(inliers)) / len(d) if len(d) > 0 else 0.0
        rmse    = float(np.sqrt(np.mean(d[inliers] ** 2))) if np.any(inliers) else 1e9
        if (fit > best["fitness"] + 1e-9) or (abs(fit - best["fitness"]) <= 1e-9 and rmse < best["rmse"]):
            best = {"yaw": yaw, "t": t, "fitness": fit, "rmse": rmse}
    return best["yaw"], best["t"]


def constrained_icp_yaw_xyz(src_pts_fixed, tgt_points, init_yaw, init_t):
    tgt_pcd = make_pcd(tgt_points)
    kdtree  = o3d.geometry.KDTreeFlann(tgt_pcd)
    yaw, t  = float(init_yaw), init_t.astype(np.float64).copy()

    for _ in range(ICP_MAX_ITERS):
        src_trans = (rot_z(yaw) @ src_pts_fixed.T).T + t
        src_idx, tgt_idx, _ = find_correspondences(src_trans, tgt_points, kdtree, ICP_DIST)
        if src_idx is None or len(src_idx) < 50:
            return None, None, 0.0, 1e9

        new_yaw, new_t = solve_yaw_and_translation(src_pts_fixed[src_idx], tgt_points[tgt_idx])
        if new_yaw is None:
            return None, None, 0.0, 1e9

        dyaw = abs(wrap_to_pi(new_yaw - yaw))
        dt   = np.linalg.norm(new_t - t)
        yaw, t = float(new_yaw), new_t

        if dyaw < np.deg2rad(CONVERGE_YAW_DEG) and dt < CONVERGE_T:
            break

    src_trans = (rot_z(yaw) @ src_pts_fixed.T).T + t
    src_idx, tgt_idx, d = find_correspondences(src_trans, tgt_points, kdtree, ICP_DIST)
    if src_idx is None or len(src_idx) == 0:
        return None, None, 0.0, 1e9

    fitness = float(len(src_idx)) / float(src_pts_fixed.shape[0])
    rmse    = float(np.sqrt(np.mean(d ** 2)))
    return yaw, t, fitness, rmse


def is_better(curr_fit, curr_rmse, best_fit, best_rmse):
    if best_fit is None:
        return True
    if curr_fit > best_fit + 1e-9:
        return True
    if abs(curr_fit - best_fit) <= 1e-9 and curr_rmse < best_rmse:
        return True
    return False


def run(mesh_path: str, world_cloud_xyz: str):
    """
    Constrained yaw-ICP registration of mesh against world-frame point cloud.

    mesh_path:       path to .ply mesh file
    world_cloud_xyz: path to input .xyz point cloud (world frame, floor removed)
    """
    R_fixed = rot_y(np.deg2rad(PITCH_FIXED_DEG)) @ rot_x(np.deg2rad(ROLL_FIXED_DEG))

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    tgt_pts  = np.loadtxt(world_cloud_xyz).astype(np.float64)
    tgt_down = make_pcd(tgt_pts[:, :3]).voxel_down_sample(VOXEL_SIZE)
    tgt_down_pts = np.asarray(tgt_down.points).astype(np.float64)

    best = {"seed": None, "fitness": None, "rmse": None,
            "yaw": None, "t": None, "T": None, "src_down_pts_fixed": None}

    for i in range(NUM_TRIALS):
        seed = BASE_SEED + i
        set_seed(seed)

        src_pts       = np.asarray(mesh.sample_points_poisson_disk(N_MESH_POINTS).points).astype(np.float64)
        src_pts_fixed = (R_fixed @ src_pts.T).T
        src_down_pts  = np.asarray(make_pcd(src_pts_fixed).voxel_down_sample(VOXEL_SIZE).points).astype(np.float64)

        init_yaw, init_t = score_yaw_candidates(src_down_pts, tgt_down)
        if init_yaw is None:
            print(f"[trial {i+1}/{NUM_TRIALS}] seed={seed} failed init yaw")
            continue

        yaw, t, fit, rmse = constrained_icp_yaw_xyz(src_down_pts, tgt_down_pts, init_yaw, init_t)
        yaw_deg = np.degrees(yaw) if yaw is not None else None
        print(f"[trial {i+1}/{NUM_TRIALS}] seed={seed}  fitness={fit:.4f}  rmse={rmse:.4f}  yaw_deg={yaw_deg}")

        if is_better(fit, rmse, best["fitness"], best["rmse"]):
            R = rot_z(yaw) @ R_fixed
            T = np.eye(4, dtype=np.float64)
            T[:3, :3], T[:3, 3] = R, t
            best.update(seed=seed, fitness=fit, rmse=rmse, yaw=float(yaw),
                        t=t.copy(), T=T, src_down_pts_fixed=src_down_pts.copy())

    print("\n=== BEST (Yaw + XYZ, Roll/Pitch fixed) ===")
    print(f"seed={best['seed']}  fitness={best['fitness']}  rmse={best['rmse']}")

    yaw1     = float(best["yaw"])
    t1       = best["t"]
    yaw1_deg = wrap_to_180(np.degrees(yaw1))

    yaw2, t2, fit2, rmse2 = constrained_icp_yaw_xyz(
        best["src_down_pts_fixed"], tgt_down_pts,
        wrap_to_pi(yaw1 + np.pi), t1
    )
    yaw2_deg = wrap_to_180(np.degrees(float(yaw2))) if yaw2 is not None else None

    print("\nYaw candidates (deg):")
    print(f"  yaw1 = {yaw1_deg:.6f}   (fit={best['fitness']:.6f}, rmse={best['rmse']:.6f})")
    if yaw2_deg is not None:
        print(f"  yaw2 = {yaw2_deg:.6f}   (fit={fit2:.6f}, rmse={rmse2:.6f})   [= yaw1±180]")

    if not CHOOSE_MIN_ABS_YAW or yaw2_deg is None:
        yaw_final, t_final, fit_final, rmse_final, label = yaw1, t1, best["fitness"], best["rmse"], "yaw1"
    else:
        if abs(yaw2_deg) < abs(yaw1_deg):
            yaw_final, t_final, fit_final, rmse_final, label = float(yaw2), t2, fit2, rmse2, "yaw2 (yaw1±180)"
        else:
            yaw_final, t_final, fit_final, rmse_final, label = yaw1, t1, best["fitness"], best["rmse"], "yaw1"

    yaw_final_deg = wrap_to_180(np.degrees(yaw_final))
    print("\n=== CHOSEN ===")
    print(f"chosen: {label}  yaw_deg={yaw_final_deg:.6f}  t={t_final}  fitness={fit_final:.6f}  rmse={rmse_final:.6f}")

    R_final       = rot_z(yaw_final) @ R_fixed
    T_final       = np.eye(4, dtype=np.float64)
    T_final[:3, :3], T_final[:3, 3] = R_final, t_final
    print("\nT_world_mesh_chosen:\n", T_final)

    np.savetxt("T_world_mesh_chosen_yaw_xyz.txt", T_final)

    mesh_w = o3d.io.read_triangle_mesh(mesh_path)
    mesh_w.transform(T_final)
    o3d.io.write_triangle_mesh("pallet_mesh_in_world_chosen_yaw_xyz.ply", mesh_w)

    if VISUALIZE_BEST:
        set_seed(best["seed"])
        src_vis = mesh.sample_points_poisson_disk(N_MESH_POINTS)
        src_vis.paint_uniform_color([1, 0, 0])
        tgt_down.paint_uniform_color([0, 1, 0])
        src_vis.transform(T_final)
        o3d.visualization.draw_geometries([tgt_down, src_vis])