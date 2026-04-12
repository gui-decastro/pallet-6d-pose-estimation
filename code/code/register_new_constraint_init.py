import numpy as np
import random
import open3d as o3d

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MESH_PATH = "Pallet_world_dim_transforms.ply"

BASE_SEED  = 7
NUM_TRIALS = 10

VOXEL_SIZE    = 0.02
N_MESH_POINTS = 12000

ICP_MAX_ITERS    = 100
ICP_DIST         = 0.03
CONVERGE_YAW_DEG = 0.05
CONVERGE_T       = 1e-4

YAW_STEP_DEG   = 2.0
YAW_SCORE_DIST = 0.04

ROLL_FIXED_DEG  = 0.0
PITCH_FIXED_DEG = 0.0

SYMMETRY_OFFSETS_DEG = [0, 180]

FITNESS_TOLERANCE = 0.06

CHOOSE_MIN_ABS_YAW = True

VISUALIZE_BEST = True
# ───────────────────────────────────────────────────────────────────────────────


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


def load_xyz_points(path):
    return np.loadtxt(path).astype(np.float64)


def rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,  0,   0],
                     [0,  ca, -sa],
                     [0,  sa,  ca]], dtype=np.float64)


def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=np.float64)


def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0],
                     [sa,  ca, 0],
                     [ 0,   0, 1]], dtype=np.float64)


def make_pcd(points):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(points)
    return p


def principal_axis_yaw(pts):
    xy           = pts[:, :2] - pts[:, :2].mean(axis=0)
    cov          = xy.T @ xy / len(xy)
    eigvals, eigvecs = np.linalg.eigh(cov)
    dominant     = eigvecs[:, np.argmax(eigvals)]
    ratio        = eigvals.max() / (eigvals.min() + 1e-9)
    angle        = np.arctan2(dominant[1], dominant[0])
    return angle, ratio


def compute_yaw_bias(src_pca_fixed, tgt_down_pts):
    scan_yaw, scan_ratio = principal_axis_yaw(tgt_down_pts)
    cad_yaw,  cad_ratio  = principal_axis_yaw(src_pca_fixed)

    print(f"[PCA] scan long axis yaw : {np.degrees(scan_yaw):.1f}°  "
          f"(elongation ratio={scan_ratio:.2f})")
    print(f"[PCA] CAD  long axis yaw : {np.degrees(cad_yaw):.1f}°  "
          f"(elongation ratio={cad_ratio:.2f})")

    if scan_ratio < 1.3:
        print("[PCA] WARNING: scan is nearly square — "
              "PCA unreliable, falling back to full 360° search")
        return None
    if cad_ratio < 1.3:
        print("[PCA] WARNING: CAD is nearly square — "
              "PCA unreliable, falling back to full 360° search")
        return None

    bias = wrap_to_pi(scan_yaw - cad_yaw)
    print(f"[PCA] yaw_bias            : {np.degrees(bias):.1f}°")
    print(f"[PCA] coarse search basins: "
          f"{wrap_to_180(np.degrees(bias)):.1f}° ± 90°  and  "
          f"{wrap_to_180(np.degrees(wrap_to_pi(bias + np.pi))):.1f}° ± 90°")
    return bias


def snap_to_180_symmetry(yaw_candidate, yaw_reference):
    opt1  = yaw_candidate
    opt2  = wrap_to_pi(yaw_candidate + np.pi)
    diff1 = abs(wrap_to_pi(opt1 - yaw_reference))
    diff2 = abs(wrap_to_pi(opt2 - yaw_reference))
    return opt1 if diff1 <= diff2 else opt2


def solve_yaw_and_translation(P, Q):
    if P.shape[0] < 3:
        return None, None

    Pxy   = P[:, :2]
    Qxy   = Q[:, :2]
    p_bar = Pxy.mean(axis=0)
    q_bar = Qxy.mean(axis=0)
    Pc    = Pxy - p_bar
    Qc    = Qxy - q_bar

    a   = np.sum(Pc[:, 0] * Qc[:, 0] + Pc[:, 1] * Qc[:, 1])
    b   = np.sum(Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0])
    yaw = np.arctan2(b, a)

    R    = rot_z(yaw)
    t_xy = q_bar - (R[:2, :2] @ p_bar)
    t_z  = Q[:, 2].max() - P[:, 2].max()

    return yaw, np.array([t_xy[0], t_xy[1], t_z], dtype=np.float64)


def build_kdtree(target_points):
    tgt_pcd = make_pcd(target_points)
    return tgt_pcd, o3d.geometry.KDTreeFlann(tgt_pcd)


def find_correspondences(src_transformed, tgt_points, kdtree, max_dist):
    inlier_src_idx = []
    inlier_tgt_idx = []
    inlier_dist    = []
    max_dist2      = max_dist * max_dist

    for i in range(src_transformed.shape[0]):
        k, idx, dist2 = kdtree.search_knn_vector_3d(src_transformed[i], 1)
        if k == 1 and dist2[0] <= max_dist2:
            inlier_src_idx.append(i)
            inlier_tgt_idx.append(idx[0])
            inlier_dist.append(np.sqrt(dist2[0]))

    if len(inlier_src_idx) == 0:
        return None, None, None

    return (np.array(inlier_src_idx, dtype=np.int64),
            np.array(inlier_tgt_idx, dtype=np.int64),
            np.array(inlier_dist,    dtype=np.float64))


def score_yaw_candidates(src_points_fixed, tgt_pcd_down, tgt_centroid, yaw_bias=None):
    tgt_pts = np.asarray(tgt_pcd_down.points)
    best    = {"yaw": None, "t": None, "fitness": -1.0, "rmse": 1e9}

    if yaw_bias is None:
        yaw_vals = [wrap_to_pi(np.deg2rad(y))
                    for y in np.arange(-180.0, 180.0, YAW_STEP_DEG)]
    else:
        yaw_vals = []
        for center in [yaw_bias, wrap_to_pi(yaw_bias + np.pi)]:
            for off in np.arange(-90.0, 90.0, YAW_STEP_DEG):
                yaw_vals.append(wrap_to_pi(center + np.deg2rad(off)))

    for yaw in yaw_vals:
        Rz      = rot_z(yaw)
        src_rot = (Rz @ src_points_fixed.T).T

        src_centroid = src_rot.mean(axis=0)
        t            = tgt_centroid - src_centroid
        t[2]         = tgt_pts[:, 2].max() - src_rot[:, 2].max()

        src_try_pcd = make_pcd(src_rot + t)
        d           = np.asarray(src_try_pcd.compute_point_cloud_distance(tgt_pcd_down))
        inliers     = d < YAW_SCORE_DIST
        fit         = float(np.sum(inliers)) / float(len(d)) if len(d) > 0 else 0.0
        rmse        = float(np.sqrt(np.mean(d[inliers] ** 2))) if np.any(inliers) else 1e9

        if (fit > best["fitness"] + 1e-9) or \
           (abs(fit - best["fitness"]) <= 1e-9 and rmse < best["rmse"]):
            best.update({"yaw": yaw, "t": t.copy(), "fitness": fit, "rmse": rmse})

    return best["yaw"], best["t"]


def constrained_icp_yaw_xyz(src_points_fixed, tgt_points, init_yaw, init_t,
                             min_inliers=50):
    _, kdtree = build_kdtree(tgt_points)
    yaw       = float(init_yaw)
    t         = init_t.astype(np.float64).copy()
    converged = False

    for _ in range(ICP_MAX_ITERS):
        src_trans            = (rot_z(yaw) @ src_points_fixed.T).T + t
        src_idx, tgt_idx, _  = find_correspondences(
            src_trans, tgt_points, kdtree, ICP_DIST)

        if src_idx is None or len(src_idx) < min_inliers:
            return None, None, 0.0, 1e9

        new_yaw, new_t = solve_yaw_and_translation(
            src_points_fixed[src_idx], tgt_points[tgt_idx])
        if new_yaw is None:
            return None, None, 0.0, 1e9

        dyaw    = abs(wrap_to_pi(new_yaw - yaw))
        dt      = np.linalg.norm(new_t - t)
        yaw, t  = float(new_yaw), new_t

        if dyaw < np.deg2rad(CONVERGE_YAW_DEG) and dt < CONVERGE_T:
            converged = True
            break

    if not converged:
        print(f"    [ICP] WARNING: hit max iterations ({ICP_MAX_ITERS}) "
              "without converging")

    src_trans            = (rot_z(yaw) @ src_points_fixed.T).T + t
    src_idx, tgt_idx, d = find_correspondences(
        src_trans, tgt_points, kdtree, ICP_DIST)

    if src_idx is None or len(src_idx) == 0:
        return None, None, 0.0, 1e9

    fitness = float(len(src_idx)) / float(src_points_fixed.shape[0])
    rmse    = float(np.sqrt(np.mean(d ** 2)))
    return yaw, t, fitness, rmse


def evaluate_symmetric_yaws(base_yaw, src_pts_fixed, tgt_down_pts, t_init):
    candidates = []
    for offset_deg in SYMMETRY_OFFSETS_DEG:
        yaw_try               = wrap_to_pi(base_yaw + np.deg2rad(offset_deg))
        yaw_out, t_out, fit, rmse = constrained_icp_yaw_xyz(
            src_pts_fixed, tgt_down_pts, yaw_try, t_init)

        if yaw_out is not None:
            print(f"    offset={offset_deg:+4d}°  fit={fit:.4f}  rmse={rmse:.4f}  "
                  f"yaw={wrap_to_180(np.degrees(yaw_out)):.3f}°")
            candidates.append((fit, rmse, yaw_out, t_out, offset_deg))
        else:
            print(f"    offset={offset_deg:+4d}°  FAILED (too few inliers)")

    if not candidates:
        return None, None, None, None, None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    b = candidates[0]
    return b[0], b[1], b[2], b[3], b[4]


def try_rotation_fix(yaw_best, t_best, fit_best, rmse_best,
                     src_pts_fixed, tgt_down_pts):
    print("\n[rot-fix] Trying current best rotated by ±90° ...")
    print(f"[rot-fix] current  fit={fit_best:.4f}  rmse={rmse_best:.4f}  "
          f"yaw={wrap_to_180(np.degrees(yaw_best)):.2f}°")

    for offset_deg in [-90, 90]:
        yaw_try              = wrap_to_pi(yaw_best + np.deg2rad(offset_deg))
        yaw_out, t_out, fit, rmse = constrained_icp_yaw_xyz(
            src_pts_fixed, tgt_down_pts, yaw_try, t_best)

        if yaw_out is None:
            print(f"[rot-fix] offset={offset_deg:+d}°  FAILED (too few inliers)")
            continue

        print(f"[rot-fix] offset={offset_deg:+d}°  fit={fit:.4f}  rmse={rmse:.4f}  "
              f"yaw={wrap_to_180(np.degrees(yaw_out)):.2f}°", end="")

        if fit > fit_best + FITNESS_TOLERANCE or \
           (abs(fit - fit_best) <= FITNESS_TOLERANCE and rmse < rmse_best):
            print("  ✓ BETTER — switching")
            yaw_best  = yaw_out
            t_best    = t_out
            fit_best  = fit
            rmse_best = rmse
        else:
            print("  (not better, keeping current)")

    return yaw_best, t_best, fit_best, rmse_best


def is_better(curr_fit, curr_rmse, best_fit, best_rmse):
    if best_fit is None:
        return True
    if curr_fit > best_fit + 1e-9:
        return True
    if abs(curr_fit - best_fit) <= 1e-9 and curr_rmse < best_rmse:
        return True
    return False


def run(world_cloud_xyz: str):
    roll    = np.deg2rad(ROLL_FIXED_DEG)
    pitch   = np.deg2rad(PITCH_FIXED_DEG)
    R_fixed = rot_y(pitch) @ rot_x(roll)

    mesh = o3d.io.read_triangle_mesh(MESH_PATH)
    mesh.compute_vertex_normals()

    tgt_pts_full = load_xyz_points(world_cloud_xyz)
    tgt_pcd_full = make_pcd(tgt_pts_full)

    print("[sanity] Scan X range: [{:.3f}, {:.3f}]  span={:.3f}".format(
        tgt_pts_full[:,0].min(), tgt_pts_full[:,0].max(),
        tgt_pts_full[:,0].max() - tgt_pts_full[:,0].min()))
    print("[sanity] Scan Y range: [{:.3f}, {:.3f}]  span={:.3f}".format(
        tgt_pts_full[:,1].min(), tgt_pts_full[:,1].max(),
        tgt_pts_full[:,1].max() - tgt_pts_full[:,1].min()))
    print("[sanity] Scan Z range: [{:.3f}, {:.3f}]  span={:.3f}".format(
        tgt_pts_full[:,2].min(), tgt_pts_full[:,2].max(),
        tgt_pts_full[:,2].max() - tgt_pts_full[:,2].min()))

    tgt_down     = tgt_pcd_full.voxel_down_sample(VOXEL_SIZE)
    tgt_down_pts = np.asarray(tgt_down.points).astype(np.float64)

    tgt_centroid = tgt_down_pts.mean(axis=0)
    print(f"[centroid] Depth cloud centroid (x,y,z): {tgt_centroid}")

    set_seed(BASE_SEED)
    src_pca       = mesh.sample_points_poisson_disk(number_of_points=N_MESH_POINTS)
    src_pca_pts   = np.asarray(src_pca.points).astype(np.float64)
    src_pca_fixed = (R_fixed @ src_pca_pts.T).T

    print()
    yaw_bias = compute_yaw_bias(src_pca_fixed, tgt_down_pts)
    print()

    best = {
        "seed":               None,
        "fitness":            None,
        "rmse":               None,
        "yaw":                None,
        "t":                  None,
        "T":                  None,
        "src_down_pts_fixed": None,
        "tgt_down":           tgt_down,
    }

    for i in range(NUM_TRIALS):
        seed = BASE_SEED + i
        set_seed(seed)

        src                = mesh.sample_points_poisson_disk(number_of_points=N_MESH_POINTS)
        src_pts            = np.asarray(src.points).astype(np.float64)
        src_pts_fixed      = (R_fixed @ src_pts.T).T
        src_down           = make_pcd(src_pts_fixed).voxel_down_sample(VOXEL_SIZE)
        src_down_pts_fixed = np.asarray(src_down.points).astype(np.float64)

        init_yaw, init_t = score_yaw_candidates(
            src_down_pts_fixed, tgt_down, tgt_centroid, yaw_bias=yaw_bias)
        if init_yaw is None:
            print(f"[trial {i+1}/{NUM_TRIALS}] seed={seed}  FAILED init yaw")
            continue

        yaw, t, fit, rmse = constrained_icp_yaw_xyz(
            src_down_pts_fixed, tgt_down_pts, init_yaw, init_t)

        yaw_deg = wrap_to_180(np.degrees(yaw)) if yaw is not None else None
        print(f"[trial {i+1}/{NUM_TRIALS}] seed={seed}  "
              f"fitness={fit:.4f}  rmse={rmse:.4f}  yaw_deg={yaw_deg}")

        if is_better(fit, rmse, best["fitness"], best["rmse"]):
            R = rot_z(yaw) @ R_fixed
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3,  3] = t
            best.update({
                "seed":               seed,
                "fitness":            fit,
                "rmse":               rmse,
                "yaw":                float(yaw),
                "t":                  t.astype(np.float64).copy(),
                "T":                  T,
                "src_down_pts_fixed": src_down_pts_fixed.copy(),
            })

    if best["seed"] is None:
        print("\nERROR: all trials failed. "
              "Try increasing ICP_DIST or loosening YAW_SCORE_DIST.")
        return

    print("\n=== BEST from trials ===")
    print(f"  seed={best['seed']}  fitness={best['fitness']:.6f}  "
          f"rmse={best['rmse']:.6f}  yaw={wrap_to_180(np.degrees(best['yaw'])):.3f}°")

    print(f"\n[symmetry] Checking 0° and 180° offsets ...")
    fit_final, rmse_final, yaw_final, t_final, chosen_offset = \
        evaluate_symmetric_yaws(
            best["yaw"],
            best["src_down_pts_fixed"],
            tgt_down_pts,
            best["t"],
        )

    if yaw_final is None:
        print("ERROR: all symmetric candidates failed. Using raw best.")
        yaw_final, t_final    = best["yaw"], best["t"]
        fit_final, rmse_final = best["fitness"], best["rmse"]
        chosen_offset         = 0

    print(f"\n[symmetry] Chosen offset={chosen_offset:+d}°  "
          f"yaw={wrap_to_180(np.degrees(yaw_final)):.3f}°  "
          f"fit={fit_final:.4f}  rmse={rmse_final:.4f}")

    yaw_after_symmetry = yaw_final

    yaw_final, t_final, fit_final, rmse_final = try_rotation_fix(
        yaw_final, t_final, fit_final, rmse_final,
        best["src_down_pts_fixed"],
        tgt_down_pts,
    )

    yaw_snapped = snap_to_180_symmetry(yaw_final, yaw_after_symmetry)

    if abs(wrap_to_pi(yaw_snapped - yaw_final)) > 0.01:
        print(f"\n[snap] Snapping yaw from "
              f"{wrap_to_180(np.degrees(yaw_final)):.2f}° → "
              f"{wrap_to_180(np.degrees(yaw_snapped)):.2f}° "
              f"(180-deg equivalent closest to symmetry-check result "
              f"{wrap_to_180(np.degrees(yaw_after_symmetry)):.1f}°)")

        yaw_out, t_out, fit_out, rmse_out = constrained_icp_yaw_xyz(
            best["src_down_pts_fixed"], tgt_down_pts,
            yaw_snapped, t_final)

        if yaw_out is not None:
            yaw_final  = yaw_out
            t_final    = t_out
            fit_final  = fit_out
            rmse_final = rmse_out
            print(f"[snap] ICP from snapped yaw: "
                  f"fit={fit_final:.4f}  rmse={rmse_final:.4f}  "
                  f"yaw={wrap_to_180(np.degrees(yaw_final)):.2f}°")
        else:
            print("[snap] ICP from snapped yaw failed — keeping pre-snap result")
            yaw_final = yaw_snapped
    else:
        print(f"\n[snap] Yaw {wrap_to_180(np.degrees(yaw_final)):.2f}° "
              f"already in correct basin — no snap needed")

    if CHOOSE_MIN_ABS_YAW:
        yaw1     = float(yaw_final)
        t1       = t_final
        fit1     = fit_final
        rmse1    = rmse_final
        yaw1_deg = wrap_to_180(np.degrees(yaw1))

        yaw2_init     = wrap_to_pi(yaw1 + np.pi)
        yaw2_init_deg = wrap_to_180(np.degrees(yaw2_init))

        yaw2, t2, fit2, rmse2 = constrained_icp_yaw_xyz(
            best["src_down_pts_fixed"], tgt_down_pts, yaw2_init, t1)

        yaw2_deg = wrap_to_180(np.degrees(float(yaw2))) if yaw2 is not None else None

        print("\n[min-abs-yaw] Yaw candidates (deg):")
        print(f"  yaw1 = {yaw1_deg:.6f}   (fit={fit1:.6f}, rmse={rmse1:.6f})")
        if yaw2_deg is not None:
            print(f"  yaw2 = {yaw2_deg:.6f}   "
                  f"(fit={fit2:.6f}, rmse={rmse2:.6f})   [= yaw1±180]")
        else:
            print(f"  yaw2 = {yaw2_init_deg:.6f}   "
                  f"(ICP refinement failed)   [= yaw1±180 init]")

        if yaw2_deg is not None and abs(yaw2_deg) < abs(yaw1_deg):
            yaw_final  = float(yaw2)
            t_final    = t2
            fit_final  = fit2
            rmse_final = rmse2
            print("[min-abs-yaw] Chosen: yaw2 (yaw1±180)")
        else:
            print("[min-abs-yaw] Chosen: yaw1")

    yaw_final_deg = wrap_to_180(np.degrees(yaw_final))

    print("\n=== FINAL RESULT ===")
    print(f"  chosen_yaw_deg  : {yaw_final_deg:.6f}°")
    print(f"  chosen_t (x,y,z): {t_final}")
    print(f"  chosen_fitness  : {fit_final:.6f}")
    print(f"  chosen_rmse     : {rmse_final:.6f}")

    R_final         = rot_z(yaw_final) @ R_fixed
    T_final         = np.eye(4, dtype=np.float64)
    T_final[:3, :3] = R_final
    T_final[:3,  3] = t_final

    det = np.linalg.det(R_final)
    print(f"\n[sanity] R determinant        : {det:.6f}  (should be +1.0)")
    print(f"[sanity] Translation magnitude: {np.linalg.norm(t_final):.4f} m")
    print("\nT_world_mesh_chosen:\n", T_final)

    np.savetxt("T_world_mesh_chosen_yaw_xyz.txt", T_final)

    mesh_w = o3d.io.read_triangle_mesh(MESH_PATH)
    mesh_w.transform(T_final)
    o3d.io.write_triangle_mesh("pallet_mesh_in_world_chosen_yaw_xyz.ply", mesh_w)

    cad_pts_world = np.asarray(mesh_w.vertices)
    dz = cad_pts_world[:,2].max() - tgt_pts_full[:,2].max()
    print(f"\n[sanity] Transformed CAD Z range : "
          f"[{cad_pts_world[:,2].min():.3f}, {cad_pts_world[:,2].max():.3f}]")
    print(f"[sanity] Scan Z range            : "
          f"[{tgt_pts_full[:,2].min():.3f}, {tgt_pts_full[:,2].max():.3f}]")
    print(f"[sanity] Top-surface Z delta     : {dz:.4f} m  "
          f"({'✓ good' if abs(dz) < 0.02 else '⚠ check Z alignment'})")

    if VISUALIZE_BEST:
        set_seed(best["seed"])
        src_full = mesh.sample_points_poisson_disk(number_of_points=N_MESH_POINTS)
        src_full.paint_uniform_color([1, 0, 0])
        tgt_vis  = tgt_down
        tgt_vis.paint_uniform_color([0, 1, 0])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        src_full.transform(T_final)
        o3d.visualization.draw_geometries(
            [tgt_vis, src_full, frame],
            window_name="ICP result — red=CAD  green=scan")

    pallet_x   = t_final[0]
    pallet_y   = t_final[1]
    pallet_z   = t_final[2]
    pallet_yaw = yaw_final_deg

    print("\n=== PALLET POSE IN WORLD FRAME ===")
    print(f"  x   : {pallet_x:.4f} m")
    print(f"  y   : {pallet_y:.4f} m")
    print(f"  z   : {pallet_z:.4f} m")
    print(f"  yaw : {pallet_yaw:.4f}°  (rotation about Z axis)")