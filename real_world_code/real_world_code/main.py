"""
Pallet pose estimation pipeline
================================
Steps:
  1. real_world_depth_crop   — YOLO detection → depth crop → camera-frame .xyz
  2. cam_to_world            — camera frame → world frame
  3. real_world_depth_clean  — voxel + SOR + ROR + normals
  4. floor_remove            — RANSAC plane removal
  5. clean_before_icp        — DBSCAN largest-cluster extraction
  6. register_new_constraint_init — constrained yaw-ICP registration

Usage:
  python main.py --rgb <path> --depth_bin <path> --intrinsics <path> --out_dir <path> [--pick {farthest,largest}]
"""

import argparse
import json
import threading
import time
from pathlib import Path

GT_POSE_FILE = Path(__file__).parent / ".gt_pose.json"

# ── Pipeline imports ──────────────────────────────────────────────────────────
import real_world_depth_crop
import cam_to_world
import real_world_depth_clean
import floor_remove
import clean_before_icp
import register_new_constraint_init
import publish_to_viz

# ── User-editable config ──────────────────────────────────────────────────────
WEIGHTS_PATH = str(Path(__file__).parent.parent / "models" / "yolo.pt")

YOLO_DEVICE  = 'cpu'
YOLO_IMGSZ   = 640
YOLO_CONF    = 0.01
Z_MIN        = 0.5
Z_MAX        = 4.0
DEPTH_STRIDE = 2
# ─────────────────────────────────────────────────────────────────────────────


def build_paths(out_dir: str):
    o = out_dir
    return {
        "crop_cam":   f"{o}/crop.xyz",
        "crop_debug": f"{o}/crop_rgb_box.png",
        "crop_world": f"{o}/crop_world.xyz",
        "cleaned":    f"{o}/crop_world_cleaned.xyz",
        "no_floor":   f"{o}/crop_world_cleaned_floor_removed.xyz",
        "icp_input":  f"{o}/crop_world_cleaned_floor_removed_cleaned.xyz",
    }


def main():
    parser = argparse.ArgumentParser(description="Pallet pose estimation pipeline")
    parser.add_argument("--rgb",        required=True, help="Path to RGB image (.png)")
    parser.add_argument("--depth_bin",  required=True, help="Path to depth binary file (.bin)")
    parser.add_argument("--intrinsics", required=True, help="Path to camera intrinsics JSON")
    parser.add_argument("--out_dir",    required=True, help="Output folder for all intermediate and final files")
    parser.add_argument(
        "--pick",
        choices=["farthest", "largest"],
        default="farthest",
        help="Floor-plane selection heuristic for RANSAC removal "
             "(default: farthest — picks plane whose inliers are farthest from camera; "
             "largest — picks plane with most inliers)",
    )
    args = parser.parse_args()

    p = build_paths(args.out_dir)

    # Ensure output directory exists
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    t_pipeline_start = time.perf_counter()

    # ── Step 1: YOLO detection + depth unprojection ───────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Depth crop (YOLO + unprojection)")
    print("="*60)
    result = real_world_depth_crop.run(
        weights_path=WEIGHTS_PATH,
        rgb_path=args.rgb,
        depth_bin_path=args.depth_bin,
        intrinsics_json_path=args.intrinsics,
        out_xyz_path=p["crop_cam"],
        out_debug_depth_vis_with_box=p["crop_debug"],
        device=YOLO_DEVICE,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        z_min=Z_MIN,
        z_max=Z_MAX,
        stride=DEPTH_STRIDE,
    )
    if result is None:
        print("Pipeline aborted: no YOLO detection.")
        return
    _, _, t_yolo, t_depth_crop = result

    # ── Step 2: Camera → world transform ─────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — Camera frame → world frame")
    print("="*60)
    cam_to_world.run(
        in_path=p["crop_cam"],
        out_path=p["crop_world"],
    )

    # ── Step 3: Point cloud cleaning ─────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 — Point cloud cleaning (voxel + SOR + ROR + normals)")
    print("="*60)
    t0 = time.perf_counter()
    real_world_depth_clean.run(
        in_path=p["crop_world"],
        out_path=p["cleaned"],
    )
    t_pc_clean = time.perf_counter() - t0

    # ── Step 4: Floor removal ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"STEP 4 — Floor removal (RANSAC, pick='{args.pick}')")
    print("="*60)
    t0 = time.perf_counter()
    floor_remove.run(
        in_path=p["cleaned"],
        out_path=p["no_floor"],
        pick=args.pick,
    )
    t_floor = time.perf_counter() - t0

    # ── Step 5: DBSCAN — keep largest cluster ────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5 — DBSCAN largest cluster (pre-ICP clean)")
    print("="*60)
    t0 = time.perf_counter()
    clean_before_icp.run(
        in_path=p["no_floor"],
        out_path=p["icp_input"],
    )
    t_dbscan = time.perf_counter() - t0

    # ── Step 6: Constrained yaw-ICP registration ─────────────────────────────
    print("\n" + "="*60)
    print("STEP 6 — Constrained yaw-ICP registration")
    print("="*60)
    t0 = time.perf_counter()
    result = register_new_constraint_init.run(
        world_cloud_xyz=p["icp_input"],
        out_dir=args.out_dir,
    )
    t_icp = time.perf_counter() - t0

    t_pipeline_total = time.perf_counter() - t_pipeline_start

    # ── Timing summary ────────────────────────────────────────────────────────
    t_pc_total = t_pc_clean + t_floor + t_dbscan
    metrics = {
        "yolo_s":           round(t_yolo,           3),
        "depth_crop_s":     round(t_depth_crop,      3),
        "pc_clean_s":       round(t_pc_clean,        3),
        "floor_remove_s":   round(t_floor,           3),
        "dbscan_s":         round(t_dbscan,          3),
        "pc_total_s":       round(t_pc_total,        3),
        "icp_s":            round(t_icp,             3),
        "pipeline_total_s": round(t_pipeline_total,  3),
    }

    # ── Compute final pose + GT errors before printing summary ───────────────
    import math
    if result is not None:
        pallet_x, pallet_y, pallet_yaw_rad, viz_fn = result
        yaw_deg            = math.degrees(pallet_yaw_rad)
        yaw_deg_normalized = ((yaw_deg + 45) % 90) - 45
        pallet_yaw_rad     = math.radians(yaw_deg_normalized)
    else:
        viz_fn = None

    if GT_POSE_FILE.exists() and result is not None:
        gt         = json.loads(GT_POSE_FILE.read_text())
        gt_x       = float(gt["x"])
        gt_y       = float(gt["y"])
        gt_yaw_deg = float(gt["yaw_deg"])

        err_x    = pallet_x - gt_x
        err_y    = pallet_y - gt_y
        err_xy   = math.hypot(err_x, err_y)
        raw_diff = yaw_deg_normalized - gt_yaw_deg
        err_yaw  = min(abs((raw_diff + 45) % 90 - 45),
                       abs((raw_diff - 45) % 90 + 45))

        # Fork pocket displacement — pockets at ±0.369 m along pallet Y-axis.
        # Rotate each pocket into world frame under both poses and compare.
        pocket_y_offset = 0.369
        pockets_local   = [(0.0, pocket_y_offset), (0.0, -pocket_y_offset)]

        def _world_pocket(lx, ly, x, y, yaw_rad):
            c, s = math.cos(yaw_rad), math.sin(yaw_rad)
            return (x + c * lx - s * ly, y + s * lx + c * ly)

        yaw_est_rad = math.radians(yaw_deg_normalized)
        yaw_gt_rad  = math.radians(gt_yaw_deg)
        pocket_errs = [
            math.hypot(
                _world_pocket(lx, ly, pallet_x, pallet_y, yaw_est_rad)[0] -
                _world_pocket(lx, ly, gt_x,     gt_y,     yaw_gt_rad)[0],
                _world_pocket(lx, ly, pallet_x, pallet_y, yaw_est_rad)[1] -
                _world_pocket(lx, ly, gt_x,     gt_y,     yaw_gt_rad)[1],
            )
            for lx, ly in pockets_local
        ]
        fork_pocket_err = sum(pocket_errs) / len(pocket_errs)

        metrics["gt_x"]             = round(gt_x,            4)
        metrics["gt_y"]             = round(gt_y,            4)
        metrics["gt_yaw_deg"]       = round(gt_yaw_deg,      4)
        metrics["err_x_m"]          = round(err_x,           4)
        metrics["err_y_m"]          = round(err_y,           4)
        metrics["err_xy_m"]         = round(err_xy,          4)
        metrics["err_yaw_deg"]      = round(err_yaw,         4)
        metrics["fork_pocket_err_m"] = round(fork_pocket_err, 4)
    else:
        gt_x = gt_y = gt_yaw_deg = None
        err_x = err_y = err_xy = err_yaw = fork_pocket_err = None

    print("\n" + "="*60)
    print("Pipeline complete.")
    print("="*60)
    print(f"\n{'─'*44}")
    print(f"  {'Step':<32} {'Time':>8}")
    print(f"{'─'*44}")
    print(f"  {'YOLO inference':<32} {t_yolo:>7.3f}s")
    print(f"  {'Depth crop (unproject)':<32} {t_depth_crop:>7.3f}s")
    print(f"  {'PC clean (voxel+SOR+ROR)':<32} {t_pc_clean:>7.3f}s")
    print(f"  {'Floor removal (RANSAC)':<32} {t_floor:>7.3f}s")
    print(f"  {'DBSCAN cluster':<32} {t_dbscan:>7.3f}s")
    print(f"  {'  PC total':<32} {t_pc_total:>7.3f}s")
    print(f"  {'ICP registration':<32} {t_icp:>7.3f}s")
    print(f"{'─'*44}")
    print(f"  {'Pipeline total':<32} {t_pipeline_total:>7.3f}s")
    print(f"{'─'*44}")

    if result is not None:
        print(f"\n{'─'*44}")
        print(f"  {'Estimated pose':<32}")
        print(f"{'─'*44}")
        print(f"  {'x':<32} {pallet_x:>8.4f} m")
        print(f"  {'y':<32} {pallet_y:>8.4f} m")
        print(f"  {'yaw':<32} {yaw_deg_normalized:>8.4f}°")
        print(f"{'─'*44}")

    if err_xy is not None:
        print(f"\n{'─'*44}")
        print(f"  {'Ground-truth error':<32}")
        print(f"{'─'*44}")
        print(f"  {'err_x':<32} {err_x:>+8.4f} m")
        print(f"  {'err_y':<32} {err_y:>+8.4f} m")
        print(f"  {'err_xy (Euclidean)':<32} {err_xy:>8.4f} m")
        print(f"  {'err_yaw':<32} {err_yaw:>8.4f}°")
        print(f"  {'fork_pocket_err':<32} {fork_pocket_err:>8.4f} m")
        print(f"{'─'*44}")

    csv_header = "yolo_s,depth_crop_s,pc_clean_s,floor_s,dbscan_s,icp_s,total_s"
    csv_row    = (f"{t_yolo:.3f},{t_depth_crop:.3f},{t_pc_clean:.3f},"
                  f"{t_floor:.3f},{t_dbscan:.3f},{t_icp:.3f},{t_pipeline_total:.3f}")
    if err_xy is not None:
        csv_header += ",err_x_m,err_y_m,err_xy_m,err_yaw_deg,fork_pocket_err_m"
        csv_row    += (f",{err_x:.4f},{err_y:.4f},{err_xy:.4f},"
                       f"{err_yaw:.4f},{fork_pocket_err:.4f}")
    print(f"\nCSV")
    print(csv_header)
    print(csv_row)
    print()

    metrics_path = Path(args.out_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # ── Step 7: Push pose to trajectory_viz simulation (RViz2) ───────────────
    if result is not None:
        print("\n" + "="*60)
        print("STEP 7 — Publishing pose to trajectory_viz (RViz2)")
        print("="*60)
        print(f"[yaw] raw={yaw_deg:.2f}°  normalized={yaw_deg_normalized:.2f}°")
        publish_to_viz.publish_pose(pallet_x, pallet_y, pallet_yaw_rad)
    else:
        print("\n[INFO] ICP returned no result — skipping RViz2 update.")

    # ── Step 8: Show ICP visualizer in background (non-blocking) ─────────────
    if viz_fn is not None:
        print("\n[INFO] Opening ICP visualizer in background (close window when done).")
        threading.Thread(target=viz_fn, daemon=False).start()


if __name__ == "__main__":
    main()