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
from pathlib import Path

# ── Pipeline imports ──────────────────────────────────────────────────────────
import real_world_depth_crop
import cam_to_world
import real_world_depth_clean
import floor_remove
import clean_before_icp
import register_new_constraint_init

# ── User-editable config ──────────────────────────────────────────────────────
WEIGHTS_PATH = "/home/sarthak_m/Capstone/runs/train_phase_1_3_4/weights/best.pt"

YOLO_DEVICE  = 0
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
    real_world_depth_clean.run(
        in_path=p["crop_world"],
        out_path=p["cleaned"],
    )

    # ── Step 4: Floor removal ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"STEP 4 — Floor removal (RANSAC, pick='{args.pick}')")
    print("="*60)
    floor_remove.run(
        in_path=p["cleaned"],
        out_path=p["no_floor"],
        pick=args.pick,
    )

    # ── Step 5: DBSCAN — keep largest cluster ────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5 — DBSCAN largest cluster (pre-ICP clean)")
    print("="*60)
    clean_before_icp.run(
        in_path=p["no_floor"],
        out_path=p["icp_input"],
    )

    # ── Step 6: Constrained yaw-ICP registration ─────────────────────────────
    print("\n" + "="*60)
    print("STEP 6 — Constrained yaw-ICP registration")
    print("="*60)
    register_new_constraint_init.run(
        world_cloud_xyz=p["icp_input"],
    )

    print("\n" + "="*60)
    print("Pipeline complete.")
    print("="*60)


if __name__ == "__main__":
    main()