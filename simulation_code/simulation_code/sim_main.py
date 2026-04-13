"""
Pallet pose estimation pipeline
================================
Steps:
  1. sim_depth_crop  — YOLO detection → EXR depth crop → camera-frame .xyz
  2. cam_to_world    — camera frame → world frame (4x4 matrix from .txt file)
  3. floor_remove    — RANSAC iterative plane removal
  4. register        — constrained yaw-ICP registration

Usage:
  python main.py --rgb <path> --depth_exr <path> --cam_to_world_txt <path> --out_dir <path> [--pick {farthest,largest}]
"""

import argparse
from pathlib import Path

# ── Pipeline imports ──────────────────────────────────────────────────────────
import sim_depth_crop
import sim_cam_to_world
import floor_remove
import register_new_constraint

# ── User-editable config ──────────────────────────────────────────────────────
WEIGHTS_PATH = "/home/sarthak_m/Capstone/runs/train_phase_1_3_4/weights/best.pt"
MESH_PATH    = "/home/sarthak_m/Capstone/Code/simulation_code/pallet_mesh/Pallet_world_dim_transforms.ply"

YOLO_DEVICE  = 0
YOLO_IMGSZ   = 640
YOLO_CONF    = 0.01
DEPTH_STRIDE = 2
# ─────────────────────────────────────────────────────────────────────────────


def build_paths(out_dir: str) -> dict:
    o = out_dir
    return {
        "crop_cam":   f"{o}/crop.xyz",
        "crop_debug": f"{o}/crop_rgb_box.png",
        "crop_world": f"{o}/crop_world.xyz",
        "no_floor":   f"{o}/crop_world_floor_removed.xyz",
    }


def main():
    parser = argparse.ArgumentParser(description="Pallet pose estimation pipeline")
    parser.add_argument("--rgb",              required=True, help="Path to RGB image (.png)")
    parser.add_argument("--depth_exr",        required=True, help="Path to EXR depth file (.exr)")
    parser.add_argument("--cam_to_world_txt", required=True, help="Path to .txt file with 4x4 cam-to-world matrix")
    parser.add_argument("--out_dir",          required=True, help="Output folder for all intermediate and final files")
    parser.add_argument("--pick", choices=["farthest", "largest"], default="farthest",
                        help="Floor-plane selection heuristic (default: farthest)")
    args = parser.parse_args()

    p = build_paths(args.out_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Step 1: YOLO detection + EXR depth unprojection ──────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Depth crop (YOLO + EXR unprojection)")
    print("="*60)
    result = sim_depth_crop.run(
        weights_path=WEIGHTS_PATH,
        rgb_path=args.rgb,
        depth_exr_path=args.depth_exr,
        out_xyz_path=p["crop_cam"],
        out_debug_bbox_img=p["crop_debug"],
        device=YOLO_DEVICE,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        stride=DEPTH_STRIDE,
    )
    if result is None:
        print("Pipeline aborted: no YOLO detection.")
        return

    # ── Step 2: Camera → world transform ─────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — Camera frame → world frame")
    print("="*60)
    sim_cam_to_world.run(
        in_path=p["crop_cam"],
        out_path=p["crop_world"],
        cam_to_world_txt=args.cam_to_world_txt,
    )

    # ── Step 3: Floor removal ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"STEP 3 — Floor removal (RANSAC, pick='{args.pick}')")
    print("="*60)
    floor_remove.run(
        in_path=p["crop_world"],
        out_path=p["no_floor"],
        pick=args.pick,
    )

    # ── Step 4: Constrained yaw-ICP registration ─────────────────────────────
    print("\n" + "="*60)
    print("STEP 4 — Constrained yaw-ICP registration")
    print("="*60)
    register_new_constraint.run(
        mesh_path=MESH_PATH,
        world_cloud_xyz=p["no_floor"],
    )

    print("\n" + "="*60)
    print("Pipeline complete.")
    print("="*60)


if __name__ == "__main__":
    main()