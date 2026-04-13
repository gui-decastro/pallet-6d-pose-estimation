# Pallet Pose Estimation Pipeline

End-to-end pipeline that takes a single RGB image + depth map, detects a pallet using YOLO, and estimates its 6-DoF pose (x, y, z, yaw) in the world frame via constrained ICP registration against a CAD mesh.

---

## Pipeline Overview

```
RGB + Depth (.bin) + Intrinsics (.json)
           │
           ▼
  1. real_world_depth_crop       YOLO detection → unproject depth ROI → camera-frame .xyz
           │
           ▼
  2. cam_to_world                Camera frame → world frame (axis flip + tilt + yaw + translation)
           │
           ▼
  3. real_world_depth_clean      Voxel downsample → SOR → ROR → normal estimation
           │
           ▼
  4. floor_remove                RANSAC plane segmentation → floor plane removal
           │
           ▼
  5. clean_before_icp            DBSCAN → keep largest cluster (the pallet)
           │
           ▼
  6. register_new_constraint_init  PCA yaw init → constrained yaw-ICP → pallet pose
           │
           ▼
  T_world_mesh_chosen_yaw_xyz.txt     (4×4 transform)
  pallet_mesh_in_world_chosen_yaw_xyz.ply  (transformed mesh)
```

---

## File Structure

All scripts must be in the same directory:

```
pipeline/
├── main.py
├── real_world_depth_crop.py
├── cam_to_world.py
├── real_world_depth_clean.py
├── floor_remove.py
├── clean_before_icp.py
├── register_new_constraint_init.py
└── Pallet_world_dim_transforms.ply     ← CAD mesh (required by step 6)
```

---

## Dependencies

```bash
pip install numpy opencv-python ultralytics open3d
```

A CUDA-capable GPU is recommended for YOLO inference. CPU fallback works but is slower.

---

## Configuration

Before running, open `main.py` and update the constants at the top:

```python
WEIGHTS_PATH = "/path/to/your/yolo/best.pt"   # YOLO model weights

YOLO_DEVICE  = 0      # GPU device index (0, 1, …) or 'cpu'
YOLO_IMGSZ   = 640    # YOLO input image size
YOLO_CONF    = 0.01   # YOLO confidence threshold
Z_MIN        = 0.5    # Min depth to keep (metres)
Z_MAX        = 4.0    # Max depth to keep (metres)
DEPTH_STRIDE = 2      # Depth unprojection stride (1 = full res, 2 = half, …)
```

Also ensure `Pallet_world_dim_transforms.ply` is present in the same directory as `main.py`, or update `MESH_PATH` in `register_new_constraint_init.py`.

---

## Usage

```bash
python main.py \
  --rgb        /path/to/rgb.png \
  --depth_bin  /path/to/depth_meters.bin \
  --intrinsics /path/to/intrinsics.json \
  --out_dir    /path/to/output_folder \
  --pick       farthest
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--rgb` | ✅ | Path to the RGB image (`.png`) |
| `--depth_bin` | ✅ | Path to the raw depth file (`.bin`, float32, row-major) |
| `--intrinsics` | ✅ | Path to the camera intrinsics JSON |
| `--out_dir` | ✅ | Output folder — created automatically if it does not exist |
| `--pick` | ❌ | Floor plane selection: `farthest` (default) or `largest` |

### `--pick` explained

- `farthest` — removes the plane whose inliers are on average **farthest from the camera origin**. Recommended when the floor is at a different depth than the pallet top surface.
- `largest` — removes the plane with the **most inlier points**. Use this if `farthest` accidentally removes the pallet top instead of the floor.

---

## Intrinsics JSON format

```json
{
  "model": "pinhole",
  "width": 1280,
  "height": 720,
  "fx": 640.0,
  "fy": 640.0,
  "cx": 640.0,
  "cy": 360.0
}
```

The `model` value does not need to be quoted in the file — the parser handles this automatically.

---

## Output files

All intermediate files are written to `--out_dir`. Final registration outputs are written to the **current working directory** (wherever you run `main.py` from).

| File | Description |
|---|---|
| `<out_dir>/crop.xyz` | Pallet depth crop in camera frame |
| `<out_dir>/crop_rgb_box.png` | Debug image — YOLO bounding box overlaid on RGB |
| `<out_dir>/crop_world.xyz` | Depth crop transformed to world frame |
| `<out_dir>/crop_world_cleaned.xyz` | After voxel + SOR + ROR cleaning |
| `<out_dir>/crop_world_cleaned_floor_removed.xyz` | After floor plane removal |
| `<out_dir>/crop_world_cleaned_floor_removed_cleaned.xyz` | Final ICP input (largest cluster) |
| `T_world_mesh_chosen_yaw_xyz.txt` | 4×4 world-to-mesh transform matrix |
| `pallet_mesh_in_world_chosen_yaw_xyz.ply` | CAD mesh transformed into world frame |

The final pallet pose (x, y, z, yaw) is printed to stdout at the end of the run.

---

## Example

```bash
python main.py \
  --rgb        /data/session_001/frame_0023_rgb.png \
  --depth_bin  /data/session_001/frame_0023_depth_meters.bin \
  --intrinsics /data/session_001/frame_0023_intrinsics.json \
  --out_dir    /data/session_001/output \
  --pick       farthest
```

Expected terminal output (abbreviated):

```
============================================================
STEP 1 — Depth crop (YOLO + unprojection)
============================================================
BBox xyxy: (312, 204, 891, 617)  conf: 0.847  cls: 0
Saved points: 48320 -> /data/session_001/output/crop.xyz

============================================================
STEP 2 — Camera frame → world frame
============================================================
[transform] camera frame → world frame
[save]      48320 points saved to .../crop_world.xyz

...

=== PALLET POSE IN WORLD FRAME ===
  x   : 0.1234 m
  y   : -0.0512 m
  z   : 0.0021 m
  yaw : -12.4871°  (rotation about Z axis)
```

---

## Troubleshooting

**No YOLO detection — pipeline aborts at Step 1**
Lower `YOLO_CONF` in `main.py` (e.g. `0.001`) or verify the weights path is correct.

**Too few points after floor removal**
Switch `--pick largest` to `--pick farthest` or vice versa. If still poor, increase `th` (RANSAC distance threshold) in `floor_remove.py`.

**DBSCAN finds no clusters at Step 5**
Increase `DBSCAN_EPS` in `clean_before_icp.py` (e.g. `0.03` → `0.05`).

**ICP fails — all trials report low fitness**
Increase `ICP_DIST` or `YAW_SCORE_DIST` in `register_new_constraint_init.py`. Also verify the camera-to-world transform parameters in `cam_to_world.py` match your physical setup.

**Open3D visualizer windows block the pipeline**
Set `VISUALIZE_BEST = False` in `register_new_constraint_init.py` to skip the interactive 3D viewer.