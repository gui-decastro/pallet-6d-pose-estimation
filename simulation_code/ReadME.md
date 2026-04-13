# Pallet Pose Estimation Pipeline

End-to-end pipeline for estimating pallet pose from a Blender-rendered RGB image and EXR depth map using YOLO detection, point cloud processing, and constrained ICP registration.

---

## Pipeline Overview

```
RGB + EXR Depth
      │
      ▼
┌─────────────────┐
│  sim_depth_crop │  YOLO detection → depth unprojection → camera-frame .xyz
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  cam_to_world   │  4×4 Blender matrix → world-frame .xyz
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  floor_remove   │  Iterative RANSAC plane removal
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    register     │  Constrained yaw-ICP mesh registration
└────────┬────────┘
         │
         ▼
  T_world_mesh.txt + aligned mesh .ply
```

---

## File Structure

```
.
├── sim_main.py                   # Orchestrator — runs the full pipeline
├── sim_depth_crop.py             # Step 1: YOLO + EXR depth unprojection
├── cam_to_world.py               # Step 2: Camera → world frame transform
├── floor_remove.py               # Step 3: RANSAC floor plane removal
├── register_new_constraint.py    # Step 4: Constrained yaw-ICP registration
└── README.md
```

---

## Requirements

```
Python >= 3.10
torch
ultralytics
opencv-python
OpenEXR
Imath
open3d
numpy
```

Install dependencies:
```bash
pip install ultralytics opencv-python openexr imath open3d numpy
```

> **Note:** `OpenEXR` may require system-level libraries. On Ubuntu:
> ```bash
> sudo apt-get install libopenexr-dev
> ```

---

## Usage

```bash
python sim_main.py \
  --rgb             /path/to/frame_rgb.png \
  --depth_exr       /path/to/frame_depth.exr \
  --cam_to_world_txt /path/to/T_world_cam.txt \
  --out_dir         /path/to/output_folder \
  --pick            farthest
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--rgb` | ✅ | Path to RGB image (`.png`) |
| `--depth_exr` | ✅ | Path to Blender-exported EXR depth file |
| `--cam_to_world_txt` | ✅ | Path to `.txt` file containing the 4×4 cam-to-world matrix |
| `--out_dir` | ✅ | Output directory for all intermediate and final files |
| `--pick` | ❌ | Floor plane selection heuristic: `farthest` (default) or `largest` |

---

## Configuration

Edit the config section at the top of `main.py` to set paths and parameters:

```python
WEIGHTS_PATH = "/path/to/yolo/weights/best.pt"
MESH_PATH    = "/path/to/Pallet_world_dim_transforms.ply"

YOLO_DEVICE  = 0      # GPU device index
YOLO_IMGSZ   = 640    # YOLO input image size
YOLO_CONF    = 0.01   # YOLO confidence threshold
DEPTH_STRIDE = 2      # Depth unprojection stride (higher = fewer points, faster)
```

Per-module config constants (e.g. ICP iterations, RANSAC thresholds) are at the top of each respective module file.

---

## Camera Intrinsics

Fixed Blender intrinsics are hardcoded in `sim_depth_crop.py`:

```python
FX = 645.3333
FY = 806.6666
CX = 640.0
CY = 400.0
```

Update these if your Blender camera settings change.

---

## Cam-to-World Matrix Format

The `--cam_to_world_txt` file should contain a row-major 4×4 homogeneous transform matrix. Lines starting with `#` are treated as comments and ignored:

```
# T_world_cam: camera-to-world transform for 'Camera'
# Row-major 4x4 matrix:
-0.00000059  0.74314487  -0.66913056  9.81021786
-1.00000000  -0.00000170  -0.00000101  3.24600267
-0.00000189  0.66913056   0.74314487  1.50000000
 0.00000000  0.00000000   0.00000000  1.00000000
```

This matrix changes when the camera position in Blender changes. Export a new one for each camera pose.

---

## Output Files

All intermediate and final files are saved to `--out_dir`:

| File | Description |
|---|---|
| `crop.xyz` | Point cloud in camera frame (Step 1 output) |
| `crop_rgb_box.png` | Debug image with YOLO bounding box overlaid |
| `crop_world.xyz` | Point cloud in world frame (Step 2 output) |
| `crop_world_floor_removed.xyz` | Point cloud with floor plane removed (Step 3 output) |
| `T_world_mesh_chosen_yaw_xyz.txt` | Final 4×4 pose transform (Step 4 output) |
| `pallet_mesh_in_world_chosen_yaw_xyz.ply` | Mesh aligned to point cloud in world frame |

---

## Floor Removal — `--pick` Heuristic

The `floor_remove` step runs RANSAC up to 3 times, finding one plane per iteration. The `--pick` argument controls which plane is selected as the floor:

- **`farthest`** *(default)* — selects the plane whose inliers are on average farthest from the origin. Recommended when the camera is elevated and looking down at the pallet.
- **`largest`** — selects the plane with the most inliers. Can misfire if the pallet top is larger than the visible floor area.

---

## ICP Registration Notes

- Registration solves for **yaw (rotation about Z) + translation (x, y, z)** only. Roll and pitch are fixed at 0°.
- A coarse yaw search (5° steps over 360°) initializes the ICP to avoid local minima.
- After finding the best yaw, the pipeline also evaluates `yaw ± 180°` and picks the one with the smaller absolute yaw value (configurable via `CHOOSE_MIN_ABS_YAW` in `register.py`).
- Multiple trials (`NUM_TRIALS = 10`) are run with different random seeds; the best result by fitness then RMSE is kept.
