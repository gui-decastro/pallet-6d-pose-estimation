# Pallet Pose Estimation Pipeline

End-to-end pipeline that takes a single RGB image + depth map, detects a pallet using YOLO, and estimates its 6-DoF pose (x, y, z, yaw) in the world frame via constrained ICP registration against a CAD mesh. The estimated pose is published to a RViz2 simulation that animates a forklift approach trajectory.

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
  publish_to_viz  →  /est_pallet_pose_in  →  RViz2 trajectory animation
```

---

## Setup

### Virtual environment

```bash
python3 -m venv ~/repositories/pallet-6d-pose-estimation/venv
source ~/repositories/pallet-6d-pose-estimation/venv/bin/activate

# PyTorch (CPU — GTX 1050 Ti not supported by recent CUDA wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Rest of dependencies
pip install matplotlib "numpy==2.2.4" "opencv-python==4.11.0.86" open3d ultralytics
```

### Make scripts executable (one-time)

```bash
chmod +x run_pipeline.sh
```

---

## Usage

All commands are run from the **repo root**.

### Option A — Existing captured files (testing)

```bash
./run_pipeline.sh \
  --rgb        /path/to/frame_rgb.png \
  --depth_bin  /path/to/frame_depth_meters.bin \
  --intrinsics /path/to/frame_intrinsics.json \
  --out_dir    /path/to/output_folder \
  --pick       farthest
```

RViz2 opens immediately. Once the pipeline finishes, the estimated pose is published and the forklift animation triggers automatically.

### Option B — Live camera (demo mode)

```bash
./run_pipeline.sh --live
```

RViz2 and a live camera preview window open. Press `Space` to capture a frame and run the full pipeline. Press `Q` to quit.

---

## Demo setup (with ground truth comparison)

Use two terminals.

**Terminal 1** — start everything:
```bash
./run_pipeline.sh --live
```

**Terminal 2** — once the pallet is in its known position, publish the ground truth pose:
```bash
source /opt/ros/jazzy/setup.bash
source ~/repositories/pallet-6d-pose-estimation/trajectory_viz/install/setup.bash
python3 real_world_code/real_world_code/set_ground_truth.py --x 2.0 --y 0.1 --yaw_deg 5.0
```

The ground truth pallet appears in RViz immediately. Then press `Space` in the camera preview to run the pipeline — the estimated pose publishes and the forklift trajectory animates.

---

## Configuration

Open `real_world_code/real_world_code/main.py` and adjust if needed:

```python
YOLO_DEVICE  = 'cpu'  # or GPU index if supported
YOLO_IMGSZ   = 640    # YOLO input image size
YOLO_CONF    = 0.01   # YOLO confidence threshold
Z_MIN        = 0.5    # Min depth to keep (metres)
Z_MAX        = 4.0    # Max depth to keep (metres)
DEPTH_STRIDE = 2      # Depth unprojection stride (1 = full res, 2 = half, …)
```

`WEIGHTS_PATH` is resolved automatically from `models/yolo.pt` relative to the script.

---

## Arguments

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

---

## Output files

Intermediate files go to `--out_dir`. In demo mode they go to `demo_output/pipeline_out/`.

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

**Ground truth pose not appearing in RViz**
Ensure the simulation node is running before calling `set_ground_truth.py`. The script waits up to 2 seconds for a subscriber — if it times out, the message is dropped.
