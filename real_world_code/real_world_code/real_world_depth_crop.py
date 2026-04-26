from pathlib import Path
import numpy as np
import cv2
import re
import time
from ultralytics import YOLO


def load_intrinsics_loose(json_path: str) -> dict:
    text = Path(json_path).read_text()
    text = re.sub(r'("model"\s*:\s*)([^"\n][^\n}]*)', r'\1"\2"', text)
    import json
    return json.loads(text)


def yolo_best_bbox_xyxy(model: YOLO, rgb_path: str, imgsz: int = 640, conf: float = 0.01, device: int = 0):
    results = model.predict(
        source=rgb_path,
        imgsz=imgsz,
        conf=conf,
        iou=0.7,
        max_det=50,
        device=device,
        verbose=False
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        return None

    confs = results.boxes.conf.detach().cpu().numpy()
    best_i = int(np.argmax(confs))

    x1, y1, x2, y2 = results.boxes.xyxy[best_i].detach().cpu().numpy().astype(int)
    best_conf = float(confs[best_i])
    cls_id = int(results.boxes.cls[best_i].detach().cpu().numpy())

    return (x1, y1, x2, y2, best_conf, cls_id)


def load_depth_bin(depth_bin_path: str, width: int, height: int, dtype=np.float32) -> np.ndarray:
    depth = np.fromfile(depth_bin_path, dtype=dtype)
    if depth.size != width * height:
        raise ValueError(f"Depth bin size mismatch: got {depth.size}, expected {width*height}")
    return depth.reshape((height, width))


def bbox_to_xyz(depth_m: np.ndarray, bbox_xyxy, fx, fy, cx, cy, z_min=0.0, z_max=10.0, stride=1):
    x1, y1, x2, y2 = bbox_xyxy

    x1 = max(0, min(depth_m.shape[1] - 1, x1))
    x2 = max(0, min(depth_m.shape[1] - 1, x2))
    y1 = max(0, min(depth_m.shape[0] - 1, y1))
    y2 = max(0, min(depth_m.shape[0] - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 3), dtype=np.float32)

    roi = depth_m[y1:y2:stride, x1:x2:stride]
    if roi.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    v_coords = np.arange(y1, y2, stride, dtype=np.float32)
    u_coords = np.arange(x1, x2, stride, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)

    Z = roi.astype(np.float32)
    valid = np.isfinite(Z) & (Z > z_min) & (Z < z_max)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    uu = uu[valid]
    vv = vv[valid]
    Z = Z[valid]

    X = (uu - cx) / fx * Z
    Y = (vv - cy) / fy * Z

    pts = np.stack([X, Y, Z], axis=1).astype(np.float32)
    return pts


def save_xyz(points_xyz: np.ndarray, out_xyz_path: str):
    Path(out_xyz_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_xyz_path, "w") as f:
        for x, y, z in points_xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def draw_bbox(image_path: str, bbox_xyxy, out_path: str, label: str = ""):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if label:
        cv2.putText(img, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, img)


def run(
    weights_path: str,
    rgb_path: str,
    depth_bin_path: str,
    intrinsics_json_path: str,
    out_xyz_path: str,
    out_debug_depth_vis_with_box: str | None = None,
    device: int = 0,
    imgsz: int = 640,
    conf: float = 0.01,
    z_min: float = 0.5,
    z_max: float = 4.0,
    stride: int = 2
):
    K = load_intrinsics_loose(intrinsics_json_path)
    width = int(K["width"])
    height = int(K["height"])
    fx = float(K["fx"])
    fy = float(K["fy"])
    cx = float(K["cx"])
    cy = float(K["cy"])

    t0 = time.perf_counter()
    model = YOLO(weights_path)
    best = yolo_best_bbox_xyxy(model, rgb_path, imgsz=imgsz, conf=conf, device=device)
    yolo_s = time.perf_counter() - t0
    if best is None:
        print("No detections on RGB. No .xyz created.")
        return None

    x1, y1, x2, y2, best_conf, cls_id = best
    bbox = (x1, y1, x2, y2)

    t1 = time.perf_counter()
    depth_m = load_depth_bin(depth_bin_path, width=width, height=height, dtype=np.float32)
    pts = bbox_to_xyz(depth_m, bbox, fx, fy, cx, cy, z_min=z_min, z_max=z_max, stride=stride)
    save_xyz(pts, out_xyz_path)
    depth_crop_s = time.perf_counter() - t1

    print("BBox xyxy:", bbox, "conf:", best_conf, "cls:", cls_id)
    print("Saved points:", pts.shape[0], "->", out_xyz_path)

    if out_debug_depth_vis_with_box is not None:
        draw_bbox(rgb_path, bbox, out_debug_depth_vis_with_box,
                  label=f"cls={cls_id} conf={best_conf:.3f}")

    return bbox, pts, yolo_s, depth_crop_s