from pathlib import Path
import numpy as np
import cv2
import OpenEXR
import Imath
from ultralytics import YOLO


# ── Fixed intrinsics from Blender ─────────────────────────────────────────────
FX = 645.3333
FY = 806.6666
CX = 640.0
CY = 400.0


def yolo_best_bbox_xyxy(model: YOLO, rgb_path: str, imgsz: int = 640, conf: float = 0.01, device: int = 0):
    """
    Returns best bbox (x1,y1,x2,y2) in ORIGINAL image pixel coordinates.
    """
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

    confs     = results.boxes.conf.detach().cpu().numpy()
    best_i    = int(np.argmax(confs))
    x1, y1, x2, y2 = results.boxes.xyxy[best_i].detach().cpu().numpy().astype(int)
    best_conf = float(confs[best_i])
    cls_id    = int(results.boxes.cls[best_i].detach().cpu().numpy())
    return (x1, y1, x2, y2, best_conf, cls_id)


def load_depth_exr(depth_exr_path: str) -> np.ndarray:
    """
    Loads depth from an EXR file using OpenEXR (Blender-exported).
    Reads the 'Depth.V' channel and returns a float32 HxW array.
    """
    exr   = OpenEXR.InputFile(depth_exr_path)
    dw    = exr.header()['dataWindow']
    w     = dw.max.x - dw.min.x + 1
    h     = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    depth = np.frombuffer(exr.channel('Depth.V', FLOAT), dtype=np.float32).reshape((h, w))
    return depth


def bbox_to_xyz(depth: np.ndarray, bbox_xyxy, stride: int = 1) -> np.ndarray:
    """
    Unprojects depth ROI to 3D points in Blender camera frame convention:
      X =  (u - cx) / fx * Z
      Y = -(v - cy) / fy * Z
      Z = -depth(u, v)
    """
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(depth.shape[1] - 1, x1))
    x2 = max(0, min(depth.shape[1] - 1, x2))
    y1 = max(0, min(depth.shape[0] - 1, y1))
    y2 = max(0, min(depth.shape[0] - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 3), dtype=np.float32)

    roi = depth[y1:y2:stride, x1:x2:stride]
    if roi.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    v_coords = np.arange(y1, y2, stride, dtype=np.float32)
    u_coords = np.arange(x1, x2, stride, dtype=np.float32)
    uu, vv   = np.meshgrid(u_coords, v_coords)

    Z     = roi.astype(np.float32)
    valid = (Z > 0) & np.isfinite(Z)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    uu, vv, Z = uu[valid], vv[valid], Z[valid]
    X  =  (uu - CX) / FX * Z
    Y  = -(vv - CY) / FY * Z
    Zc = -Z

    return np.stack([X, Y, Zc], axis=1).astype(np.float32)


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
    depth_exr_path: str,
    out_xyz_path: str,
    out_debug_bbox_img: str | None = None,
    device: int = 0,
    imgsz: int = 640,
    conf: float = 0.01,
    stride: int = 1
):
    model = YOLO(weights_path)
    best  = yolo_best_bbox_xyxy(model, rgb_path, imgsz=imgsz, conf=conf, device=device)
    if best is None:
        print("No detections on RGB. No .xyz created.")
        return None

    x1, y1, x2, y2, best_conf, cls_id = best
    bbox = (x1, y1, x2, y2)
    print(f"BBox xyxy: {bbox}  conf: {best_conf:.3f}  cls: {cls_id}")

    depth = load_depth_exr(depth_exr_path)
    pts   = bbox_to_xyz(depth, bbox, stride=stride)

    Path(out_xyz_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_xyz_path, pts)
    print(f"Saved {len(pts):,} points -> {out_xyz_path}")

    if out_debug_bbox_img is not None:
        draw_bbox(rgb_path, bbox, out_debug_bbox_img, label=f"cls={cls_id} conf={best_conf:.3f}")

    return bbox, pts