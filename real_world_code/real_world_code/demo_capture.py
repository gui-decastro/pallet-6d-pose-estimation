"""
demo_capture.py
===============
Live camera preview for the pallet pose estimation demo.
Captures a frame on SPACE, saves it to disk, then runs the full pipeline
(main.py) exactly as if you had passed --rgb/--depth_bin/--intrinsics by hand.

Run via:
    bash run_pipeline.sh --live

Controls:
    SPACE  — capture frame + run pipeline
    Q      — quit
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# ── Config ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT, FPS   = 1280, 720, 15
DEPTH_CLAMP_MIN      = 0.8   # metres  (matches C++ capture)
DEPTH_CLAMP_MAX      = 6.0

REPO_DIR  = Path(__file__).resolve().parent.parent.parent
OUT_DIR   = REPO_DIR / "demo_output"
WINDOW    = "Pallet Pipeline  |  SPACE = capture & run  |  Q = quit"
# ─────────────────────────────────────────────────────────────────────────────


def save_intrinsics(intr, path: Path):
    path.write_text(json.dumps({
        "model":  "pinhole",
        "width":  intr.width,
        "height": intr.height,
        "fx":     intr.fx,
        "fy":     intr.fy,
        "cx":     intr.ppx,
        "cy":     intr.ppy,
    }, indent=2))


def overlay_text(img, text, colour=(0, 255, 0)):
    """Draw a filled banner at the bottom of the frame."""
    h, w = img.shape[:2]
    out  = img.copy()
    cv2.rectangle(out, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(out, text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    return out


def run_pipeline(rgb_path, depth_path, intr_path):
    main_py = Path(__file__).parent / "main.py"
    cmd = [
        sys.executable, str(main_py),
        "--rgb",        str(rgb_path),
        "--depth_bin",  str(depth_path),
        "--intrinsics", str(intr_path),
        "--out_dir",    str(OUT_DIR / "pipeline_out"),
    ]
    subprocess.run(cmd, check=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)
    profile  = pipeline.start(config)

    depth_scale = (profile.get_device()
                          .first_depth_sensor()
                          .get_depth_scale())

    align = rs.align(rs.stream.color)

    spatial  = rs.spatial_filter()
    temporal = rs.temporal_filter()

    print(f"Camera ready. Output → {OUT_DIR}")

    try:
        while True:
            frames      = pipeline.wait_for_frames()
            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            display   = overlay_text(color_img, "SPACE = capture  |  Q = quit")
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord(' '):
                # Show feedback immediately
                cv2.imshow(WINDOW, overlay_text(color_img, "Capturing...", (0, 200, 255)))
                cv2.waitKey(1)

                # Apply filters + depth clamp
                depth_frame = spatial.process(depth_frame)
                depth_frame = temporal.process(depth_frame)
                depth_m     = (np.asanyarray(depth_frame.get_data()).astype(np.float32)
                               * depth_scale)
                depth_m[(depth_m < DEPTH_CLAMP_MIN) | (depth_m > DEPTH_CLAMP_MAX)] = 0.0

                # Save files
                rgb_path   = OUT_DIR / "capture_rgb.png"
                depth_path = OUT_DIR / "capture_depth_meters.bin"
                intr_path  = OUT_DIR / "capture_intrinsics.json"

                cv2.imwrite(str(rgb_path), color_img)
                depth_m.tofile(str(depth_path))
                save_intrinsics(
                    color_frame.profile.as_video_stream_profile().intrinsics,
                    intr_path)
                print(f"Frame saved → {OUT_DIR}")

                # Show "running" banner and hand off to pipeline
                cv2.imshow(WINDOW, overlay_text(color_img,
                           "Pipeline running — watch terminal...", (0, 200, 255)))
                cv2.waitKey(1)

                run_pipeline(rgb_path, depth_path, intr_path)

                cv2.imshow(WINDOW, overlay_text(color_img,
                           "Done! Check RViz. SPACE for another capture."))
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
