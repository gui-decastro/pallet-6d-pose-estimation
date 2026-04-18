"""
run_pipeline.py
===============
Glue script: launches the RealSense capture exe and auto-triggers the pose
estimation pipeline every time a new frame is captured.

Usage:
    python run_pipeline.py [--exe <path>] [--out_dir <path>] [--pick {farthest,largest}] [--no-viz]

Workflow:
    1. (Optional) Launches `ros2 launch trajectory_viz viz.launch.py` in the
       background so RViz2 is ready before the first frame arrives.
    2. Launches capture_d400e.exe, passing <out_dir>/live_capture as its output folder.
    3. Watches the session folder for new complete frame sets
       (rgb + depth_meters.bin + intrinsics.json all present).
    4. For each new frame, runs main.py with the correct arguments.
       main.py step 7 publishes the estimated pallet pose to /est_pallet_pose_in
       and triggers the forklift animation in RViz2.
    5. Continues until the capture exe exits (user presses ESC in the preview window).
"""

import argparse
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path


# ── User-editable defaults ────────────────────────────────────────────────────
DEFAULT_EXE = str(
    Path(__file__).parent
    / "camera_capture" / "d400e_cpp" / "build" / "Release" / "capture_d400e.exe"
)
DEFAULT_OUT_DIR = str(Path(__file__).parent / "live_capture")
PIPELINE_SCRIPT = str(
    Path(__file__).parent / "real_world_code" / "real_world_code" / "main.py"
)
# ─────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL   = 0.5   # seconds between directory scans
VIZ_WARMUP_SEC  = 4.0   # seconds to wait for RViz2 to finish loading


def ts() -> str:
    """Return a timestamp prefix like [14:32:01.456]."""
    return datetime.now().strftime("[%H:%M:%S.%f")[:-3] + "]"


def log(msg: str) -> None:
    print(f"{ts()} {msg}", flush=True)


def log_section(title: str, width: int = 64) -> None:
    bar = "─" * width
    print(f"\n{bar}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{bar}", flush=True)


def stream_output(proc: subprocess.Popen, prefix: str) -> None:
    """Forward a subprocess's stdout to our stdout in real time."""
    for line in proc.stdout:
        print(f"{ts()} [{prefix}] {line}", end="", flush=True)


def find_new_frames(session_dir: Path, seen: set) -> list[dict]:
    """
    Return a list of dicts for every complete frame not yet in `seen`.
    A frame is complete when all three files exist:
      frame_XXXX_rgb.png
      frame_XXXX_depth_meters.bin
      frame_XXXX_intrinsics.json
    """
    new_frames = []
    rgb_files = sorted(session_dir.glob("*_rgb.png"))
    for rgb in rgb_files:
        prefix = rgb.name.replace("_rgb.png", "")
        if prefix in seen:
            continue
        depth_bin  = session_dir / f"{prefix}_depth_meters.bin"
        intrinsics = session_dir / f"{prefix}_intrinsics.json"
        if depth_bin.exists() and intrinsics.exists():
            seen.add(prefix)
            new_frames.append({
                "prefix":     prefix,
                "rgb":        str(rgb),
                "depth_bin":  str(depth_bin),
                "intrinsics": str(intrinsics),
            })
    return new_frames


def run_pipeline(frame: dict, out_dir: Path, pick: str, frame_index: int) -> bool:
    """
    Run main.py for a single captured frame.

    Pipeline steps executed inside main.py:
      1. YOLO detection + depth unprojection  →  camera-frame XYZ
      2. Camera frame  →  world frame
      3. Point cloud cleaning (voxel + SOR + ROR + normals)
      4. Floor removal (RANSAC)
      5. DBSCAN largest-cluster extraction
      6. Constrained yaw-ICP registration  →  final 6-DOF pose
      7. Publish pose to /est_pallet_pose_in + trigger /animation/run_once

    Returns True on success, False on non-zero exit code.
    """
    frame_out = out_dir / frame["prefix"]
    frame_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        PIPELINE_SCRIPT,
        "--rgb",        frame["rgb"],
        "--depth_bin",  frame["depth_bin"],
        "--intrinsics", frame["intrinsics"],
        "--out_dir",    str(frame_out),
        "--pick",       pick,
    ]

    log_section(f"FRAME #{frame_index}  —  {frame['prefix']}")
    log(f"  RGB image   : {frame['rgb']}")
    log(f"  Depth bin   : {frame['depth_bin']}")
    log(f"  Intrinsics  : {frame['intrinsics']}")
    log(f"  Output dir  : {frame_out}")
    log(f"  RANSAC pick : {pick}")
    log(f"  Launching pipeline  "
        f"(7 steps: YOLO → cam2world → clean → floor → DBSCAN → ICP → RViz2)")
    print(flush=True)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        log(f"  Pipeline finished successfully in {elapsed:.1f}s")
    else:
        log(f"  [WARNING] Pipeline exited with code {result.returncode} "
            f"after {elapsed:.1f}s  —  frame {frame['prefix']}")

    return result.returncode == 0


def launch_viz(ros2_cmd: str) -> subprocess.Popen | None:
    """
    Launch `ros2 launch trajectory_viz viz.launch.py` as a background process.

    ros2_cmd is typically 'ros2' but can be overridden for custom ROS2 installs.
    Returns the Popen handle, or None if the launch fails (non-fatal).
    """
    cmd = [ros2_cmd, "launch", "trajectory_viz", "viz.launch.py"]
    log(f"  Launching viz: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        t = threading.Thread(target=stream_output, args=(proc, "viz"), daemon=True)
        t.start()
        log(f"  trajectory_viz started  (PID {proc.pid})")
        return proc
    except FileNotFoundError:
        log("  [WARNING] 'ros2' command not found — skipping RViz2 launch.")
        log("            Source ROS2 Jazzy and re-run, or pass --no-viz to suppress this.")
        return None
    except Exception as exc:
        log(f"  [WARNING] Could not launch trajectory_viz: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pallet 6-DOF pose estimation — capture + pipeline glue script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Pipeline steps (run per frame):\n"
            "  1. YOLO detection + depth unprojection  →  camera-frame XYZ\n"
            "  2. Camera frame  →  world frame\n"
            "  3. Point cloud cleaning (voxel + SOR + ROR + normals)\n"
            "  4. Floor removal (RANSAC)\n"
            "  5. DBSCAN largest-cluster extraction\n"
            "  6. Constrained yaw-ICP registration  →  final 6-DOF pose\n"
            "  7. Publish pose to trajectory_viz  →  RViz2 animation\n"
        ),
    )
    parser.add_argument(
        "--exe",
        default=DEFAULT_EXE,
        help=f"Path to capture_d400e.exe (default: {DEFAULT_EXE})",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help=f"Root output folder for live captures and pipeline results (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--pick",
        choices=["farthest", "largest"],
        default="farthest",
        help=(
            "Floor-plane selection heuristic for RANSAC "
            "(farthest: plane whose inliers are farthest from camera; "
            "largest: plane with most inliers). Default: farthest"
        ),
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        default=False,
        help="Skip launching trajectory_viz / RViz2 (useful when ROS2 is not available)",
    )
    parser.add_argument(
        "--ros2-cmd",
        default="ros2",
        help="ros2 executable name or full path (default: ros2)",
    )
    args = parser.parse_args()

    exe_path    = Path(args.exe)
    out_dir     = Path(args.out_dir)
    capture_dir = out_dir / "live_capture"

    # ── Startup banner ────────────────────────────────────────────────────────
    width = 64
    print("=" * width)
    print("  Pallet 6-DOF Pose Estimation — Real-World Pipeline")
    print("=" * width)
    log(f"  Session start   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Capture exe     : {exe_path}")
    log(f"  Capture folder  : {capture_dir}")
    log(f"  Output root     : {out_dir}")
    log(f"  RANSAC pick     : {args.pick}")
    log(f"  Visualization   : {'disabled (--no-viz)' if args.no_viz else 'enabled (trajectory_viz + RViz2)'}")
    log(f"  Poll interval   : {POLL_INTERVAL}s")
    print("=" * width)

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    log_section("Pre-flight checks")
    if not exe_path.exists():
        log(f"[ERROR] Capture exe not found: {exe_path}")
        log("        Build the C++ project in Visual Studio first.")
        sys.exit(1)
    log(f"  Capture exe     : OK  ({exe_path})")

    if not Path(PIPELINE_SCRIPT).exists():
        log(f"[ERROR] Pipeline script not found: {PIPELINE_SCRIPT}")
        sys.exit(1)
    log(f"  Pipeline script : OK  ({PIPELINE_SCRIPT})")

    capture_dir.mkdir(parents=True, exist_ok=True)
    pipeline_out = out_dir / "pipeline_results"
    pipeline_out.mkdir(parents=True, exist_ok=True)
    log(f"  Directories     : OK")
    log(f"  All checks passed.")

    # ── Launch trajectory_viz (RViz2) ─────────────────────────────────────────
    viz_proc = None
    if not args.no_viz:
        log_section("Starting trajectory_viz simulation (RViz2)")
        viz_proc = launch_viz(args.ros2_cmd)
        if viz_proc is not None:
            log(f"  Waiting {VIZ_WARMUP_SEC}s for RViz2 to finish loading...")
            time.sleep(VIZ_WARMUP_SEC)
            log("  RViz2 should be ready — starting capture.")
    else:
        log_section("Visualization skipped (--no-viz)")

    # ── Launch capture exe ────────────────────────────────────────────────────
    log_section("Starting RealSense capture")
    log(f"  Launching: {exe_path} {capture_dir}")
    log("  (Press ESC in the camera preview window to stop)")

    proc = subprocess.Popen(
        [str(exe_path), str(capture_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    log(f"  Capture process started  (PID {proc.pid})")

    t = threading.Thread(target=stream_output, args=(proc, "capture"), daemon=True)
    t.start()

    # ── Main watch loop ───────────────────────────────────────────────────────
    log_section("Watching for new frames")
    log(f"  Scanning: {capture_dir}")
    log(f"  Waiting for complete frame sets (rgb + depth_meters.bin + intrinsics.json)...")

    seen_frames: set = set()
    frame_count   = 0
    success_count = 0
    session_t0    = time.perf_counter()
    last_heartbeat = 0.0

    try:
        while proc.poll() is None:
            new_frames = find_new_frames(capture_dir, seen_frames)
            now = time.perf_counter()
            if not new_frames and (now - last_heartbeat) >= 10.0:
                log(f"  Watching... ({frame_count} frame(s) processed so far)")
                last_heartbeat = now
            for frame in new_frames:
                last_heartbeat = time.perf_counter()
                frame_count += 1
                ok = run_pipeline(frame, pipeline_out, args.pick, frame_count)
                if ok:
                    success_count += 1
            time.sleep(POLL_INTERVAL)

        log_section("Capture exe exited — final scan")
        log("  Running one last scan for frames saved just before exit...")
        new_frames = find_new_frames(capture_dir, seen_frames)
        if new_frames:
            log(f"  Found {len(new_frames)} additional frame(s) to process.")
        for frame in new_frames:
            frame_count += 1
            ok = run_pipeline(frame, pipeline_out, args.pick, frame_count)
            if ok:
                success_count += 1

    except KeyboardInterrupt:
        log("\n[INFO] KeyboardInterrupt — terminating capture exe...")
        proc.terminate()
        proc.wait()
        log("  Capture process terminated.")

    # ── Shut down viz ─────────────────────────────────────────────────────────
    if viz_proc is not None and viz_proc.poll() is None:
        log_section("Stopping trajectory_viz")
        log("  Terminating RViz2 process...")
        viz_proc.terminate()
        try:
            viz_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            viz_proc.kill()
        log("  trajectory_viz stopped.")

    # ── Session summary ───────────────────────────────────────────────────────
    session_elapsed = time.perf_counter() - session_t0
    failed = frame_count - success_count

    print("\n" + "=" * width)
    print("  SESSION SUMMARY")
    print("=" * width)
    log(f"  Total frames processed : {frame_count}")
    log(f"  Successful             : {success_count}")
    log(f"  Failed / aborted       : {failed}")
    log(f"  Total session time     : {session_elapsed:.1f}s")
    log(f"  Results saved to       : {pipeline_out}")
    print("=" * width)


if __name__ == "__main__":
    main()
