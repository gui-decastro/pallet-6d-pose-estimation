# C++ Capture Setup Guide — FRAMOS D400e on Linux (Ubuntu 22.04)

---

## Prerequisites

- Ubuntu 22.04
- FRAMOS D400e camera with PoE connection
- Network adapter connected to the camera's subnet (192.168.0.x)

---

## Step 1 — Install the FRAMOS SDK

Download the FRAMOS D400e Software Package for Linux from the FRAMOS customer portal. It includes a FRAMOS-specific fork of librealsense2 (v2.55.10) with GigE Vision / PoE transport support. **Do not use the standard Intel librealsense2 — it does not support PoE cameras.**

The package ships as `.deb` files. If you encounter a permissions error during install:

```bash
sudo chmod 644 *.deb
sudo apt install ./*.deb
```

This installs:
- `librealsense2` (FRAMOS fork) — headers + shared libraries
- `librealsense2-dev` — CMake config (`realsense2Config.cmake`) needed for `find_package`

---

## Step 2 — Install dependencies

```bash
sudo apt install build-essential cmake libopencv-dev inotify-tools
```

---

## Step 3 — Set up network routing

The D400e connects over PoE at `192.168.0.200`. If you have multiple interfaces on the same subnet, add a static host route so traffic reaches the camera through the correct NIC.

1. Open NetworkManager connection editor:
   ```bash
   nm-connection-editor
   ```
2. Edit the connection for the camera's NIC (e.g. `enx000ec662e40e`)
3. Go to **IPv4 Settings → Routes → Add**:
   - Address: `192.168.0.200`
   - Netmask: `255.255.255.255`
   - Gateway: `0.0.0.0`
4. Save and reconnect

Verify connectivity:
```bash
ping 192.168.0.200
```

---

## Step 4 — Build the capture binary

```bash
cd ~/repositories/pallet-6d-pose-estimation/camera_capture/d400e_cpp
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
```

`CMakeLists.txt` uses `find_package(realsense2 REQUIRED)` and `find_package(OpenCV REQUIRED)` — no hardcoded paths needed on Linux.

**Expected output:**
```
-- Found realsense2: ...
-- Found OpenCV: ... (found version "4.x.x")
[100%] Built target capture_d400e
```

Binary is at: `camera_capture/d400e_cpp/build/capture_d400e`

---

## Step 5 — Run the capture binary

The binary is called automatically by `run_pipeline.sh --live`. To run it standalone:

```bash
~/repositories/pallet-6d-pose-estimation/camera_capture/d400e_cpp/build/capture_d400e
```

**Controls:**
- `SPACE` — capture frame (saves 5 files for the pipeline)
- `ESC` — exit

**Output per capture** (saved to `camera_capture/collected_data/session_<timestamp>/`):
```
frame_0000_rgb.png              — 1280×720 color image
frame_0000_depth.png            — 16-bit raw depth
frame_0000_depth_meters.bin     — 32-bit float depth in meters
frame_0000_depth_vis.png        — colormap visualization
frame_0000_intrinsics.json      — camera parameters
```

---

## Troubleshooting

**Camera not detected**
```bash
ping 192.168.0.200
```
If this fails but `ping -I enx000ec662e40e 192.168.0.200` works, the static route is not set — repeat Step 3.

**`find_package(realsense2)` fails during cmake**
The FRAMOS dev package is not installed. Re-run Step 1 and verify:
```bash
dpkg -l | grep realsense
```

**Permission error installing `.deb` files**
```bash
sudo chmod 644 *.deb
sudo apt install ./*.deb
```

**Pipeline does not trigger after pressing SPACE**
Check `inotify-tools` is installed:
```bash
sudo apt install inotify-tools
```
