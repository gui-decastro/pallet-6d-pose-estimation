# Building D400e C++ Capture Program

## Prerequisites

1. **Visual Studio 2026** - Already installed ✓
2. **CMake** - Already installed ✓
3. **OpenCV for Windows**
   - Download from: https://opencv.org/releases/
   - Get Windows version (e.g., opencv-4.10.0-windows.exe)
   - Extract to: `C:\opencv`
   - Add to PATH: `C:\opencv\build\x64\vc16\bin`

4. **FRAMOS SDK** - Already installed ✓

---

## Build Steps

### 1. Setup Project

```bash
cd "/c/Users/guibc/Robotics Projects/Pallet 6D Pose Estimation"

# Create C++ project directory
mkdir d400e_cpp
cd d400e_cpp

# Copy files here:
# - capture_d400e.cpp
# - CMakeLists.txt
```

### 2. Build with CMake

```bash
# Create build directory
mkdir build
cd build

# Configure CMake (set OpenCV path)
cmake .. -G "Visual Studio 18 2026" -A x64 \
    -DOpenCV_DIR="C:/opencv/build"

# Build
cmake --build . --config Release

# Executable will be in: build/Release/capture_d400e.exe
```

### 3. Run

```bash
cd Release
./capture_d400e.exe

# Controls:
# - Press 's' to save current frame
# - Press 'q' to quit
```

---

## If OpenCV Not Installed

**Quick Install:**
1. Download: https://opencv.org/releases/
2. Run installer (self-extracting)
3. Extract to `C:\opencv`
4. Done!

---

## Alternative: Visual Studio Solution

If you prefer Visual Studio GUI:

1. **Open Visual Studio 2026**
2. **Create New Project** → Empty C++ Project
3. **Add** `capture_d400e.cpp`
4. **Project Properties:**
   - C/C++ → Additional Include Directories:
     - `C:\Program Files\FRAMOS\librealsense2\include`
     - `C:\opencv\build\include`
   - Linker → Additional Library Directories:
     - `C:\Program Files\FRAMOS\librealsense2\lib`
     - `C:\opencv\build\x64\vc16\lib`
   - Linker → Input → Additional Dependencies:
     - `realsense2.lib`
     - `opencv_world4100.lib` (or your version)
5. **Build** → **Run**

---

## Comparison: Python vs C++

| Setup Task | Python | C++ |
|------------|--------|-----|
| Build pyrealsense2 | 3+ hours | Not needed! |
| Install libraries | Complex | Just link |
| First capture | After long build | 30 minutes |
| Performance | Good | Better |
| Live streaming | After build | Works immediately |

**C++ is definitely easier for D400e!**
