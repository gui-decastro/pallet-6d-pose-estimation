# Complete C++ Setup Guide for FRAMOS D400e Camera
## Live Real-Time Capture on Windows

---

## 📋 Prerequisites Checklist

Before starting, you need:

- [x] Windows 10/11
- [x] FRAMOS D400e Software Package installed
  - Location: `C:\Program Files\FRAMOS\`
  - Includes: librealsense2, CameraSuite
- [x] Visual Studio 2026 Community with C++ tools
  - Workload: "Desktop development with C++"
- [x] CMake 4.x
- [x] OpenCV for Windows
- [x] GitBash terminal (optional but helpful)

---

## 🚀 Complete Setup Process

### Step 1: Install Visual Studio C++ Tools

**Check if already installed:**
```bash
ls "/c/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC"
```

**If not installed:**
1. Open Visual Studio Installer:
   ```bash
   "/c/Program Files (x86)/Microsoft Visual Studio/Installer/vs_installer.exe" &
   ```
2. Click **Modify** on Visual Studio 2026 Community
3. Check: ☑ **Desktop development with C++**
4. Click **Modify** and wait for installation (~15 minutes)

---

### Step 2: Install OpenCV

**Download:**
1. Go to: https://opencv.org/releases/
2. Download Windows version (e.g., `opencv-4.12.0-windows.exe`)

**Install:**
1. Run the downloaded `.exe` file
2. When asked "Extract to:", enter: `C:\`
3. Click **Extract**
4. Creates: `C:\opencv\`

**Verify:**
```bash
ls "/c/opencv/build"
# Should show: bin, include, x64, etc.
```

**Find OpenCV config location:**
```bash
find "/c/opencv" -name "OpenCVConfig.cmake" 2>/dev/null
# Result: /c/opencv/build/x64/vc16/lib/OpenCVConfig.cmake
```

---

### Step 3: Create Project Directory

```bash
cd "/c/Users/guibc/Robotics Projects/Pallet 6D Pose Estimation"

# Create C++ project directory
mkdir d400e_cpp
cd d400e_cpp

# Create build directory
mkdir build
```

---

### Step 4: Create Source Files

Create these 3 files in `d400e_cpp/` directory:

#### File 1: `capture_d400e.cpp`

```cpp
// capture_d400e.cpp
// Simple live capture from FRAMOS D400e camera

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char* argv[]) try
{
    // Create a RealSense pipeline
    rs2::pipeline pipe;
    
    // Configure streams (RGB + Depth)
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 30);
    
    // Start streaming
    std::cout << "Starting camera..." << std::endl;
    pipe.start(cfg);
    
    // Wait for camera to warm up
    for(int i = 0; i < 30; i++) {
        pipe.wait_for_frames();
    }
    std::cout << "Camera ready!" << std::endl;
    
    // Create alignment object (align depth to color)
    rs2::align align_to_color(RS2_STREAM_COLOR);
    
    // Capture frames
    int frame_count = 0;
    
    while(true) {
        // Wait for frames
        rs2::frameset frames = pipe.wait_for_frames();
        
        // Align depth to color
        frames = align_to_color.process(frames);
        
        // Get aligned frames
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();
        
        // Convert to OpenCV Mat
        cv::Mat color(cv::Size(848, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth(cv::Size(848, 480), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
        
        // Create depth colormap for visualization
        cv::Mat depth_8bit;
        depth.convertTo(depth_8bit, CV_8U, 0.03);  // Scale for display
        cv::Mat depth_colormap;
        cv::applyColorMap(depth_8bit, depth_colormap, cv::COLORMAP_JET);
        
        // Display
        cv::imshow("RGB", color);
        cv::imshow("Depth", depth_colormap);
        
        // Press 's' to save, 'q' to quit
        char key = cv::waitKey(1);
        
        if(key == 's') {
            // Save frames
            std::string filename = "frame_" + std::to_string(frame_count);
            cv::imwrite(filename + "_rgb.png", color);
            cv::imwrite(filename + "_depth.png", depth);
            cv::imwrite(filename + "_depth_vis.png", depth_colormap);
            
            std::cout << "Saved " << filename << std::endl;
            frame_count++;
        }
        else if(key == 'q') {
            break;
        }
    }
    
    std::cout << "Captured " << frame_count << " frames" << std::endl;
    
    return 0;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error: " << e.what() << std::endl;
    return 1;
}
catch (const std::exception& e)
{
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
```

#### File 2: `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.10)
project(D400e_Capture)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)

# RealSense paths
set(REALSENSE_INCLUDE "C:/Program Files/FRAMOS/librealsense2/include")
set(REALSENSE_LIB "C:/Program Files/FRAMOS/librealsense2/lib/realsense2.lib")

# Include directories
include_directories(${REALSENSE_INCLUDE})
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(capture_d400e capture_d400e.cpp)

# Link libraries
target_link_libraries(capture_d400e 
    ${REALSENSE_LIB}
    ${OpenCV_LIBS}
)

# Copy DLLs to output directory (Windows)
if(WIN32)
    add_custom_command(TARGET capture_d400e POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Program Files/FRAMOS/librealsense2/bin/realsense2.dll"
            $<TARGET_FILE_DIR:capture_d400e>
    )
    
    add_custom_command(TARGET capture_d400e POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "C:/Program Files/FRAMOS/CameraSuite/bin/CameraSuite.dll"
            $<TARGET_FILE_DIR:capture_d400e>
    )
endif()
```

---

### Step 5: Build with CMake

```bash
cd "/c/Users/guibc/Robotics Projects/Pallet 6D Pose Estimation/d400e_cpp/build"

# Configure (using exact OpenCV path we found)
cmake .. -G "Visual Studio 18 2026" -A x64 \
    -DOpenCV_DIR="C:/opencv/build/x64/vc16/lib"

# Build
cmake --build . --config Release
```

**Expected output:**
```
-- Found OpenCV: C:/opencv/build (found version "4.12.0")
-- Configuring done (4.4s)
-- Generating done (0.1s)
-- Build files have been written to: ...

Building...
capture_d400e.vcxproj -> .../Release/capture_d400e.exe
```

---

### Step 6: Copy OpenCV DLL

```bash
cd Release

# Copy OpenCV DLL (adjust version number if different)
cp "/c/opencv/build/x64/vc16/bin/opencv_world4120.dll" .

# Verify all DLLs are present
ls *.dll
```

**You should see:**
- `realsense2.dll` (copied by CMake)
- `CameraSuite.dll` (copied by CMake)
- `opencv_world4120.dll` (manually copied)

---

### Step 7: Connect Camera and Run

**Connect camera:**
1. Power on D400e camera (PoE or external power)
2. Connect network cable
3. Ensure camera and PC on same subnet
4. Test with RealSense Viewer first (optional):
   ```bash
   "/c/Program Files/FRAMOS/librealsense2/bin/realsense-viewer.exe" &
   ```

**Run the program:**
```bash
cd "/c/Users/guibc/Robotics Projects/Pallet 6D Pose Estimation/d400e_cpp/build/Release"

./capture_d400e.exe
```

**Expected output:**
```
Starting camera...
Camera ready!
```

Two windows appear:
- **RGB** - Live color feed
- **Depth** - Live depth colormap

**Controls:**
- Press `s` - Save current frame
- Press `q` - Quit program

---

## 📁 Final Project Structure

```
Pallet 6D Pose Estimation/
├── d400e_cpp/
│   ├── capture_d400e.cpp          # Source code
│   ├── CMakeLists.txt              # Build configuration
│   └── build/
│       ├── D400e_Capture.sln       # Visual Studio solution
│       └── Release/
│           ├── capture_d400e.exe   # ← Executable
│           ├── realsense2.dll
│           ├── CameraSuite.dll
│           └── opencv_world4120.dll
│
└── pallet_venv/                    # (Your Python venv, separate)
```

---

## 🔧 Troubleshooting

### "opencv_world4XXX.dll not found"
```bash
# Find your OpenCV version
ls "/c/opencv/build/x64/vc16/bin"/opencv_world*.dll

# Copy the exact version
cp "/c/opencv/build/x64/vc16/bin/opencv_world4XXX.dll" Release/
```

### "Could not find OpenCV"
```bash
# Use the exact path from find command
find "/c/opencv" -name "OpenCVConfig.cmake"
# Use that path in: -DOpenCV_DIR="<path>"
```

### "Camera not detected"
```bash
# Test with RealSense Viewer first
"/c/Program Files/FRAMOS/librealsense2/bin/realsense-viewer.exe" &

# Check camera power and network
# Verify IP subnet matches PC
```

### "RealSense error: No device connected"
- Check camera power (PoE or external)
- Verify network cable connected
- Check camera IP matches PC subnet
- Use ConfigureIP tool if needed

---

## ⚡ Quick Rebuild Commands

If you modify `capture_d400e.cpp`:

```bash
cd "/c/Users/guibc/Robotics Projects/Pallet 6D Pose Estimation/d400e_cpp/build"

# Rebuild (no need to reconfigure)
cmake --build . --config Release

# Run
cd Release
./capture_d400e.exe
```

---

## 🎯 Summary of Steps

1. ✅ Install Visual Studio C++ tools (or verify installed)
2. ✅ Download and extract OpenCV to `C:\opencv`
3. ✅ Create project directory and source files
4. ✅ Configure with CMake (point to OpenCV)
5. ✅ Build with CMake
6. ✅ Copy OpenCV DLL to Release folder
7. ✅ Connect camera
8. ✅ Run program

**Total setup time:** ~30 minutes (mostly downloads/installs)  
**Works:** Immediately after setup, live camera access!

---

## 📊 What You Get

**Captured files when you press 's':**
- `frame_0_rgb.png` - 848x480 color image
- `frame_0_depth.png` - 848x480 16-bit depth (millimeters)
- `frame_0_depth_vis.png` - Depth colormap visualization

**Depth values:**
- Raw depth units (typically millimeters)
- Convert to meters: `depth_meters = depth_pixels * 0.001`

---

## 🚀 Next Steps

**For your pallet detection project:**
1. Modify the code to save frames automatically
2. Process saved images in Python for pose estimation
3. Or integrate Python directly into C++ using pybind11

**Example workflow:**
```
C++ Program → Saves frames
     ↓
Python Script → Loads frames → 6D Pose Estimation
```

---

## 📞 Support

- FRAMOS Support: support@framos.com
- RealSense Documentation: https://dev.intelrealsense.com/
- OpenCV Documentation: https://docs.opencv.org/

---

**Last Updated:** February 2026  
**Tested On:** Windows 10/11, Visual Studio 2026, OpenCV 4.12, FRAMOS D400e SDK v2.10.0
