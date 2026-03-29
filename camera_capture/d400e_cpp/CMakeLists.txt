cmake_minimum_required(VERSION 3.10)
project(D400e_Capture)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV path - defaults to C:/opencv. Override with: cmake .. -DOpenCV_DIR="path/to/opencv/build/x64/vcXX/lib"
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib" CACHE PATH "Path to OpenCV cmake config directory")
find_package(OpenCV REQUIRED)

# RealSense paths (adjust if needed)
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

# Copy DLLs to output directory (Windows only)
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
