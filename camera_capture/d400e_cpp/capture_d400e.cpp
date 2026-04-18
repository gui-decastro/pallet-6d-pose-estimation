// capture_d400e_production.cpp
// Optimized for Pallet 6D Pose Estimation Pipeline
// Uses fixed optimal parameters - no interactive tuning

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <ctime>
#include <sstream>
#include <direct.h>    // _mkdir on Windows
#include <errno.h>     // EEXIST

// OPTIMAL PARAMETERS FOR PALLET DETECTION + ICP
struct OptimalParams {
    // Decimation: OFF for maximum resolution (edges are critical for ICP)
    bool use_decimation = false;
    
    // Spatial Filter: ON - reduces noise while preserving edges
    bool use_spatial = true;
    float spatial_magnitude = 2.0f;    // Medium smoothing
    float spatial_alpha = 0.5f;        // Balanced edge preservation
    float spatial_delta = 20.0f;       // Standard threshold
    
    // Temporal Filter: ON - smooths frame-to-frame for stable ICP
    bool use_temporal = true;
    float temporal_alpha = 0.4f;       // Moderate smoothing
    float temporal_delta = 20.0f;      // Standard threshold
    
    // Hole Filling: OFF - preserve pallet corner geometry for ICP
    bool use_hole_filling = false;
    
    // Depth Range: Focused on pallet distance
    float depth_min = 0.8f;            // 0.8 meters (cut out floor/foreground)
    float depth_max = 6.0f;            // 6.0 meters (typical pallet range)
};

// Creates a directory and all missing parent directories.
static void make_dir(const std::string& path) {
    // Walk each prefix and create any missing component.
    for (size_t pos = 0; pos != std::string::npos; ) {
        pos = path.find_first_of("\\/", pos + 1);
        std::string sub = path.substr(0, pos);
        if (sub.empty()) continue;
        int result = _mkdir(sub.c_str());
        if (result != 0 && errno != EEXIST) {
            std::cerr << "[ERROR] Could not create directory: " << sub
                      << " (errno=" << errno << ")\n";
            return;
        }
    }
}

// Creates a timestamped session folder.
// If output_root is non-empty, uses it directly as the session directory.
// Otherwise falls back to the default timestamped path under collected_data.
std::string create_session_folder(const std::string& output_root = "") {
    if (!output_root.empty()) {
        make_dir(output_root);
        return output_root;
    }

    const std::string default_root =
        "C:\\Users\\guibc\\Robotics Projects\\pallet-6d-pose-estimation\\camera_capture\\collected_data";

    std::time_t now = std::time(nullptr);
    std::tm t{};
    localtime_s(&t, &now);

    std::ostringstream oss;
    oss << default_root << "\\session_"
        << std::put_time(&t, "%Y-%m-%d_%H-%M-%S");

    std::string session_path = oss.str();

    make_dir(default_root);
    make_dir(session_path);

    return session_path;
}

void print_settings(const OptimalParams& p, float depth_scale) {
    std::cout << "\n";
    std::cout << "========================================================================\n";
    std::cout << "     PALLET DETECTION - OPTIMIZED FILTER CONFIGURATION\n";
    std::cout << "========================================================================\n";
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\nDECIMATION FILTER:     " << (p.use_decimation ? "ENABLED" : "DISABLED") << "\n";
    std::cout << "  Reason: Need full resolution for sharp pallet edges (ICP requirement)\n";
    
    std::cout << "\nSPATIAL FILTER:        ENABLED\n";
    std::cout << "  Magnitude: " << p.spatial_magnitude << " (iterations)\n";
    std::cout << "  Alpha:     " << p.spatial_alpha << " (edge preservation)\n";
    std::cout << "  Delta:     " << p.spatial_delta << " mm (depth threshold)\n";
    std::cout << "  Reason: Reduces noise while preserving object boundaries\n";
    
    std::cout << "\nTEMPORAL FILTER:       ENABLED\n";
    std::cout << "  Alpha:     " << p.temporal_alpha << " (smoothing strength)\n";
    std::cout << "  Delta:     " << p.temporal_delta << " mm (depth threshold)\n";
    std::cout << "  Reason: Stabilizes depth for consistent ICP alignment\n";
    
    std::cout << "\nHOLE FILLING:          DISABLED\n";
    std::cout << "  Reason: Preserve pallet corner features for accurate ICP\n";
    
    std::cout << "\nDEPTH RANGE:\n";
    std::cout << "  Min: " << p.depth_min << " meters (filter foreground noise)\n";
    std::cout << "  Max: " << p.depth_max << " meters (focus on pallet)\n";
    std::cout << "  Reason: Isolates pallet, removes floor and distant clutter\n";
    
    std::cout << "\nDEPTH SCALE:           " << depth_scale << " (raw units to meters)\n";
    
    std::cout << "\n========================================================================\n";
}

void save_frame_data(const std::string& session_dir,
                     int frame_index,
                     const cv::Mat& rgb,
                     const cv::Mat& depth_raw,
                     const cv::Mat& depth_vis,
                     float depth_scale,
                     const rs2::video_stream_profile& color_profile) {

    // Build base path: captures/session_YYYY-MM-DD_HH-MM-SS/frame_0042
    std::ostringstream oss;
    oss << session_dir << "\\frame_" << std::setw(4) << std::setfill('0') << frame_index;
    std::string base = oss.str();

    // Save RGB
    cv::imwrite(base + "_rgb.png", rgb);
    
    // Save raw depth (16-bit, in original units)
    cv::imwrite(base + "_depth.png", depth_raw);
    
    // Save depth in meters (32-bit float binary)
    cv::Mat depth_meters;
    depth_raw.convertTo(depth_meters, CV_32F, depth_scale);
    std::ofstream depth_file(base + "_depth_meters.bin", std::ios::binary);
    depth_file.write((char*)depth_meters.data, depth_meters.total() * sizeof(float));
    depth_file.close();
    
    // Save visualization
    cv::imwrite(base + "_depth_vis.png", depth_vis);
    
    // Save camera intrinsics (needed for depth-to-3D conversion)
    auto intrinsics = color_profile.get_intrinsics();
    std::ofstream intrinsics_file(base + "_intrinsics.json");
    intrinsics_file << "{\n";
    intrinsics_file << "  \"width\": " << intrinsics.width << ",\n";
    intrinsics_file << "  \"height\": " << intrinsics.height << ",\n";
    intrinsics_file << "  \"fx\": " << intrinsics.fx << ",\n";
    intrinsics_file << "  \"fy\": " << intrinsics.fy << ",\n";
    intrinsics_file << "  \"cx\": " << intrinsics.ppx << ",\n";
    intrinsics_file << "  \"cy\": " << intrinsics.ppy << ",\n";
    intrinsics_file << "  \"depth_scale\": " << depth_scale << ",\n";
    intrinsics_file << "  \"model\": " << intrinsics.model << "\n";
    intrinsics_file << "}\n";
    intrinsics_file.close();

    // Extract just the filename portion for clean display
    std::string fname = "frame_" + oss.str().substr(oss.str().rfind('\\') + 1);
    std::cout << "\n[SAVED] " << base << " - 5 files:\n";
    std::cout << "  [OK] " << base << "_rgb.png (1280x720 color)\n";
    std::cout << "  [OK] " << base << "_depth.png (16-bit raw depth)\n";
    std::cout << "  [OK] " << base << "_depth_meters.bin (32-bit float, meters)\n";
    std::cout << "  [OK] " << base << "_depth_vis.png (colormap visualization)\n";
    std::cout << "  [OK] " << base << "_intrinsics.json (camera parameters)\n";
}

int main(int argc, char* argv[]) try {
    std::cout << "\n";
    std::cout << "========================================================================\n";
    std::cout << "     D400e Pallet Detection - Production Capture\n";
    std::cout << "========================================================================\n";

    OptimalParams params;

    // Use argv[1] as output folder if provided; otherwise use default timestamped path
    std::string output_root = (argc > 1) ? std::string(argv[1]) : "";
    std::string session_dir = create_session_folder(output_root);
    std::cout << "\n[SESSION] Output folder: " << session_dir << "\n";

    // Find camera
    std::cout << "\n[INIT] Searching for camera...\n";
    rs2::context ctx;
    auto devices = ctx.query_devices();
    if (devices.size() == 0) {
        std::cerr << "[ERROR] No camera found!\n";
        std::cerr << "  - Check power and network connection\n";
        std::cerr << "  - Close RealSense Viewer if open\n";
        return 1;
    }
    
    std::cout << "[OK] Found: " << devices[0].get_info(RS2_CAMERA_INFO_NAME) << "\n";
    std::cout << "     Serial: " << devices[0].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << "\n";
    
    // Configure pipeline
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 15); // 1280x720 is the highest resolution supported. Supports up to 30fps, but choosing 15fps for now
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 15); // Support 1280x800, but sticking with 1280x720 for now to keep it consistent with RGB stream
    
    std::cout << "\n[INIT] Starting camera streams...\n";
    auto profile = pipe.start(cfg);
    
    // Get depth scale
    auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
    float depth_scale = depth_sensor.get_depth_scale();
    
    // Get camera intrinsics
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    
    // Print configuration
    print_settings(params, depth_scale);
    
    // Create filters
    rs2::spatial_filter spatial;
    rs2::temporal_filter temporal;
    rs2::align align_to_color(RS2_STREAM_COLOR);
    
    // Configure spatial filter
    spatial.set_option(RS2_OPTION_FILTER_MAGNITUDE, params.spatial_magnitude);
    spatial.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, params.spatial_alpha);
    spatial.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, params.spatial_delta);
    
    // Configure temporal filter
    temporal.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, params.temporal_alpha);
    temporal.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, params.temporal_delta);
    
    // Warm up camera
    std::cout << "\n[INIT] Warming up camera";
    for(int i = 0; i < 60; i++) {
        pipe.wait_for_frames();
        if (i % 15 == 0) std::cout << ".";
    }
    std::cout << " done!\n";
    
    std::cout << "\n[READY] Camera ready for capture\n";
    std::cout << "\nCONTROLS:\n";
    std::cout << "  SPACE - Capture frame (saves 5 files for pipeline)\n";
    std::cout << "  ESC   - Exit program\n";
    std::cout << "\n";
    
    int frame_count = 0;
    
    while(true) {
        // Get frames
        rs2::frameset frames;
        try {
            frames = pipe.wait_for_frames(5000);
        } catch (...) {
            std::cerr << "[WARNING] Frame timeout, retrying...\n";
            continue;
        }
        
        // Align depth to color
        frames = align_to_color.process(frames);
        
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        rs2::video_frame color_frame = frames.get_color_frame();
        
        if (!depth_frame || !color_frame) continue;
        
        // Apply filters (spatial + temporal only, no decimation, no hole filling)
        if (params.use_spatial) {
            depth_frame = spatial.process(depth_frame);
        }
        if (params.use_temporal) {
            depth_frame = temporal.process(depth_frame);
        }
        
        // Convert to OpenCV
        cv::Mat color(cv::Size(1280, 720), CV_8UC3, (void*)color_frame.get_data());
        cv::Mat depth_raw(cv::Size(1280, 720), CV_16UC1, (void*)depth_frame.get_data());
        
        // Apply depth range clipping
        cv::Mat depth_clipped = depth_raw.clone();
        uint16_t min_val = static_cast<uint16_t>(params.depth_min / depth_scale);
        uint16_t max_val = static_cast<uint16_t>(params.depth_max / depth_scale);
        depth_clipped.setTo(0, depth_raw < min_val);
        depth_clipped.setTo(0, depth_raw > max_val);
        
        // Create visualization with histogram-based colormap
        cv::Mat depth_colormap;
        std::vector<uint16_t> valid_pixels;
        for (int i = 0; i < depth_clipped.rows; i++) {
            for (int j = 0; j < depth_clipped.cols; j++) {
                uint16_t val = depth_clipped.at<uint16_t>(i, j);
                if (val > 0) valid_pixels.push_back(val);
            }
        }
        
        if (valid_pixels.size() > 100) {
            std::sort(valid_pixels.begin(), valid_pixels.end());
            uint16_t p1 = valid_pixels[static_cast<size_t>(valid_pixels.size() * 0.01)];
            uint16_t p99 = valid_pixels[static_cast<size_t>(valid_pixels.size() * 0.99)];
            
            if (p99 > p1) {
                cv::Mat depth_8bit;
                depth_clipped.convertTo(depth_8bit, CV_8U, 255.0 / (p99 - p1), -p1 * 255.0 / (p99 - p1));
                cv::applyColorMap(depth_8bit, depth_colormap, cv::COLORMAP_JET);
            } else {
                depth_colormap = cv::Mat::zeros(depth_clipped.size(), CV_8UC3);
            }
        } else {
            depth_colormap = cv::Mat::zeros(depth_clipped.size(), CV_8UC3);
        }
        
        // depth_colormap is clean (no text) - used for saving
        // Add overlay only to the display copy
        cv::Mat depth_display = depth_colormap.clone();
        cv::putText(depth_display, "PALLET DETECTION MODE", cv::Point(10, 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        char info[100];
        snprintf(info, sizeof(info), "Filters: Spatial+Temporal | Range: %.1f-%.1fm", 
                params.depth_min, params.depth_max);
        cv::putText(depth_display, info, cv::Point(10, 55),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // Display
        cv::imshow("RGB Camera", color);
        cv::imshow("Depth (Optimized for ICP)", depth_display);
        
        // Handle keyboard
        int key_code = cv::waitKey(1);
        if (key_code == -1) continue;
        
        char key = static_cast<char>(key_code & 0xFF);
        
        if (key == 27) {  // ESC
            break;
        }
        else if (key == ' ') {  // SPACE - capture
            save_frame_data(session_dir, frame_count, color, depth_raw,
                            depth_colormap, depth_scale, color_stream);
            frame_count++;
        }
    }
    
    std::cout << "\n[DONE] Captured " << frame_count << " frames -> " << session_dir << "\n";
    std::cout << "       Ready for YOLO detection + ICP alignment\n";
    std::cout << "\nExiting...\n\n";
    
    return 0;
}
catch (const rs2::error& e) {
    std::cerr << "\n[FATAL ERROR] RealSense: " << e.what() << "\n";
    std::cerr << "Function: " << e.get_failed_function() << "\n";
    return 1;
}
catch (const std::exception& e) {
    std::cerr << "\n[FATAL ERROR] " << e.what() << "\n";
    return 1;
}