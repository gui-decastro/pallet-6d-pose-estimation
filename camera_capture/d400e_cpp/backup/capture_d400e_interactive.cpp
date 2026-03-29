// capture_d400e_interactive.cpp
// Interactive filter tuning - adjust parameters in real-time!

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

// Filter parameters (adjustable at runtime)
struct FilterParams {
    bool use_spatial = true;
    float spatial_magnitude = 2.0f;
    float spatial_alpha = 0.5f;
    float spatial_delta = 20.0f;
    
    bool use_temporal = false;
    float temporal_alpha = 0.4f;
    float temporal_delta = 20.0f;
    
    bool use_hole_filling = true;
    int hole_fill_mode = 1;  // 0=off, 1=farest_from_around, 2=nearest_from_around
    
    float depth_min = 0.3f;  // meters
    float depth_max = 3.0f;  // meters
};

void print_controls() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CONTROLS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "FILTERS:" << std::endl;
    std::cout << "  1/2 - Spatial filter ON/OFF" << std::endl;
    std::cout << "  3/4 - Temporal filter ON/OFF" << std::endl;
    std::cout << "  5/6 - Hole filling ON/OFF" << std::endl;
    std::cout << std::endl;
    std::cout << "SPATIAL FILTER:" << std::endl;
    std::cout << "  q/w - Magnitude (strength): decrease/increase" << std::endl;
    std::cout << "  a/s - Alpha (edge preservation): decrease/increase" << std::endl;
    std::cout << "  z/x - Delta (range): decrease/increase" << std::endl;
    std::cout << std::endl;
    std::cout << "TEMPORAL FILTER:" << std::endl;
    std::cout << "  e/r - Alpha: decrease/increase" << std::endl;
    std::cout << "  d/f - Delta: decrease/increase" << std::endl;
    std::cout << std::endl;
    std::cout << "HOLE FILLING:" << std::endl;
    std::cout << "  h - Cycle hole fill mode (0/1/2)" << std::endl;
    std::cout << std::endl;
    std::cout << "DEPTH RANGE:" << std::endl;
    std::cout << "  [/] - Min depth: decrease/increase" << std::endl;
    std::cout << "  ;/' - Max depth: decrease/increase" << std::endl;
    std::cout << std::endl;
    std::cout << "OTHER:" << std::endl;
    std::cout << "  SPACE - Save current frame" << std::endl;
    std::cout << "  p - Print current settings" << std::endl;
    std::cout << "  c - Save current settings to config file" << std::endl;
    std::cout << "  ESC or q - Quit" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void print_params(const FilterParams& params) {
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "CURRENT FILTER SETTINGS" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Spatial Filter:  " << (params.use_spatial ? "ON " : "OFF") << std::endl;
    if (params.use_spatial) {
        std::cout << "  Magnitude: " << params.spatial_magnitude << std::endl;
        std::cout << "  Alpha:     " << params.spatial_alpha << std::endl;
        std::cout << "  Delta:     " << params.spatial_delta << std::endl;
    }
    std::cout << "Temporal Filter: " << (params.use_temporal ? "ON " : "OFF") << std::endl;
    if (params.use_temporal) {
        std::cout << "  Alpha:     " << params.temporal_alpha << std::endl;
        std::cout << "  Delta:     " << params.temporal_delta << std::endl;
    }
    std::cout << "Hole Filling:    " << (params.use_hole_filling ? "ON " : "OFF");
    std::cout << " (mode: " << params.hole_fill_mode << ")" << std::endl;
    std::cout << "Depth Range:     " << params.depth_min << "m - " << params.depth_max << "m" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
}

void save_config(const FilterParams& params, const std::string& filename = "filter_config.txt") {
    std::ofstream file(filename);
    file << "# D400e Filter Configuration" << std::endl;
    file << "# Generated automatically - can be edited manually" << std::endl;
    file << std::endl;
    file << "use_spatial=" << params.use_spatial << std::endl;
    file << "spatial_magnitude=" << params.spatial_magnitude << std::endl;
    file << "spatial_alpha=" << params.spatial_alpha << std::endl;
    file << "spatial_delta=" << params.spatial_delta << std::endl;
    file << std::endl;
    file << "use_temporal=" << params.use_temporal << std::endl;
    file << "temporal_alpha=" << params.temporal_alpha << std::endl;
    file << "temporal_delta=" << params.temporal_delta << std::endl;
    file << std::endl;
    file << "use_hole_filling=" << params.use_hole_filling << std::endl;
    file << "hole_fill_mode=" << params.hole_fill_mode << std::endl;
    file << std::endl;
    file << "depth_min=" << params.depth_min << std::endl;
    file << "depth_max=" << params.depth_max << std::endl;
    file.close();
    
    std::cout << "✓ Settings saved to: " << filename << std::endl;
}

int main(int argc, char* argv[]) try
{
    std::cout << "D400e Interactive Filter Tuning" << std::endl;
    std::cout << "================================" << std::endl;
    
    // Initialize filter parameters
    FilterParams params;
    
    // Find camera
    rs2::context ctx;
    auto devices = ctx.query_devices();
    if (devices.size() == 0) {
        std::cerr << "No devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found: " << devices[0].get_info(RS2_CAMERA_INFO_NAME) << std::endl;
    
    // Create pipeline
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 30);
    
    std::cout << "Starting camera..." << std::endl;
    auto profile = pipe.start(cfg);
    
    // Get depth scale
    auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
    float depth_scale = depth_sensor.get_depth_scale();
    
    // Create filters
    rs2::spatial_filter spatial;
    rs2::temporal_filter temporal;
    rs2::hole_filling_filter hole_fill;
    rs2::align align_to_color(RS2_STREAM_COLOR);
    
    // Warm up
    std::cout << "Warming up..." << std::endl;
    for(int i = 0; i < 60; i++) {
        pipe.wait_for_frames();
    }
    
    print_controls();
    print_params(params);
    
    int frame_count = 0;
    bool settings_changed = true;  // Force update on first frame
    
    while(true) {
        // Get frames
        rs2::frameset frames = pipe.wait_for_frames();
        frames = align_to_color.process(frames);
        
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        rs2::video_frame color_frame = frames.get_color_frame();
        
        // Apply filters based on settings
        if (settings_changed) {
            // Update filter parameters
            if (params.use_spatial) {
                spatial.set_option(RS2_OPTION_FILTER_MAGNITUDE, params.spatial_magnitude);
                spatial.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, params.spatial_alpha);
                spatial.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, params.spatial_delta);
            }
            
            if (params.use_temporal) {
                temporal.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, params.temporal_alpha);
                temporal.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, params.temporal_delta);
            }
            
            if (params.use_hole_filling) {
                hole_fill.set_option(RS2_OPTION_HOLES_FILL, params.hole_fill_mode);
            }
            
            settings_changed = false;
        }
        
        // Apply enabled filters
        if (params.use_spatial) {
            depth_frame = spatial.process(depth_frame);
        }
        if (params.use_temporal) {
            depth_frame = temporal.process(depth_frame);
        }
        if (params.use_hole_filling) {
            depth_frame = hole_fill.process(depth_frame);
        }
        
        // Convert to OpenCV
        cv::Mat color(cv::Size(848, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth_raw(cv::Size(848, 480), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
        
        // Apply depth range clipping
        cv::Mat depth_clipped = depth_raw.clone();
        uint16_t min_val = static_cast<uint16_t>(params.depth_min / depth_scale);
        uint16_t max_val = static_cast<uint16_t>(params.depth_max / depth_scale);
        depth_clipped.setTo(0, depth_raw < min_val);
        depth_clipped.setTo(0, depth_raw > max_val);
        
        // Create visualization
        cv::Mat depth_8bit;
        depth_clipped.convertTo(depth_8bit, CV_8U, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
        cv::Mat depth_colormap;
        cv::applyColorMap(depth_8bit, depth_colormap, cv::COLORMAP_JET);
        
        // Add text overlay with current settings
        std::string status = "Spatial:" + std::string(params.use_spatial ? "ON" : "OFF") +
                           " Temporal:" + std::string(params.use_temporal ? "ON" : "OFF") +
                           " Holes:" + std::string(params.use_hole_filling ? "ON" : "OFF");
        cv::putText(depth_colormap, status, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        char range_text[100];
        sprintf(range_text, "Range: %.2fm - %.2fm", params.depth_min, params.depth_max);
        cv::putText(depth_colormap, range_text, cv::Point(10, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        // Display
        cv::imshow("RGB", color);
        cv::imshow("Depth (Filtered)", depth_colormap);
        
        // Handle keyboard input
        char key = cv::waitKey(1);
        
        if (key == 27 || key == 'q') {  // ESC or 'q'
            break;
        }
        else if (key == '1') { params.use_spatial = false; settings_changed = true; std::cout << "Spatial filter: OFF" << std::endl; }
        else if (key == '2') { params.use_spatial = true; settings_changed = true; std::cout << "Spatial filter: ON" << std::endl; }
        else if (key == '3') { params.use_temporal = false; settings_changed = true; std::cout << "Temporal filter: OFF" << std::endl; }
        else if (key == '4') { params.use_temporal = true; settings_changed = true; std::cout << "Temporal filter: ON" << std::endl; }
        else if (key == '5') { params.use_hole_filling = false; settings_changed = true; std::cout << "Hole filling: OFF" << std::endl; }
        else if (key == '6') { params.use_hole_filling = true; settings_changed = true; std::cout << "Hole filling: ON" << std::endl; }
        
        // Spatial filter adjustments
        else if (key == 'q') { params.spatial_magnitude = std::max(1.0f, params.spatial_magnitude - 0.5f); settings_changed = true; std::cout << "Spatial magnitude: " << params.spatial_magnitude << std::endl; }
        else if (key == 'w') { params.spatial_magnitude = std::min(5.0f, params.spatial_magnitude + 0.5f); settings_changed = true; std::cout << "Spatial magnitude: " << params.spatial_magnitude << std::endl; }
        else if (key == 'a') { params.spatial_alpha = std::max(0.0f, params.spatial_alpha - 0.1f); settings_changed = true; std::cout << "Spatial alpha: " << params.spatial_alpha << std::endl; }
        else if (key == 's') { params.spatial_alpha = std::min(1.0f, params.spatial_alpha + 0.1f); settings_changed = true; std::cout << "Spatial alpha: " << params.spatial_alpha << std::endl; }
        else if (key == 'z') { params.spatial_delta = std::max(1.0f, params.spatial_delta - 5.0f); settings_changed = true; std::cout << "Spatial delta: " << params.spatial_delta << std::endl; }
        else if (key == 'x') { params.spatial_delta = std::min(50.0f, params.spatial_delta + 5.0f); settings_changed = true; std::cout << "Spatial delta: " << params.spatial_delta << std::endl; }
        
        // Temporal filter adjustments
        else if (key == 'e') { params.temporal_alpha = std::max(0.0f, params.temporal_alpha - 0.1f); settings_changed = true; std::cout << "Temporal alpha: " << params.temporal_alpha << std::endl; }
        else if (key == 'r') { params.temporal_alpha = std::min(1.0f, params.temporal_alpha + 0.1f); settings_changed = true; std::cout << "Temporal alpha: " << params.temporal_alpha << std::endl; }
        else if (key == 'd') { params.temporal_delta = std::max(1.0f, params.temporal_delta - 5.0f); settings_changed = true; std::cout << "Temporal delta: " << params.temporal_delta << std::endl; }
        else if (key == 'f') { params.temporal_delta = std::min(50.0f, params.temporal_delta + 5.0f); settings_changed = true; std::cout << "Temporal delta: " << params.temporal_delta << std::endl; }
        
        // Hole filling mode
        else if (key == 'h') { params.hole_fill_mode = (params.hole_fill_mode + 1) % 3; settings_changed = true; std::cout << "Hole fill mode: " << params.hole_fill_mode << std::endl; }
        
        // Depth range
        else if (key == '[') { params.depth_min = std::max(0.1f, params.depth_min - 0.1f); std::cout << "Min depth: " << params.depth_min << "m" << std::endl; }
        else if (key == ']') { params.depth_min = std::min(params.depth_max - 0.1f, params.depth_min + 0.1f); std::cout << "Min depth: " << params.depth_min << "m" << std::endl; }
        else if (key == ';') { params.depth_max = std::max(params.depth_min + 0.1f, params.depth_max - 0.1f); std::cout << "Max depth: " << params.depth_max << "m" << std::endl; }
        else if (key == '\'') { params.depth_max = std::min(10.0f, params.depth_max + 0.1f); std::cout << "Max depth: " << params.depth_max << "m" << std::endl; }
        
        // Save frame
        else if (key == ' ') {
            std::string base = "frame_" + std::to_string(frame_count);
            cv::imwrite(base + "_rgb.png", color);
            cv::imwrite(base + "_depth.png", depth_raw);
            cv::imwrite(base + "_depth_vis.png", depth_colormap);
            std::cout << "✓ Saved " << base << std::endl;
            frame_count++;
        }
        
        // Print current settings
        else if (key == 'p') {
            print_params(params);
        }
        
        // Save config
        else if (key == 'c') {
            save_config(params);
        }
    }
    
    std::cout << "\nCaptured " << frame_count << " frames" << std::endl;
    std::cout << "Exiting..." << std::endl;
    
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
