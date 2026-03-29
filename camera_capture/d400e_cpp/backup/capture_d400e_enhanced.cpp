// capture_d400e_enhanced.cpp
// D400e capture with filtering and multiple output formats for pallet detection

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[]) try
{
    std::cout << "D400e Capture for Pallet Detection" << std::endl;
    std::cout << "===================================" << std::endl;
    
    // Create context and find devices
    rs2::context ctx;
    auto devices = ctx.query_devices();
    
    if (devices.size() == 0) {
        std::cerr << "No devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found device: " << devices[0].get_info(RS2_CAMERA_INFO_NAME) << std::endl;
    
    // Create pipeline
    rs2::pipeline pipe;
    rs2::config cfg;
    
    // Configure streams
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 30);
    
    // Start
    std::cout << "Starting camera..." << std::endl;
    auto profile = pipe.start(cfg);
    
    // Get depth scale (conversion from raw units to meters)
    auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
    float depth_scale = depth_sensor.get_depth_scale();
    std::cout << "Depth scale: " << depth_scale << " (multiply raw values by this to get meters)" << std::endl;
    
    // Get camera intrinsics (needed for 3D reconstruction)
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto intrinsics = color_stream.get_intrinsics();
    
    std::cout << "Camera intrinsics:" << std::endl;
    std::cout << "  fx: " << intrinsics.fx << std::endl;
    std::cout << "  fy: " << intrinsics.fy << std::endl;
    std::cout << "  cx: " << intrinsics.ppx << std::endl;
    std::cout << "  cy: " << intrinsics.ppy << std::endl;
    
    // POST-PROCESSING FILTERS (for cleaner depth)
    rs2::spatial_filter spatial;
    rs2::hole_filling_filter hole_fill;
    
    // Configure spatial filter (preserve edges for pallet detection)
    spatial.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
    spatial.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f);
    spatial.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20);
    
    // Hole filling mode 1 = fill from neighboring pixels
    hole_fill.set_option(RS2_OPTION_HOLES_FILL, 1);
    
    // Alignment
    rs2::align align_to_color(RS2_STREAM_COLOR);
    
    // Warm up
    std::cout << "Warming up..." << std::endl;
    for(int i = 0; i < 60; i++) {
        pipe.wait_for_frames();
    }
    
    std::cout << "\nReady!" << std::endl;
    std::cout << "Controls: 's' = save, 'f' = toggle filters, 'q' = quit" << std::endl;
    
    int frame_count = 0;
    bool use_filters = true;
    
    while(true) {
        // Get frames
        rs2::frameset frames = pipe.wait_for_frames();
        
        // Align
        frames = align_to_color.process(frames);
        
        // Get frames
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        rs2::video_frame color_frame = frames.get_color_frame();
        
        // Apply filters if enabled
        if (use_filters) {
            depth_frame = spatial.process(depth_frame);
            depth_frame = hole_fill.process(depth_frame);
        }
        
        // Convert to OpenCV
        cv::Mat color(cv::Size(848, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth_raw(cv::Size(848, 480), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
        
        // Create visualization
        cv::Mat depth_8bit;
        depth_raw.convertTo(depth_8bit, CV_8U, 0.03);
        cv::Mat depth_colormap;
        cv::applyColorMap(depth_8bit, depth_colormap, cv::COLORMAP_JET);
        
        // Add filter status text
        std::string filter_text = use_filters ? "Filters: ON" : "Filters: OFF";
        cv::putText(depth_colormap, filter_text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // Display
        cv::imshow("RGB", color);
        cv::imshow("Depth", depth_colormap);
        
        char key = cv::waitKey(1);
        
        if(key == 's') {
            std::string base = "frame_" + std::to_string(frame_count);
            
            // 1. Save RGB image
            cv::imwrite(base + "_rgb.png", color);
            
            // 2. Save RAW depth (16-bit PNG - preserves precision)
            cv::imwrite(base + "_depth.png", depth_raw);
            
            // 3. Save depth in METERS (32-bit float)
            cv::Mat depth_meters;
            depth_raw.convertTo(depth_meters, CV_32F, depth_scale);
            
            // Save as binary file (readable by Python numpy)
            std::ofstream out(base + "_depth_meters.bin", std::ios::binary);
            out.write((char*)depth_meters.data, depth_meters.total() * sizeof(float));
            out.close();
            
            // 4. Save visualization
            cv::imwrite(base + "_depth_vis.png", depth_colormap);
            
            // 5. Save camera intrinsics (JSON format)
            std::ofstream intrinsics_file(base + "_intrinsics.json");
            intrinsics_file << "{" << std::endl;
            intrinsics_file << "  \"width\": " << intrinsics.width << "," << std::endl;
            intrinsics_file << "  \"height\": " << intrinsics.height << "," << std::endl;
            intrinsics_file << "  \"fx\": " << intrinsics.fx << "," << std::endl;
            intrinsics_file << "  \"fy\": " << intrinsics.fy << "," << std::endl;
            intrinsics_file << "  \"cx\": " << intrinsics.ppx << "," << std::endl;
            intrinsics_file << "  \"cy\": " << intrinsics.ppy << "," << std::endl;
            intrinsics_file << "  \"depth_scale\": " << depth_scale << "," << std::endl;
            intrinsics_file << "  \"model\": \"" << intrinsics.model << "\"" << std::endl;
            intrinsics_file << "}" << std::endl;
            intrinsics_file.close();
            
            std::cout << "Saved " << base << " (5 files)" << std::endl;
            std::cout << "  - RGB: " << base << "_rgb.png" << std::endl;
            std::cout << "  - Depth (raw): " << base << "_depth.png" << std::endl;
            std::cout << "  - Depth (meters): " << base << "_depth_meters.bin" << std::endl;
            std::cout << "  - Visualization: " << base << "_depth_vis.png" << std::endl;
            std::cout << "  - Intrinsics: " << base << "_intrinsics.json" << std::endl;
            
            frame_count++;
        }
        else if(key == 'f') {
            use_filters = !use_filters;
            std::cout << "Filters: " << (use_filters ? "ON" : "OFF") << std::endl;
        }
        else if(key == 'q') {
            break;
        }
    }
    
    std::cout << "\nCaptured " << frame_count << " frames" << std::endl;
    
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
