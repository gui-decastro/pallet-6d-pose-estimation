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
