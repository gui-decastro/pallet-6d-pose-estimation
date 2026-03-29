# Camera Capture Instructions
### Prerequisites
- Environment has been set up and configured according to COMPLETE_CPP__SETUP_GUIDE.md
- FRAMOS 455e Depth Camera is connected and powered on 
- Computer is on the same subnet as the camera, which is configured to the static IP address 192.168.0.200

### Development
Source code is located in `d400e_cpp/capture_d400e.cpp`

### Build
```
cd d400e_cpp/build
cmake --build . --config Release
```

### Run
```
cd Release
./capture_d400e.exe
```
