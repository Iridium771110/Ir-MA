cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(registry_custom_ops_onnx)
add_definitions(-std=c++11)

add_executable(test_fps_onnx test_custom_fps_onnx.h test_custom_fps_onnx.cpp)

include_directories(/home/dong/WS/onnxruntime-linux-x64-1.10.0/include) #please modify the path to onnxruntime
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /home/dong/WS/onnxruntime-linux-x64-1.10.0/lib) #please modify the path to onnxruntime

message("include path:" ${ONNXRUNTIME_INCLUDE_DIR})
message("lib path:" ${ONNXRUNTIME_LIBRARY})

target_link_libraries(test_fps_onnx PUBLIC ${ONNXRUNTIME_LIBRARY})
