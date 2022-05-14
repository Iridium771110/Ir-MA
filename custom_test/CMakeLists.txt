cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(onnx_ops_custom_test)
#add_definitions(-std=c++11)
set(HEAD ball_query_kd.h common.h gather_points.h grouping.h sampling.h)

message("head file: " ${HEAD})

add_executable(custom_test ${HEAD} test_custom.cpp)

include_directories(/home/dong/WS/onnxruntime-linux-x64-1.10.0/include)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /home/dong/WS/onnxruntime-linux-x64-1.10.0/lib)

target_link_libraries(custom_test PUBLIC ${ONNXRUNTIME_LIBRARY})