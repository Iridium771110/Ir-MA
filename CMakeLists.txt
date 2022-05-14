cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(onnx_ops)
#add_definitions(-std=c++11)
set(HEAD ball_query_kd.h common.h gather_points.h grouping.h sampling.h)

message("head file: " ${HEAD})

add_executable(onnx_ssg ${HEAD} test_pnt2_onnx.cpp)

include_directories(/home/dong/WS/onnxruntime-linux-x64-1.10.0/include oct_tree/include kd_tree)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /home/dong/WS/onnxruntime-linux-x64-1.10.0/lib)

target_link_libraries(onnx_ssg PUBLIC ${ONNXRUNTIME_LIBRARY})