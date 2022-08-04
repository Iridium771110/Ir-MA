cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(onnx_ops)
#add_definitions(-std=c++11)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
message ("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -std=c++11")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKERS_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(onnx_ssg test_pnt2_onnx.cpp)

include_directories(/home/dong/WS/onnxruntime-linux-x64-1.10.0/include oct_tree/include kd_tree) # please modify the path to includes
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /home/dong/WS/onnxruntime-linux-x64-1.10.0/lib) # please modify the path to onnxruntime lib

target_link_libraries(onnx_ssg PUBLIC ${ONNXRUNTIME_LIBRARY})