cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(onnx_ops)
include_directories(include)
add_executable(test_oct test_oct_tree.cpp)