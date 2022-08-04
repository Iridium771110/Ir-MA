cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(try_script)

set(Torch_DIR /home/dong/anaconda3/envs/pnt2/lib/python3.7/site-packages/torch/share/cmake/Torch) #please modify the path to libtorch
message("..:..:" ${TORCH_LIBRARIES})
message("include path:" ${CMAKE_INCLUDE_PATH})
message("lib path:" ${CMAKE_LIBRARY_PATH})
set(CUDNN_LIBRARY ~/anaconda3/pkgs/cudnn-8.2.1-cuda11.3_0/lib) #please modify the path to cuDNN
set(CUDNN_INCLUDE_DIR ~/anaconda3/pkgs/cudnn-8.2.1-cuda11.3_0/include) #please modify the path to cuDNN

find_package(Torch REQUIRED)

add_executable(try_script try_script.cpp)

add_executable(test_dataload test_dataload.cpp)

target_link_libraries(try_script "${TORCH_LIBRARIES}")

set_property(TARGET try_script PROPERTY CXX_STANDARD 14)