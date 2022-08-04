cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(py_ops)
#add_definitions(-std=c++11)
set(SRC ball_query.cpp gather_points.cpp grouping.cpp sampling.cpp register.cpp)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
message ("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O2 -std=c++11")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKERS_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_library(custom_ops SHARED ${SRC})

include_directories(/home/dong/WS/onnxruntime-linux-x64-1.10.0/include )
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /home/dong/WS/onnxruntime-linux-x64-1.10.0/lib)

target_link_libraries(custom_ops PUBLIC ${ONNXRUNTIME_LIBRARY})