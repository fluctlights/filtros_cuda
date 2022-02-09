cmake_minimum_required(VERSION 3.17)
project(CUDA_CMAKE)

set(CMAKE_CXX_STANDARD 17)

# find dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include_directories("/usr/include/opencv4")
include_directories("/usr/local/cuda/include")

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET(proyecto_cuda_files main.cpp)

CUDA_ADD_EXECUTABLE(proyecto_cuda ${proyecto_cuda_files} ${hdr} ${cu})
target_link_libraries(proyecto_cuda ${OpenCV_LIBS})
