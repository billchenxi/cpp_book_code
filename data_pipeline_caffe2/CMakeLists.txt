cmake_minimum_required(VERSION 3.18)
project(caffe2_normalize_demo CXX)
set(CMAKE_CXX_STANDARD 17)

# Point Torch_DIR to the folder that contains TorchConfig.cmake (see build steps below)
find_package(Torch REQUIRED)

add_executable(caffe2_normalize_demo caffe2_normalize_demo.cpp)
target_include_directories(caffe2_normalize_demo PRIVATE "${TORCH_INCLUDE_DIRS}")
target_link_libraries(caffe2_normalize_demo "${TORCH_LIBRARIES}")
