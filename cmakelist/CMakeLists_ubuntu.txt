cmake_minimum_required(VERSION 3.14)
project(keypointdemo)

set(CMAKE_CXX_STANDARD 11)

add_executable(keypoint src/demo11_5.cpp)

#Opencv库
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(keypoint ${OpenCV_LIBS})