cmake_minimum_required(VERSION 3.16)
project(orb_sift_feature_matching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Requires OpenCV >= 4.5 (cv::SIFT lives in the main module from 4.4+).
find_package(OpenCV REQUIRED)

add_library(custom_features STATIC src/custom_orb.cpp src/custom_sift.cpp)
target_include_directories(custom_features PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(custom_features ${OpenCV_LIBS})

add_executable(detect src/detect.cpp)
target_link_libraries(detect custom_features ${OpenCV_LIBS})

add_executable(match src/match.cpp)
target_include_directories(match PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(match ${OpenCV_LIBS})
