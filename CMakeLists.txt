#fDSST_gra
cmake_minimum_required(VERSION 2.8)

# Define project name
project(fDSST_gra)

set( CMAKE_CXX_FLAGS "-std=c++11 -O3 -DUSE_MULTI_THREAD -pthread" )

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
set(OpenCV_DIR /usr/local/opencv340/share/OpenCV)
find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#head file path
include_directories(${PROJECT_SOURCE_DIR}/include)

#source directory
aux_source_directory(${PROJECT_SOURCE_DIR}/src TRACKER_SRC)

#library tracker: my own tracker library
add_library(tracker ${TRACKER_SRC})



add_subdirectory(main)

