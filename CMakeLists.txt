cmake_minimum_required(VERSION 2.8)

# use c++ 11
set(CMAKE_CXX_STANDARD 11)

# set project name
project(orb-tracker)

# find OpenCV
find_package(OpenCV REQUIRED)
message("   ")
message(STATUS "version: " ${OpenCV_VERSION})
message(STATUS ${OpenCV_INCLUDE_DIRS})
message(STATUS ${OpenCV_INSTALL_PATH})
message(STATUS ${OpenCV_CONFIG_PATH})
message("   ")

# set include directory
include_directories(include
                    ${OpenCV_INCLUDE_DIRS})

# source files to compile
add_executable(orb-tracker
               src/main.cpp
               src/example.cpp)

# link OpenCV library
target_link_libraries(orb-tracker ${OpenCV_LIBS})