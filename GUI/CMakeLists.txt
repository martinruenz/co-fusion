cmake_minimum_required(VERSION 3.5)

project(cofusion-gui)

message("Evaluating GUI/CMAKE")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(ZLIB REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(OpenCV 3.1 REQUIRED )
find_package(Boost 1.58.0 REQUIRED)
find_package(JPEG REQUIRED)

include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Core)
include_directories(${OPENNI2_INCLUDE_DIR})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${BOOST_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

message("Include-directories: ")
message("ZLIB_INCLUDE_DIR: ${ZLIB_INCLUDE_DIR}")
message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message("OPENNI2_INCLUDE_DIR: ${OPENNI2_INCLUDE_DIR}")
message("CMAKE_CURRENT_SOURCE_DIR/../Core: ${${CMAKE_CURRENT_SOURCE_DIR}/../Core}")

file(GLOB srcs *.cpp)
file(GLOB tools_srcs Tools/*.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -msse -msse2 -msse3")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug build.")
else(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release build.")
    add_compile_options(-O3)
    add_compile_options(-Wno-maybe-uninitialized)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

add_definitions(-Dlinux=1)

add_executable(CoFusion
               ${srcs}
               ${tools_srcs}
)

target_link_libraries(CoFusion
                      cofusion
                      ${ZLIB_LIBRARY}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${OPENNI2_LIBRARY}
                      ${OpenCV_LIBRARIES}
                      boost_filesystem
                      boost_system
                      pthread
                      gSLICr_lib # this should be included by target 'cofusion'
                      ${JPEG_LIBRARIES}
)

add_library(CoFusionTools SHARED
            ${tools_srcs})

target_link_libraries(CoFusionTools
                      ${ZLIB_LIBRARY}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${OPENNI2_LIBRARY}
                      ${OpenCV_LIBRARIES}
                      boost_filesystem
                      boost_system
                      pthread
)

target_include_directories(CoFusionTools PUBLIC
                            ${CMAKE_CURRENT_SOURCE_DIR}
                            ${ZLIB_INCLUDE_DIR}
                            ${Pangolin_INCLUDE_DIRS}
                            ${CUDA_INCLUDE_DIRS}
                            ${OPENNI2_INCLUDE_DIR}
                            ${OpenCV_INCLUDE_DIRS}
                            ${BOOST_INCLUDE_DIRS}
)
