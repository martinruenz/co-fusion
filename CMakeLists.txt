cmake_minimum_required(VERSION 3.1.0)
project(cofusion)

set(CMAKE_CXX_STANDARD 14)

# Warnings / Errors
add_compile_options(-Wall -Werror -Wno-write-strings -Wno-deprecated-declarations)

# Don't follow symlinks when FILE GLOB_RECURSE (and don't warn)
cmake_policy(SET CMP0009 NEW)

set(BUILD_TESTS OFF CACHE BOOL "Build unit tests")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# boost
find_package(Boost REQUIRED)
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# gSLICr
find_package(gSLICr REQUIRED)

# Pangolin
find_package(Pangolin 0.5 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Eigen3
find_package (Eigen3 REQUIRED NO_MODULE)

# project-wide paths
include_directories(${PROJECT_SOURCE_DIR})

# -- CORE
add_subdirectory(Core)

# -- GUI
add_subdirectory(GUI)

# -- show all files
file(GLOB_RECURSE showfiles Core/* GUI/*)
add_custom_target(showthings SOURCES ${showfiles})

# export library
install(EXPORT cofusion
    DESTINATION share/${PROJECT_NAME}/cmake
    FILE ${PROJECT_NAME}Config.cmake
)
