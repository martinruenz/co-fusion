cmake_minimum_required(VERSION 3.0.2)

project(cofusion)

message(STATUS "Evaluating Core/CMAKE")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

include(GNUInstallDirs)

## External packages
find_package(CUDA REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(Eigen3 REQUIRED NO_MODULE)

if(PCL_FOUND)
    # See: https://github.com/PointCloudLibrary/pcl/issues/776
    get_directory_property(dir_defs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_DEFINITIONS)
    set(vtk_flags)
    foreach(it ${dir_defs})
        if(it MATCHES "vtk*")
        list(APPEND vtk_flags ${it})
        endif()
    endforeach()

    foreach(d ${vtk_flags})
        remove_definitions(-D${d})
    endforeach()

    message(STATUS "With PCL registration.")
    add_definitions(-DWITH_PCL)
    add_definitions(${PCL_DEFINITIONS})
endif()

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

## Options

# Max number of surfels
if(DEFINED COFUSION_NUM_SURFELS)
message(STATUS "COFUSION_NUM_SURFELS=${COFUSION_NUM_SURFELS}")
add_definitions(-DCOFUSION_NUM_SURFELS=${COFUSION_NUM_SURFELS})
endif()

# Show debug visualisations
option(SHOW_DEBUG_VISUALISATION "Show debug visualisation" OFF)
if(SHOW_DEBUG_VISUALISATION)
message(STATUS "SHOW_DEBUG_VISUALISATION enabled")
add_definitions(-DSHOW_DEBUG_VISUALISATION)
endif()

## Directories
find_package(densecrf REQUIRED)
set(cofusion_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Shaders" CACHE PATH "Where the shaders live")

message(STATUS "Directories: ")
message(STATUS "cofusion_SHADER_DIR: ${cofusion_SHADER_DIR}")

## Includes
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${SUITESPARSE_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})


file(GLOB srcs *.cpp)
file(GLOB model_srcs Model/*.cpp)
file(GLOB utils_srcs Utils/*.cpp)
file(GLOB shader_srcs Shaders/*.cpp)
file(GLOB cuda Cuda/*.cu)
file(GLOB containers Cuda/containers/*.cpp)
file(GLOB segmentation Segmentation/*.cpp)

set(CUDA_ARCH_BIN "" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
include(CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-D_FORCE_INLINES;")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-g")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

CUDA_COMPILE(cuda_objs ${cuda})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -msse -msse2 -msse3 -DSHADER_DIR=${cofusion_SHADER_DIR}")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Debug build.")
else(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Release build.")
    add_compile_options(-O3)
    add_compile_options(-Wno-maybe-uninitialized)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

add_library(${PROJECT_NAME} SHARED
    ${srcs}
    ${model_srcs}
    ${utils_srcs}
    ${shader_srcs}
    ${cuda}
    ${cuda_objs}
    ${containers}
    ${segmentation}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ZLIB_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OPENNI2_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${BOOST_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    gSLICr_lib
    densecrf
    ${Pangolin_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${SUITESPARSE_LIBRARIES}
    ${OpenCV_LIBRARIES}
)


set_target_properties(${PROJECT_NAME} PROPERTIES INTERFACE_LINK_LIBRARIES "")

# deactivate 'class-memaccess' error for direct memcpy of list of 'Eigen::Vector4f'
target_compile_options(${PROJECT_NAME} PRIVATE "-Wno-error=class-memaccess")


target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

export(PACKAGE ${PROJECT_NAME})


export(TARGETS ${PROJECT_NAME}
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
)

configure_file(
    "../CMakeModules/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    COPYONLY
)

# install library
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
