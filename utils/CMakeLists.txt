cmake_minimum_required(VERSION 2.8)
project(utils)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "-std=c++14")
     add_compile_options(-std=c++14 -Wall -Wextra)
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "-std=c++11")
     add_compile_options(-std=c++11 -Wall -Wextra)
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 or C++11 support. Please use a different C++ compiler.")
endif()

find_package (Eigen3 REQUIRED NO_MODULE)
if (TARGET Eigen3::Eigen)
  message("Eigen libs found")
endif (TARGET Eigen3::Eigen)

# Compiler build flags
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package(CUDA 9.1 REQUIRED)

# Pass options to NVCC
list(APPEND CUDA_NVCC_FLAGS "-std=c++14 -use_fast_math")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
# list(APPEND CUDA_NVCC_FLAGS "${MY_NVCC_FLAGS};--ptxas-options=-v")

# Define the arch to compile for
set(GPU_ARCHITECTURE "sm_30,sm_60,sm_61" CACHE STRING "GPU architectures to generate code for (defaults to sm_30, sm_60, sm_61)")

string(REPLACE "," ";" GPU_ARCHITECTURE ${GPU_ARCHITECTURE})
foreach(ARCH IN LISTS GPU_ARCHITECTURE)
    string(REPLACE "sm_" "compute_" COMPUTE ${ARCH})
    set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}
        -gencode=arch=${COMPUTE},code=${ARCH})
endforeach(ARCH)

# Specify include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    include
    )

# Specify library paths
link_directories(
    ${CUDA_LIBRARY_DIRS}
    )

# Output dir for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Specify target & source files 
cuda_add_executable(kernel_occupancy
    src/kernel_occupancy.cu
)

cuda_add_executable(matmul_tests
    src/matmul.cu
    )

# Specify target & libraries to link it with
target_link_libraries(kernel_occupancy
    # ${CUDA_cudart_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_NVCC_FLAGS}
)

target_link_libraries(matmul_tests
    ${CUDA_LIBRARIES}
    ${CUDA_NVCC_FLAGS}

    )
