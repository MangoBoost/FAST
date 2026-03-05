cmake_minimum_required(VERSION 3.22)
project(fastalltoall
    VERSION 0.1
    DESCRIPTION "FAST NVSHMEM all-to-all extension"
    LANGUAGES CXX CUDA
)

set(CMAKE_CUDA_ARCHITECTURES 90a CACHE STRING "CUDA architecture to target (for example: 90a)")
set(NVSHMEM_ROOT "" CACHE PATH "Optional NVSHMEM installation root")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include(FetchContent)
find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)
find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_ROOT})


add_library(torch_imported INTERFACE)
add_library(torch::py_limited ALIAS torch_imported)
target_include_directories(torch_imported SYSTEM INTERFACE ${TORCH_INCLUDE_DIRS})
target_link_libraries(torch_imported INTERFACE c10 torch torch_cpu c10_cuda torch_cuda CUDA::cudart)

add_compile_options(-Wno-deprecated-declarations)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
add_compile_definitions(Py_LIMITED_API=0x03090000)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)


add_library(flash SHARED
    alltoall_nvshmem.cpp
    ./src/fast_alltoall/alltoall_algorithm.cc
    ./src/fast_alltoall/alltoall_global_scheduler.cc
    ./src/fast_alltoall/alltoall_local_scheduler.cc
    ./src/fast_alltoall/alltoall_matrix.cc
    ./src/fast_alltoall/flash_alltoall_nvshmem.cu
)

target_link_libraries(flash PUBLIC
    torch::py_limited
    Python::Module
    CUDA::cuda_driver
    CUDA::cudart
    nvshmem::nvshmem_host
    nvshmem::nvshmem_device
)
set_target_properties(flash PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    CUDA_SEPARABLE_COMPILATION ON
)
