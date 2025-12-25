cmake_minimum_required(VERSION 3.18)
project(opencv_build VERSION 0.1.0 LANGUAGES CXX CUDA)

# ================= Paths =================
set(CMAKE_PREFIX_PATH "C:/Program Files/Microsoft MPI")  # MPI path
# Make sure CUDA is in your PATH or installed in default location

# ================= Packages =================
find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${OpenCV_INCLUDE_DIRS})

# ================= CUDA Settings =================
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # Allows multiple .cu files
set(CMAKE_CUDA_FLAGS_INIT "--allow-unsupported-compiler")
set(HEADER_FILES ${YOUR_DIRECTORY}/helper_cuda.h ${YOUR_DIRECTORY}/helper_string.h)


# Optional: specify CUDA architectres
set(CMAKE_CUDA_ARCHITECTURES 75)  # adjust for your GPU

# ================= Executables =================
add_executable(cvTest cvTest.cpp)
add_executable(serial serial.cpp)
add_executable(openmp openmp.cpp)
add_executable(mpi mpi.cpp)
add_executable(cuda cuda.cu)

# ================= Link Libraries =================
target_link_libraries(cvTest PRIVATE ${OpenCV_LIBS})
target_link_libraries(serial PRIVATE ${OpenCV_LIBS})

# OpenMP for MSVC (Windows)
target_compile_options(openmp PRIVATE /openmp)
target_link_libraries(openmp PRIVATE ${OpenCV_LIBS} MPI::MPI_CXX)

target_compile_options(mpi PRIVATE /openmp)
target_link_libraries(mpi PRIVATE ${OpenCV_LIBS} MPI::MPI_CXX)

target_compile_options(cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp>)
target_compile_options(cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(cuda PRIVATE ${OpenCV_LIBS} curand)


# ================= CTest & CPack =================
include(CTest)
enable_testing()
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
