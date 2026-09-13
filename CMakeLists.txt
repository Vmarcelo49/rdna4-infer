cmake_minimum_required(VERSION 3.21)

# Toolchain: ROCm Clang directo (o wrapper hipcc nao e aceito pelo CMake).
# Source scripts/rocm-env.sh first, ou passe -DCMAKE_HIP_COMPILER=...
if(NOT CMAKE_HIP_COMPILER AND EXISTS "/opt/rocm/bin/amdclang++")
  set(CMAKE_HIP_COMPILER "/opt/rocm/bin/amdclang++")
endif()

project(rdna4-infer LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# v1 targets gfx1201 only (SPEC.md 1.3). No other arch is ever added here.
# --offload-arch em vez de CMAKE_HIP_ARCHITECTURES: funciona em CMake antigo.
add_executable(rdna4-infer src/main.hip src/backend/gguf.cpp)

target_compile_options(rdna4-infer PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(rdna4-infer PRIVATE --offload-arch=gfx1201)

target_include_directories(rdna4-infer PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(rdna4-infer PRIVATE /opt/rocm/include)
endif()

find_library(AMDHIP64 amdhip64 HINTS /opt/rocm/lib REQUIRED)
target_link_libraries(rdna4-infer PRIVATE ${AMDHIP64})

# M1 acceptance checks: plain CXX, CPU-only (no offload flags, no amdhip64).
add_executable(check-dtype tests/check_dtype.cpp src/backend/gguf.cpp)
target_include_directories(check-dtype PRIVATE include)
