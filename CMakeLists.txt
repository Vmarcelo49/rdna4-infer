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
add_executable(rdna4-infer src/main.hip src/backend/gguf.cpp src/backend/loader.cpp
                     src/backend/model.cpp)

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

add_executable(check-loader tests/check_loader.cpp src/backend/gguf.cpp
                   src/backend/loader.cpp src/backend/model.cpp)
target_include_directories(check-loader PRIVATE include)
add_executable(check-model tests/check_model.cpp src/backend/gguf.cpp
                 src/backend/loader.cpp src/backend/model.cpp)
target_include_directories(check-model PRIVATE include)

# M2 step 1: CPU dequant oracle — links the reference dequant (llama.cpp
# libggml-base, MIT, extern "C") as ground truth for the GPU-vs-CPU test.
set(LLAMA_CPP_SRC "/home/marcelo/Projetos/llama.cpp" CACHE PATH
    "llama.cpp source tree (oracle headers)")
set(LLAMA_CPP_BUILD "/home/marcelo/Projetos/llama.cpp/build" CACHE PATH
    "llama.cpp build tree (libggml-base.so)")
add_executable(check-dequant tests/check_dequant.cpp src/backend/gguf.cpp
                 src/backend/loader.cpp)
target_include_directories(check-dequant PRIVATE include
                       ${LLAMA_CPP_SRC}/ggml/src ${LLAMA_CPP_SRC}/ggml/include)
find_library(GGML_BASE_LIB ggml-base PATHS ${LLAMA_CPP_BUILD}/bin NO_DEFAULT_PATH)
if(NOT GGML_BASE_LIB)
  message(FATAL_ERROR "libggml-base.so not found under ${LLAMA_CPP_BUILD}/bin — build llama.cpp first")
endif()
target_link_libraries(check-dequant PRIVATE ${GGML_BASE_LIB})
