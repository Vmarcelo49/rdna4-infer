cmake_minimum_required(VERSION 3.21)

# Toolchain: ROCm Clang directo (o wrapper hipcc nao e aceito pelo CMake).
# Source scripts/rocm-env.sh first, ou passe -DCMAKE_HIP_COMPILER=...
if(NOT CMAKE_HIP_COMPILER AND EXISTS "/opt/rocm/bin/amdclang++")
  set(CMAKE_HIP_COMPILER "/opt/rocm/bin/amdclang++")
endif()

project(rdna4-infer LANGUAGES CXX HIP)

# HIP has no default optimization flag in this CMake/ROCm combo: without this
# the device code is built at -O0 (100x+ slower device kernels). The quant
# kernels are NOT compiled with fast-math: bit-exactness vs llama.cpp's CPU
# oracle is a hard requirement of the M2 acceptance tests.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_HIP_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_HIP_FLAGS_DEBUG "-O0 -g")

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

# M2 step 2/3 — GPU dequant vs CPU oracle (gfx1201, real execution).
# The HIP TU holds the device kernels; the CPU oracle bridge is a plain C++
# TU so llama.cpp's headers never enter the HIP compilation.
add_executable(check-dequant-gpu tests/check_dequant_gpu.hip
                 tests/dequant_cpu_oracle.cpp src/backend/gguf.cpp src/backend/loader.cpp)
target_include_directories(check-dequant-gpu PRIVATE include
                          ${LLAMA_CPP_SRC}/ggml/src ${LLAMA_CPP_SRC}/ggml/include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(check-dequant-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(check-dequant-gpu PRIVATE ${GGML_BASE_LIB} ${AMDHIP64})
target_link_options(check-dequant-gpu PRIVATE --offload-arch=gfx1201)
set_target_properties(check-dequant-gpu PROPERTIES
    HIP_ARCHITECTURES "gfx1201"
    HIP_HIPCC_ADDITIONAL_FLAGS "-O2"
)

# M2 step 3 — fused matvec (quantized weights x q8_1 activations) vs CPU.
add_executable(check-matvec-gpu tests/check_matvec_gpu.hip
                 tests/dequant_cpu_oracle.cpp src/backend/gguf.cpp src/backend/loader.cpp)
target_include_directories(check-matvec-gpu PRIVATE include /opt/rocm/include
                          ${LLAMA_CPP_SRC}/ggml/src ${LLAMA_CPP_SRC}/ggml/include)
target_link_libraries(check-matvec-gpu PRIVATE ${GGML_BASE_LIB} ${AMDHIP64})
target_link_options(check-matvec-gpu PRIVATE --offload-arch=gfx1201)
set_target_properties(check-matvec-gpu PROPERTIES HIP_ARCHITECTURES "gfx1201")

# M3 step 1 — device primitives vs the llama.cpp eval-callback oracle.
add_executable(check-nn-gpu tests/check_nn_gpu.hip
                 tests/dequant_cpu_oracle.cpp src/backend/gguf.cpp src/backend/loader.cpp
                 src/backend/model.cpp)
target_include_directories(check-nn-gpu PRIVATE include /opt/rocm/include
                          ${LLAMA_CPP_SRC}/ggml/src ${LLAMA_CPP_SRC}/ggml/include)
target_link_libraries(check-nn-gpu PRIVATE ${GGML_BASE_LIB} ${AMDHIP64})
target_link_options(check-nn-gpu PRIVATE --offload-arch=gfx1201)
set_target_properties(check-nn-gpu PROPERTIES HIP_ARCHITECTURES "gfx1201")

# M3 step 2a — RoPE / attention kernels vs a CPU reference.
add_executable(check-rope-gpu tests/check_rope_gpu.hip)
target_include_directories(check-rope-gpu PRIVATE include /opt/rocm/include)
target_link_libraries(check-rope-gpu PRIVATE ${AMDHIP64})
target_link_options(check-rope-gpu PRIVATE --offload-arch=gfx1201)
set_target_properties(check-rope-gpu PROPERTIES HIP_ARCHITECTURES "gfx1201")
