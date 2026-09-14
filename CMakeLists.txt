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
                     src/backend/model.cpp src/backend/unicode.cpp
                     src/backend/unicode_data.cpp src/backend/tokenizer.cpp
                     src/backend/sampler.cpp src/backend/chat.cpp)

# Warnings on the shipping targets (review finding B1: they were enabled nowhere, so
# with no flags at all the build never reported anything). Measured per translation
# unit before turning this on: src/backend/*.cpp 0 warnings, src/main.hip 0,
# serve.hip 0 -- the tree is clean under this set, so anything it prints from now on
# is new. The test binaries are deliberately left out: ROCm 7.2 marks hipError_t
# [[nodiscard]] and their 422 ignored returns are test noise, not engine risk.
set(RDNA4_WARNINGS -Wall -Wextra)
target_compile_options(rdna4-infer PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_compile_options(rdna4-infer PRIVATE ${RDNA4_WARNINGS})
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
# Loader hardening probe: a forged GGUF header must be rejected cleanly instead
# of loading header bytes as weights or aborting in an allocation (audit
# findings M7/M8). Driven by scripts/check_hardening.sh.
add_executable(check-forge tests/check_forge.cpp src/backend/gguf.cpp
                 src/backend/loader.cpp)
target_include_directories(check-forge PRIVATE include)
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

# M4 — tokenizer (vendored unicode tables + BPE) and its oracle comparison.
add_library(rdna4_tokenizer STATIC src/backend/unicode.cpp src/backend/unicode_data.cpp
                                   src/backend/tokenizer.cpp)
target_include_directories(rdna4_tokenizer PUBLIC include)

add_executable(oracle-tokenize tests/oracle_tokenize.cpp)
target_include_directories(oracle-tokenize PRIVATE ${LLAMA_CPP_SRC}/include
                          ${LLAMA_CPP_SRC}/ggml/include)
target_link_libraries(oracle-tokenize PRIVATE ${LLAMA_CPP_BUILD}/bin/libllama.so ${AMDHIP64})
set_target_properties(oracle-tokenize PROPERTIES BUILD_RPATH "${LLAMA_CPP_BUILD}")

add_executable(check-tokenizer tests/check_tokenizer.cpp src/backend/gguf.cpp
                                src/backend/loader.cpp)
add_executable(check-chat tests/check_chat.cpp src/backend/chat.cpp)
target_include_directories(check-chat PRIVATE include)

# oracle-chat renders the metadata chat template with llama.cpp's Jinja engine.
add_executable(oracle-chat tests/oracle_chat.cpp)
target_include_directories(oracle-chat PRIVATE ${LLAMA_CPP_SRC}/include
                          ${LLAMA_CPP_SRC}/common ${LLAMA_CPP_SRC}/ggml/include)
target_link_libraries(oracle-chat PRIVATE ${LLAMA_CPP_BUILD}/bin/libllama-common.so
                                            ${LLAMA_CPP_BUILD}/bin/libllama.so ${AMDHIP64})
set_target_properties(oracle-chat PROPERTIES BUILD_RPATH "${LLAMA_CPP_BUILD}")
target_link_libraries(check-tokenizer PRIVATE rdna4_tokenizer)

# M4 — sampler.
add_executable(check-sampler tests/check_sampler.cpp src/backend/sampler.cpp)
target_include_directories(check-sampler PRIVATE include)

# M3 step 3 — the assembled qwen35 forward graph vs the eval-callback oracle.
add_executable(check-graph-gpu tests/check_graph_gpu.hip
                 tests/dequant_cpu_oracle.cpp src/backend/gguf.cpp src/backend/loader.cpp
                 src/backend/model.cpp)
target_include_directories(check-graph-gpu PRIVATE include /opt/rocm/include
                          ${LLAMA_CPP_SRC}/ggml/src ${LLAMA_CPP_SRC}/ggml/include)
target_link_libraries(check-graph-gpu PRIVATE ${GGML_BASE_LIB} ${AMDHIP64})
target_link_options(check-graph-gpu PRIVATE --offload-arch=gfx1201)
set_target_properties(check-graph-gpu PROPERTIES HIP_ARCHITECTURES "gfx1201")

# M3 step 4 — KV cache types + long-context behaviour.
add_executable(check-kvctx-gpu tests/check_kvctx_gpu.hip
                 tests/dequant_cpu_oracle.cpp src/backend/gguf.cpp src/backend/loader.cpp
                 src/backend/model.cpp)
target_include_directories(check-kvctx-gpu PRIVATE include /opt/rocm/include
                          ${LLAMA_CPP_SRC}/ggml/src ${LLAMA_CPP_SRC}/ggml/include)
target_link_libraries(check-kvctx-gpu PRIVATE ${GGML_BASE_LIB} ${AMDHIP64})
target_link_options(check-kvctx-gpu PRIVATE --offload-arch=gfx1201)
set_target_properties(check-kvctx-gpu PROPERTIES HIP_ARCHITECTURES "gfx1201")

# M3 step 2a — RoPE / attention kernels vs a CPU reference.
add_executable(check-rope-gpu tests/check_rope_gpu.hip)
target_include_directories(check-rope-gpu PRIVATE include /opt/rocm/include)
target_link_libraries(check-rope-gpu PRIVATE ${AMDHIP64})
target_link_options(check-rope-gpu PRIVATE --offload-arch=gfx1201)
set_target_properties(check-rope-gpu PROPERTIES HIP_ARCHITECTURES "gfx1201")

# Test tool — llama.cpp's own argmax for a raw token sequence (the acceptance
# reference for the assembled graph; see tests/oracle_next_token.cpp).
add_executable(oracle-next-token tests/oracle_next_token.cpp)
target_include_directories(oracle-next-token PRIVATE ${LLAMA_CPP_SRC}/include
                          ${LLAMA_CPP_SRC}/ggml/include)
target_link_libraries(oracle-next-token PRIVATE ${LLAMA_CPP_BUILD}/bin/libllama.so ${AMDHIP64})
set_target_properties(oracle-next-token PROPERTIES BUILD_RPATH "${LLAMA_CPP_BUILD}")

# M4: CLI streaming helper (CPU-only, no HIP): the UTF-8 boundary logic that
# keeps the engine from writing half a multibyte character to stdout.
add_executable(check-stream tests/check_stream.cpp)
target_include_directories(check-stream PRIVATE include)

# M4: end-of-generation set (the CLI must stop where llama.cpp stops).
add_executable(oracle-eog tests/oracle_eog.cpp)
target_link_libraries(oracle-eog PRIVATE ${LLAMA_CPP_BUILD}/bin/libllama.so ${AMDHIP64})
add_executable(check-eog tests/check_eog.cpp src/backend/gguf.cpp src/backend/loader.cpp)
target_include_directories(check-eog PRIVATE include)
target_link_libraries(check-eog PRIVATE rdna4_tokenizer)

# M6 — batched matvec (weight sharing across tokens): bit-exactness vs N GEMV
# calls plus the prefill scaling curve per quant type.
add_executable(check-matmul-gpu tests/check_matmul_gpu.hip
               src/backend/gguf.cpp src/backend/loader.cpp)
target_compile_options(check-matmul-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(check-matmul-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(check-matmul-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(check-matmul-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(check-matmul-gpu PRIVATE ${AMDHIP64})

# M7 — attention bench: shipping kernel vs the GQA-sharing prototype, at long
# context (the fix candidate for the 174 ms/token decode step at 64K).
add_executable(bench-mmq-wmma-gpu tests/bench_mmq_wmma_gpu.hip)
target_compile_options(bench-mmq-wmma-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(bench-mmq-wmma-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(bench-mmq-wmma-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(bench-mmq-wmma-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(bench-mmq-wmma-gpu PRIVATE ${AMDHIP64})

add_executable(bench-gemm-quant-gpu tests/bench_gemm_quant_gpu.hip)
target_compile_options(bench-gemm-quant-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(bench-gemm-quant-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(bench-gemm-quant-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(bench-gemm-quant-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(bench-gemm-quant-gpu PRIVATE ${AMDHIP64})

add_executable(bench-gemm-gpu tests/bench_gemm_gpu.hip)
target_compile_options(bench-gemm-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(bench-gemm-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(bench-gemm-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(bench-gemm-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(bench-gemm-gpu PRIVATE ${AMDHIP64})

add_executable(bench-wmma-gpu tests/bench_wmma_gpu.hip)
target_compile_options(bench-wmma-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(bench-wmma-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(bench-wmma-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(bench-wmma-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(bench-wmma-gpu PRIVATE ${AMDHIP64})

add_executable(bench-attn-gpu tests/bench_attn_gpu.hip)
target_compile_options(bench-attn-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(bench-attn-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(bench-attn-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(bench-attn-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(bench-attn-gpu PRIVATE ${AMDHIP64})

# CPU-only checks for the HTTP/JSON half: no model, no GPU, no ROCm.
add_executable(check-server-http tests/check_server_http.cpp src/server/http.cpp
                                 src/server/json.cpp)
target_include_directories(check-server-http PRIVATE include)

# ---------------------------------------------------------------------------
# OpenAI-compatible server (branch feat/openai-server, docs/servidor-openai.md).
#
# Appended by the server workstream: no line above this one was modified
# (docs/agentes-paralelos.md rule 2, "CMakeLists.txt: só append").
#
# Layout: src/server/serve.hip is the HIP translation unit (it owns the model,
# Graph, tokenizer and the generation loop and declares cmd_serve, dispatched
# from src/main.hip); src/server/http.cpp + src/server/json.cpp are plain C++
# (sockets, router, JSON) and carry no HIP at all, which is what lets
# check-server-http exercise the request routing and the JSON codec on CPU.
# ---------------------------------------------------------------------------
# A shared library, not a second object in the executable: graph.cuh (and the
# kernel headers it pulls in) defines its __global__ functions without
# `static`/`inline`, so a second translation unit that includes it collides with
# main.hip's copy at link time ("duplicate symbol: rdna4::mul_kernel"). Linking
# the server as a shared object keeps serve.hip a normal translation unit - the
# alternative, marking 21 kernels across six headers internal, would edit files
# owned by the ROCm workstream.
add_library(rdna4_serve SHARED src/server/serve.hip src/server/http.cpp src/server/json.cpp)
target_include_directories(rdna4_serve PRIVATE include)
target_compile_options(rdna4_serve PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_compile_options(rdna4_serve PRIVATE ${RDNA4_WARNINGS})
target_link_options(rdna4_serve PRIVATE --offload-arch=gfx1201)
target_link_libraries(rdna4_serve PRIVATE ${AMDHIP64})
# PIC for the shared library, explicitly and for BOTH passes. The trap: this header
# set keeps small host-side tables in headers (kv.h's type-name table, the attention
# WPB knob earlier), and a HIP TU that carries a PC32 relocation to one of them from
# its device pass fails the shared link with "recompile with -fPIC" -- which is what
# happened when the new KV types changed which translation unit instantiates the
# table. Making the whole target PIC removes the class instead of chasing tables.
set_target_properties(rdna4_serve PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(rdna4_serve PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fPIC>)
target_link_libraries(rdna4-infer PRIVATE rdna4_serve)

# M8 (estudo ROCm) — bench do matvec por forma real: quanto custa cada tensor do
# inventario por token (nao por tipo) e A/B do knob rows-por-CTA (bit-exato).
add_executable(bench-matvec-shapes-gpu tests/bench_matvec_shapes_gpu.hip
               src/backend/gguf.cpp src/backend/loader.cpp)
target_compile_options(bench-matvec-shapes-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(bench-matvec-shapes-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(bench-matvec-shapes-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(bench-matvec-shapes-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(bench-matvec-shapes-gpu PRIVATE ${AMDHIP64})

# ---------------------------------------------------------------------------
# feat/mtp — the NextN / MTP draft block (blk.64): structure, drafts, the
# acceptance rate and speculative decoding's token-for-token equivalence with
# plain greedy decode. Appended (never inserted) so the parallel-agent merge
# stays trivial.
add_executable(check-mtp-gpu tests/check_mtp_gpu.hip src/backend/gguf.cpp
               src/backend/loader.cpp src/backend/model.cpp src/backend/unicode.cpp
               src/backend/unicode_data.cpp src/backend/tokenizer.cpp
               src/backend/sampler.cpp)
target_compile_options(check-mtp-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(check-mtp-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(check-mtp-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(check-mtp-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(check-mtp-gpu PRIVATE ${AMDHIP64})

# feat/mtp — the llama.cpp oracle for the draft block: runs llama.cpp's own
# LLM_GRAPH_TYPE_DECODER_MTP graph (qwen35.cpp graph_mtp) on the CPU and dumps
# every node of one draft step, in the same format tests/check_graph_gpu.hip
# parses. Captured output goes to reference/oracle_mtp_prompt6_cpu.txt (see
# docs/mtp.md); check-mtp-gpu compares its own nodes against it.
add_executable(oracle-mtp tests/oracle_mtp.cpp)
target_include_directories(oracle-mtp PRIVATE ${LLAMA_CPP_SRC}/include ${LLAMA_CPP_SRC}/ggml/include)
target_link_libraries(oracle-mtp PRIVATE ${LLAMA_CPP_BUILD}/bin/libllama.so ${LLAMA_CPP_BUILD}/bin/libggml.so ${LLAMA_CPP_BUILD}/bin/libggml-base.so ${AMDHIP64})
set_target_properties(oracle-mtp PROPERTIES BUILD_RPATH "${LLAMA_CPP_BUILD}")

# M8 — batched prefill: bit-exactness against the per-token path + the win.
add_executable(check-batch-gpu tests/check_batch_gpu.hip
               src/backend/gguf.cpp src/backend/loader.cpp src/backend/model.cpp
               src/backend/unicode.cpp src/backend/unicode_data.cpp
               src/backend/tokenizer.cpp)
target_compile_options(check-batch-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(check-batch-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(check-batch-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(check-batch-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(check-batch-gpu PRIVATE ${AMDHIP64})

# ---------------------------------------------------------------------------
# Agente de medicoes (worktree rdna4-wt-medicoes-gpu) — orcamento medido por
# fase (eventos HIP no grafo real), pico de leitura de DRAM/L2 desta placa,
# custo de despacho/evento e trafego de KV. Nao e gate: e instrumento de
# medicao (docs/medicoes-banda-e-gargalos.md). Appended, como as outras frentes.
add_executable(bench-phases-gpu tests/bench_phases_gpu.hip
               src/backend/gguf.cpp src/backend/loader.cpp src/backend/model.cpp
               src/backend/unicode.cpp src/backend/unicode_data.cpp
               src/backend/tokenizer.cpp)
target_compile_options(bench-phases-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(bench-phases-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(bench-phases-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(bench-phases-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(bench-phases-gpu PRIVATE ${AMDHIP64})

# Precisao por formato de quantizacao: matvec quantizado (caminho do motor) vs
# matvec fp32 sobre os pesos desquantizados, e o erro da propria quantizacao da
# ativacao (q8_1) - as duas metades do orcamento de erro sem precisar de um
# modelo F16 (docs/quants-precisao.md).
add_executable(check-quant-precision-gpu tests/check_quant_precision_gpu.hip
               tests/dequant_cpu_oracle.cpp
               src/backend/gguf.cpp src/backend/loader.cpp src/backend/model.cpp
               src/backend/unicode.cpp src/backend/unicode_data.cpp
               src/backend/tokenizer.cpp)
target_compile_options(check-quant-precision-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(check-quant-precision-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(check-quant-precision-gpu PRIVATE include /opt/rocm/include
                          ${LLAMA_CPP_SRC}/ggml/src ${LLAMA_CPP_SRC}/ggml/include)
target_link_libraries(check-quant-precision-gpu PRIVATE ${GGML_BASE_LIB} ${AMDHIP64})
# task 6 (autotuning gfx1201) — gate da tabela de tuning.
#
# CPU puro: nao inclui HIP, nao abre modelo, nao toca a GPU (microssegundos), e
# por isso pode rodar em qualquer gate. Ele imprime a configuracao embarcada
# (include/rdna4/tuning.h, a unica fonte dos numeros que matvec.cuh/attn.cuh/
# graph.cuh usam) e compara com a referencia commitada tests/golden/ml_tuning.txt
# -- um refactor que troque um parametro medido quebra aqui, em vez de mudar o
# desempenho em silencio. Ver docs/autotuning-gfx1201.md.
add_executable(check-tuning tests/check_tuning.cpp)
target_include_directories(check-tuning PRIVATE include)

# ---------------------------------------------------------------------------
# task 7 (suite de regressao numerica) — gate de pre-merge.
#
# Roda um conjunto FIXO de prompts pelo caminho real do motor em greedy puro e
# compara os ids gerados (bit-exatos) e os logits do ultimo passo (rel-L2 <= 1e-5)
# contra tests/golden/regression_greedy_f16.txt. Script: scripts/check_regression.sh
# (passa pelo scripts/gpu-lock.sh). Ver docs/autotuning-gfx1201.md §gate.
add_executable(check-regression-gpu tests/check_regression_gpu.hip
               src/backend/gguf.cpp src/backend/loader.cpp src/backend/model.cpp
               src/backend/unicode.cpp src/backend/unicode_data.cpp
               src/backend/tokenizer.cpp src/backend/sampler.cpp src/backend/chat.cpp)
target_compile_options(check-regression-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(check-regression-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(check-regression-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(check-regression-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(check-regression-gpu PRIVATE ${AMDHIP64})

# ---------------------------------------------------------------------------
# Frente KV (worktree rdna4-wt-noite-kv) — formatos de KV Q5_0 e Q4_1 para o alvo
# de 131K, validados contra os quantizadores de referencia do proprio llama.cpp
# (libggml-base exporta quantize_row_q5_0_ref / q4_1_ref e dequantize_row_q5_0 /
# q4_1). Gate: bytes armazenados identicos ao llama.cpp e os dois caminhos de
# load (escalar e o vetorial kv_load8) reconstruindo exatamente o que o
# dequantizador do llama.cpp reconstroi — mais as cotas de erro medidas.
# Appended, como as outras frentes. Com um modelo como argumento ele tambem mede
# o erro sobre linhas K/V reais capturadas dos nos `Kcur`/`Vcur` do grafo.
add_executable(check-kvquant-gpu tests/check_kvquant_gpu.hip
               tests/dequant_cpu_oracle.cpp src/backend/gguf.cpp src/backend/loader.cpp
               src/backend/model.cpp)
target_compile_options(check-kvquant-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(check-kvquant-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(check-kvquant-gpu PRIVATE include /opt/rocm/include
                          ${LLAMA_CPP_SRC}/ggml/src ${LLAMA_CPP_SRC}/ggml/include)
target_link_libraries(check-kvquant-gpu PRIVATE ${GGML_BASE_LIB} ${AMDHIP64})
set_target_properties(check-kvquant-gpu PROPERTIES HIP_ARCHITECTURES "gfx1201")

# Frente KV — porta fechada do achado F6 (docs/adversarial-noite.md): as tres
# tabelas de include/rdna4/kv.h nao podem mais responder "nao sei" com um 0
# silencioso (kv_row_bytes -> passo de linha 0; kv_bytes_per_elem -> orcamento sem
# KV) nem cair no corpo do q4_0 por fallthrough. O teste forka um filho, entrega
# KvType(99) e EXIGE SIGABRT; roda em microssegundos, sem GPU e sem modelo, por
# isso pode entrar em qualquer gate. Appended, como as outras frentes.
add_executable(check-kvtype tests/check_kvtype.hip)
target_compile_options(check-kvtype PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(check-kvtype PRIVATE --offload-arch=gfx1201)
target_include_directories(check-kvtype PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(check-kvtype PRIVATE /opt/rocm/include)
endif()
target_link_libraries(check-kvtype PRIVATE ${AMDHIP64})
set_target_properties(check-kvtype PROPERTIES HIP_ARCHITECTURES "gfx1201")

# Frente KV — sonda de qualidade do cache (o que a PPL NAO mede).
#
# A quantizacao do KV e um erro de CACHE: a NLL agregada de texto natural quase
# nao se move quando o cache piora (na tabela AIME25 do upstream o KV q4_0 cai de
# 37,9% para 2,0% com a PPL andando ~0,4%), entao a PPL nao consegue ordenar dois
# formatos. Esta sonda compara a DISTRIBUICAO de saida do motor com KV f16 contra
# a do formato alvo, nas MESMAS posicoes dos MESMOS tokens, no MESMO processo:
# KL media em nats/posicao sobre todo o vocabulario, fracao de posicoes cujo argmax
# (o token ganancioso) mudou, e |dNLL| pareado. Rodar f16 duas vezes da o piso do
# harness. O stream de tokens vem de `rdna4-infer tokenize -f <corpus>`.
# Appended, como as outras frentes.
add_executable(check-kvquality-gpu tests/check_kvquality_gpu.hip
               src/backend/gguf.cpp src/backend/loader.cpp src/backend/model.cpp)
target_compile_options(check-kvquality-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(check-kvquality-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(check-kvquality-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(check-kvquality-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(check-kvquality-gpu PRIVATE ${AMDHIP64})
set_target_properties(check-kvquality-gpu PROPERTIES HIP_ARCHITECTURES "gfx1201")
# frente kernels de decode (noite 2026-09-14): bench isolado da `delta_rule` do
# GDN. Mede variantes do kernel na mesma cadeia de 100 lançamentos (o protocolo
# de docs/medicoes-banda-e-gargalos.md §4.2), com A/B intercalado, piso de ruído
# do próprio harness e checagem de bit-exatidão contra o kernel que embarca.
# Diário: docs/journal-kernels.md.
add_executable(bench-delta-gpu tests/bench_delta_gpu.hip)
target_compile_options(bench-delta-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(bench-delta-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(bench-delta-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(bench-delta-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(bench-delta-gpu PRIVATE ${AMDHIP64})

# ---------------------------------------------------------------------------
# frente kernels de decode (noite 2026-09-14): bench isolado das normas do
# decode (rms_norm/l2_norm). O `attn_norm`/`post_norm` de 1x5120 custa 10,9 us
# por lançamento contra um piso de despacho de 2,2 us e roda 128x por token.
# Cadeia de 100 lançamentos, A/B antigo vs novo no mesmo harness, piso de ruído
# do próprio harness e memcmp de bit-exatidão. Diário: docs/journal-kernels.md.
add_executable(bench-norm-gpu tests/bench_norm_gpu.hip)
target_compile_options(bench-norm-gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)
target_link_options(bench-norm-gpu PRIVATE --offload-arch=gfx1201)
target_include_directories(bench-norm-gpu PRIVATE include)
if(EXISTS "/opt/rocm/include")
  target_include_directories(bench-norm-gpu PRIVATE /opt/rocm/include)
endif()
target_link_libraries(bench-norm-gpu PRIVATE ${AMDHIP64})
# frente longctx (noite 2026-09-14) — RoPE contra o ggml REAL em posicao longa.
#
# Os dois gates de RoPE que existiam sao cegos para contexto longo:
# tests/check_rope_gpu.hip compara contra uma referencia escrita a mao (mesma
# ordem de operacoes do kernel) em posicoes 3..34, e check-graph-gpu usa um dump
# de prompt de 8 tokens. Este alvo chama ggml_rope_multi/ggml_rope_ext no backend
# CPU do llama.cpp -- as mesmas funcoes que o grafo qwen35 chama -- com os
# parametros que o llama.cpp resolve de verdade para este GGUF, e compara com o
# rope_kernel do motor em posicoes ate 262143. Ver docs/journal-longctx.md.
add_executable(check-rope-long-gpu tests/check_rope_long_gpu.hip)
target_include_directories(check-rope-long-gpu PRIVATE include /opt/rocm/include
                          ${LLAMA_CPP_SRC}/ggml/src ${LLAMA_CPP_SRC}/ggml/include)
find_library(GGML_LIB ggml PATHS ${LLAMA_CPP_BUILD}/bin NO_DEFAULT_PATH)
find_library(GGML_CPU_LIB ggml-cpu PATHS ${LLAMA_CPP_BUILD}/bin NO_DEFAULT_PATH)
if(NOT GGML_LIB OR NOT GGML_CPU_LIB)
  message(FATAL_ERROR "libggml.so/libggml-cpu.so not found under ${LLAMA_CPP_BUILD}/bin")
endif()
target_link_options(check-rope-long-gpu PRIVATE --offload-arch=gfx1201)
target_link_libraries(check-rope-long-gpu PRIVATE ${GGML_LIB} ${GGML_BASE_LIB} ${GGML_CPU_LIB}
                      ${AMDHIP64})
set_target_properties(check-rope-long-gpu PROPERTIES HIP_ARCHITECTURES "gfx1201"
                      BUILD_RPATH "${LLAMA_CPP_BUILD}/bin")
