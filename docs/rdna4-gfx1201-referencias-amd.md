# RDNA4 gfx1201 — Documentação oficial AMD útil para inferência de IA em ROCm

Foco: somente fontes oficiais da AMD (amd.com, `rocm.docs.amd.com`, GPUOpen, blogs oficiais AMD e GitHub `ROCm`).
`gfx1201` = RDNA4 (ex.: RX 9070 / 9070 GRE / 9070 XT, Radeon AI PRO R9700/R9700S/R9600D). `gfx1200` = RDNA4 (ex.: RX 9060/9060 XT).

## 1. Arquitetura RDNA4 / gfx1201 (hardware e ISA)

- [RDNA4 Instruction Set Architecture — Reference Guide (docs.amd.com)](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture) — Referência oficial do ISA RDNA4 (encoding e semântica das instruções, base para kernels de inferência).
- [RDNA4 Instruction Set Architecture (PDF direto)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf) — Versão PDF do ISA RDNA4 para consulta/offline.
- [AMD GPU specifications (ROCm)](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html) — Tabela oficial com CUs, caches, LDS, wavefront e LLVM target (`gfx1200`/`gfx1201`) de cada GPU.
- [AMD GPU architectures (ROCm)](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/index.html) — Índice de microarquiteturas com links para ISAs e white papers por geração.
- [Arquitetura AMD RDNA (amd.com)](https://www.amd.com/en/technologies/rdna.html) — Visão geral oficial da arquitetura RDNA e seus aceleradores.
- [AMD unveils next-generation AMD RDNA 4 (newsroom)](https://www.amd.com/en/newsroom/press-releases/2025-2-28-amd-unveils-next-generation-amd-rdna-4-architectu.html) — Anúncio oficial do RDNA4 (2x ray tracing por CU, AI accelerators de 3ª geração).
- [AI Acceleration with AMD Radeon Graphics (amd.com)](https://www.amd.com/en/products/graphics/radeon-ai.html) — Página oficial de IA em Radeon: RDNA4 com 4x AI compute, 2 AI accelerators por CU, novos tipos de dados e sparsity.

## 2. Matrix Cores / WMMA em RDNA4 (essencial para GEMM e inferência)

- [Using the Matrix Cores of AMD RDNA 4 GPUs (GPUOpen)](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/) — Como usar intrinsics WMMA no RDNA4, com exemplo de inferência de MLP.
- [WMMA guide for AMD RDNA 4 GPUs – part 1 (GPUOpen)](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/) — Guia prático de fusão de GEMMs em RDNA4 com código HIP e verificação via hipBLAS.
- [Matrix Compendium (GPUOpen)](https://gpuopen.com/learn/matrix-compendium/matrix-compendium-intro/) — Compêndio oficial sobre multiplicação de matrizes em GPUs AMD.
- [AMD RDNA Performance Guide (GPUOpen)](https://gpuopen.com/learn/rdna-performance-guide/) — Guia oficial de performance para arquiteturas RDNA.
- [AMD GPU Architecture Programming Documentation (GPUOpen)](https://gpuopen.com/amd-gpu-architecture-programming-documentation/) — Hub oficial com manuais de programação das arquiteturas de GPU AMD.
- [Machine-readable ISAs (GPUOpen)](https://gpuopen.com/machine-readable-isa/) — ISAs AMD em formato legível por máquina (útil para tooling e codegen).
- [Generative AI on AMD Radeon GPUs (GPUOpen)](https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/) — Notas oficiais sobre aceleração de GenAI em GPUs Radeon.

## 3. ROCm e HIP (runtime e modelo de programação)

- [ROCm Documentation Hub](https://rocm.docs.amd.com/en/latest/) — Portal central da documentação oficial do ROCm.
- [HIP Documentation](https://rocm.docs.amd.com/projects/HIP/) — Documentação oficial do HIP (API runtime + linguagem de kernels C++).
- [What is HIP?](https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html) — Definição oficial do HIP e seu papel no ROCm.
- [Introduction to the HIP programming model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html) — Modelo de programação HIP (`hipMalloc`, `hipMemcpy`, lançamento de kernels, filas e sincronização).
- [AMD GPU programming on ROCm](https://rocm.docs.amd.com/en/latest/reference/hip-programming.html) — Guia de programação de GPUs AMD sobre ROCm.
- [Use ROCm on Radeon and Ryzen](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html) — ROCm em RDNA4/Radeon (suporte a PyTorch, inferência local e workstation privada).
- [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) — Matriz oficial: GPUs (`gfx1200`/`gfx1201`), SOs, kernels e drivers suportados.
- [Data types and precision support](https://rocm.docs.amd.com/en/latest/reference/precision-support.html) — Tipos numéricos e precisões suportados por arquitetura e biblioteca ROCm.
- [AMD Radeon and Ryzen system optimization](https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna.html) — Ajustes de sistema recomendados para GPUs Radeon/RDNA.
- [ROCm LLVM Documentation](https://rocm.docs.amd.com/projects/llvm-project/en/latest) — Compiladores LLVM do ROCm (inclui targets `gfx1200`/`gfx1201`).
- [ROCm examples (GitHub ROCm)](https://github.com/ROCm/rocm-examples) — Exemplos oficiais (HIP e bibliotecas) mantidos pela AMD.

## 4. Bibliotecas de computação para inferência (GEMM, atenção, convolução)

- [Composable Kernel (CK)](https://rocm.docs.amd.com/projects/composable_kernel/en/latest) — Biblioteca de kernels fused/tileáveis (GEMM, atenção, normalização) para inferência de alta performance.
- [rocWMMA](https://rocm.docs.amd.com/projects/rocWMMA/en/latest) — API C++ portátil para usar os matrix cores via WMMA.
- [rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest) — BLAS de baixo nível do ROCm (base dos GEMMs de inferência).
- [hipBLAS](https://rocm.docs.amd.com/projects/hipBLAS/en/latest) — Interface BLAS portátil HIP (compatível com cuBLAS para portar código CUDA).
- [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest) — GEMMs leves/fundidos com epílogos para inferência (bias, GELU, quantização).
- [MIOpen](https://rocm.docs.amd.com/projects/MIOpen/en/latest) — Biblioteca de primitivas de deep learning (conv, GEMM, fusões) do ROCm.
- [MIGraphX for Radeon GPUs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-migraphx.html) — Runtime de grafos de inferência da AMD (inclui quantização INT8/INT4 via ONNX).

## 5. Frameworks e runtimes de inferência em ROCm/Radeon

- [PyTorch for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html) — Instalação e uso do PyTorch com ROCm.
- [vLLM on ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html) — Servidor de inferência de LLMs (paged attention, throughput alto).
- [SGLang on ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/sglang.html) — Runtime de serving de LLMs suportado no ROCm.
- [PyTorch for Radeon GPUs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html) — PyTorch específico para GPUs Radeon/RDNA4.
- [ONNX Runtime for Radeon GPUs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-onnx.html) — ONNX Runtime em Radeon (inferência INT8/INT4 com MIGraphX).
- [Triton for Radeon GPUs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-triton.html) — Triton para escrever kernels fused de inferência em Python.
- [JAX for Radeon GPUs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-jax.html) — JAX em Radeon (inferência).
- [Llama.cpp pre-built binaries (Radeon/Linux)](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/advanced/advancedrad/linux/llm/llamacpp.html) — Binários oficiais de llama.cpp para inferência eficiente de LLMs em Radeon.
- [GEMM tuning for model inferencing with vLLM](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/advanced/advancedrad/linux/llm/gemm-tuning.html) — Como tunar GEMMs para inferência com vLLM em Radeon.

## 6. Profiling, debug e monitoramento

- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest) — Profiler de kernels (roofline, contadores) para otimizar inferência.
- [ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest) — Profiler de sistema (CPU+GPU, timelines) para gargalos de serving.
- [AMD SMI](https://rocm.docs.amd.com/projects/amdsmi/en/latest) — Utilitário oficial de monitoramento e controle das GPUs AMD.
- [Radeon Developer Tool Suite (GPUOpen)](https://gpuopen.com/tools/) — Ferramentas oficiais Radeon (RGP, RGA, RMV) para profiling e análise de shaders/kernels.
