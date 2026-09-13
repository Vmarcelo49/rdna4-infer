# Kernels de IA no gfx1201 — guia de programação (oficial AMD + relatos da comunidade)

Foco: programar kernels de inferência (GEMM, atenção, MoE) para RDNA4/gfx1201. Complementa `rdna4-gfx1201-referencias-amd.md` (não repete os guias WMMA da GPUOpen).

## Programar kernels (oficial AMD)

- [From Theory to Kernel: FlashAttention-v2 with CK-Tile (ROCm Blogs)](https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html) — Implementação comentada de FlashAttention-v2 em ~100 linhas de CK-Tile (tiling, LDS, pipeline GEMM0/GEMM1).
- [Hands on with CK Tile: building efficient GEMM kernels (ROCm Blogs)](https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html) — Ponto de partida oficial para escrever seu primeiro GEMM em CK-Tile.
- [Optimizing with Composable Kernel (ROCm Docs)](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html) — Como instanciar e tunar GEMMs do CK para shapes de inferência.
- [AITER — AI Tensor Engine for ROCm (GitHub ROCm)](https://github.com/ROCm/aiter) — Biblioteca de kernels Triton/CK da AMD usada por vLLM e SGLang (referência de kernels prontos e de como estruturar os seus).

## Relatos de quem implementou em gfx1201/RDNA4 (comunidade)

- [rdna4-wmma-guide: WMMA lane mapping gfx12 + fused MXFP4 GEMM (GitHub)](https://github.com/JohnTDI-cpu/rdna4-wmma-guide) — Mapeamento lane→tile WMMA no RDNA4 e GEMM MXFP4 fundido a 40.8 TFLOPS na R9700 (gfx1201); corrige o erro clássico de tiles 16x16 transpostos.
- [gfx1201 enablement: rebuilding aiter/flash-attn/vLLM for RDNA4 (r/ROCm)](https://www.reddit.com/r/ROCm/comments/1u3slct/gfx1201_enablement_rebuilding_aiter/) — Receita prática: `GPU_ARCHS=gfx1201` explícito, cache persistente do autotune Triton (~15-30 min), validação e2e com Qwen3.6-35B-A3B-AWQ (repo `patcarter883/rdna4-vllm`).
- [2x R9700 GFX1201 SGLang inference (GitHub)](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference) — `num_kv_splits=64` no decode em gfx1201 para preencher as 64 CUs em contexto longo (override `SGLANG_KV_SPLITS_OVERRIDE`).
- [llama.cpp with native gfx1201 ROCm support (GitHub)](https://github.com/tlee933/llama.cpp-rdna4-gfx1201) — Build com `-DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201` (~99 tok/s); molde mínimo de flags para compilar kernels HIP para o arch exato.
- [Performance of llama.cpp on AMD ROCm (Discussão #15021)](https://github.com/ggml-org/llama.cpp/discussions/15021) — Números de flash-attn via rocWMMA em gfx1201/R9700 e sua sensibilidade a flags (`GGML_HIP_ROCWMMA_FATTN`).
- [RX 9070 llama.cpp benchmarks — The Flash Attention Discovery (r/LocalLLaMA)](https://www.reddit.com/r/LocalLLaMA/comments/1s55b0r/rx_9070_rdna4gfx1201_rocm_721_llamacpp_benchmarks/) — ROCm supera Vulkan em prefill com flash-attn + `-DGGML_CUDA_FORCE_MMQ=ON -DGGML_HIP_GRAPHS=ON` na 9070 (gfx1201).
- [Guide: AI on RDNA4 with ROCm 7.1 + performance tuning (gist)](https://gist.github.com/apollo-mg/ecba6a0c29323325a7ac3babf08e53be) — Trocar backend CK→Triton quando faltam binários pré-compilados p/ RDNA4 (Triton gera kernels JIT para o arch real).
