# Menções de referência upstream — gfx1201 + família Qwen3.5 (alvo: Qwen3.8 27B na RX 9070 XT)

Clones shallow em `.ref/` (vllm, sglang, hipfire, llama.cpp), sincronizados em 2026-09-13.
Formato: `projeto | arquivo | linhas | por que olhar`. Ranges aproximados (±5 linhas); linhas exatas variam com a revisão.

## vLLM — gfx1201 (suporte RDNA4)

- vLLM | `CMakeLists.txt` | L48-56 | `gfx1200;gfx1201` na lista `HIP_SUPPORTED_ARCHS` do build HIP.
- vLLM | `vllm/platforms/rocm.py` | L76-84 | Mapa PCI→nome: `0x7550` RX 9070 XT e `0x7551` R9700 marcados como `gfx1201`.
- vLLM | `vllm/platforms/rocm.py` | L218-226 | Flags `_ON_RDNA` / `_ON_RDNA4` (gate de paths RDNA4 por arch).
- vLLM | `vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py` | L236-260 | Heurística Triton W4A16 **tunada em gfx1201** (Radeon AI PRO R9700, 32 CUs, wavefront 32) — referência direta de tuning GEMM para nossa placa.
- vLLM | `tests/kernels/core/test_rocm_misc_ops.py` | L80-90 | Teste parametrizado `gfx1201` (RX 9070) — exemplo de como validar kernel em RDNA4.
- vLLM | `docker/Dockerfile.rocm_base` | L33-37 | `PYTORCH_ROCM_ARCH` inclui `gfx1200;gfx1201` na imagem ROCm.
- vLLM | `docker/Dockerfile.rock_base` | L22-26 e L108-114 | Idem no ROCK + comentário de mapeamento GPU→arch.

## vLLM — família Qwen3.5 / Qwen3.8

- vLLM | `vllm/model_executor/models/qwen3_5.py` | L106-120 | `Qwen3_5ProcessingInfo` / `Qwen3_5MoeProcessingInfo` (herdam Qwen3-VL).
- vLLM | `vllm/model_executor/models/qwen3_5.py` | L120-217 | `Qwen3_5DecoderLayer` (herda `Qwen3NextDecoderLayer` — Qwen3.5 reaproveita o decoder Qwen3-Next).
- vLLM | `vllm/model_executor/models/qwen3_5.py` | L217-292 | `Qwen3_5Model` (torre de texto).
- vLLM | `vllm/model_executor/models/qwen3_5.py` | L292-451 | `Qwen3_5ForCausalLMBase` (lógica causal densa/MoE compartilhada).
- vLLM | `vllm/model_executor/models/qwen3_5.py` | L451-473 | `Qwen3_5ForCausalLM` + `Qwen3_5MoeForCausalLM` (registro dos dois sabores).
- vLLM | `vllm/model_executor/models/qwen3_5.py` | L473-750 | `Qwen3_5ForConditionalGeneration` / MoE (variantes multimodais).
- vLLM | `vllm/model_executor/models/qwen3_5_mtp.py` | L67-217 | `Qwen3_5MultiTokenPredictor` (draft head MTP).
- vLLM | `vllm/model_executor/models/qwen3_5_mtp.py` | L217-329 | `Qwen3_5MTP` / `Qwen3_5MoeMTP` (spec-decoding).
- vLLM | `vllm/config/speculative.py` | L900-945 | Fiação do draft `qwen3_5_mtp` / arquiteturas `Qwen3_5MTP` (como o vLLM casa target+draft).
- vLLM | `vllm/models/qwen4_exp/nvidia/ops/qsa.py` | L415-425 | `Qwen3.8-Flash-Next` tunado p/ GB300 — **só NVIDIA**: contraste útil (ainda sem path ROCm p/ 3.8 no vLLM).
- vLLM | `tests/evals/qwen4_exp/configs/Qwen3.8-Flash-Next-FP8.yaml` | L1-10 | Config de eval do Qwen3.8 (molde de YAML modelo+quant).

## SGLang — gfx1201 (suporte RDNA4)

- SGLang | `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp8_utils.cuh` | L8-20 | `hip_fp8.h` + `SGL_ROCM_FP8_HW_CVT` habilitados para `gfx950/gfx1200/gfx1201` (conversão FP8 em HW no RDNA4).

## SGLang — família Qwen3.5 / Qwen3.8

- SGLang | `python/sglang/srt/models/qwen3_5.py` | L322-929 | `Qwen3_5GatedDeltaNet` (atenção linear híbrida — coração da arquitetura).
- SGLang | `python/sglang/srt/models/qwen3_5.py` | L929-1094 | `Qwen3_5LinearDecoderLayer` (camada p/ heads lineares).
- SGLang | `python/sglang/srt/models/qwen3_5.py` | L1094-1550 | `Qwen3_5AttentionDecoderLayer` (camada p/ full attention).
- SGLang | `python/sglang/srt/models/qwen3_5.py` | L1550-1962 | `Qwen3_5ForCausalLM` (modelo causal denso).
- SGLang | `python/sglang/srt/models/qwen3_5.py` | L1962-2173 | `Qwen3_5MoeForCausalLM` (variante MoE).
- SGLang | `python/sglang/srt/models/qwen3_5.py` | L2173-2450 | `Qwen3_5ForConditionalGeneration` / MoE (multimodais).
- SGLang | `python/sglang/srt/models/qwen3_5_text.py` | L41-223 | `Qwen3_5ForCausalLM` texto puro (menor que o multimodal — melhor ponto de partida).
- SGLang | `python/sglang/srt/models/qwen3_5_text.py` | L223-fim | `Qwen3_5MoeForCausalLM` texto MoE.
- SGLang | `python/sglang/srt/models/qwen3_5_mtp.py` | L85-fim | `Qwen3_5ForCausalLMMTP` (draft MTP no SGLang).
- SGLang | `python/sglang/srt/configs/qwen3_5.py` | L7-37 | `Qwen3_5TextConfig` (herda `Qwen3NextConfig`) + `Qwen3_5Config`.
- SGLang | `python/sglang/srt/configs/qwen3_5.py` | L112-138 | Configs MoE (`Qwen3_5MoeTextConfig` / `Qwen3_5MoeConfig`).
- SGLang | `python/sglang/srt/arg_groups/model_overrides/qwen3_5.py` | L1-54 | Overrides de `attention_backend` p/ híbridos Qwen3.5 (ex. triton em SM100 — padrão de override por arch).
- SGLang | `docs/cookbook/autoregressive/Qwen/Qwen3.8.mdx` | L1-40 | Guia day-0 de deploy do Qwen3.8 (MoE GDN/GQA, NVIDIA e AMD).
- SGLang | `test/registered/e2e/models/test_qwen4_exp_models.py` | L1-30 | E2E `Qwen3.8-Flash-Next` (molde de teste de fumaça p/ 3.8).
- SGLang | `test/lm_eval_configs/Qwen3.5-397B-A17B.yaml` | L1-5 | Eval `Qwen/Qwen3.5-397B-A17B` (molde de config de correctness).
- SGLang | `test/registered/amd/perf/mi35x/test_qwen35_fp8_perf_mi35x.py` | L50-95 | Benchmark FP8 do Qwen3.5 em GPU AMD (molde de perf em HW AMD).

## hipfire — gfx1201 (engine RDNA-nativo; o guia mais próximo da 9070 XT)

- hipfire | `crates/rdna-compute/src/rdna/gfx1201.rs` | L17-45 | Consts dos kernels MoE MQ2-Lloyd `..._gfx1201_ep` (gate_up/down + variantes compact/LDS).
- hipfire | `crates/rdna-compute/src/rdna/gfx1201.rs` | L45-62 | `Gfx1201Device` + `try_gfx1201` (detecção/admissão do arch).
- hipfire | `crates/rdna-compute/src/rdna/gfx1201.rs` | L62-129 | Launch `mq2_lloyd_moe_gate_up_compact_ep` (binding de ponteiros p/ kernel gfx1201).
- hipfire | `crates/rdna-compute/src/rdna/gfx1201.rs` | L129-fim | Launch `mq2_lloyd_moe_gate_up_ep` (variante expandida).
- hipfire | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs` | L339-360 | `PREFILL_DEFAULT_BATCH_GFX1201 = 384` — **sweet spot de prefill Qwen3.8 medido em gfx1201** (só arch exato, não gfx1200).
- hipfire | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs` | L410-420 e L722-730 | Gates de batch por arch (`gfx1200|gfx1201` vs demais).
- hipfire | `crates/hipfire-arch-qwen35/src/qwen35/forward.rs` | L3060-3070 | Prerotação V2 condicionada a `is_gfx1201()` (otimização de layout por arch).
- hipfire | `crates/hipfire-arch-qwen35/src/qwen35/forward.rs` | L3331-3360 | `qwen35_fa_epilogue_route_supported` (rota de epílogo FA difere em gfx1201).
- hipfire | `crates/hipfire-arch-qwen35/src/qwen35/forward.rs` | L5560-5570 | `gfx1201_state_fusions_enabled` (gate das fusões de decode-state portadas p/ gfx1201).
- hipfire | `crates/rdna-compute/src/kernels.rs` | L217-303 e L612-681 | Seleção de kernel por arch (`gfx1200|gfx1201` vs demais).
- hipfire | `crates/rdna-compute/src/kernels.rs` | L1126-1140 | `GATED_NORM_MQ_ROTATE_GFX1201_SRC` (kernel gated-norm/MQ exato p/ gfx1201).
- hipfire | `crates/hipfire-arch-deepseek4/src/backend/gfx1201.rs` | L1-162 | Backend gfx1201 de outra arch (molde de como plugar um arch novo no dispatch).

## hipfire — família Qwen3.5 / Qwen3.8 (inclui nosso alvo `qwen3.8:27b`)

- hipfire | `crates/hipfire-arch-qwen35/src/arch.rs` | L46-94 | `struct Qwen35` + `impl Architecture` (registro do arch no runtime).
- hipfire | `crates/hipfire-arch-qwen35/src/qwen35/config.rs` | L107-184 | `Qwen35Config` (hiperparâmetros da torre densa arch-5).
- hipfire | `crates/hipfire-arch-qwen35/src/qwen35/config.rs` | L486-504 | `validate_dense_tp` / `local_dense_tp_config` (sharding TP do denso 27B).
- hipfire | `crates/hipfire-arch-qwen35/src/qwen35/weights.rs` | L25-139 | Pesos `DeltaNet` / `FullAttn` / `Expert` + `mixed_expert_tag`.
- hipfire | `crates/hipfire-arch-qwen35/src/qwen35/weights.rs` | L1088-1277 | `Qwen35Weights` (container de pesos no device).
- hipfire | `crates/hipfire-arch-qwen35/src/qwen35/load.rs` | L2431-2470 | `load_weights` + `HfqSource` (carga do formato `.hfq`).
- hipfire | `crates/hipfire-arch-qwen35/src/qwen35/forward.rs` | L741-800 | `forward` / `forward_from_x(_gpu)` (pontos de entrada do decode).
- hipfire | `crates/hipfire-arch-qwen35/src/qwen35/forward.rs` | L139-280 | `moe_ffn_decode` + prerotação MQ4V2 gate-side.
- hipfire | `registry/models.json` | L702-870 | SKUs **`qwen3.8:27b-*`** (ladder MQ3/MQ4/MQ5/MQ6, tiers xt/base/pro + drafts) — referência exata do nosso modelo.
- hipfire | `README.md` | L125-145 | Tabela de modelos (linha `qwen3.8:27b-mq4` default) + aliases `qwen3.5/3.6/3.8`.
- hipfire | `CHANGELOG.md` | L55-68 | Entrada `Qwen 3.8 27B` (dense arch-5, contexto 262K, `qwen3.8:27b` + `qwen3.8:27b-fast`).
- hipfire | `docs/QUANTIZE.md` | L150-180 | Ladder de quantização da célula Qwen3.8 (tiers xt/base/pro, comando `hipfire-quantize`).
- hipfire | `docs/CONFIG.md` | L240-265 | Thinking config nativa do Qwen3.8 (`enable_thinking`, níveis low/medium/xhigh).

## llama.cpp — gfx1201/RDNA4 (via HIP/CUDA backend)

- llama.cpp | `ggml/src/ggml-cuda/vendors/hip.h` | L211-238 | `__GFX12__` → `#define RDNA4` (detecção de compilação do arch).
- llama.cpp | `ggml/src/ggml-cuda/common.cuh` | L84-93 | `GGML_CUDA_CC_RDNA4` (0x1200, RX 9000) + predicados `IS_RDNA4/IS_RDNA3_5`.
- llama.cpp | `ggml/src/ggml-cuda/common.cuh` | L279-281, L328-351 | Ramos RDNA4 (FP32 compute type, seleção de path).
- llama.cpp | `ggml/src/ggml-cuda/common.cuh` | L718-720 e L763-765 | Guardas `RDNA3||RDNA4` em helpers do backend.
- llama.cpp | `ggml/src/ggml-cuda/mmvq.cu` | L102-130 | `MMVQ_PARAMETERS_RDNA4` (tabela de tuning matvec p/ RDNA4).
- llama.cpp | `ggml/src/ggml-cuda/mmvq.cu` | L258-305 | `get_mmvq_mmid_max_batch_rdna4` (batch ótimo por tipo).
- llama.cpp | `ggml/src/ggml-cuda/mmvq.cu` | L417-470 e L467-495 | Seleção `nwarps=8` p/ vec_dot simples em RDNA4.
- llama.cpp | `ggml/src/ggml-cuda/mmvf.cu` | L840-850 | Ramo `IS_RDNA4` no matvec fused.

## llama.cpp — família Qwen3.5 / Qwen3.8

- llama.cpp | `src/models/qwen35.cpp` | L1-120 | Registro do modelo denso (tipos por `n_layer`/`n_embd`, tensores token/output, camadas attn+GDN).
- llama.cpp | `src/models/qwen35moe.cpp` | L149-260 | `build_arch_graph` + `build_qkvz` / `build_norm_gated` (grafo MoE).
- llama.cpp | `src/models/qwen35moe.cpp` | L278-500 | `build_layer_attn` / `build_layer_attn_linear` / `build_layer_ffn`.
- llama.cpp | `src/llama-arch.h` | L46-47 | `LLM_ARCH_QWEN35` / `LLM_ARCH_QWEN35MOE` (enum).
- llama.cpp | `src/llama-arch.cpp` | L41-42 | Nomes `"qwen35"` / `"qwen35moe"` (tabela arch→string).
- llama.cpp | `src/llama-arch.cpp` | L1081-1110 | `QWEN35` nos predicados de arch híbrida/linear (junto de QWEN3NEXT/QWEN4EXP).
- llama.cpp | `gguf-py/gguf/constants.py` | L505-520 | `QWEN35`, `QWEN35MOE`, `QWEN3NEXT`, `QWEN4EXP` (enum de archs do GGUF).
- llama.cpp | `gguf-py/gguf/tensor_mapping.py` | L250-255 e L390-395 | Tensores `linear_attn.in_proj_qkv/z` marcados `# qwen3.5`.
- llama.cpp | `gguf-py/gguf/tensor_mapping.py` | L900-930 | Tensores `in_proj_a/b` (`# qwen3.5`) vs `in_proj_ba` (`# qwen3next`) — diferença de layout entre 3.5 e Next.
- llama.cpp | `conversion/qwen4exp.py` | L10-30 | Conversor `Qwen3.8-Flash-Next` (herda mixin `_Qwen35MRopeMixin` — reaproveita mrope/GDN do 3.5).
- llama.cpp | `src/llama-model.h` | L130-140 | `LLM_TYPE_A3B // Qwen3.8 Flash Next` (tipo de modelo 3.8).
- llama.cpp | `examples/speculative-simple/README.md` | L8-20 | Receita draft/target com `Qwen3.6-27B-GGUF` (27B como o nosso alvo — molde de spec-decoding).
