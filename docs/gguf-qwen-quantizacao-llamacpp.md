# GGUF no llama.cpp com foco em Qwen + quantização

Referências ao clone em `.ref/llama.cpp`. Só cobre o que interessa para suportar `.gguf` de modelos Qwen (denso 27B tipo Qwen3.8 e MoE tipo Qwen3.5) e quantizá-los.

## 1. Formato GGUF (o que nosso engine precisa ler)

- `ggml/include/gguf.h` L1-32 — Estrutura binária: magic `"GGUF"`, versão **3**, `n_tensors`, `n_kv`; depois os pares KV (chave string + tipo + valor) e os tensores (nome, n_dims, dims, `ggml_type`, offset no blob). Strings = u64 len + bytes; enums = int32; bool = int8; alinhamento via `general.alignment` (default 32).
- `gguf-py/gguf/constants.py` L1270-1272 — `general.architecture` vale `"qwen35"` (denso) ou `"qwen35moe"` (MoE); é essa string que seleciona arch, tensores e loader.
- Hiperparâmetros seguem o padrão `{arch}.*` (`qwen35.block_count`, `qwen35.embedding_length`, `qwen35.attention.head_count`, ...). Chave obrigatória específica: **`qwen35.rope.dimension_sections`** (default `[11,11,10,0]` injetado pelo conversor — `conversion/qwen.py` L621-634).
- `gguf-py/gguf/constants.py` L2833-2864 — Tensores do `QWEN35`: `token_embd/output_norm/output`, attn GQA (`attn_q/k/v/out` + `attn_q_norm/k_norm/post_norm/gate`), `attn_qkv` fundido, `ffn_gate/down/up`, SSM (`ssm_a/conv1d/dt/norm/beta/alpha/out`) e tensores `nextn_*` (MTP, preservados mas não usados).
- `gguf-py/gguf/constants.py` L2865-2902 — `QWEN35MOE` troca o FFN denso por experts (`ffn_gate_inp`, `*_shexp` shared expert, `*_exp` routed, `ffn_gate_up_exp` fundido).
- `gguf-py/gguf/tensor_mapping.py` L252, L393, L900, L927 — Nomes HF→GGUF do `linear_attn`: `in_proj_qkv→ATTN_QKV`, `in_proj_z`, `in_proj_a→SSM_ALPHA`, `in_proj_b→SSM_BETA`. Atenção: `in_proj_ba` é **qwen3next**, não 3.5 — distinção crítica ao mapear pesos.
- `src/models/qwen35.cpp` L1-120 — Loader C++ do denso (tipos por `n_layer`/`n_embd`, criação dos tensores, QK-norm + post-norm).
- `src/models/qwen35moe.cpp` L149-500 — Grafo MoE (`build_qkvz`, `build_norm_gated`, `build_layer_attn[_linear]`, `build_layer_ffn`).
- `src/llama-arch.h` L46-47 + `src/llama-arch.cpp` L41-42 — Registro `LLM_ARCH_QWEN35/QWEN35MOE` ↔ `"qwen35"/"qwen35moe"`.
- `conversion/qwen.py` L639-646 — Conversão HF→GGUF: `Qwen3_5TextModel` / `Qwen3_5MoeTextModel` (`python convert_hf_to_gguf.py --outtype bf16`).

## 2. Quantização (pipeline e tipos)

- `tools/quantize/README.md` L10-16 — Pipeline em 2 fases: converter para GGUF (F32/BF16) e depois `llama-quantize`; usar **imatrix** para preservar qualidade.
- `tools/quantize/quantize.cpp` L34-75 — Tipos suportados (`QUANT_OPTIONS`): legados (`Q4_0`, `Q8_0`), K-quants (`Q2_K`–`Q6_K`, `Q4_K_M` é o default prático), I-quants (`IQ2_XXS`–`IQ4_XS`), ternários (`TQ1_0/TQ2_0`), `MXFP4_MOE` e `COPY`.
- `ggml/include/ggml.h` L389-434 — Tipos de tensor: `Q4_0..Q8_0`, `Q*_K`, `IQ*`, `TQ*`, **`MXFP4` (39)** e **`NVFP4` (40)** — MXFP4 é o formato nativo dos WMMA do RDNA4.
- `ggml/include/ggml.h` L460-489 — `ggml_ftype`: regra geral `MOSTLY_*` = quantiza tudo **exceto tensores 1-D** (norms ficam F16).
- `src/llama-quant.cpp` L306 — `ffn_gate_inp.weight` (roteador MoE) **nunca é quantizado**.
- `src/llama-quant.cpp` L458-530 — `output.weight` e `token_embd` têm regras próprias por ftype (ex.: Q6_K/Q5_K/Q4_K conforme agressividade) — ajustáveis via `--output-tensor-type`, `--token-embedding-type`, `--leave-output-tensor`.
- `src/llama-quant.cpp` L512 — Em quants agressivos (2-bit), modelos com `n_expert >= 4` ou GQA >= 4 caem para `Q4_K` em vez de 2-bit (proteção de qualidade que afeta MoE).
- `tools/quantize/README.md` L53-67 — Flags finas: `--imatrix`, `--tensor-type "regex=tipo"` (ex.: manter `ssm_*`/experts em precisão maior), `--pure`, `--allow-requantize`, `--override-kv` (ex.: `qwen3moe.expert_used_count`).

## 3. Implicações para o Qwen3.8 27B na 9070 XT (16 GB)

- Ordem de grandeza: `Q4_K_M` (~4,9 bpw) em 27B ≈ **16–17 GB** — no limite/estouro da VRAM; `Q4_0` (~4,3 bpw) ≈ 14,6 GB, `IQ4_XS` (~4,5 bpw) ≈ 15 GB, `Q3_K_M` (~4,0 bpw) ≈ 13,5 GB. Provável necessidade de `Q4_0`/`IQ4_XS`/`Q3_K_M` ou offload parcial.
- `Q8_0` (~8,5 bpw ≈ 29 GB) só com offload pesado para CPU — inviável full-VRAM.
- No HIP/RDNA4, os paths mais maduros são `Q4_0`/`Q8_0` (kernels `MMVQ` tunados — ver `mmvq.cu`); K-quants funcionam, MXFP4 nativo ainda em amadurecimento (ver discussão rocWMMA flash-attn #15021).
- Tensores candidatos a precisão maior via `--tensor-type`: `ssm_*` (estado recorrente do GDN), `attn_q_norm/k_norm`, `output.weight` — pequeno custo em GB, grande ganho em estabilidade.
