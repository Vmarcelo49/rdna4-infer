# PLAN — rdna4-infer v1 (Qwen3.8 27B denso em GGUF na 9070 XT)

Derivado de `SPEC.md`. Ordem é de dependência — não pular etapas. Cada passo cita a referência exata (`docs/` + `.ref/`).

## M0 — Scaffold + build HIP gfx1201 ✅ feito em 2026-09-13 (`src/main.hip`, `include/rdna4/device.h`)

Objetivo: binário que só existe no mundo gfx1201.

1. Criar o CMake mínimo (só backend HIP, `-DCMAKE_HIP_ARCHITECTURES=gfx1201`).
   Ref: `docs/kernels-ia-gfx1201.md` § "Relatos" (build `tlee933/llama.cpp-rdna4-gfx1201`: flags mínimas comprovadas) · `docs/referencias-upstream-gfx1201-qwen35.md` § "vLLM — gfx1201" (como o upstream declara o arch no build: `CMakeLists.txt` L52, dockers).
2. Implementar detecção de GPU em runtime com recusa de `!= gfx1201`.
   Ref: `docs/referencias-upstream-gfx1201-qwen35.md` § "vLLM — gfx1201" (`rocm.py` L78-84 mapa PCI→gfx1201, L220-226 flags `_ON_RDNA4`) · `docs/rdna4-gfx1201-referencias-amd.md` §3 (matriz de compatibilidade: o que conta como gfx1201).
3. Implementar checagem de VRAM pré-load (tamanho do arquivo + KV por `--ctx-size`, tabela `SPEC.md` §3).
   Ref: `SPEC.md` §1.3 e §3.
4. **Aceite:** binário imprime nome/arch da GPU; aborta com mensagem clara sem gfx1201 ou sem VRAM.

## M1 — Loader GGUF `qwen35` denso

Objetivo: ler os dois arquivos UD e validar tudo fail-fast.

1. Implementar o parser binário (magic `GGUF`, versão 3, KVs, tensores).
   Ref: `docs/gguf-qwen-quantizacao-llamacpp.md` §1 (`gguf.h` L1-32) · `.ref/llama.cpp/ggml/include/gguf.h`.
2. Validar KVs obrigatórias: `general.architecture == "qwen35"`, 65 blocos, `qwen35.rope.dimension_sections == [11,11,10,0]`, ctx 262144.
   Ref: `docs/gguf-qwen-quantizacao-llamacpp.md` §1 (`conversion/qwen.py` L621-634) · `SPEC.md` §1.2.
3. Montar o inventário por camada com os 3 layouts (48 lineares `attn_qkv`+`ssm_*`, 16 full `q/k/v`+norms, bloco 64 com `nextn_*` para ignorar) e os quirks (`ssm_a`, `ssm_dt.bias`, `ssm_alpha/beta` Q8_0, norms F32).
   Ref: `docs/gguf-qwen-quantizacao-llamacpp.md` §1 (listas `constants.py` L2833-2902, mapeamento `tensor_mapping.py`) · `SPEC.md` §1.2.
4. Aceitar exatamente o union de tipos dos arquivos UD (`F32`, `Q8_0`, `Q2/3/4/5/6_K`, `IQ1_S`, `IQ2_XXS/XS/S`, `IQ3_XXS/S`, `IQ4_NL/XS`); resto é erro.
   Ref: `docs/gguf-qwen-quantizacao-llamacpp.md` §2 (`ggml.h` L389-489) · `SPEC.md` §1.2.
5. **Aceite:** carrega `Qwen3.8-27B-UD-IQ3_S.gguf` e lista os 866 tensores (nome/dims/tipo) iguais ao inventário verificado via `gguf-py`.

## M2 — Dequant + GEMM/matvec no HIP

Objetivo: todo tipo do union M1 computando em GPU.

1. Trazer de `.ref/llama.cpp`: `vecdotq.cuh` (dequant), `mmvq.cu` + `MMVQ_PARAMETERS_RDNA4`, `mmq.cuh` + `mmq-config-rdna4.cuh` (inclui `mmq-instance-iq3_s.cu`).
   Ref: `docs/referencias-upstream-gfx1201-qwen35.md` § "llama.cpp — gfx1201/RDNA4" (linhas exatas por arquivo) · `docs/kernels-ia-gfx1201.md` § "Relatos" (`rdna4-wmma-guide`: armadilha dos tiles WMMA transpostos).
2. Cobrir com teste GPU-vs-CPU cada tipo presente nos UD, priorizando `IQ3_S`, `IQ4_XS`, `Q4_K`, `Q8_0`, `Q5_K` (output) e `Q3_K` (embed).
   Ref: `docs/gguf-qwen-quantizacao-llamacpp.md` §2 (regras por tensor: `ffn_gate_inp` nunca quantiza, norms 1-D em F32 — o teste precisa refletir isso).
3. Sem kernel para um tipo → erro explícito, nunca fallback silencioso para CPU (`SPEC.md` §1.3).
4. **Aceite:** teste unitário por tipo passa dentro de tolerância documentada.

## M3 — Grafo forward (prefill + decode com KV cache)

Objetivo: logits corretos nos dois ramos de camada.

1. Implementar o ramo linear GDN (`attn_qkv` + `attn_gate` + SSM conv/recorrência + `ssm_out`).
   Ref: `docs/referencias-upstream-gfx1201-qwen35.md` § "SGLang — Qwen3.5" (`qwen3_5.py` L322-1094: `GatedDeltaNet` + `LinearDecoderLayer`) e § "hipfire — Qwen3.5" (`forward.rs` L741-800 entradas, L139-280 MoE/decode patterns).
2. Implementar o ramo full-attention GQA (`q/k/v` + QK-norm + RoPE/MRoPE + softmax + `attn_output`), RoPE `freq_base 1e7`, `dimension_count 64`, sections `[11,11,10,0]`.
   Ref: mesmo § SGLang (`AttentionDecoderLayer` L1094-1550) · `.ref/llama.cpp/src/models/qwen35.cpp` L1-120.
3. FFN SwiGLU + RMSNorms + `output_norm`/`output` (Q5_K); KV cache incremental com tipos configuráveis (`--cache-type-k/v`: `f16`, `q8_0`, `q4_0` — K/V gravados já quantizados por bloco, atenção desquantiza on-the-fly, como o llama.cpp em `common/arg.cpp` L304-314); prefill em batch, decode token-a-token.
   Ref: `SPEC.md` §1.4 · `docs/gguf-qwen-quantizacao-llamacpp.md` §2 (tipos `Q8_0`/`Q4_0` em `ggml.h`).
4. Começar com `--ctx-size` pequeno; escalar para 32K+ com KV `q8_0`/`q4_0` (131K com `q4_0` cabe no orçamento — verificado no binário M0).
5. **Aceite:** 1 camada linear + 1 full isoladas batem com o llama.cpp HIP (oráculo) dentro de tolerância; run de ctx longo (≥64K, KV `q4_0`) dentro dos 16 GB.

## M4 — Sampler + CLI `run`

Objetivo: gerar texto determinístico com template de chat.

1. Sampler (greedy, temperature, top-k/n, top-p, min-p, repetition penalty, seed) com defaults dos metadados (top_k 20, top_p 0.95, temp 1.0).
   Ref: `SPEC.md` §1.4 · KVs `general.sampling.*` lidas em M1.
2. CLI `run` com streaming e erros em stderr/exit != 0.
   Ref: `SPEC.md` §1.1.
3. `--chat` aplicando `tokenizer.chat_template` (`pre = qwen35`, BOS 248044 / EOS 248046 / PAD 248055).
   Ref: `docs/gguf-qwen-quantizacao-llamacpp.md` §1 · `SPEC.md` §1.1.
4. **Aceite:** golden test — prompt fixo no `UD-IQ3_S`, mesma seed, mesma saída.

## M5 — Validação nos dois arquivos + docs

Objetivo: os dois modelos conversando na 9070 XT.

1. Rodar M4 nos dois `.gguf`; registrar perplexidade/tempo por arquivo.
   Ref: `docs/kernels-ia-gfx1201.md` § "Relatos" (flags que decidem perf em gfx1201: `GGML_HIP_GRAPHS`, `ROCWMMA_FATTN`, `num_kv_splits=64`).
2. Afinamentos gfx1201 de baixo risco (prefill batch, kv splits) só com medição antes/depois.
   Ref: `docs/kernels-ia-gfx1201.md` (blogs CK-Tile para o que for custom) · `docs/rdna4-gfx1201-referencias-amd.md` §6 (profilers).
3. README reproduzível + `SPEC.md` §3 com números medidos.
4. **Aceite:** coerência nos dois arquivos dentro dos 16 GB.

## Fora deste plano (futuro, ver `SPEC.md` §4)

Servidor OpenAI-compatible, `qwen35moe`, MTP/`nextn_*`, mmproj de visão, offload, outros quants/GPUs.
