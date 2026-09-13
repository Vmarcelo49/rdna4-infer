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

## M1 — Loader GGUF `qwen35` denso ✅ (todos os itens 1-5, ver Progresso)

Objetivo: ler os dois arquivos UD e validar tudo fail-fast.

### Progresso

- **Passo 1 — dtype mapping ✅** (`include/rdna4/dtype.h`, `tests/check_dtype.cpp`): `DType` = union de 15 tipos dos UD, `dtype_from_ggml()` mapeia os ids raw do GGUF, resto é erro. `check-dtype` passa nos dois arquivos (866 tensores, 15/15 tipos presentes no IQ3_S; IQ4_XS sem `IQ2_XXS`/`IQ1_S`).
- **Passo 2 — GgufLoader ✅** (`include/rdna4/loader.h`, `src/backend/loader.cpp`, `tests/check_loader.cpp`): `GgufLoader::open()` = parse GGUF v3 + whitelist + geometria fail-fast (size de cada tensor ≤ span até o próximo, fim alinhado a 32, nomes únicos) + `load_tensor()` copia os bytes quantizados de um tensor p/ host.
  - **Geometria validada nos dois arquivos** (C++ + espelho Python `scripts/check_geometry.py`): todos os 866 tensores de cada arquivo batem exatamente — último termina no EOF (`file_end_exact=yes`), zero overflow, zero desalinhamento.
  - **Fato de formato (crítico p/ o loader):** offsets dos tensores são **relativos ao início da seção de dados** = fim do header **padded a `alignment`** (`GGML_PAD`), confirmado no reader de referência (`ggml/src/gguf.cpp`: `gr.seek(GGML_PAD(tell, alignment)); ctx->offset = tell`). Header ~11 MB (vocab 248320); `data_offset=10996640` nos dois arquivos.
  - **Tabela de block size (elems, bytes)** validada empiricamente nos arquivos reais e cross-check com `llama-gguf r` (b10902): F32(1,4), Q8_0(32,34), Q2_K(256,84), Q3_K(256,110), Q4_K(256,144), Q5_K(256,176), Q6_K(256,210), IQ2_XXS(256,66), IQ2_XS(256,74), IQ3_XXS(256,98), IQ1_S(256,50), IQ4_NL(32,18), IQ3_S(256,110), IQ2_S(256,82), IQ4_XS(256,136). **O snapshot `.ref` (790cf51) está em refactoring e é auto-inconsistente para `IQ1_S` (struct 66 B vs `static_assert` 50 B) — NÃO usar como referência de formato de arquivo.**
  - Spot-loads F32 com valores plausíveis (`attn_norm` ∈ [0.86,1.2], `ssm_a` negativo pequeno) nos dois arquivos — valores idênticos entre os dois UD, como esperado para tensores F32.
- **Passo 3 — Config `qwen35` + inventário por camada ✅** (`include/rdna4/model.h`, `src/backend/model.cpp`, `tests/check_model.cpp`, `tests/gen_fake_qwen35.py`):
  - `parse_qwen35_config()` (item 2): fail-fast em `general.architecture != "qwen35"` ou KV ausente/tipo errado. Conjunto exigido: `block_count`, `full_attention_interval`, `nextn_predict_layers`, `context_length`, `embedding_length`, `feed_forward_length`, `attention.head_count`/`head_count_kv`, `key_length`/`value_length`, `ssm.{conv_kernel,state_size,group_count,time_step_rank,inner_size}`, `rope.dimension_count`, `rope.dimension_sections` (array inteiro — **I32** no arquivo, `[11,11,10,0]`), `layer_norm_rms_epsilon`, `rope.freq_base`, `bos/eos/pad_token_id`.
  - `validate_qwen35_layout()` (item 3): inventário exato por bloco — 48 GDN (14 tensores) quando `(i+1) % interval != 0`, 16 full (11 tensores) quando `(i+1) % interval == 0`, bloco 64 = MTP (conjunto full + 4 `nextn.*`); nomes **e** dims exatos, derivadas dos KVs (ex.: `attn_qkv` GDN = 5·group·state = 10240; `attn_q` full = heads·2·klen = 12288). Top-level: `token_embd`/`output`/`output_norm` (vocab não fixado; consistência emb + shared-V checada). Erro nomeia o tensor (missing / unexpected / dims mismatch).
  - Wired em `rdna4-infer` (toda execução real), `check-loader` e `check-model` (CPU-only, standalone).
  - **Fail-fast provado**: `tests/gen_fake_qwen35.py` gera mini-qwen35 de 2 blocos (dims 32-B aligned); `check-model` aceita o arquivo íntegro e rejeita tensor faltando / dim errada / tensor extra com mensagem específica. Arquivos reais passam nos dois: `65 blocks (16 full-attn, 48 GDN, 1 MTP)`.
  - **Layout ground truth (extraído dos arquivos)**: GDN = `attn_gate [5120,6144]`, `attn_norm [5120]`, `attn_qkv [5120,10240]`, `ffn_down/gate/up`, `post_attention_norm [5120]`, `ssm_a [48]`, `ssm_alpha/beta [5120,48]`, `ssm_conv1d [4,10240]`, `ssm_dt.bias [48]`, `ssm_norm [128]`, `ssm_out [6144,5120]`; full = `attn_k [5120,1024]`, `attn_k_norm [256]`, `attn_norm`, `attn_output [6144,5120]`, `attn_q [5120,12288]`, `attn_q_norm [256]`, `attn_v [5120,1024]`, `ffn_*`; MTP = full + `nextn.eh_proj [10240,5120]`, `nextn.enorm/hnorm/shared_head_norm [5120]`.
- **Itens do M1:** todos ✅ — item 1 (parser binário), item 2 (KVs obrigatórias), item 3 (inventário 48 GDN / 16 full / bloco 64 MTP), item 4 (union de tipos), item 5 (aceite: 866 tensores listados nos dois arquivos). **M1 completo.**

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

### Progresso

- **Passo 1 — oracle CPU de dequant ✅** (`tests/check_dequant.cpp`, `GgufLoader::load_tensor_range`):
  - Oracle = `dequantize_row_*` do **llama.cpp buildado** (`libggml-base.so`, MIT, `extern "C"`) — referência garantida, sem re-derivar matemática. O teste **não** roda GPU/VRAM (CPU puro).
  - **(1) Ponte de structs:** `sizeof(block_X)` (llama.cpp) == `dtype_block_bytes(X)` (tabela M1) para os 14 tipos quantizados — a tabela de bytes do M1 bate com o layout de referência.
  - **(2) Por tipo:** primeiro tensor do tipo nos arquivos reais dequantiza p/ valores finitos e plausíveis (|max| ≤ 32) — `check-dequant` passa nos dois UD (14/14 tipos no IQ3_S; IQ4_XS sem `iq2_xxs`/`iq1_s`).
  - **(3) Pipeline Q8_0 bit-exact:** fórmula trivial inline `d*qs[i]` == oracle → prova o cast raw-bytes→struct + alinhamento.
  - **(4) Cross-file:** mesmo peso, duas quantizações → rel-L2 pequeno (cap 0.3). `output.weight`/`ffn_down` rel-L2≈0 (mesma quant nos dois arquivos); `attn_q` (q2_k vs q5_k) rel-L2=0.21 OK.
  - **Nota:** `block_q*_K` usa **K maiúsculo** no llama.cpp; as grids/LUTs (`iq3s_grid` etc.) vivem em `ggml-common.h`/`ggml-quants.c`. Snapshot `.ref` (790cf51) é incompleto p/ quants (faltam `iq3xs_grid`, `block_q4_K`) — **usar o checkout novo** (`/home/marcelo/Projetos/llama.cpp` @ df03399b8) como fonte de kernels/quants.
- **Passo 2 — dequant GPU vs oracle CPU, por tipo ✅** (`include/rdna4/{quants,dequant.cuh,fp16.h,quant_tables.h}`, `tests/check_dequant_gpu.hip`, `tests/dequant_cpu_oracle.cpp`):
  - **Resultado: 14/14 tipos BIT-EXACT na GPU (gfx1201) contra o oracle CPU do llama.cpp.** IQ3_S: 14 tipos × 16 K elems (q8_0/iq4_nl 2 K). IQ4_XS: 12 tipos presentes × 262 K elems (≈3,1 M elems) — `exact=N/N`, `max|d|=0`.
  - **Kernels:** `dequantize_*` vendorados de `ggml-cuda/dequantize.cuh` (MIT) com adaptação **mecânica apenas**: structs de `quants.h`, `ggml_half`→`uint16_t`, `__low2half/__high2half`→`rdna4::fp16_to_float`, `dst_t`→`float`, `ggml_cuda_cast`→`static_cast<float>`.
  - **Thread mapping por tipo** (é o que faz o kernel escrever o bloco certo): 64 threads → `q2_K,q3_K,q5_K,q6_K`; 32 → `q4_K` + todos os IQ; `q8_0` usa a forma `float2` (16 threads, 2 elems/chamada). Fonte: `getrows.cu` `get_rows_cuda_kq<N, dst_t, dequantize_X>`.
  - **Fonte de verdade = build do llama.cpp** (`libggml-base.so`): TU separada (`dequant_cpu_oracle.cpp`) para os headers do llama.cpp **não** entrarem na TU HIP.
  - **Auditoria de layout de struct (anti-corrupção silenciosa):** `rdna4_audit_layout()` compara `sizeof` + `offsetof` de **todos** os campos nossos vs llama.cpp. Pegou um bug real: `block_q5_K` tem **`qh` antes de `qs`** (mesmo tamanho 176 B, ordem trocada → valores errados). Também corrigido: `(float)x[].d` era cast inteiro→float (nosso `d` é `uint16_t`) — precisava `fp16_to_float`.
  - **Bug de build resolvido:** TU HIP **precisa** da extensão `.hip` (com `.cpp`, o `amdclang++` compila host-only e `__device__`/`float2` não existem).
  - **Tabelas IQ vendoradas** em `quant_tables.h` (grids `iq1s_grid_gpu`/`iq2xxs`/`iq2xs`/`iq2s`/`iq3xxs`/`iq3s`, `kmask_iq2xs`, `ksigns_iq2xs`, `kvalues_iq4nl`) + `IQ1S_DELTA`, `NGRID_IQ1S`.
  - **VRAM:** liberada (hipFree nos dois buffers); pico de alocação por caso = 64–1024 blocos. GPU RX 9070 XT 16 GB, ~15,7 GB livres.
- **Passo 3 (próximo) — matvec fundido (dequant+dot) + tuning RDNA4:**
  - Trazer `vecdotq.cuh` (`vec_dot_*`), `mmvq.cu` + `MMVQ_PARAMETERS_RDNA4`, `mmq.cuh` + `mmq-config-rdna4.cuh`.
  - Teste GPU-vs-CPU: nosso matvec GPU vs. `dequantize_row_*` + dot em f32 (oracle do passo 1)/vs. `ggml` CPU.
  - Aqui a tolerância deixa de ser bit-exact (ordem de acumulação difere) → documentar tolerância.
- **Nota:** GPU/VRAM **liberada** para testes nesta sessão (antes estava proibido). Dequant já executado de verdade em gfx1201, não só compilado.

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
