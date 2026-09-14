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
- **Passo 3 — matvec fundido (dequant+dot em int8) por tipo ✅** (`include/rdna4/vecdotq.cuh`, `include/rdna4/matvec.cuh`, `tests/check_matvec_gpu.hip`):
  - **Resultado: 14/14 tipos OK** nos dois UD (`check-matvec-gpu`), 8 linhas por tipo, até 248 320 colunas.
  - **Kernels:** `vec_dot_<tipo>_q8_1` vendorados de `ggml-cuda/vecdotq.cuh` (MIT), **subset** aos 14 tipos do union M1 + helpers (`get_int_b1/b2/b4`, `get_int_from_table_16` (path HIP com `__builtin_amdgcn_perm`), `unpack_ksigns`) + emulações HIP de `__vsubss4/__vsub4/__vcmpeq4/__vcmpne4` (de `vendors/hip.h`) + `ggml_cuda_dp4a` com **RDNA4**: `__builtin_amdgcn_sudot4`.
  - **Estrutura do matvec** (espelha o `mul_mat_vec_q` do llama.cpp, caso `ncols_dst=1`, sem fusão): 1 warp por linha; `kqs = vdr*(tid % (qi/vdr))`; `slot = tid/(qi/vdr)`; `blocks_per_iter = vdr*32/qi`; `kby = kb*(qk/QK8_1)`; redução por `__shfl_xor`. Constantes `(qk, qi, vdr)` por tipo vêm de `QI*/QR*` (ggml-common.h) e `VDR_*_MMVQ`.
  - **Ativação:** `quantize_q8_1_block()` — 1 warp quantiza 32 floats → `block_q8_1` (`d = amax/127`, `qs = roundf(x/d)`, `s = fp16(d*Σqs)`), igual ao `quantize_row_q8_1_ref`.
  - **Tolerâncias (documentadas, é o critério de aceite do item 4):**
    - **1e-6** (erro de fp32) para os tipos que usam matemática exata: `q8_0, q2_K, q3_K, q4_K, q5_K, q6_K, iq3_s, iq4_nl, iq4_xs` — medido ~1e-7..4e-7.
    - **1e-3** para os tipos cujo `vec_dot` do llama.cpp usa **escala inteira truncada** (`iq2_xxs`: `sumi*ls/8`; `iq2_xs`/`iq2_s`: `(sumi0*ls0+sumi1*ls1+(sumi0+sumi1)/2)/4`; `iq3_xxs`: `(ls*sumi+sumi/2)/2`; `iq1_s`: termo delta em fp16 `s`) — medido 4e-6..9,4e-5, cresce com o nº de sub-blocos. A referência do teste é matemática exata (dequant + dot em f32), então a diferença **é esperada** e não é bug: o CPU do próprio llama.cpp usa a mesma formulação inteira.
  - **Bugs encontrados nesta etapa (mesma classe do passo 2 — campo fp16 lido como inteiro):** `const float d = bq3_K->d;`, `bq6_K->d` e `bq8_0->d` passados a parâmetro `const float&` (vira temporário inteiro→float, ex. 0x3800 → 14336 em vez de 0,5). Todos corrigidos com `fp16_to_float`. **Lição:** a auditoria de layout **não** pega isso (o offset está certo); o que pega é o teste numérico por tipo.
  - **Hardening pendente (próximo passo pequeno):** trocar o tipo dos campos fp16 por um wrapper `struct fp16 { uint16_t bits; operator float() const; }` para que "esquecer a conversão" vire **erro de compilação** em vez de valor errado silencioso.
- **Passo 4 — bench do matvec (baseline de desempenho) ✅** (`check-matvec-gpu <file> --bench`):
  - **Bug de build crítico encontrado: o código HIP estava sendo compilado em `-O0`** (o combo CMake/ROCm não define flag de otimização para HIP; `HIP_FLAGS` era só `--offload-arch=gfx1201`). Corrigido com `CMAKE_HIP_FLAGS_RELEASE = "-O3 -DNDEBUG"`. **Efeito: 4,4 GB/s → 268 GB/s agregado (62×).** Sem `-ffast-math`: o bit-exact continua garantido (verificado depois do `-O3`, valores idênticos).
  - Bench (IQ3_S, maior tensor de cada tipo, 10 iterações, pico teórico da placa = 644 GB/s):
    | tipo | tensor | GB/s |
    |---|---|---|
    | q5_k | output.weight [5120,248320] | **487** (76% do pico) |
    | q8_0 | blk.64.attn_k.weight | 393 |
    | q4_k | blk.63.ffn_down.weight | 376 |
    | q6_k | blk.64.ffn_down.weight | 290 |
    | iq1_s | blk.0.ffn_up.weight | 231 |
    | iq3_s | blk.2.ffn_down.weight | 200 |
    | q3_k | token_embd.weight [5120,248320] | 195 |
    | iq3_xxs | blk.0.ffn_down.weight | 180 |
    | iq2_s | blk.1.ffn_gate.weight | 155 |
    | iq2_xs | blk.0.ffn_gate.weight | 137 |
    | iq2_xxs | blk.1.ffn_up.weight | 124 |
    | iq4_xs | blk.21.ffn_down.weight | 117 |
    | iq4_nl | blk.11.attn_k.weight (1024 cols) | 90 |
  - **Estimativa de decode (só matvec, soma `bytes/GB/s` por tipo sobre os 866 tensores): 12,02 GB em 68,9 ms → 14,5 tok/s.** Baseline Vulkan/RADV = 37–38 tok/s → **estamos em ~38% do baseline** (a meta é ≥100%).
  - **Onde está o gargalo:** tipos k-quant (q4_k/q5_k/q6_k/q8_0) já vão a 290–487 GB/s; os **IQ dominam o arquivo e ficam em 117–200 GB/s** (mais ALU/lookup por byte: `perm`-based table lookups + escala inteira). `token_embd`/`output` (5120 linhas) têm pouca paralelidade com 1 warp/linha.
- **Passo 5b — diagnóstico do gargalo (onde o tempo realmente vai) ✅**:
  - **Metodologia primeiro:** esta GPU cai para um estado de DPM profundo entre kernels (SCLK observado em 9–16 MHz). Aquecer até ~300 ms de tempo de GPU **dobrou todas as medições** — os números anteriores mediam a rampa de clock, não o kernel. Com isso: agregado 262 → 395–407 GB/s e estimativa de decode 13,6 → **26 tok/s** (3 execuções: 25,6 / 25,8 / 26,7) vs. baseline 37–38.
  - **Onde o tempo vai (12,02 GB, ~39 ms):** `iq4_xs` 32% (196 GB/s), `iq3_s` 26% (341 GB/s), `iq3_xxs` 14% (329 GB/s) → **72% do decode em 3 tipos**.
  - **Teto de leitura (mesma travessia de blocos, lendo TODOS os bytes):** 619 GB/s agregado. O matvec faz 407 → o padrão de acesso **não** é o limite.
  - **Razão matvec/leitura por tipo** (quanto o compute custa além da memória): q8_0 0,9×, q5_k 1,0× (limitados por memória), q6_k 1,7×, q3_k 1,9×, q4_k 2,0×, iq1_s 2,0×, iq2_s 2,0×, iq3_xxs 2,4×, iq4_nl 2,6×, q2_k 3,1×, iq2_xs 3,2×, iq3_s 3,2×, iq2_xxs 5,3×, **iq4_xs 6,5×**.
  - **Causa:** pressão de registradores nos vec_dot dos IQ — `iq4_xs` **119 regs/thread**, `iq2_xs` 109, `iq3_s` 87, `iq3_xxs` 85, `iq2_s`/`iq2_xxs` 83 → cabem ~8–16 warps/CU, então a cadeia de dependência dos lookups não é escondida. Os tipos rápidos usam 43–61 regs.
  - **O que NÃO funcionou (medido):** ILP ±3% (ruído), prefetch L2 neutro/prejudicial, `__launch_bounds__` (minb) ±20% dentro do ruído — e **o próprio ruído entre execuções idênticas é ~20%** (mesma config medida 272 vs 337 GB/s em duas passagens), o que **esgota o tuning de parâmetros**: nenhuma diferença abaixo de ~20% é confiável aqui.
  - **Conclusão:** o ganho tem que vir de **mudar o vec_dot dos IQ** (contagem de instruções por elemento), não de knobs. Direções: manter os valores da ativação em registradores compartilhados entre **várias linhas de peso** por thread (o `rows_per_cuda_block > 1` do mmvq, que hoje é 1 no RDNA4) para não reler a ativação; e reduzir a cadeia dos lookups (`get_int_from_table_16` gasta ~10 ops por 8 valores).
- **Passo 4b — BUG de desempenho crítico encontrado e corrigido (o maior ganho da sessão) ✅**:
  - **`GGML_USE_HIP` nunca foi definido**, então o `get_int_from_table_16` vendorado compilava o **fallback genérico** (o caminho CUDA com `__byte_perm`) em vez do caminho RDNA4 com `__builtin_amdgcn_perm`. Diagnóstico: despejo da ISA (`amdclang++ -S`) do kernel `iq4_xs` mostrou **72 `ds_load_u8`** + 70 `s_wait_alu` — a tabela `kvalues_iq4nl` estava indo para LDS com lookup byte-a-byte.
  - **Correção:** remover o `#if defined(GGML_USE_HIP)` e manter só o corpo HIP (este motor tem um único alvo, gfx1201). Agora a ISA mostra `v_perm_b32` ✓.
  - **Efeito: `iq4_xs` 196 → ~1000 GB/s (5×)**; e a estimativa de decode **~26 → 33,4–35,0 tok/s** (3 execuções), contra o baseline Vulkan/RADV de 37–38 → **~92% do baseline**. `iq4_xs` caiu de 32% para ~9% do tempo de decode; agora o gargalo é `iq3_s` (~35%, 337–375 GB/s).
  - **Estado do matvec (matvec-only):** 12,02 GB em ~28,9 ms → 33,4–35,0 tok/s. O teto de leitura medido (mesma travessia, todos os bytes) é 619 GB/s agregado.
- **Passo 4c — iq3_s: o sign handling era o gargalo; `v_perm_b32` + linearidade do dp4a ✅✅**:
  - **Atribuição por variantes (A/B intercalado, mesmo processo):** um harness `--bench-ab` mede variantes **pareadas** (o ruído entre execuções desta GPU é ~20%, então só medição intercalada é confiável). Resultado em `iq3_s`:
    | variante | GB/s | vs shipping |
    |---|---|---|
    | shipping (`__vcmpne4` + `__vsub4`) | 333–349 | 1,00× |
    | **DIAG sem sign** (errada de propósito) | 745–770 | **2,2×** |
    | DIAG sem lookup do grid | 276–331 | 0,95× |
    | só linearidade (`2·dp4a(g&~m) − dp4a(g)`) | 418–424 | 1,20× |
    | **perm + linearidade** | **689–705** | **2,0×** |
  - **Conclusão da atribuição:** o sign handling custava **>50% do tempo do kernel**; o lookup do grid é praticamente grátis (0,95×). O `__vsub4`/`__vcmpne4` do llama.cpp para HIP são emulações por byte (a ISA mostra dezenas de `v_sub_nc_i16`/`v_lshlrev_b16`).
  - **A otimização (bit-identical!):** um único `V_PERM_B32` por grupo seleciona o byte do grid **ou** um byte zero (índices 0-3 = bytes do grid, 4-7 = dword zero), e a linearidade exata do `dp4a` substitui a negação por byte:
    `Σ (±g)·u = 2·Σ(g & ~m)·u − Σ g·u` → `sumi = 2*sumi_pos − sumi_all`.
    O seletor sai de 4 bits de sinal espalhados para os bytes (`(t * 0x00810204) & 0x04040404 | 0x03020100`), sem `__vcmpne4`.
  - **Ganhos medidos (todos bit-identical, rel-L2 = 0,000e+00):** `iq3_s` **2,02×**, `iq2_xxs` **1,63×**, `iq3_xxs` 1,28×, `iq2_xs` 1,27×, `iq2_s` 1,26×.
  - **Resultado agregado (matvec-only, 12,02 GB):** ~29 ms → **22,3–24,0 ms** ⇒ **41,6–45,0 tok/s** (era 33,4–35,0; baseline Vulkan/RADV **37–38**). **Passamos o baseline** na parte de matvec.
  - **Segunda rodada (seletor enxuto):** a variante `perm` ainda gastava com `unpack_ksigns` (broadcast por multiplicação) + `__vcmpne4` (máscara por byte). Pegando o **nibble de sinais direto** do byte empacotado (`sv ^= (popc(sv)&1)<<7`) e espalhando para os bytes do seletor (`(t * 0x00810204) & 0x04040404`), o custo de sinal praticamente zera — confirmado por diagnóstico: `iq3_xxs` com sinal = 386 GB/s, **sem sinal = 655 GB/s**, e a forma enxuta chega a ~690 GB/s (ou seja, sobrou ~6% para o sinal).
  - **Ganhos finais por tipo (todos bit-identical, `rel-L2 = 0`):**
    | tipo | antes (vendored) | depois (perm2 enxuto) |
    |---|---|---|
    | `iq3_s` | ~350 GB/s | **557–753 GB/s** |
    | `iq3_xxs` | ~300–369 | **532–692** |
    | `iq2_s` | ~259–364 | **455–591** |
    | `iq2_xs` | ~226–310 | **471–488** |
    | `iq2_xxs` | ~164–301 | **456–462** |
    | `iq4_xs` | 196 (bug do perm) → **1008–1021** | (sem sinal; fora desta rodada) |
  - **Resultado agregado (matvec-only, 12,02 GB):** **20,1–20,8 ms ⇒ 48,0–49,7 tok/s** (início desta otimização: 33,4–35,0; baseline Vulkan/RADV **37–38**). Ou seja, **~1,3× acima do baseline** só na parte de matvec.
  - **Bottleneck atual:** `iq3_s` (~25–32% do decode, 557–753 GB/s), `iq3_xxs` (13–17%, 532–692), `iq4_xs` (12–13%, ~1010).
  - **Pesquisa paralela (subagente) — contagem de instruções e alternativas:** o corpo do laço `iq3_s` (1 chamada de vec_dot = 32 elementos/thread) tem **452 instruções**, das quais só **8 são `v_dot4_i32_iu8`** e **~250–280 (55–62%) eram a emulação de bytes** (`__vcmpne4` + `__vsub4`) — batendo com a medição de 2,2× do diagnóstico "sem sinal". O teto por *issue* desse corpo é ~600 GB/s (63% de utilização a 375 GB/s), contra ~1200 GB/s de leitura pura. Confirmações úteis: (a) a saturação do `__vsub4` é **código morto** (todos os bytes de `iq3s_grid` são ímpares ≥ 1), (b) os backends CUDA/SYCL usam a *mesma* emulação (não há forma melhor no checkout), (c) o **Vulkan (baseline 37–38 tok/s) não usa int8 dot nenhum** — faz FMA em fp32 com selects de sinal e mantém o `iq3s_grid` em **LDS**, (d) o llama.cpp já força `nwarps=1` para IQ3_S em RDNA4 ("register pressure and lookup table contention"), (e) o cálculo do índice do grid já está quase mínimo (4 ops + 1 load).
  - **Alternativa testada e rejeitada (registrado para não repetir):** a proposta `(g ^ m) + t` (6 ops/grupo, 1 dp4a em vez de 2, exata — verifiquei exaustivamente: 0 divergências em 512 entradas × 16 nibbles de sinal, com `t` evitando carry entre bytes porque todo byte do grid é ímpar ≥ 1) mede **672–682 GB/s contra 722–731 GB/s** da forma `perm + linearidade` em 3 execuções A/B consistentes. Ou seja: o `dp4a` extra é mais barato que as operações extras, e a forma perm+lin **fica**.
  - **`fp16_to_float` agora usa a conversão de hardware (`V_CVT_F32_F16`)** em device (via `__half2float(__ushort_as_half(h))`), em vez de ~9 instruções de software chamadas 2× por vec_dot → **48,0–49,7 → 50,4–55,8 tok/s** (~+8%, acima dos ~4% previstos). O bit-exact é revalidado pelo próprio teste de dequant (que usa essa conversão em toda escala).
  - **Próximos candidatos (por ordem de evidência):** LUT `iq3s_grid` em **LDS** (provado pelo Vulkan; ataca o gather via L1 — 8 `global_load_b32` por chamada), e a formulação fp32 estilo Vulkan (~130–150 instruções/32 elementos, numericamente exata para este intervalo de valores, mas com 4× mais bytes de ativação).
- **Passo 5 — tuning do matvec: infraestrutura de config pronta, ganho pequeno ⚠️**:
  - Kernel generalizado `matvec_kernel_gen<T, ROWS, WPR>`: `ROWS` linhas por CTA × `WPR` warps por linha (cobre tanto 4 warps/4 linhas quanto o layout 1 linha/8 warps do `mmvq`). Redução intra-warp + shared memory quando `WPR > 1`.
  - `matvec_default_config(dt)` por tipo, escolhido por **medição** (sweep de 8 shapes × 14 tipos, 3 repetições, média). Valores medidos (GB/s) para o maior tensor de cada tipo:
    | tipo | 4x1 | 2x1 | 1x1 | 8x1 | 1x2 | 1x4 | 1x8 | 2x2 | default |
    |---|---|---|---|---|---|---|---|---|---|
    | q8_0 | 399 | 401 | **407** | 395 | 357 | 292 | 193 | 355 | 4x1 |
    | q2_k | 179 | 184 | 187 | 190 | 146 | **191** | 142 | 190 | 2x2 |
    | q3_k | 178 | 176 | 178 | 175 | 183 | 187 | **187** | 180 | 1x4 |
    | q4_k | 321 | 359 | 402 | 366 | 405 | 325 | 212 | **408** | 2x2 |
    | q5_k | 454 | 461 | 468 | 459 | **493** | 490 | 491 | 480 | 1x2 |
    | q6_k | 292 | 273 | **312** | 287 | 281 | 300 | 260 | 281 | 1x1 |
    | iq2_xxs | 124 | 111 | **128** | 123 | 126 | 107 | 100 | 109 | 1x1 |
    | iq2_xs | 130 | 142 | **146** | 127 | 139 | 126 | 113 | 131 | 1x1 |
    | iq3_xxs | 166 | 175 | 182 | **193** | 138 | 111 | 103 | 147 | 8x1 |
    | iq1_s | 236 | 198 | 235 | 234 | 235 | 225 | 229 | **249** | 2x2 |
    | iq4_nl | 95 | 70 | 42 | **99** | 36 | 49 | 48 | 60 | 8x1 |
    | iq3_s | 174 | 208 | **215** | 203 | 169 | 128 | 109 | 149 | 1x1 |
    | iq2_s | 156 | 145 | 151 | **166** | 158 | 126 | 126 | 160 | 8x1 |
    | iq4_xs | 110 | 69 | 37 | **124** | 54 | 70 | 93 | 97 | 8x1 |
  - **Resultado:** agregado 262 GB/s (era 268 com 4x1 fixo) → **13,6 tok/s** de estimativa. **Ou seja: o tuning de shape por tipo rendeu ~0** — a escolha depende mais do *formato da linha* (nº de blocos por linha) do que do tipo, e a maior parte dos 866 tensores tem formato diferente do "maior tensor do tipo" usado no sweep. **Lição registrada:** a tabela precisa ser indexada por (tipo, blocos-por-linha), não só por tipo.
  - **Onde está o gargalo real (hipóteses para o M5):** o loop por thread tem pouca ILP (5–20 iterações, dependência `sum += dot(...)`, sem prefetch) → latência de memória não escondida. Caminhos: acumuladores independentes + unroll, prefetch L2 explícito (o `mmvq_prefetch_l2` do llama.cpp), `ncols_dst > 1` (batch), e kernels específicos para os IQ (mais ALU/byte). O `-O3` já tirou o gargalo artificial de 62×.
  - **Correção de rota registrada:** a regra `nwarps=8` do `MMVQ_PARAMETERS_RDNA4` (llama.cpp, para `ncols_dst=1`) **não** transfere para este kernel: medido q8_0 8x1=395 vs 4x1=399 e q4_k 8x1=366 vs 2x2=408; as regressões que o llama.cpp reporta para Q3_K/IQ são reais aqui também (Q3_K 8x1=175 ≈ 4x1).
  - Depois (M5): `mmq.cuh` + `mmq-config-rdna4.cuh` para prefill (batch > 8; o matvec só cobre batch ≤ 8, igual ao `MMVQ_MAX_BATCH_SIZE`).
- **Revisão adversarial de código (subagente revisor, M1+M2) — corrigido nesta sessão:**
  - **C1 (crítico, no meu próprio teste):** `check_dequant_gpu` dimensionava os buffers pelo bloco ggml de 32 elementos do IQ4_NL, mas `dequant_kernel_256` itera em unidades de **256** elementos (8 sub-blocos: `ibs*(QK_K/QK4_NL)`) → **leitura e escrita 8× fora do buffer** no device, passando silenciosamente (o prefixo comparado estava certo). Corrigido: buffers dimensionados pela unidade do kernel; IQ4_NL agora verifica 16 384 elems (era 2 048) e continua bit-exact.
  - **H1:** a referência CPU do matvec consumia os **próprios bytes de ativação da GPU** → nenhum teste podia pegar bug no `quantize_q8_1_block`. Agora há cross-check contra o `quantize_row_q8_1_ref` do llama.cpp (exportado pelo `libggml-base`): **d, qs e s bit-exact** em todos os tipos.
  - **H2:** `s` era `fp16(d·Σqs)` calculado a partir do `d` já arredondado, com um `sum` float morto. Escolhido e documentado: **igual ao `quantize_row_q8_1_ref`** (bit-exact), que é o que mantém o bloco inteiro verificável (o `mmvq` do llama.cpp usa `make_half2(d, Σx)` — ~1% de diferença, só afeta `iq1_s`).
  - **H4:** o gate de 1e-6 comparava kernel f32 com referência f64; agora a tolerância escala com `sqrt(cols/256)` (documentado), e o teste diz explicitamente que permutação de threads é invisível (só bugs de cobertura mexem no número).
  - **M1/M2:** `parse_qwen35_config` agora **valida valores** (block_count ≥ 2, interval ≥ 1, nextn < block_count, 4 seções MRoPE, `2·Σsections == dimension_count`). **Isto pegou um erro meu:** eu tinha assumido `Σsections == dimension_count`, mas as seções do MRoPE cobrem **metade** dos dims rotativos (32 vs 64) — os arquivos reais eram rejeitados. Os valores exatos ([11,11,10,0]) só são exigidos para o layout real de 65 blocos.
  - **M3/M7:** `prod_dims` com detecção de overflow, dims ≤ 0 rejeitadas, **`ne0 % block_elems == 0` exigido** para tensores quantizados (ggml dimensiona por linha) e tamanho 0 rejeitado.
  - **M4:** `hipMalloc`/`hipMemcpy` verificados e sem vazamento nos `continue` do teste de matvec.
  - **M5:** `kQwen35KvElemsPerToken` contava **17** camadas com KV; são **16** (o bloco MTP não roda na v1) → o orçamento ficou mais preciso (12,78 GiB vs 12,81).
  - **M6/L9:** guards no `check_dequant` + rótulos honestos ("informational") nos gates fracos; auditoria de layout agora cobre `block_q8_1` (o campo `ds` é load-bearing).
  - **L4:** `nextn.eh_proj` = `[2·emb, emb]` (concat(hidden, embed)), não `5·g·d` (coincidem neste modelo).
  - **L5:** `check-dtype` reporta tipos ausentes e tem `--require-all` (o claim "15/15 no IQ3_S" agora é verificável).
  - **Ainda pendente da revisão (não corrigido):** L1 (SPEC cita F16/BF16 fora do whitelist), L2 (limites de string/array no parser GGUF), L3 (`pad_to` defensivo), L6 (tipos ausentes não contam como falha), L7 (tabelas de tuning duplicadas), L8 (`file_end_exact` não aplicado), L10 (sem guard de `ncols % qk`), L11 (tabelas antigas de GB/s no PLAN — **esta seção atualiza**), L12 (nits).
- **Nota:** GPU/VRAM **liberada** para testes nesta sessão (antes estava proibido). Dequant **e** matvec já executados de verdade em gfx1201, não só compilados.


1. Trazer de `.ref/llama.cpp`: `vecdotq.cuh` (dequant), `mmvq.cu` + `MMVQ_PARAMETERS_RDNA4`, `mmq.cuh` + `mmq-config-rdna4.cuh` (inclui `mmq-instance-iq3_s.cu`).
   Ref: `docs/referencias-upstream-gfx1201-qwen35.md` § "llama.cpp — gfx1201/RDNA4" (linhas exatas por arquivo) · `docs/kernels-ia-gfx1201.md` § "Relatos" (`rdna4-wmma-guide`: armadilha dos tiles WMMA transpostos).
2. Cobrir com teste GPU-vs-CPU cada tipo presente nos UD, priorizando `IQ3_S`, `IQ4_XS`, `Q4_K`, `Q8_0`, `Q5_K` (output) e `Q3_K` (embed).
   Ref: `docs/gguf-qwen-quantizacao-llamacpp.md` §2 (regras por tensor: `ffn_gate_inp` nunca quantiza, norms 1-D em F32 — o teste precisa refletir isso).
3. Sem kernel para um tipo → erro explícito, nunca fallback silencioso para CPU (`SPEC.md` §1.3).
4. **Aceite:** teste unitário por tipo passa dentro de tolerância documentada.

## M3 — Grafo forward (prefill + decode com KV cache)

Objetivo: logits corretos nos dois ramos de camada.

### Progresso

- **Oráculo escolhido: `llama-eval-callback`** (o `qwen35.cpp` do llama.cpp marca cada nó com `cb()`, e a ferramenta imprime nome, shape, 3 primeiros + 3 últimos valores e a **soma** de cada nó). `scripts/capture_oracle.sh` captura o dump (`-ngl 0`, seed/temp fixos → determinístico) em `reference/` (gitignored, ~4 MB). Isso dá um oráculo por nó — muito melhor que re-derivar semântica do source.
- **Passo 1 — primitivas de device validadas contra o oráculo ✅** (`include/rdna4/nn.cuh`, `tests/check_nn_gpu.hip`):
  - `check-nn-gpu <gguf> <dump> [token]` reproduz no GPU os 2 primeiros nós do grafo a partir do **token real** (9419 = "Hello"): linha do `token_embd.weight` (Q3_K, `GgufLoader::load_tensor_range` + dequant) → `rms_norm` → `mul` por `blk.0.attn_norm.weight`.
  - **Resultado: `norm-0` e `attn_norm-0` batem com o oráculo** — os 6 valores amostrados coincidem até a 4ª decimal e a soma relativa difere **2,5e-06** / **2,3e-06**.
  - Semânticas fixadas no caminho (todas lidas do fonte de referência, não adivinhadas): `RMSNorm` = `x * rsqrt(mean(x²) + eps)` com o `eps` do GGUF e o peso em `MUL` separado; `L2 norm` do GDN = `x / sqrt(Σx² + eps)` (de `build_gdn_l2_norm` = `scale(rms_norm(x, eps/n), 1/√n)`); `silu`/`sigmoid`/`softplus` (com o clamp `x>20` do ggml); **RoPE do qwen35 = `LLAMA_ROPE_TYPE_IMROPE`** (de `llama_model_rope_type`), `n_rot=64` dims com `freq_base=1e7`; as seções `[11,11,10,0]` só importam para posições distintas (imagem/áudio), então para texto as 4 posições são iguais.
  - ⚠️ **Correção (Passo 3): "pares adjacentes" estava errado.** O `ggml_compute_forward_rope_f32` roteia NEOX/**MROPE**/**IMROPE** por `rotate_pairs(n_dims, n_dims/2, ...)`: o elemento `i` rotaciona com `i + n_rot/2` (**split-half**), com `cache[i0]` ainda pertencendo ao par `i0/2` (ângulo `pos·freq_base^(-2p/n_rot)` inalterado). Pares adjacentes só valem para `GGML_ROPE_TYPE_NORMAL`.
  - `Qwen35Config` agora carrega `rms_norm_eps` e `rope_freq_base` (o parser já lia, mas descartava).
- **Passo 2a — RoPE + atenção causal ✅** (`include/rdna4/attn.cuh`, `tests/check_rope_gpu.hip`):
  - `rope_kernel` (split-half, `angle = pos·freq_base^(-2p/n_rot)`, primeiros `n_rot` dims de cada head) e `attn_kernel` (causal, 1 bloco por head, softmax em shared memory).
  - Validados contra implementação CPU direta da semântica documentada: **rel-L2 2,7e-08 (RoPE)** e **5,7e-08 (atenção)**. ⚠️ Esse teste é **auto-consistente** (a referência CPU foi escrita a partir da mesma leitura do ggml), então ele não pega erro de *convenção* — os dois erros de convenção abaixo só apareceram contra o oráculo do grafo completo.
- **Dimensões da arquitetura decodificadas dos shapes do oráculo** (não de suposição): `head_dim = 256`, `n_head = 24`, `n_head_kv = 4` (GQA 6×), `n_rot = 64` (**só 64 dos 256 dims rodam**), `scale = 1/√256 = 1/16`; `attn_q` = 24 × **2** × 256 = 12288 (q e gate **intercalados por head**, stride `2·head_dim`), `attn_k/v` = 4 × 256 = 1024, `attn_output` = 24 × 256 = 6144.
- **Oráculo multi-token capturado**: `reference/oracle_prompt6_cpu.txt` (8 tokens: "Hello world, this is a test." → ids `9419 1814 11 411 369 264 1228 13`). Necessário porque com 1 token a posição é 0 (RoPE = identidade) e a atenção é trivial — nenhum dos dois validaria nada.
- **Limite de validação descoberto:** o dump imprime apenas 6 valores + a soma por nó, então **não dá para semear uma camada isolada** com a entrada real dela (a entrada da camada N vem do grafo). A validação do M3 é portanto **incremental**: montar o grafo e comparar `l_out-N` (amostras + soma) camada a camada, fechando em `result_norm` e `result_output`.
- **Passo 3 — grafo montado e validado ✅** (`include/rdna4/graph.cuh`, `tests/check_graph_gpu.hip`, `tests/oracle_next_token.cpp`): ver a seção "Passo 3" abaixo.
- **Passo 2 (histórico) — ramo full-attention**: `attn_q` (com gate intercalado por head, `view_3d` com stride 2×head_dim), `attn_k`/`attn_v`, QK-norm, RoPE/MRoPE, GQA + KV cache, `sigmoid(gate)` multiplicando a saída da atenção, `attn_output`; validar contra os nós `Qcur-N`/`Kcur-N`/`attn_pregate-N`/`attn_gated-N`/`attn_output-N` do dump.


### Passo 3 — grafo montado (M3 ✅)

**O que existe agora**

- `include/rdna4/graph.cuh`: `rdna4::Graph` — upload de todos os pesos dos 63 blocos executáveis para VRAM (bytes quantizados consumidos direto pelo matvec; f32 para norms/viéses), laço de camadas token-a-token, ramo GDN (conv causal com estado rolante + regra delta + norm gateada) e ramo full-attention (split q/gate, QK-norm, RoPE, KV cache f32, `sigmoid(gate)`), FFN `down(silu(gate)·up)` com o residual do `post_attention_norm` (o FFN **não** tem residual na entrada: soma-se ao tensor de *antes* do norm), `output_norm` e LM head. `set_node_cb()` expõe cada nó intermediário com **o mesmo nome do `cb()` do llama.cpp**, que é o que torna a comparação por nó possível.
- `tests/check-graph-gpu <gguf> <dump|-> <argmax_ref|-> [ids...]`: compara nó a nó contra o dump e roda os *structural checks* + o argmax.
- `tests/oracle_next_token.cpp` (`oracle-next-token`): argmax do llama.cpp para uma sequência **crua** de ids (o dump não imprime argmax). `ORACLE_NGL=99` roda no backend Vulkan, `ORACLE_STEP=1` decodifica token-a-token (mede a variância do próprio llama.cpp entre caminhos: **0,078 no logit top-1**, ou seja ~0,4%, contra 2,1–2,6 do caminho em batch).
- `scripts/capture_oracle.sh` agora também salva `reference/argmax_<nome>_<backend>.txt`; `scripts/extract_dump_block.py` extrai o bloco de **um** token de uma captura `-ub 1`.

**Metodologia (o que foi preciso mudar no caminho)**

1. **O oráculo tem que ser capturado com `-ub 1` para comparar com este motor.** Com prefill em batch (8 tokens) o CPU do llama.cpp usa outro caminho de `MUL_MAT`, e os nós do *próprio* llama.cpp ficam 4–5% diferentes dos do caminho token-a-token (`attn_norm-3` do token 0, mesma entrada). Como o nosso motor faz um matvec por token (caminho de decode), a referência correta é a captura com `-ub 1` (`reference/oracle_prompt6_ub1_cpu.txt` + `reference/oracle_prompt6_ub1_tok7_cpu.txt` = bloco do último token). O teste detecta dump em batch pelo shape (`{5120, 8, 1, 1}`) e **pula** os structural checks com aviso.
2. **Sensibilidade medida antes de acusar ruído.** Injetando perturbação por camada (`GRAPH_NOISE`, ver abaixo) o logit top-1 varia **≤0,3 mesmo com ±30% por camada** e o *std* dos logits não se move: ou seja, desvio do oráculo **não** se explica por ruído de quantização acumulado. Isso foi o que obrigou a procurar bug em vez de aceitar o desvio — e achou dois.
3. **Amostra não serve para nós com amostras pequenas:** o dump imprime 4 decimais, então o *sample metric* tem piso ~1e-4; os checks descontam esse slack antes de normalizar.

**Os dois bugs reais que a validação pegou**

- **RoPE com pares adjacentes** (`include/rdna4/attn.cuh`): deveria ser **split-half** (`i` com `i + n_rot/2`), como em `rotate_pairs(n_dims, n_dims/2, …)` para IMROPE. Sintoma: `Qcur-3` com métrica de amostra **1,57** (agora 0,16). Curiosamente era **benigno para os scores de atenção** (a mesma rotação aplicada em q e k preserva o produto interno), então não mudava o texto gerado — mas deixava `Qcur-N`/`Kcur-N` errados.
- **GQA com agrupamento errado + índice da query fora do buffer** (`attn_kernel`): usávamos `kvh = h / (n_head/n_head_kv)`… não: usávamos `h % n_head_kv` (agrupamento "tile"), quando o correto para este checkpoint é o **contíguo** `h / (n_head/n_head_kv)` (convenção HF/`repeat_kv`); e a query era indexada como `q[t*n_head + h]` quando o buffer `d_attnout_` contém **um único token** — para `t ≥ 1` isso **lia fora da alocação**. Sintoma: `attn_pregate-3` com soma rel **3,2e-02** (agora **1,05e-04**) e argmax errado em 1 de 3 prompts. É o único bug até agora que realmente muda a resposta do modelo.

**Evidência numérica (oráculo por token, `-ub 1`, último token do prompt de 8)**

- Prefixo determinístico: `model.input_embed` soma rel **8,6e-08**; `attn_norm-0` **2,6e-08**; `beta-0`/`gate-0`/`a_softplus-0` ≤ 1,1e-07.
- Camada 0 (GDN): todas as somas ≤ 3,3e-03 (`linear_attn_qkv_mixed` 1,5e-03, `conv_output_silu` 2,1e-03, `final_output` 1,5e-04, `l_out-0` 3,2e-04) e o **estado recorrente** `new_state-0/1/2` com erro absoluto ≤ 6,1e-05 (o estado atravessa conv1d + L2-norms + regra delta + decay).
- Camada 3 (primeira full-attention): `Qcur_full-3` 6,2e-03, `attn_pregate-3` **1,05e-04**, `attn_output-3` 5,2e-03, `l_out-3` 3,1e-02.
- Precisão do matvec medida contra **f32 exato** (`GRAPH_EXACT=1`, dequant no host + dot em double): `||gpu-exact||/||exact|| = 1,1e-03` para `blk.0.attn_qkv`; o **próprio oráculo CPU fica 2,4× pior** (rms 1,6e-02 vs 6,7e-03 nas mesmas 6 linhas) — a nossa quantização de ativação (q8_1, blocos de 32) é mais fina que a q8_K dele (blocos de 256).
- **Aceite end-to-end: argmax igual ao do llama.cpp em 4/4 prompts** (`check-graph-gpu <gguf> - reference/argmax_<p>_cpu.txt <ids>`):

  | prompt | ids | llama.cpp top-1 | nós top-1 | top-5 (ordem) |
  |---|---|---|---|---|
  | "Hello world, this is a test." | `9419 1814 11 411 369 264 1228 13` | 198 @ 13,5299 | **198 @ 13,5186** | 5/5 |
  | "The capital of France is" | `760 6511 314 9338 369` | 11751 @ 17,5278 | **11751 @ 17,5793** | 5/5 |
  | "def fibonacci(n):" | `727 73111 1393 1590` | 198 @ 20,0289 | **198 @ 19,9924** | 5/5 |
  | "1 2 3 … 9" | `16 220 17 … 24` | 220 @ 16,4062 | **220 @ 16,3145** | 5/5 |

  *(números pós-fix de contagem de camadas; antes dele os "nossos" eram 12,41/14,19/16,16/19,07 — a tabela antiga ficou registrada no commit `f525b27`.)* O `std` dos logits também acompanha: 2,091 vs 2,088 do oráculo neste prompt.

**Passo 4 — tipos de KV cache + contexto longo ✅** (`include/rdna4/kv.h`, `tests/check_kvctx_gpu.hip`)

- **Tipos**: `KvType::{F32,F16,Q8_0,Q4_0}` (o default do llama.cpp é `f16`; o plano pedia `q8_0`/`q4_0` para ctx longo). As linhas do cache (uma cabeça KV de um token) são gravadas **já quantizadas** em blocos de 32 elementos, **byte a byte como o llama.cpp** (`quantize_row_q8_0_ref` e `quantize_row_q4_0_ref` — inclusive o detalhe do q4_0: escala `d = max_sinalizado / -8` e arredondamento `(int8_t)(x*id + 8.5)` com `MIN(15,·)`, não uma grade simétrica `amax/7`), e a atenção **desquantiza on-the-fly** (`kv_load<CT>`), sem cópia f32 do cache. `kv_store_row_launch` grava uma linha; `kv_fill_launch` preenche caches inteiros (hook de teste). A byte-exatidão é **testada** contra um espelho host das fórmulas do ggml em `check-rope-gpu` (`kv store (tipo): bytes differing from the ggml reference: 0`).
- **Teste isolado por tipo** (`check-rope-gpu`): para cada tipo, grava as linhas com `kv_store_row_launch` e confere que a atenção (1 chave ⇒ softmax = 1) devolve **exatamente** o que o cache contém — `max|out - cache| = 0,000e+00` nos 4 tipos ✓.
- **fim-a-fim**: os 4 tipos dão o **mesmo argmax** do llama.cpp no prompt de 8 tokens ✓.
- **Atenção reescrita (flash-style, softmax online)**: a versão anterior guardava 1 score por chave em shared memory + uma redução de bloco por chave, o que (a) limita `t` a alguns milhares de tokens (smem!) e (b) é lento. A nova: 1 bloco por cabeça, 8 warps dividindo as chaves, max/soma correntes e merge por warp no fim — sem smem proporcional ao contexto. `check-rope-gpu` valida a causal (rel-L2 6,1e-08) e o caso de 1 chave.
- **Continuação bit-exata**: `Graph::forward(emb, start_pos, ...)` continua o mesmo KV cache e o mesmo estado recorrente; `check-kvctx-gpu` compara **8 tokens numa chamada** contra **1+3+4 tokens em três chamadas** → hidden e logits **idênticos** ✓ (é o que o CLI vai precisar para prefill em blocos).
- **Contexto longo (aceite M3 item 5)** — `check-kvctx-gpu <gguf> <ctx> <tipo>`, cache preenchido com padrão determinístico e um passo de decode no fim do contexto:

  | ctx | KV | caches | VRAM em uso | decode no fim |
  |---|---|---|---|---|
  | 65536 | q4_0 | 576 MiB | **12,73 GiB** (3,19 livres) | 262 ms/token |
  | 65536 | f16 | 2,0 GiB | 15,16 GiB (0,76 livres) | 192 ms/token |
  | 65536 | f32 | 4,3 GiB | **não aloca** (`hipMalloc failed`) | — |
  | 131072 | q4_0 | 1,1 GiB | **13,51 GiB** (2,41 livres) | 432 ms/token |

  Ou seja: **64K e 131K com KV `q4_0` cabem nos 16 GB** (f16 cabe apertado em 64K; f32 não cabe) — exatamente o que o plano previa. O tempo de decode no fim do contexto é dominado pela atenção O(ctx): 262 ms/token em 64K, 432 ms/token em 131K (a atenção ainda não é tiled — item de performance).

**Bug de contagem de camadas (achado no passo 4, corrigido)**: `Graph::n_layer()` era `block_count - 1 - nextn` = **63**, mas `block_count` cobre o tronco **mais** o bloco MTP, então o tronco é `block_count - nextn` = **64** blocos (0..63; full-attention em i = 3,7,…,63 = 16 camadas — o que o `validate_qwen35_layout` já dizia). A camada 63 nunca rodava: o LLM ainda acertava o argmax (a rede é robusta, como o estudo de perturbação mostrou), mas os logits ficavam ~15–25% comprimidos. Depois do fix: **logit top-1 dentro de 0,02–0,09 do llama.cpp em 4/4 prompts**, top-5 **5/5 na mesma ordem**, desvio padrão dos logits 2,091 vs 2,088 do oráculo, e `result_output` (token 7) com soma rel **4,2e-03** e amostras dentro de 1,2%. Os *structural checks* agora incluem `l_out-63`/`result_norm`/`result_output`, então uma contagem de camadas errada falha o teste (`CHECK MISSING l_out-63`) — foi verificado revertendo o bug.

**Validação cruzada no outro arquivo UD (IQ4_XS)**: o mesmo motor, sem nenhum ajuste, dá **o mesmo argmax e top-5 5/5 na mesma ordem** do llama.cpp nos dois prompts testados (`9419 1814 11 411 369 264 1228 13` → 198; `760 6511 314 9338 369` → 11751), com logits dentro de 0,02–0,10 (ex.: top-1 17,9728 vs 17,9491) e *std* 2,124 vs 2,126 / 2,066 vs 2,062. Ou seja: a validação não depende do mix de tipos do IQ3_S.

**Estatística honesta da comparação por nó** (dump por token, último token, 1219 nós comparados): **15 nós com valores amostrados desviando >20%** (todos em camadas tardias: `l_out-{58..63}`, `attn_residual-{60..63}`, `Qcur_full-59`, `alpha/a_softplus-54`, com desvio **absoluto** 0,29–0,73 sobre valores de 5–15, isto é 2–6%) e 843 nós com erro de **soma** >1e-2 — mas a soma é um critério inútil para os ~metade dos nós cujos valores cancelam (`Qcur` pós-RoPE chega a soma rel 14×). Por isso o teste reporta os dois critérios separados e só o de valores alimenta os checks.

**Deriva residual (documentada, não "resolvida"):** depois do fix de camadas o desvio é pequeno mas não zero: nós profundos ficam tipicamente **1–10%** (`attn_norm-62` 7,9e-02 de diferença máxima nas amostras, `l_out-63` 5,7e-02, `result_output` 1,2e-02) e os logits batem dentro de ~0,1. Atribuição medida: (a) o llama.cpp não é reprodutível melhor que ~4% por nó entre os seus próprios caminhos (batched vs per-token), (b) o nosso matvec é **2,4× mais preciso** que o dele contra f32 exato (q8_1/blocos de 32 vs q8_K/blocos de 256), (c) perturbar 30% por camada não muda o argmax e muda o top-1 em ≤0,3. Ou seja: o resíduo é ruído de quantização de ativação amplificado por 64 camadas, não operação errada.

### Passo 5 — revisão adversarial do M3 (agente separado, só leitura de código)

Revisão contra `qwen35.cpp`/`delta-net-base.cpp`/`ggml-cpu/ops.cpp`, revisão do commit `d2db5c1`. **Nenhum achado crítico**; 4 achados "major", 6 "minor", 4 "nit". Todos os itens acionáveis foram corrigidos:

| achado | correção |
|---|---|
| M1 `q4_0` do cache KV **não** era o do ggml (grade simétrica `amax/7` em vez de `d = max/-8` com o máximo *sinalizado*) — a alegação "byte-idêntico" era falsa | encoder reescrito a partir do `quantize_row_q4_0_ref`; novo teste de byte-exatidão contra espelho host das fórmulas do ggml (com o bug reintroduzido: 489 bytes diferem → MISMATCH) |
| M2 `scripts/capture_oracle.sh` não sabia capturar com `-ub 1`, e um dump em batch era **pulado** deixando `PASS` | `UB=<n>`/`ORACLE_STEP=1` no script, sufixo nos nomes dos arquivos, detecção de batch pelo shape (`N > 1`) e **falha** (não skip) em `check-graph-gpu` |
| M3 `exact_check` lia fora do vetor com `GRAPH_EXACT=1`+`GRAPH_LAST_TOKEN=1` (indexava `n_tokens-1` com só 1 token acumulado) | índice derivado do tamanho acumulado; o caminho de 6 amostras do oráculo em batch é pulado quando só há 1 token |
| M4 nenhum tensor consumido como f32 tinha o dtype checado (norm weights, `ssm_a`, `ssm_dt`, conv1d) — F16 passaria silenciosamente; `ssm_a` de 1 elemento lia fora do buffer | `up_f32()` obrigatório para os papéis f32 + `ssm_a.dim0 == nvh` |
| M5 `proj()` confiava nos argumentos (nrows/ncols) e `head_dim()` usa `key_length` enquanto o layout valida `value_length` | `proj()` confere `dim0/dim1/bytes` do tensor; `key_length == value_length` exigido no `init()`; `validate_qwen35_layout()` chamado pelo próprio `Graph::init` |
| m6/m7 sem validação de `n_head % n_head_kv == 0` e de `n_rot <= head_dim` | ambos validados no `init()` |
| m8 o caminho multi-warp do merge nunca era testado (só `t ≤ 7` ⇒ só o warp 0) | `check-rope-gpu` passa a usar **32 tokens** (merges com 8 fatias não vazias + caudas ímpares) |
| m9 `kv_fill_kernel<Q4_0>` tinha corrida read-modify-write nos nibbles | cada thread < 16 calcula os dois nibbles do seu byte |
| m10 `argv[3]` NULL com `argc == 3` | guarda `argc < 4` |
| m11 batch vazio aceito (logits velhos) e embedding com cauda parcial | falha explícita |
| n12 o meio do grafo (camadas 4..61) só era pinado pelos nós finais | checks em `attn_norm`/`l_out` das camadas 10/20/30/40/50 |
| n15 tabela de aceite com números pré-fix e alegação de byte-exatidão errada | corrigidos acima |

Também corrigido por conta própria: `softplus` usava `log1pf(expf(x))` onde o ggml usa `logf(1 + expf(x))` (diferença ≤1e-8, mas é o que se compara). **Não corrigidos (conscientes)**: (n13) o teste isolado de atenção é auto-consistente por construção — o oráculo do grafo é a validação de convenção; (n14) sem fallback de embeddings amarrados (`output.weight` ausente é erro barulhento); o kernel de atenção continua sem tiling (performance, não correção — 262 ms/token em 64K).

**Knobs de diagnóstico** (usados para o estudo acima, mantidos por serem baratos): `GRAPH_NOISE=<rel>` (+`GRAPH_NOISE_COHERENT=1`) perturba a saída de cada camada antes do residual; `GRAPH_EXACT=1` roda o check f32-exato da primeira projeção; `GRAPH_SAMPLES=1` imprime os 6 valores (oráculo vs nós) de cada nó divergente; `GRAPH_LAST_TOKEN=1` compara só o último token (obrigatório com dump `-ub 1`); `diag.hidden_pre_norm` reporta RMS/min/max do estado residual que entra no `output_norm`.

1. Implementar o ramo linear GDN (`attn_qkv` + `attn_gate` + SSM conv/recorrência + `ssm_out`).
   Ref: `docs/referencias-upstream-gfx1201-qwen35.md` § "SGLang — Qwen3.5" (`qwen3_5.py` L322-1094: `GatedDeltaNet` + `LinearDecoderLayer`) e § "hipfire — Qwen3.5" (`forward.rs` L741-800 entradas, L139-280 MoE/decode patterns).
2. Implementar o ramo full-attention GQA (`q/k/v` + QK-norm + RoPE/MRoPE + softmax + `attn_output`), RoPE `freq_base 1e7`, `dimension_count 64`, sections `[11,11,10,0]`.
   Ref: mesmo § SGLang (`AttentionDecoderLayer` L1094-1550) · `.ref/llama.cpp/src/models/qwen35.cpp` L1-120.
3. ✅ FFN SwiGLU + RMSNorms + `output_norm`/`output` (Q6_K no IQ3_S); **KV cache incremental com tipos configuráveis** (`KvType::{F32,F16,Q8_0,Q4_0}`, gravado já quantizado por bloco de 32, atenção desquantiza on-the-fly — `include/rdna4/kv.h`); decode token-a-token (prefill em blocos via `Graph::forward(..., start_pos, ...)`, bit-exato).
   Ref: `SPEC.md` §1.4 · `docs/gguf-qwen-quantizacao-llamacpp.md` §2 (tipos `Q8_0`/`Q4_0` em `ggml.h`).
4. ✅ 64K (e 131K) com KV `q4_0` cabem nos 16 GB e rodam; f16 cabe apertado em 64K, f32 não cabe (tabela no Passo 4).
5. ✅ **Aceite cumprido:** 1 camada GDN + 1 full-attention batem com o oráculo por nó (`attn_pregate-3` soma rel 1,1e-04; estado recorrente ≤ 6,1e-05); o grafo completo dá o **mesmo argmax do llama.cpp em 4/4 prompts** e top-5 **5/5 na mesma ordem** com logits dentro de ~0,1; run de ctx longo (64K/131K, KV `q4_0`) dentro dos 16 GB.

## M4 — Sampler + CLI `run` ✅ concluído em 2026-09-13

Objetivo: gerar texto determinístico com template de chat.

### Passo 1 — tokenizer BPE (pré-requisito que o plano não listava)

`include/rdna4/tokenizer.h` + `src/backend/tokenizer.cpp`, com `unicode.h`/`unicode.cpp`/`unicode_data.*` vendorizados de `llama.cpp/src/unicode.cpp` + `unicode-data.cpp` (MIT) — só o que o `pre = qwen35` usa (`unicode_regex_split_custom_qwen35`, byte-encoding GPT-2, `unicode_tolower`, as tabelas de ranges/categorias), mais um wrapper novo `unicode_regex_split()`. Pipeline = `llm_tokenizer_bpe`: split nos tokens especiais (`USER_DEFINED` sempre, `CONTROL`/`UNKNOWN` com `parse_special`) → regex do qwen35 → byte-encode → merges por rank.

- Oráculo: `tests/oracle_tokenize.cpp` (libllama) vs `tests/check_tokenizer.cpp`.
- `tests/tokenizer_corpus.txt` (205 linhas) + `tests/golden/tokenizer_ids.txt` commitados; além do corpus, **3000 strings aleatórias → 0 divergências de id e 0 falhas de roundtrip** (`decode(encode(s)) == s`).
- Dois defeitos de teste encontrados e corrigidos no caminho: (a) o oráculo rodava com `vocab_only = true`, então os tensores não eram carregados e o `llama-tokenize` devolvia ids **errados** (1814 → 1206); (b) `llama_tokenize` devolve `-n` quando o buffer é curto e o oráculo tratava o negativo como "o tamanho", alocando 0 tokens → o teste passava com **0 comparações** (vacuidade); agora `need = -need` e `check-tokenizer` **falha** se faltar linha no oráculo.

### Passo 2 — sampler

`include/rdna4/sampler.h` + `src/backend/sampler.cpp`: cadeia e semântica por estágio copiadas de `src/llama-sampler.cpp` (penalties → top_k → top_p → min_p → temperature → dist), com as fórmulas exatas (`logit <= 0 ? logit*penalty : logit/penalty`; top_k por `partial_sort`; top_p via softmax com max subtraído e o menor prefixo ≥ p; `min_p` com `max + log(min_p)`; `temp <= 0` ⇒ greedy; dist = softmax + 1 sorteio uniforme). RNG próprio (`std::mt19937_64` + uniforme de 53 bits) e **deliberadamente não** o stream do llama.cpp — está documentado no header: a comparação direta com o llama.cpp é feita no modo greedy, que não usa RNG.

`tests/check_sampler.cpp` (`check-sampler`): 20+ asserções sobre a cadeia, os defaults dos metadados (`top_k 20`, `top_p 0.95`, `temp 1.0`), o greedy e o determinismo por seed. Dois bugs de teste achados: usar o **valor** de `max_element` como índice e assertar a penalidade no token vencedor (que deixa de ser o vencedor) — a asserção passou a usar penalidade 1,0001 e a checar os dois sinais com `temp 1`.

### Passo 3 — `--chat`

`include/rdna4/chat.h` + `src/backend/chat.cpp` renderiza o template qwen35 direto (sem engine Jinja): merge de system/developer, instrução de `reasoning_effort` (default `xhigh`), `<|im_start|>user\n…<|im_end|>\n`, assistant com `<think>`, tool response, e o prompt de geração `<|im_start|>assistant\n<think>\n` (ou a variante com `</think>` fechado quando thinking está desligado).

`tests/oracle_chat.cpp` (libllama-common, `common_chat_templates_apply`) gera `tests/golden/chat_renders.txt` com 13 conversas; `tests/check_chat.cpp` compara **byte a byte** → todas idênticas (300/270/74/330/308/313/396/297/345/320/313/397/433 bytes) + 3 caminhos de erro (papel desconhecido, system após o primeiro turno) rejeitados.

### Passo 4 — entrada por token no grafo + CLI `run`

- `Graph::forward_tokens(ids, pos, …)` (`include/rdna4/graph.cuh`): `token_embd.weight` é **quantizado** nos UD (q3_K no IQ3_S, q4_K no IQ4_XS) e o grafo de referência alimenta a linha **desquantizada** (`ggml_get_rows` → f32). A linha é desquantizada no device reusando exatamente os kernels já validados bit-exatos (`include/rdna4/dequant_row.cuh`, extraído de `check_dequant_gpu.hip`), sem round-trip pelo host e sem uma segunda implementação para manter em sincronia. `forward()` (embeddings f32 do host) e `forward_tokens()` compartilham o mesmo corpo (`forward_run`).
  - Esse refactor pegou um bug real: o launcher compartilhado do `q8_0` subia com **32 threads** onde o kernel usa o mapeamento `float2` de **16** threads (32 elementos por bloco) → escrevia fora do bloco; `check-dequant-gpu` acusou `q8_0 MISMATCH` e voltou a `OK (14 types tested)` com o conserto.
  - `Graph::init` agora exige `token_embd.dim0 == n_embd && dim1 == output.dim1` (o índice por id de token sairia do buffer sem isso).
- `src/main.hip`: subcomando `run` (o `info`/orçamento do M0-M2 continua sendo o comportamento padrão). Flags: `-m -p -n --chat --system --no-thinking --reasoning-effort --temp --greedy --top-k --top-p --min-p --repeat-penalty --repeat-last-n --seed --ctx-size --cache-type-k/-v -v --no-stats`.
  - **stdout só leva texto gerado**; todo diagnóstico (timings, ids, avisos) vai para stderr — é o que permite comparar a saída byte a byte.
  - Streaming com fronteira de UTF-8: re-decodifica o prefixo gerado e emite só o que termina em sequência completa (`utf8_keep_len`), guardando bytes de um caractere multibyte partido entre dois tokens.
  - Para em EOS (`<|im_end|>` = 248046), clipa `-n` ao contexto disponível, e falha com exit ≠ 0 em prompt/token fora de faixa, parâmetros inválidos, flag desconhecida, `-m` ausente, ctx pequeno demais.

### Passo 5 — aceite do M4

**Golden test** (`scripts/check_golden_run.sh`, prompt fixo "The capital of France is", `-n 32`):

| arquivo em `tests/golden/` | o que fixa |
|---|---|
| `run_greedy_IQ3_S.txt` + `run_greedy_ids_IQ3_S.txt` | saída e ids com `--greedy` (independente de RNG) |
| `run_sample_IQ3_S.txt` + `run_sample_ids_IQ3_S.txt` | saída e ids com `--temp 1 --seed 42` |

O script roda o greedy, roda a amostragem **duas vezes** (mesma seed ⇒ mesma saída byte a byte), roda com `seed 43` (tem que **diferir**: sem isso um sampler que ignorasse a seed passaria) e roda com KV `q4_0` (continua coerente). **Controle negativo feito:** corrompendo cada golden, o script falha (`FAILED` + diff) — o teste não é vacuoso. `UPDATE=1` regenera os goldens.

**Comparação com o llama.cpp (aceite forte):** `scripts/compare_llama_greedy.sh` roda o mesmo prompt (ids vindos do **nosso** tokenizer, já validado bit-exato) no nosso motor e no llama.cpp (`oracle-next-token` estendido com `ORACLE_GREEDY=N`, backend Vulkan, decodificação token-a-token):

```
rdna4: 32 tokens, llama.cpp: 32 tokens
IDS MATCH: all 32 greedy tokens identical
text: rdna4 130 bytes, llama.cpp 130 bytes, IDENTICAL
" Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome.\n
 The capital of Spain is Madrid.\nThe capital of Portugal is"
```

32 passos de decode consecutivos (64 camadas cada) sem **nenhuma** divergência — é o mesmo tipo de evidência que o argmax do M3, agora encadeado.

**Uso real medido** (IQ3_S, KV f16, 9070 XT): carga 3,5-16,6 s (page cache), prefill 19 tokens 0,84 s (22,6 tok/s), decode **~26 tok/s**; `--chat --no-thinking -p "What is the capital of France?"` responde `The capital of France is **Paris**.` e para no EOS em 8 tokens. `--chat` com thinking produz o rastro de raciocínio normalmente.

### Passo 6 — revisão adversarial do M4 (agente separado, só leitura de código)

Revisão contra `llama-sampler.cpp`, `llama-vocab.cpp`, `common/*` e o próprio `tokenizer.chat_template` extraído do GGUF. **1 achado crítico**, 2 "major", 7 "minor", 4 "nit". Todos os acionáveis foram corrigidos:

| achado | correção |
|---|---|
| **C1 (crítico) `dequant_row.cuh`**: o *block count* do IQ4_NL era 8× grande demais → escrita **fora do buffer** no device. O kernel `dequantize_iq4_nl` funde 8 sub-blocos de 32 por chamada (`x + ibs*(QK_K/QK4_NL)`, escreve `yy + i*256`), mas o launcher derivava a unidade de `dtype_block_elems()` = 32; o teste imprimia `BIT-EXACT` porque só comparava o prefixo correto. Regressão **introduzida pela extração do launcher** (commit anterior), e o defeito já tinha sido visto no M2 (C1) | unidade **explícita** (`q8_0` = 32, todo o resto = 256) em vez de derivada do layout de armazenamento, + grid limitado a 4096 como na cópia validada. O `check-dequant-gpu` agora **hipMemseta o destino com 0xAB, aloca uma guard band de 4096 floats e falha se qualquer elemento passado do fim for tocado** (ou se algum elemento pedido ficar sem escrever). Controle negativo: reintroduzindo o bug, `iq4_nl ... guard=OVERFLOW` + `wrote 4096 element(s) past the 4096-element destination` e exit ≠ 0; com a correção, `guard=ok` nos 14 tipos. A evidência de bit-exatidão do IQ4_NL (M2) fica re-estabelecida com o bloco certo |
| **M1 `scripts/compare_llama_greedy.sh`**: passava com **zero tokens comparados** (se os dois motores não emitissem nada, `a == b == []` e o script saía 0) | falha explícita `VACUOUS` quando qualquer lado fica vazio + nota quando sai menos que o pedido |
| **M2 EOG**: o CLI parava só no EOS (248046), enquanto o llama.cpp para em **qualquer** `llama_vocab_is_eog` — neste vocab são 5 ids: 248044 `<|endoftext|>`, 248046 `<|im_end|>` e 248063/248064/248065 (os FIM, detectados por *texto*, não pelo metadata) | `Tokenizer::is_eog()` construído como o llama.cpp constrói (lista fixa de textos control + FIM por texto + os ids do metadata) e usado no laço do CLI; `tests/oracle_eog.cpp` (libllama) + `tests/golden/eog_ids.txt` + `tests/check_eog.cpp` fixam o conjunto contra a referência (5/5 ids). O `oracle-next-token` em modo greedy também passou a parar em EOG, para a comparação continuar justa |
| m3 `repeat_last_n <= 0` significava "história inteira" aqui; no llama.cpp `penalty_last_n = max(n,0)` e **0 desliga** a penalidade | semântica igual à do llama.cpp (+ testes: 0 e negativo desligam, janela positiva aplica) |
| m4 o CLI rejeitava `--top-k 0`, `--top-p 0`, `--min-p -1`, `--repeat-penalty 0`, todos válidos no llama.cpp (e já tratados como "desligado" pelo sampler) | validação removida: `top_k <= 0` desliga, `top_p <= 0` mantém 1 candidato, `min_p <= 0` desliga (documentado no `--help`) |
| m5 defaults do sampler fixos no código (20/0.95/1.0) em vez de lidos dos metadados, e `min_p` divergindo do default efetivo do llama-cli (0.05) | `apply_metadata_sampling()` lê `general.sampling.{top_k,top_p,temp,min_p}` do GGUF (flag do usuário sempre vence); `min_p` continua 0 (desligado) e está documentado — o SPEC §1.4 lista só os três KVs que este arquivo tem |
| m6 no modo greedy a ordem vinha de um `std::sort` **não estável** (empates e NaN indefinidos, e comparador sem *strict weak ordering* com NaN); o llama.cpp faz uma varredura pelo primeiro máximo | estágio greedy reescrito como a varredura do `llama_sampler_temp_impl` (primeiro máximo estrito, empate fica com o menor id, `-inf`/NaN caem no primeiro candidato) e movido para **antes** dos outros filtros (o máximo sempre sobrevive a top_k/top_p/min_p). Testes novos: all-`-inf`, NaN, empate, `n_vocab == 1`, `top_k > n_vocab` |
| m7 `token_to_piece` usava `map.at()` do mapa de byte-encoding → `std::out_of_range` (e `std::terminate`) para um code point fora do mapa; tokens `BYTE` sairiam como `<0xXX>` literal | `unicode_utf8_to_byte_safe()` (variante que não lança, adicionada ao arquivo vendorizado) + fallback que emite o code point como está; token `BYTE` agora emite o byte cru, como o llama.cpp |
| m8 BOS duplicado quando `tokenizer.ggml.add_bos_token=true` (`encode()` já insere e o `main` inseria de novo) | inserção do `main` removida |
| m9 `UPDATE=1` do `check_golden_run.sh` sobrescrevia os goldens commitados mesmo com uma execução falha | só atualiza quando a execução correspondente teve sucesso |
| n10/n11 comentário dizendo que o stdout termina em `\n` (não termina) e `stream.h` citando um flag `--stream` inexistente | comentários corrigidos |
| n12 `parse_u64` aceitava `-1` (o `strtoull` dá a volta) em `--seed`/`--ctx-size` | `-` na frente é rejeitado |
| n13 `chat.cpp` validava `reasoning_effort` mesmo com thinking desligado, onde o template não valida | validação só com thinking ligado |

**Lacunas de teste fechadas junto com os achados:** guard band no `check-dequant-gpu` (C1); `check-eog` + oracle (M2); asserção de mínimo de tokens no script cross-engine (M1); entradas degeneradas do sampler (m6); **pureza do stdout** (rodar com e sem `-v` e comparar o stdout byte a byte) e combinações de cache KV misturadas (`f16/q8_0`, `q8_0/f16`, `f32/q4_0`) no `check_golden_run.sh`; `check-stream` (fronteira UTF-8 do streaming, com controle negativo).

**Verificado e considerado correto pela revisão** (sem achado): a ordem e as fórmulas de cada estágio do sampler contra o `llama-sampler.cpp` (janela/sinal da penalidade, top_k, top_p "menor prefixo ≥ p", min_p `max+log(min_p)`, dist com o fallback do último elemento, e a cadeia default reduzida porque dry/top_n_sigma/typical/xtc/mirostat estão desligados); determinismo (RNG por instância, `filter()` const e sem consumo de RNG — logo `-v` não perturba o stream, um sorteio por passo, nenhum estado global mutável, nenhum `printf` de float no stdout); `utf8_keep_len` em 2/3/4 bytes completos e partidos, caudas inválidas e a premissa de prefixo do `decode()`; o laço greedy do `oracle-next-token` (posições, `logits_ith`, KV f16 dos dois lados, EOS antes do push); `check-tokenizer` falhando com linha faltante; e o `chat.cpp` contra o template real do GGUF (todos os construtos batem).

**Ainda em aberto (consciente):** o teste cross-engine depende de o EOG ser o mesmo dos dois lados (agora é, mas nenhum caso força um EOG no meio da geração); a comparação de `--chat` fim-a-fim usa o tokenizer já validado bit-exato em separado, sem um teste que amarre "prompt renderizado → ids do llama.cpp"; o laço de chat do CLI é single-turn (multi-turn é suportado e testado só no `chat_render`).

1. ✅ Sampler (greedy, temperature, top-k/n, top-p, min-p, repetition penalty, seed) com defaults dos metadados (top_k 20, top_p 0.95, temp 1.0).
   Ref: `SPEC.md` §1.4 · KVs `general.sampling.*` lidas em M1.
2. ✅ CLI `run` com streaming e erros em stderr/exit != 0.
   Ref: `SPEC.md` §1.1.
3. ✅ `--chat` aplicando `tokenizer.chat_template` (`pre = qwen35`, BOS 248044 / EOS 248046 / PAD 248055).
   Ref: `docs/gguf-qwen-quantizacao-llamacpp.md` §1 · `SPEC.md` §1.1.
4. ✅ **Aceite cumprido:** golden test — prompt fixo no `UD-IQ3_S`, mesma seed, mesma saída (e greedy **idêntico ao llama.cpp por 32 tokens**).


## M5 — Validação nos dois arquivos + docs ✅ concluído em 2026-09-13

Objetivo: os dois modelos conversando na 9070 XT.

### Passo 1 — ferramenta de medição (`bench`) e baselines dos dois arquivos

`rdna4-infer bench -m <gguf> [-n N] [--warmup N] [--reps N] [--prefill N] [--ctx-size N]
[--cache-type-k/-v T] [--fill-cache] [--layers N]` reporta prefill/decode em tok/s,
VRAM em uso, banda efetiva (bytes do modelo por token ÷ tempo) e a divisão do tempo
por token entre thread chamadora e device. Aquece fora da janela medida (esta GPU cai
para DPM profundo entre kernels) e usa greedy (sem RNG). `--layers N` roda só as N
primeiras camadas: é o que permitiu **atribuir** o custo por token em vez de supor.

| | IQ3_S | IQ4_XS |
|---|---|---|
| peso | 11,21 GiB | 13,27 GiB |
| decode (4K, KV f16) | 27,6 tok/s | 27,0 tok/s |
| prefill (512 tokens, por token) | 28,8 tok/s | 27,4 tok/s |
| banda efetiva | 336 GB/s | 364 GB/s |
| PPL (wikitext-2, 10×512) vs llama.cpp | ≤ 0,25 % de desvio | ≤ 0,15 % |

Referência na mesma máquina (`llama-bench -ngl 99`, Vulkan, mesmo arquivo): **decode
39,7 tok/s**, **prefill 440 tok/s**. Tabelas de VRAM por ctx/KV (4K→131K) em
`docs/medicoes-m5.md` e em `SPEC.md` §3 — inclui uma **correção** da estimativa do M0:
o IQ4_XS com 32K f16 **cabe** (15,66 GiB, 0,26 GiB livres); o que não cabe é 131K q4_0.

### Passo 2 — onde o tempo vai (medido, não suposto)

Decomposição com `--layers` (IQ3_S, 32 tokens, 3 repetições):

| parte | custo | fração |
|---|---|---|
| embeddings + norm final + LM head + cópia de logits + sampler | 1,96 ms | 5,5 % |
| 64 camadas do tronco | 33,87 ms (0,529 ms/camada) | 94,5 % |
| — quantização de ativação (teto, medido pulando **todos** os `quantize_q8_1`) | 1,5 ms | 4,1 % |

O matvec foi tunado no M2 contra um teto de leitura medido de **619 GB/s**; a
estimativa matvec-only (12,02 GB em ~28,9 ms = 416 GB/s) explica ~29 ms dos 35,8 ms,
ou seja ~6,9 ms são os ~1300 kernels pequenos por token (norms, conv/delta do GDN,
atenção, ~450 quantizações de ativação, somas de residual). **Duas hipóteses foram
medidas e rejeitadas:** (a) *HIP graphs* — o caminho de lançamento desta máquina
custa **1,04 µs/launch** (medido com kernel nulo 2000×), então ~1300 lançamentos
custam ~1,4 ms; o tempo que a thread passa no laço de lançamento é *back-pressure* da
fila cheia, não custo de host; (b) *fundir a quantização de ativação* — pular todas as
450 por token muda 27,73 → 28,82 tok/s (**4,1 %**). Nenhuma das duas virou mudança de
código, e ficou registrado o motivo.

Única otimização de fato aplicada (aritmética idêntica, ganho pequeno mas estrito):
buffer de logits reaproveitado entre tokens (era `hipMalloc`/`hipFree` de 1 MB por
token), leitura de `hidden` opcional (`set_want_hidden`, uma sincronização a menos por
token no CLI) e `reset_state()` para reusar o grafo em sequências independentes.

### Passo 3 — perplexidade (o aceite de qualidade do SPEC §1.5)

`rdna4-infer ppl -m <gguf> -f <corpus> [--ctx-size 512] [--stride 512] [--chunks N]
[--nll-out FILE]` reproduz a metodologia do `llama-perplexity` modo strided: janela de
`ctx + stride/2` tokens, pontuando as posições `[window-stride-1, window-1)`, softmax
com max subtraído e `PPL = exp(média NLL)`.

`scripts/compare_ppl.sh` fecha o aceite **posição a posição**: tokeniza o corpus com os
dois tokenizadores (ids idênticos nos 297 193 tokens), roda o motor sobre as janelas,
roda **o mesmo modelo no llama.cpp** sobre as janelas idênticas
(`oracle-next-token` com `ORACLE_NLL_OUT`, um token por vez) e compara NLL por posição.
Resultado: 5120 posições por arquivo, desvio por chunk **0,004–0,25 %** (IQ3_S) e
**0,001–0,15 %** (IQ4_XS); pior diferença numa única posição 0,458 / 0,873 nats.

⚠️ **Pegadinha encontrada no caminho (documentada porque muda a interpretação de
qualquer comparação futura):** o `llama-perplexity` **não** pontua uma janela limpa. Para
o chunk 2 deste corpus ele imprime PPL **5,8045**, mas o *mesmo modelo* sobre a *mesma*
janela de 768 tokens (estado zerado, mesmas posições) dá **8,7591** — o número da
ferramenta carrega contexto dos chunks anteriores. A primeira leitura (nossa 8,75 contra
5,80 da ferramenta) parecia um bug de 51 % no motor; a comparação por posição mostrou
que o motor estava certo (2,16905 vs 2,17010 de NLL média, 0,05 %). **Lição:** comparar
agregados de uma ferramenta sem entender o que ela condiciona custa horas; comparar por
posição localiza o problema na hora.

### Passo 4 — docs

`README.md` ganhou uso completo (5 subcomandos, flags, tipos de KV, códigos de saída),
números medidos, os comandos de validação e as limitações conhecidas.
`docs/medicoes-m5.md` é o log de medição (baseline da referência, decomposição por
token, tabelas de VRAM/ctx/KV dos dois arquivos, PPL por chunk). `SPEC.md` §3 passou a
ter os números medidos, com a correção do IQ4_XS/32K.

### Passo 5 — o que **não** foi feito (e por quê)

**Prefill em batch** é a única lacuna grande: 28,8 tok/s (por token) contra 440 tok/s
da referência = **16×**, e é o que dói em prompt longo (>200 s para 4K tokens de
contexto, contra ~10 s do llama.cpp). Não foi feito no M5 porque não é "afinamento de
baixo risco": é trocar GEMV por GEMM nos 14 kernels quantizados (cada thread passa a
acumular N tokens com o mesmo bloco de peso desquantizado) e revalidar bit-exatidão —
trabalho de milestone, com mudança de numérica. Fica como **M6** com o número que o
justifica. Atenção (não tiling) continua sendo limite de performance em contexto longo
(262 ms/token em 64K), também registrado.

1. ✅ Rodar M4 nos dois `.gguf`; registrar perplexidade/tempo por arquivo.
2. ⚠️ Afinamentos gfx1201 de baixo risco: **medidos antes/depois** e, dos dois
   candidatos, **nenhum** se pagou (HIP graphs 1,04 µs/launch, quantização de ativação
   4,1 % como teto); aplicadas só as mudanças de custo estritamente menor. O
   afinamento grande (prefill batch) foi para o M6 com medição que o justifica.
3. ✅ README reproduzível + `SPEC.md` §3 com números medidos.
4. ✅ **Aceite cumprido:** coerência nos dois arquivos dentro dos 16 GB — greedy
   idêntico ao llama.cpp por 32 tokens, PPL por posição dentro de 0,25 %, 131K de
   contexto no IQ3_S e 64K no IQ4_XS (o 131K do IQ4_XS não cabe, documentado).

## M6 — Prefill em batch (passo 1 feito; escopo re-medido)

### Passo 1 — kernel batched + curva de escala ✅ (feito em 2026-09-13)

`matvec_kernel_batch<T, ROWS, WPR, ILP, N>` + `matvec_launch_batch()` em
`include/rdna4/matvec.cuh`: N vetores de ativação compartilham **uma** passagem pelos
pesos. Para um dado token o k-walk, os slots de ILP, a soma por slot e os dois
estágios de redução são as mesmas operações na mesma ordem do `matvec_kernel_gen`,
então cada elemento de saída é **bit-idêntico a N chamadas GEMV separadas** —
verificado, não assumido (`tests/check_matmul_gpu.hip`: *bit-exact in every
configuration tested*).

**Resultado medido (IQ3_S, working set maior que o L2, aquecimento com sync):**
ganho de **~2-5× por tipo**, não N× (iq3_s 4,35×, iq3_xxs 5,14×, q3_k 5,45×,
iq4_xs 2,62×, q4_k 3,34×; alguns tipos *regridem* em N=16 por pressão de registradores).
Motivo: os `vec_dot` são **issue-bound** (o custo de ALU por byte de peso cresce com N),
exatamente o que o M2 mediu no corpo do iq3_s (452 instruções por vec_dot, 55-62 % de
emulação de bytes).

**Projeção para o modelo** (bytes reais por tipo × banda medida):

| | por token | taxa |
|---|---|---|
| matvec GEMV (hoje) | 35,9 ms | 27,9 tok/s |
| matvec batched | 8,97 ms | — |
| + andaime por token (norms, atenção, GDN, quantização: 6,9 ms medidos no M5) | 15,9 ms | **~63 tok/s** |

**Duas conclusões que mudam o escopo deste milestone:**

1. Batching do matvec vale **~2,2× de prefill** (28,8 → ~63 tok/s; num prompt de 512
   tokens: 17,8 s → 8 s), e exige a reestruturação layer-major do grafo.
2. O andaime por token sozinho **limita o prefill a ~145 tok/s** mesmo com matvec
   infinitamente rápido. Chegar aos 440 tok/s do llama.cpp exige em batch *tudo*
   (norms, atenção, recorrência GDN, quantização) **e** um laço interno muito mais
   barato (kernel tiled estilo MMQ com peso em shared memory e dot int8 largo — a rota
   que o M2 deliberadamente não tomou, porque troca exatidão bit a bit por velocidade).

**Correção de registro (afeta o M2):** a tabela de GB/s por tipo do M2 (iq3_s 557-753
GB/s) foi medida com tensores que cabem no L2 de 64 MB — são números de **L2**, não de
DRAM. Com working set realista, o N=1 fica em **143-552 GB/s**, o que explica a banda
efetiva de 336 GB/s medida ponta a ponta no M5.

**Duas armadilhas de metodologia** (registradas em `docs/medicoes-m6.md`): working set
precisa exceder o L2 (senão mede-se L2), e o laço de aquecimento precisa de `sync`
(sem ele a fila de comandos absorve milhares de iterações e o relógio de parede mede
vazão de lançamento — a primeira execução do M6 enfileirou ~20 minutos de trabalho).

### Passo 2 (a decidir, com os números acima)

- **2a. Reestruturar para prefill batched** (layer-major: projeções em batch, atenção e
  recorrência GDN por token): ~63 tok/s de prefill, bit-exatidão verificável contra o
  caminho por token. Trabalho grande, ganho real de 2,2×.
- **2b. Batching completo** (norms + atenção + GDN + quantização em batch): teto ~145
  tok/s; bem mais trabalho (máscara causal em batch, recorrência sequencial).
- **2c. Kernel tiled estilo MMQ**: única rota para ~300-440 tok/s; perde bit-exatidão,
  precisa de rodada própria de validação numérica; trabalho de milestão separado.
- **2d. Tiling da atenção** (262 ms/token em 64K hoje): mudança contida, ganho visível
  em contexto longo — provavelmente o melhor ganho por hora se contexto longo importa.

## M7 — Atenção de contexto longo ✅ concluído em 2026-09-13

O `bench --start-pos` (corrigido no M5) mostrou que contexto longo era **limitado pela
atenção**, não pelos pesos: 45 ms por passo a 4K, 174 ms a 64K (KV f16) e 441 ms a 131K,
contra ~36 ms de fluxo de pesos constante. O diagnóstico
(`tests/bench_attn_gpu.hip`) apontou **latência de memória**, não softmax nem banda:
o tempo por camada não segue os bytes do cache (7,0-9,4 ms para 36/128/256 MiB), a banda
alcançada a 64K era 27 GB/s de ~600 GB/s, e a mesma quantidade de trabalho ia de 9,42 ms
(8 warps) para 3,69 ms (32 warps). Ou seja: o problema era **requisições em voo**.

Correções, cada uma medida: cargas vetorizadas do cache (`kv_load8<CT>()`, 1 acesso por
lane em vez de 8 escalares), mais warps por CTA, e split-KV (a faixa de chaves dividida
entre CTAs com merge de softmax online). Resultado final (decode, IQ3_S):

| contexto / KV | M5 | agora | ganho |
|---|---|---|---|
| 4 096 f16 | 22,13 | **26,82** | 1,21× |
| 16 384 f16 | 13,72 | **24,29** | 1,77× |
| 65 536 f16 | — | **não cabe** (transborda 2,1 GB para GTT: 2,16-7,97 tok/s; ver §2.4 do doc das tarefas 2/5) | — |
| 65 536 q4_0 | 4,18 | **17,88** | 4,3× |
| 131 072 q4_0 | 2,27 | **12,99** | 5,7× |
| 65 536 **q8_0** | — | **18,28** (atenção 19,87 ms) | — |

Split-KV **não** é bit-exato (a soma das chaves muda de ordem) e por isso tem gate próprio:
`scripts/check_attn_split.sh` compara em texto real (PPL 5,1989 contra 5,1917, 0,14 %) e o
`check-kvctx-gpu` compara logits a contexto fixo (rel-L2 ~1e-6). `docs/medicoes-m7.md` traz
o detalhamento, inclusive a correção do registro "32 warps": o valor **embarcado** sempre
foi 8, e o `a830570` dizia 32.

## M8 — prefill em batch, `proj_qq` e política de splits ✅ concluído em 2026-09-13

- **Prefill em batch** (`Graph::forward_batch`, 2..16 tokens, layer-major): projeções,
  quantização da ativação, norms e elementwise em uma passada; KV, atenção e recorrência
  GDN continuam por token porque são sequenciais. **Bit-idêntico** ao caminho por token
  (`check-batch-gpu`, rel-L2 0 em N=2/3/4/8/16) e **28,6 → 70,2 tok/s** (2,45-2,62×).
- **`proj_qq`**: a projeção reusa a ativação já quantizada em q8_1 (384 lançamentos e uma
  passagem de leitura a menos por token, +1,3 %, bit-exato).
- **Política de splits** 2048 → 512 chaves: +14,7 % a 4K (`docs/rocm-estudo.md`).
- **MTP: decidido não fazer.** O rascunho do NextN tem 86,7 % de aceitação (o driver do
  llama.cpp, 87,5 %) e o greedy sai idêntico, mas cada modo ainda roda um forward do tronco
  por token aceito: `--mtp` é 9-10 % **mais lento**. Com verificação em batch a projeção
  medida dá 1,3-1,5×, não os 2,5× que a razão passo-rascunho/passo-tronco sugere. Fica
  documentado em `docs/medicoes-m8.md` e `docs/mtp.md` em vez de embarcado.

## Frente paralela — 9 tarefas, 6 agentes, 6 worktrees ✅ concluída em 2026-09-13

Trabalho independente em worktrees separados (um agente por worktree, uma placa de GPU
serializada por `scripts/gpu-lock.sh`, ver `docs/gpu-queue.md`), mergeado em série pelo
coordenador:

| # | frente | entrega |
|---|---|---|
| 1 | auditoria de qualidade | `docs/auditoria-qualidade.md` (4 CRÍTICOS, 19 IMPORTANTES); os 4 CRÍTICOS e 3 IMPORTANTES foram corrigidos na `main` com prova antes/depois (§7 do doc, `scripts/check_hardening.sh`) |
| 2 | kernels do Qwen e matrix cores | `docs/qwen-kernels.md` |
| 3 | Vulkan (llama.cpp) vs HIP, código a código | `docs/vulkan-vs-hip.md` |
| 4 | inventário de quantizações | `docs/quants-inventario.md` |
| 5 | desenho do cache KV e orçamento de tráfego | `docs/kv-memoria-desenho.md` |
| 6-7 | autotuning e regressão | `docs/autotuning-gfx1201.md`, `include/rdna4/tuning.h` (tabela única), `check-tuning` + `scripts/check_regression.sh` como gates |
| 8 | build reprodutível | `docs/build-repro.md` |
| 9 | README com os números reais | `README.md` |

Regras do lote: cada agente só commita na sua branch; o merge é em série; GPU só com o
lock; agente que não precisa de GPU não usa GPU.

### Frente prefill da rodada noturna (merge `264044e`)

O andaime que rodava por token **dentro** do `forward_batch` virou kernels em lote (atenção
com máscara causal, recorrência GDN com os tokens andados dentro do kernel, `conv1d`, normas,
escalares e `kv_write` em um lançamento), mais o `delta_rule` lendo a linha do estado com
`float4` (a warp lia a 512 B de distância = 8× de amplificação de setor). Medido em janela
limpa, `bench --prefill 512 --prefill-reps 3`, melhor de 3, piso de ruído 1,2 %:

| | prefill 512 tokens |
|---|---|
| baseline da noite | 73,05 tok/s |
| + andaime em lote | 104,5 tok/s (**+43 %**, A/B intercalado) |
| + `delta_rule` com `float4` | **123,9 tok/s (+18,6 % sobre o anterior; +69,7 % no total)** |

A atribuição acima é a do diário da frente (`journal-prefill.md` §4.1, A/B intercalado em
janela limpa): o README e este PLAN chegaram a atribuir os mesmos +69,7 % a mudanças
diferentes — quem herdasse a alavanca pelo PLAN superestimaria o `float4` em 3,7×.

`check-batch-gpu` continua **BIT-EXACT** em N=2/3/4/8/16 e no prompt completo, e o
`check-graph-gpu` passa. O que sobrou: o matvec em lote lê 12,0 GB por chunk de 16 em 110 ms
(**109 GB/s**) contra 446 GB/s do caminho por token, porque o `vec_dot` reexecuta a
dequantização/LUT/sinais N vezes — dequantizar o bloco uma vez e fazer N `dp4a` é o item
seguinte (estimativa de 2-2,5× no matvec ⇒ ~200 tok/s de prefill).

### O que o autotuning embarcou (medido, com A/B intercalado e piso de ruído de 1,001×)

- **`UNROLL=2`** nos quatro tipos de ILP=1 (`iq3_s` **1,073×**, `iq3_xxs` 1,031×,
  `iq2_s` 1,011×, `iq4_nl` 1,055×) e **`rows=1`** em `iq2_xs` (1,020×) e `iq2_xxs`
  (1,002-1,011×): **bit-exatos** (mesmas operações, mesma ordem; `rows` só muda qual CTA
  calcula qual linha). Somados: matvec **25,2 → 24,5 ms/token (1,028×)** sobre o inventário
  real (497 tensores, 10,36 GiB por token).
- **Regra de duas pontas no `WPB` da atenção** (`kAttnSplitWpbWide`): 131K com KV `q4_0` mede
  **1,087×** (24×16 contra 16×8, 12/15 rodadas); abaixo disso ≤ 1 %. Não é bit-exato (muda a
  ordem do merge), então é a mesma classe numérica do split-KV, coberta pelo
  `check_attn_split.sh`.
- **Rejeitados com dado**: `rows` global (1,010/1,005/0,991/0,988×), ILP global
  (0,82/0,76/0,77×), `unroll=4` (0,917×), prefetch L2 real (0,984×), `kAttnMaxSplits` 16→24
  (0,921× a 16K), e "poucos splits + CTA larga" em 4K/16K (real no kernel, 0,2-0,5 % no
  total). Armadilha medida: `rows=1` e `unroll=2` **não compõem** (`iq3_s`: 8,081 ms contra
  7,723 ms = 4,6 % pior).
- **Fim a fim** (A/B com dois binários, janela limpa): contexto curto 27,71 → **28,35 tok/s
  (+2,3 %, 3/3)**, fim de 4K +1,2 % (2 empates), 131K `q4_0` +0,8 % (dentro do ruído daquela
  configuração).
- **Gate novo da tabela** (`check-tuning`, CPU puro): compara a configuração embarcada com
  `tests/golden/ml_tuning.txt` (22 linhas e 22 chaves obrigatórias) — um refactor que troque um
  parâmetro medido quebra o gate em vez de mudar o desempenho em silêncio.
- **Suíte de regressão** (`scripts/check_regression.sh` + `check-regression-gpu`): 7 casos
  fixos (curto, prosa, código, CJK, chat, prefill real de 4096 tokens, 32K com cache semeado),
  greedy puro, ids bit-exatos e logits rel-L2 ≤ 1e-5; **79 s**. Controle negativo colado: com
  `amax/127 → amax/126` na quantização q8_1 da ativação ela falha 15 vezes (rel-L2 5,4e-3 a
  1,6e-2, e o caso de 32K diverge de id no passo 0) e volta a passar quando revertido.


## Rodada noturna (2026-09-14) — sete frentes, um alvo: 131K + KV `q5_0`/`q4_1` + MTP

Baseline: tag `noite-baseline-2026-09-14` (todas as nove tarefas anteriores mergeadas,
`check_all.sh` PASS). Regras e diário: `docs/noite-regras.md`, `docs/journal-noite.md`
(relatório da manhã, com TL;DR). Duas revisões adversariais rodaram: `docs/adversarial-noite.md`
(a árvore de partida, F1-F15) e `docs/adversarial-noite2.md` (o código escrito durante a noite,
R1-R11 — **dois críticos, ambos sem cobertura de gate**).

**Resultado em números, todos com gate ou A/B intercalado:**

| frente | antes | depois | exatidão |
|---|---|---|---|
| prefill 512 tokens | 73,05 tok/s | **123,68** (+69 %) | bit-exato (`check-batch-gpu`) |
| decode 4K real | 28,97 tok/s | **34,00** (+17,4 %) | bit-exato (gate por tipo) |
| KV `q5_0`/`q4_1` a 131K | não existia | **14,83 tok/s**, 14,26 GiB, zero GTT | bytes idênticos ao llama.cpp |
| MTP (verificação em lote) | 0,88× (serial) | **1,22×** pré-kernels, **0,96×** na árvore final, **2,03×** em código | md5 do stdout == ganancioso |
| contexto longo | — | RoPE vs `ggml` até 262 143; top-5 idêntico a 17 639 tokens | gate novo |

**O que ficou aberto, com o número que o prioriza**: `matvec_kernel_batch` roda a
**109 GB/s** contra 446 GB/s do mesmo trabalho por token — é 83,7 % do prefill **e** o que
decide se o MTP paga em prosa; a hipótese da dequantização reexecutada foi refutada por ISA, e
a questão (banda amortizada × issue/ocupação) segue aberta. Depois vêm o GQA compartilhado só
para KV quantizado (até +18 % a 64K), `UNROLL=4` com LUT em LDS (sem medida) e as fusões de
kernels pequenos (bloqueadas pelo `emit()` bloqueante do oráculo).

**Lições de processo registradas**: (1) gate vermelho por contenção é pior que gate não rodado —
`gpu-lock.sh` agora espera a VRAM do dono anterior drenar e marca os filhos (`GPU_LOCK_HELD`),
porque ninho de lock custou ~2 h de GPU; (2) um ganho vale o que vale o baseline (o número do
MTP ficou inflado por 40 min por causa de um baseline quebrado); (3) revisão adversarial no
código escrito na própria sessão pegou dois críticos que nenhum gate pegava.
