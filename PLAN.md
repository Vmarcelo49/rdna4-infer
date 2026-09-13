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

  | prompt | ids | llama.cpp | nós |
  |---|---|---|---|
  | "Hello world, this is a test." | `9419 1814 11 411 369 264 1228 13` | 198 | 198 (12,41 vs 13,53) |
  | "The capital of France is" | `760 6511 314 9338 369` | 11751 | 11751 (14,19 vs 17,53) |
  | "def fibonacci(n):" | `727 73111 1393 1590` | 198 | 198 (16,16 vs 20,03) |
  | "1 2 3 … 9" | `16 220 17 … 24` | 220 | 220 (19,07 vs 16,41) |

  Top-5 também bate em ordem/ids em 4/5 (um ligeiro desacordo no 4º/5º colocado em 3 dos 4 prompts).

**Passo 4 — tipos de KV cache + contexto longo ✅** (`include/rdna4/kv.h`, `tests/check_kvctx_gpu.hip`)

- **Tipos**: `KvType::{F32,F16,Q8_0,Q4_0}` (o default do llama.cpp é `f16`; o plano pedia `q8_0`/`q4_0` para ctx longo). As linhas do cache (uma cabeça KV de um token) são gravadas **já quantizadas** em blocos de 32 elementos, byte a byte como `quantize_row_q8_0_ref`/`quantize_row_q4_0_ref` do llama.cpp, e a atenção **desquantiza on-the-fly** (`kv_load<CT>`), sem cópia f32 do cache. `kv_store_row_launch` grava uma linha; `kv_fill_launch` preenche caches inteiros (hook de teste).
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

**Knobs de diagnóstico** (usados para o estudo acima, mantidos por serem baratos): `GRAPH_NOISE=<rel>` (+`GRAPH_NOISE_COHERENT=1`) perturba a saída de cada camada antes do residual; `GRAPH_EXACT=1` roda o check f32-exato da primeira projeção; `GRAPH_SAMPLES=1` imprime os 6 valores (oráculo vs nós) de cada nó divergente; `GRAPH_LAST_TOKEN=1` compara só o último token (obrigatório com dump `-ub 1`); `diag.hidden_pre_norm` reporta RMS/min/max do estado residual que entra no `output_norm`.

1. Implementar o ramo linear GDN (`attn_qkv` + `attn_gate` + SSM conv/recorrência + `ssm_out`).
   Ref: `docs/referencias-upstream-gfx1201-qwen35.md` § "SGLang — Qwen3.5" (`qwen3_5.py` L322-1094: `GatedDeltaNet` + `LinearDecoderLayer`) e § "hipfire — Qwen3.5" (`forward.rs` L741-800 entradas, L139-280 MoE/decode patterns).
2. Implementar o ramo full-attention GQA (`q/k/v` + QK-norm + RoPE/MRoPE + softmax + `attn_output`), RoPE `freq_base 1e7`, `dimension_count 64`, sections `[11,11,10,0]`.
   Ref: mesmo § SGLang (`AttentionDecoderLayer` L1094-1550) · `.ref/llama.cpp/src/models/qwen35.cpp` L1-120.
3. ✅ FFN SwiGLU + RMSNorms + `output_norm`/`output` (Q6_K no IQ3_S); **KV cache incremental com tipos configuráveis** (`KvType::{F32,F16,Q8_0,Q4_0}`, gravado já quantizado por bloco de 32, atenção desquantiza on-the-fly — `include/rdna4/kv.h`); decode token-a-token (prefill em blocos via `Graph::forward(..., start_pos, ...)`, bit-exato).
   Ref: `SPEC.md` §1.4 · `docs/gguf-qwen-quantizacao-llamacpp.md` §2 (tipos `Q8_0`/`Q4_0` em `ggml.h`).
4. ✅ 64K (e 131K) com KV `q4_0` cabem nos 16 GB e rodam; f16 cabe apertado em 64K, f32 não cabe (tabela no Passo 4).
5. ✅ **Aceite cumprido:** 1 camada GDN + 1 full-attention batem com o oráculo por nó (`attn_pregate-3` soma rel 1,1e-04; estado recorrente ≤ 6,1e-05); o grafo completo dá o **mesmo argmax do llama.cpp em 4/4 prompts** e top-5 **5/5 na mesma ordem** com logits dentro de ~0,1; run de ctx longo (64K/131K, KV `q4_0`) dentro dos 16 GB.

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
