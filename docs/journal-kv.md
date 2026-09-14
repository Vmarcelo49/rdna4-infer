# Diário da frente KV — quantização do cache e orçamento de memória para 131K

Worktree `rdna4-wt-noite-kv`, branch `feat/noite-kv`, base `main` = tag
`noite-baseline-2026-09-14` (`ea0e670`). Contrato: `docs/noite-regras.md`.
Formato: §5 (referência → hipótese → comando + janela → resultado com piso de
ruído → veredito).

**Estado de partida, dito primeiro:** o motor **não tinha** `q5_0` nem `q4_1`.
`include/rdna4/kv.h` só tinha `KvType{F32, F16, Q8_0, Q4_0}`. Então "K em q5_0 /
V em q4_1" não era um resultado deste motor — era uma hipótese de fora do repo,
que esta frente teve de **implementar e depois testar**.

---

## 0. Baseline de VRAM da janela (para saber o que é "janela limpa")

```
$ cat /sys/class/drm/card*/device/mem_info_vram_used
333688832          # 318 MiB: Xwayland + plasmashell + firefox, nenhum modelo
$ cat /sys/class/drm/card*/device/mem_info_gtt_used
55017472
```
Ou seja: **~0,32 GiB é o piso do desktop**, não um processo pendurado. Toda
medição abaixo desconta esse valor antes de dizer "vram em uso" do motor.

---

## 1. Implementação de `Q5_0` e `Q4_1` no cache de KV

- **Referência**: `include/rdna4/kv.h` (layout e os quatro tipos existentes),
  `include/rdna4/attn.cuh:94-196` e `:293-456` (os dois kernels de atenção que
  carregam K/V elemento a elemento), `include/rdna4/graph.cuh:406-577`
  (alocação) e `:724-741` (`kv_write`).
  Layout e aritmética dos formatos novos, do llama.cpp:
  - `ggml/src/ggml-common.h:203-212` → `block_q4_1` = `{half d; half m; uint8_t qs[16];}`
    = **20 B / 32 elementos**; `ggml-common.h:228-235` → `block_q5_0` =
    `{half d; uint8_t qh[4]; uint8_t qs[16];}` = **22 B / 32**. Os dois têm
    `static_assert` de tamanho no próprio llama.cpp, e eu repliquei com
    `static_assert` em `kv.h` (22 e 20).
  - `ggml/src/ggml-quants.c:150` `quantize_row_q4_1_ref` (`d = (max-min)/15`,
    `m = min`, `(int8_t)(x*id + 0.5f)`), `ggml-quants.c:187` `quantize_row_q5_0_ref`
    (`d = max/-16`, `(int8_t)(x*id + 16.5f)`, bit 4 do `qh` em `j` e `j+16`),
    `ggml-quants.c:479` `dequantize_row_q4_1` (`q*d + m`),
    `ggml-quants.c:500` `dequantize_row_q5_0` (`((qs[j]&0xF)|xh_0) - 16`, vezes `d`).

- **Hipótese**: com os dois formatos implementados, o alvo de 131K com
  `q5_0`/`q4_1` passa a ser (a) medível e (b) testável quanto a qualidade.
  Conta preliminar de VRAM por token: 16 camadas de atenção completa × 4 cabeças
  KV × (256 de K + 256 de V) elementos = 16 384 elementos de K e 16 384 de V por
  token ⇒ 21 504 B/token ⇒ **2,63 GiB** em 131 072 tokens, contra 2,25 GiB do
  `q4_0/q4_0` e 4,25 GiB do `q8_0/q8_0`. Com 11,20 GiB de pesos (ver §6) isso
  **não caberia** em 15,9 GiB junto com ~0,5 GiB de buffers — hipótese a testar,
  ver §5.

- **Comando**: implementação + `cmake --build build -j8`
  (`source scripts/rocm-env.sh`, `-DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++`).
  Sem GPU.

- **O que mudou, arquivo por arquivo** (tudo dentro de `kv.h`, `attn.cuh`,
  `graph.cuh`, `device.h`, `main.hip`/`serve.hip`, `tests/`, `CMakeLists.txt`
  appendado — nada de `matvec.cuh`, `mtp.cuh`, `tuning.h`):
  - `kv.h`: `block_q5_0`/`block_q4_1` (locais do kv.h, com `#ifndef QK5_0/QK4_1`,
    para não colidir com um `quants.h` futuro); `KvType::Q5_0 = 4`, `Q4_1 = 5`
    (apendados, nunca renumerados, para não mexer em nenhum valor existente);
    `kv_type_name`/`kv_type_parse`/`kv_bytes_per_elem` (22/32 e 20/32);
    `kv_row_bytes`; `kv_load<Q5_0>` e `kv_load<Q4_1>`; `kv_load8<Q5_0>` e
    `kv_load8<Q4_1>`; os dois ramos de quantização em `kv_store_row_kernel`;
    os dois ramos em `kv_fill_kernel`; os despachos de `kv_fill_launch` e
    `kv_store_row_launch`.
  - `attn.cuh`: `attn_launch` e `attn_launch_split` agora instanciam **o produto
    6×6 completo** (36 pares; antes era uma lista mantida à mão de 16 pares sem
    `f32/f32`), então `--cache-type-k q5_0 --cache-type-v q4_1` não pode cair
    fora do despacho em silêncio. Os dois pontos de entrada `_wpb` (só usados por
    `bench_attn-gpu`) ganharam apenas os pares diagonais novos
    (`Q5_0/Q5_0`, `Q4_1/Q4_1`) para não triplicar instanciações por nada.
  - `graph.cuh`: `kv_write` já usava `kv_row_bytes(kv_v_)` no passo da linha, mas
    **a alocação usava o tamanho de K para os dois caches**. Com K=q5_0 e V=q4_1
    isso super-alocava cada camada de V em 10%. Agora `kv_bytes_` (K) e
    `kv_bytes_v_` (V) são separados e todos os strides de V usam o segundo
    (novo acessor `kv_layer_bytes_v()`). Ver §7.
  - `device.h`: nada a mudar — `kv_cache_bytes`, `required_bytes` e o orçamento
    de `serve` já delegam em `kv_bytes_per_elem(KvType)`, que é a única fonte do
    número; foi para isso que a M4 os centralizou.
  - `main.hip`/`serve.hip`: `parse_kv_type` virou uma chamada a
    `rdna4::kv_type_parse` (eram três cópias da mesma lista, e um tipo acrescentado
    em duas delas seria aceito pela CLI e rejeitado pelo despacho); `kv_type_cstr`
    e os textos de help/erro ganharam `q5_0`/`q4_1`.
  - `CMakeLists.txt`: alvo novo `check-kvquant-gpu` **appendado no fim**.

- **Detalhe de implementação que vale registrar**: `qh` fica no offset 2 de uma
  struct de 22 bytes, então um load de 32 bits em `qh` é **desalinhado** — e
  `std::memcpy` não é `__device__` (erro de compilação real na primeira
  tentativa). Por isso o bit plano é lido byte a byte: `kv_load<Q5_0>` usa
  `b->qh[j>>3]` bit `(j&7)`, e `kv_load8<Q5_0>` usa o fato de que os 8 dims da
  lane L são exatamente `b->qh[lane & 3]` (bits `8*(lane&3)..+8`), um único byte
  por lane, sem máscara. A escrita é feita byte a byte (little-endian explícito)
  em vez de `memcpy`.

- **Resultado**: compila limpo (`-Wall -Wextra` no binário do motor: zero
  warnings novos; os `-Wswitch` que apareceram durante a edição foram o
  compilador apontando exatamente as cinco tabelas de `kv.h` que faltavam
  preencher). `build/check-kvquant-gpu` e `build/rdna4-infer` linkados.

- **Veredito**: MANTIDO (é a base de tudo o resto; sem número de desempenho
  próprio).

---

## 2. Validação dos formatos novos — oráculo CPU independente

- **Referência**: `libggml-base.so` **exporta** os quantizadores de referência do
  llama.cpp como símbolos C (`nm -D`: `quantize_row_q5_0_ref`, `quantize_row_q4_1_ref`,
  `quantize_row_q4_0_ref`, `quantize_row_q8_0_ref`, `dequantize_row_q5_0`,
  `dequantize_row_q4_1`, `dequantize_row_q4_0`, `dequantize_row_q8_0`;
  declarações em `ggml/src/ggml-quants.h:19-52`). Isso é melhor do que um espelho
  meu das minhas próprias fórmulas: é o código do llama.cpp, compilado por outro
  compilador, comparado byte a byte com o que a GPU grava.

- **Hipótese**: se `kv_store_row_kernel` for fiel, os bytes gravados são
  **idênticos** aos de `quantize_row_q5_0_ref`/`q4_1_ref`; e se os dois caminhos
  de load forem fiéis, `kv_load`/`kv_load8` reconstroem exatamente o que
  `dequantize_row_q5_0`/`q4_1` reconstroem daqueles bytes. Nenhuma das duas
  afirmações usa a saída do meu kernel como referência.

- **Comando**:
  `timeout 900 ./scripts/gpu-lock.sh ./build/check-kvquant-gpu` (sem modelo,
  ~1 s de GPU) e
  `timeout 900 ./scripts/gpu-lock.sh ./build/check-kvquant-gpu <IQ3_S>` (captura
  linhas K/V reais dos nós `Kcur`/`Vcur` do próprio grafo durante um forward).
  Janela: ver §2.1.

- **Resultado**: _(preenchido em §2.1 abaixo)_

- **Veredito**: _(ver §2.1)_
