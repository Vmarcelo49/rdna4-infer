# KV cache e tráfego de memória — análise e desenho

> **Escopo.** Análise de código e orçamento de tráfego do cache KV do `rdna4-infer`
> (Qwen3.8-27B, `gfx1201`, KV f16/q8_0/q4_0). Toda linha vem de arquivo lido
> (citação `arquivo:linha`) ou de aritmética mostrada; estimativas estão marcadas como
> **estimativa**. **Nada foi executado na GPU** (nem testes, nem `bench`): a medição é
> de outro agente, pela fila de `docs/gpu-queue.md`. Onde falta medição, o documento diz
> qual gate a provaria.
>
> Base: branch `feat/build-repro`, commit `aa15eea` (M8). Data: 2026-09-13.

---

## 1. Layout: contíguo, por camada, sem paging e sem indireção

### 1.1 Como o cache é alocado e endereçado

`include/rdna4/kv.h:22-27` define os tipos (`F32`, `F16`, `Q8_0`, `Q4_0`) e
`kv_row_bytes(t, head_dim)` (`kv.h:66-75`) dá o tamanho de **uma linha = um head de um
token**:

```cpp
const std::uint64_t blocks = (head_dim + 31) / 32;
case F32 : head_dim * 4;      case F16 : head_dim * 2;
case Q8_0: blocks * sizeof(block_q8_0);   // 8 blocos × 34 B
case Q4_0: blocks * sizeof(block_q4_0);   // 8 blocos × 18 B
```

Com `head_dim = 256` (`attention.key_length`, lido do GGUF) → `blocks = 8`:

| tipo | bytes/linha | bpw efetivo |
|---|---|---|
| f32 | 1024 | 32,0 |
| **f16** (default do llama.cpp) | **512** | 16,0 |
| q8_0 | 272 | 8,5 |
| q4_0 | 144 | 4,5 |

A alocação está em `graph.cuh:493-502`:

```cpp
// KV cache: `n_attn` caches of max_ctx rows, each row NKV heads of HD elements
const std::size_t kv_bytes = (std::size_t)max_ctx * NKV * (std::size_t)kv_row_bytes(kv_k_, HD);
kv_bytes_ = kv_bytes;
if (hipMalloc(&d_k_, (std::size_t)n_attn * kv_bytes) != hipSuccess ||
    hipMalloc(&d_v_, (std::size_t)n_attn * kv_bytes) != hipSuccess) { … }
```

Ou seja: **duas alocações** (`d_k_`, `d_v_` — `graph.cuh:285`), cada uma
`n_full_attn × kv_bytes_`, e o cache de uma camada é um bloco **contíguo** de
`max_ctx × n_head_kv` linhas. O índice da camada de atenção `il` vira um *slot*
compacto por `attn_slot()` (`graph.cuh:249-253`), que conta quantas camadas de atenção
plena existem antes dela:

```cpp
int attn_slot(int il) const { int a = 0; for (int i = 0; i < il; ++i) a += is_recr(i) ? 0 : 1; return a; }
```

O endereço de uma linha é, então, uma multiplicação tripla — `kv_write`
(`graph.cuh:694-711`) e a atenção (`attn.cuh:119,142`) usam **a mesma** fórmula:

```
endereço = base[d_k_ ou d_v_] + attn_slot(il) · kv_bytes_
                            + t · n_head_kv · kv_row_bytes(tipo)
                            + h · kv_row_bytes(tipo)
```

Consequências do layout, todas verificáveis no código:

- **Dentro de um token, os 4 KV heads são contíguos**: a "linha de token" tem
  `n_head_kv · row_bytes` = **2048 B (f16)**, 1088 B (q8_0), 576 B (q4_0).
- **Não há paging, block table, arena, indireção, nem tabela de slots de sequência.**
  Não existe identificador de sequência, nem `block_table`, nem realocação: o cache é um
  array `[camada][token][head][dim]` com strides fixos, dimensionado uma vez por
  `max_ctx`, e `reset_state()` (`graph.cuh:1221-1236`) só zera o estado GDN — o KV nem é
  limpo, porque a atenção é causal e toda linha é reescrita conforme a sequência avança
  (comentário em `graph.cuh:159-165`).
- A "página" que existe é de **1 token × 1 head** (512 B em f16), alinhada à linha de
  cache de 256 B desta placa (`rocminfo`: cacheline 256 B): `kv_load8<F16>` lê 16 B por
  lane e, com 32 lanes, **cobre exatamente uma linha de 512 B de forma contígua**
  (`kv.h:117-120,136-146`) — o mesmo argumento que o commit do M7 usou para tirar a
  atenção de 27 GB/s.

### 1.2 Tamanhos reais (aritmética com os números do modelo)

`n_full_attn = 16`, `n_head_kv = 4`, `head_dim = 256` (`graph.cuh:209-213`; conferido no
header do IQ3_S: `full_attention_interval 4`, `block_count 65`, `nextn_predict_layers 1`
→ 64 camadas de tronco, 16 de atenção plena `blk.3,7,…,63`). Por camada:
`max_ctx × 4 heads × row_bytes`; o total é isso × 16 camadas × 2 (K **e** V):

| tipo | ctx | por camada (K só) | **K+V total** |
|---|---|---|---|
| f16 | 4 096 | 8 388 608 B = 8,00 MiB | 268 435 456 B = 0,250 GiB |
| f16 | 65 536 | 134 217 728 B = 128,0 MiB | 4 294 967 296 B = **4,000 GiB** |
| f16 | 131 072 | 268 435 456 B = 256,0 MiB | 8 589 934 592 B = 8,000 GiB |
| q8_0 | 4 096 | 4 456 448 B = 4,25 MiB | 142 606 336 B = 0,133 GiB |
| q8_0 | 65 536 | 71 303 168 B = 68,0 MiB | 2 281 701 376 B = 2,125 GiB |
| q8_0 | 131 072 | 142 606 336 B = 136,0 MiB | 4 563 402 752 B = 4,250 GiB |
| q4_0 | 4 096 | 2 359 296 B = 2,25 MiB | 75 497 472 B = 0,070 GiB |
| q4_0 | 65 536 | 37 748 736 B = 36,0 MiB | 1 207 959 552 B = 1,125 GiB |
| q4_0 | 131 072 | 75 497 472 B = 72,0 MiB | 2 415 919 104 B = **2,250 GiB** |

**Conferência contra a medição de VRAM já existente** (`docs/medicoes-m5.md:140-155`):
IQ3_S 4K/f16 = 11,87 GiB e 131K/q4_0 = 13,86 GiB → diferença 1,99 GiB; pela tabela,
2,250 − 0,250 = **2,00 GiB** ✓ (0,5% de diferença, atribuível ao arredondamento da própria
tabela). O único caso que não cabe é IQ4_XS + 131K q4_0 (`medicoes-m5.md:155`), coerente
com 14,25 GB de pesos + 2,25 GiB de KV + ativações.

O buffer de *parciais* do split-KV é separado e minúsculo (`graph.cuh:302-304,534-538`):
`attn_partial_bytes(24, 256, 16)` = `24 × 16 × (2+256) × 4` = **396 288 B** (387 KiB),
alocado **uma vez** e reutilizado por toda camada e todo token
(`attn.cuh:498-500`).

### 1.3 Por que contíguo é a decisão certa aqui (e quando deixaria de ser)

O motor tem **um modelo, uma sequência, um pedido por vez**: o servidor é single-request,
sem keep-alive e **sem reúso de prefixo** (`README.md:149-150`, `docs/servidor-openai.md`).
Nesse regime, o que paging compraria é:

| recurso de paging | serve para | existe aqui? |
|---|---|---|
| block table / páginas de N tokens | várias sequências concorrentes com crescimento independente | não (uma sequência) |
| reúso de prefixo (copy-on-write de blocos) | cachear o prompt entre requisições | não (servidor sem prefix cache) |
| alocação sob demanda / fragmentação | servir N contextos de tamanhos diferentes | não (um `max_ctx` fixo) |

Em troca, o layout contíguo dá três coisas que este kernel *sente*:

1. **Zero indireção no caminho quente.** O endereço é 3 multiplicações + 1 soma; o kernel
   de atenção é *issue-bound* por byte lido (`docs/rocm-estudo.md` §B.2: 171 instruções por
   `vec_dot`) — somar um lookup de block table por linha seria trabalho no lugar mais caro.
2. **Linha de token contígua (2048 B em f16)** = os 4 heads caem em 8 linhas de cache
   consecutivas, e o par K/V de um token cabe em 2 chunks de 512 B por lane-warp.
3. **Alocação única por buffer** = `hipMalloc` de 4 GiB em duas chamadas, sem fragmentação
   nem espaçador: é o que faz "16K f16" caber com 3,77 GiB livres (`medicoes-m5.md:145`).

**Custo de mudar de ideia depois** (se o servidor ganhar prefix cache): a mudança é
*localizada* — todos os endereços passam por `kv_row_bytes()` + base, em exatamente dois
sítios (`graph.cuh:694-711` no write e `attn.cuh:119,142/317,338` no read) — mas ela
quebraria a propriedade (2) e obrigaria o kernel a carregar a block table. Vale registrar
como "paging-ready por acidente": a interface já é "base + strides", só não há tabela.

---

## 2. Leituras e escritas por token (decode)

Ponto de entrada: `forward_run` (`graph.cuh:1125-1219`), um token por vez, 64 camadas
(`graph.cuh:1158-1177`); 16 delas são `full_attn` (`graph.cuh:619-691`) e 48 são
`gdn_layer` (`graph.cuh:714+`). Tudo abaixo é **por token de decode**.

### 2.1 Escrita: `kv_write` — 8 lançamentos por camada, 64 KiB por token

`graph.cuh:694-711`:

```cpp
for (int h = 0; h < NKV; ++h) {
  kv_store_row_launch(kv_k_, d_ksrc + h*HD, krow + h*kv_row_bytes(kv_k_, HD), HD) ||
  kv_store_row_launch(kv_v_, d_vsrc + h*HD, vrow + h*kv_row_bytes(kv_v_, HD), HD)
}
```

O kernel (`kv.h:185-236`) quantiza **uma linha** de `head_dim` floats para o tipo do cache
(reproduzindo `quantize_row_q8_0_ref`/`quantize_row_q4_0_ref` byte a byte) e o launcher
(`kv.h:328-336`) usa 128 threads por linha. Ou seja:

| | valor |
|---|---|
| lançamentos por camada | 2 × `n_head_kv` = **8** |
| lançamentos por token | 8 × 16 = **128** (confere com `docs/rocm-estudo.md` §A.2.8) |
| bytes escritos por camada | 8 × row_bytes = 4 096 B (f16) / 2 176 B (q8_0) / 1 152 B (q4_0) |
| **bytes escritos por token** | **65 536 B (64 KiB) f16** · 34 816 B q8_0 · 18 432 B q4_0 |
| tráfego de leitura associado | `d_kstage_`/`d_vstage_` (f32, 4 KiB cada por camada) |

Os dois buffers de origem são `d_kstage_`/`d_vstage_` (`graph.cuh:284`, `NKV*HD` floats),
escritos pelas projeções `attn_k`/`attn_v` via `proj_qq` (`graph.cuh:630-631`) — o mesmo
caminho que o M8 otimizou (§3.4).

### 2.2 Leitura: a atenção — duas variantes, com e sem split

`attn.cuh` caminha as chaves `j = w; j <= t; j += WPB` no kernel sem split
(`attn.cuh:118`) e `j = w + WPB·s; j <= t; j += WPB·n_splits` no kernel com split
(`attn.cuh:315-316`), lendo `(krow + vrow)` por chave e por head de query. A política de
splits é `attn_splits_for()` (`graph.cuh:309-323`), com
`kAttnSplitMin = 512` (`graph.cuh:308`, ajustado no M8) e teto `kAttnMaxSplits = 16`:

| contexto (chaves = pos+1) | splits | grade do kernel | lançamentos/camada | lançamentos/token |
|---|---|---|---|---|
| 4 096 | 8 | 24 × 8 = 192 CTAs | 2 (split + merge) | **32** |
| 16 384 | 16 (teto) | 384 CTAs | 2 | 32 |
| 65 536 | 16 (teto) | 384 CTAs | 2 | 32 |
| < 512 | 1 | 24 CTAs | 1 (sem merge) | 16 |

Com split, o kernel escreve um parcial `(m, l, acc[head_dim])` por (head, split) —
`attn.cuh:380-397` — e `attn_merge_kernel` (`attn.cuh:401-428`) lê **todos** os parciais de
um head e combina com a mesma aritmética de online softmax. Bytes do vaivém dos parciais
(`attn_partial_bytes` = `n_head · n_splits · (2+head_dim) · 4`):

| ctx | splits | por camada (ida) | por token (ida) | ida + volta |
|---|---|---|---|---|
| 4 096 | 8 | 198 144 B | 3 170 304 B = 3,02 MiB | **6,34 MB** |
| 65 536 | 16 | 396 288 B | 6 340 608 B = 6,05 MiB | **12,68 MB** |
| 131 072 | 16 | 396 288 B | 6 340 608 B | 12,68 MB |

### 2.3 O inventário completo do que o cache toca por token

| # | operação (arquivo:linha) | lançamentos/token | bytes lidos | bytes escritos |
|---|---|---|---|---|
| 1 | `kv_write` K+V, 8 por camada (`graph.cuh:701-709`) | 128 | 8 KiB por camada (f32 staging) = 128 KiB | 64 KiB (f16) |
| 2 | atenção, leitura de K/V (`attn.cuh:118-150` / `:315-346`) | 16 (ou 32 com merge) | **1,611 GB lógicos a 4K** · 25,77 GB a 64K (f16) | — |
| 3 | parciais do split (`attn.cuh:380-397`) | (dentro do #2) | 3,02 MiB a 4K · 6,05 MiB a 64K | 3,02 MiB · 6,05 MiB |
| 4 | `attn_merge_kernel` (`attn.cuh:401-428`) | 16 a 4K+ | 3,02 MiB · 6,05 MiB | 16 × `NH·HD·4` = 393 KB |
| 5 | `deinterleave_q_gate` + sigmoid/mul do gate (`graph.cuh:635,682-684`) | 48 | 48 KB + 48 KB por camada | idem |
| 6 | LM head + `hipMemcpy` dos logits (`graph.cuh:1206-1212`) | 1 + 1 cópia | escrita 993 KB, lida 993 KB (D2H) | 993 KB |
| 7 | `hipMemcpy(d_pos_)` H2D **síncrono**, 16× | 16 cópias | 4 B cada | 4 B cada |
| 8 | estado GDN: `delta_rule` lê **e** escreve o estado inteiro (`gdn.cuh:57-86`) | 48 | **144 MiB** | **144 MiB** |
| 9 | linha do embedding (`dequant_row`, `graph.cuh:1146`) | 1 | 2 200 B (q3_K) | 20 KB |

O item 8 é o maior tráfego "não-peso" do motor e **não** é o KV: cada camada recorrente
guarda `nvh · S · S = 48 × 128 × 128` floats = **3,00 MiB** (`graph.cuh:539`), e o
`delta_rule_kernel` multiplica e reescreve essa matriz inteira a cada token
(`gdn.cuh:73-84`) → 48 camadas × 3 MiB × 2 = **302 MB por token** (2,7% do stream de pesos).
É também o estado que o MTP teria de fazer *checkpoint* (`docs/medicoes-m8.md:87` fala em
157 MB — o número de lá inclui conv state + estado; pela minha conta o estado é
`48 × 48×128×128 × 4 B = 144 MiB` e o conv state
`48 × (K−1) × chan = 48 × 3 × 10240 × 4 B = 5,6 MiB`).

---

## 3. Round-trips evitáveis (o que é escrito e lido de volta)

| # | vaivém | bytes/token | evitável? | como |
|---|---|---|---|---|
| 3.1 | **K/V escrito e lido de volta** pelo próprio token | 64 KiB escritos; a leitura do token `t` inclui a linha `t` (1 de `t+1`) | **não** (é o cache) | — |
| 3.2 | **Amplificação 6× do GQA** na leitura | 1,611 GB → 25,77 GB lógicos | **sim, parcialmente** | compartilhar a linha entre as 6 cabeças do grupo (§5, proposta 1) |
| 3.3 | **Parciais do split escritos e relidos pelo merge** | 3,02 MiB (4K) / 6,05 MiB (64K) por sentido | sim (parcial) | fundir o merge no kernel (última CTA) ou reduzir splits |
| 3.4 | **Re-quantização da ativação** | 192 de 305 quantizações eram redundantes | **já corrigido no M8** | `proj_qq()` (`graph.cuh:586`, `:630-631,738-740,800`) |
| 3.5 | **Cópia dos logits D2H + sync** | 993 KB + 1 sincronização | sim (no caminho greedy) | argmax no device |
| 3.6 | `hipMemcpy` de 4 B da posição, 16× **síncrono** | 64 B | sim | passar `pos` por valor para a RoPE |
| 3.7 | Cópia D2D de `d_inner` floats por token/camada recorrente, **só no prefill** (`graph.cuh:976-977`) | 24 KB × 48 = **1,18 MB por token de prefill** + 48 cópias | sim | dar à `ssm_out` uma origem com stride (é o `v_c` dentro do buffer de conv) |
| 3.8 | `d_kstage_`/`d_vstage_` f32 → cache quantizado | 128 KiB lidos por token | sim (fusão difícil) | epílogo fundido na projeção K/V (mexe em `matvec.cuh`) |
| 3.9 | `d_proj_` (12288 floats) escrito pela projeção Q e relido pelo `deinterleave` | 48 KB + 48 KB por camada | sim, mas caro | projetar Q já com q e gate em buffers separados — muda o `matvec` ou o peso |

Notas de qualificação (importantes para não exagerar):

- **3.2 é o único vaivém grande.** 6× de leitura redundante = 5/6 de 25,77 GB a 64K. Em
  compensação, o L2 desta placa é de **8 MiB** (`rocminfo`: `L2: 8192 KB`;
  `docs/rocm-estudo.md` §0.2)
  e as 6 cabeças de um grupo rodam em CTAs vizinhas da mesma grade (`dim3 grid(n_head,
  n_splits)`, `attn.cuh:439`, com `headIdx` como dimensão rápida) — a 1ª das 6 paga o DRAM e
  as outras 5 devem acertar o L2. É por isso que o número medido de L2 (≈1,35 TB/s, §4.3)
  convive com apenas ~225 GB/s de DRAM: **o 6× é hoje um custo de L2 + ALU de dequantização,
  não de DRAM**.
- **3.3 é barato em banda, caro em lançamento.** O buffer de parciais é **um só**,
  reutilizado por todas as camadas (396 288 B, `graph.cuh:534-538`) → ele vive quente em L2
  durante o token inteiro; o custo real é o lançamento extra do merge (16/token × 3,5 µs de
  gap de despacho = **~0,06 ms**, [estimativa]) mais a serialização (a CTA que espera o merge).
  Fundir merge + último split economiza 16 lançamentos e um ponto de sincronização de fluxo.
- **3.4 já foi feito**: `proj_qq` reaproveita os blocos `q8_1` entre projeções que leem a
  mesma ativação, com contrato no header (`graph.cuh:238-245`); medido em +1,3% de decode e
  384 lançamentos a menos por token (`docs/medicoes-m8.md:51-66`). Restam **113**
  quantizações por token (305 − 192) que são genuinamente necessárias.
- **3.5 é o vaivém mais "puro" que sobrou**: 993 KB de logits (`248320 × 4 B`) copiados
  device→host **por token**, com o `hipMemcpy` bloqueante como ponto de sincronização do
  pipeline (`graph.cuh:1194` e `:1211`; o custo medido de cópia+amostragem é ~0,55 ms, dos
  quais 40-60 µs são a cópia, `docs/rocm-estudo.md` §A.2.4). O que se paga é o **sync**, não
  o KB.
- **3.7 não é decode**: aparece só no caminho de prefill em batch (`forward_batch_layer`,
  `graph.cuh:976-980`), e o comentário no código já explica por que ele existe (o `v_c` está
  dentro do buffer de conv do token, não contíguo entre tokens).

---

## 4. Orçamento de tráfego por token

### 4.1 (a) Pesos

Do header do `Qwen3.8-27B-UD-IQ3_S.gguf` (parser próprio, só o cabeçalho; 866 tensores):

| parcela | bytes |
|---|---|
| soma dos tensores (é o `loader.total_bytes()`) | 12 029 886 464 = **12,030 GB** |
| − `token_embd.weight` (Q3_K, 546,3 MB) — no decode lê-se **uma linha** | −546 304 000 |
| − bloco MTP `blk.64.*` (351,0 MB) — não executa na v1 | −351 008 768 |
| + a linha de embedding efetivamente lida (q3_K, 5120 elem) | +2 200 |
| **= pesos lidos por token** | **11 132 575 896 B = 11,133 GB** |

Isso **confirma dentro de 0,1%** o número já documentado (11,122 GB em `docs/rocm-estudo.md`
§A.2.6) — mas **corrige a decomposição** de lá: o bloco MTP não tem "42,7 MB", tem **351 MB**
(medido no header), e `token_embd` tem 546 MB (não 556 MB). A conta que fecha é
12,030 − 0,546 − 0,351 = 11,133 GB.

Tipos por volume (úteis para saber onde 1 byte de peso dói mais, §B.3 do estudo):
IQ3_S 30,9% · IQ4_XS 21,6% · IQ3_XXS 15,5% · Q5_K 8,2% · Q3_K 7,9% · IQ2_S 4,8% (o resto
≤ 3% cada).

### 4.2 (b) KV lido por token, **com a duplicação do GQA**

Cada um dos 24 heads de query percorre as `t+1` chaves do seu grupo KV (`kvh = h / (n_head /
n_head_kv)`, `attn.cuh:100`), lendo K e V. Com `n_head/n_head_kv = 6`, cada linha é lida
**6 vezes**. Leitura lógica por token = `16 camadas × 24 heads × ctx × (krow + vrow)`:

| tipo | ctx | leitura **lógica** (L2) | linhas únicas (DRAM, se o L2 absorver o 6×) |
|---|---|---|---|
| f16 | 4 096 | 1,611 GB | 0,268 GB |
| f16 | 65 536 | **25,770 GB** | **4,295 GB** |
| q8_0 | 4 096 | 0,856 GB | 0,143 GB |
| q8_0 | 65 536 | 13,690 GB | 2,282 GB |
| q4_0 | 4 096 | 0,453 GB | 0,075 GB |
| q4_0 | 65 536 | 7,248 GB | 1,208 GB |
| q4_0 | 131 072 | 14,496 GB | 2,416 GB |

A coluna DRAM é o "pior caso favorável" (todo o 6× resolvido em L2); a coluna lógica é o
"pior caso absoluto" (nenhum reúso). O valor real está entre os dois e é medível
(§6, gate do `bench-attn-gpu`).

### 4.3 (c) e (d) Ativações, outros, e o total

| parcela | 4K f16 | 64K f16 | 131K q4_0 |
|---|---|---|---|
| pesos | 11,133 GB | 11,133 GB | 11,133 GB |
| KV escrito | 0,066 MB | 0,066 MB | 0,018 MB |
| KV lido (DRAM, com 6× no L2) | 0,268 GB | 4,295 GB | 2,416 GB |
| KV lido (lógico/L2) | 1,611 GB | 25,770 GB | 14,496 GB |
| parciais do split (ida+volta) | 6,34 MB | 12,68 MB | 12,68 MB |
| estado GDN (ida+volta) | 302 MB | 302 MB | 302 MB |
| logits (escrita + cópia D2H) | 1,99 MB | 1,99 MB | 1,99 MB |
| **total DRAM** | **11,71 GB** (105% dos pesos) | **15,74 GB** (141%) | **13,87 GB** (125%) |
| **total lógico (L2+DRAM)** | **13,05 GB** (117%) | **37,22 GB** (334%) | **25,95 GB** (233%) |

Contra a banda medida (`README.md:97`, `docs/medicoes-m5.md:161-164`): efetiva de
**336,2 GB/s** (IQ3_S) / **364 GB/s** (IQ4_XS), teto de leitura medido **619 GB/s**
(`PLAN.md:102,111`, `docs/rocm-estudo.md` §B.1), pico teórico 640 GB/s:

| contexto | total DRAM | a 336 GB/s | a 619 GB/s | medido (tok/s) | leitura |
|---|---|---|---|---|---|
| 4K f16 | 11,71 GB | 34,8 ms | 18,9 ms | 26,8-29,3 → 34-37 ms (`README.md:94`, `medicoes-m8.md:61-62`) | **dominado pelos pesos** |
| 64K f16 | 15,74 GB | 46,8 ms | 25,4 ms | 18,91 → 52,9 ms (`medicoes-m7.md:107`) | pesos 33 ms + atenção 19 ms |
| 131K q4_0 | 13,87 GB | 41,2 ms | 22,4 ms | 12,99 → 77,0 ms (`medicoes-m7.md:109`) | idem + margem |

### 4.4 O que é limitado por banda e o que é limitado por latência/issue

Usando os tempos decompostos já medidos (`docs/rocm-estudo.md` §A.1 e `medicoes-m7.md:145-151`):

| componente | ms/token (4K) | ms/token (64K) | natureza |
|---|---|---|---|
| pesos (497 tensores, medida por forma real) | 26,4-27,7 | 26,4-27,7 | **banda de DRAM** (402-421 GB/s medidos; 336 GB/s no total do token) |
| dos quais LM head (874 MB) | 1,39 (627 GB/s) | 1,39 | **banda, no teto** (98% dos 619 GB/s) |
| atenção (16 camadas) | 1,2 (8 splits) | 19,0 | **issue-bound na dequantização**, não banda |
| logits + cópia + sampler | ~0,5 | ~0,5 | **latência/sync** (drena o pipeline) |
| ~1900 kernels pequenos (norms, GDN, rope, `kv_write`, quantizações) | ~2-3 | ~2-3 | **latência de despacho** (gap de 3,5 µs) |

A atenção merece o número mais forte desta análise. Leitura lógica e tempo medido dão,
**nos dois contextos**, a mesma taxa de L2:

```
4K : 1,611 GB / 1,2 ms  = 1,34 TB/s      (16 camadas, 8 splits, docs/rocm-estudo.md §A.1)
64K: 25,770 GB / 19,0 ms = 1,36 TB/s     (1,19 ms/camada × 16, medicoes-m7.md:147-151)
```

e a mesma taxa de DRAM efetiva, ~225 GB/s:

```
4K : 0,268 GB / 1,2 ms  = 224 GB/s
64K: 4,295 GB / 19,0 ms = 226 GB/s
```

Ou seja: **o kernel de atenção satura em ~1,35 TB/s de L2 independentemente do contexto, e
só chega a ~225 GB/s de DRAM porque 6× do que ele lê é redundante.** O piso de DRAM da
atenção a 64K f16 é 4,295 GB / 619 GB/s = **6,9 ms** contra 19,0 ms medidos → **~12 ms por
token são ALU de dequantização + tráfego de L2**, não memória. É exatamente aí que as duas
alavancas restantes moram (menos L2 por byte útil, ou menos instruções por byte — §B.2 do
estudo).

Confirmação cruzada por tipo de KV: a 64K, `q4_0` move 1,21 GB de DRAM contra 4,29 GB do f16
(3,5× menos bytes!) e ainda assim é **mais lento** (17,88 vs 18,91 tok/s, `medicoes-m7.md:107-108`)
— prova experimental de que a atenção é *issue-bound na desquantização*, e que **q4_0 é
alavanca de VRAM, não de velocidade** (`README.md:138`, `docs/rocm-estudo.md` §A.2.2).

Resumo da atribuição: em 4K, **~95% do tempo é banda de pesos** (34,8 dos 36-37 ms);
a partir de ~16K a atenção cresce (19 ms a 64K) mas os pesos continuam sendo a maior parcela
única; os itens de latência/despacho somam ~3-4 ms/token em qualquer contexto.

**Atenção ao somar a tabela acima:** ela **não** fecha por adição, e a razão é informativa —
o `bench-matvec-shapes-gpu` mede os 497 matvecs **isolados** a 402-421 GB/s (26,4-27,7 ms),
enquanto dentro do motor os mesmos pesos andam a **336 GB/s** (11,133 GB → **33,1 ms**). A
diferença (~6 ms) não é trabalho extra: é a interferência e o gap de despacho dos ~1900
kernels pequenos que se intercalam entre os matvecs (`docs/rocm-estudo.md` §B.5). Para o
orçamento, o número de pesos que vale é o **de dentro do motor** (33,1 ms), não o do replay.

---

## 5. Propostas, ranqueadas por (ganho esperado × confiança) / esforço

Legenda: **[G]** = ganho esperado (estimativa, com a base) · **[C]** = confiança ·
**[E]** = esforço · a última coluna diz se mexe em `graph.cuh` (o arquivo é de outro agente,
então precisa entrar na fila do coordenador).

| # | proposta | [G] | [C] | [E] | arquivos | gate | `graph.cuh`? |
|---|---|---|---|---|---|---|---|
| 1 | **Compartilhar a linha KV entre as 6 cabeças do grupo GQA** (grade já larga: 24×16 = 384 CTAs) | até **+18% a 64K** (19,0 → ~7-10 ms; teto de DRAM 6,9 ms), ~0 a 4K | média — o protótipo do M7 foi 8-12× **mais lento** por colapso de grade; agora a grade é larga, mas o risco de pressão de registrador é real (6 q + 6 acc × 8 dims = 96 VGPR só de estado) | grande | `attn.cuh` (+ `tests/bench_attn_gpu.hip`) | `bench-attn-gpu` (A/B no mesmo processo) + `scripts/check_attn_split.sh` (PPL) + `check-kvctx-gpu` | **não** (só o kernel; só mexeria no `graph.cuh` se a política de splits/WPB mudar) |
| 2 | **Escrever os 4 heads de K e de V num único lançamento** (`kv_store_rows_launch`) | −120 lançamentos/token ≈ **0,42 ms (1,2%)** [estimativa: 120 × 3,5 µs] | alta (a fusão preserva a aritmética por linha: mesmo kernel, mesmo tipo) | pequeno | `kv.h` (helper novo) + `graph.cuh:701-709` | `check-graph-gpu` (oráculo por nó, bit-exato) + `scripts/check_golden_run.sh` | **sim** |
| 3 | **Fundir o merge do split no último split** (ou em um kernel que faça merge + gate sigmoid + mul) | 16 lançamentos + 1 ponto de sync/token ≈ 0,1-0,2 ms; tira 6,34 MB de vaivém a 4K (que hoje é L2, não DRAM) | média (muda a ordem da soma → deixa de ser bit-exato; mesmo regime do M7) | médio | `attn.cuh`, `graph.cuh:671-680` | `scripts/check_attn_split.sh` + `check-kvctx-gpu` | **sim** |
| 4 | **Argmax no device no caminho `--greedy`** (mata a cópia de 993 KB e o sync do token) | **0,3-0,5 ms (1-1,5%)** (§A.2.4) | alta para o ganho, média para a exatidão (em empate de logits o desempate tem de ser o mesmo do host) | pequeno-médio | kernel em `nn.cuh`/`matvec.cuh` + `graph.cuh`/`main.hip` | `scripts/check_golden_run.sh` + `compare_llama_greedy.sh` (greedy token a token) | **sim** |
| 5 | **Passar `pos` por valor para a RoPE** (elimina 16 `hipMemcpy` síncronos de 4 B/token) | 0,05 ms (0,15%), mas tira 16 sincronizações do caminho quente | alta (bit-exato) | trivial | `attn.cuh` (firma do kernel) + `graph.cuh:651-657` | `check-rope-gpu` + `check-graph-gpu` | **sim** |
| 6 | **Fundir a cadeia escalar do GDN** (sigmoid+add+softplus+mul = 4 lançamentos de 48 elementos) | ~0,5 ms (1,4%) | alta (bit-exato: mesmas ops, mesma ordem) | pequeno | kernel novo em `gdn.cuh` + `graph.cuh:746-748` | `check-graph-gpu` + golden run | **sim** |
| 7 | **Fundir `rms_norm` + `quantize_q8_1`** (a norma já tem o valor em registrador) | ~0,45 ms (1,3%) — 130 lançamentos | alta (o próprio comentário do estudo exige manter a ordem de soma da norma) | médio | `nn.cuh` + `graph.cuh` (o primeiro `proj` de cada camada) | `check-graph-gpu` (bit-exato) | **sim** |
| 8 | **Staging em LDS da linha KV dentro do CTA** (variante do #1 que evita 96 VGPR: uma linha de 512 B por vez, 6 cabeças consumindo) | mesmo teto do #1, com menos risco de registrador | baixa-média (não há `global_load_lds` em gfx1201 — o staging tem de passar por registrador, e o `__syncthreads` por chave pode custar mais que o ganho) | grande | `attn.cuh` | `bench-attn-gpu` primeiro (experimento isolado) | **não** |
| 9 | **Tirar a cópia D2D do prefill** (`graph.cuh:976-977`) | 1,18 MB/token de prefill + 48 cópias → ~0 no tempo (2 µs de banda) | alta | médio (mexe no `ssm_out` batched) | `graph.cuh` + `mtp`/batch | `check-batch-gpu` (bit-exato) | **sim** |
| 10 | **Paging / block table** | 0 para este workload | — | grande | `kv.h` + `graph.cuh` + `attn.cuh` | só faz sentido junto com prefix cache no servidor | **sim** |

**Rejeitados com motivo (para não repetir):**

- **Inverter o layout para head-major** (`[head][token][dim]`): deixaria a caminhada de um
  head com stride `row_bytes` em vez de `n_head_kv · row_bytes`, mas destrói (a) a linha de
  token contígua de 2048 B que dá o acesso coalescido de 512 B por warp
  (`kv.h:117-120`) e (b) a localidade entre as 6 cabeças do grupo — que é justamente o
  mecanismo que hoje faz o 6× cair no L2 e não no DRAM.
- **Guardar uma cópia f32/f16 além da quantizada**: custa exatamente a VRAM que o `q4_0`
  existe para economizar (§A.2.2) — a 131K seriam +2,25 GiB (f16 ao lado de q4_0).
- **Reduzir splits para economizar o merge**: medido — a 4K sair de 8 splits para 1 custa
  23,48 → 26,92 tok/s (+14,7% com 8 splits, `docs/rocm-estudo.md` §E.4 e `medicoes-m7.md:105`).
  A economia de 16 lançamentos não paga a perda de paralelismo.
- **Padding entre camadas / entre K e V** para evitar conflito de conjunto no L2: no caso
  f16 (`max_ctx × 4 × 512 B`) o stride entre camadas é uma potência de 2 (8 MiB a 4K,
  128 MiB a 64K), o que em teoria alinha todas as camadas nos mesmos conjuntos. **Não**
  considerei problema: o laço de camadas é sequencial (uma camada por vez,
  `graph.cuh:1158-1177`), então não há duas camadas em voo disputando conjunto. Se algum dia
  houver (prefill em batch com camadas sobrepostas), medir antes de mudar.

---

## 6. O que este desenho **não** prova (e o gate que provaria)

| afirmação neste doc | status | como medir |
|---|---|---|
| 6× do GQA cai no L2 (coluna "DRAM" do §4.2) | **hipótese** sustentada pela taxa de 1,35 TB/s medida | contadores de L2 via `bench-attn-gpu` variando o número de heads por grupo; ou comparar tempo do kernel com 4 heads vs 24 heads na MESMA grade |
| parciais do split ficam em L2 (396 KB reutilizados) | **estimativa** (o buffer é 387 KiB contra 8 MiB de L2) | `bench-attn-gpu` com e sem o merge, medindo só o kernel de merge |
| atenção = 19 ms a 64K dos quais ~6,9 ms são DRAM | **estimativa** por aritmética | `bench-attn-gpu f16 65536` com um tensor sintético (o bench já aquece a GPU e isola o kernel) |
| ganho de +18% a 64K com GQA sharing | **estimativa** (§C.7 do estudo e teto de DRAM) | protótipo no `bench-attn-gpu` antes de tocar no `attn.cuh` |
| custo do merge = 16 × gap de despacho | estimativa | `bench --start-pos 65000` com e sem merge fundido, via `scripts/check_attn_split.sh` |

Os gates citados já existem no repositório: `tests/bench_attn_gpu.hip` (mede o kernel
isolado, com A/B no mesmo processo), `scripts/check_attn_split.sh` (PPL split vs unsplit em
texto real, 5,1989 vs 5,1917 = 0,14%, `medicoes-m7.md:132-143`), `check-kvctx-gpu`
(split vs unsplit, contexto longo), `check-graph-gpu` (oráculo por nó, bit-exato) e
`bench --start-pos` (decode no fim do contexto).

---

## 7. Divergências com os documentos existentes

1. **`docs/rocm-estudo.md` §A.2.6** diz que o bloco MTP tem "42,7 MB" e `token_embd` 556 MB.
   Medido no header do IQ3_S: MTP `blk.64.*` = **351 008 768 B (351 MB)** e
   `token_embd.weight` = **546 304 000 B (546 MB)**. O total (11,122 GB de pesos por token)
   continua certo dentro de 0,1% (meu número: 11,133 GB); a decomposição é que estava errada.
2. **`include/rdna4/device.h:33`** tem um comentário obsoleto ("**17** full-attention
   layers") logo acima dos dois comentários que o corrigem para 16 (`device.h:37-40`), e a
   constante (`kQwen35KvElemsPerToken = 16·4·(256+256)`, `device.h:40`) está certa. É
   vestígio do erro que o `PLAN.md:174` registra. Vale limpar o parágrafo.
3. **A atenção não é mais o gargalo de contexto que `medicoes-m5.md:191` descreve**
   ("a 64K um passo de decode custa ~262 ms") — isso é o baseline pré-M7; o M7 fechou em
   19 ms de atenção (53 ms/token a 64K, `medicoes-m7.md:107`). O texto do M5 continua no
   repositório sem nota de revisão.
4. **`README.md:132-137`** atribui o ganho do M7 a "32 warps/CTA", mas `attn.cuh:82` fixa
   `kAttnWarpsPerBlock = 8` desde o M7 e o próprio estudo registra isso como rejeitado
   (`docs/rocm-estudo.md` §D.7: "`attn.cuh:82` nunca teve o valor trocado"). O ganho real
   veio do `kv_load8` vetorizado + do split-KV. (Já apontado pelo estudo; reforço aqui porque
   o README é a porta de entrada.)
5. **`medicoes-m7.md:86`** descreve a política de splits como `keys / 2048` (era o valor no
   momento daquela medição); o código de hoje é `keys / 512` (`graph.cuh:308`, ajustado no
   M8: +14,7% a 4K).
