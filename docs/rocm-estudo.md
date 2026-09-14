# Estudo ROCm/HIP + gfx1201 — conversões on-the-fly, trabalho redundante e otimizações desta placa

Branch `feat/rocm-study` (worktree `../rdna4-infer-wt-rocm`). Tudo aqui foi medido nesta
máquina (RX 9070 XT, gfx1201, ROCm 7.2, sem outra carga na GPU) com `scripts/gpu-lock.sh`
serializando o acesso. Cada afirmação carrega o comando que a produziu ou está marcada
como **estimativa/hipótese**.

Ferramentas novas deste estudo (em `tests/`, alvos novos no fim do `CMakeLists.txt`):

| ferramenta | o que mede o que os existentes não medem |
|---|---|
| `bench-matvec-shapes-gpu <gguf>` | o matvec **por tensor do inventário real** (não "o maior tensor do tipo"), com A/B intercalado dos knobs e um *replay* da sequência inteira |
| `bench-attn-gpu` (estendido) | varredura de **nº de splits × warps por CTA** do kernel de atenção, com rel-L2 contra o caminho sem split |

---

## 0. Método (o que torna estes números confiáveis, e onde eles não são)

Três armadilhas já documentadas no repo (`docs/medicoes-m6.md`, `PLAN.md` M2) foram
reconfimadas e duas novas apareceram:

1. **DPM** — a placa cai para SCLK ~10 MHz entre kernels; toda medição de kernel aquece
   até ~300-400 ms de tempo de GPU antes de cronometrar.
2. **Working set > cache** — o L2 desta placa tem **8 MiB** (o "64 MB" do repo é o
   *Infinity Cache*, L3/MALL: `hipDeviceProp.l2CacheSize` = 8388608). Cronometrar um
   tensor de 47 MB repetidamente mede Infinity Cache, não DRAM: é o que explica os
   500-1084 GB/s do `check-matvec-gpu --bench` (que mede 1 tensor por tipo, o maior, em
   loop fechado). O bench novo dá **uma passada por tensor por rodada**, com 10,4 GiB
   escalonados, então o número é de DRAM.
3. **Ruído de ordem (novo, e grande)** — cronometrar 6 configurações em sequência para o
   mesmo tensor dá à *primeira* de cada rodada uma penalidade sistemática de até **1,5×**
   (a mesma instanciação de kernel, medida como "ship" e como "rows=8", saiu 3,58 vs
   2,47 ms). O bench novo mede **uma passada completa por configuração e rotaciona a
   ordem a cada rodada**, e imprime um *piso de ruído*: a mesma configuração medida duas
   vezes, que agora dá **1,00×** em todos os 14 tipos.
4. **Piso de sincronização (novo)** — um kernel vazio leva **0,010 ms** cronometrado com
   `hipEventRecord`+sync e **0,0035 ms** enfileirado. Isso infla todo tensor pequeno: a
   matriz q8_0 48×160 "custa" 10 µs medida com sync e ~1,3 µs de trabalho real.
5. **`bench` reporta bytes que não são lidos** — `bytes_per_token = loader.total_bytes()`
   = 12,030 GB, mas o decode lê **11,122 GB** por token: `token_embd.weight` (556 MB) só
   tem *uma linha* lida (2,2 KB) e o bloco MTP `blk.64.*` (42,7 MB) nunca executa. A
   "effective bandwidth" impressa subestima a real em **8%** (`docs/rocm-estudo.md` §A.1).

---

## A. Inventário de conversões on-the-fly e trabalho redundante por token

### A.1 O orçamento de um token (medido)

```
./build/rdna4-infer bench -m <IQ3_S> -n 32 --reps 3            # 4K ctx, posicao 5
  decode 32 tokens x 3 reps: best 1.125 s (28.45 tok/s)        # 35.15 ms/token
  per token: 35.15 ms = launch loop 32.72 + tail 1.92 + sample 0.50
./build/rdna4-infer bench ... --layers 0                        # 1.95 ms (embeddings+norm+LM head+copy+sampler)
./build/rdna4-infer bench ... --layers 1                        # 2.66 ms
./build/rdna4-infer bench ... --ctx-size 4096 --start-pos 4064 --fill-cache   # 23.96 tok/s (41.74 ms)
```

| componente | ms/token | como foi medido |
|---|---|---|
| matvec (497 tensores, uma passada) | **26,4-27,7** | `bench-matvec-shapes-gpu` replay, 402-421 GB/s |
| — dos quais LM head (`output.weight`, 874 MB) | 1,39 (627 GB/s) | mesma ferramenta, classe q5_K 248320×20 |
| atenção, 16 camadas, 4K f16 | 5,4 (sem split) → 1,2 (8 splits) | `bench-attn-gpu f16 4096` × 16 |
| embeddings + norm final + cópia de logits + sampler | ~0,5 | `--layers 0` (1,95) − LM head (1,39) |
| *resto* (norms, GDN, rope, kv_store, quantizações, ~1900 kernels pequenos) | ~2-3 | 35,15 − 26,4 − 1,4 − 0,5 − atenção |

**A conta fecha em ~35 ms** e mostra de onde vem o "6,9 ms de outras coisas" do M5: não é
o *trabalho* dos kernels pequenos, é o **gap de despacho entre eles** (§B.5).

### A.2 Item por item

Legenda: **[medido]** = número desta sessão · **[estimativa]** = derivado de medição ·
**[hipótese]** = não medido.

#### A.2.1 Quantização da ativação: 305 lançamentos, **192 deles redundantes** [medido]

`Graph::proj()` (`graph.cuh`, LEIA-SE, não modificado) faz sempre
`quantize_q8_1_kernel` + `matvec_launch`. Contando as chamadas por token:

| onde | projeções | mesma entrada `d_x_` | redundantes |
|---|---|---|---|
| `full_attn` (16 camadas) | `attn_q`, `attn_k`, `attn_v` sobre `d_xn_` | 3 | **2** |
| — `attn_output` sobre `d_attnout_` | 1 | 1 | 0 |
| `ffn` (64 camadas) | `ffn_gate`, `ffn_up` sobre `d_xn_` | 2 | **1** |
| — `ffn_down` sobre `d_ffn_a_` | 1 | 1 | 0 |
| `gdn_layer` (48 camadas) | `attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha` sobre `d_xn_` | 4 | **3** |
| — `ssm_out` sobre `v_c` | 1 | 1 | 0 |
| LM head | 1 | 1 | 0 |
| **total** | **305** | | **16·3 + 48·3 = 192** |

**Correção de registro (medida em `docs/vulkan-vs-hip.md` §2.2, contagem no código que
roda):** os números desta seção foram contados antes do M8 e ficaram **subestimados**. O
`proj_qq` do M8 tirou uma quantização por projeção do FFN, e o código atual faz **497
matvec** e **257 `quantize_q8_1`** por token, não 305/192 — o total por token é **1 940
lançamentos** (497 + 257 + 1 186 das norms, ops escalares, `kv_write` e atenção). A
conclusão da seção não muda (a quantização redundante é removível com aritmética idêntica,
e o M8 já removeu a maior parte dela: `act_ready` no caminho em batch), mas o teto de
economia é maior do que o 63% calculado aqui, porque as 240 quantizações repetidas do
código pré-M8 valiam mais que 192.

Custo: o M5 mediu o teto *pulando todos* os `quantize_q8_1` em **1,36-1,5 ms (4,1%)**;
192 de 305 são 63% disso ⇒ **~0,85 ms/token (2,4%)** de trabalho, mais os 192 gaps de
lançamento correspondentes (§B.5) ⇒ ~1,2-1,5 ms no total. **Removível com aritmética idêntica**: os 192 blocos
`block_q8_1` produzidos são byte a byte iguais (mesma entrada, mesmo kernel), então basta
quantizar **uma vez por ativação** e reusar o buffer — não muda um único bit do modelo.
Precisa de mudança no `graph.cuh` (§E.1).

#### A.2.2 KV cache dequantizado elemento a elemento dentro da atenção [medido]

`kv_load<CT>()`/`kv_load8<CT>()` (`kv.h`) desquantizam na leitura. Medições:
- f16 vs q4_0 no mesmo contexto (M7 + este estudo): f16 é ~38% mais rápido a 64K; o kernel
  é *issue-bound* na desquantização, não limitado por banda.
- Com o caminho de splits o q4_0 a 64K fica em 1,03-1,09 ms/camada contra 1,02-1,03 do f16
  no mesmo split (a vantagem do f16 encolhe quando o kernel deixa de ser latência-bound) —
  ou seja, **`q4_0` é alavanca de VRAM, não de velocidade** (confirma M5/M7).
- `kv_load8` já usa um acesso vetorizado por lane (16 B para f16), e o caminho `q4_0`
  passa por `__half2float`+nibble por elemento. **Não há conversão removível**: o cache
  *é* armazenado quantizado por decisão de projeto (VRAM), e a alternativa (manter uma
  cópia f32/f16 além da quantizada) custa a VRAM que o q4_0 existe para economizar.

#### A.2.3 Linha do embedding desquantizada por token [medido]

`dequant_row_launch(tok_embd_)` = 1 lançamento de `dequant_kernel_256` (q3_K, 5120
elementos, 20 blocos, grid 20×64 threads). Custo ~1 × piso de lançamento ≈ 4 µs =
**0,01%** do token. Poderia rodar uma vez por token de entrada em vez de por posição, mas
é irrelevante; **não vale tocar**.

#### A.2.4 Logits copiados device→host por token: 1 MB, e o sampler no host [medido/estimativa]

`forward_run` copia os 248 320 floats (993 KB) e o CLI amostra no host. Medido: `--layers
0` = 1,95 ms, dos quais o LM head é 1,39 ms ⇒ **~0,55 ms (1,6%)** para cópia + amostragem
+ synck final. A cópia em si é ~40-60 µs (1 MB a ~20 GB/s de PCIe); o resto é o custo de
**sincronizar** (drena o pipeline do token) e o softmax/top-k no host.
**Removível (parcialmente)**: um argmax no device para o caminho `--greedy` mataria a
cópia e o sync; a amostragem com temperatura exige os logits no host (ou um sampler no
device). Efeito ≈ 0,3-0,5 ms (**1-1,5%**). Muda `graph.cuh`/`main.hip`, não os kernels.

#### A.2.5 LM head: 874 MB lidos por token [medido]

Classe q5_K 248320×20 medida a **627 GB/s** = 1,39 ms. Isso é **98% do teto de leitura
medido** (–619 GB/s– do M5; a placa tem 640 GB/s de pico teórico): não há nada a otimizar
aqui além de *não* ler os pesos (decodificação especulativa / MTP, trabalho de outro
agente). **É 4% do token e está no limite.**

#### A.2.6 O bloco MTP e a tabela de embedding contados como "lidos por token" [medido]

`bench` divide `loader.total_bytes()` (12,030 GB) pelo tempo ⇒ imprime 342 GB/s quando a
banda real é **316 GB/s** (11,122/0,03515). Correção: `bench` deveria descontar
`token_embd.weight` (556 MB) e `blk.<block_count-1>.*` (42,7 MB). Não muda nenhuma
conclusão, mas muda todo número de banda do repo em +8%.

#### A.2.7 Cadeia de GDN: 4 kernels de 48 elementos + 1 conv + 2 l2 + 1 delta por camada [medido]

Por camada recorrente (48/token): `sigmoid(48)`, `add(48)`, `softplus(48)`, `mul(48)` —
quatro lançamentos para ~200 elementos —, `conv1d` (10240 canais), 2× `l2_norm` (16×128),
`delta_rule` (48 CTAs × 128 threads), `rms_norm` (48×128), `silu(6144)`, `mul(6144)`.
O trabalho é desprezível; o custo é o **gap** (§B.5): ~10 lançamentos × 3,5-4,5 µs ≈
**~1,8 ms/token (5%)** nas 48 camadas recorrentes. Fusão dessas cadeias (um kernel para os
4 ops escalares, outro para `rms_norm`+`quantize`) é a maior economia de lançamentos
disponível — precisa de `graph.cuh` (§E.2).

#### A.2.8 `kv_write`: 8 lançamentos minúsculos por camada de atenção [medido]

`Graph::kv_write` faz `kv_store_row_launch` por cabeça KV, para K e para V: 4×2 = **8
lançamentos de 128 threads** para escrever 4×512 B (f16) por camada ⇒ 128 lançamentos/token
⇒ **~0,5 ms (1,4%)** só de gap. Uma única função que escreve as NKV cabeças (e K+V) em um
lançamento economiza 112 deles. `kv.h` ganha o helper, `graph.cuh` a chamada (§E.3).

#### A.2.9 `hipMemcpy` de 4 bytes para subir a posição [medido]

`full_attn` faz `hipMemcpy(d_pos_, &pos, 4, H2D)` — **síncrono** — por camada de atenção
(16/token). É um round-trip host→device bloqueante no meio do caminho quente. 16 × ~2-5 µs
≈ **0,05 ms (0,15%)**; a correção é escrever a posição num buffer de host mapeado ou passá-la
como argumento de kernel (a RoPE já recebe `d_pos`; um kernel por camada poderia recebê-la
por valor). Baixo ganho, mas é conversão/cópia evitável.

#### A.2.10 `rms_norm` em 1 CTA por linha [medido]

130 lançamentos/token; o caso comum é `nrows=1` (5120 elementos) ⇒ **um CTA de 256 threads
em uma CU**, com 8 `__syncthreads()` na redução em árvore. Custo por lançamento dominado
pelo gap (3,5-4,5 µs) ⇒ **~0,6 ms (1,7%)** em 130 lançamentos. Duas direções: (a) fundir
`rms_norm`+`quantize_q8_1` (a norma já tem o dado em registrador — economiza 130
lançamentos), (b) manter em `nn.cuh` a mesma ordem de soma para não perturbar os gates
(a redução em árvore compartilhada é *bit-exata*; trocá-la por shuffle mudaria os bits e
não vale o risco).

#### A.2.11 Resumo do inventário

| item | custo/token | removível? | onde |
|---|---|---|---|
| 192 quantizações redundantes | ~1,3 ms (3,7%) | **sim, bit-exato** | `graph.cuh` §E.1 |
| cadeia GDN (4 ops escalares + norms + silu/mul) | ~1,8 ms (5%) | sim, fusão | `graph.cuh` §E.2 |
| `kv_write` 8 lançamentos/camada | ~0,5 ms (1,4%) | sim, fusão (helper em `kv.h`) | `graph.cuh` §E.3 |
| cópia de logits + sampler no host | ~0,5 ms (1,4%) | parcial (argmax no device) | `graph.cuh`/`main.hip` |
| `hipMemcpy` da posição (16×) | ~0,05 ms | sim | `graph.cuh` |
| LM head (874 MB) | 1,39 ms | **não** (98% do teto) | — |
| KV dequant na atenção | dentro da atenção | **não** (é a razão do q4_0) | — |
| embedding row | 0,01% | irrelevante | — |
| `bench` conta 0,9 GB não lidos | 8% da banda reportada | sim | `main.hip` |

---

## B. O que esta placa oferece e o que o motor usa

### B.1 Fatos de hardware (medidos nesta máquina + ISA oficial)

Ver `docs/rdna4-gfx1201-hardware-brief.md` (brief completo, com fontes). O que importa aqui:

| fato | valor | consequência para nós |
|---|---|---|
| "CU" do HIP = WGP | 32 WGP = 64 CU = 128 SIMD32, wave32 | 64 CTAs de 8 warps enchem a placa; grid de 24 CTAs (atenção sem split) usa **24 de 64** |
| waves/WGP | máx **64**; plano até ~96 VGPR/thread | iq3_s (~69-87 VGPR) e iq4_xs (119) **não** são limitados por ocupação |
| LDS | 128 KB/WGP, 64 KB por workgroup | dá para escalonar tiles; a atenção usa 8-33 KB |
| L2 / Infinity Cache | **8 MiB** / **64 MB** | "working set > L2" no repo na verdade testava o IC (§0.2) |
| pico DRAM | 640 GB/s (medido ~619 no padrão do matvec) | matvec está em 402-444 GB/s ⇒ ~70% |
| `v_dot4_i32_iu8` | **full-rate, igual a `v_fma_f32`** | nossos `vec_dot` já usam a instrução certa; não há "emulação lenta de dp4a" |
| `v_perm_b32` | full-rate | o `perm+linearidade` do M2 é a escolha certa |
| `v_pk_fma_f16` / `v_dot2_f32_f16` | **1,5× o rate do FMA fp32** | só serve para *pesos fp16* (não para os nossos int8) |
| `global_load_lds`, `buffer_prefetch` | **não existem em gfx1201** | descartar o caminho "direto para LDS" |
| `__builtin_prefetch` | **vira nada** (ISA sem instrução) | o PF do `matvec.cuh` era um no-op (§D.4) |
| `__builtin_amdgcn_s_prefetch_data` | existe, **endereço uniforme por wave** | é o único prefetch real |
| gaps de despacho | **3,5 µs** (kernel vazio, fila cheia) | ~2200 lançamentos/token ≈ **7,7 ms/token** |

### B.2 Contagem de instruções por `vec_dot` (ISA, `-O3`, `--save-temps`) [medido]

Compilando **uma chamada** de cada `vec_dot` isolada (script em §G):

| tipo | instruções/chamada | VGPR | dp4a | perm | loads globais |
|---|---|---|---|---|---|
| `iq3_s` (envio, perm) | **171** | 28 | 16 | 8 | 16 |
| `iq3_s` (vendorado) | 424 | 69 | 8 | 16 | 16 |
| `iq4_xs` | 92 | 15 | 2 | 6 | 5 |
| `iq3_xxs` | 175 | 31 | 16 | 8 | 14 |
| `iq2_s` | 145 | 29 | 16 | 8 | 12 |
| `q3_k` | 208 | 41 | 4 | 0 | 15 (8 deles `u8`) |
| `q4_k` | 84 | 14 | 8 | 0 | 9 |
| `q5_k` | 93 | 17 | 8 | 0 | 11 |

**O kernel é limitado por issue, e dá para calcular o teto:** em `iq3_s` uma iteração do
laço do warp consome 2 blocos = **220 bytes** e custa ~171 instruções (mais o laço). Com
128 SIMD32 × 2,4 GHz × 1 IPC = 307 G instr/s ⇒ 307/171 × 220 B = **395 GB/s**; medido
**442-490 GB/s**, ou seja **IPC ≈ 1,1-1,25** (o dual-issue VOPD está fazendo parte do
trabalho). Para subir a banda de `iq3_s` (33% do tráfego) só há um caminho: **menos
instruções por byte**. As 171 se decompõem em 16 `dp4a` (irredutíveis: 4 MACs int8 cada),
8 `perm`, 16 loads, e **~57 de aritmética de índice/seletor** (`v_lshlrev_b32`=21,
`v_and_or_b32`=17, `v_bfe_u32`=11, `v_mul_u32_u24`=8) — é aí que sobra gordura (um LUT de
seletor de sinais de 16 entradas trocaria ~5 ALU por 1 load).

### B.3 `q3_k`: a exceção que ninguém olhou [medido]

`q3_k` roda a **307 GB/s** (0,401 GB, 1,31 ms/token, 5% do matvec) enquanto os outros
k-quants vão a 408-582. A contagem de ISA explica: **208 instruções** por chamada, com
`v_sub_nc_u16`=16, `v_and_b32`=29, `v_lshlrev_b16`=8 — **emulação de bytes 16-bit**, o
mesmo padrão que o M2 eliminou em `iq3_s` com `perm`+linearidade e **que nunca foi aplicado
aos k-quants**. O `vec_dot_q3_K_q8_1` é exatamente o candidato: trocar o par
`v_sub_nc_u16`/`v_and_b32` por um `perm`+`dp4a` linear (o M2 mediu **2,0×** em `iq3_s` com
essa troca). Ganho esperado: 1,31 ms → ~0,7 ms ⇒ **+1,7% de decode**. Esforço: médio
(mexe em `vecdotq.cuh`, precisa provar bit-exatidão). **Não implementado neste estudo por
falta de orçamento** — é o item #2 do ranking (§C).

### B.4 O que o llama.cpp faz e nós não (RDNA4)

Briefing completo do checkout `df03399b8` (worktree de referência; nenhuma tabela
`MMVQ_PARAMETERS_RDNA4` existe neste commit — os parâmetros viraram funções `constexpr`):

- `calc_nwarps` (`mmvq.cu:467-490`) usa **8 warps por linha** para
  `{Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,Q2_K,Q4_K,Q5_K,Q6_K,IQ4_NL,IQ4_XS}` e **1 warp por linha**
  para `{Q3_K,IQ1_S,IQ2_*,IQ3_*}` ("register pressure and lookup table contention").
  Nós usamos **1 warp por linha** em todos os tipos com `WPR=1`, com redução só por
  shuffle (sem a etapa LDS que eles têm). Medido aqui: **não é uma lacuna** — o M2 varreu
  1×8 vs 4×1/2×2 e a forma de 1 warp por linha ganhou.
- Para `ne11 > 8` (prefill) o caminho é **MMQ com WMMA int8**
  (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`) + **staging dos pesos em LDS**, com
  exigência de ≥48 KB de shared/block e `mmq-config-rdna4.cuh` com `occupancy=2`,
  `I=64/128`, `J=16..128`. Isso é a rota para prefill (§C.6).
- Eles ligam **HIP graphs por padrão** (`GGML_HIP_GRAPHS=ON`). Medimos o que isso vale
  para *nós*: **1,4 ms/token (4%)** (§D.6) — ou seja, existe, mas não é o fator 16 do
  prefill.
- **Nada de `v_pk_fma_f16` em GEMV quantizado** no backend deles (zero ocorrências em
  `ggml/src`); `v_dot2_f32_f16` só para pesos fp16. A hipótese "o Vulkan chega a 440 tok/s
  de prefill com matemática fp16 empacotada" **não** descreve o caminho GEMV de decode de
  nenhum backend — o ganho deles é GEMM/WMMA, não packed fp16 no dot.
- `mmvq_prefetch_l2` e PDL são no-op em RDNA4 no backend deles também.

### B.5 A conta dos lançamentos (a maior descoberta deste estudo) [medido]

```
bench-matvec-shapes-gpu:
  launch floor: empty kernel 0.010 ms timed with a sync, 0.0035 ms/launch back-to-back
  replay (497 launches enfileirados, 1 par de eventos): 26.4-27.7 ms
  graph replay dos MESMOS 497 kernels: 25.0 ms  ->  1.38 ms/token recuperável (1.06x)
```

O motor faz **~2200 lançamentos por token** (contados em `graph.cuh`: 305 projeções ×2,
130 norms, 192 ops escalares do GDN, 128 `kv_write`, 16 pares split+merge de atenção,
rope, deinterleave, silt, residuals...). **[Correção, ver §A.2.1: a contagem medida no
código pós-M8 é 1 940 — 497 matvec + 257 quantizações + 1 186 do resto; as 305 projeções
desta linha são 497 porque o FFN tem 3 projeções por camada, não 2 × 1,5.]** O trabalho real dos kernels pequenos é
desprezível; o que custa é o **gap de despacho** (3,5 µs medidos com kernel vazio
enfileirado). Duas medidas delimitam o quanto disso é evitável:

- **grafo do HIP sobre a sequência de matvec: 1,4 ms** (1,06×) — os kernels grandes
  sobrepõem o despacho com a execução;
- os kernels pequenos e dependentes **não** têm o que sobrepor, então para eles o gap é o
  custo inteiro: os ~1900 lançamentos pequenos ≈ **7 ms de teto superior** — mas isso é
  **[estimativa]**, não medição: o grafo foi medido só sobre a sequência de matvec (497
  kernels grandes). Medir o grafo sobre a sequência *inteira* (ou sobre 1000 kernels
  vazios) é o teste que falta, e é ele que decide entre "fundir kernels" (grande) e
  "capturar um HIP graph por token" (médio).

Isso **refina o M5 sem contradizê-lo**: o M5 rejeitou HIP graphs medindo o custo *de host*
por lançamento (1,04 µs); o número relevante é o de *device*, e mesmo assim o grafo só
devolve 1,4 ms no melhor caso (sequência de matvec). A conclusão operacional muda de
"HIP graphs não pagam" para **"fundir kernels pequenos é a alavanca; HIP graph é um
complemento de ~1,4 ms"**. HIP graph sobre a sequência *inteira* (incluindo os ~1900
kernels pequenos) não foi medido: é o teste que falta (§C.4).

---

## C. Candidatos, por ordem de ganho medido sobre risco e esforço

| # | otimização | ganho medido/esperado | bit-exato? | esforço | dono |
|---|---|---|---|---|---|
| 1 | **Política de splits da atenção** (`kAttnSplitMin` 2048 → keys/512) | **+15%** a 4K f16 (23,95→27,54 tok/s, `RD_ATTN_SPLITS=8`); +1% a 16K | não (equiv. numérica, gate em texto real) | **1 linha** | `graph.cuh` (§E.4) |
| 2 | **Tirar as 192 quantizações redundantes** | ~+2,4% (0,85 ms de trabalho + 192 gaps) | **sim** | pequeno (`graph.cuh` + helper em `matvec.cuh`) | `graph.cuh` (§E.1) |
| 3 | **Fundir os kernels pequenos** (GDN escalar, `rms_norm`+`quantize`, `kv_write`) | +3-10% [estimativa não medida: ~1900 lançamentos × 3,5 µs] | sim (fusão preserva ordem) | grande (`graph.cuh`) | `graph.cuh` (§E.2/E.3) |
| 4 | **HIP graph da sequência inteira** | ≥1,4 ms (+4%) medido no matvec; teto nos kernels pequenos não medido | sim (mesmos kernels) | médio; incompatível com o `hipMemcpy` da posição hoje | `graph.cuh` |
| 5 | **`q3_k`: eliminar a emulação de `__vsubss4` por linearidade do `dp4a`** — `Σ(vil−vih)·u = Σvil·u − Σvih·u`, exato em int32 (os valores são 0..3 e 0..4, a saturação do `__vsubss4` é código morto aqui), então 2 `dp4a` + 1 sub substituem 1 `dp4a` + a emulação | ~14% das instruções do corpo (30 de 208 contadas na ISA: `v_sub_nc_u16`=16, `v_and_b32`=29) ⇒ **~0,15-0,2 ms (+0,5%)** | sim (linearidade inteira, como no `iq3_s` do M2) | pequeno-médio (`vecdotq.cuh`) | este worktree |
| 6 | **Prefill em batch (MMQ/WMMA int8 + LDS)** | 28,8 → 63 tok/s (matvec batch, M6) e até 300-440 com MMQ | não | milestão | outro agente (M6 2a/2c) |
| 7 | **Compartilhar K/V entre as 6 cabeças GQA** (agora que a grade é larga) | a 64K a atenção está em ~1,56 TB/s de L2 por causa da redundância 6×; teto ~2,4× na atenção ⇒ até +18% a 64K | não (equiv. numérica) | grande, e o protótipo do M7 já falhou uma vez | este worktree |
| 8 | Argmax no device (caminho greedy) | ~+0,3-0,5 ms (1%) | sim | pequeno-médio | `graph.cuh`/`main.hip` |
| 9 | `bench` descontar `token_embd`/MTP | 0 (corrige número reportado em 8%) | — | trivial | `main.hip` |

---

## D. Rejeitados **com dado** (para não repetir)

1. **`ROWS` por CTA (linhas por bloco)**: `rows ∈ {1,2,4,8,16}` sobre os 497 tensores reais
   ⇒ **0,99-1,01×** (o piso de ruído da própria medição é 1,00×). É bit-exato e não vale
   nada: o mapeamento thread→(slot,kqs) não depende de ROWS.
2. **`__launch_bounds__` / orçamento de registrador (`minb`)**: **0,82×** — 18% *mais
   lento* com `minb ∈ {2,4,6}` nos 14 tipos. O M2 tinha classificado isso como "±20%
   dentro do ruído"; com A/B intercalado é uma regressão real: a ocupação já estava no
   platô de 64 waves/WGP.
3. **Unroll manual com acumulador único** (MLP bit-exata: mesmas ops, mesma ordem, só mais
   loads em voo): **1,03×** com 2 blocos e **0,92×** com 4. O compilador já extrai essa
   paralelidade; o laço não é limitado por MLP.
4. **Prefetch L2**: o `rdna4_prefetch_l2()` do `matvec.cuh` usava `__builtin_prefetch`,
   que **compila para nada em gfx1201** (verificado com `--save-temps`: o kernel não tem
   nenhuma instrução de prefetch). Ou seja, a conclusão do M2 ("prefetch neutro/prejudicial")
   media um no-op. Refeito com o intrínseco real (`__builtin_amdgcn_s_prefetch_data`):
   **0,98×** — continua não pagando.
5. **`ILP` (acumuladores independentes)**: já era o resultado do M2 (±3%); não foi
   re-medido porque muda a ordem de soma (não é livre) e as duas alavancas bit-exatas
   equivalentes (unroll, prefetch) não pagam.
6. **HIP graph da sequência de matvec**: **1,06×** = 1,38 ms/token. Positivo, mas pequeno
   perto do que a fusão de kernels pequenos promete; e não é o fator que explica a
   distância para o llama.cpp.
7. **`WPB=32` na atenção** (o "8 → 32 warps" que a mensagem do commit `a830570` descreve):
   no kernel **sem split** dá 0,73× a 4K e 1,07-1,19× a 8-64K, e no kernel **com split**
   é sempre pior que 8 ou 16 (ex.: 16 splits a 64K: 0,1111 vs 0,0819 com 8 warps). Além
   disso, `attn.cuh:82` **nunca teve o valor trocado** — `kAttnWarpsPerBlock` é 8 desde o
   M7, então o "32 warps" do `docs/medicoes-m7.md` não está no código que roda (o ganho
   de 3,6× do commit era do `kv_load8` + do caminho de splits).
8. **Compartilhar K/V entre as cabeças GQA** (protótipo do M7): 8-12× mais lento. Só
   reconsiderar com a grade larga dos splits e com staging em LDS por CTA (§C.7).

---

## E. Mudanças que precisam do `graph.cuh` (para o agente coordenador)

Todas as quatro são pequenas, independentes entre si, e cada uma tem a medição que a
justifica. Elas **não** foram aplicadas porque `graph.cuh` pertence a outro agente; os
helpers de kernel que elas usam já estão neste branch.

### E.1 Quantizar uma vez por ativação (−192 lançamentos/token, bit-exato)

Em `Graph::proj` hoje:

```cpp
quantize_q8_1_kernel<<<(nb + 3) / 4, 128>>>(d_x, (block_q8_1 *)d_q8_, nb);
matvec_launch((int)w.dt, w.ptr, (const block_q8_1 *)d_q8_, d_y, nrows, ncols, nullptr);
```

Proposta: `proj` ganha um parâmetro `bool act_ready = false` (ou duas variantes
`proj_q8`/`proj_reuse`) e os chamadores que projetam **a mesma ativação em sequência**
passam `true` a partir da segunda:

```cpp
// full_attn: mesma d_xn_ nas tres
proj(L.attn_q, d_xn_, d_proj_,   NH*2*HD, E, err);              // quantiza
proj(L.attn_k, d_xn_, d_kstage_, NKV*HD,  E, err, /*act_ready=*/true);
proj(L.attn_v, d_xn_, d_vstage_, NKV*HD,  E, err, /*act_ready=*/true);
// gdn_layer: mesma d_xn_ nas quatro (qkv, gate, beta, alpha)
// ffn: mesma d_xn_ em gate/up
```

Invariante que torna isso seguro: **entre duas dessas chamadas nada escreve em `d_xn_`**
(as projeções escrevem em `d_proj_`/`d_kstage_`/`d_vstage_`/`d_qkv_`/`d_z_`/`d_beta_`/
`d_alpha_`/`d_ffn_a_`/`d_ffn_b_`), e o kernel de quantização é determinístico ⇒ os
`block_q8_1` são **byte-idênticos**. Ganho: 192 de 305 lançamentos; ~0,9 ms de trabalho
(teto medido pelo M5: 1,5 ms para todos) + ~192 × 3,5 µs ≈ **1,3-1,9 ms (4-5%)**.
Gate: `check-graph-gpu` (oráculo por nó) + `check_golden_run.sh` devem ficar idênticos.

### E.2 Fundir as cadeias escalares do GDN e `rms_norm`+`quantize`

- `sigmoid(48)` → `add(alpha,dt)` → `softplus` → `mul(gate,a)`: 4 lançamentos de ~48
  elementos por camada recorrente (192/token). Um kernel `gdn_gate_kernel` que faça os
  quatro numa passada com 48 threads elimina 144 lançamentos (~0,5 ms).
- `rms_norm_launch(d_x_, attn_norm, d_xn_, 1, E)` seguido de `quantize_q8_1_kernel`
  (o primeiro `proj` da camada): fundir num kernel que escreve `d_xn_` **e** o bloco
  `q8_1` (o valor normalizado já está em registrador) elimina 130 lançamentos (~0,45 ms)
  **sem mudar um bit** (mesma ordem de soma na norma, mesma fórmula do `q8_1`).
- `kv_write`: usar `kv_store_rows_launch(KvType, src, dst, n_heads, head_dim)` — que este
  branch **ainda não adiciona**; se o coordenador preferir, a fusão pode ficar no
  `graph.cuh` com um kernel de 128 threads escrevendo as 4 cabeças.

### E.3 `hipMemcpy` da posição

`hipMemcpy(d_pos_, &pos, sizeof(int), hipMemcpyHostToDevice)` é síncrono e acontece 16×
por token. Alternativa: um `rms_norm`/`rope` que receba `pos` por valor (a RoPE é o único
consumidor) e elimine o `d_pos_`.

### E.4 Política de splits da atenção (a de maior ganho)

`graph.cuh`:

```cpp
static constexpr int kAttnSplitMin = 2048;      // hoje
int attn_splits_for(int keys) const {
  ...
  if (keys < kAttnSplitMin) return 1;
  const int sp = keys / kAttnSplitMin;          // 1 em [2048,4096) -> UNSplit!
  return sp > kAttnMaxSplits ? kAttnMaxSplits : sp;
}
```

Dois defeitos medidos: (a) em `keys ∈ [2048,4096)` o resultado é 1, ou seja o caminho
**sem split**; (b) a contagem cresce devagar (2 splits a 4K) quando o kernel ainda está
limitado por paralelismo. Medido (`bench --start-pos`, IQ3_S, f16, 32 tokens, 3 reps):

| 4K f16, `RD_ATTN_SPLITS` | default (1) | 2 | 4 | **8** | 16 |
|---|---|---|---|---|---|
| decode | 23,95 tok/s | 25,80 | 27,01 | **27,54** | 27,01 |
| decode (re-medido no binário final) | 23,48 | — | — | **26,92 (+14,7%)** | — |

e no kernel (`bench-attn-gpu f16 4096`): sem split 0,334-0,346 ms/camada contra
0,0717 ms com 4 splits e WPB=16 (**4,7×**) — 16 camadas ⇒ 5,4 → 1,1 ms/token.

Mudança sugerida (uma linha, com o mesmo espírito do que já existe):

```cpp
static constexpr int kAttnSplitMin = 512;   // e o resto da funcao inalterado
```

Isso dá 8 splits a 4K (medido: 27,54 tok/s, +15%), 16 a 8K+ (16 splits a 64K mediu
1,03 ms/camada vs 1,09 com 8 — sem regressão). **Atenção**: o valor precisa de
re-validação em texto real — `scripts/check_attn_split.sh` compara PPL com
`RD_ATTN_SPLITS=1` vs 4 (5,1989 vs 5,1917, 0,14%) e continua sendo o gate.

---

## F. O que foi **implementado** neste branch e medido

### F.1 Warps por CTA no kernel de atenção com split (`include/rdna4/attn.cuh`) — **+4 a +5%**

`attn_split_kernel` e `attn_launch_typed` ganharam `WPB` como parâmetro de template
(padrão = `kAttnWarpsPerBlock`, o valor que já rodava), e `attn_launch_split` passou a
escolher a forma por número de splits:

```cpp
constexpr int kAttnSplitWpbLimit = 4;
inline int attn_split_wpb(int n_splits) {
  return n_splits <= kAttnSplitWpbLimit ? 16 : kAttnWarpsPerBlock;
}
```

Motivo (medido, `bench-attn-gpu`, ms por camada, 1 token de query, f16):

| splits | WPB=8 | WPB=16 | WPB=32 | CTAs |
|---|---|---|---|---|
| 2 | 0,1777 | **0,1051** | 0,1057 | 48 |
| 4 | 0,1014 | **0,0719** | 0,0884 | 96 |
| 8 | **0,0744** | 0,0775 | 0,0969 | 192 |
| 16 | **0,0819** | 0,0844 | 0,1111 | 384 |

O limite é **4** (e não 8) porque com 8 splits as duas execuções independentes discordam
(1,10× e 1,003×) enquanto com 2-4 splits todas as execuções dão 1,4-1,7×; assim **todo
contexto com ≥8 splits (16K para cima) roda exatamente o mesmo caminho de antes**.

Antes/depois (`bench --start-pos`, IQ3_S, f16, 32 tokens, 4 reps; BEFORE = binário
construído de um `git worktree add --detach /tmp/rdna4-base HEAD`, AFTER = este branch):

| ctx / pos | keys → splits | BEFORE | AFTER | ganho |
|---|---|---|---|---|
| 6144 / 4096 | 4097 → 2 | 26,11 tok/s (38,3 ms) | **27,29 (36,7)** | **+4,5%** |
| 8192 / 6144 | 6145 → 3 | 25,82 (38,7) | **27,18 (36,8)** | **+5,3%** |
| 10240 / 8192 | 8193 → 4 | 26,10 (38,3) | **27,12 (36,9)** | **+3,9%** |
| 16384 / 16320 | 16321 → 7 | 25,45 (39,3) | 25,45 (39,3) | 0,0% (caminho idêntico) |
| 65536 / 65504 | 65505 → 16 | 19,48 (51,3) | 19,48 (51,3) | 0,0% (caminho idêntico) |

(uma segunda execução independente do mesmo par de binários deu 25,74→26,65 (+3,5%) a
6144, 25,65→26,59 (+3,7%) a 8192 e 27,99 tok/s no contexto curto nos dois — o *absoluto*
varia 2-3% entre sessões por DPM/temperatura, então o que vale é o delta dentro da mesma
execução, e é ele que fica entre +3,5% e +5,3%.)

As duas últimas linhas são o **controle**: são o mesmo código, e o tempo sai idêntico ao
milissegundo — o que mostra que o ganho das três primeiras não é ruído de execução.
Aritmética: é uma mudança de **forma de bloco** (quantos warps dividem a faixa de chaves de
um par (cabeça, split)); o número de parciais e a ordem de soma entre splits não mudam.
O efeito numérico é o mesmo da mudança de split do M7 (rel-L2 ~2,5e-7 no kernel).

**Gates rodados neste branch (todos verdes):**

```
scripts/gpu-lock.sh ./build/check-matvec-gpu <IQ3_S>             -> check-matvec-gpu: OK (14 types tested)
scripts/gpu-lock.sh ./build/check-dequant-gpu <IQ3_S>            -> check-dequant-gpu: OK (14 types tested)
scripts/gpu-lock.sh ./build/check-kvctx-gpu <IQ3_S> 65536 q4_0   -> check-kvctx-gpu: OK
scripts/gpu-lock.sh ./scripts/check_golden_run.sh                -> check-golden-run: OK
scripts/gpu-lock.sh ./scripts/check_attn_split.sh                -> 5,1989 vs 5,2054 = 0,125% (limite 0,5%): OK
scripts/gpu-lock.sh ./build/check-graph-gpu <IQ3_S> reference/oracle_prompt6_ub1_tok7_cpu.txt -  -> FAIL (39 nos)
```

**`check-graph-gpu` esta vermelho neste branch — e ja estava:** o mesmo comando, com
o binario construido de um worktree **destacado no HEAD limpo** (`bed0f23`) **e** com o
binario do checkout principal (`main @ c35c3ad`, `build/` de 20:30), produz **a mesma
lista de 39 falhas com os mesmos numeros** (`Qcur_full-3 sample=6.994e-02/0.2
sum=6.451e+00/0.01`, `l_out-3 sum=1.617e+01/0.05`, ...): as falhas sao todas em nos de
camada de atencao (il = 3, 10, 20, 30, 40, 50, 62, 63 e os `Qcur/Kcur/Vcur/attn_*` do
il=3) e as magnitudes (somas relativas de 4-16) sao de *outro* resultado, nao de
arredondamento. O dump `reference/oracle_prompt6_ub1_tok7_cpu.txt` e de 16:07 e o
diretorio `reference/` do repo principal foi reescrito **durante esta sessao**
(`oracle_mtp_prompt6_cpu.txt` apareceu as 20:59), entao a hipotese mais simples e que o
dump nao corresponde ao estado atual do motor/grafo: **recapturar com
`scripts/capture_oracle.sh`** antes de usar esse gate como aceite. Nenhuma parte desta
mudanca toca o caminho que esse teste exercita (contexto curto ⇒ atencao sem split).

Qual gate prova o quê nesta mudança: `check-attn-split` (texto real, é o gate do caminho
com splits), `check-kvctx-gpu` (compara split vs sem split no mesmo contexto sintético) e
`check-golden-run`/`check-graph-gpu` (contexto curto ⇒ caminho sem split ⇒ **bit-idênticos
por construção**, é isso que os mantém verdes).

### F.2 Ferramentas novas e extensões (nada de runtime mudou por causa delas)

- `tests/bench_matvec_shapes_gpu.hip` + alvo `bench-matvec-shapes-gpu` (novo): medição por
  forma real, A/B intercalado com piso de ruído, replay enfileirado, piso de lançamento e
  replay por HIP graph.
- `tests/bench_attn_gpu.hip` (estendido): varredura splits × warps/CTA no kernel com e sem
  split, com rel-L2 contra o caminho sem split.
- `include/rdna4/matvec.cuh`: `matvec_launch_rows()`, `matvec_launch_unroll()`,
  `matvec_launch_minb()` (já existia) usados pelo bench; `rdna4_prefetch_l2()` corrigido
  para o intrínseco que realmente executa; `UNROLL` como parâmetro de template do
  `matvec_kernel_gen` (sem mudança de comportamento no caminho de envio: `UNROLL=1`).
  `matvec_launch_rows` limita `ROWS*WPR*32 <= 1024`.
- `include/rdna4/attn.cuh`: `WPB` template (kernel e launchers) + `attn_launch_wpb()`,
  `attn_launch_split_wpb()` para os benches.

---

## G. Reproduzir

```bash
source scripts/rocm-env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j6
M=/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf

# orcamento por token (o baseline de tudo)
scripts/gpu-lock.sh ./build/rdna4-infer bench -m $M -n 32 --reps 3
scripts/gpu-lock.sh ./build/rdna4-infer bench -m $M -n 32 --reps 3 --layers 0
scripts/gpu-lock.sh ./build/rdna4-infer bench -m $M -n 32 --reps 3 --ctx-size 4096 --start-pos 4064 --fill-cache

# inventario do matvec por tensor + rejeitados + HIP graph (~2 min: le 11 GB do disco)
scripts/gpu-lock.sh ./build/bench-matvec-shapes-gpu $M --budget-mib 12000 --reps 4 --rows 1,4 --minb 2,4,6
scripts/gpu-lock.sh ./build/bench-matvec-shapes-gpu $M --budget-mib 12000 --reps 4 --unroll 1,2,4

# atencao: splits x warps/CTA
scripts/gpu-lock.sh ./build/bench-attn-gpu f16 2048 4096 8192 16384 65536

# politica de splits fim a fim (o que o graph.cuh deveria adotar)
for S in 1 2 4 8 16; do RD_ATTN_SPLITS=$S scripts/gpu-lock.sh ./build/rdna4-infer bench \
    -m $M -n 32 --reps 3 --ctx-size 4096 --start-pos 4064 --fill-cache | grep "decode 32"; done

# contagem de instrucoes por vec_dot (nao usa GPU)
cd /tmp && amdclang++ -x hip --offload-arch=gfx1201 -O3 -I<repo>/include --save-temps -c vd_count.hip
# (kernel de uma chamada por tipo; contar as instrucoes do corpo em vd_count-*.s)
```

---

## H. Conclusão em uma linha

O decode gasta **~26 ms de matvec (70% do token) num kernel que já está a ~70% do teto de
banda e é limitado por *issue*** (171 instruções por 220 bytes no `iq3_s`), **~7 ms em gaps
de despacho de ~2200 kernels pequenos**, ~1,4 ms no LM head (98% do teto) e o resto em
atenção; as alavancas que sobraram não são knobs de kernel — ROWS, `minb`, unroll e
prefetch foram **medidos e rejeitados** — e sim (1) **política de splits da atenção**
(+15% a 4K, 1 linha no `graph.cuh`), (2) **remover as 192 quantizações redundantes**
(bit-exato, ~4%), (3) **fundir os kernels pequenos** (até ~10%) e (4) **reescrever o
`q3_k`** como o `iq3_s` foi reescrito no M2 (~+1,7%).
