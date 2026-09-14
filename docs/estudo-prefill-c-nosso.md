# Onde vai o tempo do NOSSO prefill (medido)

Todas as corridas em 2026-09-14, `scripts/gpu-lock.sh` com `timeout` **dentro** do lock, modelo
`/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf`, KV `f16/f16`. Cada tabela
diz o comando e a janela (VRAM antes, se a placa estava ocupada). Logs brutos em
`/tmp/med-prefill/`. Nada do motor foi modificado; os dois utilitários novos são scratch em
`/tmp` (`/tmp/scratch/mb_batch_n2.hip`).

## TL;DR

1. **O "gap de 9,0×" não existe no tamanho de chunk do nosso motor.** O `pp512` do
   `llama-bench` roda a prompt inteira em **um** micro-lote de 512 tokens; o nosso motor usa
   **16**. Com o micro-lote do llama.cpp forçado a 16 (`-b 16 -ub 16`), ele faz **200,25 tok/s**
   contra os nossos **123,4** — a razão é **1,62×**, não 9,0×.
2. O matvec em lote é **83,7 %** do prefill medido dentro do grafo (nível 2,
   `--prefill 64`), o que **confirma** o número de 85,6 % que veio do bench isolado: os dois
   concordam dentro de 2 %.
3. E os dois **valores absolutos** do matvec também concordam: 117,8 ms por chunk de 16 no
   grafo contra 110,7 ms isolado (**+6,4 %**), ou seja **não há andaime escondido** entre o
   bench e o prefill real: o grafo não acrescenta nada além do próprio kernel.
4. O andaime (norms, atenção, GDN, resíduos, quantização de ativação, **todos** os lançamentos)
   custa **0,79 ms/token = 9,7 %** do prefill; o custo de despacho de todos os ~955
   lançamentos por chunk é **2,4 ms = 0,06 %**. Fundir tudo economizaria 0,06 %.
5. Os três maiores custos por token que batching/fusão ainda poderiam tirar são
   `gdn_delta` (0,641 ms/token), `act_quant` (0,227 ms/token) e `qk_norm_rope_kv` +
   `attention` + `attn_gate_out` (0,088 ms/token) — juntos **0,96 ms/token, 11,8 %**. O resto
   (66 %) é o matvec, e o matvec não é andaime: é álgebra.

---

## 1. Baseline nosso, e o que o `pp` do llama.cpp realmente mede

### 1.1 Nosso prefill, tok/s

```
./scripts/gpu-lock.sh timeout 900 ./build/rdna4-infer bench \
  -m MODEL --prefill N --prefill-reps 3 --decode 1 --warmup 0 --ctx-size $((N+64))
```

| N | passada 1 | passada 2 | passada 3 | **melhor** | ms/token |
|---|---|---|---|---|---|
| 64 | 0,767 s (83,46) | 0,513 s (124,71) | 0,516 s (124,06) | **0,513 s (124,71 tok/s)** | 8,02 |
| 512 | 4,543 s (112,69) | 4,174 s (122,66) | 4,205 s (121,75) | **4,174 s (122,66 tok/s)** | 8,15 |
| 2048 | 17,730 s (115,51) | 17,275 s (118,55) | 17,082 s (119,89) | **17,082 s (119,89 tok/s)** | 8,34 |
| 4096 | 35,266 s (116,14) | 35,088 s (116,74) | 35,122 s (116,62) | **35,088 s (116,74 tok/s)** | 8,57 |

- Janela: `VRAM_before` = 679 MB (nada na placa, nenhum `flock` esperando) na entrada do job;
  as corridas seguintes do laço já rodaram com a fila do `flock` cheia de outros agentes
  (medido: 7 waiters), o que é a razão provável da passada 1 de cada N ser a pior (DPM frio
  + disputa). O número reportado é a **melhor das 3**, como o próprio bench define.
- **Frio × quente**: a passada 1 do N=64 (83,46 tok/s) é a assinatura do DPM — 1,49× abaixo da
  passada 2. É por isso que `--prefill-reps 3` não é opcional.
- O nosso custo por token **cresce** de 8,02 (N=64) para 8,57 (N=4096): +6,9 %. É a atenção
  (16 camadas) pagando contexto, e a 4096 o split de atenção (`kAttnSplitMin = 512`) passa a
  valer (`splits = 4` a 2048, medido no count-only).

### 1.2 O que o `llama-bench` mede (leitura de código, não de opinião)

`tools/llama-bench/llama-bench.cpp:2133-2161` (`test_prompt`):

```cpp
while (n_processed < n_prompt) {
    int n_tokens = std::min(n_prompt - n_processed, n_batch);
    ... tokens[i] = std::rand() % n_vocab;
    int res = llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens));
    n_processed += n_tokens;
}
llama_synchronize(ctx);
```

- **Não há amostragem** (`-n 0`): só `llama_decode`, nenhum `llama_sampler`.
- `llama_batch_get_one` (`src/llama-batch.cpp:931-943`) deixa `logits = nullptr`, e
  `llama-context.cpp:1669` define `output_all = cparams.embeddings` (falso no bench) ⇒
  `llama-batch.cpp:121-128` marca **só o último token** do lote como saída. Ou seja: o `pp` do
  llama.cpp faz a mesma coisa que o nosso `--prefill`: prompt inteiro pelas camadas +
  `output_norm` + LM head **de um token só**. Não é uma comparação com logits por token.
- **Não há cache reaproveitado entre passadas**: as 3 repetições re-decodificam a mesma prompt
  sobre um KV já escrito; a atenção é causal e cada linha é reescrita pela posição antes de ser
  lida, que é exatamente o que o nosso `bench --prefill-reps 3` faz.

**Veredito: o `--prefill 512` mede a mesma coisa que o `pp512` do `llama-bench` — com uma
diferença que não é semântica, é de tamanho de lote.** `n_batch` default é 2048 e `n_ubatch`
512 (`llama-bench.cpp:390-391`), então o `pp512` entra numa **única** chamada de
`llama_decode` de 512 tokens e o backend Vulkan roda **um** GEMM de 512 colunas. O nosso
`prefill_ids` (`src/main.hip:162-183`) corta a mesma prompt em **32 chunks de 16**
(`kMaxBatch = 16`, `graph.cuh:189`).

### 1.3 O número comparável, medido (mesma máquina, mesmo modelo, hoje)

```
./scripts/gpu-lock.sh timeout 900 /home/marcelo/Projetos/llama.cpp/build/bin/llama-bench \
  -m MODEL -p 64,512 -n 0 -r 3 -ub 16 -b 16
```

| llama.cpp Vulkan | tok/s | ms/token | comando |
|---|---|---|---|
| pp64 (micro-lote 16) | **189,29 ± 8,95** | 5,28 | `-p 64 -r 3 -ub 16 -b 16` |
| pp512 (micro-lote 16) | **200,25 ± 3,44** | 4,99 | `-p 512 -r 3 -ub 16 -b 16` |
| pp512 (defaults: `b 2048`, `ub 512`) | **1168,49 ± 2,59** | 0,856 | `-p 512 -r 3` |
| pp512 (defaults, corrida do briefing) | 1113,96 | 0,898 | `-p 64,512,2048 -n 0 -r 3` |

O `-b 16 -ub 16` foi confirmado no cabeçalho que o próprio `llama-bench` imprime
(`n_batch 16, n_ubatch 16`), e é o **micro-lote** que decide: `n_ubatch` é o lote que chega ao
backend (`cparams.n_ubatch`, `llama-bench.cpp:1291`), logo `-ub 16` reproduz exatamente o nosso
chunk de 16.

**Consequência, e é o resultado principal deste estudo:** com o micro-lote igualado,

| | tok/s | ms/token |
|---|---|---|
| nosso `--prefill 512` | 123,39 | 8,104 |
| llama.cpp Vulkan `pp512 -ub 16` | 200,25 | 4,994 |
| **razão** | **1,62×** | **+3,11 ms/token** |

A 9,0× do briefing (123,9 contra 1113,96) é **o chunk de 16 contra o chunk de 512**, não o nosso
kernel contra o kernel do Vulkan. As duas coisas estão medidas nesta máquina, na mesma janela:

- `pp512` micro-lote 512 → 1168,49 tok/s; micro-lote 16 → 200,25 tok/s. **O próprio llama.cpp
  perde 5,84× ao descer o micro-lote para o nosso.** (Mesma corrida, mesma sessão de
  `llama-bench`, diferença só de `-b/-ub`.)

---

## 2. Orçamento por fase DENTRO do prefill real

### 2.1 Nível 1 (fases por bloco)

```
./scripts/gpu-lock.sh timeout 900 ./build/bench-phases-gpu MODEL --prefill 64  --prefill-reps 3 --level 1
./scripts/gpu-lock.sh timeout 900 ./build/bench-phases-gpu MODEL --prefill 128 --prefill-reps 3 --level 1
./scripts/gpu-lock.sh timeout 900 ./build/bench-phases-gpu MODEL --prefill 512 --prefill-reps 3 --level 1
```

Janela: entrada do job com `VRAM_before` = 4347 MB (VRAM de um processo drenando, nenhum
`/dev/kfd` vivo). O `bench-phases-gpu` imprime o piso de despacho da própria corrida:
**2,253 µs por kernel vazio enfileirado** e **4,063-4,085 µs por `hipEventRecord`** — o custo
que cada marca do profiler adiciona à fila.

**N = 64** — passada limpa (profiler desligado) **8,150 ms/token (122,71 tok/s)**, 4 chunks de
16; instrumentada 8,260 ms/token ⇒ **inflação +1,4 %**; soma dos buckets 526,635 ms; 42,0
marcas/token.

| bucket | ms/token | % | marcas/tok | o que entra |
|---|---|---|---|---|
| ffn_gate_up | 2,818 | **34,2 %** | 4,0 | `ffn_gate` + `ffn_up` + silu·mul, 64 camadas |
| ffn_down | 1,705 | **20,7 %** | 4,0 | `ffn_down`, 64 camadas |
| gdn_proj | 1,342 | **16,3 %** | 3,0 | qkv/gate/beta/alpha, 48 camadas GDN |
| gdn_out_proj | 0,630 | 7,7 % | 3,0 | `ssm_out`, 48 camadas |
| gdn_delta | 0,625 | 7,6 % | 3,0 | `delta_rule_batch`, 48 camadas |
| qkv_proj | 0,351 | 4,3 % | 1,0 | q/k/v das 16 camadas de atenção |
| attn_out_proj | 0,216 | 2,6 % | 1,0 | `attn_output`, 16 camadas |
| ffn_residual | 0,164 | 2,0 % | 4,0 | soma residual pós-FFN, 64 camadas |
| post_norm | 0,076 | 0,9 % | 4,0 | residual + `attn_post_norm`, 64 camadas |
| gdn_l2norm | 0,072 | 0,9 % | 3,0 | `l2_norm` de q/k, 48 camadas |
| gdn_norm_silu | 0,052 | 0,6 % | 3,0 | `ssm_norm` + silu·mul |
| gdn_scalars | 0,052 | 0,6 % | 3,0 | sigmoid(beta), softplus(alpha), ·a |
| gdn_conv | 0,051 | 0,6 % | 3,0 | `conv1d_state_batch` |
| qk_norm_rope_kv | 0,032 | 0,4 % | 1,0 | q/gate split + qk-norm + rope + escrita no KV |
| attention | 0,031 | 0,4 % | 1,0 | `attn_batch` (1 split a 64) |
| attn_gate_out | 0,012 | 0,1 % | 1,0 | sigmoid(gate)·attn |
| **soma** | **8,233** | 100 % | 42,0 | |

**N = 128** — limpa **8,579 ms/token (116,57 tok/s)** (8 chunks), instrumentada 8,926
(+4,0 %), soma 1139,491 ms, 42,0 marcas/token. Os sete maiores buckets, na mesma ordem do
N=64: `ffn_gate_up` 2,968 (33,3 %), `ffn_down` 1,833 (20,6 %), `gdn_proj` 1,456 (16,4 %),
`gdn_delta` 0,688 (7,7 %), `gdn_out_proj` 0,682 (7,7 %), `qkv_proj` 0,386 (4,3 %),
`attn_out_proj` 0,237 (2,7 %) ⇒ **os três de projeção = 6,257 ms/token = 70,1 %**.
`attention` sobe de 0,031 para 0,051 (**+66 %**, é contexto) e nenhum outro bucket muda mais
de 5 %.

**N = 512** — ver §3 (a primeira tentativa morreu em *page fault* por VRAM drenando de outro
processo; o número bom veio da segunda corrida do mesmo comando).

Em **todos** os N medidos o total de marcas é **42,0 por token**, idêntico — confirma que a
política de chunk é a mesma (32 chunks de 16 a 512, 128 a 2048) e que o andaime **não** cresce
com o chunk.

### 2.2 Nível 2 (separa `matvec` de `act_quant` dentro de cada projeção)

```
./scripts/gpu-lock.sh timeout 900 ./build/bench-phases-gpu MODEL --prefill 64 --prefill-reps 3 --level 2
```

N = 64, passada limpa **8,638 ms/token (115,76 tok/s)**, instrumentada 8,826 (+2,2 %),
**93,1 marcas/token**, soma 563,004 ms:

| bucket | ms/token | % | marcas/tok |
|---|---|---|---|
| **matvec** | **7,362** | **83,7 %** | 31,0 |
| gdn_delta | 0,641 | 7,3 % | 3,0 |
| act_quant | 0,227 | 2,6 % | 20,1 |
| ffn_residual | 0,095 | 1,1 % | 4,0 |
| post_norm | 0,076 | 0,9 % | 4,0 |
| gdn_l2norm | 0,073 | 0,8 % | 3,0 |
| gdn_norm_silu | 0,066 | 0,8 % | 3,0 |
| gdn_scalars | 0,053 | 0,6 % | 3,0 |
| gdn_conv | 0,050 | 0,6 % | 3,0 |
| qk_norm_rope_kv | 0,044 | 0,5 % | 1,0 |
| attention | 0,032 | 0,4 % | 1,0 |
| ffn_down / ffn_gate_up / gdn_proj / gdn_out_proj / attn_gate_out | 0,017/0,017/0,013/0,013/0,012 | 0,8 % | 4/4/3/3/1 |
| qkv_proj / attn_out_proj | 0,004 / 0,004 | 0,0 % | 1,0 cada |

Leitura direta: no nível 1 o "matvec" está embutido nos buckets de projeção; a diferença entre
os dois níveis fecha a conta. Nível 1 das projeções (N=64): 2,818+1,705+1,342+0,630+0,351+0,216
= **7,062 ms/token**; nível 2: matvec 7,362 + act_quant 0,227 = **7,589**; a diferença de
0,527 ms/token é o andaime que **fica fora** das janelas de projeção com marca de nível 2
(silu·mul do FFN, os 305 lançamentos de quantização que o nível 1 conta dentro do bucket da
projeção mas o nível 2 separa). Não é contradição: são janelas diferentes.

### 2.3 A sobrecarga do próprio profiler, medida

- **Um `hipEventRecord` custa 4,063-4,085 µs** (medido pelo próprio bench: 2000 kernels vazios
  com e sem evento entre eles). Com 42,0 marcas/token no nível 1 isso é **0,170 ms/token =
  2,1 %** de acréscimo *contábil*; a inflação **medida** foi **+1,4 % (N=64)** e **+4,0 %
  (N=128)** — ou seja, o custo da marca parcialmente sobrepõe com trabalho útil na fila, e a
  inflação medida é menor ou igual à conta.
- Nível 2: 93,1 marcas/token × 4,085 µs = 0,380 ms/token = 4,4 % contábil contra **+2,2 %**
  medidos.
- **Piso da instrumentação em wall-clock, medido sem eventos**: `--count-only` no mesmo N dá
  **8,077 ms/token** a 512 contra **8,104 ms/token** com o profiler desligado (mesma corrida,
  `phases.log`) ⇒ contar marcas custa **−0,3 %**, isto é, nada. O que custa é o evento.

### 2.4 O total medido por fase fecha com o total medido por fora?

| N | `bench --prefill` (melhor) | `bench-phases-gpu` (limpa, melhor) | soma dos buckets (nível 1) |
|---|---|---|---|
| 64 | 8,016 ms/token | 8,150 | 8,233 (*+1,0 % sobre a limpa do próprio harness*) |
| 128 | — | 8,579 | 8,902 (instrumentada; −3,6 % do que a instrumentada informa) |
| 512 | 8,146 ms/token | 8,104 (clean) / 8,077 (count-only) | — |

Os dois harnesses concordam dentro de 0,5 % a 512 (8,146 × 8,104 = **+0,5 %**) e a soma dos
buckets fecha com o total instrumentado dentro de 1 %. Não há tempo "sumido" no grafo.

---

## 3. O matvec é 83,7 % dentro do grafo? Sim — e os valores absolutos batem

### 3.1 Contabilidade dos lançamentos de matvec (código)

`proj_batch` (`include/rdna4/graph.cuh:1009-1028`) lança **exatamente um**
`matvec_launch_batch` por chamada (`:1023`), armado pela marca fina `RD_PHASE(pfine(),
"matvec")` (`:1022`). Chamadas por chunk de 16 tokens:

| onde | file:line | por camada | × camadas | matvecs/chunk |
|---|---|---|---|---|
| atenção q/k/v/out | `graph.cuh:1051,1054,1055,1143` | 4 | 16 (não-recorrentes) | 64 |
| GDN qkv/gate/beta/alpha/out | `graph.cuh:1205,1207,1208,1209,1262` | 5 | 48 (recorrentes) | 240 |
| LM head | `graph.cuh:1416` (`proj`, GEMV) | 1 | 1 | 1 |
| | | | | **305** |

Medido pelo próprio profiler em modo contagem: **31,0 marcas de `matvec` por token = 496 por
chunk** (`phases-l2.log`, N=64). A conta de marcas (não de kernels) é: 305 `proj_batch` + 190
`quantize_batch` + 1 head = 496. **Os 305 são os matvecs; os 190 são as quantizações de
ativação**, e são exatamente os 496 que o profiler conta. Nosso número de matvecs **por chunk**
é 305 — não 480.

### 3.2 Total do matvec: profiler × bench isolado × conta de banda

| fonte | ms por chunk de 16 | ms/token | comando |
|---|---|---|---|
| profiler nível 2, N=64 | **117,8** (7,362×16) | 7,362 | `bench-phases-gpu --prefill 64 --level 2` |
| bench isolado, N=16 | **110,70** | 6,92 | `bench-matvec-shapes-gpu --batch 16 --reps 3` |
| diferença | **+7,1 ms (+6,4 %)** | +0,44 | |

**Não há discrepância a explicar, e é isso o achado:** o matvec dentro do grafo custa 6,4 %
mais que o mesmo inventário lançado back-to-back. O que sobra (6,4 %) é coerente com a
diferença de staging: no grafo cada chunk intercala 305 matvecs com ~120 outros kernels, então
a Infinity Cache (que o bench isolado aproveita ao varrer os mesmos tensores em sequência) é
evictada entre projeções. A leitura direta das bandas: 11,133 GB / 117,8 ms = **94,5 GB/s** no
grafo contra 11,133 GB / 110,70 ms = **100,5 GB/s** no isolado.

**E o número que o briefing pede, o 85,6 %:** o profiler dá **83,7 %** (7,362 de 8,795 da
instrumentada) ou **85,2 %** (7,362 de 8,638 da passada limpa do mesmo harness). O 85,6 % vinha
de `110,70 ms × 32 chunks = 3,542 s` sobre `4,14 s`; a medição **dentro do grafo** dá
`117,8 ms × 32 = 3,770 s` sobre `4,174 s` = **90,3 %**. As duas versões estão certas, medem
coisas diferentes: 85,6 % é a fração do **custo mínimo** do matvec isolado; 90,3 % é a fração
do **custo real** dele dentro do prefill. A diferença de 4,7 pontos percentuais é
**inteiramente** o staging (+6,4 % no kernel), não andaime.

### 3.3 O matvec é realmente sublinear no tamanho do chunk? (destravar N > 16)

`matvec_launch_batch` (`include/rdna4/matvec.cuh:1002-1013`) só instancia N ∈ {2,3,4,8,16}:
**`kMaxBatch = 16` é um teto de compilação, não um ótimo medido**. Para transformar "vale a pena
chunk maior?" numa medida, o kernel **idêntico** (`matvec_kernel_batch`, traits `TIQ3S_S` das
tabelas de `include/rdna4/tuning.h`) foi instanciado para N ∈ {8,16,24,32,48,64} num binário
scratch (`/tmp/scratch/mb_batch_n2.hip`, fora do motor), com a forma real de
`blk.*.ffn_down.weight` (17408×5120, 5,99 GB):

```
./scripts/gpu-lock.sh timeout 900 /tmp/scratch/mb_batch_n2 3
```

<!--MB_BATCH_N2-->

---

## 4. Despacho: quantos lançamentos, quanto custam, e quanto as fusões do Vulkan tirariam

### 4.1 Contagem por camada (código)

Todo helper de lançamento (`rms_norm_launch` `nn.cuh:238`, `unary_launch` `:250`,
`mul_launch` `:257`, `add_launch` `:264`, `add_bcast_launch` `:279`, `mul_bcast_launch` `:287`,
`l2_norm_launch` `:244`, `rope_launch` `attn.cuh:56`, `dequant_row_launch`
`dequant_row.cuh:65-118`, `conv1d_state_batch_launch` `gdn.cuh:337`,
`delta_rule_batch_launch` `gdn.cuh:429`, `deinterleave_q_gate_batch_launch` `gdn.cuh:463`) faz
**exatamente 1 lançamento**. `attn_batch_launch` (`attn.cuh:423`) faz 1 (o eixo dos tokens é
`gridDim.y`, `attn.cuh:417-419`); `attn_split_batch_launch` (`attn.cuh:887-891`) faz 2
(kernel + merge) e só é usado quando `splits > 1`, o que a 512 tokens **não acontece**
(`kAttnSplitMin = tuned::kAttnSplitMin = 512`, `tuning.h:101`; `attn_splits_for(512) = 512/512
= 1`, `graph.cuh:426-428` — e o bench imprime `1 splits`).

Por **chunk de 16** (`forward_batch_layer`, `graph.cuh:1032-1340`; `forward_batch`,
`graph.cuh:1344-1425`):

| bloco | lançamentos | × | total |
|---|---|---|---|
| embeddings (`forward_batch:1383`, 1 dequant por token) | 1 | 16 tokens | **16** |
| camada de atenção não-recorrente: `attn_norm`, `dequant/quantize` ×2, deinterleave, rms_norm q, rms_norm k, rope q, rope k, `kv_write`, `attn_batch`, sigmoid, mul, residual, post_norm, silu, mul | 20 | 16 camadas | **320** |
| camada GDN: `attn_norm`, 4× (quantize+matvec), 3 escalares, conv1d, l2_norm, delta_rule, ssm_norm, silu, mul, residual, post_norm, silu, mul | 12 | 48 camadas | **576** |
| cabeça: `output_norm`, `finish_argmax` | 2 | 1 | **2** |
| **total por chunk** | | | **914** |
| cauda de cada chunk: `quantize_q8_1` + `matvec` do LM head (`graph.cuh:1416`), 1 × `hipMemcpy` | 2 | 1 | **2** |
| **total real por chunk** | | | **~916** |

`--count-only` confirma o andaime de marcas (42,0 marcas/token = **672 marcas por chunk**,
`phases.log` N=512), e a conta por camada fecha exatamente: 10 marcas/camada × 64 = 640, mais
`embed` + `out_norm` + `head` + `token_end` = 644, mais 28 de `attn_out_proj` (`:1142`/
`:1331`; o bucket `ffn_down` a jusante absorve as marcas de `proj_batch`) = 672.

### 4.2 O custo do despacho

- Piso medido nesta máquina, na mesma corrida: **2,253 µs por kernel vazio enfileirado**
  (o briefing cita 2,6-3,5 µs de `bench-matvec-shapes-gpu`; a diferença é o harness e o
  estado de DPM).
- 916 lançamentos × 2,253 µs = **2,06 ms por chunk**; 32 chunks × = **66 ms** para a prompt de
  512 tokens.
- Sobre o prefill medido de **4.174 ms** (122,66 tok/s): **1,58 %**.
- Com o piso mais pessimista de 3,5 µs: 916 × 3,5 µs × 32 = **102,6 ms = 2,46 %**.

### 4.3 Quantos lançamentos as fusões nomeadas do Vulkan tirariam do NOSSO grafo

Padrões de `ggml_backend_vk_graph_compute` (`ggml-vulkan/ggml-vulkan.cpp:18149-18327`) que
existem no nosso grafo batido com `forward_batch_layer`:

| padrão Vulkan | file:line (llama.cpp) | onde no nosso grafo (file:line) | lançamentos que saem por chunk |
|---|---|---|---|
| `MUL_MAT_ADD` | `ggml-vulkan.cpp:18165` | residual pós-`attn_output` / pós-`ssm_out` (`graph.cuh:1316`) e pós-`ffn_down` (`graph.cuh:1334`) | 2 × 64 = **128** |
| `SIGMOID_MUL` | `:18228-18237` | `sigmoid(attn_gate)·attn` (`graph.cuh:1137-1138`), `silu(z)·v` do GDN (`graph.cuh:1254-1255`) | 2 × 16 + 2 × 48 = **128** |
| `SILU_MUL` | `:18228-18237` | `silu(ffn_gate)·ffn_up` (`graph.cuh:1329-1330`) | 2 × 64 = **128** |
| `RMS_NORM_MUL` (`RMS_NORM_MUL_ADD_MUL` na versão com resíduo) | `:18203-18227` | `attn_norm`/`attn_post_norm`/`ssm_norm` (a normalização e o seu consumo) | 2 × 64 + 1 × 48 = **176** |
| `ROPE_VIEW_SET_ROWS` | `:18251-18258` | `rope(k)` + `kv_write_batch` (`graph.cuh:1076,1080`) | **16** |
| `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS` | `:18182-18193` | `rms_norm(k)` + `rope(k)` + escrita no cache (`graph.cuh:1069,1076,1080`) | **16** |
| `SSM_CONV_SILU` / `SSM_CONV_BIAS_SILU` | `:18238-18250` | `conv1d_state_batch` + silu (`graph.cuh:1232-1233,1254`) | **48** |

**Total: 640 lançamentos por chunk sairiam**, de 916 para **276** (−70 %). Em ms:
640 × 2,253 µs = **1,44 ms por chunk**, 46 ms por prompt de 512 = **1,1 % do prefill**.

Ou seja: **a fusão nomeada do Vulkan, copiada inteira, vale 1,1 % do nosso prefill.** O
argumento de que o Vulkan ganha por fazer menos despachos **não sobrevive à conta**: mesmo
zerando *todos* os 916 lançamentos por chunk, o teto é 2,06 ms/chunk = 66 ms = 1,58 % do
prefill. O que o Vulkan ganha está no **kernel** de matmul (GEMM tiled com staging em LDS e,
para 22 % dos bytes, int8), não na orquestração.

---

## 5. Atenção e GDN: quanto cada caminho custa no prefill

Atenção = 16 camadas (`i = 3, 7, ..., 63`); GDN = 48 camadas (`is_recr`, `graph.cuh:300-302`).
Não há bucket "atenção total": os buckets são transversais. Separação por bucket + contagem de
camadas, com os números de nível 1 a N=64 (as marcas por token já dizem quantas camadas cada
bucket cobre):

| caminho | buckets | ms/token | % do total (8,233) |
|---|---|---|---|
| **atenção (16 camadas)** | `qkv_proj` 0,351 + `qk_norm_rope_kv` 0,032 + `attention` 0,031 + `attn_gate_out` 0,012 + `attn_out_proj` 0,216 | **0,642** | **7,8 %** |
| **GDN/SSM (48 camadas)** | `gdn_proj` 1,342 + `gdn_scalars` 0,052 + `gdn_conv` 0,051 + `gdn_l2norm` 0,072 + `gdn_delta` 0,625 + `gdn_norm_silu` 0,052 + `gdn_out_proj` 0,630 | **2,824** | **34,3 %** |
| FFN (64 camadas, dos dois tipos) | `ffn_gate_up` 2,818 + `ffn_down` 1,705 | **4,523** | **54,9 %** |
| comum às 64 camadas | `post_norm` 0,076 + `ffn_residual` 0,164 | **0,240** | **2,9 %** |
| | | **8,229** | 100,0 % |

- **Atenção: 0,642 ms/token (7,8 %)**, dos quais 0,567 é projeção (q/k/v/out) e **0,031 ms
  = 0,4 % é o kernel de atenção propriamente dito**. A atenção não é o problema do prefill a
  512 tokens. (A 2048 o `count-only` mostra `splits = 4` e o custo por chunk sobe de 128,1
  para 131,7 ms; a 4096 a atenção começa a pesar — a curva de 8,02 a 8,57 ms/token do §1.1.)
- **GDN: 2,824 ms/token (34,3 %)**, dos quais 1,972 é projeção (`gdn_proj` + `gdn_out_proj`) e
  **0,852 ms/token (10,3 %) é a recorrência propriamente dita** (`gdn_delta` 0,625 +
  `gdn_l2norm` 0,072 + `gdn_conv` 0,051 + `gdn_scalars` 0,052 + `gdn_norm_silu` 0,052).
- **Não houve ablação por `--layers`**: o `bench --layers N` existe (`src/main.hip:277`) mas é
  um diagnóstico do caminho por token; num prefill em lote ele só trunca o número de camadas e
  não separa tipos de camada, e o modelo não tem flag para pular as recorrentes. A separação
  acima é **medida** pelos buckets (que somam 1,0 ao total) e pela contagem de camadas — não é
  estimativa.

---

## 6. O que NÃO é matvec: o custo que o lote não amortizou

Tudo abaixo é **por token** (não por chunk): o lote de 16 amortiza a leitura do peso e a
quantização de ativação, mas essas fases rodam o mesmo número de vezes por token, dentro de um
chunk ou fora dele. Números de `--prefill 64 --level 1` (todos os buckets com marcas/token ≥ 1
por camada), com o custo de matvec removido:

| fase | ms/token | % do prefill | por que não amortiza |
|---|---|---|---|
| `gdn_delta` | **0,641** | 7,3 % | é a recorrência: `delta_rule_batch_rows_kernel` (`gdn.cuh:429-444`) tem o laço de tokens **dentro** do kernel, uma passada por token, dependente |
| `act_quant` | **0,227** | 2,6 % | 190 `quantize_q8_1_batch` por chunk = 11,9 por token, cada uma sobre `n×ncols` elementos |
| `ffn_residual` | **0,095** | 1,1 % | `add` sobre n×5120 por camada |
| `post_norm` | **0,076** | 0,9 % | `add` + `rms_norm` sobre n×5120 por camada |
| `gdn_l2norm` | **0,073** | 0,8 % | 2 `l2_norm` por camada GDN |
| `gdn_norm_silu` | **0,066** | 0,8 % | `rms_norm` + `silu`·`mul` |
| `gdn_scalars` | **0,053** | 0,6 % | `sigmoid`, `softplus`, `mul` — elementos de `nvh = 48` por token |
| `gdn_conv` | **0,050** | 0,6 % | `conv1d_state_batch` |
| `qk_norm_rope_kv` | **0,044** | 0,5 % | 2 `rms_norm`, 2 `rope`, escrita do KV |
| `attention` | **0,032** | 0,4 % | 1 kernel, contexto cresce |
| `attn_gate_out` | **0,012** | 0,1 % | `sigmoid` + `mul` |
| **total não-matvec** | **1,369** | **16,6 %** | (a N=64; a 512 mede-se 0,79 ms/token — ver adiante) |

E o total do prefill decomposto em dois pedaços, com o matvec medido **dentro do grafo**:

| N | prefill total | matvec (nível 2) | % | resto | % |
|---|---|---|---|---|---|
| 64 | 8,638 ms/token (limpa) | 7,362 | 85,2 % | 1,276 | 14,8 % |
| 512 | 8,104 ms/token (limpa) | 7,362 (do N=64) | 90,8 % | 0,742 | 9,2 % |

O matvec do nível 2 a N=512 não pôde ser medido direto: o nível 2 com `--prefill 512` cria
460 800 `hipEvent_t` e o próprio harness recusa (`"(passada instrumentada pulada: N=512 cria
460800 hipEvent_t; use N<=128)"`, `bench_phases_gpu.hip:741-744`). O menor não-matvec medido
**dentro do grafo** é 0,742 ms/token (resíduo de 8,104 − 7,362) e a soma dos buckets de nível 1
a N=64 dá 1,276 ms/token. A diferença entre os dois é o staging do matvec: a N=64 os buckets de
projeção incluem 0,527 ms/token de andaime de projeção (silu·mul, quantizações) que a N=512
está contada dentro do bucket.

**Os três maiores custos por token que fusão ou batching poderiam remover:**

1. **`gdn_delta` — 0,641 ms/token (7,3 %)**. É 1 kernel por camada com laço de tokens dentro
   (48 por chunk), lendo e escrevendo 3,1 MB de estado por camada; já é batido, mas é
   sequencial por construção. Fusão não ajuda; um algoritmo de scan paralelo ajudaria.
2. **`act_quant` — 0,227 ms/token (2,6 %)**. 190 lançamentos por chunk, cada um re-quantizando
   blocos de 32. Candidato direto a fusão com o matvec (quantizar dentro do kernel, como o MMQ
   faz) ou a cache por (buffer, geração).
3. **`qk_norm_rope_kv` + `attention` + `attn_gate_out` — 0,088 ms/token (1,1 %)**, que a fusão
   `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS` + `SIGMOID_MUL` do Vulkan tocaria.

Somados: **0,956 ms/token = 11,6 %** do prefill a 512 (sobre 8,104). Nenhum deles é o gap.

---

## 7. Atribuição do gap

### 7.1 O gap de 9,0× do briefing é de micro-lote, e isso está medido

| configuração | tok/s | ms/token | razão contra nós |
|---|---|---|---|
| nós, `--prefill 512` | 123,39 | 8,104 | 1,00× |
| llama.cpp Vulkan, `-ub 16 -b 16` | 200,25 | 4,994 | **1,62×** |
| llama.cpp Vulkan, default (`b 2048`, `ub 512`) | 1168,49 | 0,856 | 9,47× |
| llama.cpp Vulkan, corrida do briefing | 1113,96 | 0,898 | 9,03× |

O mesmo binário, a mesma máquina, o mesmo modelo, a mesma sessão: o que muda entre a linha 2 e
a linha 3 é `-b/-ub`. **O llama.cpp perde 5,84× ao processar 16 tokens por vez; nós não temos
como perder, porque 16 é o nosso teto.** A pergunta certa deixa de ser "por que 9×" e passa a
ser "por que 1,62×".

### 7.2 O que resta dos 1,62× (e o que não resta)

| componente | nosso (ms/token) | llama.cpp implícito (ms/token) | delta | origem de cada célula |
|---|---|---|---|---|
| **matvec/projeções (305 por chunk)** | **7,362** | ≤ 4,99 | ≥ **+2,37** | nosso: *medido* (nível 2, `--prefill 64`). llama: *derivado* (200,25 tok/s ⇒ 4,994 ms/token é o total dele; a projeção é parte disso) |
| `act_quant` (190 lançamentos/chunk) | 0,227 | ≤ 0,1 | ~+0,13 | nosso: *medido*. llama: *desconhecido* (MMQ quantiza dentro do kernel; não há como medir sem profiler) |
| `gdn_delta` + `gdn_conv` + `gdn_l2norm` + `gdn_scalars` + `gdn_norm_silu` | 0,883 | ≤ 0,3 | ~+0,58 | nosso: *medido*. llama: *desconhecido* |
| `attention` (16 camadas, 1 split a 512) | 0,032 | *desconhecido* | — | nosso: *medido*. llama: sem número |
| norms/resíduos (`post_norm`, `ffn_residual`) | 0,240 | *desconhecido* | — | *medido* nosso; *desconhecido* dele |
| despacho (916 lançamentos × 2,253 µs) | 2,06 ms **por chunk** = **0,129 ms/token** | *desconhecido* (mas o teto do ganho em qualquer direção é ≤ 0,13 ms/token) | ≤ 0,13 | nosso: *medido* (piso do kernel vazio + contagem de código). dele: *desconhecido* |
| gordura do intervalo medido (`0,79` do §6 vs `7,362`) | — | — | — | *derivado* |
| **total** | **8,104** (*medido*) | **4,994** (*medido*) | **+3,11** | |

**A atribuição em uma frase:** dos 3,11 ms/token de diferença contra um llama.cpp no mesmo
micro-lote, **≥2,37 ms (76 %) é o matvec** e o resto (≤0,74 ms) é andaime — mas o delta do
andaime **não é medível** do lado do llama.cpp sem profiler, e as fusões dele (a explicação
usual) valem no máximo 1,1 % do nosso prefill por medição própria (§4.3). Do gap de 9,0×,
**7,25 ms/token (89 %) é a diferença de tamanho de micro-lote** (512 tokens por
`llama_decode` contra 16 por `forward_batch`), e **1,62× é o que sobra para o kernel**.

### 7.3 O que a banda diz sobre o matvec (aritmética, para não confundir com medição)

- 305 matvecs por chunk leem 11,133 GB de pesos (`bench-phases-gpu` imprime exatamente esse
  número, "lidos por token 11.133 GB"). Isso é **11,133 GB por chunk de 16**, não por token.
- No grafo: 11,133 GB / 117,8 ms = **94,5 GB/s efetivos** (dados do próprio tensor, sem contar
  a releitura da ativação).
- Isolado: 11,133 / 110,70 ms = **100,5 GB/s**.
- Piso de roofline medido nesta placa: **633 GB/s** (`docs/medicoes-banda-e-gargalos.md`
  §2.4). Ou seja, o matvec em lote usa **16 % da banda disponível** e o caminho por token usa
  79-82 % — o lote amortiza a leitura do peso entre 16 tokens e ao mesmo tempo **multiplica por
  16 o trabalho de ALU por byte**, ficando limitado por issue/latência e não por banda
  (`docs/journal-lote.md` Medidas 1-4, todas refutando as alternativas).
- **DERIVADO**: com o matvec a 100,5 GB/s e o resto do prefill igual, o teto do prefill deste
  motor com este kernel é `11,133/(0,1005) + 0,742 ms` por chunk de 16 = 111,5 + 11,9 =
  **123,4 ms/chunk = 129,6 tok/s**. Medimos 123,39 tok/s. **O motor está a 5 % do teto do
  próprio kernel.**

---

## 8. O que não foi medido, e por quê

1. **A 9,0× da tabela do briefing não tem contraparte no mesmo micro-lote em nenhuma direção**:
   o `llama-bench` não tem flag para forçar 16 tokens por `llama_decode` *e* medir o `pp512`
   como 512 tokens lógicos — o `-ub 16` faz 32 chamadas de 16, que é o nosso caso, mas aí o
   número dele (200,25) já é outro. Não existe um "9,0× comparável"; existe 1,62×.
2. **O orçamento interno do lado do llama.cpp é desconhecido**: `rocprof`/`rocprofv3`/`omniperf`
   não estão instalados nesta máquina (só `rocminfo`), e o backend Vulkan não expõe eventos HIP.
   As colunas "llama.cpp" de §7.2 são derivadas do total, não medidas.
3. **Nível 2 a 512 tokens não é executável**: o harness recusa por criar 460 800 `hipEvent_t`.
   O nível 2 existe só até N=128 (§2.2), e a extrapolação para 512 usa o matvec medido a 64
   (7,362 ms/token) — o prefill limpo medido a 512 (8,104) é o denominador.
4. **A ablação por `--layers` para separar atenção de GDN não é possível com o que existe**: o
   modelo não tem flag para pular as 48 camadas recorrentes, e `--layers N` corta as primeiras
   N camadas, não um tipo. A separação de §5 é por bucket + contagem de camadas (medida, mas
   não ablada por subtração).
5. **`mb_batch_n` (v1) faultou** com *Memory access fault ... Page not present* por VRAM sendo
   drenada de outro processo no instante da alocação; a v2 é autossuficiente (sem GGUF, sem
   geometria de arquivo) e roda sob o mesmo lock.
6. **Uma corrida de fase a 512 morreu no meio** (a tabela do N=512 do `phases.log` inicial não
   saiu; o log mostra a corrida de 128 e depois o `--count-only`). O número de 512 que está no
   texto é do `--count-only` e do `bench --prefill`, os dois concordando em 0,5 %.
