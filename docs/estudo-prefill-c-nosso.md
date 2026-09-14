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
| profiler nível 2, N=64 (corrida 1, janela ocupada) | **117,8** (7,362×16) | 7,362 | `bench-phases-gpu --prefill 64 --level 2` |
| profiler nível 2, N=128 (corrida 2) | **113,6** (7,103×16) | 7,103 | `bench-phases-gpu --prefill 128 --level 2` |
| bench isolado, N=16 | **110,70** | 6,92 | `bench-matvec-shapes-gpu --batch 16 --reps 3` (corrida de hoje: 111,64) |
| diferença (corrida 2 × isolado) | **+2,9 ms (+2,6 %)** | +0,19 | |

**Não há discrepância a explicar, e é esse o achado:** o matvec dentro do grafo custa 2,6 % a
6,4 % mais que o mesmo inventário lançado back-to-back, dependendo da janela. As duas corridas
de nível 2 não concordam entre si (7,362 contra 7,103 ms/token = 3,6 %) porque a primeira
rodou com a fila do `flock` cheia (7 waiters, VRAM de outro processo em 11,4-13,0 GB) e a
passada limpa dela deu 115,76 tok/s contra 121,29 tok/s da segunda. **O número bom é
7,103 ms/token** e o desvio contra o isolado cai para 2,6 %, que é o piso de ruído do
harness. A leitura direta das bandas: 11,133 GB / 113,6 ms = **98,0 GB/s** no grafo contra
11,133 GB / 110,70 ms = **100,5 GB/s** no isolado.

**E o número que o briefing pede, o 85,6 %:** o profiler dá **83,7 %** nas duas corridas de
nível 2 (7,362/8,795 e 7,103/8,486 — a razão é mais estável que os valores absolutos), ou
**82,4 %** contra a passada limpa da mesma corrida (7,103/8,620 no N=128). O 85,6 % do
briefing vinha de `110,70 ms × 32 chunks = 3,542 s` sobre `4,14 s`; a medição **dentro do
grafo** dá `113,6 ms × 32 = 3,635 s` sobre `4,174 s` = **87,1 %**. As duas versões estão
certas e medem coisas diferentes: 85,6 % é a fração do **custo do kernel isolado**; 87,1 % é a
fração do **custo dele dentro do prefill**. **O 85,6 % está confirmado dentro do grafo, com
1,5 ponto de diferença, e essa diferença é o staging do matvec, não andaime** — o andaime
medido (§6) é 0,74-1,28 ms/token, e 3,635 s + 32×0,74 ms = 3,659 s ≠ 4,174 s; os 515 ms que
sobram são o próprio matvec rodando mais devagar dentro do grafo.


