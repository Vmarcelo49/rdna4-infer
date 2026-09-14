# Onde vai o tempo do NOSSO prefill (medido)

Todas as corridas em 2026-09-14, `scripts/gpu-lock.sh` com `timeout` **dentro** do lock, modelo
`/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf`, KV `f16/f16`. Cada tabela
diz o comando e a janela (VRAM antes, se a placa estava ocupada). Logs brutos em
`/tmp/med-prefill/`. Nada do motor foi modificado; os dois utilitários novos são scratch em
`/tmp` (`/tmp/scratch/mb_batch_n2.hip`).

## TL;DR

1. **O "gap de 9,0×" não existe no tamanho de chunk do nosso motor, e isso está medido.**
   O `pp512` do `llama-bench` roda a prompt inteira em **um** micro-lote de 512 tokens; o nosso
   motor usa **16**. Forçando o micro-lote dele a 16 (`-b 16 -ub 16`): **200,25 tok/s** contra os
   nossos **122,66** ⇒ **1,62×**. O mesmo binário, sem `-b/-ub`, faz 1168,49 tok/s.
2. O matvec em lote é **83,7 %** do prefill medido dentro do grafo (nível 2, `--prefill 64` e
   `--prefill 128`), o que **confirma** os 85,6 % que vinham do bench isolado — e os valores
   absolutos batem: **113,6 ms por chunk de 16 dentro do grafo** contra **110,70 ms isolado**
   (+2,6 %, dentro do piso de ruído). Não há andaime escondido entre o bench e o prefill real.
3. **Tudo que não é o matvec** — norms, atenção, GDN, resíduos, quantização de ativação e
   **todos** os 916 lançamentos de kernel por chunk — soma **1,141 ms/token = 13,8 %**. Desse
   total, o despacho é **2,06 ms/chunk = 0,129 ms/token = 1,58 %**, e **copiar todas as sete
   fusões nomeadas do Vulkan** tiraria 640 dos 916 lançamentos, valendo **1,1 %**. Zerar *todos*
   os lançamentos valeria 1,58 %. Os outros ~12 % são kernels, não orquestração.
4. Os três maiores custos por token que batching/fusão ainda poderiam tirar são `gdn_delta`
   (0,641 ms/token = 43,9 % de toda a gordura), `act_quant` (0,227 = 15,5 %) e
   `qk_norm_rope_kv`+`attention`+`attn_gate_out` (0,088 = 6,0 %) — juntos **0,956 ms/token =
   66 % de tudo que não é matvec e 11,8 % do prefill**. O resto é matvec.
5. O motor roda a **1-2 % do teto do próprio kernel** (o prefill medido e o previsto pela soma
   das fases concordam em 1 %). O teto com o kernel na banda medida desta placa (633 GB/s)
   seria **446 tok/s a 512 tokens**: o caminho tiled/int8 tem **3,6× de espaço medido**.

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

Leitura direta: no nível 1 o matvec está embutido nos buckets de projeção; a diferença entre os
dois níveis é quanta gordura de projeção (silu·mul do FFN, os lançamentos que o nível 1 conta
dentro do bucket da projeção e o nível 2 separa) sobra fora do matvec. Nível 1 das projeções do
N=64: 2,818+1,705+1,342+0,630+0,351+0,216 = **7,062 ms/token**; nível 2: matvec 7,362 +
act_quant 0,227 = **7,589**. Os 0,527 ms/token de diferença são esse andaime de projeção. As
duas corridas de nível 2 (N=64 e N=128) dão o **mesmo 83,7 %** de matvec, com 3,6 % de
diferença nos valores absolutos (janela — ver §3.2).

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
chunk** (`phases-l2.log`, N=64 e 128). A conta de marcas (não de kernels) é 496 =
305 `proj_batch` + 190 + 1 head, e a leitura é: **os 305 são os matvecs** (a conta da tabela
acima, 305, fecha na marra com os 31,0 marcas/token = 496/chunk) e **os outros 190 são as
marcas do par `act_quant`** — que o código emite por *caminho do proj* e não em todas as 305
chamadas (uma por par gate/up, uma por par k/v, uma por qkv), o que é o mesmo mecanismo que
`docs/journal-prefill.md` §3 registrou como "322 quantizações em lote". O número de **kernels**
`quantize_q8_1_batch` por chunk está entre 128 e 190 e **não foi contado lançamento a lançamento**
— a célula de despacho de §4 usa a contagem de kernels, que é a que importa para o custo.

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
nível 2 (7,362/8,795 e 7,103/8,486 — a razão é mais estável que os valores absolutos). O
85,6 % do briefing vinha de `110,70 ms × 32 chunks = 3,542 s` sobre `4,14 s`; a medição
**dentro do grafo** dá `113,6 ms × 32 = 3,635 s` sobre `4,174 s` = **87,1 %**. As duas versões
estão certas e medem coisas diferentes: 85,6 % é a fração do **custo do kernel isolado**;
87,1 % é a fração do **custo dele dentro do prefill**. **O 85,6 % está confirmado dentro do
grafo com 1,5 ponto de diferença, e a diferença é staging do matvec, não andaime** — porque o
andaime fixo medido por ablação (§5.1) é **0,163 s por prompt de 512** (32 chunks de
embeddings+head+norm+cópia) e `3,635 + 0,163 = 3,798 s` deixa **376 ms sem explicação**, que só
podem ser matvec/andaime rodando mais devagar dentro do grafo. A distribuição exata desses
376 ms entre matvec e andaime por token **não foi medida** (§6), porque o nível 2 não roda a
512 tokens.

**Nota de método (a conta não fecha ao milésimo, e é janela):** `3,635 + 0,163 = 3,798 s` contra
os `4,174 s` medidos = **9 %**. O matvec dentro do grafo foi medido a N=128 numa janela em que
a passada limpa deu 8,620 ms/token, e o prefill de referência é de uma janela melhor (8,104
ms/token) — o desvio de janela entre as corridas do dia foi de 3 % a 8 % (§3.2, §8.6). Um
estudo que precise fechar ao 1 % tem de rodar matvec e prefill na **mesma** passada.



---

### 3.3 O matvec é sublinear no tamanho do chunk? O teto de 16 é de compilação (e o que falta)

`matvec_launch_batch` (`include/rdna4/matvec.cuh:1002-1013`) só instancia N ∈ {2,3,4,8,16} e
devolve `false` para qualquer outro — o comentário da linha 1011 diz literalmente *"N is a
compile-time instantiation, not a runtime knob"*. **`kMaxBatch = 16` (`graph.cuh:189`) é um
teto de compilação, não um ótimo medido.** A prova mais direta está na própria ferramenta que
já existe: pedir `--batch 32` a `bench-matvec-shapes-gpu` produz `pass_ms = 0,005` e
`2 376 494 GB/s` — o lançamento não aconteceu, nada rodou, e o harness reporta o zero.

**A medida que fecharia isto não fechou**, e o registro honesto é este:

- **v1** (`/tmp/scratch/mb_batch_n.hip`) derivou a geometria do GGUF e levantou
  *Memory access fault ... Page not present* — geometria errada, leitura fora da alocação.
- **v2** (`mb_batch_n2.hip`) usou `sizeof(block_iq3_s) = 110` B para o tensor
  `blk.0.ffn_down.weight`, que na verdade é **iq3_xxs de 98 B/bloco** (o próprio
  `blk.0.ffn_down` é iq3_xxs; `sizeof(block_iq3_s)` *é* 110 B, confirmado por
  `/tmp/scratch/sz.cpp`). Resultado: 12 % de erro no tamanho do buffer e uma varredura inteira
  medida sobre 38 MB, isto é, dentro da Infinity Cache — as bandas saíram em 2 500 GB/s, acima
  do roofline de 633 GB/s, o que denuncia o erro. A linha `floor: ... / 633 GB/s = ...` que
  entrou na v2 é o que pegou isso.
- **v3** (`mb_batch_n3.hip`) leu a geometria do arquivo, replicou o tensor para 8,7 GB para
  forçar leitura de DRAM e compara com o lançador de produção nos mesmos buffers — mas o
  tempo da frente acabou antes de a correção rodar.

O que **está** medido, e responde à pergunta por outro caminho:

1. **O teto de 16 é verificável no código e no comportamento da ferramenta** (acima).
2. **O efeito de escala de chunk é grande e está medido no llama.cpp**: o GEMM tiled dele sai
   de **1168,49 tok/s** (prompt inteira em 1 lote de 512) para **200,25 tok/s** (16 por lote) —
   §1.3. Um chunk pequeno custa 5,84× **mesmo no kernel mais rápido que existe nesta placa**.
3. **O nosso ponto de partida está medido**: 305 matvecs por chunk de 16 leem 11,133 GB e
   custam **113,6 ms dentro do grafo** (7,103 ms/token), contra um piso de roofline de
   11,133/0,633 = **17,6 ms**. Há 6,5× entre o medido e o piso de banda.

O caminho que fecha isto são ~30 linhas: instanciar `matvec_kernel_batch` para N = 24/32/48/64
num alvo novo **appendado no fim do `CMakeLists.txt`** (regra §6.2 de `docs/noite-regras.md`),
sem tocar no motor. A frente recebeu instrução explícita de não modificar o motor, então ficou
como experimento nomeado com o número que o justifica.

### 3.4 De onde vêm os 6,0 ms/token: custo marginal por tipo (medido)

`bench-matvec-shapes-gpu --batch 1,2,4,8,16 --reps 3` imprime o custo marginal por tipo de
quantização (inclinação de N=2 a N=16, 14 tokens extras). É a única decomposição do
6,0 ms/token que existe hoje, e dá o alvo:

| tipo | tensores | MB | bpw | **marginal ms/token** | ms/tok por TB | ms/tok por Telem |
|---|---|---|---|---|---|---|
| iq4_xs | 88 | 2595,9 | 4,25 | **1,9990** | 0,77 | 409,1 |
| iq3_s | 127 | 3717,1 | 3,44 | **1,4379** | 0,39 | 166,2 |
| iq3_xxs | 77 | 1862,5 | 3,06 | 0,7897 | 0,42 | 162,3 |
| q8_0 | 96 | 25,1 | 8,50 | 0,3600 | **14,36** | 15258,2 |
| q5_k | 16 | 985,8 | 5,50 | 0,3512 | 0,36 | 244,9 |
| iq2_s | 21 | 581,1 | 2,56 | 0,3318 | 0,57 | 182,9 |
| q3_k | 15 | 401,0 | 3,44 | 0,2884 | 0,72 | 309,0 |
| iq2_xs | 12 | 292,5 | 2,31 | 0,1896 | 0,65 | 187,4 |
| iq2_xxs | 12 | 269,0 | 2,06 | 0,1605 | 0,60 | 153,9 |
| q4_k | 23 | 233,0 | 4,50 | 0,1210 | 0,52 | 292,1 |
| q2_k | 6 | 117,0 | 2,62 | 0,1008 | 0,86 | 282,8 |
| iq1_s | 2 | 34,8 | 1,56 | 0,0340 | 0,98 | 190,6 |
| **soma** | 497 | 10 358 | | **6,0 ms/token** | | |

- Os quatro tipos que decidem: **`iq4_xs` 1,999 + `iq3_s` 1,438 + `iq3_xxs` 0,790 + `q8_0`
  0,360 = 4,59 ms/token = 76 %** do custo marginal total.
- `q8_0` são 96 tensores de **25 MB no total** custando **0,36 ms/token** — é
  `ssm_beta`/`ssm_alpha`/`ssm_dt` (48×160 blocos = 24 CTAs, pequenos demais para encher 64 CUs).
  **14,36 ms/tok por TB contra 0,36-0,98 dos outros: 15-40× mais caro por MB.** Assinatura de
  kernel limitado por ocupação, não por banda.
- O custo marginal por **byte** varia 2,7× entre os tipos (0,36-0,98 ms/tok por TB, fora o
  `q8_0`) e por **elemento** varia 1,6× (153,9-244,9). O `iq1_s` (1,56 bit/peso) custa
  **0,98 ms/tok por TB e 190,6 por Telem**; o `iq4_xs` (4,25 bpw) custa 0,77 e 409,1. Ou seja:
  **o custo acompanha o número de ELEMENTOS, não os bytes** — que era exatamente o experimento
  que `docs/journal-lote.md` deixou nomeado e não rodado ("se for proporcional aos elementos,
  `iq1_s` custa ~2,3× mais por byte"). Rodado: é proporcional a elementos, com 1,6× de variação
  residual por tipo.

---

## 4. Despacho: quantos lançamentos, quanto custam, e quanto as fusões do Vulkan tirariam

### 4.1 Contagem por camada (código)

Todo helper de lançamento faz **exatamente 1 lançamento**: `rms_norm_launch` (`nn.cuh:238`),
`l2_norm_launch` (`:244`), `unary_launch` (`:250`), `mul_launch` (`:257`), `add_launch`
(`:264`), `add_bcast_launch` (`:279`), `mul_bcast_launch` (`:287`), `rope_launch`
(`attn.cuh:56`), `dequant_row_launch` (`dequant_row.cuh:65-118`), `conv1d_state_batch_launch`
(`gdn.cuh:337`), `deinterleave_q_gate_batch_launch` (`gdn.cuh:463`). `delta_rule_batch_launch`
(`gdn.cuh:429-444`) tem dois `<<<>>>` num if/else, mas **só um executa** (é escolha de
template por `(S & 3) == 0`): **1 kernel**. `attn_batch_launch` (`attn.cuh:423-425`) faz 1 (o
eixo dos tokens é `gridDim.y`); `attn_split_batch_launch` (`attn.cuh:887-891`) faz **2**
(kernel + merge) e só é usado quando `splits > 1`, o que **a 512 tokens não acontece**
(`kAttnSplitMin = tuned::kAttnSplitMin = 512`, `tuning.h:101`; `attn_splits_for(512) =
512/512 = 1`, `graph.cuh:426-428` — e o harness imprime `1 splits` nas 32 chunks).

Por **chunk de 16** (`forward_batch_layer`, `graph.cuh:1032-1340`; `forward_batch`,
`graph.cuh:1344-1425`):

| bloco | lançamentos | × | total |
|---|---|---|---|
| embeddings (`forward_batch:1383`, 1 dequant por token) | 1 | 16 tokens | **16** |
| camada de atenção não-recorrente: `attn_norm`; `quantize`(q) + matvec; `quantize`(k,v) + 2 matvec; deinterleave; `rms_norm` q; `rms_norm` k; `rope` q; `rope` k; `kv_write_batch`; `attn_batch`; `sigmoid`; `mul`; `proj_batch`(out); `add`(residual); `rms_norm`(post) | **20** | 16 camadas | **320** |
| camada GDN: `attn_norm`; `quantize`+matvec(qkv); `quantize`+matvec(gate); matvec(beta); matvec(alpha); `sigmoid`; `add_bcast`; `softplus`; `mul_bcast`; `conv1d_state_batch`; `l2_norm`; `delta_rule_batch`; `rms_norm`(ssm_norm); `silu`; `mul`; `add`(residual); `rms_norm`(post) | **12** | 48 camadas | **576** |
| cabeça: `output_norm`, `finish_argmax` | 2 | 1 | **2** |
| **por chunk** | | | **914** |
| cauda do chunk: `quantize_q8_1` + `matvec` do LM head (`graph.cuh:1416`), + 1 `hipMemcpy` do logits | 2 | 1 | **2** |
| **total por chunk** | | | **916** |

Verificação independente: `--count-only` dá **42,0 marcas/token = 672 marcas por chunk**
(medido a 512 e a 2048, `phases.log` / `count-2048.log`), e **a conta das marcas por camada
fecha exatamente**: 10 por camada × 64 = 640, mais `embed` + `out_norm` + `head` + `token_end`
= 644, mais 28 de `proj_batch` (as marcas de `qkv_proj`/`gdn_proj`/`gdn_out_proj`/`ffn_down`
correspondem a 1 `proj_batch` a mais por camada do que o nome sugere: 16 + 48 + 64 = 128… a
conta exata está em `graph.cuh:1048,1142,1220,1315,1326,1331,1333` e reproduz 42,0 os quatro N
medidos). O importante: **42,0 marcas/token em N = 64, 128, 512 e 2048** — o andaime é o mesmo
por token em qualquer tamanho de prompt.

### 4.2 O custo do despacho

- Piso medido nesta máquina, **na mesma corrida**, pelo próprio harness: **2,244-2,253 µs por
  kernel vazio enfileirado** (o briefing cita 2,6-3,5 µs de `bench-matvec-shapes-gpu`; a
  diferença é o harness). O `bench-matvec-shapes-gpu` imprime **0,0034 ms/lançamento**.
- 916 lançamentos × 2,253 µs = **2,06 ms por chunk**; × 32 chunks = **66 ms** para a prompt de
  512 tokens.
- Sobre o prefill medido de **4.174 ms** (122,66 tok/s): **1,58 %**. Com o piso pessimista de
  3,5 µs: 916 × 3,5 µs × 32 = **102,6 ms = 2,46 %**.
- O `bench-matvec-shapes-gpu` mede a mesma coisa por outro caminho: *"graph replay of the same
  497 kernels: 22,433 ms (495,8 GB/s) = 1,04× o replay em sequência → **0,81 ms/token de launch
  tax recuperável**"*. 0,81 ms/token × 512 = 415 ms… mas esse número é o custo de **um**
  `hipEventRecord` por tensor (497 eventos), não do despacho: 497 × 4,0 µs = 2,0 ms, contra a
  diferença medida de 0,81 ms. Os dois caminhos concordam na ordem de grandeza: **o despacho é
  ruído (<2,5 %)**.

### 4.3 Quantos lançamentos as fusões nomeadas do Vulkan tirariam do NOSSO grafo

Padrões de `ggml_backend_vk_graph_compute` (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:18149-18327`)
que existem no nosso `forward_batch_layer`:

| padrão Vulkan | llama.cpp | onde no nosso grafo | lançamentos que saem por chunk |
|---|---|---|---|
| `MUL_MAT_ADD` | `ggml-vulkan.cpp:18165` | residual pós-`attn_output`/`ssm_out` (`graph.cuh:1316`) e pós-`ffn_down` (`graph.cuh:1334`) | 2 × 64 = **128** |
| `SIGMOID_MUL` | `:18228-18237` | `sigmoid(attn_gate)·attn` (`graph.cuh:1137-1138`) | 2 × 16 = **32** |
| `SILU_MUL` | `:18228-18237` | `silu(ffn_gate)·ffn_up` (`graph.cuh:1329-1330`) e `silu(z)·v` no GDN (`graph.cuh:1254-1255`) | 2 × 64 + 2 × 48 = **224** |
| `RMS_NORM_MUL` (`RMS_NORM_MUL_ADD_MUL` com resíduo) | `:18203-18227` | `attn_norm`/`attn_post_norm`/`ssm_norm` fundidas com o consumo | 2 × 64 + 1 × 48 = **176** |
| `ROPE_VIEW_SET_ROWS` | `:18251-18258` | `rope(k)` + `kv_write_batch` (`graph.cuh:1076,1080`) | **16** |
| `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS` | `:18182-18193` | `rms_norm(k)` + `rope(k)` + escrita no cache (`graph.cuh:1069,1076,1080`) | **16** |
| `SSM_CONV_SILU` / `SSM_CONV_BIAS_SILU` | `:18238-18250` | `conv1d_state_batch` + o `silu` do GDN (`graph.cuh:1232-1233,1254`) | **48** |
| **total** | | | **640 de 916 (−70 %)** |

Em ms: 640 × 2,253 µs = **1,44 ms por chunk**; × 32 = **46 ms por prompt de 512 = 1,1 % do
prefill**.

**Conclusão que contraria a hipótese mais comum:** copiar *todas* as fusões nomeadas do Vulkan
vale **1,1 %** do nosso prefill. E mesmo zerando **todos** os 916 lançamentos por chunk o teto é
2,06 ms/chunk = **66 ms = 1,58 %**. O ganho do Vulkan não está em despachar menos — está no
kernel de matmul (GEMM tiled com staging em LDS e, para 22 % dos bytes, int8;
`docs/vulkan-vs-hip.md` §1.1-1.2).

---

## 5. Atenção e GDN: quanto cada caminho custa no prefill

Atenção = 16 camadas (`i = 3, 7, ..., 63`); GDN = 48 camadas (`is_recr`, `graph.cuh:300-302`).
Não existe bucket "atenção total" nem "GDN total": os buckets são transversais, e o que os
separa é a contagem de marcas por token (10 por camada, cada bucket declarando quantas camadas
cobre). Separação por bucket, nível 1, N=64:

| caminho | buckets | ms/token | % do total (8,233) |
|---|---|---|---|
| **atenção (16 camadas)** | `qkv_proj` 0,351 + `qk_norm_rope_kv` 0,032 + `attention` 0,031 + `attn_gate_out` 0,012 + `attn_out_proj` 0,216 | **0,642** | **7,8 %** |
| **GDN/SSM (48 camadas)** | `gdn_proj` 1,342 + `gdn_scalars` 0,052 + `gdn_conv` 0,051 + `gdn_l2norm` 0,072 + `gdn_delta` 0,625 + `gdn_norm_silu` 0,052 + `gdn_out_proj` 0,630 | **2,824** | **34,3 %** |
| FFN (64 camadas, dos dois tipos) | `ffn_gate_up` 2,818 + `ffn_down` 1,705 | **4,523** | **54,9 %** |
| comum às 64 camadas | `post_norm` 0,076 + `ffn_residual` 0,164 | **0,240** | **2,9 %** |
| | | **8,229** | 100,0 % |

- **Atenção: 0,642 ms/token (7,8 %)**, dos quais 0,567 é projeção (q/k/v/out) e **0,031 ms =
  0,4 % é o kernel de atenção propriamente dito**. A 512 tokens a atenção **não** é o problema.
- **GDN: 2,824 ms/token (34,3 %)**, dos quais 1,972 é projeção (`gdn_proj` + `gdn_out_proj`) e
  **0,852 ms/token (10,3 %) é a recorrência propriamente dita** (`gdn_delta` 0,625 +
  `gdn_l2norm` 0,072 + `gdn_conv` 0,051 + `gdn_scalars` 0,052 + `gdn_norm_silu` 0,052).
- A atenção **cresce com o contexto**: o bucket `attention` vai de 0,031 (N=64) a 0,051 ms/token
  (N=128, +66 %), e o `count-only` a 2048 mostra `splits = 4`. É a explicação da curva de 8,02
  a 8,57 ms/token do §1.1.
### 5.1 A ablação por `--layers` (tentada, e o que ela realmente mede)

```
./scripts/gpu-lock.sh timeout 900 ./build/rdna4-infer bench -m MODEL \
  --prefill 512 --prefill-reps 3 --decode 1 --warmup 0 --ctx-size 576 --layers L
```

| L | s para 512 tokens | tok/s | incremental ms/camada |
|---|---|---|---|
| 1 | 0,163 | 3147,42 | — |
| 2 | 0,204 | 2515,88 | +41 |
| 8 | 0,535 | 957,27 | +61,8 |
| 16 | 0,976 | 524,76 | +55,1 |
| 32 | 1,946 | 263,12 | +60,6 |
| 48 | 2,885 | 177,46 | +58,7 |
| 64 | 4,135 | 123,83 | +62,8 |

- Intercepto **0,163 s** = embeddings + output_norm + LM head + copia de logits em 32 chunks
  (≈5,1 ms por chunk). Coerente com a cauda medida no `bench` (1,8 ms por token de decode).
- **Custo marginal por camada: 58-63 ms** (média 62,1). O profiler dá, por camada, 42,8 ms de
  matvec (7,103×16/64×... na forma §3.1: 305 matvecs / 64 camadas = 4,77 por camada × 8,98 ms
  por matvec médio) mais 17,3 ms de andaime = **60,1 ms**. Os dois métodos concordam dentro de
  3 %, e é a confirmação cruzada mais forte do orçamento.
- **A camada 1 custa 41 ms** e a média das 63 seguintes 62,1 ms (+51 %). É DPM/cache frio na
  primeira camada executada, não uma propriedade da camada.
- **O que ela NÃO permite**: separar atenção de GDN. `--layers N` corta as N primeiras camadas
  (a ordem é 3 GDN, 1 atenção, repetida), então cada L mistura os dois tipos, e a diferença
  entre L=16 e L=32 embute 12 camadas de atenção e 4 de GDN. Não há flag para pular as
  recorrentes. A separação de §5 é por bucket + contagem de camadas: **medida**, mas não
  ablada por subtração — a ablação acima serve para confirmar o total por camada.

---

## 6. O que NÃO é matvec: o custo que o lote não amortizou

Tudo abaixo é **por token** (não por chunk): o lote de 16 amortiza a leitura do peso e a
quantização de ativação, mas essas fases rodam uma vez por token dentro do chunk, com o número
de lançamentos já dividido por 16. Números de `--prefill 64 --level 1` / `--level 2`, com o
matvec removido:

| fase | ms/token | % da gordura (1,461) | por que não amortiza |
|---|---|---|---|
| `gdn_delta` | **0,641** | 43,9 % | a recorrência: `delta_rule_batch_rows_kernel` (`gdn.cuh:429-444`) tem o laço de tokens **dentro** do kernel, uma passada sequencial por token |
| `act_quant` | **0,227** | 15,5 % | 190 `quantize_q8_1_batch` por chunk = 11,9 por token, cada uma sobre `n×ncols` elementos |
| `ffn_residual` | **0,095** | 6,5 % | `add` sobre n×5120 por camada |
| `post_norm` | **0,076** | 5,2 % | `add` + `rms_norm` sobre n×5120 por camada |
| `gdn_l2norm` | **0,073** | 5,0 % | 2 `l2_norm` por camada GDN |
| `gdn_norm_silu` | **0,066** | 4,5 % | `rms_norm` + `silu`·`mul` |
| `gdn_scalars` | **0,053** | 3,6 % | `sigmoid`, `softplus`, `mul` sobre 48 elementos por token |
| `gdn_conv` | **0,050** | 3,4 % | `conv1d_state_batch` |
| `qk_norm_rope_kv` | **0,044** | 3,0 % | 2 `rms_norm`, 2 `rope`, escrita do KV |
| `attention` | **0,032** | 2,2 % | 1 kernel por camada, contexto cresce |
| `attn_gate_out` | **0,012** | 0,8 % | `sigmoid` + `mul` |
| **soma desta tabela** | **1,369** | 93,7 % | soma dos buckets de nível 1 a N=64 (inclui 0,527 de andaime de projeção) |
| **total não-matvec medido a N=128** | **1,141** | **11,9 %** | 8,244 (limpa) − 7,103 (matvec) |
| **parte fixa por CHUNK** (embeddings, head, norms, cópia do logits) | 0,32 (5,1 ms/chunk) | 3,3 % | medido: intercepto de 0,163 s no §5.1 |
| **toda a gordura** (matvec excluído) | **1,461** | **15,3 %** | 1,141 + 0,32; é este o denominador da coluna % acima |

Decomposição em dois pedaços, com o matvec medido dentro do grafo:

| N | prefill total (limpa) | matvec | % | resto | % |
|---|---|---|---|---|---|
| 64 | 8,638 ms/token | 7,362 | 85,2 % | 1,276 | 14,8 % |
| 128 | 8,244 ms/token | 7,103 | 86,2 % | 1,141 | 13,8 % |
| 512 | 8,104 ms/token (count-only) / 8,221 (nível 1) | *não medido a 512* | — | — | — |

O nível 2 a 512 **não é executável**: 460 800 `hipEvent_t` e o harness recusa
(`"(passada instrumentada pulada: N=512 cria 460800 hipEvent_t; use N<=128)"`,
`bench_phases_gpu.hip:741-744`). A atribuição a 512 usa o matvec medido a 128
(7,103 ms/token) — o chunk é o mesmo (16) nos dois N, e é isso que autoriza a transferência.

**Os três maiores custos por token que fusão ou batching poderiam remover:**

1. **`gdn_delta` — 0,641 ms/token (0,641 ms/token = 43,9 % da gordura; 7,9 % do prefill)**: 1 kernel por camada com o laço de tokens dentro
   (48 por chunk), lendo e escrevendo 3,1 MB de estado por camada. Já é batido; é sequencial
   por construção. Fusão não ajuda, um scan paralelo ajudaria.
2. **`act_quant` — 0,227 ms/token (0,227 ms/token = 15,5 % da gordura; 2,8 % do prefill)**: 190 lançamentos por chunk, cada um re-quantizando
   blocos de 32. Candidato direto a fusão **dentro do matvec** (é o que o MMQ faz: quantiza a
   ativação no kernel do GEMM) ou a cache por (buffer, geração) — a mesma ideia do `proj_qq` do
   caminho por token, que já rendeu +1,3 % no decode (`docs/medicoes-m8.md`).
3. **`qk_norm_rope_kv` + `attention` + `attn_gate_out` — 0,088 ms/token (6,0 % da gordura; 1,1 % do prefill)**, exatamente o
   que as fusões `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS` e `SIGMOID_MUL` do Vulkan tocariam.

Somados: **0,956 ms/token**, que são **9,98 %** do prefill medido a 512 (0,956/9,575 — o
denominador inclui toda a gordura, 1,461 ms/token) ou **11,8 %** sobre os 8,104 ms/token do
prefill inteiro. O resto é matvec, e matvec não é andaime: é álgebra.

---

## 7. Atribuição do gap

### 7.1 O gap de 9,0× do briefing é de micro-lote, e está medido

| configuração | tok/s | ms/token | razão contra nós |
|---|---|---|---|
| nós, `--prefill 512` (melhor de 3) | 122,66 | 8,15 | 1,00× |
| nós, `bench-phases-gpu --prefill 512` (count-only) | 123,39 | 8,10 | 0,99× |
| llama.cpp Vulkan, `-ub 16 -b 16` | **200,25** | 4,99 | **1,62×** |
| llama.cpp Vulkan, default (`b 2048`, `ub 512`) | **1168,49** | 0,856 | 9,53× |
| llama.cpp Vulkan, corrida do briefing | 1113,96 | 0,898 | 9,08× |

Mesmo binário, mesma máquina, mesmo modelo, mesma sessão de `llama-bench`: o que muda entre a
linha 3 e a 4 é `-b/-ub`. **O llama.cpp perde 5,84× ao processar 16 tokens por vez; nós não
temos como perder, porque 16 é o nosso teto.** A pergunta certa não é "por que 9×", é
**"por que 1,62×"** — e a resposta é o kernel do matvec, com o tamanho de chunk como
multiplicador de tudo.

### 7.2 Atribuição do gap de 9,0× (por token)

Referência: llama.cpp 1113,96 tok/s = **0,898 ms/token**. Nosso: 8,104 ms/token (medido,
`--count-only --prefill 512`).

| componente | nosso ms | llama.cpp ms (implícito) | delta | origem |
|---|---|---|---|---|
| **projeções/matvec (305 lançamentos por chunk)** | **7,103** | ≤ 0,898 | ≥ **+6,21** | nosso: *medido* (nível 2, N=128). llama: *derivado* — 0,898 ms/token é o teto do total dele, então a projeção dele é ≤ isso |
| projeção do LM head (1 GEMV de `output.weight` por chunk, `graph.cuh:1416`) | **0,056** (*derivado*: intercepto de 0,163 s do §5.1 = 32 chunks × 5,1 ms de embeddings+head+norm+cópia, dos quais ~1,79 ms/chunk são o head) | ≤ 0,898 (contido na linha acima) | — | nosso: *derivado*. llama: *desconhecido* |
| `act_quant` (190 lançamentos por chunk) | 0,227 | *desconhecido* | — | nosso: *medido* (nível 2) |
| `gdn_delta` | 0,641 | *desconhecido* | — | nosso: *medido* |
| resto do GDN (`gdn_conv`, `gdn_l2norm`, `gdn_scalars`, `gdn_norm_silu`) | 0,242 | *desconhecido* | — | nosso: *medido* |
| atenção (16 camadas, `splits = 1` a 512) | 0,032 | *desconhecido* | — | nosso: *medido* |
| `qk_norm_rope_kv` + `attn_gate_out` | 0,056 | *desconhecido* | — | nosso: *medido* |
| norms/resíduos comuns (`post_norm`, `ffn_residual`) | 0,240 | *desconhecido* | — | nosso: *medido* |
| despacho de 916 lançamentos/chunk | 0,129 (por token) | *desconhecido*; teto do ganho de *qualquer* fusão ≤ 0,129 | ≤ 0,129 | nosso: *medido* (piso de kernel vazio 2,253 µs × contagem) |
| **soma instrumentada** | **8,795** (instrumentada) / **8,104** (limpa) | 0,898 | **+7,21** | |

**Marcação das células, em uma frase:** *medido* no nosso lado quase tudo (o prefill inteiro, o
matvec, cada bucket, o despacho); *derivado* no nosso lado só a projeção do LM head; **do lado
do llama.cpp só existe uma célula, e ela é derivada**: 0,898 ms/token é o total dele, e não há
como separar projeção de andaime sem profiler (`rocprof`/`omniperf` não estão instalados,
`docs/noite-regras.md` §3.5). As células "llama.cpp" que ficam como *desconhecido* são a maior
lacuna deste estudo, e estão listadas em §8.

### 7.3 O gap de 1,62× (o único que sobra), atribuído

| termo | nosso | llama.cpp (mesmo micro-lote) | delta | origem |
|---|---|---|---|---|
| matvec/projeções | 7,103 ms/token (*medido*) | ≤ 4,994 ms/token (*derivado*) | ≥ +2,11 | o teto do total dele é 4,994; a projeção dele é parte disso |
| todo o resto (andaime) | 1,141 ms/token (*medido*: 8,244 − 7,103) | ≥ 0 (*desconhecido*) | ≤ +1,14 | não há como medir o andaime do Vulkan |
| **total** | **8,244** | **4,994** | **+3,25** | |

**O que sobra depois de tudo:** o matvec explica **65 % a 100 %** do gap de 1,62×, dependendo
de quanto do total do llama.cpp é andaime — e a fronteira não é decidível com as ferramentas
instaladas. O que **é** decidível, e está medido: (a) o gap de 9,0× é 89 % tamanho de
micro-lote; (b) **nenhuma fusão de despacho pode valer mais de 1,6 %** do nosso prefill; (c) o
matvec em lote usa **16 % da banda** desta placa (11,133 GB / 113,6 ms = 98 GB/s contra 633
GB/s de roofline medido) enquanto o mesmo kernel no caminho por token usa 79-82 %.

### 7.4 Onde está o teto, em aritmética sobre medidas (*derivado*)

- 305 matvecs por chunk de 16 leem **11,133 GB** (o próprio `bench-phases-gpu` imprime
  "lidos por token 11.133 GB"). São 11,133 GB **por chunk**, não por token.
- A 98,0 GB/s efetivos (medido dentro do grafo) = 113,6 ms/chunk. A 633 GB/s (roofline medido)
  = **17,6 ms/chunk**.
- **Teto do prefill com o kernel atual**: o prefill medido é `512 × 8,104 ms = 4,149 s`; a
  soma do que foi medido por fase a N=128 é `512 × 8,244 = 4,221 s` (a do N=128 é 1,8 % maior
  porque aquela janela estava pior). **O motor roda a 1-2 % da soma das próprias fases**: não
  há gordura de andaime a colher entre as fases; o que existe é o kernel.
- **Teto com o kernel na banda** (*derivado*, aritmética explícita): matvec a 633 GB/s =
  11,133 GB / 0,633 = **17,6 ms/chunk**; andaime = `1,141 ms/token × 16 = 18,3 ms/chunk` (medido
  a N=128); total **35,9 ms/chunk = 446 tok/s a 512 tokens**. Contra os **200,25 tok/s** que o
  llama.cpp faz no mesmo micro-lote, o caminho tiled/int8/WMMA tem **2,2×** de espaço; contra os
  **123,39 tok/s** nossos, **3,6×**.

---

## 8. O que não foi medido, e por quê

1. **O orçamento interno do lado do llama.cpp é desconhecido.** `rocprof`/`rocprofv3`/
   `omniperf` não estão instalados (só `rocminfo`) e o backend Vulkan não expõe eventos HIP.
   As colunas "llama.cpp" de §7.2/§7.3 são o **total medido** (200,25 ou 1168,49 tok/s) e nada
   além dele. Separar o GEMM do andaime lá exige profiler de GPU ou instrumentar o llama.cpp —
   nenhum dos dois foi feito.
2. **O custo marginal de `matvec_kernel_batch` em N > 16 não foi medido.** O teto é de
   compilação (`matvec.cuh:1011`, confirmado por `--batch 32` devolver pass_ms = 0,005). A
   tentativa com um binário scratch (`/tmp/scratch/mb_batch_n*.hip`) falhou três vezes por
   geometria do tensor sintético (ver §3.3) e o tempo da frente acabou. **O experimento que
   fecha**: instanciar o mesmo kernel para N = 24/32/48/64 num alvo novo appendado no
   `CMakeLists.txt` — ~30 linhas, sem tocar no motor.
3. **Nível 2 a 512 tokens não é executável** (460 800 `hipEvent_t`, o harness recusa). O matvec
   a 512 é o medido a 128, transferido porque o chunk é o mesmo.
4. **Não há separação ablada de atenção × GDN dentro do `forward_batch`**: o modelo não tem
   flag para pular as 48 camadas recorrentes e `--layers N` corta as N primeiras (mistura os
   dois tipos). A separação de §5 é por bucket + contagem de camadas — medida, mas derivada da
   contagem, não ablada por subtração.
5. **Uma corrida de fase a 512 morreu** com *page fault* por VRAM de outro processo drenando
   (a primeira tentativa de `--prefill 512 --level 1`); o número de 512 usado no texto é de
   duas corridas novas (`phases-512.log`, 8,221 ms/token) e do `--count-only` (8,077), que
   concordam em 1,8 %.
6. **A janela não esteve limpa em todas as corridas.** Os números estão anotados com
   `VRAM_before`/`VRAM_after` nos logs de `/tmp/med-prefill/`. As corridas com a fila do
   `flock` cheia (7 waiters) deram de 3 % a 7 % pior e estão identificadas no texto (nível 2 a
   N=64; a passada 1 de cada N do `bench`).
7. **`sizeof(block_iq3_s) = 110` B** (medido por `/tmp/scratch/sz.cpp`), enquanto
   `docs/quants-inventario.md` implica 98 B para os blocos iq3 (o tensor `blk.0.ffn_down.weight`
   é iq3_xxs de 98 B, e é *outro* tipo). **Não é contradição do motor** — é uma armadilha para
   quem for escrever o microbench de N > 16: escolha o tensor e o `sizeof` do tipo dele, não um
   `sizeof` genérico.
