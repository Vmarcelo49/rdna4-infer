# Atenção e KV no prefill (medido)

Todas as corridas em **2026-09-14, 09:34-10:02**, sob `scripts/gpu-lock.sh` com o `timeout` **dentro**
do lock, modelo `/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf`
(64 camadas, 16 de atenção + 48 GDN), KV `f16/f16` salvo onde dito. **12 aquisições do lock**, todas
sem espera (o wall time de cada corrida bate com o trabalho de GPU: o N=4096 gastou 132 s de parede
para 4,4 s de load + 2×40,8 s de passada limpa + 41,9 s de instrumentada), `VRAM_before` em
**236-248 MiB em todas** (nada na placa), nenhum OOM. Logs brutos em `/tmp/med-atencao/log*.txt`,
scratch em `/tmp/attn-scratch/`.

Nota de integridade: às 09:33, antes da primeira aquisição, eu rodei `./build/bench-attn-gpu --help`
**fora do lock** — a placa estava ociosa (236 MiB), a chamada durou ~1 s e **nenhum número dela é
usado neste documento** (todas as curvas do `bench-attn-gpu` da §2.3 e da §3.1 saíram de dentro do
lock, em `/tmp/med-atencao/log1-attn-isolado.txt`).

Duas notas de método que valem para tudo abaixo:

1. **O teto de N do `bench-phases-gpu` não é do nível 2 — é de qualquer nível.** O briefing (e a §C)
   dizem "`--level 2` é limitado a N≤128 por causa dos eventos"; o código diz outra coisa
   (`tests/bench_phases_gpu.hip:741`): *"passada instrumentada pulada: N=%d cria %d hipEvent_t; use
   N<=128"*, sem consultar o nível. Para medir 512/2048/4096 no nível 1 eu compilei uma **cópia** do
   `.hip` em `/tmp/attn-scratch/` com **só esse teto trocado** (por um teto de eventos) e linkei
   contra os `.o` do backend do próprio `build/`; nenhum arquivo do repositório foi tocado. A cópia
   foi validada contra o binário oficial em N=64 e N=128 (o `attention` dá 0,032 contra 0,033 em
   N=64 e 0,048 contra 0,044 em N=128 — dentro do ruído de ±10 % do bucket; a passada limpa dá
   9,381 contra 9,465 e 9,452 contra 9,441). A 4096 no nível 1 são 42,0 marcas/token ⇒ 172 032
   eventos × 5,3 µs = 0,9 s de custo de evento num total de 40,8 s = **+2,2 %**, e o custo por bucket
   é impresso e subtraído.
2. **Os meus ms/token absolutos estão 13-18 % acima dos da frente C** (mesmos comandos, mesmos
   binários de 07:55): ela mediu 8,02 (N=64) e 8,22 (512), eu meço 9,38-9,49 e 9,51. **Não é o
   harness**: `rdna4-infer bench` e `bench-phases-gpu` concordam *dentro da minha janela*
   (104,65 contra 105,11 tok/s a 512; 100,56 contra 100,38 a 4096). É estado da máquina. Todas as
   porcentagens e razões deste documento são da janela de hoje; o **crescimento** de 64→4096 foi
   confirmado em três janelas (§2).

---

## TL;DR

1. **O bloco de atenção é 8,7 % do prefill a 512, 10,5 % a 2048 e 12,9 % a 4096** (16 camadas:
   `qkv_proj`+`qk_norm_rope_kv`+`attention`+`attn_gate_out`+`attn_out_proj`); **o kernel de atenção
   sozinho é 1,1 % / 3,2 % / 5,6 %**.
2. **O crescimento de +6,2 % por token de 64 → 4096 é o kernel de atenção**: +0,521 dos +0,581
   ms/token (**89,7 %**). As projeções de atenção (+0,068) e a escrita do KV (`qk_norm_rope_kv`,
   −0,003) estão **dentro do ruído** (±0,05 ms/token).
3. **Quantizar o KV é neutro para o prefill** (105,11 tok/s f16, 105,01 q8_0, 104,79 q4_0 a 512;
   102,59 contra 102,43 a 2048) e **piora o kernel** (+24 % com q4_0 a 2048). O par
   **K=`q5_0`/V=`q4_1` — o "ponto doce" documentado — quebra a GPU** (page fault) no prefill.
4. **A atenção do llama.cpp é 0,34 % do chunk a 512 e 1,06 % a 2048** (nó `FLASH_ATTN_EXT`), e o
   bloco inteiro deles dá **7,6 % e 7,7 %** — *a mesma fatia que a nossa*. A diferença é absoluta:
   **11-14× no bloco e 30-34× no kernel**, e o mecanismo é medido (1536× mais tráfego de KV).
5. **Atenção não é alavanca de prefill**: o kernel custa **1,13 % do prefill a 512 e 5,55 % a 4096**
   (zerá-lo daria +1,2 % e +5,9 % de velocidade), contra **83-88 %** dos buckets de projeção (matvec)
   e **10,3 %** da recorrência GDN.

---

## 1. A fatia da atenção no prefill, por tamanho de prompt

### 1.1 Como separei as 16 camadas de atenção das 48 GDN (e o que o `--gdn` faz)

O profiler de nível 1 tem 16 buckets e **nenhum bucket é "atenção total"**: o que separa os dois
caminhos é a **contagem de marcas por token**, que o próprio tool imprime e que fecha exatamente:

| grupo | buckets | marcas/token medidas | aritmética |
|---|---|---|---|
| **atenção (16 camadas)** | `qkv_proj`, `qk_norm_rope_kv`, `attention`, `attn_gate_out`, `attn_out_proj` | 1,0 × 5 = **5,0** | 5 buckets × 16 camadas / 16 tokens |
| **GDN (48 camadas)** | `gdn_proj`, `gdn_scalars`, `gdn_conv`, `gdn_l2norm`, `gdn_delta`, `gdn_norm_silu`, `gdn_out_proj` | 3,0 × 7 = **21,0** | 7 buckets × 48 camadas / 16 tokens |
| **comum (64 camadas)** | `post_norm`, `ffn_gate_up`, `ffn_down`, `ffn_residual` | 4,0 × 4 = **16,0** | 4 buckets × 64 camadas / 16 tokens |
| | | **42,0** | impresso: `42.0 marks/token` em todas as corridas |

Ou seja: o número de camadas cobertas por cada bucket é **medido** (pela contagem de marcas), não
suposto. As marcas estão em `include/rdna4/graph.cuh:1048` (qkv/gdn_proj), `:1062`, `:1081`, `:1136`,
`:1142` (atenção) e `:1220-1261` (GDN).

**O `--gdn` NÃO separa camadas no perfil.** Ele roda uma **bancada isolada de kernels GDN fora do
grafo** (comentário no próprio tool: *"GDN isolado (fora do grafo)"*), com as formas reais
(`nvh 48, nkh 16, S 128`):

```
./scripts/gpu-lock.sh timeout 900 ./build/bench-phases-gpu $M --gdn --prefill 16 --level 1 --prefill-reps 1
```
imprime, por lançamento: `delta_rule(48 cabeças, S 128) 8,45 µs`, `conv1d(4 taps, 10240 canais)
3,53 µs`, `kv_store_row(f16,256) 2,81 µs`, `kv_store_row(q4_0,256) 3,15 µs`, `rms_norm(48×128)
3,39 µs`, `rope(1×24×256) 3,22 µs`, `deinterleave_q_gate(24×256) 2,83 µs`, e o piso de despacho
2,19 µs. Serve para preço de kernel isolado; **não** muda a atribuição por bucket.

### 1.2 Os números (f16/f16, nível 1, chunks de 16)

Comando de cada linha (os de 512/2048/4096 usam a cópia em `/tmp` descrita acima; o de 64/128 usa o
binário oficial — os dois concordam):

```
./scripts/gpu-lock.sh timeout 900 ./build/bench-phases-gpu $M --prefill {64,128} --level 1 --prefill-reps 3
./scripts/gpu-lock.sh timeout 900 /tmp/attn-scratch/bench-phases-attn-gpu $M --prefill {512,2048,4096} --level 1 --prefill-reps {2,3}
```

| N | passada limpa (ms/token) | tok/s | splits | **`attention`** | `qkv_proj` | `qk_norm_rope_kv` | `attn_gate_out` | `attn_out_proj` | **bloco** | **% kernel** | **% bloco** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **64** | 9,381-9,486 (4 medidas) | 105,4-106,6 | 1 | **0,032-0,033** | 0,373-0,453 | 0,033-0,067 | 0,012-0,025 | 0,230-0,275 | **0,694-0,805** | **0,34-0,35 %** | **7,4-8,5 %** |
| 128 | 9,441 | 105,92 | 1 | 0,044 | 0,370 | 0,043 | 0,012 | 0,264 | 0,733 | 0,47 % | 7,8 % |
| **512** | 9,495-9,542 (4 medidas) | 104,8-105,3 | 1 | **0,108-0,118** | 0,390-0,429 | 0,037-0,045 | 0,014-0,019 | 0,250-0,286 | **0,822-0,852** | **1,13-1,24 %** | **8,65-8,97 %** |
| **2048** | 9,748 | 102,59 | 4 | **0,309** | 0,406 | 0,042 | 0,016 | 0,254 | **1,027** | **3,17 %** | **10,54 %** |
| **4096** | 9,962 | 100,38 | 8 | **0,553** | 0,408 | 0,044 | 0,015 | 0,263 | **1,283** | **5,55 %** | **12,88 %** |

- "% kernel" e "% bloco" são sobre a **passada limpa** do próprio N (o profiler infla o total em
  +1,9 % a +3,4 %; a coluna `%` da tabela impressa é sobre a soma dos buckets, não sobre o prefill).
- **Correção de evento**: cada marca custa 5,1-5,7 µs e cai no bucket que a possui — é preciso
  subtrair **0,005 ms/token** do `attention` (a 64 isso é 16 % do bucket: 0,033 → 0,028) e
  0,005 ms/token de cada projeção. A 4096 a correção é 1 % do kernel e irrelevante para as
  conclusões.
- **Ruído por bucket (nível 1, uma passada instrumentada)**: 4 medidas a 512 dão
  `qkv_proj` 0,390-0,429 (**±10 %**), `attn_out_proj` 0,250-0,286 (±14 %), `qk_norm_rope_kv`
  0,037-0,045 (±22 %), `attn_gate_out` 0,014-0,019 (±36 %), `attention` 0,108-0,118 (±9 %);
  a 64 o `attention` deu 0,032/0,032/0,033/**0,055** (um outlier de 1,7×). A passada **limpa** é
  estável: 9,495-9,542 a 512 (**±0,25 %**) e 9,381-9,486 a 64 (±0,55 %). **Um único número de
  bucket pequeno não vale nada; a curva de 64 → 4096 vale porque é 17× maior que o ruído.**

---

## 2. O crescimento de 64 → 4096 é o kernel de atenção, e mais nada

### 2.1 O crescimento, em três janelas independentes

| janela | N=64 | N=4096 | Δ |
|---|---|---|---|
| frente C, `rdna4-infer bench` (`docs/estudo-prefill-c-nosso.md` §1.1) | 8,02 ms/token | 8,57 | **+6,9 %** |
| hoje, `rdna4-infer bench --prefill N --prefill-reps 3 --ctx-size N+64` | 9,434 (106,00 tok/s) | 9,944 (100,56) | **+5,4 %** |
| hoje, `bench-phases-gpu --prefill N --level 1` (melhor das passadas limpas) | 9,381 | 9,962 | **+6,2 %** |

O crescimento é de **+0,51 a +0,58 ms/token** nas duas janelas de hoje e **+0,55** na da frente C.

### 2.2 Decomposição por bucket (nível 1, melhor passada; 64 = cópia scratch, 4096 = cópia scratch)

| bucket | N=64 | N=4096 | **Δ** | % do Δ total | classe |
|---|---|---|---|---|---|
| **`attention`** | 0,032 | **0,553** | **+0,521** | **89,7 %** | kernel |
| `qkv_proj` | 0,373 | 0,408 | +0,035 | 6,0 % | projeção |
| `attn_out_proj` | 0,230 | 0,263 | +0,033 | 5,7 % | projeção |
| `attn_gate_out` | 0,012 | 0,015 | +0,003 | 0,5 % | elementwise |
| `qk_norm_rope_kv` | 0,047 | 0,044 | **−0,003** | −0,5 % | **escrita do KV** |
| `gdn_delta` | 0,715 | 0,741 | +0,026 | 4,5 % | recorrência |
| `gdn_proj` | 1,546 | 1,587 | +0,041 | 7,1 % | projeção |
| `gdn_out_proj` | 0,708 | 0,737 | +0,029 | 5,0 % | projeção |
| `ffn_gate_up` | 3,328 | 3,275 | **−0,053** | −9,1 % | projeção |
| `ffn_down` | 2,070 | 1,989 | **−0,081** | −13,9 % | projeção |
| `ffn_residual` | 0,208 | 0,245 | +0,037 | 6,4 % | elementwise |
| `post_norm` | 0,108 | 0,098 | −0,010 | −1,7 % | norm |
| `gdn_conv`/`gdn_l2norm`/`gdn_norm_silu`/`gdn_scalars` | 0,274 | 0,283 | +0,009 | 1,5 % | GDN |
| **soma** | 9,381 | 9,962 | **+0,581** | 100 % | (fecha com a passada limpa) |

Resposta direta à pergunta:

- **kernel de atenção: +0,521 ms/token (89,7 % do crescimento)**;
- **projeções de atenção (`qkv_proj` + `attn_out_proj` + `attn_gate_out`): +0,071 ms/token**, que é
  **menor que o próprio ruído de ±0,08** medido para esses buckets a 64 — **plano**;
- **escrita do KV (`qk_norm_rope_kv`, que é 2 `rms_norm` + 2 `rope` + `kv_write_batch`): −0,003
  ms/token — plana**, e o motivo é aritmético: o chunk escreve 16 tokens × 4 cabeças KV × 256 dim ×
  2 B × 2 (K e V) × 16 camadas = **1,0 MB por chunk = 64 KB/token**, independente do contexto;
- **todo o resto: −0,002 ms/token somado** (as projeções de GDN e FFN sobem e descem dentro do
  ruído). *Não há um "terceiro" componente escondido: o crescimento é o kernel de atenção.*

### 2.3 O que o `bench-attn-gpu` isolado diz — e por que ele **não** serve para o prefill

`./scripts/gpu-lock.sh timeout 900 ./build/bench-attn-gpu f16 64 256 512 1024 2048 4096`
(política de ship; **ms por camada**, para **1 token de consulta**; converter em ms/token = ×16):

| t (chaves) | unsplit | ship (splits×wpb) | ×16 = ms/token (INFERIDO) | medido no prefill | razão medido/isolado |
|---|---|---|---|---|---|
| 64 | 0,014 | 0,014 (sem split) | 0,224 | **0,032** | **0,14×** |
| 512 | 0,048 | 0,048 (sem split) | 0,768 | **0,111** | 0,14× |
| 1024 | 0,089 | 0,033 (2×16) | 0,528 | — | — |
| 2048 | 0,168 | 0,041 (4×16) | 0,654 | **0,309** | 0,47× |
| 4096 | 0,330 | 0,071 (8×8) | 1,131 | **0,553** | 0,49× |

**O bench isolado superestima o kernel do prefill em 2-7×** e a razão muda com o contexto. O motivo
é estrutural, não um erro do bench: ele mede a forma do **decode** — grade de **24 CTAs** (uma por
cabeça), 1 token de consulta — enquanto o prefill lança `attn_batch_kernel` com grade
**(24, n_tokens) = 384 CTAs** (`include/rdna4/attn.cuh:320,416`), o que enche a placa e faz as 16
consultas de um chunk concorrerem pelas mesmas linhas de KV no L2 (**INFERIDO**; o que está medido é
o fator 2-7×). **Para o kernel do prefill vale o bucket
do profiler; o isolado serve de teto, nunca de previsão.** (Ainda assim o `unsplit` isolado a 4096,
0,330 × 16 = 5,28 ms/token, mostra o tamanho do absurdo que é o kernel sem split.)

---

## 3. Formato do KV no prefill: neutro para o total, pior para o kernel, e um par que quebra

### 3.1 Total do prefill e bucket da atenção

```
./scripts/gpu-lock.sh timeout 900 /tmp/attn-scratch/bench-phases-attn-gpu $M --prefill N --level 1 \
  --prefill-reps 3 --kv-k K --kv-v V        # (o oficial com --prefill 64/128 dá o mesmo)
```

| N=512 | `f16/f16` | `q8_0/q8_0` | `q4_0/q4_0` | `q5_0/q4_1` |
|---|---|---|---|---|
| passada limpa (ms/token) | 9,514 | 9,523 | 9,543 | **FALHA (page fault)** |
| tok/s | **105,11** | 105,01 | 104,79 | — |
| `attention` (ms/token) | 0,118 | 0,117 | **0,134** | — |

| N=2048 | `f16/f16` | `q4_0/q4_0` |
|---|---|---|
| passada limpa (ms/token) | 9,748 | 9,762 |
| tok/s | **102,59** | 102,43 |
| `attention` (ms/token) | **0,309** | **0,383** (+24 %) |

**Resposta: quantizar o KV não ajuda o prefill — é neutro no total (±0,3 %, dentro do ruído
entre corridas, que é 0,5 %) e *piora* o kernel de atenção (+14 % com `q4_0` a 512, +24 % a 2048).**
O mesmo padrão aparece no kernel isolado (`bench-attn-gpu`, ms/camada, política de ship):

| t | f16 | q8_0 | q4_0 | q5_0 |
|---|---|---|---|---|
| 64 | 0,014 | 0,012 | 0,012 | 0,013 |
| 256 | 0,028 | 0,028 | 0,028 | 0,028 |
| 512 | 0,048 | 0,048 | 0,049 | *fault* |
| 1024 | 0,089 (split 0,033) | 0,090 (0,036) | 0,092 (0,038) | *fault* |
| 2048 | 0,168 (0,041) | 0,172 (0,048) | 0,178 (0,052) | — |
| 4096 | **0,330 (0,071)** | 0,340 (0,086) | **0,351 (0,092)** | — |

### 3.2 Por que (a conta fecha com o diagnóstico da noite)

Fórmula do repo (`docs/medicoes-banda-e-gargalos.md` §3): **único = chaves × 4 cabeças KV × linha ×
2 (K e V) × 16 camadas**, **emitido = 6× isso** (as 6 cabeças de consulta de cada grupo KV releem a
mesma linha, porque a nossa grade é `(n_head, n_tok)` e não agrupa consultas).

| N | emitido/token (média) | medido do kernel | **taxa emitida** | % do IC (1.500 GB/s) |
|---|---|---|---|---|
| 512 | 100,8 MB | 0,111 ms | **908 GB/s** | 61 % |
| 2048 | 402,8 MB | 0,309 ms | **1.304 GB/s** | 87 % |
| 4096 | 805,5 MB | 0,553 ms | **1.457 GB/s** | **93-97 %** |

Duas leituras, e a segunda é a resposta à pergunta 3:

1. A partir de ~2048 o kernel **f16** está no **muro do Infinity Cache** (1.500 GB/s medido em
   `docs/medicoes-banda-e-gargalos.md:307`; 1.430-1.561 GB/s em `docs/journal-longctx.md:250`).
   Isso **reproduz, no prefill, o achado que a noite fez no decode** ("a 4K e 16K a atenção roda a
   1,2-1,3 TB/s, 78-88 % do IC").
2. Com `q4_0` a 2048 o kernel emite **296 GB/s** (mesma fórmula, linha de 144 B por K/V) — 4,4× menos
   eficiente por byte e 24 % mais lento apesar de mover **3,6× menos bytes**. É exatamente o regime
   "**limitado por desquantização/issue, não por banda**" que a noite mediu a 64K (`346 GB/s
   emitidos`). Ou seja: **o prefill não muda o veredito do decode sobre quantização de KV — o
   mecanismo é o mesmo, e o número que decide é a taxa emitida (296 contra 1.304 GB/s).**

### 3.3 O par K=`q5_0`/V=`q4_1` (o ponto doce documentado) quebra a GPU no prefill

`Memory access fault by GPU node-1 (...) Reason: Page not present or supervisor privilege.`
Matriz completa, medida com o **binário oficial** (`--prefill 64 --level 1`) e com o **caminho de
produção** (`rdna4-infer bench --prefill 512`):

| K \ V | f16 | q4_1 | q4_0 | q5_0 | q8_0 |
|---|---|---|---|---|---|
| **f16** | OK | **FALHA** | — | — | — |
| **q5_0** | OK | **FALHA** | — | OK | — |
| **q8_0** | — | OK (99,65 tok/s, a 64 com 1 passada) | — | — | — |
| **q4_0** | — | OK | OK | — | — |
| **q4_1** | OK | OK | — | OK | — |

- O conjunto que falha é **K ∈ {f16, q5_0} com V = q4_1**; `q4_0/q4_1`, `q8_0/q4_1`, `q4_1/q4_1`,
  `q5_0/q5_0`, `q5_0/f16`, `q4_1/q5_0` **passam**.
- **Não é do meu scratch**: falha igual no binário oficial e no `rdna4-infer` de produção
  (`build/rdna4-infer`, 07:55).
- **Não é do caminho por token**: decode com `q5_0/q4_1` **funciona** (23,10 tok/s, `-n 8`); a falha
  aparece já com **um único chunk de 16** (`--prefill 16`) e também a 64/512. É o **caminho em lote**.
- O `(Q5_0,Q4_1)` **está** instanciado na atenção em lote (`include/rdna4/attn.cuh:465`), então a
  hipótese de `switch` incompleto está descartada por leitura; o suspeito passa a ser o caminho de
  **escrita/quantização do KV em lote** com K e V de tamanhos de linha diferentes (K `f16` = 512 B,
  K `q5_0` = 176 B, V `q4_1` = 160 B) — a mesma classe do bug de 128 MiB que a noite corrigiu.
  **INFERIDO**; a confirmação é leitura de código (frente irmã) ou um `compute-sanitizer`/`hip-memcheck`.
- `bench-attn-gpu q5_0` também **quebra** (mesma mensagem) a partir de t=256 (`HG=3 rel-L2 nan`, depois
  fault) — dois tools independentes apontam para o mesmo caminho de `q5_0`.

**Isto é um bug, não um ajuste de tuning, e cai justamente na configuração recomendada pela noite
(K=`q5_0`, V=`q4_1`).** Se o motor for para produção com KV quantizado, esse par precisa de
correção ou de um veto explícito.

---

## 4. A atenção do llama.cpp, mesmo modelo, mesma janela (`GGML_VK_PERF_LOGGER=1`, binário limpo)

```
./scripts/gpu-lock.sh timeout 900 env GGML_VK_PERF_LOGGER=1 \
  /tmp/llama-clean/build/bin/llama-bench -m $M -p 512 -n 0 -r 1     # e -p 2048
```

Referências na mesma janela: `llama-bench` limpo, **sem** logger, `-r 3` = **1047,09 ± 1,11 tok/s**;
`-r 1` = 1039,71; **com o logger = 1017,07** (o logger custa ~2 % no host). `-p 2048` com logger =
**997,53 tok/s**.

O logger imprime uma tabela **por chunk de 512 tokens** (1 811 despachos cada). Tabela do chunk
cronometrado de `-p 512` (soma dos nós **498,13 ms** = 0,973 ms/token; o `pp512` de parede dá
489 ms/chunk, 2 % abaixo):

| nó | despachos | ms/chunk | **% do chunk** | µs/token |
|---|---|---|---|---|
| **`FLASH_ATTN_EXT`** | 16 | 1,683 | **0,34 %** | **3,29** |
| `ROPE_VIEW_SET_ROWS` (rope K + escrita no KV) | 16 | 1,084 | 0,22 % | 2,12 |
| `ROPE` (Q) | 16 | 0,574 | 0,12 % | 1,12 |
| `SET_ROWS` (K no cache) | 16 | 0,162 | 0,03 % | 0,32 |
| `CONT` (recorte q/gate) | 16 | 0,640 | 0,13 % | 1,25 |
| `SIGMOID_MUL` (portão de atenção) | 16 | 0,653 | 0,13 % | 1,28 |
| `RMS_NORM_MUL(256,24)` (q-norm) | 16 | 2,702 | 0,54 % | 5,28 |
| `RMS_NORM_MUL(256,4)` (k-norm) | 16 | 0,215 | 0,04 % | 0,42 |
| **subtotal "nós de atenção"** | | **7,714** | **1,55 %** | **15,07** |
| projeção q+gate (`m=12288 k=5120`) | 16 | 16,961 | 3,40 % | 33,13 |
| projeção k+v (`m=1024 k=5120`) | 32 | 3,885 | 0,78 % | 7,59 |
| projeção o (`m=5120 k=6144`, 25,0 % **DERIVADO**) | 16 | 9,343 | 1,88 % | 18,25 |
| **bloco de atenção (nós + projeções)** | | **37,902** | **7,61 %** | **74,03** |
| `CONCAT` (48 = nº de camadas **GDN** — atribuição ambígua) | 48 | 9,368 | 1,88 % | 18,30 |
| `CPY` (96, ambíguo: pelo menos a escrita de V) | 96 | 0,510 | 0,10 % | 1,00 |

Para `-p 2048` (quatro chunks cronometrados, soma 2 007,1 ms para 2048 tokens = 0,980 ms/token), por
chave crescente 512/1024/1536/2048:

| nó | 512 chaves | 1024 | 1536 | 2048 | agregado µs/token | % |
|---|---|---|---|---|---|---|
| `FLASH_ATTN_EXT` | 2,644 | 4,546 | 5,691 | 8,288 | **10,34** | **1,06 %** |
| nós de atenção (sem projeções) | 6,082 | 7,970 | 10,074 | 11,730 | 17,51 | 1,79 % |
| projeções de atenção (o = 25 % MAC) | 27,933 | 28,308 | 31,579 | 30,639 | 57,84 | 5,91 % |
| **bloco de atenção** | **34,014** | **36,278** | **41,653** | **42,369** | **75,35** | **7,69 %** |

Como as fatias foram atribuídas: `FLASH_ATTN_EXT`, `ROPE*`, `SET_ROWS`, `CONT`, `SIGMOID_MUL` e as
duas `RMS_NORM_MUL` de 16 despachos são **medidos** e só existem no caminho de atenção (16 camadas);
as projeções são **DERIVADO** por forma de tensor — `m=12288 k=5120` = `attn_q` (q ⊕ gate, 12 288 =
24×256×2) e `m=1024 k=5120` = `attn_k`+`attn_v`, ambos exclusivos da atenção; `m=5120 k=6144` mistura
`attn_output` (16) com `ssm_out` do GDN (48) e foi rateada por **MAC**: 503,3 M / (503,3 + 1 509,9) M
= **25,0 %** (formas lidas do GGUF com `python3 -c "from gguf import GGUFReader..."`; o tronco dá
**24,3525 G MACs/token**, e as projeções de atenção são **1,6777 G = 6,89 %** dele).
`CONCAT` e `CPY` ficam **fora** do bloco por não serem atribuíveis sem leitura de código:
`CONCAT` tem **48** despachos = exatamente o número de camadas GDN (concat da entrada do `ssm_conv`),
mas 48 também é 3×16; se forem todos de atenção, o bloco de 512 sobe para 9,59 %. (`CPY` vale 0,1 %;
não muda nada.)

**Ruído deles, medido**: o mesmo `FLASH_ATTN_EXT` com a mesma geometria (512 consultas × 512 chaves)
deu **3,29 µs/token** no `-p 512` e **5,16 µs/token** no 1º chunk do `-p 2048` — **+57 %**;
`ROPE_VIEW_SET_ROWS` deu 67,8 µs/dispatch na primeira corrida e 9,8-13,0 µs nas outras (primeiro
toque de página no cache de K, provavelmente). As projeções de atenção do lado deles são estáveis
(3,4 % do chunk em todas as tabelas não poluídas).

---

## 5. Comparação lado a lado — 512 e 2048

| grandeza (por token) | **nós** | **llama.cpp Vulkan** | razão | status |
|---|---|---|---|---|
| **512 tokens, kernel de atenção** | 0,111 ms (0,108-0,118) | **3,29 µs** (3,29-5,16) | **21,6-33,9×** | medido / medido |
| 512 tokens, **bloco** (kernel + normas + rope + portão + projeções) | 0,839 ms (0,822-0,852) | **74,0 µs** (66,4-74,0) | **11,3-12,6×** | nosso medido / deles **derivado** (projeções por MAC) |
| 512, % do prefill — kernel | **1,13-1,24 %** | **0,34-0,53 %** | 2,3-3,6× | medido / medido |
| 512, % do prefill — bloco | **8,65-8,97 %** | **7,61 %** (6,82-7,61) | 1,14-1,27× | medido / derivado |
| 512, prefill inteiro | **9,51 ms/token** (105,1 tok/s) | **0,955 ms/token** (1047,09 tok/s) | 9,96× | medido / medido |
| **2048 tokens, kernel** | 0,309 ms | **10,34 µs** | **29,9×** | medido / medido |
| **2048 tokens, bloco** | 1,027 ms | **75,35 µs** | **13,6×** | medido / derivado |
| 2048, % do prefill — kernel | **3,17 %** | **1,06 %** | 3,0× | medido / medido |
| 2048, % do prefill — bloco | **10,54 %** | **7,69 %** | 1,37× | medido / derivado |
| 2048, prefill inteiro | **9,75 ms/token** (102,59 tok/s) | **1,003 ms/token** (997,53 tok/s) | 9,72× | medido / medido |
| **KV emitido no prompt inteiro, 512** | **51,6 GB** (`Σ(i+1)×24 cabeças×1 024 B×16`) | **≥ 33,6 MB** (piso: o cache lido uma vez) | **≥ 1 536×** | nosso **derivado** da forma do kernel / deles **piso** |
| **KV emitido no prompt inteiro, 2048** | **825 GB** | **≥ 335 MB** (prefixo por chunk) | **≥ 2 460×** | idem |
| taxa emitida do kernel (nosso) | 908 GB/s (512) / 1 304 (2048) / 1 457 (4096) | *desconhecido* | — | nosso derivado; deles só se souber a redundância do workgroup |

**O que isto diz, sem adjetivo:**

- A **fatia** de atenção no prefill é quase a mesma nos dois (8,7-9,0 % contra 7,6 % a 512; 10,5 %
  contra 7,7 % a 2048). A atenção **não** é um problema desproporcional: o bloco inteiro é
  11-14× mais lento, praticamente o mesmo fator do prefill inteiro (9,7-10,0×).
- O que é desproporcional é **só o kernel: 30-34×**. E o mecanismo está medido: a nossa grade é
  `(24 cabeças, 16 tokens)` = 384 CTAs e cada uma relê o seu contexto inteiro ⇒ **1536× o piso de
  tráfego de KV** (51,6 GB contra 33,6 MB a 512). O `FLASH_ATTN_EXT` deles processa as 512 consultas
  de um chunk num único despacho por camada, com blocos de 64 consultas (`m(512,512,1,1)` é a máscara
  quadrada) — daí, com fator de reuso 4,5 (INFERIDO: 8 workgroups sobre 512 consultas), o tráfego
  deles seria de ~151 MB contra os nossos 51,6 GB, **342× menos bytes em 33× menos tempo**.
  **A confirmação dessa parte é leitura de código (frente irmã); o que é meu é o 51,6 GB e o 1536×.**
- A 4096 o nosso kernel f16 já está a **93-97 % do Infinity Cache**: mesmo com a instrução perfeita,
  ele não fica mais rápido sem **ler menos** — e é isso que o bloqueio de consultas faz.

---

## O que isto muda no plano

**A atenção não é alavanca do prefill. É um rodapé, e o número que decide é 5,55 %.**

| componente (a 4096, f16) | ms/token | % do prefill | onde está na fila |
|---|---|---|---|
| buckets de projeção (`ffn_*`, `gdn_proj`, `gdn_out_proj`, `qkv_proj`, `attn_out_proj`) | **8,259** | **82,9 %** | **P0/D2-D4 — é aqui que o gap mora** (frente C: matvec = 83,7 %) |
| recorrência GDN (`gdn_delta`+`conv`+`l2norm`+`norm_silu`+`scalars`) | 1,024 (frente C: 0,852) | 10,3 % | scan paralelo, e não é atenção |
| **bloco de atenção inteiro (16 camadas)** | **1,283** | **12,9 %** | rodapé — e 53,5 % dele são as projeções (0,408+0,263+0,015), que já estão no P0 |
| **só o kernel de atenção** | **0,553** | **5,55 %** | teto de qualquer trabalho em atenção para o prefill |

Consequências concretas, na ordem em que valem:

1. **Zerar o kernel de atenção** (impossível, mas é o teto) leva o prefill de 9,962 para 9,409
   ms/token em 4096: **+5,9 %** de velocidade. A 512, **+1,1-1,2 %**. Nada do que está em
   `docs/plano-prefill.md` (P0 tile+LDS, P1 matrix cores) fica atrás disso: o P0 medido vale
   **3,86-3,99×**. **A atenção continua sendo item de decode**, onde a 64K/131K ela já é 34-54 % do
   passo (`docs/journal-longctx.md:245`).
2. **O P3 ("atenção GQA por workgroup, +9-11,8 % em protótipo") vale, no prefill, no máximo 5,5 %**
   (4096 com o kernel zerado) e realistamente bem menos: o protótipo GQA do `bench-attn-gpu` a 4096
   custa **0,166 ms/camada contra 0,071-0,084 do kernel de ship** — **2,0-2,3× mais lento** na forma
   do prefill (16 consultas por chunk); ele é bom para *contexto longo com poucas consultas*, não
   para isto. **Não mover o P3 para antes do P0.**
3. **A política de splits no prefill não compensa trabalho novo**: a melhor célula do sweep a 4096
   é 0,0672 contra 0,0707 ms da política de ship = **1,052×**; aplicado ao kernel medido dá
   0,553/1,052 = 0,526, ou seja **0,027 ms/token = 0,27 % do prefill** (a diferença absoluta do bench
   isolado, 0,0035 ms/camada × 16 = 0,056 ms/token, é o limite superior dessa conta). É ajuste de
   tabela, no máximo.
4. **O KV quantizado não entra no plano de velocidade do prefill.** Ele é neutro no total (±0,3 %) e
   custa 14-24 % no kernel. A escolha tem de continuar sendo por **qualidade e VRAM**
   (`docs/journal-kv.md` §7: K `q8_0` 16 % melhor que `q5_0` por KL), nunca por tok/s.
5. **Há um bug de corretude a corrigir antes de qualquer coisa**: o par **K=`q5_0`/V=`q4_1`** (e
   K=`f16`/V=`q4_1`) dá **page fault** no prefill em lote — inclusive no `rdna4-infer` de produção —
   e é justamente o "ponto doce" que a noite recomendou. Isso é um gate, não um tuning: ou corrige,
   ou o par sai da lista permitida. **Passam**: `q5_0/q5_0` 104,50, `q4_1/q4_1` 104,52 e `q4_0/q4_0`
   104,79 tok/s (a 512, `--prefill-reps 2-3`), e `q8_0/q4_1` 99,65 tok/s (a 64, `reps 1` — as
   primeiras passadas de qualquer formato a 64 ficam em 101-102 tok/s por DPM, então esse número não
   é conclusivo por si).
6. **O crescimento do custo por token com o contexto (o "+6,9 %" da frente C) é 89,7 % kernel de
   atenção e 0 % escrita de KV** — e o KV é a única parte que o *formato* poderia aliviar. Fica
   registrado o que **não** vale perseguir: nenhuma fusão de `kv_write`/`rope`/`qk-norm` muda a
   curva, porque o bucket inteiro (`qk_norm_rope_kv`) é **plano** entre 64 e 4096 (0,047 → 0,044).

### O que não consegui medir, e por quê

- **A atribuição de `CONCAT` (48 despachos, 1,88 % do chunk deles) e de `CPY` (96, 0,10 %)** sem
  leitura de código: o logger não imprime forma para esses nós. Dou as duas pontas (bloco deles =
  7,61 % sem `CONCAT`; 9,59 % com ele inteiro).
- **A redundância de leitura de KV do `FLASH_ATTN_EXT`** (o fator de workgroup): o nó é um despacho
  por camada cobrindo 512 consultas; o quanto ele relê depende do bloqueio interno do shader.
  Reporto o **piso** (33,6 MB a 512) e, como **INFERIDO**, 4,5× para BM=64 em 8 workgroups. O nosso
  lado (51,6 GB, 1536×) é derivado da forma do nosso kernel, que é código nosso e verificável.
- **`bench-attn-gpu` com `q5_0`**: o tool quebra a partir de t=256 (mesmo page fault), então a curva
  isolada de `q5_0` só existe até t=64. A curva do prefill com `q5_0/q4_1` **não existe** — o
  caminho de produção também quebra (§3.3).
- **Nada de `--level 2` acima de N=128**: o teto é do tool, não do nível (nota de método 1). A 4096
  coexistiriam ~900 marcas/token ⇒ ~3,7 M eventos × 5,3 µs ≈ 20 s de evento em 41 s de corrida
  (+48 %), o que inviabiliza a medida de qualquer forma.
- **Se a máquina estava mais lenta hoje** (13-18 % acima da frente C) por DPM, térmica ou outro
  agente: não medi. O que posso afirmar é que os dois harnesses concordam entre si na minha janela
  (104,65 contra 105,11 tok/s a 512) e que o `flock` foi adquirido sem espera em todas as 12
  chamadas, com `VRAM_before` de 236-248 MiB.
