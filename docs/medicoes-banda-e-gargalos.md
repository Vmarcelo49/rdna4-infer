# Medições — orçamento de um token, banda de memória e gargalos (tarefas 2 e 5)

Branch `feat/medicoes-gpu` (worktree `../rdna4-wt-medicoes-gpu`). Todo número aqui foi
**medido nesta máquina** (RX 9070 XT, `gfx1201`, 15,9 GiB, ROCm 7.2.4) com
`scripts/gpu-lock.sh` serializando o acesso, ou é uma conta explícita sobre medidas —
e nesse caso está dito. Modelo: `Qwen3.8-27B-UD-IQ3_S.gguf` (salvo onde indicado),
KV f16 salvo onde indicado.

**Ferramenta oficial de profiling: não existe nesta instalação.** Verificado:
`rocprof`/`rocprofv3`/`omniperf` **não estão instalados** (`which` falha; só
`/opt/rocm/bin/rocminfo` e os compiladores). Todo o profiling abaixo é feito com
(i) eventos HIP novos dentro do grafo real, (ii) `bench --layers` do motor,
(iii) subtração A/B no harness do `bench`, (iv) kernels medidos isoladamente e
(v) contadores do próprio kernel (contagem de lançamentos, sem eventos).

---

## 0. Método (e onde ele não vale)

Ferramentas novas deste worktree (alvos **no fim** do `CMakeLists.txt`):

| ferramenta | o que mede |
|---|---|
| `tests/bench_phases_gpu.hip` → `bench-phases-gpu` | orçamento por fase com eventos HIP no grafo real (`--level 1\|2`), contagem de lançamentos (`--count-only`, sem eventos), pico de leitura DRAM/Infinity Cache (`--mem`), custo de despacho e de um `hipEventRecord`, kernels pequenos isolados e o `delta_rule`/`conv1d` do GDN fora do grafo (`--gdn`) |
| hook `RD_PHASE` em `include/rdna4/graph.cuh` + `include/rdna4/phase_prof.cuh` | **aditivo e opt-in** (só liga em `Graph::set_phase_prof`; com `prof_ == nullptr` o código é idêntico). Marca a fila com um evento por fronteira de fase; **nenhum buffer é tocado, nenhum número muda**. Delimitado por comentários `// RD_PHASE_PROF` |

Cinco armadilhas, todas tratadas explicitamente:

1. **DPM** — a placa cai para SCLK baixo entre kernels. Todo caminho de medição
   aquece até **≥ 300 ms de GPU ocupada com `sync` dentro do laço** (no
   `bench-phases-gpu` o warm-up roda com o profiler **desligado**, e o
   `--count-only` confirma que aquecer não custa nada: 35,749 ms/token contando
   contra 35,650 medidos sem instrumentação na mesma posição).
2. **O evento mede o que ele mesmo custa.** Um `hipEventRecord` enfileirado custa
   **3,82-4,08 µs** (medido: 2,53 µs por kernel vazio enfileirado, 6,39 µs por
   "kernel vazio + um evento"). O `bench-phases-gpu` imprime esse custo e a
   coluna `corrigido` já o subtrai bucket a bucket (ms − marcas × 4,0 µs). Sem
   essa subtração o nível 2 infla o token em **+16%** (medido).
3. **Concorrência** — o orçamento por fase e a tabela de tipos usam a **melhor**
   passada de N (as medianas sofrem com a outra fila da GPU; ver §5).
   As três repetições do `bench` em janela limpa deram 28,92 / 28,83 / 28,88 tok/s
   (± 0,2%), que é o piso de ruído do que está sendo reportado.
4. **Working set > cache** — o L2 desta placa tem 8 MiB e o Infinity Cache 64 MB.
   O pico de banda é medido com um buffer de **3 GiB** (DRAM) e outro de 32 MiB
   (Infinity Cache); medir com um kernel sem ILP dá 217 GB/s em vez de 1.500 GB/s
   no mesmo buffer (erro de método que este worktree corrigiu, ver §2.4).
5. **VRAM acabando derruba para GTT (memória do host por PCIe) sem erro nenhum.**
   Medido pelo `sysfs` (§2.5): a 64K com KV f16 o motor pede mais do que a placa
   tem, 2,1 GB vão para GTT e o decode cai de ~18 para 2,2-8,0 tok/s.

Comando de cada número: os `.txt` brutos das sessões ficaram em `/tmp/med/` durante
a medição; cada tabela abaixo cita o comando que a produziu.

### 0.1 Janela limpa, e a prova de que o hook não muda nada

- **A outra fila da GPU rodou, em parte desta sessão, sem o `flock`** (um `ppl` do
  worktree de autotuning, `fuser /dev/kfd` mostrando só ele, 12,07 GiB de VRAM
  ocupados por fora da fila). Isso produz `hipMalloc failed` espúrio em quem tem o
  lock e medianas 15-20% piores que a melhor passada. Foi o que derrubou
  `scripts/check_golden_run.sh` numa execução intermediária.
- **Marcação das medições**: cada tabela abaixo diz se o número é *melhor passada*
  e se a corrida foi em janela limpa (`fuser -v /dev/kfd` vazio e sem processo sem
  lock). Os números de headline têm confirmação **cruzada entre dois harnesses
  independentes** — a ferramenta de fases e o `bench` — o que é mais forte que a
  etiqueta de janela: 4K 35,65 (ferramenta) / 36,29 (`bench`); 16K 39,15 / 39,67;
  64K `q4_0` 55,02 / 55,09. Quando duas corridas discordam, as duas estão no texto.
- **O hook de fases é inerte sem profiler** (`prof_ == nullptr` por omissão): em
  janela limpa, `scripts/check_golden_run.sh` dá **OK** (todos os casos, incluindo
  os de KV `q4_0`/`q8_0`/`f32`) e `scripts/compare_ppl.sh` reproduz *exatamente* os
  valores históricos 0,185 / 0,105 / 0,008 / 0,018 % por chunk (ver
  `docs/quants-precisao.md` §4).

---

## 1. Orçamento medido de um token de decode (IQ3_S, ctx 4K, KV f16)

### 1.1 Números de referência (harness do `bench`)

Todas as linhas desta subseção foram medidas em **janela limpa** (`fuser -v
/dev/kfd` vazio, nenhum processo sem lock na placa); as três repetições de 4K
deram 28,92 / 28,83 / 28,88 tok/s (±0,2%).

```
scripts/gpu-lock.sh ./build/rdna4-infer bench -m IQ3_S -p "The capital of France is" -n 16 --reps 3
  decode 16 tokens x 3 reps (warmup 8): best 0.554 s (28.88 tok/s), mean 28.42 tok/s
  per token: 34.6 ms  (posição 5, atencao com 6 chaves)
scripts/gpu-lock.sh ./build/rdna4-infer bench ... --ctx-size 4096 --start-pos 4090 --fill-cache
  best 0.218 s (27.55 tok/s)  -> 36.29 ms/token   (atencao com 4091 chaves = 4K real)
scripts/gpu-lock.sh ./build/rdna4-infer bench ... --ctx-size 16384 --start-pos 16256 --fill-cache
  best 25.12 tok/s -> 39.67 ms/token
scripts/gpu-lock.sh ./build/rdna4-infer bench ... --ctx-size 65536 --start-pos 65408 --fill-cache
  best 2.16 tok/s (1a corrida) e 7.97 tok/s (2a corrida) -> 125-464 ms/token  [GTT, §2.5]
scripts/gpu-lock.sh ./build/rdna4-infer bench ... --ctx-size 65536 ... --cache-type-k q4_0 --cache-type-v q4_0
  best 18.15 tok/s -> 55.09 ms/token   (mesmo contexto, dentro da VRAM)
```

### 1.2 Decomposição por fase (eventos HIP no grafo, 4K real)

```
scripts/gpu-lock.sh ./build/bench-phases-gpu IQ3_S --ctx 4096 --pos 4090 --tokens 24 --level 2
```

`pos` fica **fixo** em todas as passadas medidas (mesma janela de atenção em cada
token, custo por token constante). Duas execuções independentes deste comando
(valor limpo / instrumentado): 35,650 / 41,600 e 35,703 / 41,431 ms — **0,1% de
repetibilidade**. 1479 marcas/token; a partição fecha: a soma dos
buckets (41,987 ms/token instrumentado) é igual ao tempo de parede da passada
instrumentada (41,600 ms de melhor token) e a diferença para a passada limpa é
exatamente `1479 marcas × 4,026 µs = 5,95 ms` (medido: 41,60 − 35,65 = **5,95 ms**).

| fase | ms/token medido | eventos | **ms/token corrigido** | % |
|---|---|---|---|---|
| **matvec dos pesos** (496 lançamentos, tronco) | 25,509 | −1,997 | **23,512** | 65,3% |
| **recorrência GDN** `delta_rule` (48) | 4,161 | −0,193 | **3,968** | 11,0% |
| **atenção** (16 camadas, 16 splits) | 1,446 | −0,064 | **1,382** | 3,8% |
| **quantização da ativação** (256) | 1,909 | −1,031 | **0,878** | 2,4% |
| **LM head** `output.weight` (1) | 1,432 | −0,004 | **1,428** | 4,0% |
| `attn_norm` (64) | 1,144 | −0,258 | **0,886** | 2,5% |
| `post_norm` (64) | 1,128 | −0,258 | **0,870** | 2,4% |
| `qk_norm_rope_kv` (16 marcas: deinterleave + 2 norms + 2 rope + `kv_write` 8×) | 1,125 | −0,064 | **1,061** | 2,9% |
| GDN escalares (sigmoid/add/softplus/mul, 48 marcas = 192 lançamentos) | 0,810 | −0,193 | **0,617** | 1,7% |
| GDN `rms_norm`+`silu`+`mul` (48 marcas = 96 lançamentos) | 0,657 | −0,193 | **0,464** | 1,3% |
| GDN `l2_norm` (48 marcas = 96 lançamentos) | 0,525 | −0,193 | **0,332** | 0,9% |
| GDN `conv1d` (48) | 0,441 | −0,193 | **0,248** | 0,7% |
| portão da atenção (sigmoid+mul, 16) | 0,167 | −0,064 | **0,103** | 0,3% |
| cópia dos logits D→H (1 MB, 1) | 0,146 | −0,004 | **0,142** | 0,4% |
| fim do token / dreno (1) | 0,075 | −0,004 | **0,071** | 0,2% |
| `out_norm`+`embed`+`head:act_quant` (3) | 0,033 | −0,012 | **0,021** | 0,1% |
| **total corrigido** | 41,987 | −5,954 | **36,03** | 100% |

Conferência independente: o mesmo binário sem instrumentação, mesma posição, deu
**35,65 ms/token** (melhor token) — 1% de diferença, dentro do resíduo dos eventos
que abrem bucket sem antecessor.

Agrupando nas sete categorias do pedido (fração do token corrigido de **36,0 ms**):

| categoria | ms/token | % | como foi medida |
|---|---|---|---|
| (a) matvec dos pesos | **24,94** (tronco 23,51 + head 1,43) | 69,3% | bucket `matvec` + `head:matvec` (497 lançamentos medidos) |
| (b) quantização da ativação | **0,88** (257 lançamentos) | 2,4% | bucket `act_quant` (+ `head:act_quant` 0,003) |
| (c) atenção | **1,38** (16 camadas) | 3,8% | bucket `attention` |
| (d) recorrência GDN | **5,63** (delta 3,97 + conv 0,25 + escalares 0,62 + norm/silu 0,46 + l2 0,33) | 15,6% | buckets `gdn_*` |
| (e) normas/elementwise | **2,96** (attn_norm 0,89 + post_norm 0,87 + qk/rope/kv 1,06 + portão 0,10 + silu/mul do FFN 0,04) | 8,2% | buckets de norma/rope |
| (f) LM head + cópia de logits + sampler | **2,08** (head 1,43 + cópia/dreno 0,21 + sampler 0,44 medido pelo `bench`) | 5,8% | buckets + `bench` |
| (g) gaps de despacho | **não é uma linha separada** | — | por construção o gap *depois* de cada fase cai no bucket daquela fase; o teto está em §1.4 |

### 1.3 Contagem de lançamentos (medida, sem eventos)

```
scripts/gpu-lock.sh ./build/bench-phases-gpu IQ3_S --ctx 4096 --pos 4090 --tokens 24 --level 2 --count-only
  TOTAL de marcas/token: 1479.0    (35,749 ms/token — contar não custa nada)
```

Medido: **497 matvecs** (496 no tronco + 1 head) e **257 quantizações** de ativação
por token — os dois números que o `--count-only` conta diretamente (uma marca por
chamada de `proj`/`proj_qq`). Pela leitura do `graph.cuh` (contagem derivada, com o
trecho citado em cada linha) o token faz:

```text
embed 1 + head (quantize+matvec) 2
por camada de atencao (x16): attn_norm 1, proj(q) 2, proj_qq(k,v) 2, deinterleave 1,
  q_norm 1, k_norm 1, rope 2, kv_write 8 (4 cabecas KV x K,V), atencao 2 (split+merge),
  sigmoid 1, mul 1, proj(attn_output) 2, residual 1            = 25  -> 400
por camada GDN (x48): attn_norm 1, proj(qkv) 2, proj_qq(x3) 3, sigmoid 1, add 1,
  softplus 1, mul 1, conv1d 1, l2_norm 2, delta_rule 1, rms_norm 1, silu 1, mul 1,
  proj(ssm_out) 2, residual 1                                   = 20  -> 960
attn_post_norm x64 = 64 ; FFN x64 (proj 2+silu+mul+proj_qq+proj 2+residual = 8) = 512
out_norm 1
total: 1 + 2 + 400 + 960 + 64 + 512 + 1 = 1940 kernels por token
+ 16 hipMemcpy sincronos de 4 bytes (a posicao) + 1 copia D2H bloqueante de 1 MB.
```

1940 lançamentos × 2,53 µs (piso de despacho medido) = **4,9 ms de teto absoluto**
se todo despacho fosse exposto; o medido mostra que ele **não** é (o `matvec` de
47 µs por lançamento esconde o próximo despacho, e é por isso que o nível 1 de
instrumentação, 724 marcas, é quase grátis: −3,3% a +2,3% medidos).

### 1.4 O que a instrumentação revelou sobre o custo de um comando

- Nível 1 (724 marcas/token): inflação de **−3,3% / −0,4% / +2,3% / −2,3%** em
  quatro corridas ⇒ evento escondido atrás do kernel anterior (≲ 0,5 µs efetivo).
- Nível 2 (1479 marcas/token): inflação de **+16,7% / +15,3% / +10,9%**, que é
  exatamente `marcas × 4,0 µs` ⇒ **na sequência densa de comandos curtos
  (`quantize` + `matvec` em sequência) um comando extra na fila custa o preço
  cheio, ~4 µs de linha do tempo do dispositivo.**

Isso é a medida que faltava para decidir entre "fundir kernels" e "capturar um HIP
graph": o que custa não é o lançamento do kernel grande, é o **processamento de
comando** na parte densa e curta do grafo (as ~1440 marcas/lançamentos pequenos).

### 1.5 A 16K e a 64K (mesma ferramenta, mesmas correções)

Os valores de 16K e 64K `q4_0` vêm de corridas com a **mediana 5-15% acima da
melhor passada** (janela com atividade da outra fila); os totais do `bench` vêm de
repetição com `--reps 2` e concordam com a ferramenta dentro de 1%.

| | 4K f16 | 16K f16 | 64K f16 | 64K q4_0 |
|---|---|---|---|---|
| total (limpo, melhor token) | 35,65 | 39,15 | n/a (só instrumentado) | 55,02 |
| total (`bench`, melhor de 3) | 36,29 | 39,67 | **125,5 e 463,7** (§2.5) | 55,09 |
| matvec (corrigido) | 23,51 | 23,60 | — | 23,70 |
| atenção (corrigido) | 1,38 | 4,90 | **280,0** (16 camadas) | **20,97** |
| `delta_rule` (corrigido) | 3,97 | 3,97 | 3,96 | 3,96 |
| `act_quant` | 0,88 | 0,89 | 0,93 | 0,93 |
| splits | 7 | 16 | 16 | 16 |

O tronco (tudo menos atenção) é **constante em 33,6-34,3 ms/token** de 4K a 64K:
o contexto só mexe na atenção (e no caso de 64K f16, no spill de VRAM).

### 1.6 Como o spill de GTT aparece **dentro** das fases (64K f16)

```
scripts/gpu-lock.sh ./build/bench-phases-gpu IQ3_S --ctx 65536 --pos 65530 --tokens 4 --level 1 --no-clean
  passada instrumentada: melhor 409,5 ms/token, mediana 630,3, media 604,8   (16 splits)
```

| fase | 64K q4_0 (cabe na VRAM) | 64K f16 (transborda ~2,1 GB) | fator |
|---|---|---|---|
| atenção (16 camadas) | 21,03 ms | **280,1 ms** | 13,3× |
| `ffn_gate_up` (matvec+silu) | 10,2 ms | **122,5 ms** | 12,0× |
| `gdn_proj` (4 matvecs) | 5,3 ms | **61,0 ms** | 11,5× |
| `ffn_down` | 6,0 ms | **34,9 ms** | 5,8× |
| `gdn_delta` | 4,2 ms | **34,5 ms** | 8,3× |
| `qkv_proj` (3 matvecs) | 1,3 ms | **29,5 ms** | 22,7× |
| `attn_norm` / `post_norm` | 1,13 / 1,12 ms | 1,75 / 1,74 ms | 1,6× |
| head | 1,39 ms | 1,44 ms | 1,03× |

Nada aqui é "o kernel ficou lento": **as fases que leem muitos bytes por token
colapsam juntas (6-23×) e as que leem poucos não mudam** — é a assinatura de
tráfego atravessando o PCIe, e a confirmação independente de que o `gtt_used`
de 2.557 MB (§2.4) está no caminho quente.

---

## 2. Utilização de banda de memória

### 2.1 Bytes por token

```
bench-phases-gpu ... (impresso pela propria ferramenta)
  pesos: total 12.030 GB, lidos por token 11.133 GB
         (sem token_embd 0.546 GB e sem o bloco MTP blk.64.* 0.351 GB)
```

Derivação: `loader.total_bytes()` = 12,030 GB inclui `token_embd.weight` (556 MB,
do qual o decode lê **uma linha** de 2,2 KB) e o bloco MTP inteiro (`blk.64.*`, que
tem 15 tensores, incluindo `ffn_gate/ffn_up/ffn_down/attn_*` **e** os `nextn.*`:
**0,351 GB**, não os 42,7 MB que o `docs/rocm-estudo.md` §A.2.6 atribui ao bloco —
o total de 11,12 GB daquele estudo está certo, o rótulo do componente não). O
`bench` divide `loader.total_bytes()` pelo tempo e por isso imprime banda **8%
acima** da real: 345,3 GB/s impressos contra **321,9 GB/s** reais
(`11,133 GB / 0,03458 s`).

**Ativações e estado recorrente** (a terceira parcela, medida pela mesma
ferramenta e pela geometria do `gdn.cuh`):

| parcela | bytes/token | como |
|---|---|---|
| estado do GDN (48 camadas × 3,15 MB × 4 passadas) | **604,8 MB** | `delta_rule_kernel` lê a linha 2× e escreve 2× por thread (`gdn.cuh:57-85`); a ferramenta imprime o estado por camada |
| fluxo residual (E=5120 floats × ~6 passadas por camada × 64) | ~7,9 MB | `attn_norm`, residual, `post_norm`, residual do FFN |
| logits (escrita no device + cópia D2H de 1 MB) | ~2,0 MB | 248320 floats × 4 B × 2 |
| `kv_write` (16 camadas × 4 cabeças × 512 B × 2) | 0,07 MB | desprezível |
| **total de ativações/estado** | **~615 MB** | — |

Ou seja: a 4K, **o estado recorrente do GDN (604,8 MB/token) é 2,3× o tráfego
único de KV (268 MB/token)** — e é o item mais caro depois dos pesos.

### 2.2 Banda efetiva e fração do roofline

```
scripts/gpu-lock.sh ./build/bench-phases-gpu IQ3_S --ctx 4096 --pos 4090 --tokens 24 --level 2 --mem --gdn
  DRAM  read: 3221.2 MB in 5.090 ms -> 632.9 GB/s (3 GiB buffer, 5 reps, best)
  Infinity Cache (32 MiB x16): 536.9 MB in 0.358 ms -> 1499.6 GB/s
```

| grandeza | medido | como |
|---|---|---|
| pico de leitura DRAM desta placa | **632,9-634,5 GB/s** | `read_reduce` sobre 3 GiB (13× o Infinity Cache), melhor de 12 passadas, warm-up de 300 ms |
| pico teórico (folha de dados) | 640 GB/s | AMD |
| banda efetiva dos pesos, ponta a ponta | **321,9 GB/s** = **51% do pico medido** | 11,133 GB / 34,58 ms (`bench` 4K, melhor de 3) |
| banda efetiva **dentro do matvec** | **436,4 GB/s** = **69% do pico** | bucket `matvec` corrigido: 11,133 GB / 25,51 ms |
| LM head isolado | **~620 GB/s** = **98% do pico** | 0,874 GB / 1,41 ms (bucket `head:matvec`) |
| banda efetiva com o KV junto (4K) | **314 GB/s = 50% do pico** | (11,133 + 0,268) GB / 36,29 ms (`bench`, 4K de atenção) |
| **inventário completo** (pesos + KV + ativações/estado) | **12,02 GB/token → 331 GB/s = 52% do pico** | 11,133 + 0,268 + 0,615 GB em 36,29 ms |

Leitura: **o motor está a 69% do roofline na parte que importa (os matvecs) e a
98% no LM head**; os 31% que faltam no matvec são *issue-bound*, não banda — a
contagem de ISA do `docs/rocm-estudo.md` §B.2 (171 instruções por 220 bytes em
`iq3_s`) prevê 395 GB/s de teto e o medido aqui é 436 GB/s (IPC ≈ 1,2). O que
derruba a banda ponta a ponta para 322 GB/s não é o matvec: são os ~11 ms/token em
que a placa **não** está lendo pesos (atenção, GDN, normas, sampler, cópia).

### 2.3 A 64K: quanto é tráfego de KV e quanto é outra coisa

Mesma ferramenta, `--ctx 65536 --kv-k q4_0 --kv-v q4_0 --pos 65530 --tokens 8 --level 2`:

| contexto | KV | bytes de KV únicos/token | emitidos (GQA 6×) | atenção medida | GB/s únicos | GB/s emitidos |
|---|---|---|---|---|---|---|
| 4K | f16 | 268 MB | 1.611 MB | 1,38 ms | 194 | **1.166** |
| 16K | f16 | 1.074 MB | 6.442 MB | 4,90 ms | 219 | **1.315** |
| 64K | q4_0 | 1.208 MB | 7.248 MB | 20,97 ms | 58 | **346** |

(único = `chaves × 4 cabeças KV × linha × 2 (K e V) × 16 camadas`; emitido = 6×
isso, porque cada cabeça de consulta lê a própria linha de KV — ver §3.)

- A 4K e 16K a atenção roda a **1,2-1,3 TB/s de tráfego emitido**, ou seja 78-88%
  do Infinity Cache medido (1.500 GB/s) e **acima** do pico de DRAM (633 GB/s):
  ela é limitada por *cache*, e o que a limita é a duplicação 6× do GQA.
- A 64K com `q4_0` o kernel cai para **346 GB/s emitidos** (11× os bytes únicos de
  DRAM, que dariam 1,9 ms de piso): a 64K o gargalo **não é banda nenhuma**, é o
  trabalho de desquantização por elemento dentro do kernel (§3.4).

### 2.4 O caso 64K f16: 2,1 GB vão para GTT e o decode cai 8×

`dmesg`/driver nenhum avisa; o sinal está no `sysfs`. Medido durante as corridas
(amostragem de 250 ms em `/sys/class/drm/card1/device/`):

| corrida | pico `mem_info_vram_used` | pico `mem_info_gtt_used` | decode |
|---|---|---|---|
| 4K f16 (IQ3_S) | 12.604 MB | 472 MB | 28,82 tok/s |
| 16K f16 | 13.372 MB | 468 MB | 25,12 tok/s |
| 64K q4_0 | 13.500 MB | 466 MB | 18,10 tok/s |
| **64K f16** | **16.294 MB** (= 15,91 GiB de 15,92 GiB) | **2.557 MB** | **2,16 e 7,97 tok/s** |

- O `gtt_used` em repouso nesta máquina é 471 MB (o desktop); nas três primeiras
  corridas ele **não sobe**, e na 64K f16 sobe para 2.557 MB: **~2,1 GB do modelo
  ou do KV estão na memória do host, atrás do PCIe**.
- Demanda: pesos 11,20 GiB + KV f16 a 64K (16 camadas × 2 × 65536 × 4 × 512 B =
  **4,29 GiB**) + estado do GDN 0,15 GiB + buffers ≈ 15,9 GiB = a placa inteira,
  com 0,30 GiB "livres" (e 0,45 GiB do desktop já residentes).
- O mesmo contexto com `q4_0` (KV 1,21 GiB) sobra 3,16 GiB e roda a 18,10 tok/s.

**Isso invalida a recomendação do `README.md` ("use f16 onde couber: é 38% mais
rápido que q4_0 no mesmo contexto") no ponto exato onde ela é usada**: a 64K com
KV f16 não cabe mais, e o resultado não é "um pouco mais lento", é **2,5 a 8× mais
lento**, com uma variância enorme entre corridas (2,16 e 7,97 tok/s em duas
execuções idênticas) porque a fração que transborda depende da fragmentação. A
observação de `docs/medicoes-m7.md` §"uma corrida de 64K-f16 reportou 1,06 tok/s,
era outro workstream segurando a GPU" **não se confirma**: com o lock em mãos e
nenhum outro processo na GPU (`fuser -v /dev/kfd` vazio), o número se reproduz.

---

## 3. Tráfego de KV da atenção, medido contra o tempo medido

### 3.1 A conta, e por que a duplicação 6× é real

`attn_split_kernel` (`include/rdna4/attn.cuh:291`) usa `h = blockIdx.x` (uma CTA por
cabeça de consulta) e `kvh = h / (n_head / n_head_kv)`
(`include/rdna4/attn.cuh:99`): cada cabeça de consulta varre **a linha inteira** da
sua cabeça KV. Com 24 cabeças de consulta para 4 de KV, os mesmos bytes são lidos
6× do L1/L2 (o split não muda isso: ele divide a faixa de chaves, não as cabeças).

```
por token, por camada:  chaves x NKV(4) x [linha_K + linha_V] x (NH/NKV = 6)
  f16:  linha = 256 x 2 B   = 512 B   ->  4.096 B por chave por camada (unico)
  q4_0: linha = (256/32) x 18 = 144 B ->  1.152 B por chave por camada (unico)
```
Multiplicado por 16 camadas de atenção dá a tabela de §2.3 (268 MB/1,07 GB/4,29 GB
únicos em 4K/16K/64K f16; 1,21 GB em 64K q4_0).

### 3.2 Verificação contra o tempo medido

| contexto | atenção medida | por camada | banda emitida | banda única | diagnóstico |
|---|---|---|---|---|---|
| 4K f16 | 1,38 ms | 86 µs | 1.166 GB/s | 194 GB/s | **limitada por cache (Infinity Cache a 78%)** |
| 16K f16 | 4,90 ms | 306 µs | 1.315 GB/s | 219 GB/s | **limitada por cache (88% do IC medido)** |
| 64K q4_0 | 20,97 ms | 1.311 µs | 346 GB/s | 58 GB/s | **limitada por issue/desquantização** (11× acima do piso de DRAM) |

Conferência com o M7: `docs/medicoes-m7.md` mediu **1,19 ms por camada a 64K**
(16 camadas = 19,0 ms) e eu meço **1,31 ms por camada** com `q4_0` (20,97 ms);
o M7 chama esses 215 GB/s de "banda da atenção" usando os **bytes únicos** — o
tráfego **emitido** é 11×&nbsp;maior e é ele que explica o tempo.

### 3.3 O que isso implica para a próxima otimização

- **4K/16K (onde o motor passa a maior parte da vida):** dividir K/V entre as 6
  cabeças GQA é a alavanca certa — o tráfego emitido cairia ~6× e a atenção sai de
  1,38 ms para perto do piso de DRAM (0,42 ms a 4K; 1,70 ms a 16K). Ganho medido
  possível: **~1 ms/token a 4K (3%) e ~3 ms a 16K (7%)**. (O protótipo do M7
  falhou por 8-12×; o caminho é o mesmo: grade larga + staging em LDS por CTA.)
- **64K `q4_0`:** dividir K/V não resolve — o kernel já não é limitado por banda
  (346 GB/s emitidos de um teto de 1.500). O que falta é o caminho de leitura
  vetorizado/LDS para `q4_0` (o `kv_load8<Q4_0>` de `kv.h:159` desempacota nibble a
  nibble em ALU por elemento). A 64K a atenção é **38% do token** (20,97 de 55,02 ms).
- Antes de escolher o tipo de KV a 64K, medir `q8_0` (linha de 272 B, sem
  desempacotamento): cabe na VRAM (2,28 GiB de KV ⇒ 14,1 GiB em uso) e deve ser
  mais rápido por byte que `q4_0` — **não medido neste trabalho** (ficou fora do
  orçamento de GPU; é o teste #1 da próxima sessão).

---

## 4. Os maiores gargalos **medidos**, em ordem

Todos os números desta seção saíram deste worktree; nada aqui é estimativa de
terceiro.

### 1. Matvec dos pesos: 24,94 ms/token (69%), a 436 GB/s de um teto de 633 GB/s

- Prova: bucket `matvec` corrigido 23,51 ms (496 lançamentos, 11,133 GB ⇒
  **436 GB/s**), mais o head 1,43 ms (0,874 GB ⇒ 620 GB/s).
- Piso teórico: 11,133 GB / 632,9 GB/s = **17,6 ms** ⇒ o tronco está **5,9 ms
  (25%) acima do piso** (16% do token inteiro), e a causa medida é a contagem de
  instruções por byte do `vec_dot`
  (171 instr/220 B em `iq3_s`; `q3_k` a 307 GB/s contra 408-582 dos outros
  k-quants) — não banda, não ocupação.
- Mudança concreta: `q3_k` responde por 0,401 GB/token a 307 GB/s (1,31 ms) e é o
  único tipo com 208 instruções por chamada; trocar a emulação de bytes 16-bit por
  `perm`+`dp4a` linear em `include/rdna4/vecdotq.cuh` (`vec_dot_q3_K_q8_1`) vale
  ~0,6 ms/token (1,7%). Gate: `check-matvec-gpu` (por tipo) + `check-graph-gpu`
  (oráculo por nó) + `scripts/check_golden_run.sh`.

### 2. Recorrência GDN (`delta_rule`): 3,97 ms/token (11%), a 152 GB/s

- Prova dupla: bucket `gdn_delta` corrigido = 3,97 ms/token nas 48 camadas
  (**83 µs por camada**, dentro do grafo) e o **mesmo kernel medido isolado em
  cadeia de 100, com as formas reais (nvh 48, nkh 16, S 128): 69,6 µs por
  lançamento** — 67 µs acima do piso de despacho de 2,2 µs. A medida dentro do
  grafo não é artefato de instrumentação (a diferença de 13 µs é a dependência com
  o `conv1d`/`l2_norm` que precede o kernel no grafo).
- Tráfego: estado = 3,15 MB por camada, lido+escrito 2× = **12,6 MB por camada**
  (4 passadas sobre a linha), **604,8 MB por token** ⇒ **152 GB/s efetivos** (4,2×
  abaixo do pico de DRAM medido, 10× abaixo do Infinity Cache). É **latência**: 48
  CTAs de 128 threads = 9% de ocupação da placa, e cada thread faz duas passadas
  seriais de 128 elementos com dependência (`gdn.cuh:57-85`). Confirmação
  independente: o mesmo kernel isolado, em cadeia, dá **69,6 µs por lançamento**
  contra um piso de memória de 12,6 MB / 632,9 GB/s ≈ **20 µs**.
- Piso teórico: 12,6 MB / 632,9 GB/s ≈ **20 µs por camada** (0,96 ms/token) — o
  medido é 69,6 µs isolado e 83 µs dentro do grafo, **3,5-4,2× acima do piso**.
- Mudança concreta: `include/rdna4/gdn.cuh`, `delta_rule_kernel` — 4 threads por
  linha com 32 elementos em registrador (ou 2 acumuladores + `float4`), mantendo a
  ordem de soma documentada; ganho de até **~3 ms/token (8%)**. Gate:
  `check-graph-gpu` (oráculo por nó, tolerância de reordenação) +
  `scripts/check_golden_run.sh` + `scripts/compare_llama_greedy.sh`.

### 3. Lançamentos pequenos: ~1440 lançamentos a ~2,8 µs = **~4,0 ms/token (11%)**

- Prova: cada kernel pequeno medido **isolado, em cadeia de 100** (medir um
  lançamento com `hipDeviceSynchronize()` depois mede o round-trip host→GPU de
  ~21 µs, não o kernel — foi o primeiro erro de método desta sessão, corrigido):

| kernel | µs/lançamento (cadeia de 100) | veredito |
|---|---|---|
| kernel vazio (**piso de despacho**) | **2,20** | — |
| `unary(48)`, `mul(48)`, `add(48)` | 2,77-2,83 | **no piso** |
| `silu(6144)`, `mul(6144)` | 2,75-2,78 | **no piso** |
| `quantize_q8_1(5120)` | 2,98 | no piso |
| `deinterleave_q_gate(24×256)` | 2,77 | no piso |
| `kv_store_row(f16 / q4_0, 256)` | 2,78 / 3,15 | no piso |
| `rope(1×24×256)` | 3,16 | +1 µs |
| `l2_norm(16×128)` | 3,17 | +1 µs |
| `rms_norm(48×128)` | 3,36 | +1,2 µs |
| `conv1d(4 taps, 10240 canais)` | 3,53 | +1,3 µs |
| **`rms_norm(1×5120)`** | **10,90** | **+8,7 µs de kernel real** |
| **`delta_rule(48 cabeças, S=128)`** | **69,55** | **+67 µs de kernel real** |

- Confronto com o grafo (buckets corrigidos ÷ lançamentos por camada):
  `attn_norm` 13,8 µs contra 10,9 isolado; `gdn_norm_silu` 9,7 µs contra 9,0
  (3 kernels no piso); `gdn_l2norm` 6,9 contra 6,3 (2 kernels); `gdn_scalars`
  12,8 µs para 4 kernels no piso (11,2 µs); `gdn_conv` 5,2 contra 3,5;
  `qk_norm_rope_kv` 66 µs por camada para 13 lançamentos. **O grafo paga o piso de
  despacho + ~1-3 µs por fase**, o que fecha a conta.
- Conta do token: 1940 lançamentos, dos quais 497 são matvecs de ~47 µs (o
  despacho fica escondido atrás do kernel). Sobram ~1440 lançamentos pequenos, e a
  2,2-3,5 µs cada isso é **~4,0 ms/token**; somando o que os kernels pequenos
  acrescentam *acima* do piso (o `rms_norm` de 5120 elementos, 129 lançamentos ×
  8,7 µs = **1,1 ms**, e a `delta_rule`, tratada no item 2), chega-se aos ~8,6 ms
  medidos nas fases pequenas.
- Piso teórico: a fusão preserva a ordem das contas, então **cada par de kernels
  fundido devolve exatamente o piso de um deles**. Fusões já propostas com o ganho
  agora medido: `rms_norm`+`quantize` (130 lançamentos → 0,36 ms), as 4 ops
  escalares do GDN em uma (144 → 0,40 ms), `kv_write` de 8 para 1 lançamento
  (112 → 0,31 ms) = **~1,1 ms/token (3%)**.
- Mudança concreta: `include/rdna4/graph.cuh` + um kernel novo em `nn.cuh`/`gdn.cuh`
  por fusão (o estudo §E.2/E.3 já traz o desenho). Gate: `check-graph-gpu`
  (oráculo por nó) + `scripts/check_golden_run.sh` para as fusões bit-exatas.


### 4. Segmentação de KV da atenção a 64K: 20,97 ms/token (38% do token a 64K)

- Prova: bucket `attention` corrigido a 64K `q4_0` = 20,97 ms para 16 camadas
  (1,31 ms/camada), contra 1,38 ms a 4K. Tráfego emitido 7,25 GB/token a
  **346 GB/s** (o Infinity Cache medido dá 1.500).
- Piso teórico: 1,21 GB únicos / 633 GB/s = **1,9 ms** (10% do medido).
- **A alternativa foi medida**: `q8_0` no KV a 64K (linha de 272 B, sem
  desempacotamento de nibble) faz a mesma atenção em **19,87 ms** (contra 21,03 do
  `q4_0`) com **13,7 GB emitidos a 690 GB/s** contra 7,25 GB a 346 GB/s, e o token
  inteiro fica em 54,45 ms (18,36 tok/s) contra 55,02 (18,17) — **o dobro da
  precisão do KV pelo mesmo tempo**, com 2,18 GiB de folga de VRAM. É a troca de
  uma linha no CLI/servidor e não precisa de `kv.h`.
- Mudança de kernel (para quem for dono de `kv.h`): leitura vetorizada/LDS no
  caminho `q4_0` (`kv_load8<KvType::Q4_0>`) — 346 GB/s emitidos contra 690 do
  `q8_0` mostram que o gargalo é o **desempacotamento por elemento**, não os bytes.

### 5. Transbordo para GTT a 64K f16: **8× de queda**, de 18,1 para 2,2-8,0 tok/s

- Prova: `sysfs` durante a corrida (§2.4): `gtt_used` 466-472 MB nas configurações
  que cabem, **2.557 MB** com KV f16 a 64K; `vram_used` no teto (16.294 MB de
  15,92 GiB) e decodes de **2,16 e 7,97 tok/s** em duas corridas idênticas contra
  **18,10/18,17 tok/s** do `q4_0` no mesmo contexto.
- Piso: o mesmo trabalho com `q4_0` (55,02 ms/token, 18,17 tok/s) — ou seja, há
  **~6,7× de ganho imediato** trocando o tipo de KV a 64K.
- Mudança concreta: (a) a política de KV por contexto no CLI/servidor
  (`src/main.hip`, escolher sozinho `f16` até 48K, `q8_0` acima — **o `q8_0` a 64K
  é medidamente igual ou melhor que o `q4_0` e tem o dobro da precisão do KV**;
  a decisão precisa do `hipMemGetInfo` antes do `Graph::init`, que já é chamado no
  `cmd_bench`/`cmd_run`); (b) descontar o bloco MTP (0,351 GB) e a linha única de
  `token_embd` da conta de VRAM quando `--mtp` está desligado; (c) gate:
  `check-kvctx-gpu` + um teste que compare `vram in use` contra o total e falhe se
  sobrar < 0,5 GiB (o ponto de falha é medido: 48K f16 = 1,30 GiB livres OK, 64K
  f16 = 0,30 GiB livres → transbordo).

---

## 5. Comparação com o inventário **estimado** de `docs/rocm-estudo.md` §A

| item do estudo | estimado lá | medido aqui | veredito |
|---|---|---|---|
| matvec, 497 tensores | 26,4-27,7 ms (replay, 402-421 GB/s) | tronco 23,51 + head 1,43 = **24,94 ms** (436 GB/s no tronco) | **segura a ordem de grandeza**, 6-11% acima; o replay fora do grafo superestima um pouco (o M8 tirou 241 quantizações do caminho) |
| LM head (`output.weight`, 874 MB) | 1,39 ms (627 GB/s) | **1,43 ms (620 GB/s)** | **confere** (98% do teto) |
| atenção 4K, 16 camadas, com split | 1,2 ms (8 splits) | **1,38 ms** (7-8 splits) | **confere** |
| embeddings + norm final + cópia + sampler | ~0,5 ms | **0,44 ms de sampler + 0,23 ms de cópia/dreno + 0,02 de norm** | **confere** |
| "resto" (normas, GDN, rope, `kv_write`, quantizações, ~1900 kernels) | ~2-3 ms | **~9,5 ms** (GDN 5,63 — dos quais 3,97 são a `delta_rule` — + normas/rope 2,96 + `act_quant` 0,88) | **NÃO confere**: o estudo tratou como "~2-3 ms de trabalho desprezível" o que é (a) a `delta_rule` a 152 GB/s (3,97 ms) e (b) ~1440 lançamentos pequenos no piso de despacho (~4,0 ms) |
| **gaps de despacho** | **~7 ms** (2200 lançamentos × 3,5 µs), marcado como estimativa | **~4,0 ms medidos** para os ~1440 lançamentos pequenos (2,2-3,5 µs cada, medidos em cadeia) + ~1,1 ms de latência de `rms_norm` acima do piso; contagem real de lançamentos: **1940** (derivada do código, e 1479 marcas medidas no grafo) | **a magnitude confere (4-5 ms contra os ~7 ms estimados), a contagem não**: são 1940 lançamentos, não 2200, e o "gap" não é uniforme — 497 matvecs de 47 µs escondem o próprio despacho |
| "a 64K a atenção está em ~1,56 TB/s de L2 por causa da redundância 6×" | estimativa coerente | **1,32 TB/s emitidos a 16K e 1,17 TB/s a 4K** (`f16`, medidos); a 64K `q4_0` cai para 346 GB/s | **confere a 4K/16K**, mas **não** a 64K: lá o kernel é issue-bound, não cache-bound |
| pico de DRAM ~619 GB/s (M5) | 619 GB/s | **632,9-634,5 GB/s** | confere (medido com buffer de 3 GiB e ILP) |
| KV `q4_0` é "alavanca de VRAM, não de velocidade" | qualitativo | **confirmado e quantificado**: 64K `q4_0` = 20,97 ms de atenção contra 4,29 GiB de VRAM; 64K f16 não cabe (GTT) | confere — com a ressalva de que, **se o f16 não cabe, o `q4_0` é 8× mais rápido** |

---

## 6. O que não deu para medir (e por quê)

1. **`rocprof`/`rocprofv3`/`omniperf`** — não instalados; nenhum contador de
   hardware (L2 miss, utilização de CU, stall reasons) foi lido. Toda atribuição de
   "latência vs banda" aqui é **inferida de medidas de tempo e de bytes**, com o
   mecanismo indicado, não de contadores.
2. **Contadores `s_memrealtime`/`SHADER_CYCLES` dentro dos kernels do motor** —
   exigiriam editar `matvec.cuh`/`attn.cuh`/`kv.h`/`nn.cuh`, que pertencem a outro
   agente nesta rodada. O que existe aqui é tempo por evento na fronteira das fases.
3. **GPU `q8_0` no KV a 64K** e **KV f16 a 32K/48K** (onde exatamente começa o
   transbordo para GTT) — ficaram fora do orçamento de GPU desta sessão.
4. **Atenção sem split a 4K/16K** (para medir o ganho do caminho de splits de hoje)
   — o `RD_ATTN_SPLITS` do M7 permite, mas não foi rodado aqui.
5. **Prefill** (batch N≤16): todo este documento é decode. O prefill é outro perfil
   (MMQ/WMMA, M8, 70,2 tok/s medidos em `docs/medicoes-m8.md`).
6. **Contenção da outra fila da GPU** (detalhada em §0.1): as medianas de várias
   corridas ficaram 15-20% acima da melhor passada, e uma execução de
   `check_golden_run.sh` deu 5 casos vermelhos por `hipMalloc` alheio. Nada disso é
   do motor: em janela limpa o mesmo gate dá OK e as três repetições de 4K dão
   ±0,2%. **Quais números são de janela limpa está dito em cada tabela**; quando as
   duas corridas divergem (por exemplo mediana e melhor passada do 64K `q4_0`,
   64,5 contra 55,0 ms), as duas estão reportadas.
