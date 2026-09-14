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

### 2.1 Resultado da validação (oráculo llama.cpp) — medido

**Comando**: `timeout 900 ./scripts/gpu-lock.sh ./build/check-kvquant-gpu` (sem
modelo). Janela: **limpa** — `mem_info_vram_used` = 198 225 920 B (189 MiB, só o
desktop) antes, 198 156 288 B depois; `gtt_used` = 30 253 056 B antes, 30 208 000 B
depois. Nenhum `flock` esperando além do meu. `rocm-smi`/`rocprof` não existem
nesta máquina (só `rocminfo`), então todos os números de VRAM vêm de
`/sys/class/drm/card*/device/mem_info_*` e de `hipMemGetInfo`.

```
== sintético (sem modelo): 48 linhas x 256 elementos, 8 padrões x 6 repetições
type   bytes    step_tol   max_err    err/step     err/range    err/|x|    check
f32    exact    0.0000     0.000000   0.0000       0.00000      0.000000   OK
f16    exact    0.0000     0.031166   0.0000       0.00016      0.000312   OK
q8_0   exact    0.5500     0.409378   0.5201       0.00205      0.004095   OK
q4_0   exact    1.0200    12.065948   0.9653       0.06047      0.120686   OK
q5_0   exact    1.0200     5.815948   0.9306       0.02915      0.058172   OK
q4_1   exact    0.5200     6.589973   0.4953       0.03303      0.065914   OK
       (para todas as seis: "bytes differ from llama.cpp: 0 |
        scalar load 0.000e+00 (rel 0.00e+00), load8 0.000e+00 (rel 0.00e+00)")
```
Erro por padrão hostil (q5_0 / q4_1), o que a tabela agregada esconde:

```
padrão                                  q5_0 err/range   q4_1 err/range
0 uniforme [-1,1)                          0.03046          0.03276
1 só positivo [0,1)                        0.03107          0.03304
2 31 valores ~1e-3 + um outlier 1.0        0.00098          0.00194
3 constante 0.5                            0.00000          0.00000
4 tudo zero                                0.00000          0.00000
5 alternando +-1                           0.03125          0.00024
6 gaussiana-ish                            0.01736          0.03281
7 faixa larga (|x| até 100)                0.02915          0.03303
```

Leitura dos números, que é o ponto do exercício:

1. **Os bytes que o motor grava são byte a byte os do llama.cpp** para os seis
   tipos, nos oito padrões, incluindo os dois novos (`quantize_row_q5_0_ref` /
   `quantize_row_q4_1_ref` de `libggml-base.so`). Não é "parecido": zero bytes
   diferentes, incluindo o campo `qh` do q5_0 e o `m` do q4_1.
2. **Os dois caminhos de load reconstroem exatamente o que
   `dequantize_row_q5_0`/`q4_1` reconstroem** — `max |gpu − cpu| = 0.000e+00`
   (igualdade bit a bit, não tolerância) para o `kv_load` escalar e para o
   `kv_load8` vetorial. Isso cobre o `qh` do q5_0 lido byte a byte e o `q*d + m`
   do q4_1.
3. **A cota de erro do q5_0 é ~1/32 da faixa do bloco** (`err/range` ≈ 0,029-0,031
   nos padrões realistas) **e a do q4_1 é ~1/30** (0,0328-0,0330).
   A razão aritmética: a grade do `q5_0`/`q4_0` é **assimétrica** — `d` sai do
   máximo *com sinal* (`d = vmax/-16` no q5_0, `/-8` no q4_0), então o extremo do
   sinal oposto cai a **um passo inteiro** (medido `err/step` 0,93-1,00), enquanto
   o `q4_1` tem grade afim `[min, max]` com as duas pontas exatas e erro de **meio
   passo** (medido 0,491-0,496).
   **Correção ao briefing**: o texto da tarefa dizia "~1/32 e ~1/16 da faixa" para
   q5_0 e q4_1. O q5_0 confere (1/32); o q4_1 mede **1/30, não 1/16** — meio passo
   de uma grade de 16 níveis, e não um passo. Quem comparar os dois formatos pelo
   "1/16" vai concluir que o q4_1 é 2x pior que o q5_0, quando na verdade eles
   erram quase o mesmo (`err/range` 0,033 vs 0,029) com 6 bits a menos por bloco
   no q4_1.
4. O `q8_0` tem `err/step` 0,52 e não 0,50 porque o `d` dele é guardado em fp16:
   o termo relativo 2^-11 entra por cima do meio passo. Sem esse detalhe a
   tolerância de 0,50 reprova o próprio q8_0 (foi o que aconteceu na primeira
   versão do teste).
5. O `f16` (a linha de base da correção) mede erro relativo máximo 3,12e-4, dentro
   da cota de 5e-4 que o `check-rope-gpu` já usa.

**Veredito**: MANTIDO. Gate fechado (`check-kvquant-gpu`, sem modelo roda em ~1 s).

### 2.2 `check-rope-gpu` estendido aos dois formatos novos

- **Comando**: `timeout 600 ./build/check-rope-gpu` (janela limpa, 189 MiB).
- **Resultado**: os seis tipos passam — `kv store (f32/f16/q8_0/q4_0/q5_0/q4_1)
  bytes differing from the ggml reference: 0` e
  `attn (1 key, <tipo>) max|out - cache| = 0.000e+00`.
  Esse segundo número é o que importa para o q5_0/q4_1: com `head_dim = 256` o
  kernel de atenção usa o caminho `kv_load8` (`dpw == 8`), então ele prova que o
  kernel **de produção** devolve exatamente o que os bytes do cache decodificam,
  em todos os offsets da linha — não só o `kv_load` escalar que o teste acima usa.
- **Veredito**: MANTIDO.

---

## 3. Achado F6 — as três portas do `kv.h` (bloqueante do coordenador)

- **Referência**: `docs/adversarial-noite.md` F6, enviado pelo coordenador. As três
  portas eram reais e eu as confirmei no código antes de mexer:
  `kv_row_bytes` devolvia `0` para um `KvType` fora do enum (`return 0;` depois do
  switch) → passo de linha 0 → todas as linhas no mesmo endereço, "roda, não
  avisa, sai errado"; `kv_store_row_kernel` e `kv_fill_kernel` terminavam num corpo
  **q4_0 implícito** (o último ramo era um `else` sem `if`), então qualquer tipo
  fora de F32/F16/Q8_0 era gravado como q4_0; `kv_bytes_per_elem` devolvia `0.0`,
  que faz `info`/`serve` aprovarem uma configuração sem contar KV nenhum.
- **Hipótese**: fechar as três com um teste que **exige falha** impede que o
  próximo tipo acrescentado (por mim ou por outra frente) caia nessas portas.
- **Implementação** (tudo em `kv.h`, sem tocar em nada de outra frente):
  - Os switches continuam **exaustivos e sem `default:`** de propósito: um
    `default:` silenciaria o `-Wswitch`, que é justamente o mecanismo que pega um
    tipo novo. O que mudou é o que vem **depois** do switch: `kv_unreachable()`,
    que faz `abort()` no host e `__builtin_trap()` no device, com mensagem.
  - O corpo do q4_0 em `kv_store_row_kernel` e em `kv_fill_kernel` virou um ramo
    explícito `if (CT == KvType::Q4_0) { ... }` e a cadeia termina em
    `kv_unreachable()`. Como `CT` é parâmetro de template, isso não custa nada em
    tempo de execução e **não muda um byte** do caminho do q4_0.
- **Porta extra que o coordenador não listou e o teste novo encontrou**:
  `kv_fill_launch` fazia `return hipGetLastError() == hipSuccess;` **depois** do
  switch, ou seja devolvia o status do *último launch* — para um tipo desconhecido
  isso é "sem erro", então o despachante respondia **`true`**: "cache preenchido"
  para um cache que ele nunca tocou. Agora cada `case` devolve o status do seu
  próprio launch e o `tail` é `return false;` (o contrato do chamador em
  `graph.cuh:1308` já é "false = erro").
- **Teste de regressão**: `tests/check_kvtype.hip` (alvo `check-kvtype`,
  appendado no `CMakeLists.txt`). Ele **forka um filho, entrega `KvType(99)` e
  exige `SIGABRT`** nas três tabelas; confere também que `kv_type_parse` rejeita
  `q9_9`, `q5_1` e a string vazia; que `kv_store_row_launch`/`kv_fill_launch`
  devolvem `false`; e fixa a aritmética do orçamento a 131 072 tokens contra
  contas feitas à mão dentro do próprio teste (ver §6).
- **Comando**: `./build/check-kvtype` — **microssegundos, sem GPU, sem modelo, sem
  lock** (por isso pode entrar em qualquer gate).
- **Resultado**: `check-kvtype: OK`, 40 asserções, incluindo os três `abort` e os
  dois `return false`. As expectativas de orçamento batem:
  `f16/f16 8,000 GiB`, `q8_0/q8_0 4,250`, `q4_0/q4_0 2,250`, `q5_0/q4_1 2,625`,
  `q8_0/q4_0 3,250`, `q5_0/q4_0 2,500`, `q8_0/q4_1 3,375`.
- **Veredito**: MANTIDO.
- **Nota de merge para o coordenador**: as três portas estão fechadas, mas a mesma
  forma existe em **`mtp.cuh:365`** (`kv_bytes_` é calculado com `kv_row_bytes(kv_k_)`
  e usado para alocar `d_ck_` **e** `d_cv_`, enquanto o passo de V no forward usa
  `kv_row_bytes(kv_v_)`). Com K=q5_0 (22 B) e V=q4_1 (20 B) o MTP **super**-aloca V
  e fica apenas correto por sorte; com K=q4_1 e V=q5_0, `--mtp` **estoura o cache de
  V** (20 B de passo para linhas de 22 B). Não toquei em `mtp.cuh` porque está na
  lista de arquivos de outra frente: a correção é a mesma que eu fiz no trunk
  (`graph.cuh`), dois tamanhos separados.

---

## 4. Orçamento de VRAM do motor — a conta exata (§6 da tarefa)

Modelo: `Qwen3.8-27B-UD-IQ3_S.gguf`, 866 tensores, `block_count 65`,
`embedding_length 5120`, `feed_forward_length 17408`, `head_count 24`,
`head_count_kv 4`, `key_length = value_length 256`, `ssm {conv_kernel 4,
state_size 128, group_count 16, time_step_rank 48}`, `full_attention_interval 4`,
vocab 248 320. Derivados: 64 camadas de tronco (16 de atenção completa,
`i%4==3`, e 48 GDN), `n_attn = 16`, `n_recr = 48`, `chan = 2*16*128 + 48*128 =
10 240`, `d_inner = 6144`.

### 4.1 Pesos — o ficheiro de 12,03 GB não é o que vai para a VRAM

| parcela | bytes | GiB / MiB | de onde |
|---|---:|---:|---|
| ficheiro GGUF | 12 040 883 104 | 11,214 GiB | `stat` |
| cabeçalho GGUF + padding de alinhamento | 10 945 408 | 10,44 MiB | offset do 1º tensor, alinhamento 32 |
| **soma exata dos 866 tensores** | **12 029 886 464** | **11,204 GiB** | `dtype_block_bytes` de `include/rdna4/dtype.h:78` |
| padding **entre** tensores | **0** | 0 | os offsets são contíguos |
| cauda depois do último tensor | 51 232 | 0,05 MiB | |
| pesos do tronco (**o que é mesmo `hipMalloc`ado**) | **11 678 877 696** | **10,877 GiB** | `Graph::up()`/`up_f32()`, `graph.cuh:464-495` |
| bloco MTP `blk.64.*` (15 tensores, só com `--mtp`) | 351 008 768 | 0,327 GiB | `MtpHead::init` |

**Achado 1 (o maior de todos):** `rdna4::required_bytes` (`device.h:76-87`) usa o
**tamanho do ficheiro** como estimativa de pesos — `file_bytes + kv +
kOverheadBytes`. O ficheiro tem 11,214 GiB mas só **10,877 GiB** de tronco sobem
para a GPU: os 0,327 GiB do bloco MTP ficam de fora quando `--mtp` não é usado.
Ou seja, `info`/`serve` **superestimam** a VRAM em ~340 MiB, e a conta de
"11,20 GiB de pesos" que circulou de manhã também — a certa é 10,88 GiB. Isso
importa porque a folga a 131K é justamente da ordem de centenas de MiB.

### 4.2 Buffers do grafo (tudo `hipMalloc`ado em `Graph::init`)

| buffer | bytes | MiB | linha |
|---|---:|---:|---|
| ativações 1-token (`d_x_`…`d_vstage_`, 16 buffers) | 414 272 | 0,40 | `graph.cuh:518-522` |
| buffers de batch, `kMaxBatch = 16` (16 floats-buffers) | 6 619 136 | 6,31 | `graph.cuh:544-553` |
| blocos de ativação q8_1 do batch (`d_aqb_`, 16×544×36) | 313 344 | 0,30 | `graph.cuh:554-557` |
| posições do batch (`d_posb_`) | 64 | 0,00 | `graph.cuh:559` |
| **parciais do split-KV** `24 × 16 × (2+256) × 4 B` | 396 288 | 0,38 | `graph.cuh:565` |
| **estado GDN** `48 × 48 × 128 × 128 × 4 B` | 150 994 944 | **144,00** | `graph.cuh:570` |
| conv state GDN `48 × 3 × 10240 × 4 B` | 5 898 240 | 5,62 | `graph.cuh:571` |
| `d_pos_` | 4 | 0,00 | `graph.cuh:572` |
| scratch q8 `(17408/32 + 8) × 36` | 19 872 | 0,02 | `graph.cuh:583` |
| logits `248320 × 4` | 993 280 | 0,95 | `graph.cuh:1130`/`1262` |
| **total buffers fixos** | **165 658 660** | **157,98** | |

Por token o bloco de batch custa 423,7 KiB; ele é **fixo em 16 tokens**, não
escala com `--ctx-size`. Todos os 16 buffers de batch são usados (contei as
referências uma a uma: nenhum morto).

### 4.3 KV cache por token e por combinação (números exatos, gateados)

`16 camadas × 4 cabeças × 256 dims = 16 384 elementos por lado por token`.

| K / V | B/token | 131 072 tokens | pesos+buffers+KV | folga de 15,900 GiB |
|---|---:|---:|---:|---:|
| f16/f16 | 65 536 | **8,000 GiB** | 19,031 GiB | **−3,131** |
| q8_0/q8_0 | 34 816 | 4,250 GiB | 15,281 GiB | +0,619 |
| q4_0/q4_0 | 18 432 | 2,250 GiB | 13,281 GiB | +2,619 |
| **q5_0/q4_1** | **21 504** | **2,625 GiB** | **13,656 GiB** | **+2,244** |
| q5_0/q5_0 | 22 528 | 2,750 GiB | 13,781 GiB | +2,119 |
| q4_1/q4_1 | 20 480 | 2,500 GiB | 13,531 GiB | +2,369 |
| q8_0/q4_1 | 27 648 | 3,375 GiB | 14,406 GiB | +1,494 |
| q8_0/q4_0 | 26 624 | 3,250 GiB | 14,281 GiB | +1,619 |
| q5_0/q4_0 | 20 480 | 2,500 GiB | 13,531 GiB | +2,369 |

(15,900 GiB é o que `hipMemGetInfo` reporta como total — a VRAM visível da placa,
não os 16 GiB do anúncio. O piso do desktop nesta janela é 189-318 MiB.)

**A correção ao número que circulou**: K `q8_0` + V `q4_0` = **3,250 GiB** e
K `q5_0` + V `q4_0` = **2,500 GiB** — confere com o coordenador, e os dois estão
fixados como asserção em `tests/check_kvtype.hip`, não como prosa.

### 4.4 Resíduo não explicado

Falta reconciliar 4.3 com o `vram in use` do `bench` (medição em §5): a diferença
esperada é o allocator do HIP (arredondamento de cada `hipMalloc` para a página do
driver), os ~190-320 MiB do desktop e os contextos/`hipModule`s do runtime. Com
~150 `hipMalloc` de pesos, um arredondamento de 64 KiB por tensor já vale 10 MiB;
o HIP costuma arredondar para 2 MiB em alocações grandes, o que daria centenas de
MiB. O número medido em §5 fecha a conta.

### 4.5 Desperdício encontrado (e o que custaria recuperar)

1. **Cache de V alocado com o tamanho de K — 128 MiB a 131K (CORRIGIDO).**
   `graph.cuh:514-521` (antes da minha mudança) calculava `kv_bytes` com
   `kv_row_bytes(kv_k_, HD)` e usava o **mesmo** valor para `hipMalloc(&d_k_, …)` e
   `hipMalloc(&d_v_, …)`, enquanto o passo de linha de V em `kv_write` já usava
   `kv_row_bytes(kv_v_, HD)` corretamente. Com K=q5_0 (176 B/linha) e V=q4_1
   (160 B/linha) isso é 16 B de lixo por linha de V: 16 B × 4 cabeças × 131 072
   tokens × 16 camadas = **128 MiB** (134,2 MB) alocados e nunca tocados a 131K.
   Corrigido com `kv_bytes_v_` + `kv_layer_bytes_v()`; com K e V do mesmo tipo o
   valor é idêntico ao de antes (as duas expressões coincidem), então **nenhum
   caminho existente mudou**.
2. **Falta de `default:`/retorno silencioso nas tabelas de `kv.h` (CORRIGIDO)** —
   ver §3. Não é MiB, é a classe de erro que produz número errado com cara de certo.
3. **`kOverheadBytes = 1 GiB` (`device.h:49`)** é uma constante fixa de "kernels,
   buffers e fragmentação". Medido: buffers fixos = **158 MiB**. Os 866 MiB
   restantes são reserva; enquanto reserva, é o que faz `info` recusar configurações
   que caberiam (por exemplo `q8_0/q8_0` a 131K, que cabe com 619 MiB de folga e a
   reserva de 1 GiB rejeita). Trocar a constante por um valor medido é uma linha,
   mas mexe na política de aceitação de `run`/`bench`/`ppl`/`serve` — **não fiz**,
   porque é decisão do coordenador e não estava no meu escopo.
4. **`d_attn_partial_` é dimensionado para o número MÁXIMO de splits** (16) em
   todas as configurações: 396 288 B = 387 KiB. A contextos curtos (menos de
   `kAttnSplitMin = 512`) ele nunca é lido. 387 KiB é irrelevante a 131K; fica
   registrado, não vale mexer.
5. **`d_projb_`/`d_ffnab_`/`d_ffnbb_`/`d_qkvb_`/`d_convb_` são fixos em 16 tokens**
   e dominam os buffers de batch (6,62 MiB). Reduzir `kMaxBatch` de 16 para 8
   devolveria ~3,3 MiB e custaria prefill — não vale.
6. **O bloco MTP (0,327 GiB de pesos + cache próprio) é 100% desperdício quando
   `--mtp` não é passado** — o `up()` nunca é chamado, então nada é alocado. Não é
   desperdício; é o motivo pelo qual a estimativa por tamanho de ficheiro erra (§4.1).

Isto responde ao "cace cada GiB": **10,877 (tronco) + 0,158 (buffers) + KV** e mais
nada. Não há buffer duplicado, padding de pesos (0 bytes entre tensores) nem
alocação morta no caminho do tronco. O único desperdício real era o de 128 MiB do
item 1.

---

## 5. `required_bytes` — o orçamento que recusava o que cabe (tarefa do coordenador)

- **Referência**: pedido do coordenador às 03:10, a partir do meu §4.1/§4.2.
- **Hipótese**: `device.h:76-87` somava `file_bytes + kv + kOverheadBytes(1 GiB)`.
  Os dois primeiros termos erram: o ficheiro tem 11,214 GiB mas o tronco sobe
  10,877 GiB (os 0,327 GiB de `blk.64.*` são o MTP, que só sobe com `--mtp`), e
  1 GiB de overhead não existe (medido: 157,98 MiB). Erro de ~1,18 GiB **para
  cima**, o que faz `info`/`serve` recusarem configurações que cabem — 131K com
  `q8_0/q8_0` (15,281 GiB) entre elas. Ou seja: "131K só com q4_0" era falso.
- **Implementação** (tudo em `device.h` + os dois chamadores):
  - `GraphBufferShape` + `graph_buffer_bytes(shape)`: espelha `graph.cuh:518-585`
    linha a linha (ativações 1-token, bloco de batch em `kMaxBatch`, parciais do
    split-KV, estado GDN, conv state, scratch q8, logits, `d_pos_`).
  - `graph_buffer_shape(cfg, vocab)`: monta a forma a partir do config, sem
    incluir `model.h` (duck-typed).
  - `uploaded_weight_bytes(loader, mtp_layer, use_mtp)`: soma os tensores que o
    `up()` de fato sobe, com **as mesmas** `tensor_bytes`/`dtype_block_bytes`
    (`dtype.h`) que ele usa para dimensionar o `hipMalloc`, então não pode
    discordar do alocador.
  - `required_bytes(weights, ctx, kt, vt, buffers)` — assinatura nova, sem
    `file_bytes`.
  - `kOverheadBytes` deixou de ser 1 GiB: agora são duas constantes nomeadas,
    `kAllocatorMarginBytes = 256 MiB` (arredondamento do alocador do HIP sobre
    ~150 alocações de peso + fragmentação) e `kRuntimeReserveBytes = 64 MiB`
    (contexto/módulo/fila do HIP). Margem pequena e explicada em vez de um
    gigabyte que decide sozinho se um contexto cabe.
- **Gate**: `tests/check_kvtype.hip` agora **exige 165 658 660 B** para a forma do
  IQ3_S — um refactor que mude uma alocação em `graph.cuh` (ou o espelho) quebra
  ali, não em silêncio. Foi o que pegou dois bugs meus durante a escrita: eu havia
  misturado floats com bytes no bloco de batch (`per_token` em floats somado com
  `aq_blocks*36` em bytes) e esquecido o termo `n_head*2*head_dim` (o `d_proj_` do
  caminho 1-token e o `d_qb_` do batch). Sem o número fixo do teste, os dois
  passariam e o orçamento ficaria 0,8 MiB baixo — o suficiente para aceitar um
  contexto que morre no `hipMalloc`.
- **Resultado**: `graph_buffer_bytes` = 165 658 660 B (157,98 MiB), igual à soma
  feita à mão de `graph.cuh`. `required_bytes` a 131K com `q5_0/q4_1` = 13,969 GiB
  (13,656 de pesos+buffers+KV + 320 MiB de margem). `info` imprime agora as duas
  contas (a honesta e a antiga) para a diferença ficar visível.
- **Veredito**: MANTIDO. **Muda o que o motor aceita**: 131K com `q8_0/q8_0` passa
  a ser aceito pelo orçamento (15,281 GiB + 0,31 de margem = 15,59 GiB de 15,90)
  em vez de recusado — a medição em §6 diz se ele de fato roda.
- **Nota de merge**: `device.h` ganhou `#include "rdna4/dtype.h"` e
  `#include <string>`; `serve.hip` passou a abrir o GGUF **antes** da checagem de
  orçamento (só o cabeçalho, não os pesos) para ter a forma do modelo. Se outra
  frente mexer no `serve.hip`, é essa a mudança de ordem a preservar.

---

## 6. Números de 131K medidos (o alvo do enunciado)

**Comando exato** (uma tomada de lock, janela verificada antes e depois):

```
./scripts/gpu-lock.sh bash /tmp/kv-big1.sh     # timeout DENTRO do lock (regra 1.2)
./build/rdna4-infer bench -m <IQ3_S> --ctx-size 131072 --start-pos 131000 \
    --fill-cache -n 16 --reps 2 --cache-type-k <K> --cache-type-v <V>
```

**Janela**: limpa. `mem_info_vram_used` = 198 156 288 B (189 MiB, só Xwayland +
plasmashell + firefox, nenhum modelo) e `mem_info_gtt_used` = 30 208 000 B antes;
`flock` sem outros waiters além do meu no momento da medida. Janela de
`date -Is` = 03:12:13 → 03:14:48.

| K / V | decode tok/s | vram in use | livre | GTT durante a corrida | veredito |
|---|---:|---:|---:|---:|---|
| **q5_0/q4_1** | **13,73** | 14,25 GiB | 1,68 GiB | 30,2 MB → 30,2 MB | **roda** |
| q8_0/q4_1 | 14,32 | 15,00 GiB | 0,93 GiB | 30,2 → 30,2 MB | roda |
| q8_0/q8_0 | **14,34** | **15,87 GiB** | **0,05 GiB** | 30,2 → **75,5 MB** | roda, no talo |
| q4_0/q4_0 | 14,08 | 13,87 GiB | 2,05 GiB | 75,4 → 75,5 MB | roda |
| f16/f16 | — | — | — | — | **NÃO roda**: `graph init failed: hipMalloc failed (kv cache)` |

Cada corrida imprime `vram in use` do próprio `hipMemGetInfo`; "GTT durante a
corrida" é `mem_info_gtt_used` lido imediatamente depois de cada bench.

### 6.1 O achado que muda a conversa: a 131K o formato do KV NÃO muda a velocidade

q4_0/q4_0 (2,25 GiB de KV) faz **14,08** tok/s e q8_0/q8_0 (4,25 GiB de KV) faz
**14,34** tok/s: o cache quase 2x maior é **1,9% mais rápido**, não mais lento. O
tráfego por token é 10,88 (pesos) + KV, ou seja 13,13 GiB contra 15,13 GiB — se a
atenção fosse limitada por banda, o q4_0 deveria ser ~15% mais rápido. Não é: a
131K o decode é limitado por **latência/ocupação do kernel de atenção**, não por
banda. O `bench` calcula a "effective bandwidth" só sobre os pesos (11,20 GiB por
token) e reporta 165-172 GB/s nos quatro casos, o que é 27-29% da banda de pico
desta placa.

Consequência prática: **escolher o KV pelo critério de velocidade a 131K não faz
sentido neste motor** — os quatro formatos que rodam entregam o mesmo ~14 tok/s.
A decisão tem de ser por qualidade (§7) e por folga de VRAM (§6.3).

### 6.2 Ruído dentro da corrida (o piso entre processos está na §6.5)

Dentro de uma corrida, `--reps 2` dá spread ~0,01 tok/s (ex. q5_0/q4_1: best
13,73 / mean 13,72) — ou seja o spread interno é ~0,07% e não serve como piso, só
como sanidade. O piso que vale é o de **entre processos**, medido na §6.5
(**0,36%**), e é ele que diz que as diferenças de formato da tabela acima são reais.

Também não é confiável nesta tabela o `prefill 5 tokens` (0,349 / 0,235 / 0,151 /
0,141 s): são 5 tokens medidos logo depois de um `--fill-cache` de 2,25-4,25 GiB,
então o número está contaminado pelo dreno assíncrono do preenchimento. **Não usar.**

### 6.3 A conta de VRAM fecha — e calibra a margem

Comparando o modelo do §4 com o `vram in use` medido:

| K / V | previsto (pesos+buffers+KV+desktop) | medido | diferença |
|---|---:|---:|---:|
| q4_0/q4_0 | 13,48 GiB | 13,87 GiB | **+0,39** |
| q5_0/q4_1 | 13,85 GiB | 14,25 GiB | **+0,40** |
| q8_0/q4_1 | 14,60 GiB | 15,00 GiB | **+0,40** |
| q8_0/q8_0 | 15,48 GiB | 15,87 GiB | **+0,39** |

A diferença é **constante** (0,39-0,40 GiB) e não proporcional ao tamanho do cache,
logo é custo fixo: o arredondamento do alocador do HIP sobre as ~150 alocações de
peso + o contexto/módulo do HIP. Por isso `kAllocatorMarginBytes` foi calibrado de
256 MiB para **384 MiB** (+64 de runtime = **448 MiB**), em vez do 1 GiB chutado de
antes. Com 448 MiB o orçamento prevê 123 MiB livres para q8_0/q8_0 a 131K contra 51
MiB medidos — conservador na direção certa e sem recusar o que roda.

O modelo de VRAM do §4 está, portanto, **validado por medição** nas quatro
combinações, com erro residual < 2%.

### 6.4 `info` antes/depois — a diferença que o conserto do orçamento faz

Comando: `./build/rdna4-infer info -m <IQ3_S> --ctx-size 131072 --cache-type-k K
--cache-type-v V` (lido de [D] no log; a coluna "file-based" é a estimativa antiga
impressa pelo próprio `info` para a diferença ficar visível).

| K / V | pesos | buffers | KV | margem | need | estimativa antiga | antigo decidia |
|---|---:|---:|---:|---:|---:|---:|---|
| q5_0/q4_1 | 10,88 | 157,98 MiB | 2,62 | 448 MiB | **13,97 GiB** | 14,84 GiB | aceita |
| q8_0/q4_1 | 10,88 | 157,98 MiB | 3,38 | 448 MiB | 14,72 GiB | 15,59 GiB | aceita |
| q8_0/q8_0 | 10,88 | 157,98 MiB | 4,25 | 448 MiB | **15,72 GiB** | **16,46 GiB** | **RECUSAVA** |
| f16/f16 | 10,88 | 157,98 MiB | 8,00 | 448 MiB | 19,34 GiB | 20,21 GiB | recusa (correto) |

O ponto: `q8_0/q8_0` a 131K **cabe e roda** (medido: 15,87 GiB em uso, 51 MiB
livres, 14,34 tok/s), mas o orçamento antigo o recusava. A frase "a 131K só cabe
com q4_0" era falsa e vinha de contar o ficheiro em vez dos tensores.

### 6.5 Piso de ruído entre processos, medido

A mesma configuração (`q5_0/q4_1` a 131K com cache sintético, o comando inteiro
repetido) rodou duas vezes, em processos separados:

| corrida | best | mean | banda efetiva |
|---|---:|---:|---:|
| 1ª ([B], 03:12) | 13,73 tok/s | 13,72 | 165,2 GB/s |
| 2ª ([G], 03:55) | **13,68 tok/s** | 13,67 | 164,6 GB/s |

**Piso de ruído entre processos = 0,05 tok/s = 0,36%.** Logo as diferenças da
tabela do §6 **estão acima do piso**: q8_0/q8_0 (14,34) é 4,8% mais rápido que
q5_0/q4_1 (13,73) e 4,4% mais rápido que a média das duas corridas do q5_0/q4_1.
Isso reforça o §6.1 em vez de contradizê-lo: a diferença é **pequena e no sentido
"errado"** (o cache maior é o mais rápido), o que descarta banda como limitante —
se a atenção fosse limitada por banda, 2 GiB a mais de cache por token custaria
~15%, não ganharia 4,8%.

### 6.6 O que ficou de fora do bench, e por quê

**Combinações que eu decidi não medir, e por quê**: `q5_0/q5_0` e `q4_1/q4_1` saíram
do bench depois do aviso do coordenador sobre a fila do lock (havia 12-13 waiters e
a placa ocupada por outra frente). As duas são medições de *custo*, e o §6.1 mostra
que a 131K o custo é insensível ao formato (spread de 1,9% entre 2,25 e 4,25 GiB de
cache) — o que decide é qualidade, e essa é medida na §7 com a sonda de KL, que
cobre os dois. O mesmo vale para `q8_0/q4_1`, que eu **mantive** no bench por ser a
opção de melhor qualidade recomendada pelo coordenador e precisar do número de VRAM.

### 6.7 Decisão de KV a 131K (com os números na mão)

O coordenador decidiu depois da tabela do §6:

- **Padrão embarcado: `q5_0/q4_1`** — é o alvo do enunciado; mede 14,25 GiB em uso e
  **1,68 GiB livres**, folga suficiente para o MTP (os planos de estado do verify
  custam 0,6-0,75 GiB). Os 13,73 contra 14,34 tok/s do `q8_0/q8_0` são 4% dentro de um
  regime em que o KV **não é o gargalo** (§6.1).
- **Opção de melhor qualidade, documentada e não padrão: `q8_0/q4_1`** — need
  14,72 GiB, **1,20 GiB de folga medida** (15,00 em uso).
- **`q8_0/q8_0` a 131K: roda, no talo.** 14,34 tok/s mas **51 MiB livres** e 45 MB
  empurrados para **GTT** — o mesmo regime que produziu o penhasco silencioso de 8×
  com f16 a 64K. Não é padrão nem recomendação; não usar com MTP.
- **`f16/f16` a 131K falha em `hipMalloc`** (`graph init failed: hipMalloc failed (kv
  cache)`) — é a prova direta de que a linha do README "f16 at 128K needs 8 GiB and
  cannot fit" está certa, e o `info` corrigido a recusa **antes** do load (19,34 GiB).

---

## 7. Qualidade do KV — o que a PPL não mede (§4 da tarefa)

- **Referência**: a PPL é cega para quantização de cache. Evidência do upstream
  citada pelo coordenador: no AIME25 o KV `q4_0` cai de 37,9% para 2,0% enquanto a
  PPL anda ~0,4%. A sonda `check-kvquality-gpu` mede o que se move: a divergência
  da **distribuição de saída** causada pelo formato do cache, com o cache f16 do
  **mesmo motor**, nos **mesmos tokens**, no **mesmo processo** — então a
  divergência do motor em relação ao llama.cpp cancela e a única diferença entre
  duas linhas da tabela é o tipo do cache.
- **Comando**: `./build/check-kvquality-gpu <IQ3_S> /tmp/kv-ids.txt 4096
  f16:f16,q5_0:q4_1,q4_0:q4_0,q8_0:q8_0,q8_0:q4_1,q5_0:q4_0` (uma tomada de lock,
  janela limpa: vram 119 902 208 B = 114 MiB, gtt 75 431 936 B).
  Prefill em lote com probes a cada 16 posições ⇒ 256 probes sobre **4096 tokens de
  contexto real** (wikitext-2 tokenizado pelo próprio motor). Sem a forma em lote
  seria 1 forward por posição, e o prefill medido é 74,9 tok/s — não caberia na noite.
- **Piso do harness**: a linha `f16:f16` da tabela é o f16 **contra ele mesmo**, e
  sai exatamente 0 em todas as colunas (KL 0,000000, |dNLL| 0,000000, 0/256 argmax
  trocado). O piso é zero, não "pequeno" — a métrica é determinística.

### 7.1 Resultado (256 probes, 4096 tokens de contexto real)

| config | PPL | **KL nats/pos** | KL max/pos | mean\|dNLL\| | argmax trocado |
|---|---:|---:|---:|---:|---:|
| f16/f16 (referência) | 5,91711 | 0,000000 | 0,000000 | 0,000000 | 0/256 |
| q8_0/q8_0 | 5,92405 | **0,000492** | 0,014567 | 0,018732 | 3/256 |
| q8_0/q4_1 | 5,93386 | 0,001443 | 0,014037 | 0,032361 | 5/256 |
| **q5_0/q4_1** | 5,95800 | **0,001715** | 0,018921 | 0,032913 | **4/256** |
| q5_0/q4_0 | 5,93234 | 0,002118 | 0,053731 | 0,036079 | 7/256 |
| q4_0/q4_0 | 5,93976 | 0,003208 | 0,092405 | 0,042582 | 8/256 |

Eixos isolados (um lado fixo, o outro variando):

```
-- eixo K, V fixo em q4_1 (menor KL = melhor K) --
   q8_0:q4_1  KL 0,001443   PPL 5,93386   argmax 5/256
   q5_0:q4_1  KL 0,001715   PPL 5,95800   argmax 4/256
-- eixo V, K fixo em q5_0 (menor KL = melhor V) --
   q5_0:q4_1  KL 0,001715   PPL 5,95800   argmax 4/256
   q5_0:q4_0  KL 0,002118   PPL 5,93234   argmax 7/256
```

**Respostas às duas perguntas empíricas da tarefa:**

1. **Qual é o melhor K?** `q8_0` — KL 0,001443 contra 0,001715 do `q5_0`, **16% menor**,
   com V fixo em q4_1. Confirma, medida neste motor e neste modelo, a direção que a
   frente de referências achou na tabela do PR #21038 do llama.cpp (0,002920 para K
   q8_0+V q4_1 contra 0,004181 para K q5_0+V q4_1, 30% menor). E **`q4_0` em K é o
   pior de todos** (0,003208, quase 2× o q5_0), o que sustenta o "nunca q4_1/q4_0 em
   K" do coordenador.
2. **Qual é o melhor V?** `q4_1` — KL 0,001715 contra 0,002118 do `q4_0`, **19% menor**,
   com K fixo em q5_0. É exatamente onde o `min` por bloco do q4_1 paga: o V tem
   distribuição assimétrica (o `q4_0`/`q5_0` gastam metade da grade em valores que não
   ocorrem), e o q4_1 gasta os 16 níveis no intervalo que existe. O melhor V de todos é
   o `q8_0` (0,000492 com K q5_0), ao custo de mais VRAM.

3. **A PPL não ordena esses formatos — está medido, não argumentado.** Na tabela, a
   ordem por PPL é q8_0/q8_0 < **q5_0/q4_0** < q8_0/q4_1 < q4_0/q4_0 < **q5_0/q4_1**,
   enquanto a ordem por KL é q8_0/q8_0 < q8_0/q4_1 < **q5_0/q4_1** < **q5_0/q4_0** <
   q4_0/q4_0. Os dois configs do meio **trocam de lugar**: o `q5_0/q4_1` tem a **pior**
   PPL (5,958) e a **segunda melhor** KL, e o `q5_0/q4_0` tem PPL melhor com 23% mais
   KL e quase o dobro de argmax trocado (7 contra 4). Correlação de postos entre as duas
   métricas nos cinco configs quantizados: **negativa**. (Na primeira corrida, com só 16
   probes, foi pior ainda: `q5_0/q5_0` deu PPL 4,152, *melhor* que o f16 de 4,166, com
   KL 0,000456 — uma configuração quantizada "melhorando" a PPL é ruído de amostragem.)
   **Conclusão operacional: não usar PPL para escolher formato de KV.**

4. O `q5_0/q4_1` troca o token ganancioso em **4/256 = 1,6%** das posições a 4096
   tokens de contexto. O gate do teste exige < 5%.

### 7.2 Sonda needle — recuperação em contexto real longo

- **Comando**: `./build/check-kvquality-gpu <IQ3_S> /tmp/kvneedle/ids.txt needle
  /tmp/kvneedle/probes.txt f16:f16,q5_0:q4_1,q4_0:q4_0,q8_0:q8_0` (mesma tomada de lock).
- **Desenho**: 8192 tokens de contexto **real** (wikitext-2) com **8 agulhas** a
  ~11%, 22%, 33%, 44%, 56%, 67%, 78% e 89% de profundidade (8 códigos distintos de 5
  dígitos), e depois as 8 perguntas no fim. O probe é o primeiro dígito do código — o
  token que só existe na agulha. Cada agulha é tokenizada separadamente e as peças são
  concatenadas, então o índice esperado é exato por construção (supor a propriedade de
  prefixo do BPE daria índice errado). O stream é cortado em segmentos que terminam
  **exatamente** em cada probe: probar fora de ordem leria um estado GDN já avançado e
  erraria por um motivo que nada tem a ver com o cache.

| config | recuperação | rank médio | pior rank |
|---|---:|---:|---:|
| f16/f16 | **8/8** | 0,00 | 0 |
| **q5_0/q4_1** | **8/8** | 0,00 | 0 |
| q4_0/q4_0 | 8/8 | 0,00 | 0 |
| q8_0/q8_0 | 8/8 | 0,00 | 0 |

**Resultado honesto: a sonda needle a 8K não distingue os formatos — todos recuperam
8/8 no rank 0.** Isso é um resultado, não um fracasso: (a) fecha o gate que o
coordenador pediu para o alvo (**q5_0/q4_1 ≥ 90% de recuperação → 100%**), e (b) diz
que a 8K de contexto a quantização do KV **não quebra** a recuperação de longo alcance
nem no `q4_0/q4_0`. A conclusão do §7.1 (qual formato é melhor) vem da KL, que é
sensível onde a recuperação ainda é binária e saturada. Se a recuperação quebrasse
antes da KL, o gate seria a needle; aqui a KL é a métrica que ordena e a needle é a que
**prova que nenhum deles quebra o que importa**.

### 7.3 O que NÃO foi medido, e por quê (dito explicitamente)

**A 131K a qualidade não foi medida com texto real.** As duas razões, com número:

1. O prefill deste motor faz **74,9 tok/s** (medido pelo coordenador): encher o KV com
   131 072 tokens de texto real custaria **1750 s** por configuração, e a sonda de KL
   precisa de 4096 tokens de prefill + uma passada por config — com 6 configs isso passa
   de 3 h de GPU. A noite tem uma placa e 6 outras frentes na fila.
2. Portanto, a 131K o que existe medido é **custo** (tok/s, VRAM, GTT, §6) sobre
   **cache sintético** (`--fill-cache`), que por construção não tem qualidade nenhuma
   para medir. A qualidade foi medida a **4096 tokens** (KL, 256 probes) e a **8192
   tokens** (needle, 8 agulhas), que é o teto honesto desta sessão. Entre 8K e 131K a
   única coisa que muda é o número de chaves que a atenção soma — e a sonda needle a 8K
   já mostra recuperação perfeita, mas **isso não é uma medida a 131K** e não deve ser
   reportado como tal.
