# Estudo: o GDN em blocos do NInfer — especificação de port

Fonte primária: `/tmp/ninfer` HEAD `d492968`. A forma em blocos está em
`ninfer src/ops/linear_attention/gated_delta_net/chunked/` (não em `src/ops/gated_delta_net/`).
Nosso lado: `nosso include/rdna4/gdn.cuh`, `nosso include/rdna4/graph.cuh`,
`nosso docs/estudo-prefill-c-nosso.md` §5, `nosso docs/plano-prefill.md` §1b.
Terceira fonte para desempate: llama.cpp `41abbfd599fbdd3470fcae0a1fb6530ad8403cd7`
(grafos $O(\text{CS})$ de `src/models/delta-net-base.cpp`) — citado como `llama.cpp <path>:<line>`.

Abreviações de citação neste documento: `ninfer .../chunked/X` = `ninfer src/ops/linear_attention/gated_delta_net/chunked/X`;
`ninfer .../X` = `ninfer src/ops/linear_attention/gated_delta_net/X`; `nosso X` = `nosso <raiz>/X`
(raiz = `/home/marcelo/Projetos/rdna4-infer`).

Sem GPU. Tudo abaixo é leitura de código e aritmética; onde eu medi, eu digo o que medi.

---

## TL;DR

1. **A forma em blocos é EXATA, não é aproximação.** Protótipo em `float64` (aritmética exata
   para os fins desta pergunta), nossa recorrência sequencial contra a forma em blocos transcrita
   termo a termo: desvio relativo máximo **1,0e-15 … 1,7e-15** (máximo sobre saída **e** estado),
   para $B \in \{8,16,32,64\}$ e $T \in \{64,256,1024,2048,8192\}$; em $T=128$ com 5 sementes, a
   saída fica em 1,5e-16 … 2,9e-16 e o estado em 4,4e-16 … 1,6e-15. Só há reordenação de
   somas e arredondamento de `float`.
2. **Em `float32` puro (nosso caminho, sem BF16 em nada)** o desvio medido foi **5,0e-7 (T=64) …
   1,0e-6 (T=8192)** — a mesma ordem do erro que o **nosso próprio** caminho sequencial já tem
   contra `float64` (3,1e-7 … 4,4e-7). O caminho em blocos não é menos exato que o sequencial.
3. **Os dtypes do ninfer NÃO podem ser copiados.** `W`, `U`, `v_new` e `h_chunk` são **BF16**
   (`ninfer .../chunked/launch.h:38-43`) e as MMAs são BF16/TF32. Emulando só essa escolha de
   dtype, com o resto igual: desvio **3,0e-3 … 4,9e-3** — 5 000× pior. O gate declarado deles
   para o próprio caminho é `relative_l2` 4,1e-3 (saída BF16) e 2,7e-3 (estado FP32)
   (`ninfer tests/ops/test_gated_delta_net.cpp:25-33`), coerente com isso.
4. **A inversa nova não amplifica.** $\mathrm{cond}(I-A) \le 6{,}2$ e $|T|_{\max} = 1{,}00$ em todos
   os casos de estresse que rodei (chaves correlacionadas até posto 1, $\beta \to 1$). Não há
   amplificação de erro introduzida pela transformada.
5. **Três convenções derrubam a equivalência se erradas** (medido, desvio relativo contra o nosso
   sequencial): cumsum **exclusivo** em vez de inclusivo = **0,275**; diagonal da saída **estrita**
   em vez de inclusiva = **0,824**; as duas = **0,848**. Cada uma vira um teste unitário obrigatório.
6. **A consequência de gate é grande e é uma decisão, não um detalhe**: o caminho em blocos **não
   pode** ser bit-exato contra o caminho por token. Hoje isso é contrato
   (`nosso include/rdna4/gdn.cuh:295-301`, assegurado por `nosso tests/check_batch_gpu.hip:163-169`
   e `:203-204`, que exigem `BIT-EXACT`). Ou o switch fica desligado por padrão, ou aquele teste
   passa a tolerância declarada. Ver §4.3.
7. **Alvo**: `gdn_delta` = 0,625 ms/token e a recorrência inteira 0,852 ms/token
   (`nosso docs/estudo-prefill-c-nosso.md:501-503`), medidos a N=64 com chunk 16. Hoje isso é
   `0,625/48 = 13,0 µs` por camada por token = **208 µs por camada por chamada de 16 tokens** para
   37,7 M fma — ~1,1 % do pico de fma da placa, e 20× acima do piso de memória (6,3 MB de tráfego
   de estado por camada por chamada ≈ 10 µs a 633 GB/s). O ganho tem que vir de ILP/ocupação.

---

## 1. A nossa recorrência, termo a termo

### 1.1 O kernel

`nosso include/rdna4/gdn.cuh:57-87` (`delta_rule_kernel`) e a variante em lote
`nosso include/rdna4/gdn.cuh:363-444` (`delta_rule_batch_rows_kernel`), que é o caminho de
produção do prefill. Para um value head $h$, key head $kh = h \bmod n_{kh}$
(`nosso include/rdna4/gdn.cuh:65,374`), dim de estado $S=128$, token $t$:

- $a_t = \texttt{expf}(g_t)$ — `nosso include/rdna4/gdn.cuh:71` (lote: `nosso include/rdna4/gdn.cuh:383`). $g$ é **log-gate**,
  não gate: quem monta é `nosso include/rdna4/graph.cuh:1293-1298`
  ($g = \mathrm{softplus}(\alpha + \texttt{ssm\_dt}) \cdot \texttt{ssm\_a}$), layout $[n][n_{vh}]$ fp32
  (`nosso include/rdna4/graph.cuh:419,671`).
- $\beta_t = \sigma(b_t) \in (0,1)$ já pós-sigmoid na entrada do kernel
  (`nosso include/rdna4/graph.cuh:1293` → `:1315`), layout $[n][n_{vh}]$ fp32.
- $q_t, k_t$ são as linhas **já normalizadas em L2** (`nosso include/rdna4/graph.cuh:1310`,
  `l2_norm_launch` sobre $n \cdot 2 n_{kh}$ linhas de $S$, eps = `rms_norm_eps`) — logo
  $\lVert k_t \rVert_2 = 1$ e $q_t$ idem.

Estado: $M \in \mathbb{R}^{d_v \times S}$ **transposto**,
$M[j][i] = S[i][j]$ (`nosso include/rdna4/gdn.cuh:8-10`), $j$ = dim de valor, $i$ = dim de chave.
Por token, por linha $j$ (`nosso include/rdna4/gdn.cuh:74-86`):

$$
\begin{aligned}
\tilde M[j][\cdot] &= a_t \, M[j][\cdot] &&\text{(passo 1, 128 multiplicações)}\\
s_j &= \sum_{i=0}^{S-1} \tilde M[j][i]\, k_t[i] &&\text{fma ascendente, acumulador único}\\
\delta_j &= \beta_t\,(v_t[j] - s_j)\\
M[j][i] &\leftarrow \tilde M[j][i] + k_t[i]\,\delta_j &&\text{(fma, elemento a elemento)}\\
o_t[j] &= \Bigl(\sum_{i=0}^{S-1} M[j][i]\, q_t[i]\Bigr)\cdot \frac{1}{\sqrt S}
\end{aligned}
$$

Duas observações que importam para a definição do gate:

- a saída usa o estado **já atualizado** (o `acc` é acumulado depois do `fmaf` de atualização,
  `nosso include/rdna4/gdn.cuh:409-414`);
- $1/\sqrt{S}$ = `rsqrtf`, e o ninfer usa exatamente a mesma escala (`ninfer
  src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp:65`, `1/sqrt(kStateDim)`).

### 1.2 Em notação matricial (dim de chave $\times$ dim de valor)

Com $S_t \in \mathbb{R}^{S \times S}$ (transposta de $M$, $S_t = M_t^{\top}$), $k_t, v_t$ como
colunas:

$$
S_t \;=\; a_t\bigl(I - \beta_t k_t k_t^{\top}\bigr)\,S_{t-1} \;+\; \beta_t\, k_t v_t^{\top},
\qquad
o_t \;=\; \frac{1}{\sqrt S}\, S_t^{\top} q_t
$$

**Derivação** (não assumida, tirada do código): $\delta_t = \beta_t(v_t - a_t S_{t-1}^{\top}k_t)$ e
$k_t \delta_t^{\top} = \beta_t k_t v_t^{\top} - \beta_t k_t k_t^{\top}(a_t S_{t-1})$, logo
$S_t = a_t S_{t-1} - a_t\beta_t k_tk_t^{\top}S_{t-1} + \beta_t k_tv_t^{\top}$ e o fator
$(I-\beta_tk_tk_t^{\top})$ multiplica **pela esquerda**.

$k_t$ unitário e $\beta_t \in (0,1)$ dão autovalores $\{1, 1-\beta_t\}$: o operador é
**não-expansivo** e a contração $a_t \le 1$. É isso que garante que erro pequeno por bloco não
cresce ao longo da sequência — e é o que a medição de desvio confirma (§3.3: 1,0e-6 em 128 blocos,
sem crescimento).

### 1.3 O que fica fora da recorrência (e por que)

| etapa | kernel nosso | muda com o port? |
|---|---|---|
| conv1d causal + silu, estado de $K-1$ taps | `nosso include/rdna4/gdn.cuh:315-335` | **não**: ponto a ponto / por tap, mantém a ordem → segue bit-exato |
| L2-norm de q/k | `nosso include/rdna4/nn.cuh:69-101`, chamado em `nosso include/rdna4/graph.cuh:1310` | **não** |
| $\sigma(\beta)$, $\mathrm{softplus}(\alpha+dt)\cdot A$ | `nosso include/rdna4/graph.cuh:1293-1298` | **não** |
| recorrência | `nosso include/rdna4/gdn.cuh:363-444` | **sim — é o alvo** |
| rms_norm (ssm_norm) + silu·mul | `nosso include/rdna4/graph.cuh:1321-1327` | **não** |

Ou seja: do bucket `gdn_delta` 0,625 sobra o alvo; `gdn_conv` 0,051, `gdn_l2norm` 0,072 e
`gdn_scalars` 0,052 só entram se a gente **fundir** as etapas (§6.4, passo S5), não por causa da
matemática em blocos.

---

## 2. A forma em blocos do ninfer, termo a termo

### 2.1 Constantes e forma de execução

- $S = 128$ (`ninfer .../gated_delta_net/common.h:7`), $B = 64$ (`:8`, com `static_assert` nos três
  kernels: `ninfer .../chunked/prepare_wy_wu.cuh:28`, `.../chunked/state_passing.cuh:21`,
  `.../chunked/output.cuh:36`), sub-bloco $BC = 16$ e MMA $16\times8\times8$
  (`ninfer .../chunked/common.cuh:12-19`).
- Ordem de lançamento: `prepare_wy_wu` → `state_passing` → `output`
  (`ninfer .../chunked/launch.cu:31-72`).
- Grades: `prepare` = $(N_T, H_v)$ CTAs (`ninfer .../chunked/prepare_wy_wu.cu:41-43`); `state_passing`
  = $H_v \cdot D_{strips}$ CTAs (faixas de valor de 16 ou 32; $D_{strips}=8$ ou 4)
  (`ninfer .../chunked/state_passing.cu:19,42-49`); `output` = $(N_T, H_v)$ com multi-job por CTA
  (`ninfer .../chunked/output.cu:43-54`).
- Cauda: $T$ tem de ser múltiplo de 64 ou o caminho em blocos **recusa** — a mensagem é
  literalmente *"route tail tokens through AR instead"*
  (`ninfer .../chunked/launch.h:119-128`); quem faz isso é o dispatcher
  (`ninfer .../gated_delta_net/gated_delta_net.cpp:249-288`: `T_full = (T/64)*64` vai para
  `launch_chunked`, o resto para `launch_recurrent_inout`). llama.cpp faz diferente: **preenche**
  o último bloco com zeros inertes ($\beta=0 \Rightarrow u=0$, $k=0 \Rightarrow$ update 0,
  $g=0$ não muda o cumsum) (`llama.cpp src/models/delta-net-base.cpp:63-70`).

### 2.2 Denominações

Bloco de $B$ tokens, índice local $r = 0..B-1$, cumsum **inclusivo e local ao bloco**
(`ninfer .../chunked/prepare_wy_wu.cuh:417-443`, scan de Hillis-Steele sobre os 64 valores de $g$):

$$
G_r = \sum_{s \le r} g_s, \qquad
e^{G_r} = \prod_{s\le r} a_s, \qquad
e^{G_r - G_c} = \prod_{s=c+1}^{r} a_s \;\;(r>c)
$$

com $K$ = matriz $B\times S$ das chaves normalizadas, $V$ = $B \times S$ dos valores,
$\beta = \mathrm{diag}(\beta_0..\beta_{B-1})$, $H_{\text{in}}$ = estado na entrada do bloco
(em $S\times S$, dim de chave $\times$ dim de valor).

### 2.3 Estágio 1 — `prepare_wy_wu`: a transformada WY/UT intra-bloco

**(a) $KK^{\top}$ decaído e com sinal.** O kernel monta as $B\times B$ entradas
(`ninfer .../chunked/prepare_wy_wu.cuh:465-516` para o MMA; `:580-595` para o escalonamento):

$$
L[r][c] \;=\; K K^{\top}[r][c]\; e^{G_r-G_c} \;=\; (k_r \cdot k_c)\, e^{G_r - G_c},
\qquad r > c, \quad L[r][c] = 0 \text{ caso contrário (estritamente triangular inferior)}
$$

e guarda $A = -\mathrm{diag}(\beta) L$, isto é $A[r][c] = -\beta_r (k_r\cdot k_c) e^{G_r-G_c}$.
Dois detalhes que não são cosméticos: **$\beta$ é indexado pela LINHA $r$**, e a **diagonal é
excluída** (o comentário do próprio kernel avisa por que o mascaramento tem de ser `select` e não
`* mask`: $G$ é monótona decrescente, então no triângulo superior $e^{G_r-G_c}$ estoura para
$+\infty$ e `inf*0 = NaN`, `ninfer .../chunked/prepare_wy_wu.cuh:571-579`).

**(b) A inversa.** Com $I$ na diagonal (`ninfer .../chunked/prepare_wy_wu.cuh:604,675`), o kernel
resolve a triangular inferior em dois níveis: blocos diagonais de $16\times16$ por substituição
serial em $i$ com as 16 colunas em paralelo (`ninfer .../chunked/prepare_wy_wu.cuh:217-232`) e
complemento de Schur por blocos em 3 ondas (`:640-672`). O resultado é

$$
\boxed{\;T \;=\; (I - A)^{-1} \;=\; \bigl(I + \mathrm{diag}(\beta)\,\mathrm{tril}_{\text{estrito}}(K K^{\top} \odot D)\bigr)^{-1}\;}
$$

com $D[r][c] = e^{G_r-G_c}$. **Esta é a mesma expressão do llama.cpp**
(`llama.cpp src/models/delta-net-base.cpp:152-167`: `tri(..., LOWER)` para tirar a diagonal,
$\text{lhs}=A+I$, `solve_tri(lhs, -A, left, lower, unit=false)`, $+I$ de volta).

**(c) $W$ e $U$** (`ninfer .../chunked/prepare_wy_wu.cuh:681,690-721`):

$$
U \;=\; T\,\bigl(\mathrm{diag}(\beta) V\bigr),
\qquad
W \;=\; T\,\bigl(\mathrm{diag}(\beta)\,\mathrm{diag}(e^{G}) K\bigr)
$$

$U[r][\cdot] = \sum_c T[r][c]\beta_c v_c$, $W[r][\cdot] = \sum_c T[r][c]\beta_c e^{G_c} k_c$.
Nota de precisão: $T$, o $V/K$ escalonado e o produto ficam em **FP32** (TF32 só no MMA, e o
comentário diz explicitamente que $T$ e o $V/K$ escalonado *"are never down-cast to BF16"*,
`ninfer .../chunked/prepare_wy_wu.cuh:677-680`). O que vai para o workspace é que é BF16 (§5.1).

### 2.4 Estágio 2 — `state_passing`: a passagem de estado entre blocos

`ninfer .../chunked/state_passing.cuh`. O estado é carregado uma vez para registrador
(`h_frag`, `:254-274`), no layout **transposto AR** — índice `state[col_d * S + row_g]`, com
`col_d` = dim de valor e `row_g` = dim de chave (`:266-272`) — e devolvido no fim no mesmo layout
(`:522-535`). O laço é sobre os blocos, em ordem (`:313`). Por bloco:

$$
\begin{aligned}
\text{v\_new} &= U - W H_{\text{in}} &&\text{Phase B/C, } \texttt{:351-423}\\
\text{vd}[t] &= \text{v\_new}[t]\; e^{G_{B-1}-G_t} &&\text{Phase D, } \texttt{:426-467}\\
H_{\text{out}} &= e^{G_{B-1}} H_{\text{in}} + \sum_{t} k_t \otimes \text{vd}[t]
   &&\text{Phase E, } \texttt{:474-499}
\end{aligned}
$$

com $e^{G_{B-1}} = \texttt{exp2\_approx}(g_C \log_2 e)$
(`ninfer .../chunked/state_passing.cuh:426-429`) e a saída de cada bloco
$H_{\text{in}}$ (o estado **na entrada**) gravada em `h_chunk` em BF16 (`:358-376`), que é o que o
estágio 3 consome. O estado persistente `ssm_state_in/out` é **FP32**
(`ninfer .../gated_delta_net/gated_delta_net.cpp:80,166`), ao contrário dos intermediários.

Juntando com os nomes de $U$ e $W$:

$$
\text{v\_new} \;=\; T\,\mathrm{diag}(\beta)\bigl(V - \mathrm{diag}(e^{G}) K H_{\text{in}}\bigr)
$$

que é exatamente o $\delta$ "efetivo" de cada token do bloco, medido contra o estado de entrada e
transformado pela WY. É a mesma expressão do llama.cpp (`llama.cpp
.../delta-net-base.cpp:183,243-247`, com o tensor chamado `k_cumdecay`).

### 2.5 Estágio 3 — `output`

O cabeçalho do kernel dá a fórmula e o código a cumpre:

```
A[t,s] = (s <= t) ? dot(q[t], k[s]) * exp(g[t] - g[s]) : 0
out    = scale * (exp(g) * q @ h_chunk^T + A @ v_new)
```
(`ninfer .../chunked/output.cuh:10-11`; máscara e decaimento em `:226-252`, com a **diagonal
incluída**, `s <= t`; termo do estado com $\times e^{G_t}$ em `:233-234,304-307`; $A\cdot V$ em
`:321`; $\times$ `scale` em `:327-329`.)

Em notação:

$$
o_r \;=\; \frac{1}{\sqrt S}\Bigl[\,e^{G_r}\,\bigl(q_r \cdot H_{\text{in}}\bigr)
\;+\; \sum_{s \le r} (q_r\cdot k_s)\, e^{G_r-G_s}\, \text{v\_new}[s] \,\Bigr]
$$

Mesmo esquema no llama.cpp: máscara `LOWER_DIAG` (diagonal dentro)
(`llama.cpp .../delta-net-base.cpp:143-147`), termo intra $A v_{new}$ ($:251$), termo do estado
$e^{g_t}(q_t\cdot S)$ ($:255,259$), escala $1/\sqrt{S_k}$ aplicada em $q$ uma única vez
($:45-47`) — a diferença de *onde* a escala entra (deles em $q$; a nossa e a do ninfer, na saída) é
equivalente em aritmética exata e diferente em arredondamento.

### 2.6 Shapes e onde cada tensor vive

| tensor | shape lógico | quem é | dtype ninfer | nosso candidato |
|---|---|---|---|---|
| $g$ (log-gate) | $[T][H_v]$ | por token × value head | FP32 entrada | **cópia exata** do nosso `d_gate2b_` (`nosso include/rdna4/graph.cuh:419`) |
| $\beta$ | $[T][H_v]$ | por token × value head | FP32 entrada (pós-sigmoid fora) | **cópia exata** do nosso `d_betab_` |
| $K, Q, V$ | $[T][H_{qk}\text{ ou }H_v][S]$ | por token × head × dim | BF16 | nosso é FP32 (`nosso include/rdna4/graph.cuh:416,671`) |
| $g\_cumsum$ | $[H_v][T]$ | por head × token, **local ao bloco** | FP32 | FP32 |
| $T$ | $[B][B]$ por (bloco, head) | intra-bloco | FP32, **só em smem** | FP32, smem (1 KB com $B{=}16$) |
| $W, U$ | $[B][H_v][S]$ | por bloco × head × dim | workspace **BF16** | **FP32** (§5.1) |
| `v_new` | $[B][H_v][S]$ | idem | workspace **BF16** | **FP32** |
| `h_chunk` | $[N_T][H_v][S][S]$ | por bloco × head, estado na **entrada** do bloco | workspace **BF16** | **FP32** (ou eliminado, §6.4) |
| `ssm_state` | $[S][S][H_v]$ fp32 | por head, persistente | **FP32** | `nosso d_state_` (`nosso include/rdna4/graph.cuh:695`) |

Layout do workspace no ninfer: `ninfer .../chunked/launch.h:32-46`
(`g_cumsum` FP32 `{H_v, T}`, `W/U/v_new` BF16 `{S, H_v, T}`, `h_chunk` BF16 `{S, S, H_v, N_T}`).
Layout do estado do ninfer: `state[d \cdot S + k]` (valor $\times$ chave), `ninfer
.../chunked/state_passing.cuh:266-272,531-534` — **idêntico ao nosso** $M[j][i] = S[i][j]$
(`nosso include/rdna4/gdn.cuh:8-10`), e ambos FP32.

---

## 3. Equivalência: exata, com prova numérica

### 3.1 O que eu rodei

Sem GPU. Protótipo em numpy (em `/tmp/gdn_check/`, fora do repositório): a recorrência sequencial
transcrita de `nosso include/rdna4/gdn.cuh:71-86` (com `expf` por token, fma ascendente,
saída pós-atualização) contra a forma em blocos transcrita de §2.3–2.5 ($A$, $T=(I-A)^{-1}$,
$U$, $W$, `v_new`, `vd`, $H_{\text{out}}$, $A_{\text{out}}$ com diagonal, $o_r$). Entradas
realistas: $k$ e $q$ normalizados em L2 (é o que o nosso `l2_norm` entrega), $g \le 0$
decaimento por token, $\beta \in (0,1]$, estado inicial não-nulo.

### 3.2 Resultado em aritmética exata (`float64`)

| $B$ | $T$ | desvio rel. máx. |
|---|---|---|
| 8 | 256 (3 sementes) | 1,0e-15 (máx. sobre saída **e** estado) |
| 16 | 256 (3 sementes) | 1,1e-15 (idem) |
| 32 | 256 (3 sementes) | 1,1e-15 (idem) |
| 64 | 256 (3 sementes) | 1,7e-15 (idem) |
| 64 | 64 (1 bloco) | saída 2,1e-16 · estado 1,7e-15 |
| 64 | 128 (2 blocos, 5 sementes) | saída 1,5e-16 … 2,9e-16 · estado 4,4e-16 … 1,6e-15 |
| 64 | 256 / 1024 / 2048 | estado 1,9e-15 / 1,5e-15 / 9,5e-16 |

Ou seja: **a igualdade não depende de $B$** (o que autoriza escolher $B$ pelo custo, §6.2) e é
igualdade matemática, não coincidência numérica. Sustentam isso, independentemente:

- o **próprio referencial de teste do ninfer** é a recorrência sequencial em `float64`
  (`ninfer tests/ops/gdn_ref.h:99-137`: $\alpha = e^{g}$, $\delta = \beta(v - \alpha\, S^{\top}k)$,
  $S \leftarrow \alpha S + \delta k^{\top}$, leitura **pós-atualização**) — aritmeticamente igual à
  nossa §1.2 —, e o dispatcher manda $T \ge 64$ para o caminho em blocos com cauda em AR
  (`ninfer .../gated_delta_net.cpp:249-288`), com os casos "exact chunk"/"chunk-tail"/"two-chunk"
  no mesmo teste (`ninfer tests/ops/test_gated_delta_net.cpp:459-469`);
- o **llama.cpp** implementa a mesma decomposição como grafo, termo a termo: $T$, $u = T(\beta\odot V)$,
  $v_{new} = u - T\mathrm{diag}(\beta)\mathrm{diag}(e^{g})KS$, $S \leftarrow S e^{g_{last}} + K^{\top}\mathrm{diag}(e^{g_{last}-g})v_{new}$,
  saída $e^{g}(q\cdot S) + A v_{new}$ (`llama.cpp .../delta-net-base.cpp` citado em §2).

Duas implementações independentes (kernels CUDA do ninfer e grafo ggml do llama.cpp) com os mesmos
termos, e o meu protótipo fechando em 1e-15 contra a nossa recorrência: **é equivalência, não
aproximação**. Nenhum dos dois códigos comenta equivalência ou aproximação — o grep por
`equival|exact|identical|approxim` só acha, no llama.cpp, uma citação do referencial torch
(`llama.cpp src/models/delta-net-base.cpp:191`).

### 3.3 Resultado em `float32` — e o efeito dos dtypes

| configuração | desvio relativo máx. contra o nosso sequencial fp32 |
|---|---|
| bloco, tudo FP32 — T=64 / 256 / 1024 / 2048 | 5,0e-7 / 5,9e-7 / 6,2e-7 / 9,5e-7 |
| bloco, tudo FP32 — T=8192 (128 blocos) | 1,0e-6 (estado 2,3e-7) → **não cresce com o número de blocos** |
| o nosso próprio sequencial fp32 contra `float64` | 3,1e-7 … 4,4e-7 |
| bloco com os dtypes do ninfer (`W`,`U`,`v_new`,`h_chunk` BF16) | **3,0e-3 … 4,9e-3** |

Consequência direta para o port: **workspace em FP32, não BF16.** Em FP32 o desvio do caminho em
blocos é da mesma ordem do erro que o caminho sequencial já tem — inclusive porque a recorrência é
não-expansiva (§1.2) e o desvio não acumula. Em BF16 seria 5 000× pior e obrigaria o gate a ser
frouxo (é o gate deles: 2,7e-3/4,1e-3, `ninfer tests/ops/test_gated_delta_net.cpp:25-33`).

### 3.4 Estabilidade da inversa nova

A forma em blocos cria quantidades que o nosso caminho não tem ($KK^{\top}$, $T$). Medi o
condicionamento em cinco regimes, incluindo o pior caso concebível (chaves praticamente iguais
dentro do bloco):

| caso | $\mathrm{cond}(I-A)$ | $\lvert T\rvert_{\max}$ | desvio fp32 |
|---|---|---|---|
| chaves iid, $\beta \le 1$ | 1,32 | 1,00 | 4,4e-7 |
| chaves iid, $\beta \le 0{,}5$ | 1,15 | 1,00 | 5,4e-7 |
| chaves em posto 8 + ruído | 2,54 | 1,00 | 5,8e-7 |
| chaves em posto 4 + ruído | 3,44 | 1,00 | 4,7e-7 |
| chaves em posto 2, quase sem ruído | 4,41 | 1,00 | 8,3e-7 |
| chaves iguais dentro do bloco | 6,15 | 1,00 | 6,6e-7 |

A razão é estrutural: $A$ é estritamente triangular inferior com $|A[r][c]| \le \beta_r \lvert k_r\cdot k_c\rvert e^{G_r-G_c} \le 1$,
a diagonal de $I-A$ é 1 e o decaimento só encolhe os termos — não há amplificação. **INFERIDO**
que isso vale para os $k$ reais do modelo: confirma-se rodando o teste de §6.3 com os $k$
normalizados de um prompt real (o mesmo teste já cobre isso se o estado inicial não for zero).

### 3.5 As três armadilhas (medidas)

Desvio relativo da forma em blocos contra o nosso sequencial quando uma convenção é trocada:

| convenção trocada | desvio |
|---|---|
| cumsum **exclusivo** em vez de inclusivo | **0,275** |
| máscara da saída **estrita** ($s<t$) em vez de inclusiva ($s\le t$) | **0,824** |
| $T=(I + L\,\mathrm{diag}(\beta))^{-1}$ em vez de $(I+\mathrm{diag}(\beta)L)^{-1}$ ($\beta$ na coluna) | **1,1e-2 … 2,6e-2** |
| as duas primeiras linhas juntas | 0,848 |

Não são erros que um gate de PPL pegue com folga — são erros de ordem 1. Cada um vira uma
verificação no teste de §6.3 (o teste de $T$ contra $I$ é o mais barato: $T\cdot(I-A) = I$ a 1e-6).

---

## 4. Onde a ordem das somas muda, e que gate isso exige

### 4.1 O inventário

Por (token, value head), o nosso caminho tem exatamente **duas** somas: os dois produtos internos
de comprimento $S=128$ (`nosso include/rdna4/gdn.cuh:75-78` e `:82-85`), em ordem ascendente de
$i$, acumulador único. O histórico de tokens **não** é uma soma explícita: é um esquema de Horner
(128 elementos $\times$ 64 tokens, cada passo com uma multiplicação e uma soma arredondadas).

Na forma em blocos:

| # | o que muda | como |
|---|---|---|
| 1 | os dois produtos internos de comprimento 128 | passam a ser produtos matriciais com k-tiles separados e somados no fim → reassociação de uma soma de 128 termos |
| 2 | o histórico de 64 tokens por elemento do estado | vira **um** produto interno de comprimento 64 ($K^{\top}\,\mathrm{vd}$) com o decaimento aplicado **uma vez** ($e^{G_{B-1}-G_t}$) em vez de 64 multiplicações encadeadas |
| 3 | **novas** somas que não existiam | $KK^{\top}$ ($\binom{B}{2}$ produtos de comprimento $S$), a inversa triangular de $B\times B$, $W$ e $U$ (somas de comprimento até $B$), a atenção intra-bloco ($\le B$ termos) |
| 4 | a exponencial do decaimento | nossa: `expf` por token, multiplicada passo a passo; deles: `exp(G_r - G_c)` — **uma** exponencial de uma diferença de cumsums (que por sua vez é um scan de $B$ somas) |
| 5 | o $\beta$ | nosso multiplica $\beta_t$ no fim ($\delta = \beta(v - \cdot)$); o deles empurra $\beta$ para dentro de $T$ e de $A$, então $\beta$ multiplica **parciais diferentes** |
| 6 | a escala $1/\sqrt S$ | nosso no fim da saída; llama.cpp aplica em $q$ antes (equivalente exato, arredondamento diferente) |

Nenhuma dessas mudanças altera o resultado em aritmética exata — §3. Em `float32`, o desvio
agregado medido é 5e-7 … 1e-6 (§3.3).

### 4.2 Que tipo de gate

1. **Teste de unidade novo (o decisivo): bloco contra sequencial, mesmas entradas.**
   Rodar `delta_rule_batch_launch` (sequencial, `nosso include/rdna4/gdn.cuh:429-444`) e o
   caminho em blocos sobre **os mesmos** $q,k,v,g,\beta$ e o **mesmo** estado inicial (copiar
   `d_state_` antes), comparando (a) o estado final, (b) a saída, (c) o estado após cada bloco.
   Este é o gate que substitui o "bit-exato" e é o único que testa a matemática de verdade.
2. **Tolerância defensável: rel-L2 $\le$ 1e-5** no estado e na saída. Justificativa com número:
   o medido é 5e-7 … 1,0e-6 (fp32, emulado); 1e-5 dá 10× de folga sobre o medido e ainda é
   **270× mais apertado** que o gate que o ninfer usa para o próprio caminho em blocos
   (2,7e-3, `ninfer tests/ops/test_gated_delta_net.cpp:31`). Não é carimbo.
   O que a tolerância **não** pode ser: zero (bit-exato é impossível, §4.3) nem da ordem de 1e-3
   (aí não distingue o caminho em blocos de uma implementação com um erro de convenção, §3.5).
3. **PPL**: `nosso scripts/compare_ppl.sh`. O script gate hoje em **1 % no pior chunk**
   (`nosso scripts/compare_ppl.sh:142,149-151`), enquanto `nosso docs/plano-prefill.md:210` declara
   **0,5 %**. O desvio esperado aqui é ~1e-6 no estado, três ordens abaixo dos dois. Declarar 0,5 %
   (o número do plano) e registrar a discrepância 1 % × 0,5 % num item separado.
4. **Suíte de regressão**: `check-regression-gpu` compara ids (bit-exatos) e logits do último passo
   com `kLogitRelL2 = 1e-5` / `kLogitNormRel = 1e-5`
   (`nosso tests/check_regression_gpu.hip:65-66,576,583`; wrapper `nosso scripts/check_regression.sh`).
   **Risco a medir antes de virar default**: uma perturbação de ~1e-6 no estado do GDN propaga para
   os logits na mesma ordem de grandeza (1e-6), ou seja a menos de 10× do limite. Se estourar, o
   movimento correto é **declarar** o novo número com a medição, não afrouxar em silêncio.
5. **Atenção ao que já existe e vai quebrar**: ver §4.3.

### 4.3 O contrato que este port quebra (a decisão que é do dono do repo)

Hoje o caminho em lote **é bit-idêntico** ao caminho por token, e isso é contrato explícito:
`nosso include/rdna4/gdn.cuh:295-308` ("every token keeps the arithmetic it has in the per-token
path, so the result is BIT-IDENTICAL") e o teste exige exatamente isso
(`nosso tests/check_batch_gpu.hip:163-169` falha se `!exact`, e `:203-204` idem para o caso
"chunks mistos 16 + cauda por token"). A forma em blocos **não consegue** satisfazer isso: ela
reorganiza o histórico de 16 tokens em um produto matricial, e nenhum rearranjo devolve o
bit-a-bit. Portanto:

- enquanto o switch estiver desligado por padrão, `check-batch-gpu` continua valendo como está;
- para ligar o caminho em blocos por padrão, `check_batch_gpu.hip` precisa comparar com a
  tolerância de §4.2 em vez de exigir `BIT-EXACT` para o caminho GDN, e isso é a mesma classe de
  decisão que fixou o chunk default em 16 (`nosso include/rdna4/device.h:209-221`: misturar dois
  caminhos não bit-exatos foi o que fez o gate do `serve` falhar com chunk 128). É decisão do
  dono, não do implementador — e há um agravante de desenho: quem corta o prompt (CLI, servidor,
  MTP) pode misturar o caminho em blocos com o sequencial na mesma sequência se o bloco não
  cobrir todos os tamanhos aceitos por `chunk_ok` (`nosso include/rdna4/graph.cuh:204-206`).

**Duas saídas honestas** (a escolha é do dono):
(a) ligar o caminho em blocos **para todos** os tamanhos de lote que o motor aceita, preenchendo o
último grupo com zeros inertes (como o llama.cpp faz, `llama.cpp src/models/delta-net-base.cpp:63-70`),
com um $B$ que divida os tamanhos aceitos — aí a aritmética depende só das posições, não de como
o prompt foi cortado, e `serve`/MTP voltam a ser reproduzíveis entre cortes diferentes;
(b) manter `RD_GDN_CHUNKED=0` por padrão e usar o caminho novo só atrás de flag declarada, com os
números medidos.

---

## 5. Análise de gap: o que eles assumem e nós não temos, e vice-versa

### 5.1 O que o ninfer assume (e nós não temos)

| # | premissa do ninfer | situação nossa | custo |
|---|---|---|---|
| 1 | workspace de 5 tensores (BF16), dimensionado por `tokens` (`ninfer .../chunked/launch.h:32-46`) | **não alocamos nada para o GDN** além de `d_state_` e `d_convst_` (`nosso include/rdna4/graph.cuh:695-696`) | buffer novo + espelho em `device.h` + pino em `tests/check_kvtype.hip:199-201` (§6.3) |
| 2 | intermediários em BF16 | nosso caminho é **todo** FP32 (`nosso include/rdna4/graph.cuh:506-516`) | não copiar: §3.3 mostra 3-5e-3 contra 1e-6 |
| 3 | $T$ múltiplo de 64, senão cauda em AR (`ninfer .../chunked/launch.h:119-128`) | nosso chunk default é **16** (`nosso include/rdna4/device.h:206-233`) e a cauda existe (`RD_PREFILL_CHUNK`) | escolher $B$: §6.2 |
| 4 | $k$ e $q$ normalizados **dentro** do op, eps 1e-6 (`ninfer .../gated_delta_net.cpp:71-101`, `ninfer .../recurrent.cuh:32`) | nós normalizamos antes, num kernel próprio, com `rms_norm_eps` (`nosso include/rdna4/graph.cuh:1310`) | nenhum: o estágio 1+3 só precisa ler $q,k$ já normalizados do `d_qkb_` — e a etapa continua bit-exata |
| 5 | mapeamento de cabeça `qk_head(h_v) = h_v / G`, $G=H_v/H_{qk}$ (`ninfer .../common.cuh:112-118`, confirmado no referencial `ninfer tests/ops/gdn_ref.h:33-36` e na doc `ninfer docs/maintainer/qwen3.6-27b-model.md:77`) | o nosso é $kh = h \bmod n_{kh}$ (`nosso include/rdna4/gdn.cuh:65,374`) | **não copiar o deles**; ver §5.3 |
| 6 | MMAs BF16/TF32 e layouts de smem entrelaçados para `ldmatrix` (`ninfer .../chunked/prepare_wy_wu.cuh:336-350`) | nosso GDN é SIMT fp32 | o port não reusa nenhum layout deles; sobra a matemática |
| 7 | dispatcher com geometria por $H_v$ (32 vs $\ge 48$) (`ninfer .../chunked/prepare_wy_wu.cu:42-43`, `.../state_passing.cu:42-49`) | nós temos $H_v=48$, $H_{qk}=16$ fixos (`nosso tests/check_kvtype.hip:189-191`) | grade fixa: 48 value heads, faixas de valor |

### 5.2 O que nós temos e o deles não tem

- **A conv1d com estado rolante e o silu** (`nosso include/rdna4/gdn.cuh:25-46,315-345`): no ninfer
  isso é outro op (fora do caminho em blocos). O port tem de continuar produzindo
  `d_qkb_`/`d_vcb_` iguais; a conv fica onde está (ou fundida, §6.4 S5), e é bit-exata em relação
  a hoje.
- **A cauda por token**: nosso `chunk_ok` aceita 2/3/4/8/16 e qualquer $n \le$ `batch_max_`
  (`nosso include/rdna4/graph.cuh:204-206`), e o caminho por token existe para $n=1$
  (`nosso include/rdna4/graph.cuh:1338-1381`); o ninfer simplesmente recusa $T$ não múltiplo de 64.
- **O snapshot/restore do estado para o MTP** (`nosso include/rdna4/graph.cuh:1592-1654`) e o
  nó de grafo `new_state-*` (`nosso tests/check_graph_gpu.hip:396-398` com limiar 1e-3): o caminho
  em blocos tem de escrever **o mesmo buffer, no mesmo layout, no mesmo dtype** — e escreve; é por
  isso que o layout `[valor][chave]` fp32 importa (§2.6).
- **A escala e o ssm_norm+silu** fora do kernel (`nosso include/rdna4/graph.cuh:1321-1327`): o
  ninfer tem `build_norm_gated` no grafo do modelo. Mantemos como está.
- **O gate bit-exato do caminho em lote** (§4.3) — o preço da troca é nosso, não deles.

### 5.3 Divergência real entre os dois: o mapeamento de cabeça K↔V

Não é diferença de estilo, é função diferente sobre os mesmos pesos:

- **ninfer**: "every group of three V heads shares one Q/K head"
  (`ninfer docs/maintainer/qwen3.6-27b-model.md:77`), implementado como $kh = \lfloor h/3 \rfloor$
  (`ninfer .../common.cuh:112-118`) — as 3 V heads **consecutivas** compartilham um K head;
- **nosso**: $kh = h \bmod 16$ (`nosso include/rdna4/gdn.cuh:65`) — as V heads $0,16,32$ compartilham
  o K head 0 (intercalado);
- **llama.cpp** (o referencial de onde o nosso veio): repete $q,k$ para $H_v$ heads com
  `ggml_repeat_4d` quando $H_{qk}\ne H_v$ (`llama.cpp src/models/qwen35.cpp:436-442`), e `ggml_repeat`
  indexa a dimensão repetida por **módulo** → $h \bmod H_{qk}$ = **o nosso**.

Portanto: nós concordamos com o llama.cpp; o ninfer discorda dos dois. Com $G=3$ os dois
mapeamentos são permutações diferentes dos pares (K,V), e não podem estar ambos certos para os
mesmos pesos. **INFERIDO**: o que confirmaria — rodar o mesmo prompt no ninfer e no llama.cpp sobre
o mesmo GGUF e comparar a PPL de um dos dois lados; ou ler o `repeat_interleave`/`repeat` do
`transformers` para Qwen3.x. Não bloqueia o port (a especificação é: manter $kh = h \bmod n_{kh}$,
que é o que o nosso caminho sequencial valida), mas é um achado para o dono do repo, não para este
documento resolver.

---

## 6. Plano de port com switch

### 6.1 Arquivos

| arquivo | mudança |
|---|---|
| `include/rdna4/gdn.cuh` | kernels novos: `gdn_prepare_chunk_kernel`, `gdn_state_output_chunk_kernel` (e a variante de 3 estágios, §6.4), launchers. **Nada é removido**: `delta_rule_kernel`, `delta_rule_kernel_ilp` e `delta_rule_batch_rows_kernel` ficam intactos (histórico e fallback) |
| `include/rdna4/graph.cuh` | (a) a leitura do switch, no padrão da casa (função local `static const bool` + `getenv`, como `split_batch_enabled()` em `nosso include/rdna4/graph.cuh:166-172` e `batch_ops()` em `:213-220`); (b) o `if` no ramo em lote do GDN (`nosso include/rdna4/graph.cuh:1285-1337`), trocando `delta_rule_batch_launch` pelos launchers novos; (c) alocação do workspace em `init()` junto de `nosso include/rdna4/graph.cuh:695-696`; (d) marcas `RD_PHASE` novas |
| `include/rdna4/device.h` | o espelho do orçamento: novo termo em `graph_buffer_bytes()` (`nosso include/rdna4/device.h:105-143`) |
| `tests/check_kvtype.hip` | o pino do orçamento (`nosso tests/check_kvtype.hip:199-201`) passa a exigir `172 258 340 + ws_bytes`; o valor tem de ser somado à mão, como o comentário de `:197-198` já faz |
| `tests/check_gdn_chunked_gpu.hip` (**novo**) + `CMakeLists.txt` | o teste de unidade de §4.2/§6.3 |
| `docs/` | o resultado da medição (§7) quando ela existir |

### 6.2 Escolha de $B$

O port não precisa de 64. Medido: a igualdade vale para qualquer $B$ (§3.2), e o custo por token é

$$
\text{fma/token/head} \;=\; \underbrace{3S^2}_{\text{estado}} \;+\; \underbrace{\approx 4\,B\,S}_{\text{intra-bloco}}
$$

A parcela de estado é **igual** à do sequencial (que também gasta $3S^2$: um fma por elemento no
passe do `sum`, dois no passe de atualização+saída, `nosso include/rdna4/gdn.cuh:75-85`); o que se
paga a mais é só o intra-bloco. Com $S=128$: $B=16 \to 57\,344$ fma/token/head (1,17× o
sequencial de 49 152); $B=64 \to 81\,920$ (1,67×). $B$ menor é mais barato em fma e troca mais
vezes o estado entre blocos (tráfego de estado $\propto 1/B$). Além disso, $T = B\times B$ fp32 vive
em smem: $B=16 \to 1$ KB, $B=64 \to 16$ KB, $B=128 \to 64$ KB (o CU inteiro).

**Proposta**: $B = 16$ fixo, casando com o chunk de produção (`RD_PREFILL_CHUNK` default 16), com
preenchimento de zeros no último grupo — o mesmo truque do llama.cpp
(`llama.cpp src/models/delta-net-base.cpp:63-70`) — o que faz **todos** os tamanhos de lote
aceitos usarem a mesma aritmética (isso resolve o agravante de §4.3: a aritmética passa a depender
só da posição, não do corte). Instanciar também $B=64$ para medir as duas configurações com
`RD_PREFILL_CHUNK=64` — a escolha final é o número de §7.

### 6.3 Workspace e o pino do espelho

Variante de 2 estágios (recomendada; o estágio 3 fundido no 2, §6.4):

```
por (bloco, value head), fp32:
  g_cumsum  : B              floats
  W         : B x S          floats
  U         : B x S          floats
  (T fica só em smem: B x B)
total = ceil(n/B) * nvh * (B + 2*B*S) * 4 bytes
```

Com $n=16$, $B=16$, $nvh=48$, $S=128$: **789 504 B** (0,75 MiB). Com `RD_PREFILL_CHUNK=512` →
32 blocos → **25,2 MB**. O pino do teste vira `172 258 340 + 789 504 = 173 047 844 B` no shape
IQ3_S (o espelho tem de usar `s.max_batch`/`prefill_chunk_cap()`, como
`nosso include/rdna4/device.h:255`).

A variante de 3 estágios (a do ninfer, mais fácil de depurar) materializa também `v_new` e
`h_chunk` em FP32: +$B\cdot S$ floats por (bloco, head) = **+0,39 MB** e +$S^2$ floats =
**+3,15 MB por bloco**, o que com `RD_PREFILL_CHUNK=512` (32 blocos) dá 63,0 MB no total.
É por isso que a fusão da saída no estágio de estado é a recomendação: ela elimina o $h_{\text{chunk}}$
inteiro (e o seu tráfego), porque o termo $e^{G_r}(q_r\cdot H_{\text{in}})$ é independente por
faixa de dim de valor, que é exatamente como o CTA do estado já está fatiado.

### 6.4 Desenho dos kernels (nosso estilo: SIMT fp32)

1. **`gdn_prepare_chunk_kernel`** — grade `(blocos, nvh)`, 128 threads.
   smem: $K$ do bloco ($B\times S$ fp32 = 8 KB com $B{=}16$), $T$ ($B^2$ = 1 KB), $\beta$ e $g$
   ($B$ cada). Cada warp/thread faz: (a) $KK^{\top}$ do bloco; (b) $A = -\mathrm{diag}(\beta)L$
   com $\mathrm{diag}(\beta)$ na **linha** e máscara **estrita**, em `select` (nunca `* mask`,
   §2.3a); (c) $T = (I-A)^{-1}$ por substituição em blocos de 16 (o mesmo esquema de
   `ninfer .../chunked/prepare_wy_wu.cuh:217-232,640-672`, que é o algoritmo clássico e não tem
   nada de proprietário); (d) $U = T(\beta V)$ e $W = T(\beta e^{G} K)$; (e) escreve $W$, $U$,
   $g\_cumsum$ no workspace. $T$ **não** sai do smem.
2. **`gdn_state_output_chunk_kernel`** — grade `(nvh, S/D_{strip})` com $D_{strip}=32$ → 192 CTAs
   de 128 threads (a mesma contagem do kernel atual, mas com trabalho por thread muito mais
   paralelo). Cada CTA carrega a sua faixa do estado ($S \times D_{strip}$ fp32 = 16 KB) para
   registrador/smem, laça os blocos em ordem e por bloco:
   (a) `v_new[·][faixa] = U[:, faixa] − W · H[:, faixa]` (produto 16×128×32, acumuladores
   independentes); (b) $A_{\text{out}}[r][s] = (q_r\cdot k_s)e^{G_r-G_s}$, $s \le r$, e
   $o_r[\text{faixa}] = e^{G_r}(q_r\cdot H_{\text{in}}[:, \text{faixa}]) + \sum_{s\le r}A_{\text{out}}[r][s]\text{v\_new}[s][\text{faixa}]$,
   já multiplicado por $1/\sqrt S$ e escrito **em cima de `d_vcb_`** (o buffer que o
   `ssm_out`/rms_norm lê hoje, `nosso include/rdna4/graph.cuh:1321-1332`) — assim nada depois do
   GDN muda; (c) $H \leftarrow e^{G_{B-1}}H + \sum_t k_t \otimes (\text{v\_new}[t]e^{G_{B-1}-G_t})$;
   (d) no fim dos blocos, o estado volta para `d_state_` **no mesmo layout fp32**
   (`nosso include/rdna4/gdn.cuh:8-10`), para o snapshot/MTP continuarem válidos (§5.2).
3. **Fusões opcionais (S5)**: conv1d + l2_norm + `sigmoid/softplus` no estágio 1 (os kernels
   atuais não são o gargalo, mas `gdn_conv` 0,051 + `gdn_l2norm` 0,072 + `gdn_scalars` 0,052 =
   0,175 ms/token entram no mesmo balde de leitura de $q,k$). Não fazer na primeira versão: muda
   etapas hoje bit-exatas e amplia o que o gate tem de cobrir.

### 6.5 Switch e ordem de implementação

Switch: **`RD_GDN_CHUNKED`**, lido uma vez, no padrão da casa
(`nosso include/rdna4/graph.cuh:166-172`). Regra do plano (`nosso docs/plano-ninfer.md:53-54`):
`RD_GDN_CHUNKED=0` mantém o caminho antigo. Entrada em produção em duas etapas — enquanto os gates
de §4.2/§4.3 não fecharem, o default é **desligado** (ou seja, temporariamente `RD_GDN_CHUNKED=1`
liga), e só depois de a decisão de §4.3 estar tomada o default vira ligado com `=0` como escape.

Ordem, cada passo verificável sozinho:

| passo | entrega | como se verifica sozinho |
|---|---|---|
| **S1** | workspace + `prepare` (escreve $W$, $U$, $g\_cumsum$) | (i) $T\cdot(I-A) = I$ a 1e-6; (ii) $W,U$ contra uma implementação ingênua em laço no mesmo teste; (iii) `g_cumsum` contra um laço em CPU |
| **S2** | `state_output` (a parte de estado: `v_new`, estado final) | comparar o estado final contra `delta_rule_batch_launch` nas mesmas entradas: rel-L2 ≤ 1e-5 (é o gate central, e isola a lógica de estado da de saída) |
| **S3** | a parte de saída, ligando S1+S2 ponta a ponta | saída contra o sequencial: rel-L2 ≤ 1e-5, mais o caso $B=16$ **e** $B=64$ (prova a independência de $B$), mais estado inicial não-nulo |
| **S4** | fiação em `graph.cuh` atrás do switch + espelho/pino | `check-kvtype` (pino novo), `bench-phases-gpu --level 1` (§7), `compare-ppl.sh`, `check-regression-gpu` |
| **S5** | fusões (conv/l2/scalars no preparo) e/ou $B=64$ | mesma bateria do S4 + o delta de `gdn_conv`/`gdn_l2norm`/`gdn_scalars` no profiler |

Teste novo (S3), esboço: preencher `d_qkb_` com $q,k$ normalizados pseudoaleatórios (a mesma
normalização do caminho), `d_betab_` com $\sigma(\cdot)$, `d_gate2b_` com log-gates $\le 0$,
`d_state_` com padrão determinístico não-nulo; rodar o sequencial, guardar estado+saída; restaurar
o estado, rodar o em blocos; comparar os três (estado, saída, estado por bloco) com rel-L2 e
`max|d|`. Um teste só, duas configurações ($n=16$, $n=64$), ~5 s.

---

## 7. A medição que decide

Método obrigatório: o mesmo da frente C, porque é dele que vem o número alvo —
`./scripts/gpu-lock.sh timeout 900 ./build/bench-phases-gpu MODEL --prefill 64 --prefill-reps 3 --level 1`
(`nosso docs/estudo-prefill-c-nosso.md:134-136`; a doc do §5 fixa modelo, KV e a regra do lock
dentro do `timeout` em `nosso docs/estudo-prefill-c-nosso.md:3-4`). Chunk default 16
(`nosso include/rdna4/device.h:206-233`), janela limpa, 3 reps, `RD_GDN_CHUNKED` alternado no mesmo
binário.

O que ler, e o que decide:

| medida | hoje (`nosso docs/estudo-prefill-c-nosso.md:154,159,162,166,494,501-503`) | barra para manter |
|---|---|---|
| `gdn_delta` | **0,625 ms/token** | **≤ 0,30** (corta ≥ 50 % do item) |
| recorrência inteira (delta+conv+l2+scalars+norm_silu) | **0,852 ms/token** | **≤ 0,45** |
| prefill total | 8,233 ms/token (N=64) / 8,244 a N=128 | ganho visível fora do ruído do harness |
| tok/s a 512 tokens | 123,39 tok/s (`bench-phases-gpu` count-only, `nosso docs/estudo-prefill-c-nosso.md:603`) | ≥ 2 % |

Complementos, na mesma sessão de GPU:

1. `--prefill 64` **e** `--prefill 128` (a atribuição a 512 não é executável no nível 2,
   `nosso docs/estudo-prefill-c-nosso.md:575-577`);
2. `RD_PREFILL_CHUNK=64` com $B=64$ e $B=16$: é o que escolhe $B$ (§6.2);
3. o contador de lançamentos por token: a doc mede ~1.940 lançamentos/token
   (`nosso docs/plano-ninfer.md:79-80`); o port acrescenta ≤ 2 lançamentos por camada por chunk —
   o profiler tem de mostrar que isso não come o ganho;
4. `bench-delta-gpu` (`nosso docs/journal-kernels.md:135`) estendido com a cadeia do caminho em
   blocos, para o número isolado por camada contra os **8,42 µs/camada** do `float4` de hoje
   (`nosso docs/journal-kernels.md:146`) — com a ressalva registrada lá de que a cadeia reusa o
   mesmo estado e mede cache, não DRAM (`nosso docs/journal-kernels.md:157-161`).

Aritmética do que é razoável esperar (para calibrar a barra, não para prometer): os 208 µs por
camada por chamada de 16 tokens correspondem a 37,7 M fma (48 heads × 16 tokens × $3S^2$) =
~181 G-fma/s, **~1,1 % do pico de fma da placa** (~16,4 T-fma/s a 64 CU × 128 lanes × 2 GHz), e
20× acima do piso de memória (lê+escreve os 3,1 MB de estado da camada = 6,3 MB ≈ 10 µs a
633 GB/s, `nosso docs/estudo-prefill-c-nosso.md:580-581`). Mesmo com 1,2-1,7× mais fma (§6.2) e
chegando a 10-30 % do pico, o item cai para 0,01-0,07 ms/token — ou seja, a barra de 0,30 é
conservadora e o que pode frustrá-la é **ocupação de grade**: 48 heads × 4 faixas = 192 CTAs é o
mesmo teto de hoje, e o que muda é o trabalho por CTA. Se a medição mostrar que o gargalo virou
outra coisa, o item 5 da tabela do §0.1 de `nosso docs/plano-prefill.md:97-120` (Amdahl: 0,852 é
29 % do que sobra no melhor caso, ~52 % no D4) é quem diz se vale continuar.

---

## 8. O que não consegui determinar (e o que confirmaria)

Tudo abaixo é `INFERIDO` ou `NAO VERIFICADO`, marcado no lugar onde aparece:

1. **Que o condicionamento de $I-A$ é benigno nos dados reais do modelo** (§3.4). Confirmaria: o
   teste de §6.3 com $q,k$ de um prompt real (o teste com estado inicial não-nulo já cobre isso).
2. **Qual mapeamento K↔V o modelo usa** (§5.3): nós concordamos com o llama.cpp e discordamos do
   ninfer. Confirmaria: comparação de PPL dos dois motores sobre o mesmo GGUF, ou o
   `repeat`/`repeat_interleave` do `transformers`.
3. **Se `check-regression-gpu` a 1e-5 de rel-L2 sobrevive** a uma perturbação de ~1e-6 no GDN.
   É medição de GPU, não leitura; `NAO VERIFICADO` por construção.
4. **A barra de 0,30 ms/token para `gdn_delta`**: é minha, derivada do alvo de 0,852 (§7), não é
   um número do repositório. O número que o repo tem é o alvo, não a barra.
5. **O custo de despacho dos 1-2 lançamentos extras por camada por chunk** em tok/s: medido
   indiretamente na doc (`nosso docs/plano-ninfer.md:79-80`), não por este estudo.
6. **A sub-frente "RPT>1"**: não reabri. O diário já mediu que ela perde
   (`nosso docs/journal-kernels.md:161-163`, `0,377-0,518×`) — a forma em blocos não é RPT, é outra
   álgebra, mas o aviso de que **menos warps no ar** derruba o kernel continua valendo para a
   grade de 192 CTAs (§7).

## 9. Não refazer (trabalho anterior nesta mesma função)

- `float4` na linha do estado: 71,70 → 8,42 µs/camada, 8,515×, bit-exato
  (`nosso docs/journal-kernels.md:144-146`, comentário em `nosso include/rdna4/gdn.cuh:207-218`);
  o kernel escalar fica preservado para medição (`nosso include/rdna4/gdn.cuh:237-244`).
- `RPT>1` (64/2, 32/4): mediu **pior** (0,377-0,518×) e foi abandonado
  (`nosso docs/journal-kernels.md:147-150,161-166`).
- O laço de tokens dentro do kernel (batching do prefill): ganho de 2,31× em M=64 e a
  bit-exatidão contra o caminho por token (`nosso include/rdna4/gdn.cuh:291-308`,
  `nosso docs/journal-prefill.md`). O port **substitui** essa última propriedade — é a decisão de
  §4.3, e ela tem de ser explícita.
