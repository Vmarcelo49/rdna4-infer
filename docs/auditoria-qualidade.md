# Auditoria de qualidade — rdna4-infer (frente 1 de 9)

Auditoria **estática** de qualidade de código do motor, na árvore do worktree
`/home/marcelo/Projetos/rdna4-wt-audit-qualidade`, branch `feat/audit-qualidade`,
commit **`aa15eea`** ("M8: proj_qq reusa a quantizacao da ativacao…").

**Escopo:** 67 arquivos C++/HIP (`include/rdna4/**` = 8 373 linhas, `src/**` =
9 570, `tests/**` = 8 186; **26 129 linhas** no total), mais `CMakeLists.txt`,
`scripts/` e os `docs/`.

**Método:** leitura estática + dois builds completos (CPU apenas, nenhum binário,
teste, bench ou modelo foi executado — a placa está na fila de GPU,
`docs/gpu-queue.md` regra 2). Varreduras mecânicas com `grep`/Python sobre a
árvore, e `clang-tidy` sobre 18 unidades de tradução. **Nada foi corrigido**: as
sugestões são descrições, não patches.

**Ferramentas:** `clang-tidy` **existe** (`/usr/bin/clang-tidy`, LLVM 22.1.8) e
funcionou tanto em TU de CPU quanto em TU HIP (via `compile_commands.json` de um
build `Release` externo); foi usado com conjuntos de checks curados
(`bugprone-*`, `performance-*`, `portability-*`, alguns `readability-*`).
**`cppcheck` não está instalado** nesta máquina (`command -v cppcheck` → vazio);
não foi instalado nada. `-Wall -Wextra` **não** está ligado no `CMakeLists.txt`
(ver §4), então o build padrão só emite os avisos ligados por omissão.

**Como ler:** cada achado tem `arquivo:linha`, o que está errado, o cenário de
falha concreto e o conserto mínimo sugerido. Confiança é declarada quando algo
não pôde ser verificado só por leitura. No fim há uma seção de **falsos
positivos** (inclusive das ferramentas) e outra de itens de baixa confiança.

---

## 0. Sumário

Contagem por severidade: **4 CRÍTICOS** (H1, M7, M8, E1), **19 IMPORTANTES** e
**11 itens COSMÉTICOS** (a maioria agrupando vários casos). Top 10 por
(impacto × certeza):

| # | Sev | Achado | Local |
|---|---|---|---|
| 1 | **H1** CRÍTICO | Números do JSON de request convertidos sem checar finitude/intervalo → `static_cast<uint64_t>(inf)` é UB e `temp=inf`/`repeat_penalty=inf` geram logits 0/NaN em vez de 400 | `src/server/serve.hip:668-738` |
| 2 | **M7** CRÍTICO | Somas de `offset + bytes` sem checagem de overflow derrotam a única validação geométrica do loader → lê o offset errado e "carrega" lixo como pesos | `src/backend/loader.cpp:117,153,157` |
| 3 | **M8** CRÍTICO | Comprimento de string vindo do arquivo usado direto como tamanho de `resize` → `bad_alloc`/`length_error` não capturados ⇒ abort | `src/backend/gguf.cpp:27-32,90-94` |
| 4 | **E1** CRÍTICO | Escape hexadecimal com *maximal munch* corrompe o literal de EOG (`\x9Cend` = `0x9CE`) — **provado nos bytes do binário** | `src/backend/tokenizer.cpp:140` |
| 5 | **M1** IMPORTANTE | `Graph::release()` não é idempotente: 18 ponteiros ficam pendurados depois do `hipFree` (os outros 13 são zerados) → duplo free se `release()` for chamado duas vezes | `include/rdna4/graph.cuh:1271-1316` |
| 6 | **M2** IMPORTANTE | `attn_split_kernel` deixa parte do buffer de parciais sem escrever quando `head_dim > WPB*32`; o merge lê VRAM não inicializada | `include/rdna4/attn.cuh:392-397` |
| 7 | **E2** IMPORTANTE | O parâmetro de template `WPB` do kernel **sem split** é ignorado (o corpo usa a constante) ⇒ a varredura "WPB 8/16/32" do bench e de `docs/rocm-estudo.md:347-351` mede trabalho redundante, não mais warps | `include/rdna4/attn.cuh:118,162,168,175` |
| 8 | **M4** IMPORTANTE | Orçamento de VRAM do `serve` ignora `--cache-type-v` ⇒ K/V de tipos diferentes passam na checagem e falham com `hipMalloc failed` (contrariando o comentário 2 linhas acima) | `src/server/serve.hip:374-378` |
| 9 | **E4** IMPORTANTE | O README dá **dois números diferentes para o decode a 4K** e ainda afirma "32 warps/CTA" (o que roda é 8) | `README.md:132-134` vs `README.md:94` |
| 10 | **B1** IMPORTANTE | `-Wall -Wextra` não ligado: se ligado, o build emite **1 304 avisos** (784 de uma única linha) e não passaria limpo | `CMakeLists.txt:27-33` |

---

## 1. Tratamento de erro em chamadas HIP/ROCm

### 1.1 Censo (varredura mecânica)

Todos os sítios de chamadas HIP que devolvem status na árvore (`include/rdna4/**`,
`src/**`, `tests/**`):

| métrica | valor |
|---|---|
| sítios de chamada com retorno de status | **335** |
| com teste de erro na própria expressão/janela imediata | 95 |
| **sem nenhum teste** | **240** |
| ↳ desses, `(void)` explícito (intencional) | **20** (17 `hipFree`, 2 `hipMemcpy`, 1 `hipMemGetInfo`) |
| ↳ desses, instrução solta sem `(void)` | **220** |

A divisão "com/sem teste" vem de uma varredura automática (o teste de erro tem de
aparecer na expressão ou numa janela de 3 linhas); os 240 "sem teste" foram
depois conferidos linha a linha **em `include/` e `src/`** (é a lista abaixo), e
o resto está em `tests/`. Onde a janela erra, ela erra para o lado de *não* ver
uma checagem que existe (caso do `return` no fim de `dequant_row_launch` e
`kv_fill_launch`) — os números de produção citados neste relatório já levam isso
em conta.

Por função (só os sem teste): `hipFree` 71 · `hipMemcpy` 49 · `hipMalloc` 39 ·
`hipEventRecord` 22 · `hipDeviceSynchronize` 11 · `hipEventSynchronize` 11 ·
`hipEventElapsedTime` 11 · `hipEventCreate` 10 · `hipEventDestroy` 10 ·
`hipMemGetInfo` 3 · `hipMemset` 2 · `hipStreamDestroy` 1.

Por arquivo: `tests/check_matvec_gpu.hip` 106 · `tests/check_rope_gpu.hip` 40 ·
`tests/check_matmul_gpu.hip` 19 · `tests/bench_attn_gpu.hip` 17 ·
`tests/bench_matvec_shapes_gpu.hip` 16 · `include/rdna4/graph.cuh` 12 ·
`tests/check_dequant_gpu.hip` 8 · `tests/check_nn_gpu.hip` 8 ·
`include/rdna4/mtp.cuh` 7 · `tests/check_mtp_gpu.hip` 3 · `tests/check_kvctx_gpu.hip` 2 ·
`src/main.hip` 1 · `tests/check_graph_gpu.hip` 1.

**A leitura importante desse censo é o contrário do que ele sugere à primeira
vista:** em código de produção (`include/` + `src/`) as **20** chamadas sem teste
são todas `(void)` deliberado em caminho de liberação/leitura, e **não existe
nenhum `hipMalloc` sem checagem** no motor:

- **`hipMalloc`: 19 sítios em `include/` + `src/`, 19/19 checam o retorno** (12 em
  `graph.cuh`, 7 em `mtp.cuh`). Só um deles não preenche `err` (achado **H3**).
- **lançamentos de kernel: 41 expressões `<<<…>>>` em produção** (35 linhas, em 8
  arquivos: `dequant_row.cuh` 14, `nn.cuh` 6, `kv.h` 5, `attn.cuh` 4,
  `graph.cuh` 4, `matvec.cuh` 4, `gdn.cuh` 3, `mtp.cuh` 1) **e todas têm o
  resultado checado** — imediatamente (`if (hipGetLastError() != hipSuccess)`)
  ou pelo `return hipGetLastError() == hipSuccess;` que fecha o launcher
  (`dequant_row.cuh:103`, `kv.h:325`). Os 9 lançamentos sem checagem
  (`dequant_row.cuh:89-97`, `kv.h:309-317`) **estão cobertos** por esse return.
- Os **10** lançamentos restantes estão em `tests/` (`check_matvec_gpu.hip` 5,
  `bench_matvec_shapes_gpu.hip` 3, `check_matmul_gpu.hip` 1, `bench_attn_gpu.hip` 1)
  e são test-only.
- **propagação ao chamador:** dos ~95 resultados de launcher chamados em
  `include/` + `src/`, **0 são descartados** — todos passam por
  `if (!launch(...)) { err = ...; return false; }`. A única exceção é um método
  de diagnóstico (`src/main.hip:436`, achado **H4**).

Ou seja: **a disciplina existe e é boa**. O que segue são as exceções reais — e
só uma é CRÍTICA (H1), porque é a única alcançável por um pedido HTTP trivial e a
única que produz números errados *em silêncio* num caminho de produção.

### H1 — CRÍTICO — números do JSON de request convertidos sem checar finitude/intervalo

`src/server/serve.hip:733-738` (mesmo padrão em `:668-674`, `:692-697`, `:701-707`)

```cpp
    if (body.get_number("seed", &d)) {
      if (d < 0.0) {
        *e = {400, "seed must be >= 0", "invalid_request_error", "invalid_seed", "seed"};
        return false;
      }
      gr->sp.seed = static_cast<std::uint64_t>(d);      // :738
    }
```

`get_number` devolve o `double` cru do parser
(`src/server/json.cpp:261 out = Value::number(std::strtod(tok.c_str(), nullptr));`,
sem checar `errno`/`ERANGE`). **Cenário 1 (UB):** `{"seed": 1e400}` → `strtod`
devolve `+inf`; `inf < 0.0` é falso, a guarda passa e
`static_cast<std::uint64_t>(inf)` é **comportamento indefinido** (na prática
`cvttsd2si` devolve `0x8000000000000000`). **Cenário 2 (número errado em
silêncio, pior):** `{"temperature": 1e400}` é aceito como `temp = inf`; em
`Sampler::filter` (`src/backend/sampler.cpp:132`) todo logit vira
`logit / inf = 0`, o softmax fica uniforme e a resposta é lixo determinístico;
com `{"repeat_penalty": 1e400}`, `0 * inf = NaN` para candidatos de logit zero
(`sampler.cpp:61-67`). Tudo isso em vez de um `400`.

Escopo: `top_p` (`:677`), `presence/frequency_penalty` (`:718`, `:726`) e
`top_k` (via `get_int`/`as_int`, `json.cpp:313-315`) **estão** corretamente
limitados nos dois extremos; o defeito é só dos campos validados de um lado.

**Conserto:** exigir `std::isfinite(d)` e um teto explícito (`d < 1e15` para
`seed`, `d <= 1e6` para `temperature`/`min_p`/`repeat_penalty`) em
`parse_sampling`, ou rotear todo campo numérico por um helper que valide os dois
extremos — ~6 linhas em um único ponto (`serve.hip:664-740`).

### H2 — IMPORTANTE — leitura D2H descartada devolve lixo silencioso

`include/rdna4/graph.cuh:177-180`

```cpp
  void readback(const float *d, std::size_t n, std::vector<float> &out) const {
    out.resize(n);
    (void)hipMemcpy(out.data(), d, n * sizeof(float), hipMemcpyDeviceToHost);
  }
```

`mtp.cuh:647-651` faz o mesmo em `read_h_out`. **Cenário:** uma falha de cópia
(contexto perdido, `hipMemcpy` recusando um ponteiro, OOM de staging) deixa `out`
com os zeros do `resize` e o chamador segue como se fosse o estado oculto real —
no caso do MTP, `MtpHead::step_host` recebe `h_prev = 0` e o rascunho vira
ruído, sem nenhuma mensagem. É também o único caminho onde o resultado de
`hipMemcpy` é jogado fora num laço de decode (`forward_run` → `readback` quando
`want_hidden_`). **Conserto:** dar retorno `bool` a `readback` (ou receber
`std::string &err`) e propagar; `want_hidden_` já controla se a cópia acontece.

### H3 — IMPORTANTE — `Graph::init` falha sem preencher `err`

`include/rdna4/graph.cuh:541`

```cpp
  if (hipMalloc(&d_pos_, sizeof(int)) != hipSuccess) return false;
```

Todos os outros 15 `hipMalloc`/`hipMemset` do `init` escrevem `err` antes de
devolver `false`; este não. **Cenário:** num OOM de 4 bytes (VRAM esgotada nesse
ponto) o chamador imprime `err` que sobrou de uma chamada anterior — em
`src/main.hip:1161-1164` isso vira `model init failed: <mensagem antiga>` ou uma
mensagem vazia, mandando o usuário investigar a coisa errada. **Conserto:**
`err = "hipMalloc failed (pos)";` antes do `return false` (mesmo texto do
equivalente em `mtp.cuh:359`).

### H4 — IMPORTANTE — resultado de `debug_fill_caches` descartado no bench

`src/main.hip:436`

```cpp
    if (fill_cache) (void)graph.debug_fill_caches((unsigned)seed, err);
```

**Cenário:** `bench --fill-cache --start-pos …` (a medição de contexto longo):
se o `kv_fill` falhar, o `bench` mede a atenção sobre VRAM não inicializada e
imprime números de tok/s como se estivesse tudo bem — o número não é
"errado" por pouco, é de outro experimento. **Conserto:**
`if (fill_cache && !graph.debug_fill_caches(seed, err)) { …; return 1; }`.

### H5 — COSMÉTICO — `hipMemGetInfo` descartado

`src/main.hip:446-448` (`(void)hipMemGetInfo(&free_b, &tot_b)`). `free_b`/`tot_b`
são inicializados com 0, então **não há UB** — a falha só imprime
`vram in use: 0.00 GiB (free 0.00 of 0.00 GiB)`. Se o valor for usado para
decidir algo no futuro, passa a ser enganoso. **Conserto:** checar o retorno e
imprimir "vram: indisponível" quando falhar.

### H6 — COSMÉTICO — 17 `hipFree` silenciosos em `release()`

`graph.cuh:1273,1289,1291,1292,1299,1304,1305,1308,1309,1310,1311` e
`mtp.cuh:655,664,666,667,668,669`. Ignorar o retorno de `hipFree` num destrutor
é defensável (não há o que fazer), mas uma falha ali **é** um vazamento de VRAM
e não deixa rastro nenhum. **Conserto:** acumular os erros num `bool` e, no
caminho de CLI/servidor, imprimir um aviso em `stderr` (não lançar).

### H7 — IMPORTANTE (testes) — 422 avisos `nodiscard` ignorados

Grep pedido no enunciado, build completo por omissão:

```
433 warning:   (total do build Release padrão)
422 ignoring return value of type 'hipError_t' declared with 'nodiscard'
  5 ‘IQ3S_N_SCALE’ redefinido            (header de terceiros, ver §4)
  4 sequência de escape hexa fora de alcance (tokenizer.cpp:140, VER §3 E1)
  2 format specifies type 'size_t' but the argument has type 'int' (check_kvctx_gpu.hip:289)
```

Os 422 estão **todos** em `tests/` e benches (218 em `check_matvec_gpu.hip`, 80
em `check_rope_gpu.hip`, 38 em `check_matmul_gpu.hip`, 36 em `bench_attn_gpu.hip`,
20 em `check_dequant_gpu.hip`, 16 em `check_nn_gpu.hip`, 10 em
`bench_matvec_shapes_gpu.hip`, 4 em `check_kvctx_gpu.hip`, 2… ), **0 em
`include/` e `src/`** — justamente porque ali ou se checa ou se escreve `(void)`.

O caso que importa não é o ruído: `hipEventCreate`/`hipEventRecord`/
`hipEventSynchronize`/`hipEventElapsedTime` sem checagem (54 sítios) significam
que **um bench pode imprimir `0.0000 ms` ou um tempo não medido** sem falhar —
por exemplo `tests/bench_matvec_shapes_gpu.hip:105-118` e `:253`. Nesses a
conversão de `(void)` seria enganosa; o certo é checar. **Ranking:** bench-only,
portanto abaixo de todos os achados de produção deste relatório, mas acima de
qualquer item de estilo.

Nota de honestidade: os 422 avisos são *por instanciação* — a contagem por linha
de código é bem menor (os mesmos 4 sítios de `check_matvec_gpu.hip` aparecem em
14 instanciações de template). Para o número por sítio, ver a tabela §1.1.

### H8 — IMPORTANTE (semântica das checagens) — `hipGetLastError` pós-launch detecta só erro de configuração

O padrão dominante do repo é:

```cpp
  kernel<<<grid, block, 0, stream>>>(...);
  return hipGetLastError() == hipSuccess;      // ex.: nn.cuh:121-122, attn.cuh:190-192
```

Isso é o idiom correto para *detectar* um lançamento inválido, mas tem duas
propriedades que valem registro porque a pergunta do enunciado é se a checagem é
"significativa":

1. **É assíncrono.** `hipGetLastError()` depois do `<<<>>>` só vê erros de
   configuração/argumento; uma falha *durante a execução* (fault de memória,
   `s_endpgm` com exceção) aparece muito depois, numa chamada qualquer. Como o
   motor só sincroniza nos `hipMemcpy` de leitura, uma falha de execução tende a
   ser atribuída ao `hipMemcpy` seguinte (`graph.cuh:1211` → `"logits readback
   failed"`), ou a **nada**, quando o resultado não é lido (bench com
   `--no-stats`).
2. **Erro "velho" vira falso positivo.** `hipGetLastError()` *limpa* o erro. As
   220 chamadas sem checagem em `tests/` deixam o último erro sujo; o próximo
   launcher que rodar `return hipGetLastError() == hipSuccess;` pode acusar
   falha de um kernel que rodou bem. **Conserto sugerido (uma linha, no
   idiom):** `(void)hipPeekAtLastError();` imediatamente antes do `<<<>>>` — ou
   checar com `hipPeekAtLastError()` depois —, o que isola a atribuição; e usar
   `hipDeviceSynchronize()` (checado) nos pontos em que o resultado é consumido.

### H9 — COSMÉTICO — conversão de string sem checagem no knob de diagnóstico

`include/rdna4/graph.cuh:314-316` (`atoi` em `RD_ATTN_SPLITS`, apontado por
`clang-tidy: bugprone-unchecked-string-to-number-conversion`). `RD_ATTN_SPLITS=abc`
vira 0 (política automática) e `RD_ATTN_SPLITS=4x` vira 4, silenciosamente.
**Conserto:** `strtol` com checagem de `end` e de intervalo, ou mensagem de erro.

---

## 2. Gerenciamento de memória

Mapa de alocação/liberação conferido ponta a ponta: `Graph::init` (19 `hipMalloc`
+ 2 em `alloc()`), `Graph::release`, os buffers de batch (`d_xb_`…`d_gate2b_`,
`d_aqb_`, `d_posb_`), o scratch da atenção (`d_attn_partial_`), o `MtpHead`
(16 `hipMalloc`), o KV (`kv.h`), os buffers de host do servidor (`http.cpp`,
`json.cpp`) e do loader/tokenizer.

**Resultado global (bom):** `Graph` e `MtpHead` têm destrutor que chama
`release()`, são não-copiáveis (`graph.cuh:76-78`, `mtp.cuh:91-94`), e o
`Engine` do servidor declara `GgufLoader loader_` (`serve.hip:341`) **antes** de
`std::unique_ptr<Graph> graph_` (`:344`), então o grafo morre primeiro — a
referência `const GgufLoader &ld_` (`graph.cuh:258`) nunca fica pendurada.
Todos os totais de bytes dos buffers de ativação vêm de `cfg`/`n_*()` e são
idênticos aos usados pelos kernels (os gates bit-exatos conferem). Não há
`new`/`malloc`/`mmap`/`pthread` em `src/` ou `include/`.

### M1 — IMPORTANTE — `Graph::release()` não é idempotente (18 ponteiros pendurados)

`include/rdna4/graph.cuh:1284-1303`

```cpp
  float *ptrs[] = {d_x_, d_xn_, d_proj_, d_ffnout_, d_attnout_, d_attngate_,
                   d_qkv_, d_conv_, d_z_, d_alpha_, d_beta_,
                   d_gate_, d_state_, d_convst_, d_ffn_a_, d_ffn_b_, d_kstage_,
                   d_vstage_};
  for (float *p : ptrs) {
    if (p) (void)hipFree(p);
  }
```

O laço libera e **não zera** os 18 membros. Todo o resto do método zera (os 13
ponteiros de batch em `:1301-1303`, `d_aqb_`/`d_posb_`/`d_q8_`/`d_pos_`/
`d_logits_`/`d_attn_partial_` em `:1306-1315`, e os `GpuTensor` via `fr()`), e
`d_k_`/`d_v_` também (`:1293-1294`) — a inconsistência é exatamente esse vetor.
**Cenário:** hoje `release()` só é chamado pelo destrutor (`graph.cuh:76`), então
não há duplo free *no código atual*; mas qualquer segunda chamada (um
`graph.release()` explícito para liberar VRAM, um `init()` de novo no mesmo
objeto, ou mover o `Graph` para um `optional`/reuso) faz `hipFree` sobre 18
ponteiros já liberados ⇒ corrupção do heap do runtime HIP, tipicamente um abort
dentro de `libamdhip64`. Num motor cujo produto é "caber em 16 GB", liberar e
reinicializar o grafo é uma operação plausível de se querer. **Conserto:** trocar
o array por um laço que zere (`for (float *&p : ptrs) { if (p) (void)hipFree(p); p = nullptr; }`),
o que também torna `release()` seguro de chamar duas vezes.

### M2 — IMPORTANTE (latente) — parte do buffer de parciais da atenção fica sem escrever

`include/rdna4/attn.cuh:392-397` (kernel com split) contra `:437-440` (launch):

```cpp
  if (threadIdx.x < head_dim) {                 // 392
    ...
    out[2 + threadIdx.x] = a;                   // 396
  }
```

O CTA tem `threads = WPB * 32` threads. Com o valor de envio (`WPB = 8`) isso dá
256 threads e `head_dim = 256`, então cobre exatamente. Mas a validação de
entrada aceita `head_dim` até `32 * kAttnMaxDimsPerLane = 512` (`:83`, `:187`,
`:435`): com `head_dim = 384` ou `512` (qualquer cabeça maior que 256), **os
elementos `[256, head_dim)` de `out[2..]` nunca são escritos**.
`d_attn_partial_` vem de `hipMalloc` sem `memset` (`graph.cuh:535`), e o
`attn_merge_kernel` (`:401-427`) lê `ps[2 + lane*dpw + i]` de todos os parciais
⇒ média ponderada contra VRAM não inicializada: saída não determinística, NaN
possível, e o gate bit-exato (que compara dois `hipMalloc`) pode até "passar" por
coincidência de zeros. **Cenário concreto:** qualquer variante de qwen35 com
`key_length > 256` (o modelo atual é 256, então hoje não dispara — daí
"latente"). **Conserto:** no `attn_launch_split_typed`, exigir
`head_dim <= WPB * 32` (ou fazer o laço de escrita das dimensões ser `for (int d = threadIdx.x; d < head_dim; d += blockDim.x)`); e, por robustez, um
`hipMemset(d_attn_partial_, 0, part_bytes)` no `init` para que uma falha de
lógica apareça como zero e não como lixo.

### M3 — IMPORTANTE — `MtpHead::proj` não valida o tamanho em bytes do tensor

`include/rdna4/mtp.cuh:382-387` checa só `dim0`/`dim1`:

```cpp
  if (w.dim0 != ncols || w.dim1 != nrows) {
    err = "MTP: proj shape mismatch";
```

O equivalente do tronco (`graph.cuh:562-568`) valida **também**
`w.bytes != tensor_bytes(w.dt, nrows*ncols)`. **Cenário:** um `blk.64.*` com dims
certas mas dtype/contagem de blocos inconsistente (GGUF malformado ou arquivo de
outro quant) passa, e o `matvec` lê `nbytes` além do fim do tensor — leitura
dentro da VRAM do próprio processo (o próximo tensor), portanto **sem fault**:
números errados em silêncio no rascunho. **Conserto:** replicar a comparação de
`bytes` (uma linha).

### M4 — IMPORTANTE — orçamento de VRAM do `serve` ignora o tipo do cache V

`src/server/serve.hip:373-380`

```cpp
  // Budget check before the (slow) load, like `info` does: a context that does
  // not fit must fail with a message, not with hipMalloc.
  const std::uint64_t kv_bytes =
      2 * (std::uint64_t)a.ctx_size * rdna4::kQwen35KvElemsPerToken *
      rdna4::kv_bytes_per_elem(a.kv_k) / 2;
```

`kv_bytes_per_elem(a.kv_v)` **nunca** entra na conta, embora `kQwen35KvElemsPerToken`
(`device.h:40`) conte as dimensões de K **e** de V e o `serve` aceite os dois
tipos independentes (`serve.hip:184-192`, `--cache-type-k`/`--cache-type-v`) e os
passe ao grafo (`:398`). **Cenário:** `serve -m modelo --ctx-size 45000 --cache-type-k f16 --cache-type-v f32`
subestima o KV em ~1/3 (o certo é `ctx*32768*3`, a conta usa `*2`), a checagem
de `:380` passa e o usuário recebe `graph init failed: hipMalloc failed (kv cache)`
— exatamente o que o comentário duas linhas acima promete evitar. **Conserto:**
`... * (kv_bytes_per_elem(a.kv_k) + kv_bytes_per_elem(a.kv_v)) / 2;` (é a
fórmula que `src/main.hip:902-905` já usa, somando `kv_k` e `kv_v` separados).
O `2 * … / 2` é resíduo: sobra da conta em dobro e não faz nada.

Nota de coerência: esse mesmo cálculo existe em **três** lugares — `main.hip:902-905`
(certo), `serve.hip:374-378` (errado) e `device.h:51-56 required_bytes()`, que é
**código morto** (0 chamadas, ver §3 E8) — ou seja, o helper compartilhado que
existe para não deixar os dois divergirem é justamente o que não é usado.

### M5 — IMPORTANTE — `-(int)` do `ctx_size` (u64) sem validação, e orçamento que estoura

`src/server/serve.hip:78-85` (`parse_u64` aceita até 2^64-1) → `:398`
`graph_->init(static_cast<int>(a.ctx_size), …)` mas `:504` guarda o valor **não
truncado** em `ctx_size_`, usado em `:567` (`if (pos >= ctx)`), e `:322` devolve
`static_cast<int>(ctx_size_)`. Somado a isso, `2 * (std::uint64_t)a.ctx_size`
(`:376`) é uma multiplicação sem checagem. **Cenário:** `serve --ctx-size 9223372036854800000`
faz `2*ctx` dar a volta, o orçamento passa com ~6,5 GB, `graph.init(100000)`
sucede, `fit_context` (`:928`) limita o `max_tokens` a 100000 mas o laço de
`generate` só para em 2^63+10^5 ⇒ o pedido morre no meio do stream com 500
(ou exit 1 no CLI, `main.hip:1161`), depois de já ter emitido texto.
**Confiança:** alta no truncamento e no overflow; **baixa** de que exista falha
de memória — todas as posições são validadas em `graph.cuh:1031` e `:1133`, então
não há escrita fora das linhas do KV. **Conserto:** rejeitar
`ctx_size > INT_MAX` logo depois do parse (em `serve.hip` e nos laços de
`main.hip`) e usar aritmética checada no orçamento.

### M6 — IMPORTANTE — `block_count` do arquivo vira tamanho de container sem limite superior

`src/backend/model.cpp:337-338`

```cpp
  std::vector<std::map<std::string, std::vector<std::int64_t>>>
      block_tensors(cfg.block_count);
```

`cfg.block_count` é o U32 `qwen35.block_count`, validado só por `< 2`
(`model.cpp:238`); não há teto. **Cenário:** GGUF com `block_count = 10000000`
aloca ~0,5 GB de mapas e depois roda 10^7 iterações de `expected_block_tensors()`
(`:372`), cada uma criando 11-14 `std::pair<string, vector>` — minutos de CPU num
`info`/`serve` que deveria ser rejeitado na hora; com `2^32-1` o construtor do
vetor lança `bad_alloc`, que **não** é capturado em lugar nenhum do caminho de
carga (único `catch` do loader: `model.cpp:354-359`, em volta de `std::stoul`)
⇒ `std::terminate`. **Conserto:** `if (cfg.block_count > 1024) { err = …; return false; }`
ao lado da checagem que já existe.

### M7 — CRÍTICO — overflow nas somas de geometria do loader ⇒ lê o offset errado e "carrega" lixo

`src/backend/loader.cpp:114-133` — a **única** validação de offset/tamanho que
existe:

```cpp
    const std::uint64_t abs_end = doff + f_.tensors[i].offset + bytes_[i];   // :117
    ...
    if (abs_end > bound) { … "size overflows into next tensor / EOF" … }     // :120
    if (align != 0 && abs_end % align != 0) { … "end not aligned" … }        // :127
```

Os três operandos vêm do arquivo: `t.offset` é lido cru
(`src/backend/gguf.cpp:182 t.offset = c.get<std::uint64_t>();`), `bytes_[i]`
vem de `tensor_bytes()` (`include/rdna4/dtype.h:100-105`), que multiplica
`blocks * dtype_block_bytes(t)` **sem** checagem de overflow; só um resultado
exatamente `== 0` é rejeitado (`loader.cpp:95`). A mesma soma é refeita no
momento da leitura: `src/backend/loader.cpp:157`
`const long where = static_cast<long>(f_.data_offset + t.offset + off);`.
**Cenário:** tensor F32 `dims=[8]` (`bytes=32`), `doff=128`, `align=32`,
`offset = 2^64 - 128 + 32`. Então `abs_end = (128 + offset + 32) mod 2^64 = 64`,
que é `<= bound` (`file_size`) e múltiplo de 32: **as duas guardas passam**; o
teste interno de `load_tensor_range` (`:153 if (off + count > bytes_[i])`) também
passa; `where` dá a volta para 32 — o `fseek` cai dentro do cabeçalho do GGUF, o
`fread` devolve 32 bytes válidos, a função retorna `true` e o `Graph::up()`
(`graph.cuh:358-363`) sobe esses bytes **como pesos**: o modelo "carrega" e o
decode dá números errados sem nenhum aviso. A variante oposta (`dims=[INT64_MAX]`
em F32 ⇒ `bytes = 2^64-4`, ≠ 0, então passa por `:95`) estoura em
`out.resize(count)` (`loader.cpp:162`) com `std::length_error` não capturado ⇒
abort. **Conserto:** rejeitar `t.offset > file_size` e trocar as três somas por
adições checadas (`if (a > UINT64_MAX - b) fail;`), em `:117`, `:153` e `:157`.
**Confiança:** alta na aritmética (todas as linhas lidas); média de que um
arquivo forjado completo passe por `validate_qwen35_layout` — as *dims* precisam
ser consistentes, mas os *offsets*, que é onde mora o wrap, são livres e podem
ser construídos coerentes entre si; não foi executado nada.

### M8 — CRÍTICO — comprimento vindo do arquivo usado direto como tamanho de alocação

`src/backend/gguf.cpp:26-36`

```cpp
  std::string get_str() {
    const std::uint64_t n = get<std::uint64_t>();
    ...
    s.resize(n);                       // :32
```

Alcançado por nome de tensor (`:171`), chave de KV (`:153`) e valor STR (`:76`).
**Cenário:** GGUF corrompido com uma string de 2^40 bytes ⇒ `resize` tenta ~1 TiB
(`std::bad_alloc`) ou lança `std::length_error` quando `n > max_size()`; nada no
caminho (`cmd_*` → `GgufLoader::open` → `gguf::read` → `get_str`) captura, então
o processo aborta em vez de imprimir `err`. Mesma classe uma linha abaixo, com
teto que é contagem e não bytes — `gguf.cpp:90-94`:

```cpp
      if (n > (1u << 28)) { c.ok = false; }   // "sanity cap: 256M elementos"
      out.arr.resize(n);
```

2^28 `gguf::Value` (string + 2 vetores) são ≈ 21 GB alocados antes de ler o
primeiro elemento. **Confiança:** alta de que a checagem não existe; média sobre
qual dos três desfechos (abort/OOM/erro limpo) acontece — com overcommit um `n`
grande mas não absurdo pode passar e falhar limpo no `fread`. **Conserto:**
limitar `n` por bytes (`n * sizeof(Value) <= kMaxKvBytes`) nos dois lugares, com
`err` em vez de `c.ok = false` silencioso.

### M9 — COSMÉTICO — donos de recurso copiáveis/reabríveis, acessor sem bounds

- `src/backend/loader.cpp:30` `fp_ = std::fopen(path, "rb");` — um segundo
  `open()` no mesmo loader sobrescreve `fp_` sem `fclose` (vaza um `FILE*` e o
  descritor). Nenhum chamador reabre hoje (todos os `cmd_*` usam um loader novo),
  então é latente; o destrutor está correto (`loader.cpp:24-27`) e a cópia é
  deletada (`include/rdna4/loader.h:30-31`). **Conserto:** `if (fp_) std::fclose(fp_);`
  no topo de `open()`.
- `include/rdna4/server.h:120` (classe `HttpServer`) é implicitamente copiável e
  tem `~HttpServer` fechando `listen_fd_` (`src/server/http.cpp:550-555`): uma
  cópia fecha o mesmo fd duas vezes. Não é copiado na árvore (`serve.hip:1266`).
  **Conserto:** `HttpServer(const HttpServer &) = delete;`.
- `include/rdna4/tokenizer.h:75` `const std::string &token_text(std::int32_t id) const { return tokens_[id]; }`
  não valida `id` enquanto o irmão valida (`src/backend/tokenizer.cpp:273`).
  Latente (o único chamador passa id vindo do corpus, `main.hip:773`).
- `src/backend/sampler.cpp:16-17` reserva `cands_` (membro em `sampler.h:67`)
  que nenhuma outra linha usa: ~512 KB de heap ociosos por processo. Ver E8.

### M10 — IMPORTANTE (padrão) — tamanhos de buffer calculados em `int` e só depois alargados

`clang-tidy` (`bugprone-implicit-widening-of-multiplication-result`) aponta **186
avisos em 47 linhas distintas**, todas do padrão `alloc(p, A * B, err)` ou
`*ptr + i * A * B` com `A`, `B` `int` vindos de `cfg`/`n_*()` e o produto só
convertido para `std::size_t`/`int64_t` **depois** de calculado em `int`. As mais
relevantes para o gerenciamento de memória:

| linha | expressão | alimenta |
|---|---|---|
| `include/rdna4/graph.cuh:486:72` | `NH * 2 * HD` | `alloc(d_proj_, …)` |
| `include/rdna4/graph.cuh:487:55` | `NH * HD` | `alloc(d_attnout_, …)`, `d_attngate_` |
| `include/rdna4/graph.cuh:503:25,61` | `NKV * HD` | `alloc(d_kstage_, …)`, `d_vstage_` |
| `include/rdna4/graph.cuh:761:16` | `2 * key_dim` | ponteiros no meio de `d_conv_` |
| `include/rdna4/graph.cuh:911,912,955` | `NH * HD` | offsets no batch |
| `include/rdna4/mtp.cuh:340-345` | `2 * E`, `2 * NH * HD`, `NH * HD`, `NKV * HD` | os 16 buffers do bloco MTP |
| `include/rdna4/kv.h:259,286,287` | `blk * 32 + lane` | índices no cache |

**Cenário:** um GGUF com `head_count`/`key_length` grandes (ou
`embedding_length`/`ssm_*` escolhidos para estourar `int`) faz o produto dar a
volta **antes** de virar `size_t` e alocar um buffer pequeno; a validação de
layout compara *dims*, não o produto, então o caminho fica: buffer pequeno +
kernel indexando pelos mesmos números estourados. Não há entrada hoje que
dispare (o modelo é fixo e `validate_qwen35_layout` amarra as dims aos tensores),
por isso **não** é CRÍTICO — mas é a classe "buffer dimensionado a partir de
config não validada" que o enunciado pede. **Conserto:** converter na *primeira*
multiplicação (`(std::size_t)NH * 2 * HD`) nos 47 sítios, ou centralizar um
`bytes_of(rows, cols)` checado.

---

## 3. Estilo, nomenclatura, duplicação, código morto e comentários que mentem

### 3.1 O que está bem (para não "consertar" o que está certo)

- **0 `using namespace`** em qualquer header (e na árvore toda). **0 `#if 0`.**
  **0 `TODO`/`FIXME`/`XXX`/`HACK`.** **0 blocos de código comentado.**
- **0 identificadores camelCase** em variáveis locais, funções, membros ou
  kernels: os 22 kernels são todos `snake_case` + sufixo `_kernel` (22/22). O
  camelCase que existe é a família de constantes `kPascalCase` (48 nomes, 371
  ocorrências) — convenção consistente, não violação.
- **0 dimensões de modelo hard-coded** nos kernels: `151936` não aparece nenhuma
  vez na árvore; `5120` só em comentário (`mtp.cuh:64`). A geometria vem de
  `cfg_.*`/`n_embd()`/`n_head()`.
- Os parâmetros de kernel/launcher são 100 % `d_`-prefixados.

### E1 — CRÍTICO — escape hexadecimal corrompe o literal de EOG (bytes provados)

`src/backend/tokenizer.cpp:140`

```cpp
      "<turn|>", "<|tool_response>", "<\xEF\xBD\x9Cend\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>",
```

`\x9Cend` é interpretado por *maximal munch*: os dígitos hexadecimais são
`9`,`C`,`e` (o `e` de "end" **é** um dígito hexadecimal), então o escape vale
`0x9CE` — fora do alcance de `char` — e o `e` de "end" é engolido. O compilador
avisa (`warning: sequência de escape hexa fora de alcance`, 4× no build padrão) e
trunca para `0xCE`. Prova nos objetos compilados (`/tmp/b-default/rdna4-infer`,
`check-eog`, `check-tokenizer`, com Python sobre os bytes):

```
corrupt-munch=True   intended=False
literal:  3C EF BD CE 6E 64 …   (< U+FF4E? "nd" …)
esperado: 3C EF BD 9C 65 6E 64 … (< ｜ e n d …)
```

O texto pretendido é `＜|end▁of▁sentence|＞` (U+FF5C = `EF BD 9C`), que é o token
de fim do DeepSeek/Qwen listado em `llama.cpp:src/llama-vocab.cpp:2680`
(`t.first == "<｜end▁of▁sentence｜>"`, lá escrito em UTF-8 cru — a transcrição
para escapes hex foi o que introduziu o defeito). Versão do mesmo literal escrita
com concatenção de strings (`"\xEF\xBD\x9C" "end…"`) dá os bytes corretos.
**Cenário:** qualquer vocab que contenha esse token (variantes DeepSeek/Qwen3
fora dos dois UD atuais) tem o EOG silenciosamente **fora** do conjunto de parada
⇒ a geração não para no fim da resposta e vai até `-n`/`max_tokens`.
**Impacto hoje:** **nenhum** nos dois arquivos UD — o golden mostra o conjunto
real de EOG (`tests/golden/eog_ids.txt`: 248044, 248046, 248063, 248064, 248065)
e nenhum deles é esse token, o que explica o `check-eog` verde. É um defeito
objetivo e comprovado, mascarado pelo vocab. **Conserto:** escrever o literal em
UTF-8 cru (como no llama.cpp) ou separar em dois literais adjacentes; e um caso
de teste que compare os bytes do texto do EOG com o esperado, não só os ids.

### E2 — IMPORTANTE — o parâmetro de template `WPB` do kernel sem split é ignorado

`include/rdna4/attn.cuh:91-181`: o kernel é `template <KvType KT, KvType VT, int WPB = kAttnWarpsPerBlock>`,
o lançamento passa `threads = WPB * 32` (`:188-190`), mas o **corpo** usa a
constante global em todos os quatro pontos que dependem de WPB:

```cpp
  for (int j = w; j <= t; j += kAttnWarpsPerBlock) {     // :118
  for (int s = 0; s < kAttnWarpsPerBlock; ++s) mm = …    // :162
  for (int s = 0; s < kAttnWarpsPerBlock; ++s) {         // :168
    for (int s = 0; s < kAttnWarpsPerBlock; ++s) {       // :175
```

(o kernel **com** split usa `WPB` corretamente: `:315-316, 360, 365, 370, 377, 394`).
**Consequência:** com `WPB = 16/32` (só alcançável por `attn_launch_wpb`,
`:198-214`, cujo único chamador é `tests/bench_attn_gpu.hip:233`), os warps
`8..WPB-1` recalculam as mesmas fatias de chave dos warps `w mod 8` e suas
parciais de shared memory **nunca são mescladas** — o resultado é idêntico ao de
8 warps, com 2-4× o trabalho. Ou seja: a "varredura WPB do kernel sem split" que
o bench imprime (`bench_attn_gpu.hip:227-247`, comentário `:224-228`) e que
`docs/rocm-estudo.md:347-349` cita ("no kernel **sem split** dá 0,73× a 4K e
1,07-1,19× a 8-64K") mede **warps redundantes**, não "mais warps dividindo a
faixa de chaves"; a conclusão de política que ela sustenta não se apoia nela.
A tabela de `splits × WPB` (`attn.cuh:505-512`, `rocm-estudo.md:468-473`) é do
kernel *com* split e **não** é afetada. O valor enviado (`8`) continua certo.
**Conserto:** usar `WPB` nos quatro pontos (ou falhar alto quando
`WPB != kAttnWarpsPerBlock`), e re-rotular/re-medir os números do sweep sem split.

### E3 — IMPORTANTE — comentário e doc que descrevem um kernel sem split que não existe mais

`include/rdna4/attn.cuh:85-90` e `:195-197` afirmam que mudar `WPB` "changes the
number of partial slices merged at the end of the CTA, i.e. the summation order
across key slices". No kernel sem split isso é **falso** (E2: a ordem de soma é
sempre a de 8 fatias). **Conserto:** corrigir os dois comentários junto com E2.

### E4 — IMPORTANTE — o README dá dois números para a mesma medição e um warp que não existe

- `README.md:134` — "Vectorized cache loads, **32 warps/CTA** and splitting the
  key range across CTAs (with an online-softmax merge) bought 1.1-5.8×" e
  `SPEC.md:74` — "cargas vetorizadas, **32 warps por CTA** e a faixa de chaves
  dividida entre CTAs". O valor que roda é `kAttnWarpsPerBlock = 8`
  (`include/rdna4/attn.cuh:82`, com o comentário `// key slices per block
  (shipped value)`). O próprio repo já registra a correção
  (`docs/medicoes-m7.md:29-36`, `docs/agentes-paralelos.md:69-70`,
  `docs/rocm-estudo.md:347-352`) — **`docs/medicoes-m7.md` está certo**; o que
  ficou para trás foram o README e o SPEC. Com split ativo o kernel usa 16 ou 8
  warps por CTA (`attn.cuh:535-538`).
- `README.md:132-133` — "decode is 23.7 tok/s at 4K, 24.1 at 16K, 19.0 at 64K
  (f16 KV) and 13.2 at 131K" contradiz **o mesmo README**: `README.md:94`
  (`26.8 tok/s` a 4K f16) e `README.md:99` (`18.9 / 13.0`). O texto de `:132-134`
  é a prosa pré-M7 de `docs/medicoes-m7.md:116`; a árvore final medida em
  `docs/agentes-paralelos.md:59-60` é **26,8 / 24,3 / 18,9 / 13,0**.
  **Conserto:** trocar os dois trechos (o número certo já está no repo) e apontar
  o "1.2–5.7×".
- `README.md:138` — "`f16` KV … is 38 % faster than `q4_0` at the same context"
  é o número de **M5** (174 vs 239 ms/token a 64K, `docs/medicoes-m5.md:135`);
  depois de M7/M8 é ~6 % (18,91 vs 17,88 tok/s, `docs/medicoes-m7.md:107-108`), e
  `SPEC.md:78-79` diz explicitamente que "em 64K o f16 passou a ser mais rápido
  que o q4_0 (19,0 vs 17,8)". **Conserto:** "~6 % depois de M7/M8 (era 38 % antes
  das reescritas da atenção)".

### E5 — IMPORTANTE — comentários de código que o código já não cumpre

| local | comentário | realidade |
|---|---|---|
| `include/rdna4/graph.cuh:15-16` | "Full-attention layers keep a f32 KV cache (quantized cache types are a later step)." | `init` recebe `KvType kv_k, kv_v` (`:82`) e o default do CLI/servidor é **F16** (`main.hip:927`, `serve.hip:147`); Q8_0/Q4_0 existem e são medidos. Só o overload de teste (`:83`) usa F32. |
| `include/rdna4/device.h:33` | "17 full-attention layers" | 4 linhas abaixo o mesmo comentário diz "16 KV-bearing full-attention layers"; o `PLAN.md:174` registra o bug do 17 como corrigido no M5. |
| `include/rdna4/device.h:38` | "The 65th block is the MTP block, **which v1 does not run**, so it holds no KV." | O bloco MTP roda desde `feat/mtp` (`main.hip:1046-1057`, `--mtp`) e o `MtpHead` **tem** KV próprio (`mtp.cuh:365-379`) — que, aliás, o orçamento de `info`/`serve` não conta. |
| `src/main.hip:194-195` | "prefill: the per-token path (**this engine has no batched prefill**), so the prompt is timed token by token" | `main.hip:183` chama `forward_batch`, `main.hip:463` imprime "batched N<=16", e `graph.cuh:150` é o `forward_batch` do M8. |
| `include/rdna4/device.h:33-36` | "M1/M3 replace the layer/head/dim constants with real hparams from the GGUF." | M1-M3 estão concluídos e `kQwen35KvElemsPerToken` continua fixo em `16u*4u*(256u+256u)` (`:40`), usado no orçamento de `main.hip:902` e `serve.hip:376`. |
| `SPEC.md:33` | "Kernels GEMM/matvec reaproveitados do llama.cpp com tuning RDNA4 (`MMVQ_PARAMETERS_RDNA4`, `mmq-config-rdna4.cuh`)" | Nenhum dos dois existe na árvore (só citações em `PLAN.md:164,182` e docs) e **não há GEMM/MMQ**: o caminho em batch é GEMV batched (`matvec_launch_batch`). |
| `SPEC.md:95-97` | "prefill é a lacuna grande (16×), porque este motor processa o prompt token a token" | Falso desde M8: `prefill_ids` usa `forward_batch` (`main.hip:167-188`) e a medição é 69,5 tok/s a 512 tokens (`docs/medicoes-m8.md:44`) contra 17,75 s → 7,37 s (2,45×). O SPEC §3 (`:91-92`) ainda traz a coluna de prefill antiga. |
| `SPEC.md:21` vs `dtype.h:10-11` | SPEC lista "+ `F16`/`BF16` se aparecerem" no whitelist de tipos | `dtype.h` diz explicitamente que F16/Q4_0 **não** estão no whitelist e `dtype_from_ggml` não tem caso para eles. O `PLAN.md:178` já registra isso como pendência da revisão. |
| `SPEC.md:42-43`, `SPEC.md:106`, `docs/exploracao-qwen38.md:17` | "MTP/draft heads, speculative decoding" e "servidor HTTP/OpenAI-compatible" listados como **não-objetivo** (§2) e, no §4 "Futuro", "MTP … hoje lidos e ignorados" | Ambos existem, são testados e documentados (`src/server/**`, `include/rdna4/mtp.cuh`, `check-mtp-gpu`, `docs/servidor-openai.md`, `docs/mtp.md`). |
| `README.md:32` | "16 checks (CPU + GPU)" | São **19** alvos `check-*` em `CMakeLists.txt` (o 20º arquivo é `dequant_cpu_oracle.cpp`, TU de apoio). |
| `README.md:155` | "`PLAN.md` — M0–M5 milestones" | `PLAN.md` vai até **M6** (`PLAN.md:506`) e a árvore já tem M7/M8 (`docs/medicoes-m7.md`, `medicoes-m8.md`, commits `fec7065`/`39feff8`). |
| `README.md:38-39` | "one binary plus one test binary per check; there is no library to install" | Existe `librdna4_serve.so` (`CMakeLists.txt:229-234`), documentada em `docs/servidor-openai.md:191`. |
| `README.md:29` | "src/main.hip # CLI: info \| run \| bench \| ppl \| tokenize" | São 6 subcomandos: falta `serve` (`main.hip:1421-1437`). |
| `README.md:121` | "The oracles (`build/oracle-*`) are the only binaries that link llama.cpp" | Seis alvos `check-*` linkam `libggml-base.so` (`CMakeLists.txt:68,80,92,102,141,151`). |
| `README.md:84-85` | "Exit codes: … `3` insufficient VRAM" | `return 3` só existe em `main.hip:912` (`cmd_info`) e em `serve.hip:1277`; o `cmd_run` mapeia qualquer falha de init para 1 (`main.hip:1161-1164`), inclusive `hipMalloc failed (kv cache)`. |
| `docs/medicoes-m6.md:22-23` | "Timing one 20-30 MB tensor repeatedly keeps it in the **64 MB L2**" | O próprio repo diz que o L2 é **8 MiB** e os 64 MB são o Infinity Cache (`docs/rdna4-gfx1201-hardware-brief.md:13-14`, `docs/rocm-estudo.md:24-29`). O rótulo do nível de cache está errado (a conclusão metodológica, não). |
| `docs/mtp.md:269` | "a cache KV do bloco é f16 e custa 8 MiB a 4K (2 MB por 1K …), então a 64K são 128 MiB" | `mtp.cuh:367-368` aloca **duas** caches (`d_ck_` **e** `d_cv_`), 4 KiB/token ⇒ 4 MB por 1K, 16 MiB a 4K, 256 MiB a 64K. |
| `scripts/check_attn_split.sh:28` | "(see `scripts/get-wikitext-2.sh`)" | O script não existe em `scripts/` nem em lugar nenhum da árvore. |
| `tests/README.md:5-6` e `scripts/README.md:7-8` | "a definir em cada milestone" / "Futuros: `bench.py`, `smoke.py`" | Todos os milestones estão fechados; existem 19 checks e os equivalentes a `bench`/`smoke` são subcomandos + `scripts/check_golden_run.sh`. O `scripts/README.md` documenta 2 dos 11 scripts. |

Ver também E9 (`required_bytes`) e E10 (`attn_splits_last`), que são comentários
certos em volta de código morto.

### E6 — IMPORTANTE — duplicação entre o caminho por token e o caminho em batch

`include/rdna4/graph.cuh` mantém duas implementações da mesma aritmética, com a
obrigação explícita de serem **bit-idênticas** (`graph.cuh:139-151`, gate
`tests/check_batch_gpu.hip`). Duplicações medidas:

| bloco | cópia por token | cópia em batch | linhas iguais (normalizadas) |
|---|---|---|---|
| recorrência GDN (`sigmoid`→`+dt`→`softplus`→`*ssm_a`→`conv1d`→`l2`→`delta_rule`→`ssm_norm`→`silu`→`mul`→`copy`) | `746-782` (37 L) | `945-971` (27 L) | 6 (o resto difere só nos nomes dos buffers e no prefixo `"batch "` das mensagens) |
| cauda da atenção (write do KV → decisão de split → attn → sigmoid → mul) | `662-685` (24 L) | `895-912` (18 L) | 8 |
| norm final + LM head + alocação/leitura dos logits | `1180-1219` (40 L) | `1066-1089` (24 L) | 13 |
| validação de token id (mensagens byte-idênticas) | `1114-1126` | `1023-1036` | 10 |
| desempacotamento de dims (`d_inner`,`key_dim`,`chan`) | `721-723` | `922-924` | 6/6 (e uma **terceira** cópia em `397-399`) |
| FFN (gate→silu→up→mul→down→residual) | `797-810` | `996-1008` | 3 |
| laço `slot` (`is_recr`) | `725-726` | `926-927` | 2 (e ainda `count_recr()` `:285-289` e `attn_slot()` `:321-325`) |

≈ **150 linhas duplicadas** em um arquivo. Cada operação aritmética nova tem de
ser escrita duas vezes, e é a segunda que costuma esquecer (a ordem `enorm` antes
de `hnorm` do MTP, o `NaN` do split vazio, o `+8` do scratch já foram correções
pontuais desse tipo). **Conserto (descritivo):** extrair o corpo por token para
funções que operam sobre "a linha `t` de um buffer de N linhas" (stride
parametrizado), de modo que o caminho em batch seja um laço `for (t…)` sobre a
mesma função — a identidade bit-exata passa a ser garantida por construção, e o
gate continua valendo como rede.

### E7 — COSMÉTICO — convenção `_launch` e prefixo `d_`

- **32 definições de launcher** (30 nomes; `attn_launch_split_wpb` e
  `kv_store_row_launch` têm 2 overloads) em `include/rdna4/`: `matvec.cuh` 10,
  `attn.cuh` 9, `nn.cuh` 6, `gdn.cuh` 3, `kv.h` 3, `dequant_row.cuh` 1. Sete
  funções que lançam kernel **não** seguem o sufixo: `Graph::proj`
  (`graph.cuh:557`, lança em `:574`), `Graph::add_residual` (`:606`, lança em
  `:608`), `Graph::debug_fill_caches` (`:1238`, lança em `:1259`/`:1261`),
  `MtpHead::proj` (`mtp.cuh:382`, lança em `:393`), e `launch_gen`/
  `launch_shape_tuned`/`launch_read_only` (`matvec.cuh:434/605/692`) que usam o
  prefixo em vez do sufixo.
- **Assimetria:** existe `quantize_q8_1_batch_launch` (`matvec.cuh:77`) mas
  **não** existe `quantize_q8_1_launch`; o kernel cru é lançado inline em
  `graph.cuh:574`, `mtp.cuh:393` e 6 sítios de teste — ou seja, o único
  lançamento de produção **sem** launcher com checagem encapsulada é justamente
  o do caminho quente (a checagem existe inline em `graph.cuh:575`, o que salva
  o caso).
- **`d_`:** 20 alvos de `hipMalloc` em `include/` + `src/` (74 contando testes);
  17/20 com prefixo. Os 3 sem (`graph.cuh:376`, `graph.cuh:511`, `mtp.cuh:243`)
  são o parâmetro de saída de `alloc()`/`balloc()`, onde o nome `p` é local ao
  helper — aceitável, mas é a exceção que faz a regra "todo handle de VRAM é
  `d_*`" precisar de ressalva. A quebra real está nas **views por offset** dentro
  de `graph.cuh` (28 sítios: `:664-665 kc/vc`, `:697-699 base/krow/vrow`,
  `:727-728 state/convst`, `:759-761 q_c/k_c/v_c`, `:874-878 q/k/v/an/ag`,
  `:896-897`, `:928-944`, `:953-955`, `:1066 xlast`) — metade delas sem prefixo,
  o que dificulta ver de relance que apontam para dentro de um `d_*`.
- **Tipos de largura fixa:** `std::int64_t` 280 usos contra `int64_t` **137**
  (33 %); `std::uint64_t` 166 contra 3; `std::int32_t` 203 contra 14. Todos os
  `int64_t`/`uint64_t` crus estão em `matvec.cuh` (56), `dequant.cuh` (50) e
  testes (31) — são os arquivos vendorados do llama.cpp. **Conserto:** normalizar
  ao mexer neles.
- **Intrínsecos:** `__shfl_xor` (sem `_sync`, deprecado) em
  `matvec.cuh:687` (`read_only_kernel`) contra `__shfl_xor_sync` no resto.

### E8 — COSMÉTICO — código morto com prova (1 ocorrência na árvore = a definição)

| local | símbolo | prova |
|---|---|---|
| `include/rdna4/nn.cuh:152` | `scale_launch` (e o `scale_kernel` de `:110`, que só ele alcança) | 1 ocorrência |
| `include/rdna4/device.h:51` | `required_bytes()` | 1 ocorrência; os chamadores previstos (`main.hip:902-906`, `serve.hip:374-378`) reimplementaram a fórmula **e divergiram** (M4) |
| `include/rdna4/matvec.cuh:501` | `matvec_default_ilp` | 1 (substituído por `MtIlp<Dt>::value`) |
| `include/rdna4/vecdotq.cuh:58` | `__vcmpeq4` | 1 (`__vcmpne4` em `:70` **é** usado) |
| `include/rdna4/vecdotq.cuh:102` | `get_int_b1` | 1 (`get_int_b2`/`get_int_b4` usados) |
| `include/rdna4/matvec.cuh:123-125` | traits `TIQ2XXS_PERM2`, `TIQ2XS_PERM2`, `TIQ2S_PERM2` | 1 cada (a implementação `_perm2` continua viva via `*_S`) |
| `include/rdna4/graph.cuh:324` | `attn_splits_last()` | getter sem chamador; o campo é **write-only** (`:670`, `:899`) |
| `include/rdna4/model.h:49,50,51` | `bos_token_id`, `eos_token_id`, `pad_token_id` | escritos em `model.cpp:305-313`, **nunca lidos** (o tokenizer relê as mesmas KVs em `tokenizer.cpp:120-121`) |
| `include/rdna4/sampler.h:67` + `sampler.cpp:16-17` | `cands_` | só `clear()`/`reserve()` (~512 KB ociosos) |
| `src/backend/model.cpp:51` | `prod()` | 1 ocorrência; avisado pelo compilador (`-Wunused-function`) |
| `src/backend/unicode.cpp:24` | `unicode_cpts_to_utf8` | avisado pelo compilador (`-Wunused-function`) |
| `src/backend/model.cpp:450` | `(void)need;` | `need` **é** usado em `:440`; a linha é resíduo |
| `include/rdna4/kv.h:295` | `(void)amx;` | `amx` é calculado em `:288` e nunca usado — a redução `__shfl_xor_sync` de `:271` também é desperdiçada nesse ramo |
| `include/rdna4/attn.cuh:107-108` | `const char *kbase = (const char *)k + ((std::int64_t)kvh * head_dim) * 0; (void)kbase;` | multiplicação por zero e descarte: duas linhas mortas |
| `include/rdna4/graph.cuh:1056` | `const int L = n_layer();` | avisado (`-Wunused-variable`); o laço usa `debug_layer_limit()` |
| `src/main.hip:476` | `(void)total_tokens_timed;` | variável só para silenciar |
| `include/rdna4/mtp.cuh:153` / `graph.cuh:232-248` | métodos privados declarados e usados (falso positivo do meu sweep — ver §5) | — |

**Instrumentação de diagnóstico test-only** (a família `TIQ3S_NOSIGN` citada no
enunciado): `matvec.cuh:138-140` define `TIQ3XXS_NOSIGN`, `TIQ3S_NOSIGN`,
`TIQ3S_NOLOOKUP`, alcançáveis **apenas** por `matvec_launch_variant`
(`matvec.cuh:857`, casos `// DIAG` em `:885-888`), cujo único chamador é
`tests/check_matvec_gpu.hip`. O mesmo vale para `TIQ3S_LIN`/`TIQ3S_XORADD`
(`:119-121`) e para `matvec_launch_tuned`/`_rows`/`_unroll`/`_minb`/
`_read_only`/`matvec_kernel_attrs`/`MatvecShape`/`matvec_config`: 15 símbolos
públicos de header de produção sem nenhum chamador em `src/`. **Não** recomendo
apagar sem confirmar a intenção: `docs/rocm-estudo.md:461` e os docs de medição
os apresentam como evidência de A/B (instrumentação permanente), e o custo é só
de compilação. O que **recomendo** é marcá-los (`// bench-only, sem uso em src/`)
para que ninguém os trate como API.

### E9 — COSMÉTICO — números mágicos que deveriam ser constantes nomeadas

A convenção dominante e correta é `kPascalCase` (48 nomes, 371 usos:
`kAttnWarpsPerBlock` `attn.cuh:82`, `kMaxBatch` `graph.cuh:152`,
`kAttnSplitMin` `graph.cuh:308`, `kRmsNormThreads` `nn.cuh:21`,
`kOverheadBytes` `device.h:49`). Destoam:

| local | literal | significado |
|---|---|---|
| `include/rdna4/gdn.cuh:42`, `:114`, `include/rdna4/graph.cuh:1255` | `256` | threads por CTA (`kRmsNormThreads` existe e vale o mesmo) |
| `include/rdna4/graph.cuh:574`, `include/rdna4/mtp.cuh:393` | `(nb + 3) / 4, 128` | 4 = empacotamento por lane, 128 = CTA — 4 números sem nome numa linha |
| `include/rdna4/graph.cuh:608` | `(n + 255) / 256), 256` | idem, para o kernel de perturbação |
| `include/rdna4/dequant_row.cuh:76` | `4096` | teto da grade (grid-stride) |
| `include/rdna4/matvec.cuh:762` | `1024 / (… * 32)` | 1024 = limite de threads, 32 = warp |
| `include/rdna4/graph.cuh:549` | `+ 8` | folga do scratch de q8 |
| `include/rdna4/attn.cuh:538` | `? 16 :` | mistura o nomeado `kAttnSplitWpbLimit`/`kAttnWarpsPerBlock` com um 16 cru (é o `WPB` preferido — merece nome) |
| `include/rdna4/nn.cuh:85` | `20.0f` | limiar do clamp do softplus (fiel ao ggml; só falta nome) |
| `include/rdna4/matvec.cuh:195` | `constexpr int PF_DIST = 2;` | **único** `constexpr` em SCREAMING_SNAKE da árvore, 4 linhas acima de `constexpr` em snake_case (`:199`) |

### E10 — COSMÉTICO — três escalas de nomenclatura para a mesma coisa

Sete enums usam **cinco** convenções de enumerador: `ValueType` (SCREAMING,
`gguf.h:16`), `DType` (SCREAMING com `_`, `dtype.h:20`), `KvType` (SCREAMING,
`kv.h:22`), `TokenAttr` (**`k`-prefix**, `tokenizer.h:25`), `UnOp` e
`json::Type` (**PascalCase**, `nn.cuh:88`, `server_json.h:39`), mais o enum
anônimo de `unicode.h:14` (SCREAMING, herdado). E constantes em três estilos:
`kPascalCase` (48), macros SCREAMING (`quants.h`, `quant_tables.h`, `VDR_*`) e
`PF_DIST`. **Conserto:** escolher um por categoria e anotar a regra no topo de
`include/rdna4/` (ex.: "constantes `kPascalCase`; macros herdadas do ggml ficam
SCREAMING; enumeradores PascalCase em `enum class`").

### E11 — COSMÉTICO — headers, alvos e nomes em `tests/` e `scripts/`

- **`include/rdna4/tokenizer.h` é o único dos 28 headers sem `#pragma once`** (e
  sem include guard). É incluído por 7 TUs (`main.hip:30`, `serve.hip:55`,
  `tokenizer.cpp:2`, `check_batch_gpu.hip:28`, `check_eog.cpp:18`,
  `check_mtp_gpu.hip:46`, `check_tokenizer.cpp:17`); nenhuma inclui duas vezes
  hoje, então é latente. Fora ele, 27/28 têm `#pragma once`, com **colocação
  mista**: 13 na linha 1, 14 depois do comentário de cabeçalho (o mais tardio é
  `server.h:23`). **Conserto:** `#pragma once` na linha 1 em todos.
- **Sem CTest:** não existe `enable_testing()`/`add_test()` em lugar nenhum; os
  25 binários de teste são build-only e cada um tem de ser chamado à mão (o
  `README.md:104-119` lista os comandos). Os 27 alvos do `CMakeLists.txt`
  conferem com os arquivos (0 referência quebrada, 0 fonte órfã) — o que muda de
  caso é que **target é kebab-case e arquivo é snake_case** (`check-dequant-gpu`
  ← `check_dequant_gpu.hip`): sistemático, mas anotar evita "consertar".
- **Dois arquivos sem alvo:** `tests/check_server.py` (601 linhas, o teste de
  aceitação real do servidor) só é alcançável por `scripts/check_server.sh:37`, e
  `tests/gen_fake_qwen35.py` só é citado em prosa no `PLAN.md`.
- **`tests/`:** 19 `check_*`, 2 `bench_*`, 5 `oracle_*` e 1 fora do padrão —
  `dequant_cpu_oracle.cpp`, que é um oráculo **sem** o prefixo `oracle_`.
  `check_server.py` e `check_server_http.cpp` compartilham assunto e não o nome.
- **`scripts/`:** 11 arquivos; `gpu-lock.sh` e `rocm-env.sh` são os únicos com
  hífen; `rocm-env.sh` não tem shebang (é para `source`) e não é executável (ok);
  os 3 `.py` estão em modo 644 **apesar** do `#!/usr/bin/env python3`; e o
  `scripts/README.md` documenta 2 dos 11.

### E12 — COSMÉTICO — referências cruzadas quebradas (ou não verificáveis)

- `include/rdna4/dequant_row.cuh:60` cita `dequant.cuh:286` para
  `x + ibs*(QK_K/QK4_NL)`: a linha 286 é a **assinatura** de `dequantize_iq4_nl`;
  a expressão está em `:288`. A segunda citação do comentário (`yy + i*256`) não
  existe na árvore. **Conserto:** citar `:288` e escrever a expressão real
  (`y = yy + 32*ib + 4*il`).
- `README.md:108` documenta `./build/check-graph-gpu <model> <dump> -` **sem**
  `GRAPH_LAST_TOKEN=1`, que é obrigatório com dump `-ub 1`
  (`tests/check_graph_gpu.hip:124,613`; a armadilha já está registrada em
  `docs/agentes-paralelos.md:65-68`). **Conserto:** documentar a variável.
- `docs/rocm-estudo.md:417` cita `kAttnSplitMin = 2048` como "hoje"; o valor é
  **512** (`graph.cuh:308`, mudado no M8) — e os itens 1-2 de §C/§E do mesmo doc
  ("aplicar a política de splits", "tirar as 192 quantizações redundantes")
  aparecem como candidatos com dono, mas **já foram aplicados**. A referência
  `attrs.cuh:82`/`rocm-estudo.md:350` está **correta** (é o único
  doc→código linha-por-linha que resolve).
- `scripts/compare_llama_greedy.sh:9` e `PLAN.md:316` afirmam "3000 strings
  aleatórias" de fuzz do tokenizer; não existe fuzz nem artefato de 3000 strings
  na árvore (`tests/tokenizer_corpus.txt` tem 205 linhas). **Confiança:** média —
  pode ter sido um experimento não commitado; o texto é que não é verificável.
- `docs/servidor-openai.md:202` diz "68 CPU checks" e `docs/agentes-paralelos.md:55`
  diz "70 checagens"; `tests/check_server_http.cpp` tem 72 sítios `CHECK*`
  estáticos. Os dois números estão velhos em relação ao código.

---

## 4. Higiene de build

### B1 — IMPORTANTE — `-Wall -Wextra` não está ligado em nenhum alvo

`CMakeLists.txt:27-33` (e todos os 24 alvos seguintes) não passam flags de aviso;
o build usa só `-O3 -DNDEBUG` (`:18`). Consequência: os avisos que o compilador
dá por omissão (todos os `nodiscard` dos HIP) aparecem, mas tudo o mais fica
invisível.

**Build completo com `-Wall -Wextra` (feito num diretório de build externo,
`/tmp/b-warn`, sem tocar na árvore): exit 0, mas com 1 304 avisos:**

| flag | nº | onde |
|---|---|---|
| `-Wsign-compare` | **784** | **uma única linha**: `include/rdna4/attn.cuh:392` (`threadIdx.x < head_dim`, `unsigned` vs `int`), repetida por instanciação de template |
| `-Wunused-value` | 422 | todos em `tests/`+benches (os `nodiscard` de HIP do §1.1) |
| `-Wunused-parameter` | 42 | `attn.cuh:402` (16×, `n_head` em `attn_merge_kernel`), `gdn.cuh:60` (12×, `n_v_heads`), `graph.cuh:619` (12×, `t` em `full_attn`), + 2 em teste |
| `-Wunused-variable` | 20 | `graph.cuh:1056` (12×, `L`), `check_matvec_gpu.hip:416-417`, `check_matmul_gpu.hip:196`, `check_graph_gpu.hip:305` |
| `-Wunused-function` | 16 | `src/backend/model.cpp:51` (`prod`, 8×), `src/backend/unicode.cpp:24` (4×), llama.cpp (`jinja::*`) |
| `-Wunused-but-set-variable` | 4 | `check_matvec_gpu.hip:414` (`maxthr`), `check_graph_gpu.hip:638` (`exact_ok`) |
| `-Wmissing-field-initializers` | 4 | `tests/check_chat.cpp:197,202,210` (`ChatMessage::reasoning_content`) |
| `-Wformat` | 2 | `tests/check_kvctx_gpu.hip:289` (ver B3) |
| `-Wcomment` | 1 | `tests/oracle_mtp.cpp:17` — comentário `//` terminando em `\` engole a linha seguinte (aqui também é comentário, então o efeito é nulo) |

Por arquivo: `attn.cuh` 800, `check_matvec_gpu.hip` 224, `check_rope_gpu.hip` 80,
`check_matmul_gpu.hip` 40, `bench_attn_gpu.hip` 36, `graph.cuh` 24,
`check_dequant_gpu.hip` 22, `check_nn_gpu.hip` 16, `gdn.cuh` 12,
`bench_matvec_shapes_gpu.hip` 10, `model.cpp` 8, `check_kvctx_gpu.hip` 6,
`check_graph_gpu.hip` 6, `unicode.cpp` 4, `tokenizer.cpp` 4, `check_chat.cpp` 4,
`oracle_mtp.cpp` 1.

**Leitura correta desse número:** os 784 de `attn.cuh` são um **único** defeito de
tipo (`unsigned` de `threadIdx.x` contra `int` de `head_dim`) que merece correção
justamente porque é a mesma linha do achado M2; os 20 `-Wunused-variable` e 16
`-Wunused-function` são código morto confirmado (§E8); os 42
`-Wunused-parameter` são parâmetros que sobraram de refatorações (`full_attn`
recebe `t` e nunca usa — `gdn_layer` silencia com `(void)t;` em `:788`, o que só
existe para não avisar).

**Conserto mínimo:** acrescentar `-Wall -Wextra` (e as supressões pontuais
justificadas) aos alvos de `src/` primeiro, deixando `tests/` para depois — assim
o sinal não se perde nos 422 `nodiscard` de teste.

### B2 — IMPORTANTE — avisos reais do build padrão

```
src/backend/tokenizer.cpp:140:38: warning: sequência de escape hexa fora de alcance   (4×)
tests/check_kvctx_gpu.hip:289:27: warning: format specifies type 'size_t' but the argument has type 'int'
/home/marcelo/Projetos/llama.cpp/ggml/src/ggml-common.h:414:9: warning: 'IQ3S_N_SCALE' redefinido  (5×)
```

- O primeiro é o **bug E1** (literal de EOG corrompido) — o aviso é a única pista
  que o compilador dá de um defeito comprovado.
- O segundo é um `printf` com `%zu` para um argumento `int`:
  `tests/check_kvctx_gpu.hip:289:27` (o argumento é `ctx`, declarado
  `const int ctx` em `:68`) contra o `%zu` do formato em `:287`. Em x86-64 isso
  lê 8 bytes onde 4 foram passados — UB e, na prática, um número impresso errado
  no teste de contexto longo. **Conserto:** trocar o `%zu` por `%d` ou declarar
  o argumento como `std::size_t`.
- O terceiro **não** está no código do projeto, mas é causado por ele:
  `include/rdna4/quants.h:15` define `#define IQ3S_N_SCALE 4` e o
  `ggml-common.h:414` de llama.cpp define `#define IQ3S_N_SCALE QK_K/64` — mesmo
  valor (4), sequência de tokens diferente ⇒ redefinição em toda TU que inclua os
  dois (as 5 TUs de oráculo que linkam `libggml-base`). **Conserto:** não exportar
  macros do ggml de um header público (`#undef` no fim de `quants.h`, ou `constexpr`).

### B3 — COSMÉTICO — `nodiscard` e `const`

- `hipError_t` já é `nodiscard`: **o compilador aponta** cada chamada HIP sem
  checagem (os 422 avisos de §1.1) — a auditoria de erro é, em boa parte,
  automatizável com `-Wall` (ou com o alvo já existente, no build padrão).
- Violações de `const`/uso de `const`: `clang-tidy` não apontou
  `readability-non-const-parameter` em nenhuma TU amostrada; o uso de
  `const`/`constexpr` é consistente (só o `-Wmissing-field-initializers` de
  `check_chat.cpp` indica inicialização por agregado incompleta). Sem achados
  relevantes além desses.

### B4 — Resumo do `clang-tidy` (18 TUs, checks curados)

| check | avisos | leitura |
|---|---|---|
| `bugprone-implicit-widening-of-multiplication-result` | 186 (47 linhas) | **real**: tamanhos calculados em `int` e alargados depois (achado M10) |
| `portability-avoid-pragma-once` | 103 | **falso positivo** neste projeto (`#pragma once` é a convenção deliberada e funciona em amdclang++/clang/gcc) |
| `bugprone-macro-parentheses` | 84 (21 linhas) | real e barato: as macros `RD_SHIP`/`RD_BATCH`/`RD_TUNED`/`RD_RO`/`RD_ROWS`/`RD_UNROLL` de `matvec.cuh:95-101,226,521,560,622,704,769-805` passam `Dt`/`QK` sem parênteses |
| `performance-enum-size` | 22 | ruído (tamanho de enum não importa aqui) |
| `performance-inefficient-string-concatenation` | 20 | menor: `err = "layer " + std::to_string(il) + ": " + err;` em `graph.cuh:432,440,1059,1162,1173` — o custo é irrelevante em caminho de erro, mas a construção é O(n²) em `err` |
| `bugprone-signed-char-misuse` | 20 (5 linhas) | **falso positivo**: `(int)(std::int8_t)(x0 + 8.5f)` em `kv.h:230-231,293-294` e `vecdotq.cuh:334` é a transcrição fiel de `quantize_row_q4_0_ref` do ggml (validada byte a byte) |
| `bugprone-reserved-identifier` | 16 (4 linhas) | real, herdado: `__vcmpeq4`/`__vcmpne4`/`__vsub4`-like em `vecdotq.cuh:34,54,58,70` — identificadores com `__` são reservados |
| `bugprone-unchecked-string-to-number-conversion` | 9 | 1 em produção (`graph.cuh:315`, achado H9) e 5 em testes (`atoi`/`atoll`) |
| `bugprone-branch-clone` | 7 | real e inofensivo: ramos idênticos consecutivos em `dtype.h:72-73` (Q8_0 e IQ4_NL ambos 32) — dá para fundir em `case …: case …:` |
| `bugprone-random-generator-seed` | 3 | **falso positivo** (aponta para a declaração de `Sampler::init`, `sampler.h:43`) |
| `bugprone-throwing-static-initialization` | 3 | real, herdado (`unicode.cpp`, tabelas estáticas com `std::string`/`vector` que podem lançar antes de `main`) |
| `readability-redundant-inline-specifier` / `-casting` | 7 | cosmético em `unicode.h`/`utf8.h` vendorados |
| `bugprone-misplaced-widening-cast` | 1 | real: `src/main.hip:664` alarga depois de uma operação que já pode ter perdido precisão |

---

## 5. Falsos positivos e itens de baixa confiança

**Falsos positivos das ferramentas** (não "consertar"):

1. `portability-avoid-pragma-once` (103 avisos) — o projeto **quer** `#pragma once`
   e o toolchain suporta; a única ação devida é pôr no `tokenizer.h` (§E11).
2. `bugprone-signed-char-misuse` (20) e o `(void)amx` de `kv.h:295` — o cast
   `(int)(std::int8_t)(x + 8.5f)` é o comportamento **do ggml**, validado
   bit-exato; trocar por `MIN(15, …)` mudaria bytes.
3. `bugprone-random-generator-seed` (3) — a mensagem aponta para `Sampler::init`
   (`sampler.h:43`), que recebe a seed do usuário; não há semente default.
4. `performance-enum-size` (22) — irrelevante para o domínio.
5. `bugprone-easily-swappable-parameters` (centenas, excluído dos conjuntos
   finais) — ruído estrutural em kernels com muitos `int` e `const void *`;
   os nomes são explícitos (`d_q`, `d_k`, `d_v`) e as chamadas usam parâmetros
   nomeados por posição fixa há 8 milestones.
6. `bugprone-narrowing-conversions` (dezenas) — aritmética de índice de GPU
   (`blockIdx.x * blockDim.x + threadIdx.x` para `int`) é idiomática e o domínio
   (≤ 2^31 elementos) é respeitado pelos tamanhos do modelo.
7. `-Wunused-parameter` em `full_attn(…, int t, …)` (`graph.cuh:619`) é sintoma
   real de refatoração (§B1), mas o `(void)t;` de `gdn_layer:788` **não** é bug.

**Itens em que a leitura estática não fecha a questão** (marcados para quem for
corrigir):

- **M7** (overflow do loader): a aritmética está provada linha a linha, mas
  montar um GGUF forjado que passe por `validate_qwen35_layout` **não** foi feito
  (exigiria gerar o arquivo e rodar o loader — fora do escopo desta tarefa, que
  não executa binários). A construção parece viável porque os offsets são livres.
- **M8** (tamanho de string): com overcommit do Linux um comprimento grande mas
  não absurdo pode passar e falhar limpo no `fread`; qual dos desfechos
  (abort/OOM/erro limpo) sai depende do valor. A ausência da checagem é certa.
- **M5** (truncamento do `ctx_size`): nenhuma falha de memória foi construída —
  as posições são validadas em `graph.cuh:1031,1133`; o defeito é de propagação
  de erro/mensagem, não de corrupção. **Confiança baixa** de que exista corrupção.
- **E12** (3000 strings de fuzz): pode ter sido um experimento local não
  commitado; o que se afirma é que **não é verificável** na árvore.
- Números de `docs/` que dependem de GPU (tok/s, PPL, VRAM): **não** foram
  re-medidos (proibido nesta tarefa); as contradições apontadas são **internas
  aos documentos** e ao código, não medições novas.

**Verificado e correto** (não são achados, e foram checados por serem suspeitos):
`is_recr` não divide por zero (`full_attention_interval >= 1` validado em
`model.cpp:242`); `attn_splits_for` limita a `kAttnMaxSplits` nos dois caminhos
(env e automático, `graph.cuh:317-322`), então `d_attn_partial_` **não** estoura;
as posições são validadas antes de escrever o KV (`graph.cuh:1031`, `:1133`);
`release()` só é chamado pelo destrutor hoje (M1 é latente); `dequant_row_launch`
checa todos os ramos pelo `return` final; `Graph::proj` valida dims **e** bytes
antes de ler os pesos (`graph.cuh:562-568`); `forward_batch` valida
`n`/`batch_supported`/token ids/posições antes de qualquer lançamento;
`hipEventElapsedTime`/`hipEventCreate` aparecem em produção apenas em `bench`
(`main.hip:300-460`), não no caminho de decode; `kAttnSplitWpbLimit = 4` e
`attn_split_wpb` (`attn.cuh:535-538`) são consistentes com a medição citada.

---

## 6. Reproduzir as varreduras

Nada aqui executa a GPU. O diretório de build usado fica **fora** da árvore
(`/tmp/b-default`, `/tmp/b-warn`), para não sujar o `git status`.

```bash
# build padrão (avisos por omissão)
source scripts/rocm-env.sh
cmake -S . -B /tmp/b-default -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/b-default -j8 2>&1 | tee /tmp/b-default-build.log
grep -E "warning:" /tmp/b-default-build.log | sed 's/.*warning: //' | sort | uniq -c | sort -rn

# build -Wall -Wextra (o que o projeto não liga)
cmake -S . -B /tmp/b-warn -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-Wall -Wextra" -DCMAKE_HIP_FLAGS="-Wall -Wextra"
cmake --build /tmp/b-warn -j8 2>&1 | tee /tmp/b-warn-build.log
grep -E "warning:" /tmp/b-warn-build.log | sed 's/.*\[-W/[-W/' | grep -o '\[-W.*\]' | sort | uniq -c | sort -rn

# clang-tidy (existe; cppcheck NÃO está instalado)
command -v clang-tidy cppcheck
clang-tidy -p /tmp/b-warn src/backend/model.cpp --checks='-*,bugprone-*,performance-*'
clang-tidy -p /tmp/b-warn src/main.hip --checks='-*,bugprone-*,-bugprone-easily-swappable-parameters,-bugprone-narrowing-conversions'
```

Contagens pontuais citadas neste relatório (todas por `grep`/script no worktree):
335 sítios HIP com status, 240 sem teste, 20 `(void)`, 41 expressões de
lançamento em produção (35 linhas), 32 launchers, 19/19 `hipMalloc` checados,
47 linhas de alargamento implícito, 137 `int64_t` crus contra 280
`std::int64_t`, 27/28 headers com `#pragma once`, 19 alvos `check-*`,
1 304 avisos com `-Wall -Wextra`.
