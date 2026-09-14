# Revisão adversarial da árvore mergeada — `noite-baseline-2026-09-14`

Frente 1 da rodada noturna (task 1). Worktree `/home/marcelo/Projetos/rdna4-wt-noite-review`,
branch `feat/noite-review`, base `main` = tag `noite-baseline-2026-09-14` = commit `e48f3c4`.
Contrato: `docs/noite-regras.md`. Diário: `docs/journal-review.md`. Backlog: `docs/backlog-noite.md`.

**O que esta revisão é.** Uma leitura adversarial do código (não do texto que o descreve) das seis
frentes mergeadas — auditoria de qualidade, as duas de medição, autotuning, Vulkan e build repro —
com o objetivo de achar o que está **errado, incompleto ou silenciosamente quebrado**, e de
**falsificar** as afirmações que sustentam decisões. Cada achado tem severidade, local, cenário de
falha, como foi verificado e um conserto **descrito, não aplicado** (esta frente é read-only em
relação ao código do motor: `include/`, `src/`, `tests/`, `scripts/` e `CMakeLists.txt` não foram
tocados; só os três `.md` desta frente em `docs/`).

**O que foi executado de fato** (para separar medida de argumento):

| verificação | custo | resultado |
|---|---|---|
| build completo da árvore (`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` + `--build -j3`) | CPU, ~6 min | exit 0; **428** avisos, **todos** em `tests/*.hip`, **zero** em `src/` e `include/` |
| `CHECK_ALL_CPU_ONLY=1 scripts/check_all.sh` | CPU, ~5 s | 10 gates, PASS |
| o mesmo com `build/check-tuning` renomeado | CPU, ~5 s | **9 gates, PASS** (achado F3) |
| `git archive aa15eea include src` + `g++` do probe antigo + `gen_forged_gguf.py` | CPU, ~10 s | a tabela antes/depois da §7.1 da auditoria **reproduzida linha a linha** |
| `timeout 900 ./scripts/gpu-lock.sh bash /tmp/attn-wpb-ppl.sh` (PPL 8K, `RD_ATTN_SPLITS` 15/16/default) | GPU, 1 lock | fila da GPU (11-12 esperando); **abandonado**, ver "o que não consegui verificar" |
| leitura integral de `attn.cuh`, `kv.h`, `phase_prof.cuh`, `device.h`, `loader.cpp`, `gguf.cpp`, `check_tuning.cpp`, `check_regression.sh`, `check_hardening.sh`, `gen_forged_gguf.py`, `check_forge.cpp` | — | base dos achados |

---

## 0. Sumário

| # | sev. | achado | local |
|---|---|---|---|
| **F1** | **ALTO** | O `WPB` "largo" **não** é lido de `tuning.h`: `attn_split_wpb()` devolve o literal `16` e o despacho compara com outro literal `16` — a tabela só fornece o *limiar*. Trocar `kAttnSplitWpbWide`/`kAttnWarpsPerBlock` + regravar o golden deixa `check-tuning` **verde** com o kernel intacto | `include/rdna4/attn.cuh:560-566,573-582` |
| **F2** | **ALTO** | O parâmetro de template `WPB` do kernel **sem** split continua ignorado (achado E2 da auditoria, **não** corrigido): as warps 8-15 refazem as fatias de chave das warps 0-7 e o merge ignora as fatias ≥ 8. A varredura "unsplit com warps forçadas" do bench mede trabalho duplicado, e o comentário que a justifica descreve código que não existe | `include/rdna4/attn.cuh:121,165,171,178`; comentário `:88-93`; `tests/bench_attn_gpu.hip:400,409,504` |
| **F3** | **ALTO** | `check_all.sh` **pula um gate em silêncio** quando o binário não existe: `[ -x … ] && step …`. Provado: sem `build/check-tuning` a bateria imprime 9 gates e `PASS`. O mesmo construto protege a suíte de regressão (o gate numérico mais forte) | `scripts/check_all.sh:50` e `:88` |
| **F4** | **ALTO** | O `bench` usa `loader.total_bytes()` (o **arquivo**, 12,030 GB) como tráfego **por token** (11,122 GB): todo "GB/s de pesos" publicado está **8,2% otimista** — inclusive `352 GB/s = 56% da roofline` do README | `src/main.hip:373,451,480`; `README.md:199,205-206` |
| **F5** | MÉDIO-ALTO | `kAttnSplitWpbWide` preserva os números **por construção**, mas a combinação embarcada ≥16 splits × 16 warps não tem gate em texto real: o único que a toca usa cache sintético e tolerância 5e-2; o caso de 32K do golden foi gravado **com** a regra nova. E a regra é chaveada só no nº de splits, embora a medida que a motivou seja de `q4_0` | `include/rdna4/attn.cuh:546-566`; `tests/check_kvctx_gpu.hip:219-294`; `docs/autotuning-gfx1201.md:164-196` |
| **F6** | MÉDIO-ALTO | `kv_row_bytes` devolve **0** para `KvType` novo e `kv_store_row_kernel`/`kv_fill_kernel` tratam qualquer tipo não-F32/F16/Q8_0 como **q4_0** (fallthrough). A frente KV desta noite vai acrescentar `q5_0`/`q4_1`: sem exceção, sem mensagem, todas as linhas no mesmo endereço | `include/rdna4/kv.h:66-75,199-235,261-280`; consumidores `graph.cuh:728-735`, `mtp.cuh:367` |
| **F7** | MÉDIO | `kv_fill_kernel` no ramo `Q4_0`: **16 lanes escrevem o `b->d` do mesmo bloco** com valores diferentes, e cada lane quantiza seu par com o próprio `id` — bloco internamente inconsistente e vencedor da corrida indefinido, num hook que promete padrão *determinístico* | `include/rdna4/kv.h:280-298` |
| **F8** | MÉDIO | `required_bytes()` (duas sobrecargas) tem **zero chamadores** e a de um tipo só ainda contém a fórmula **pré-M4** — o bug que o M4 fechou segue no arquivo, com o nome mais natural de chamar | `include/rdna4/device.h:76-87` |
| **F9** | MÉDIO (latente) | M10 continua aberto (47 sítios de alargamento implícito em `int`): inalcançável com o modelo real (o `validate_qwen35_layout` amarra as dims aos tensores), mas é exatamente a classe que a frente KV vai tocar | `include/rdna4/graph.cuh:505-509,522,795,907-916`; `mtp.cuh:340-345` |
| **F10** | MÉDIO | `check-tuning` protege a **tabela**, não a **fiação**: nenhuma linha imprime o valor *efetivo* de um consumidor (é por isso que F1 passa por ele); e o comentário conta 22 linhas onde há 21, com `kMinLines` igual ao total | `tests/check_tuning.cpp:33-34,60-82,157-186` |
| **F11** | MÉDIO | 12 números do README checados: **7 ok, 5 com problema** — dois valores para o decode a 64K `q8_0` no mesmo arquivo (19,3 × 18,28, e 18,36 no doc), a tabela de fases que soma **37,87** contra o total declarado de **36,0**, "402-421 GB/s medidos" que é estimativa, avisos do `-Wall` descritos como pendentes, e uma linha cuja soma não fecha | `README.md:169,184,202,240-248,298-299,353-355` |
| **F12** | BAIXO-MÉDIO | A prova de forja do loader **é sólida** (reproduzi as cinco linhas por `git archive`), mas (a) o `check_all.sh` chama o script **sem** o argumento, então a metade "antes" nunca roda na bateria; (b) o argumento é um *worktree* pré-correção que já não existe; (c) três dos cinco casos aceitam uma mensagem genérica, então uma rejeição por motivo errado passaria | `scripts/check_hardening.sh:22-23,56-60,62-89`; `scripts/check_all.sh:49` |
| **F13** | BAIXO | Comentários/código morto que descrevem código que não existe: `kbase` mutiplicado por zero, "17 full-attention layers" no mesmo comentário que diz 16, `attn.cuh` prometendo que o `WPB` muda a ordem de soma, `rocm-estudo` §F.1 prometendo que ≥8 splits rodam "exatamente o mesmo caminho de antes" | `attn.cuh:110-111`; `device.h:33-40`; `rocm-estudo.md:486-492` |
| **F14** | BAIXO | Nuances do profiler de fases (que **é** inerte): macro avalia o argumento duas vezes; eventos nunca destruídos; falha de `hipEventCreate` desliga o profiler no meio e a tabela continua a ser impressa sem aviso | `include/rdna4/phase_prof.cuh:82-121,161-164` |
| **F15** | MÉDIO | O `MtpHead` aloca **duas** caches KV próprias (`max_ctx × NKV × row`) fora do orçamento de `info`/`serve` (+0,25 GiB a 64K, +0,5 GiB a 128K), e `docs/mtp.md:269` descreve **uma** — então `--mtp` pode estourar uma configuração que o `info` aprovou | `include/rdna4/mtp.cuh:365-372`; `src/main.hip:918-923`; `docs/mtp.md:269` |

Nenhum achado novo é CRÍTICO no sentido da auditoria (nada aqui é alcançável por um arquivo ou
pedido HTTP trivial e produz saída errada com o modelo real). O que a revisão encontrou de mais
grave é **de gate**: um gate que some sem falhar (F3) e um gate que fica verde enquanto a constante
que ele protege deixa de ser usada (F1/F10) — a mesma classe de falha que `docs/gpu-queue.md:41-43`
já tinha identificado ("um gate vermelho por contenção é pior que um gate não rodado").

---

## F1 — ALTO — o `WPB` largo é um literal, não a constante da tabela (e o gate não vê)

**Local**: `include/rdna4/attn.cuh:560-566` (`attn_split_wpb`) e `:573-582` (despacho);
afirmação contrariada: `include/rdna4/tuning.h:5-12` ("este e o UNICO lugar onde os parametros
medidos deste motor vivem … nao existe uma segunda copia dos numeros em lugar nenhum").

**Código**:

```cpp
inline int attn_split_wpb(int n_splits) {
  if (n_splits <= kAttnSplitWpbLimit) return 16;   // literal
  if (n_splits >= kAttnSplitWpbWide) return 16;    // limiar da tabela, valor literal
  return kAttnWarpsPerBlock;
}
...
if (attn_split_wpb(n_splits) == 16)              // terceiro literal
  return attn_launch_split_typed<…, 16>(…);
return attn_launch_split_typed<…, 8>(…);         // quarto literal
```

**Cenário de falha**: a próxima passada de autotuning decide que a CTA larga deve ter 32 warps (a
instanciação **existe**: `attn_launch_split_wpb` tem `case 32` em `attn.cuh:471-473`, e o
`bench-attn-gpu` varre 8/16/32). A mudança correta seria `kAttnSplitWpbWide`/um novo
`kAttnSplitWpbNarrow = 32` em `tuning.h` + regravar o golden. O que acontece de fato: o *limiar*
muda, o **valor devolvido continua 16**, o despacho `== 16` casa e o kernel roda com 16 warps. O
`check-tuning` imprime `attn.split_wpb_wide = 32` igual ao golden novo (verde), o doc passa a dizer
32 warps e **nenhum número de desempenho muda** — uma alegação de ganho sem ganho, exatamente o
"mudar em silêncio" que o gate existe para impedir. O simétrico também vale: com
`kAttnWarpsPerBlock = 4` a ponta estreita devolveria 4 e o despacho cairia no `else` de 8.

**Como verifiquei**: `grep -rn "tuned::" include/ src/ tests/` (49 ocorrências) + leitura de
`attn.cuh:554-601`. Não medi nada em GPU: o achado é de *fiação*, e a prova é a comparação entre o
valor devolvido (literal) e a constante da tabela. Um `grep -n "16"` em `attn.cuh` na região mostra
o literal quatro vezes.

**Conserto (descrito, não aplicado)**: nomear as duas pontas e derivar do template —
`inline constexpr int kAttnSplitWpbNarrow = tuned::kAttnWarpsPerBlock;` e devolver
`tuned::kAttnSplitWpbWide` / `kAttnSplitWpbNarrow`; trocar o `== 16` por um `switch` sobre o valor
devolvido com `static_assert` de que as duas constantes estão no conjunto instanciado ({8,16,32}).
E — o que fecha a classe — imprimir no `check-tuning` o valor **efetivo**
(`attn.split_wpb(2)`, `attn.split_wpb(16)`, `attn_splits_for(4096)`), de modo que um consumidor que
deixe de ler a tabela quebre o gate (ver F10).

---

## F2 — ALTO — o `WPB` do kernel **sem** split continua ignorado (E2 aberto)

**Local**: `include/rdna4/attn.cuh:94` (template), `:121` (passo do laço de chaves),
`:165,:171,:178` (merge) — todos usam `kAttnWarpsPerBlock`, a constante global, não o `WPB` do
template; só o lançamento (`:191-193`) usa o template. Comentário que afirma o contrário:
`:88-93` ("Changing it changes the number of partial slices merged at the end of the CTA, i.e. the
summation order across key slices"). A auditoria já tinha este achado (**E2**,
`docs/auditoria-qualidade.md:601-629`) e a §7.3 dele (linha 1116-1119) o tinha adiado "para depois
do merge da frente de autotuning" — o commit `e48f3c4` corrigiu **a cauda do kernel com split**
(M2, `attn.cuh:395-407`) e **não** este.

**Cenário de falha (medida inválida, não corrupção)**: com `WPB = 16` no template, as warps 8-15
executam `for (j = w; j <= t; j += kAttnWarpsPerBlock)` — ou seja, a warp 8 percorre
`j = 8, 16, 24, …`, que é **a mesma fatia da warp 0**, e as fatias 8-15 do smem nunca são lidas (o
merge vai até `kAttnWarpsPerBlock`). Resultado: saída **idêntica** ao `WPB = 8` e o dobro de
trabalho de caminhada de chaves. Portanto a linha "unsplit kernel with forced warps/CTA" de
`tests/bench_attn_gpu.hip:400,409,504` — e a conclusão de `docs/rocm-estudo.md:359-363`
("`WPB=32` é sempre pior que 8 ou 16") — comparam células que **não** são "mais warps": são "as
mesmas 8 warps mais 8 warps refazendo o serviço". Uma decisão de política tomada a partir dessa
varredura estaria tomada sobre o experimento errado.

**Como verifiquei**: leitura de `attn.cuh:94-217` + `grep -n "WPB\|kAttnWarpsPerBlock" attn.cuh`
(mostra que as 4 linhas do corpo usam a constante e só o launcher usa o template) + leitura do
bench (`tests/bench_attn_gpu.hip:390-412,504-506`). **Não** re-executei a varredura (a placa é de
outra frente nesta janela): o efeito está argumentado, não medido, e o diário diz isso.

**Conserto (descrito)**: usar `WPB` no corpo (`j += WPB` no laço e limite `WPB` no merge) e
re-rotular/re-medir a varredura sem split — **ou**, se a intenção era que o kernel sem split
tivesse forma fixa, apagar `attn_launch_wpb()`/`attn_launch_wpb_typed()` e a linha do bench, e
corrigir o comentário. O kernel **com** split honra o `WPB` (`:318-319` e `:363-407`), então o
caminho embarcado não é afetado por F2.

---

## F3 — ALTO — `check_all.sh` pode imprimir PASS com um gate a menos

**Local**: `scripts/check_all.sh:50` (`[ -x "$B/check-tuning" ] && step "check-tuning" "$B/check-tuning"`)
e `:88` (`if [ -x "$B/check-regression-gpu" ]; then step "regression suite" … fi`). O script não usa
`set -e`, e o resultado da linha 50 não entra em `fail`.

**Cenário de falha**: qualquer build parcial, alvo renomeado, `--target` esquecido ou binário
apagado transforma "10 gates de CPU + 15 de GPU, todos verdes" (`docs/gpu-queue.md:62-67`,
`README.md:308`) em 9 + 14 verdes **com a mesma linha de PASS**. O pior caso é o segundo: a suíte de
regressão numérica (79 s, "o gate numérico mais forte", `docs/noite-regras.md:53`) desaparece do
relatório sem uma palavra — e o relatório da manhã vai afirmar que ela passou.

**Como verifiquei (medido)**: `mv build/check-tuning /tmp/check-tuning.bak && CHECK_ALL_CPU_ONLY=1
./scripts/check_all.sh` → imprime **9** gates (`check-dtype … check-hardening`) e
`check_all (CPU half): PASS`, exit 0; nenhuma linha `SKIP`, nenhuma menção ao gate ausente.
Depois restaurei o binário (o `build/` é gitignored e é meu).

**Conserto (descrito)**: tratar ausência como falha explícita —
`if [ -x "$B/check-tuning" ]; then step …; else echo "  FAIL  check-tuning not built"; fail=1; fi` —
ou, no mínimo, imprimir `SKIP (não construído)` e contá-lo num total de gates verificados que
apareça na linha final. O mesmo para `check-regression-gpu`.

---

## F4 — ALTO — o `bench` conta o arquivo inteiro como tráfego por token (8,2% otimista)

**Local**: `src/main.hip:373` `const std::size_t bytes_per_token = loader.total_bytes();`, usado em
`:451` (`gb_s = bytes_per_token / sec_per_token`) e impresso em `:480`.
`GgufLoader::total_bytes()` soma **todos** os tensores do arquivo (`src/backend/loader.cpp:137`).

**Cenário de falha (número publicado errado)**: para o IQ3_S, o arquivo tem **12,030 GB** e o
tráfego real de um token de decode é **11,122 GB** — a diferença é a matriz `token_embd.weight`
(da qual só uma linha é lida) e o bloco MTP `blk.64.*` (que o `bench` não executa). Isso não é uma
opinião: é a tabela do próprio `README.md:126-127` e o inventário de
`docs/quants-inventario.md` §3.3. Fator 12,030/11,122 = **1,082**. Logo a linha
`README.md:199` ("34.2 ms/token, **352 GB/s** of weights") é `12,030 GB / 0,0342 s = 351,8`, e o
valor honesto é `11,122 / 0,0342 = 325 GB/s` = **51%** da roofline de 633 GB/s, não 56%. O mesmo
viés contamina "weight bandwidth, end to end 352 GB/s" (`:205`) e toda comparação de banda derivada
de `bench` (M5/M6/M7, `docs/vulkan-vs-hip.md:393-401`).

**Como verifiquei**: leitura de `src/main.hip:370-380,445-485` +
`grep -rn "total_bytes" include src` (só `main.hip:373` e `:886`, este último correto, é o tamanho
do modelo) + aritmética contra a tabela do README. Não re-executei o `bench` (fila da GPU).

**Conserto (descrito)**: expor no loader um `per_token_bytes()` (Σ `bytes_[i]` exceto a matriz de
`token_embd.weight`, mantendo uma linha, e exceto `blk.64.*` quando o MTP não roda) e usar isso em
`bench`; `total_bytes()` fica só para a linha "model size" (`main.hip:886`). É a mesma conta que
`docs/medicoes-banda-e-gargalos.md:507-508` já recomenda e que `docs/rocm-estudo.md` §C.9 pede —
os docs tratam como feito, o código não faz.

---

## F5 — MÉDIO-ALTO — `kAttnSplitWpbWide`: garantia por construção, evidência em falta

**Local**: `include/rdna4/attn.cuh:512-566`; gates: `scripts/check_attn_split.sh`,
`tests/check_kvctx_gpu.hip:219-294`; medida: `docs/autotuning-gfx1201.md:164-196`.

**A parte que se sustenta (por construção)**: para qualquer `WPB`, o conjunto de chaves de um CTA
(`h,s`) é `{w + WPB·s : w ∈ [0,WPB)}` com passo `WPB·S`; a união sobre `s ∈ [0,S)` é exatamente
`[0, WPB·S)`, então a cobertura é completa e sem contagem dupla para **qualquer** `WPB ∈ {8,16,32}`.
O que muda é só a partição das chaves entre as warps e, com ela, a ordem de soma dentro de cada par
(cabeça, split) — a mesma classe do split-KV do M7 (`rel-L2 ~2,5e-7` no kernel, declarado em
`attn.cuh:536-538`). Não há caminho para "resultado errado" fora dessa classe, e o caso de split
vazio está tratado (`:373-381`: `cmax == -INFINITY` → pesos 0).

**A parte que não se sustenta (evidência)**:
1. O único gate que exercita a **ponta larga embarcada** é `check-kvctx-gpu 65536 q4_0`, que força
   16 splits a 64K contra cache **sintético aleatório** (`debug_fill_caches`) com tolerância
   **5e-2** — declarada frouxa no próprio arquivo (`check_kvctx_gpu.hip:279-286`: "a diferença
   relativa … é amplificada pelo cancelamento, medido rel-L2 ~1,5e-2"). Ele pega NaN/lixo, não
   reordenação.
2. `scripts/check_attn_split.sh` (PPL em texto real, limite 0,5%) roda `splits = 4` a `ctx = 1024`
   → cai na ponta **estreita** da regra (`n_splits ≤ kAttnSplitWpbLimit`), não na larga.
3. O caso longo do golden de regressão (`32K`, `tests/golden/regression_greedy_f16.txt`) foi
   **gravado no commit `9ef1526`**, o mesmo que introduziu a regra de duas pontas
   (`git log -- tests/golden/regression_greedy_f16.txt` → só `9ef1526`), então ele não pode
   distinguir `WPB` 8 de 16: a referência já é a regra nova.
4. A regra é chaveada **só no número de splits**, mas a medida que a motivou é explícita sobre o
   tipo de KV ("sobretudo com KV `q4_0`, cuja linha é 3,5× menor", `attn.cuh:548-550`); a varredura
   f16 do próprio arquivo mostra `WPB=16` **3% pior** a 16 splits (0,0844 contra 0,0819,
   `attn.cuh:518-522`), e o A/B intercalado a 64K/16×16 deu **1,004x em 6/15 rodadas** — isto é,
   "nada acima do ruído" (`docs/autotuning-gfx1201.md:172`). A decisão de embarcar é defensável
   (troca ≤1% em um contexto por +5% a 131K `q4_0`, `:176-180`), mas o lado f16 é uma
   **não-regressão dentro do piso**, não um ganho.

**Consequência prática**: em ctx 8192..65536 com KV `f16`/`q8_0` o motor roda hoje uma combinação
(16 splits × 16 warps) que **nenhum** gate de texto real validou com o binário embarcado. Se houver
um bug de forma nessa combinação, o sintoma será um PPL pior em alguns contextos longos, sem nenhum
gate vermelho.

**Como verifiquei**: leitura do kernel (cobertura exata, aritmética acima) + leitura dos dois gates
+ `git log` do golden. Desenhei e tentei a medida que fecha a lacuna (PPL a 8192 chaves com
`RD_ATTN_SPLITS` 15 vs 16 vs default, ordem rotacionada) — **abandonada na fila da GPU** (11-12
processos esperando o lock, um `rdna4-infer` de outra frente segurando 12,02 GiB); ver o diário.

**Conserto (descrito)**: (a) acrescentar ao `scripts/check_attn_split.sh` um segundo par em
`ctx = 8192` ou `16384` com `SPLITS` = 16 (ou, melhor, comparar `RD_ATTN_SPLITS=15` × `16`, que isola
o `WPB` mantendo a contagem de splits — foi esse o desenho que eu tentei); (b) condicionar a ponta
larga ao tipo de KV (a medida é de `q4_0`) ou registrar na tabela de tuning que o lado f16 é
não-regressão dentro do ruído; (c) documentar em `attn.cuh` qual gate cobre qual faixa de contexto.

---

## F6 — MÉDIO-ALTO — `kv_row_bytes` devolve 0 e o armazenamento cai em `q4_0` (armadilha para a frente KV)

**Local**: `include/rdna4/kv.h:66-75` (`kv_row_bytes`, `return 0` no fim, sem `default` que grite),
`:55-63` (`kv_bytes_per_elem` → 0,0), `:199-235` (`kv_store_row_kernel`: os ramos F32/F16 saem
antes, `Q8_0` tem `if`, e o corpo de **q4_0** é o *fallthrough*), `:261-280` (`kv_fill_kernel`,
mesma forma); consumidores: `graph.cuh:728-735` e `mtp.cuh:367`.

**Cenário de falha**: `docs/noite-regras.md:16-23` declara que o alvo da noite é K em `q5_0` e V em
`q4_1` — nenhum dos dois existe hoje. Ao acrescentar um valor ao `enum class KvType`, o compilador
**não** reclama de nada: `kv_load<CT>`/`kv_load8<CT>` são templates declarados sem definição
primária (então *lá* é erro de compilação — a única rede de segurança), mas
`kv_store_row_kernel<CT>` **compila e armazena como q4_0**, `kv_fill_kernel<CT>` idem, e
`kv_row_bytes` devolve **0**. Com 0 byte de passo de linha, todas as linhas de todos os tokens caem
no **mesmo endereço** (sobrescrevendo-se) e a atenção lê o que sobrou: saída errada, sem exceção,
sem mensagem, sem gate vermelho. Pior: `kv_bytes_per_elem` = 0,0 faz `kv_cache_bytes` devolver 0 e
o orçamento de `info`/`serve` **aprovar** a configuração ("need N GiB" sem o KV) — o `hipMalloc`
estoura no load, ou não estoura e o motor roda com o cache aliased.

**Como verifiquei**: leitura integral de `kv.h` + `grep -rn "kv_row_bytes\|kv_bytes_per_elem"
include src` para listar os consumidores. É leitura de código, sem execução: o cenário exige um
`KvType` novo, que ainda não existe.

**Conserto (descrito, para a frente que tem o arquivo)**: fechar as três portas —
(i) `kv_row_bytes`/`kv_bytes_per_elem` com `default: assert(false); __builtin_unreachable();`
(ou `switch` sem `default` + `-Wswitch`, que já é aviso por omissão no clang);
(ii) `kv_store_row_kernel`: transformar o corpo final em `if (CT == KvType::Q4_0) { … } return;` de
modo que um tipo novo caia no vazio (erro de compilação por `-Wreturn-type`/uso não escrito) em vez
de virar q4_0;
(iii) `kv_fill_launch`: hoje o `switch` sobre `t` **não tem** `default` e cai no `return hipGetLastError()`
— devolver `false` explicitamente num tipo não tratado;
(iv) um `static_assert`/teste CPU que force a lista de `KvType` a casar com a lista de
`kv_type_parse`/`kv_type_name` (o teste `check-rope-gpu` já compara as linhas armazenadas com o
espelho host por tipo — estender a lista lá é o gate natural).

---

## F7 — MÉDIO — `kv_fill_kernel` no ramo `Q4_0`: corrida de escrita no `b->d` e bloco inconsistente

**Local**: `include/rdna4/kv.h:280-298`.

**Código (resumido)**: cada lane `< 16` calcula o **seu** par `a0, a1`, deriva
`signed_max` e `d = signed_max / -8`, escreve **o mesmo** `b->d` (`:290-292`) e quantiza
`b->qs[lane]` com o **seu** `id` (`:293-298`). Ou seja: 16 lanes escrevem 16 valores diferentes no
mesmo endereço (o vencedor é indefinido) e os bytes armazenados foram quantizados com escalas
diferentes da que ficou gravada.

**Cenário de falha**: o comentário (`kv.h:238-241`) promete "padrão pseudo-aleatório
**determinístico** … para um teste de contexto longo ter uma cache realista e não degenerada".
O que sai é um bloco `q4_0` internamente inconsistente e **não determinístico** entre arquiteturas
(qual lane ganha a corrida de escrita). Hoje isso é inofensivo *por acidente*: os dois consumidores
(`check-kvctx-gpu` e `bench --fill-cache`) comparam dois caminhos sobre a **mesma** cache, então o
erro cancela. No dia em que alguém escrever um gate que valide *valores* a contexto longo, ou que
compare a cache sintética com uma real, a falha aparece sem causa óbvia. O `Q8_0` logo acima faz
certo (redução de warp para o `amax`, um `d` por bloco, `:269-278`) — a assimetria é o próprio bug.
Item menor na mesma função: `amx` é calculado e jogado fora (`:288` e `:295 (void)amx;`).

**Como verifiquei**: leitura de `kv.h:250-299`; a corrida é evidente pela ausência de qualquer
redução/eleição antes das escritas, e o `Q8_0` acima mostra a forma correta. Não executei nada (é
um caminho de preenchimento de teste; o efeito visível exigiria ler os bytes de volta).

**Conserto (descrito)**: replicar a estrutura do `Q8_0` — reduzir o *máximo assinado* do bloco por
warp (`__shfl_xor_sync` sobre 32 lanes, como em `:271`), deixar **uma** lane publicar `b->d`, e
depois cada lane quantizar o seu par com `id = 1/d` do bloco; apagar `amx`.

---

## F8 — MÉDIO — `required_bytes()` morto, com a fórmula pré-M4 dentro

**Local**: `include/rdna4/device.h:76-87` (as duas sobrecargas).

**Cenário de falha**: `grep -rn "required_bytes" src include tests scripts` devolve **apenas as
definições** — zero chamadores. A sobrecarga de três argumentos
(`file_bytes + ctx * kQwen35KvElemsPerToken * kv_bytes_per_elem(kv_type)`) é **exatamente** a
fórmula que o achado M4 provou errada para K/V de tipos diferentes (ela multiplica a contagem de
elementos de K **e** V pelo bpe de **um** tipo: com K `q4_0`/V `f16` subestima 1,44 GiB,
`docs/auditoria-qualidade.md:1145-1147`). O README afirma (`:348`) que
"`kv_cache_bytes(ctx, k, v)`, one source of truth" — verdade para o caminho vivo
(`src/main.hip:918`, `src/server/serve.hip:385`), **falsa** para o arquivo, onde um nome mais
natural de chamar (`required_bytes`) continua disponível com a semântica velha. É a mesma classe do
§7.5 da auditoria ("corrigir a instância que o relatório citou não é corrigir a classe").

**Como verifiquei**: grep de chamadores (0) + leitura das duas funções + conferência de quem usa
`kv_cache_bytes`.

**Conserto (descrito)**: apagar as duas sobrecargas (ou reimplementar a de um tipo como wrapper de
duas chamadas a `kv_cache_bytes`, se algum dia alguém precisar do nome).

---

## F9 — MÉDIO (latente) — M10 continua aberto, e é a classe que a frente KV vai tocar

**Local (amostra verificada por leitura)**: `include/rdna4/graph.cuh:505-509`
(`alloc(d_proj_, NH * 2 * HD, err)`, `alloc(d_attnout_, NH * HD, err)`),
`:522` (`alloc(d_kstage_, NKV * HD, err)`), `:795` (`float *v_c = d_conv_ + 2 * key_dim;`),
`:907-916` (`d_qb_ + (std::size_t)t * NH * 2 * HD`), `:418`/`:753`
(`chan = 2 * key_dim + d_inner`), `include/rdna4/mtp.cuh:340-345`,
`include/rdna4/kv.h:259,286,287`. A auditoria conta 186 avisos de
`bugprone-implicit-widening-of-multiplication-result` em 47 linhas
(`docs/auditoria-qualidade.md:519-546`) e a §7.3 adiou o conserto por conflito de arquivos.

**Cenário de falha**: o produto é calculado em `int` e só convertido para `std::size_t` no parâmetro
de `alloc()`; um `head_count`/`key_length`/`embedding_length` grande o bastante para estourar
`int` faz o buffer ficar **pequeno** enquanto os kernels indexam com os mesmos números estourados.
**Não é alcançável hoje**: `validate_qwen35_layout` (`src/backend/model.cpp:336-413`) casa cada dim
com as dims reais dos tensores, e o loader limita cada tensor ao tamanho do arquivo
(`src/backend/loader.cpp:126-132` e `:155-192`). É por isso que a auditoria o classificou IMPORTANTE e não
CRÍTICO, e eu concordo — mantenho como latente.

**Por que ainda vale consertar**: `kv_row_bytes`/`kv_bytes_per_elem` vão ganhar `q5_0`/`q4_1` com
tamanhos de bloco novos, e o padrão que a frente vai copiar (`bytes = (int)tipo_bytes * (int)elems`)
é o mesmo. Um `bytes_of(rows, cols)` checado, usado nos 47 sítios, mata a classe inteira de uma vez.

**Como verifiquei**: grep dos sítios citados + leitura de `alloc()`/`balloc()` +
`sed -n '336,413p' src/backend/model.cpp` para a alcançabilidade. Não re-rodei `clang-tidy` (a
contagem de 186/47 é da auditoria; a *classe* está confirmada por leitura em 8 sítios).

**Conserto (descrito)**: converter na **primeira** multiplicação (`(std::size_t)NH * 2 * HD`) nos
47 sítios, ou centralizar `std::size_t bytes_of(std::size_t rows, std::size_t cols)` com checagem
de overflow; fazer isso numa passada única em quem é dono desses arquivos (esta frente não é).

---

## F10 — MÉDIO — `check-tuning` protege a tabela, não a fiação (e conta errado as próprias linhas)

**Local**: `tests/check_tuning.cpp:33-34` (comentário e `kMinLines`), `:60-82`
(`report_lines()`), `:157-186` (as três checagens).

**O que o gate faz bem**: exige presença de **todas** as 22 chaves obrigatórias (`:38-46`), exige
`ref.size() == now.size()` (`:180-186`) e compara linha a linha com a referência commitada — não é
um teste vazio, e uma mudança *acidental* em `tuning.h` quebra. O `--record` é o caminho
documentado para mudanças medidas.

**O que ele não vê**: `report_lines()` imprime exclusivamente valores de `tuned::*`. Nenhuma linha
imprime o valor **efetivo** que o motor usa. É exatamente por isso que F1 (literal `16` em
`attn.cuh`) e F2 (template `WPB` ignorado) convivem com um gate verde: a constante é lida e
impressa, o consumidor é que não a usa. O gate responde "a tabela é a medida", não "o motor usa a
tabela" — e o comentário do cabeçalho (`:3-9`) promete a segunda coisa.

**Detalhes menores**: o comentário da linha 33 ("14 tipos de matvec + 4 da atenção + 3 do batch +
1 do total = 22 linhas") conta errado nos dois termos — são **21** linhas (1 + 14 + 4 + 1 + 1) e há
**2** linhas de batch (`batch.ns`, `batch.cap`, `:74-80`), não 3. E `kMinLines = 21` é igual ao
total exato, então a checagem "anti-teste-vazio (1)" é, na prática, redundante com a igualdade de
tamanho da linha 180.

**Como verifiquei**: leitura integral de `check_tuning.cpp` + execução
(`./build/check-tuning` dentro da bateria CPU: OK, 21 linhas contra `tests/golden/ml_tuning.txt`).

**Conserto (descrito)**: acrescentar ao relatório as linhas derivadas — `attn.split_wpb(2)`,
`attn.split_wpb(16)`, `attn.splits_for(4096)`, `MtShape<9>::rows/wpr`, `batch_supported(5)` — para
que a referência cubra a fiação e não só a tabela; corrigir o comentário; separar `kMinLines` do
tamanho exato (ex.: 18) para que a checagem anti-vazio continue tendo sentido.

---

## F11 — MÉDIO — números do README contra os docs (12 checagens, 5 problemas)

Método: greps dirigidos por número (`grep -rn "18\.28\|13\.86\|402-421\|1479\|575" docs/ README.md`)
+ aritmética. Os **7 que conferem**: `tg64 = 40,0 ± 0,02` e `39,7` (M5);
`pp64 = 575 ± 65` e `pp512 = 1143 ± 30`; `436 GB/s = 69%`, `620 = 98%`, `352/633 = 56%`
(aritmética); `1479 marcas × 4,03 µs = 5,95 ms` = 16% (`docs/medicoes-banda-e-gargalos.md:106-109`
com 4,026 µs); `497 matvec + 257 quantizações` (`:154`); `iq3_xxs 16,8%` =
`docs/quants-inventario.md:124` (16,75% — a tabela do autotuning diz 16,7%, diferença de
arredondamento entre docs); `13,86 GiB` do 128K `q4_0` = `quants-inventario.md:332` e
`medicoes-m5.md:149`. Os **5 problemas**:

1. **Dois números para o mesmo decode a 64K `q8_0`** (`README.md:202` = **19,3 tok/s**;
   `:184` e `:369` = **18,28**; e o doc de origem, `docs/medicoes-banda-e-gargalos.md:488`, diz
   **18,36**). A pegada de VRAM citada é a mesma (13,75 GiB), logo é a mesma configuração. Ver o
   veredito completo no diário (entrada 11): o efeito de +0,5% a +1,8% é menor que a variabilidade
   entre sessões (2-3%, `docs/rocm-estudo.md:504-507`), os dois braços foram medidos em corridas
   **separadas** (não no A/B intercalado que o próprio repo exige), e "também é *mais rápido*"
   (`:183-184`) é uma afirmação acima da evidência. A recomendação de usar `q8_0` acima de ~56K
   **sobrevive** — ela se sustenta na folga de VRAM (2,18 GiB) e na precisão dobrada, não na
   velocidade.
2. **A tabela "Where a token goes" não fecha**: as seis linhas somam
   `24,94 + 5,63 + 2,96 + 2,08 + 1,38 + 0,88 = **37,87 ms**` contra o total declarado de **36,0**.
   A causa é dupla e está na fonte (`docs/medicoes-banda-e-gargalos.md:137-145`): o LM head
   (1,43 ms) está **dentro** da linha (a) 24,94 ("tronco 23,51 + head 1,43") **e** na linha (f) 2,08
   ("head 1,43 + cópia 0,21 + sampler 0,44"); e a linha (f) soma o sampler de 0,44 ms, que é
   **host** e não faz parte do total de 36,0 (a linha de tempo do dispositivo). As **porcentagens**
   (69,3 + 15,6 + 8,2 + 5,8 + 3,8 + 2,4 = 100,1%) escondem o problema porque cada uma é calculada
   contra 36,0 — a soma das parcelas é que não é 36,0. A tabela fina do mesmo doc (`:113-129`) soma
   **35,98** ✅, então o erro é só na agregação: ou o head sai da linha (a), ou sai da (f), e o
   sampler precisa de uma linha própria fora do total do dispositivo.
3. **"this engine's measured 402-421 GB/s"** (`README.md:298-299`): 402-421 GB/s é a **estimativa
   de replay** de `docs/rocm-estudo.md:61` (§A.1) e de `docs/kv-memoria-desenho.md:382`; a medida
   é **436 GB/s no tronco** (`docs/medicoes-banda-e-gargalos.md:519`), que o próprio README usa em
   `:206`. Trocar "measured" por "estimated (replay)" ou usar 436 — e notar que o 436 também
   carrega o viés de F4.
4. **Avisos de build descritos como pendentes** (`README.md:353-355`: "the engine compiles with 0
   warnings under `-Wall -Wextra` **except** one line repeated in `attn.cuh` (98 × `-Wsign-compare`)
   and three unused parameters — those are queued with the attention changes"): isso descreve a
   árvore **antes** de `e48f3c4`. Medido agora: os alvos do motor compilam com **0** avisos
   (`CMakeLists.txt:38,40,246`), a linha de `-Wsign-compare` foi corrigida pela cauda strideada
   (`attn.cuh:402-407`) e os três parâmetros levam `[[maybe_unused]]` (`attn.cuh:412`,
   `gdn.cuh:60`, `graph.cuh:644`). Em `:72-73`, "422 of the 433" também é da árvore pré-merge: hoje
   são **428** avisos, todos de `tests/*.hip` (nenhum do motor).
5. **Somos que não fecham**: `README.md:169` mostra a linha de 128K `q4_0` com parcelas
   `11,133 + 2,416 + 0,302 = 13,851` e total **13,87** (a de 64K erra 0,01 do mesmo jeito). A
   tabela vem de `docs/kv-memoria-desenho.md:325` e o erro está **lá** — corrigir na fonte e
   re-copiar. (Nota: 13,86 GiB em `README.md:130` é outra grandeza — VRAM em uso — e está certa.)

---

## F12 — BAIXO-MÉDIO — gate de forja do loader: prova sólida, fiação fraca

**O que é sólido (e eu reproduzi)**: `tests/gen_forged_gguf.py` reescreve **um** campo de um GGUF
falso válido; para o caso do offset, `off_wrap = 2⁶⁴ − doff + 64` (o script imprime
`doff=2912 offset_wrap=18446744073709548768`), logo `doff + off_wrap = 2⁶⁴ + 64` → `add_checked`
(`src/backend/loader.cpp:24-28`) falha em **i=0**, o primeiro tensor — o único sem antecessor, e
portanto o único cujo limite não é fixado pela guarda do tensor anterior
(`docs/auditoria-qualidade.md:1076-1082`). Reconstruí a árvore pré-correção com
`git archive aa15eea include src` + `g++` e comparei os dois probes nas cinco forjas:

```
ok                 antigo: accepted (512 bytes, mesmos bytes)   novo: accepted (idêntico)
forge_offset_wrap  antigo: accepted offset=18446744073709548768  novo: rejected "offset/size sum overflows 64 bits"
                   (head 71 77 65 6e 33 35 = ASCII "qwen35")        ← a soma guardada é o que rejeita
forge_dim_huge     antigo: rejected "end not aligned" (razão errada) novo: rejected "computed size … exceeds the 29312-byte file"
forge_str_len      antigo: rc=134 std::bad_alloc                 novo: rejected "truncated or corrupt header"
forge_arr_len      antigo: rejected                              novo: rejected
```

Ou seja: a tabela §7.1 da auditoria é uma medida, não retórica, e a forja exercita **a soma
guardada** e não uma guarda vizinha. Três fragilidades de fiação, todas de gate:

1. `scripts/check_all.sh:49` chama `"$ROOT/scripts/check_hardening.sh"` **sem argumento**, então a
   metade "antes/depois" (`check_hardening.sh:62-89`) nunca roda na bateria; o que roda é só
   "o loader endurecido rejeita" — que é a metade fraca.
2. O argumento documentado é "o root de um checkout pré-correção" — e o worktree que a auditoria
   usou (`rdna4-wt-medicoes-gpu`) já está mergeado na `main`, logo **não existe mais** um caminho
   pré-correção na máquina. Sem conserto, a coluna "antes" deixa de ser reproduzível.
3. `check()` exige que a mensagem **contenha** um trecho: para `forge_dim_huge` é `"exceeds the"`, e
   para `forge_str_len`/`forge_arr_len` a **mesma** mensagem genérica
   `"truncated or corrupt header"`. Uma rejeição por qualquer outro motivo de parse passaria nos
   dois casos; só o caso do offset assere uma mensagem exclusiva da guarda corrigida.

**Conserto (descrito)**: trocar o parâmetro de *worktree* por um *commit-ish* e extrair com
`git archive <rev> include src | tar -x -C "$TMP/old"` (foi o que eu fiz aqui em ~10 s, sem
worktree, sem tocar em `git worktree`/`main`); passar esse rev no `check_all.sh` (ou imprimir
`SKIP (sem rev pré-correção)` quando ausente, para não fingir cobertura); e, onde possível, asserir
o trecho de mensagem que identifica a guarda específica em vez do genérico.

---

## F13 — BAIXO — comentários que descrevem código que não existe

| local | o que o comentário diz | o que o código faz |
|---|---|---|
| `include/rdna4/attn.cuh:88-93` | trocar o `WPB` do kernel sem split muda "the number of partial slices merged at the end of the CTA, i.e. the summation order across key slices" | o merge vai até `kAttnWarpsPerBlock` e o passo do laço é a constante: o `WPB` do template não muda nada (F2) |
| `include/rdna4/attn.cuh:110-111` | `kbase` "set per key below" | `kbase = k + (kvh * head_dim) * 0` (zero), seguido de `(void)kbase` — morto |
| `include/rdna4/attn.cuh:198-200` | "the number of partial slices merged per CTA changes the summation order" (no launcher do kernel sem split) | mesma coisa que a linha 88-93 |
| `include/rdna4/device.h:33-40` | "17 full-attention layers × 4 KV heads × (256+256)" | duas linhas depois: "16 KV-bearing full-attention layers"; a constante é `16*4*(256+256)` — a primeira linha é resíduo do M1 |
| `include/rdna4/kv.h:238-241` | "padrão determinístico … cache realista e não degenerada" | no ramo `q4_0` a escala do bloco é escrita por 16 lanes com valores diferentes e os `qs` usam escalas que não são a que ficou (F7) |
| `docs/rocm-estudo.md:486-492` | "assim **todo contexto com ≥8 splits (16K para cima) roda exatamente o mesmo caminho de antes**" | falso depois de `kAttnSplitWpbWide = 16` (o doc do autotuning registra a mudança em `:74`, este não foi atualizado); a 8K-16K chaves o caminho **mudou** para 16 warps |
| `tests/check_tuning.cpp:33` | "14 tipos de matvec + 4 da atenção + 3 do batch + 1 do total = 22 linhas" | 21 linhas, 2 do batch (F10) |
| `include/rdna4/device.h:76-87` | — | as duas funções são código morto, e uma delas contém a fórmula pré-M4 (F8) |
| `README.md:353-355` | avisos do `-Wall -Wextra` "queued with the attention changes" | já corrigidos (F11.4) |

**Como verifiquei**: cada linha lida no arquivo (o grep de TODOs do repo volta vazio — a árvore
não tem `TODO`/`FIXME`/`XXX`/`HACK`, então este é o formato em que a dívida aparece aqui).
**Conserto**: corrigir os comentários no mesmo commit que mexer nos respectivos arquivos; o de
`rocm-estudo.md` merece uma nota de revisão explícita, porque é um argumento de não-regressão que
deixou de valer.

---

## F14 — BAIXO — o profiler de fases: inerte, sim; três arestas

**O que confirmei**: `RD_PHASE` é **realmente** inerte com `prof_ == nullptr` — enumerei **todos**
os sítios (`grep -rn "RD_PHASE\|prof_\|pfine\|phase_prof" include/ src/ tests/`): 11 marcas por
macro, 2 usos diretos ambos guardados por `if (prof_ != nullptr)` (`graph.cuh:1255,1260`), nenhum
outro uso de `prof_` no motor; `pfine()` devolve `nullptr` com o profiler desligado
(`graph.cuh:112`). A base de tempo também é válida: **não existe `hipStreamCreate` na árvore** e
`graph.cuh` nunca passa stream a um launcher, então `hipEventRecord(e, nullptr)`
(`include/rdna4/phase_prof.cuh:99`) está na mesma fila dos kernels; `bench_phases_gpu.hip:564-566`
sincroniza antes de `finish()`, que é o que `hipEventElapsedTime` exige.

**Arestas (não são bugs hoje, são armadilhas)**:
1. O macro avalia o argumento **duas vezes** (`if ((p) != nullptr) (p)->mark(name)` com
   `p = pfine()`): hoje `pfine()` é puro e barato, mas um argumento com efeito colateral seria
   chamado 2× por marca, 1479 vezes por token.
2. Um `hipEvent_t` novo por marca, **nunca destruído** (`:95-99`; não há `hipEventDestroy` em
   `phase_prof.cuh`): o pool cresce com `tokens × marcas/token` (35k eventos numa passada de 24
   tokens a nível 2). Para a ferramenta de medição é aceitável; se alguém ligar o profiler num
   `serve` de vida longa, é vazamento.
3. Se `hipEventCreate`/`hipEventRecord` falhar, `on = false` e o `finish()` **retorna cedo**
   (`:96-99`, `:109-121`): o relatório continua sendo impresso com os buckets parciais e **sem
   nenhum aviso de "perfil incompleto"**. O mesmo vale para `hipEventElapsedTime` que falhe (o
   bucket fica com 0 ms em silêncio, `:113-116`).
4. `count_only` (`:55-58`, usado por `bench-phases-gpu --count-only`) é de fato custo zero no
   dispositivo — nenhum evento é criado —, o que sustenta a contagem de 1479 marcas/token como
   "medida sem perturbar" (`docs/medicoes-banda-e-gargalos.md:151`).

**Conserto (descrito)**: guardar o argumento numa variável local (macro ou `inline` helper),
destruir os eventos em `finish()`, e imprimir `WARNING: profile incomplete (N marks dropped)` quando
`on` cair para `false`.

---

## F15 — MÉDIO — o KV próprio do `MtpHead` não entra em nenhum orçamento de VRAM

**Local**: `include/rdna4/mtp.cuh:365-372` (`alloc_kv`), consumido por
`src/main.hip:918-925` e `src/server/serve.hip:385` via `device.h:64-87`.

**Cenário de falha**: `--mtp` aloca **duas** caches (`d_ck_`, `d_cv_`) de
`max_ctx × n_head_kv × kv_row_bytes(tipo)`. Para `NKV = 4`, `HD = 256`, KV `f16` isso é 2 × 2048 B
por token = **4 KiB/token** ⇒ **0,25 GiB a 64K** e **0,5 GiB a 128K** — e o orçamento é
`file_bytes + kv_cache_bytes(ctx, k, v) + kOverheadBytes` (`device.h:84-87`), sem esse termo.
Consequência: uma configuração que o `info` aprova ("budget OK") pode morrer em
`graph init failed: hipMalloc failed` ao ligar `--mtp`, exatamente o sintoma que o achado M4 e a
§2.4 da medição passaram a noite consertando. Ninguém mediu o "cabe/não cabe" da tabela de
`docs/quants-precisao.md:242-249` com `--mtp` ligado (a coluna não diz), e a 64K `f16` a margem é de
0,30 GiB — **menor** que a cache do MTP.

**Contradição de documento no mesmo item**: `docs/mtp.md:269` descreve "8 MiB a 4K (2 MB por 1K de
contexto), então a 64K são 128 MiB" — **uma** cache; o código aloca duas, e a auditoria já tinha
notado (`docs/auditoria-qualidade.md` §E5): o certo é 4 MB por 1K, 16 MiB a 4K, 256 MiB a 64K.

**Como verifiquei**: leitura de `alloc_kv` + `grep -n "kv_bytes_\|d_ck_\|d_cv_" mtp.cuh` + leitura do
orçamento em `device.h` e do sítio de checagem em `main.hip:918-925`.

**Conserto (descrito)**: somar o termo quando `--mtp` estiver ligado
(`2 * ctx * n_head_kv * kv_row_bytes(kv_k)` + o mesmo para V, que pode ter tipo diferente) tanto no
`info` quanto no `serve`; e corrigir `docs/mtp.md:269`.

---

## O que eu **não** consegui verificar (e por quê)

1. **A medida que fecha F5** (PPL a 8192 chaves, `RD_ATTN_SPLITS` 15/16/default, ordem rotacionada):
   **abandonada na fila da GPU**. Das ~1500 s em que o comando existiu, ~260 s foram espera de
   `flock` com 11-12 processos na frente e um `rdna4-infer` de outra frente segurando 12,02 GiB; o
   `timeout 900` que a regra 2 do contrato exige não cobre a espera **mais** as 6 corridas de PPL
   (~9 min). O desenho completo está no diário (entrada 13) e é re-executável em uma janela livre.
   **Consequência**: F5 fica com a garantia por construção e sem a medida — eu não afirmo que há
   regressão, afirmo que a evidência que o doc apresenta não cobre a combinação embarcada.
2. **Os números da tabela final do README** (30,4 / 29,3 / 26,7 / 19,3 / 14,0 tok/s, 72,9 prefill):
   existem **só** no README (nenhum doc de origem os contém: grep por `30,4`, `72,9`, `19,3`), foram
   medidos pelo coordenador numa janela que eu não posso re-entrar de noite. Só pude checá-los por
   **consistência interna** — e é daí que saem os problemas de F11. Não sei qual dos dois números
   de 64K `q8_0` é o "certo"; sei que dois não podem estar certos ao mesmo tempo.
3. **A contagem de 186 avisos/47 linhas do M10**: confirmei a *classe* por leitura em 8 sítios e a
   inalcançabilidade por leitura do validador; não re-rodei `clang-tidy` (o binário existe, mas
   refazer a varredura curada e reconciliar as contagens de 47 linhas não cabia na caixa de tempo
   desta frente, e o resultado não mudaria a prioridade).
4. **Os números de desempenho do autotuning** (ganhos de `unroll=2` por tipo, as rodadas de A/B,
   o piso de ruído 1,001x, o custo de 4,03 µs por marca): GPU-only, fora do meu orçamento (2-3
   execuções, e a placa estava ocupada). O que fiz com eles foi **checagem de consistência interna**
   (unidades, aritmética, se o veredito segue da tabela) — não re-media.
5. **A varredura "unsplit com warps forçadas"** de F2: o efeito está argumentado a partir do código
   (as 4 linhas do corpo usam a constante), não medido por mim. A leitura é conclusiva quanto à
   *fiação* (o `WPB` do template não entra no corpo), mas o número específico que a varredura
   publica (o quanto "16 warps" parece melhor ou pior) eu não re-mediei.
6. **O backlog de 221 linhas**: verifiquei pessoalmente os ponteiros das ~30 linhas que sustentam o
   topo do ranking (as que cito nesta revisão); as demais vieram de uma extração mecânica com grep
   por linha e estão marcadas no `docs/backlog-noite.md` com a coluna de verificação. Não re-li os
   16 documentos-fonte inteiros: leitura integral foi feita em `auditoria-qualidade.md`,
   `autotuning-gfx1201.md`, `medicoes-banda-e-gargalos.md` (seções citadas), `gpu-queue.md`,
   `noite-regras.md`, `journal-noite.md` e nos arquivos de código listados na tabela do topo.
7. **A validação de `serve`/HTTP de ponta a ponta** (H1) e os gates de GPU: a bateria de GPU não é
   minha nesta janela; `check-server.sh` precisa da placa. Rodei só a metade CPU (10 gates, PASS).

---

## Falsos positivos que descartei (para não virarem trabalho à toa)

- **Split vazio** em `attn_split_kernel` (`:373-381`) e em `attn_merge_kernel` (`:420-423`):
  corretos; `expf(-inf - x)` = 0 e o caso "todos vazios" devolve zeros. Sem NaN.
- **`Graph::release()`**: idempotente de verdade — os 18 ponteiros de `ptrs[]` **e** os 16 de
  `batch_ptrs[]` são zerados depois do `hipFree` (`graph.cuh:1341-1378`). O M1 está fechado.
- **`check_regression.sh` "gate vazio"**: a checagem de shell (`grep -q 'rel-L2'`) é fraca, mas o
  **binário** é forte: `want.size() != got.size()` → FAIL, `kMinCases`, checagem por caso de
  nome/`n_prompt`/`n_gen`/tamanho e stride da subamostra, ids bit-exatos, `rel-L2` e `|Δ‖logits‖|`
  (`tests/check_regression_gpu.hip:478-589`). Não é um gate vazio; o `grep` é redundante.
- **Chamadas HIP sem checagem no motor**: `hipMemcpy`/`hipMemset`/`hipMemGetInfo` estão **todas**
  checadas (ou devolvem `== hipSuccess`); as únicas sem teste são os 17 `hipFree` de liberação, com
  `(void)` deliberado (achado H6, cosmético). O censo da auditoria se confirma.
- **Teto de `--ctx-size`**: existe nos cinco comandos (`run`/`bench`/`ppl`/`info`/`serve`,
  `main.hip:258,598,830,1043` e `serve.hip`) e vem **antes** da checagem de dispositivo, então é
  testável sem GPU, como a §7.5 afirma.
- **`phase_prof` fora de faixa**: `pending = -1` no início e o primeiro `mark` não empilha par;
  `ev[p.open_ev]`/`ev[p.close_ev]` sempre com índices válidos.
- **O `WPB` no kernel com split**: honrado no corpo (`:318-319`, `:363-405`) — não confundir com F2,
  que é só o kernel sem split.
- **`gguf.cpp` caps do M8**: `kMaxStrBytes`/`kMaxArrElems` e o teto por bytes restantes existem;
  a única ressalva é que `remaining()` vale `UINT64_MAX` quando o `fseek(SEEK_END)`/`ftell` falha
  (`gguf.cpp:24,61-67,163-171`), isto é, a metade "cabe no que resta do arquivo" da correção fica
  **inativa em silêncio** para um arquivo não-seekable. Registro aqui como observação, não como
  achado: com arquivo regular (o caso real) os dois freios valem, e o teto absoluto continua.
