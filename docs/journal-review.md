# Diário — frente 1: revisão adversarial + backlog (task 1 da noite)

Worktree `/home/marcelo/Projetos/rdna4-wt-noite-review` (branch `feat/noite-review`, base
`main` = tag `noite-baseline-2026-09-14` = `e48f3c4`). Regras: `docs/noite-regras.md`.
Read-only em relação ao código do motor: **nenhum arquivo de `include/`, `src/`, `tests/`,
`scripts/` ou `CMakeLists.txt` foi tocado**; os únicos arquivos escritos são os três `.md`
desta frente em `docs/`.

Formato de cada entrada: referência · hipótese (o que eu queria *falsificar*) · comando · resultado ·
veredito. Onde não houve comando de GPU, a verificação é leitura de código + aritmética, e isso está
dito explicitamente.

Condição de janela da sessão inteira: `mem_info_vram_used` começou em **0,31 GiB** (limpo) e passou a
**12,02 GiB** às 02:1x com um `rdna4-infer` de outra frente segurando o lock — todas as medições de
GPU desta frente foram feitas depois disso e ficaram **na fila** (ver entrada 13).

---

## 1. `-Wall -Wextra` está mesmo ligado, e a árvore está limpa?

- **Referência**: commit `e48f3c4` ("… `-Wall -Wextra` no motor …"); `README.md:353-355`;
  `docs/auditoria-qualidade.md` §B1/§7.3.
- **Hipótese a falsificar**: (a) as flags estão ligadas nos alvos do motor; (b) a árvore compila
  sem avisos nelas; (c) o README já reflete isso.
- **Comando**: `source scripts/rocm-env.sh && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release &&
  cmake --build build -j3` (CPU puro, exit 0) + `grep -i warning: /tmp/build.log | grep -o '^[^:]*' | sort -u`.
- **Resultado**: build exit 0; **428** linhas `warning:` — **todas** de `tests/*.hip`
  (`-Wunused-value` de `hipError_t [[nodiscard]]`), **nenhuma** de `src/` nem de `include/`.
  `CMakeLists.txt:38,40` (`rdna4-infer`) e `:246` (`rdna4_serve`) aplicam `RDNA4_WARNINGS`.
  Logo (a) e (b) são verdadeiros; (c) é **falso**: o README ainda diz que os avisos estão
  "queued with the attention changes" e que `attn.cuh` produz 98 `-Wsign-compare`.
- **Veredito**: MANTIDO para a árvore; **achado** para o README (review F11).

## 2. O `RD_PHASE` é inerte quando `prof_ == nullptr`? (todos os sítios, não o macro)

- **Referência**: `include/rdna4/graph.cuh:103-113,1186-1271`; `include/rdna4/phase_prof.cuh:159-164`.
- **Hipótese a falsificar**: existe algum sítio que toca o profiler (ou faz trabalho) fora do
  `if ((p) != nullptr)` do macro.
- **Comando**: `grep -rn "RD_PHASE\|prof_\|pfine\|phase_prof" include/ src/ tests/` (lista completa)
  + leitura de cada sítio.
- **Resultado**: 11 marcas por macro (`prof_` em 1186/1209/1216/1229/1256/1263/1271; `pfine()` em
  596/602/622) e exatamente 2 usos diretos, ambos guardados (`set_tag` em 1255 e 1260).
  `pfine()` devolve `nullptr` quando `prof_ == nullptr`. **Nenhum** outro uso de `prof_` no motor.
  O macro avalia o argumento duas vezes (`RD_PHASE(pfine(), …)` chama `pfine()` 2×) — sem efeito
  colateral hoje, mas é armadilha para um argumento futuro.
- **Veredito**: MANTIDO (é inerte de verdade). Nuances registradas no review F14.

## 3. A base de tempo dos eventos do profiler é a mesma dos kernels?

- **Referência**: `phase_prof.cuh:99` (`hipEventRecord(e, nullptr)`); `graph.cuh` inteiro.
- **Hipótese a falsificar**: o grafo lança em stream não-default, o que tornaria as fronteiras de
  bucket inválidas.
- **Comando**: `grep -rn "hipStream_t\|hipStreamCreate\|hipStreamDestroy" include/ src/ tests/`.
- **Resultado**: **0** `hipStreamCreate` na árvore; todos os launchers têm
  `hipStream_t stream = nullptr` como default e `graph.cuh` **nunca** passa outro stream (o arquivo
  não contém nenhuma ocorrência de `stream` em chamada de launcher). Logo eventos e kernels estão na
  mesma fila. `bench_phases_gpu.hip:564-566` sincroniza **antes** de `finish()` (sem isso,
  `hipEventElapsedTime` falharia e o bucket ficaria 0 ms em silêncio).
- **Veredito**: MANTIDO. Risco residual descrito em F14 (falha de `hipEventCreate` desliga o
  profiler no meio e o relatório continua sendo impresso sem aviso).

## 4. `tuning.h` é mesmo a única cópia dos números?

- **Referência**: `tuning.h:5-12` ("nao existe uma segunda copia dos numeros em lugar nenhum");
  `tests/check_tuning.cpp:3-9`.
- **Hipótese a falsificar**: algum consumidor usa um literal em vez da constante.
- **Comando**: `grep -rn "tuned::" include/ src/ tests/` (49 linhas) + leitura de
  `attn.cuh:554-566,573-582` e `graph.cuh:321-341`.
- **Resultado**: **falsificado**. `attn.cuh:563` e `:564` devolvem o literal `16` (o valor "largo"
  **não** é lido de `tuned::kAttnSplitWpbWide`, que só fornece o *limiar*), e `attn.cuh:575` compara
  o resultado com outro literal `16` para escolher entre as instanciações 16 e 8. Trocar
  `kAttnSplitWpbWide` para 32 (ou `kAttnWarpsPerBlock` para 4) em `tuning.h`, atualizar o golden e
  rodar `check-tuning` deixa o gate **verde** enquanto o kernel continua com 16/8 warps.
  Segundo achado da mesma classe: `graph.cuh:164` (`kMaxBatch = 16`) e `:167-169`
  (`batch_supported()` com `{2,3,4,8,16}`) duplicam `tuned::kBatchCap`/`kBatchNs`.
- **Veredito**: **achado** (review F1), P1 no backlog. O gate protege a tabela, não a fiação.

## 5. O gate `check-tuning` pode passar depois de uma mudança silenciosa?

- **Referência**: `tests/check_tuning.cpp`; `tests/golden/ml_tuning.txt`.
- **Hipótese a falsificar**: o gate é uma comparação vazia ou comparável demais.
- **Comando**: `./build/check-tuning` (via `check_all.sh`), leitura do `report_lines()` e do
  caminho `--record`.
- **Resultado**: o gate compara 21 linhas com chave obrigatória presente e `ref.size() == now.size()`
  — não é vazio. Mas (a) ele lê **apenas** `tuning.h`: nenhuma linha imprime o valor *efetivo* que um
  consumidor usa (por isso o achado 4 passa por ele); (b) `kMinLines = 21` é igual ao total exato, ou
  seja o "mínimo" é de fato uma igualdade; (c) o comentário da linha 33 ("14 + 4 + 3 + 1 = 22") conta
  errado (são 21 linhas, e há 2 do batch, não 3).
- **Veredito**: MANTIDO com ressalva — **achado** (review F10).

## 6. `check_all.sh` pode passar por motivo errado quando um gate não está compilado?

- **Referência**: `scripts/check_all.sh:50` (`[ -x "$B/check-tuning" ] && step …`) e `:88`
  (`if [ -x "$B/check-regression-gpu" ]; then step … fi`).
- **Hipótese a falsificar**: um binário ausente faz o gate sumir sem falhar.
- **Comando**: `mv build/check-tuning /tmp/ && CHECK_ALL_CPU_ONLY=1 ./scripts/check_all.sh;
  mv /tmp/check-tuning.bak build/check-tuning`.
- **Resultado**: **falsificado**: a bateria imprime **9** gates em vez de 10 e termina com
  `check_all (CPU half): PASS` (exit 0), sem nenhuma menção ao gate que não rodou. O mesmo
  construto existe na fase B para a suíte de regressão (o gate numérico mais forte).
- **Veredito**: **achado** (review F3) — o mais grave da classe "gate".

## 7. A prova de forja do `check_hardening.sh` se sustenta? (reproduzir antes/depois por conta própria)

- **Referência**: `docs/auditoria-qualidade.md` §7.1; `tests/gen_forged_gguf.py`;
  `tests/check_forge.cpp`; `scripts/check_hardening.sh`.
- **Hipótese a falsificar**: (a) o offset forjado não exercita a soma guardada; (b) a coluna "antes"
  é retórica, não medida.
- **Comando** (CPU puro): `git archive aa15eea include src | tar -x -C /tmp/prefix` (o commit
  **pré-correção** da auditoria), `python3 tests/gen_forged_gguf.py /tmp/forged`,
  `g++ -O2 -std=c++17 -I/tmp/prefix/include tests/check_forge.cpp
  /tmp/prefix/src/backend/{gguf,loader}.cpp -o /tmp/prefix/check-forge-old`, e os dois probes nos
  cinco arquivos.
- **Resultado**: **as cinco linhas da §7.1 se reproduzem exatamente**, sem usar worktree nenhum:
  `ok` aceito nos dois; `offset_wrap` **aceito** pelo antigo (`offset=18446744073709548768`, cabeça
  `71 77 65 6e 33 35` = ASCII `qwen35`) e rejeitado pelo novo com
  `offset/size sum overflows 64 bits`; `dim_huge` rejeitado pelo antigo por **motivo errado**
  (`end not aligned`) e pelo novo com `computed size … exceeds the 29312-byte file`; `str_len`
  antigo **`rc=134`, `std::bad_alloc`** e novo rejeitado; `arr_len` rejeitado pelos dois.
  Aritmética do caso forjado: `off_wrap = 2⁶⁴ − doff + 64`, logo `doff + off_wrap = 2⁶⁴ + 64` →
  `add_checked` (loader.cpp:24-28) falha em **i=0**, que é justamente o único tensor sem
  antecessor (é o predecessor que limitaria os outros) — a forja explora o caso certo.
- **Veredito**: MANTIDO (a prova é real e agora reproduzível por `git archive`). Duas ressalvas no
  review F12: o `check_all.sh` chama o script **sem** o argumento, então a metade "antes" nunca roda
  na bateria; e o argumento documentado é um *worktree* pré-correção que já não existe
  (`rdna4-wt-medicoes-gpu` está mergeado na `main`).

## 8. A cauda do kernel split (achado M2) foi mesmo fechada?

- **Referência**: `attn.cuh:395-407`; `docs/auditoria-qualidade.md` §M2 e §7.3.
- **Hipótese a falsificar**: sobrou dimensão sem escrita, ou a correção mudou a aritmética do caso
  que já funcionava.
- **Comando**: leitura de `attn.cuh:383-408` + aritmética por configuração.
- **Resultado**: o laço `for (d = threadIdx.x; d < head_dim; d += blockDim.x)` cobre
  `[0, head_dim)` para qualquer `WPB`; `out[0]`/`out[1]` são escritos por `threadIdx.x == 0`.
  Para `head_dim = 256, WPB = 8` (o embarque) rodam 256 threads ⇒ **uma** iteração por thread, ou
  seja a mesma ordem de antes da correção — bit-exatidão preservada onde já valia. Para
  `head_dim > WPB*32` (o caso do M2, até 512) agora escreve tudo.
- **Veredito**: MANTIDO (M2 fechado).

## 9. O `WPB` no kernel **sem** split é mesmo ignorado? (achado E2, ainda aberto)

- **Referência**: `docs/auditoria-qualidade.md` §E2 (cita `attn.cuh:118,162,168,175`);
  `attn.cuh:88-93`.
- **Hipótese a falsificar**: E2 foi corrigido junto com a cauda (o briefing sugere que "a cauda do
  kernel split" e o WPB saíram no mesmo commit `e48f3c4`).
- **Comando**: `grep -n "WPB\|kAttnWarpsPerBlock" include/rdna4/attn.cuh` + leitura de 94-217.
- **Resultado**: **E2 continua aberto**: `attn.cuh:121` (passo do laço de chaves), `:165`, `:171`,
  `:178` (merge) usam `kAttnWarpsPerBlock`, não o `WPB` do template; só o lançamento (191-193) usa o
  template. Com `WPB = 16` as warps 8-15 **repercorrem** as fatias de chave das warps 0-7
  (`j = w + k*8`) e as fatias 8-15 do smem nunca são lidas pelo merge (que vai até
  `kAttnWarpsPerBlock`) ⇒ resultado idêntico ao `WPB = 8` e tempo de trabalho duplicado. Portanto
  `tests/bench_attn_gpu.hip:400,409,504` ("unsplit kernel with forced warps/CTA") mede trabalho
  redundante, e o comentário `attn.cuh:88-93` ("changing it changes the number of partial slices
  merged … the summation order") descreve código que não existe. O kernel **com** split honra o
  `WPB` (318-319, 363-405), então o caminho embarcado não é afetado.
- **Veredito**: **achado** (review F2), P1. Confirmado por leitura; a varredura do bench não é
  re-executável nesta frente (a placa é de outra frente), então o efeito é argumentado, não medido.

## 10. `kAttnSplitWpbWide` preserva os números em **todo** contexto?

- **Referência**: `attn.cuh:512-566`; `docs/autotuning-gfx1201.md` §3.2;
  `scripts/check_attn_split.sh`; `tests/check_kvctx_gpu.hip:219-294`.
- **Hipótese a falsificar**: existe contexto em que o `WPB = 16` embarcado muda o resultado além da
  classe de reordenação, ou não há evidência para a combinação embarcada.
- **Comando**: leitura do kernel + aritmética de cobertura + leitura dos dois gates (CPU); e um
  experimento de GPU desenhado para fechar isto (entrada 13).
- **Resultado (garantia por construção)**: para qualquer `WPB`, o conjunto
  `{w + WPB·s : w ∈ [0,WPB), s ∈ [0,S)}` é exatamente `[0, WPB·S)`, então a cobertura de chaves é
  exata e não há contagem dupla; muda apenas a ordem de soma dentro de cada par (cabeça, split) —
  a classe `rel-L2 ~2,5e-7` declarada em `attn.cuh:536-538`.
  **Resultado (evidência)**: os gates **não** cobrem a ponta larga em texto real com a política
  embarcada: `check_attn_split.sh` roda `splits = 4` a `ctx = 1024` (ponta estreita, mutação
  `RD_ATTN_SPLITS`); o único gate que toca 16 splits é `check-kvctx-gpu 65536 q4_0`, com cache
  **sintético aleatório** e tolerância **5e-2** (declarada frouxa no próprio arquivo, 279-286); o
  caso de 32K do golden de regressão foi **gravado com a regra nova** (commit `9ef1526`), então não
  distingue `WPB` 8 de 16. Além disso a regra é chaveada só no *número* de splits, enquanto a medida
  que a motivou é específica de `q4_0` (`attn.cuh:546-551`): a própria varredura f16 do arquivo dá
  `WPB=16` **3% pior** a 16 splits (0,0844 contra 0,0819) e o A/B a 64K deu 1,004x em 6/15 rodadas
  ("nada acima do ruído").
- **Veredito**: **parcialmente falsificado** — a garantia vale, a *evidência por contexto* não
  (F5). É o item P1 mais caro de fechar e o experimento da entrada 13 é o desenho mínimo para isso.

## 11. O `q8_0` ganha do `q4_0` a 64K? (efeito de 0,2-1,8%)

- **Referência**: `docs/medicoes-banda-e-gargalos.md:485-490` (54,45 contra 55,02 ms; 18,36 contra
  18,17 tok/s; atenção 19,87 contra 21,03 ms); `docs/quants-precisao.md:249-254` (18,28 contra
  18,10-18,17 tok/s); `README.md:184,369` (18,28) contra `README.md:202` (19,3).
- **Hipótese a falsificar**: o delta é maior que o piso de ruído do harness e as duas pernas foram
  medidas na mesma janela.
- **Comando**: leitura dos três documentos + `grep -rn "18\.28\|18,28\|19\.3\|13\.75" docs/ README.md`
  (não rodei nada: é uma comparação de *registros*, e a placa não é minha nesta janela).
- **Resultado**: **falsificado como afirmação de velocidade**. Os dois braços vêm de **corridas
  separadas** (não há A/B intercalado — o protocolo que o próprio repo exige para decidir,
  `docs/autotuning-gfx1201.md` §2), o efeito (+0,5% a +1,8%) é **menor** que a variabilidade
  entre sessões documentada (2-3%, `docs/rocm-estudo.md:504-507`), e o README cita **três** números
  para a mesma configuração (mesma pegada de 13,75 GiB): 19,3 na tabela final, 18,28 na prosa
  (herdado do doc) e 18,36 no doc de medições. A metade *qualidade* da conclusão (o `q8_0` tem o
  dobro da precisão do KV, 2,18 GiB de folga e não é mais lento) sobrevive — é ela que sustenta a
  recomendação acima de ~56K.
- **Veredito**: **achado** (review F11) — rebaixar "também é *mais rápido*" para "não é mais lento
  dentro do ruído do harness"; a troca continua recomendada pelo eixo precisão/folga.

## 12. Números do README contra os docs (amostra de 12)

- **Referência**: `README.md` §"Measured numbers", §"Where a token goes", §"Code quality".
- **Hipótese a falsificar**: a tabela final da árvore mergeada é consistente com os docs que cita.
- **Comando**: greps dirigidos (`grep -rn "18\.28\|13\.86\|402-421\|72,9\|575\|1479" docs/ README.md`)
  + aritmética linha a linha.
- **Resultado** (12 checagens, 7 ok / 5 com problema):
  1. ✅ `tg64 = 40,0 ± 0,02` (`baseline-vulkan-iq3s.md:46`) e `39,7` em M5 (`medicoes-m5.md:24`).
  2. ✅ `pp64 = 575 ± 65` (`baseline:45`), `pp512 = 1143 ± 30` (`baseline:48`).
  3. ✅ `436 GB/s = 69%` e `620 GB/s = 98%` e `352/633 = 56%` (aritmética confere).
  4. ✅ `1479 marcas × 4,03 µs = 5,95 ms` e "16% de inflação" (`medicoes-banda:106-109`, 4,026 µs).
  5. ✅ `497 matvec + 257 quantizações` (`medicoes-banda:154`).
  6. ✅ `iq3_xxs 16,8%` = `quants-inventario.md:124` (16,75%); a tabela do autotuning (`:122`) diz
     16,7% — divergência de arredondamento entre docs, não erro do README.
  7. ✅ `13,86 GiB` do 128K `q4_0` = `quants-inventario.md:332` e `medicoes-m5.md:149`.
  8. ❌ `64K q8_0`: 19,3 (README:202) contra 18,28 (README:184/369) contra 18,36 (doc) — ver
     entrada 11.
  9. ❌ Tabela de fases: as 6 linhas somam **37,87 ms** contra o total declarado de **36,0**. Causa
     exata: o LM head (1,43) está **dentro** da linha (a) 24,94 ("tronco 23,51 + head 1,43") **e**
     na linha (f) 2,08; e a linha (f) ainda soma o sampler de 0,44 ms, que é *host* e não existe no
     total de 36,0 (que é a linha de tempo do dispositivo). A tabela fina do doc
     (`medicoes-banda:113-129`) soma 35,98 ✅ — só a tabela agrupada (`:137-145`) e a cópia dela no
     README é que contam duas vezes.
  10. ❌ `README:298-299` diz "this engine's **measured** 402-421 GB/s", mas 402-421 GB/s é a
      **estimativa de replay** de `rocm-estudo.md:61` (§A.1); a medida é 436 GB/s no tronco
      (`medicoes-banda:519`) — o próprio README:206 usa 436.
  11. ❌ `README:353-355` (avisos "queued") e `:72-73` ("422 de 433") descrevem a árvore
      pré-merge: nesta são 428 avisos, todos de `tests/`, e as 3 linhas de aviso do motor já foram
      corrigidas (`[[maybe_unused]]` em `attn.cuh:412`, `gdn.cuh:60`, `graph.cuh:644`).
  12. ❌ `README:169`: a linha de 128K não fecha com as próprias parcelas
      (11,133 + 2,416 + 0,302 = **13,851**, e o texto diz 13,87; a de 64K erra 0,01 do mesmo jeito).
      O erro vem da fonte (`kv-memoria-desenho.md:325`) — corrigir lá primeiro.
- **Veredito**: **achado** (review F11), P2/P3 por item; a tabela final precisa de uma passada de
  consistência interna antes de virar "o" número do repo.

## 13. Fechar a lacuna de evidência do `WPB` largo com uma medida de GPU (8K chaves, 15 vs 16 splits)

- **Referência**: `attn.cuh:546-566`; `docs/autotuning-gfx1201.md` §3.2; `RD_ATTN_SPLITS`
  (`graph.cuh:329-336`).
- **Hipótese a falsificar**: com a **política embarcada** a 8192 chaves (= 16 splits → `WPB=16`,
  ponta larga) o PPL de texto real difere de `RD_ATTN_SPLITS=15` (= `WPB=8`, mesma classe de
  contagem de splits) mais do que a classe de reordenação (esperado: ≤ 0,01%).
- **Comando** (desenhado, ordem rotacionada para cancelar deriva de DPM):
  `timeout 900 ./scripts/gpu-lock.sh bash /tmp/attn-wpb-ppl.sh`, que roda
  `rdna4-infer ppl -m IQ3_S -f reference/data/wikitext-2-raw/wiki.test.raw --ctx-size 8192
  --stride 8192 --chunks 2 --no-stats` com `RD_ATTN_SPLITS` = 15, 16, default, 16, 15, default.
- **Resultado**: **ABANDONADO (fila da GPU, com número)**. Cronologia medida: o comando entrou na
  fila às 02:20 (`flock` com 11-12 processos esperando e um `rdna4-infer` de outra frente segurando
  12,02-12,9 GiB), **conseguiu o lock ~4,5 min depois** (`window before: vram_used=1527418880`), e a
  primeira corrida (`RD_ATTN_SPLITS=15`) **ficou >7 min sem terminar** (máquina com load average ~8,
  outras frentes carregando o modelo de `/mnt/raid0` ao mesmo tempo): eram precisas 6 corridas de
  ~4-5 min e o `timeout 900` (regra 2 do contrato) mataria o comando no meio, sem par completo.
  **Matei o meu próprio comando** (regra 5) para devolver o lock às outras cinco frentes — o `fuser`
  mostrou outro processo assumindo o lock 3 s depois. **Zero pontos de dados.**
- **Veredito**: ABANDONADO com o motivo medido (fila + parede). A lacuna de evidência do achado 10
  **continua aberta**; em janela livre o desenho custa ~30 s de lock por corrida (6 corridas, uma
  aquisição de lock) e é o gate que falta (`rev-attn-wide-branch-gate` no backlog, posição 15).

## 14. `kv_row_bytes` devolve 0 para tipo desconhecido — o que quebra quando `q5_0`/`q4_1` entrarem?

- **Referência**: `include/rdna4/kv.h:55-75,199-235,261-280`; `docs/noite-regras.md:16-23` (o alvo
  da noite é justamente K em `q5_0` / V em `q4_1`).
- **Hipótese a falsificar**: o motor falha alto ao ganhar um `KvType` novo.
- **Comando**: leitura de `kv.h` inteiro + `grep -rn "kv_row_bytes\|kv_bytes_per_elem" include/ src/`
  para achar os consumidores.
- **Resultado**: **falsificado**: `kv_row_bytes` termina em `return 0` (`:74`) sem `default` que
  grite; `kv_store_row_kernel` **não tem** `if (CT == Q4_0)` — o corpo do `q4_0` é o *fallthrough*
  (`:211-235`), então um tipo novo é **armazenado como q4_0**; `kv_fill_kernel` idem (`:280`);
  `kv_bytes_per_elem` devolve 0,0 (`:62`), o que faz `kv_cache_bytes` devolver 0 e o orçamento de
  `info`/`serve` **aprovar** uma configuração sem contar KV. Do lado host, `graph.cuh:728-735` e
  `MtpHead::alloc_kv` (`mtp.cuh:367`) usam `kv_row_bytes` para o passo de linha: com 0 byte, toda
  linha cai no mesmo endereço — sem exceção, sem mensagem, saída errada.
  `kv_load<CT>`/`kv_load8<CT>` são templates declarados sem definição primária, então *lá* um tipo
  novo é erro de compilação (a rede de segurança que falta no resto).
- **Veredito**: **achado** (review F6), P1 para a frente KV desta noite.

## 15. O `bench` conta os bytes certos por token? (banda de memória)

- **Referência**: `src/main.hip:373,451,480`; `README.md:121-128,205-206`;
  `docs/quants-inventario.md` §3.3.
- **Hipótese a falsificar**: a banda reportada usa o tráfego real por token.
- **Comando**: `sed -n '360,385p;440,485p' src/main.hip` + `grep -rn "total_bytes" include src`.
- **Resultado**: **falsificado**: `bytes_per_token = loader.total_bytes()` — o arquivo inteiro
  (**12,030 GB**), incluindo a matriz `token_embd.weight` (da qual só uma linha é lida) e o bloco
  MTP `blk.64.*` (que o `bench` não executa). O tráfego real por token é **11,122 GB** (a tabela do
  próprio README:127 e `quants-inventario` §3.3). Fator **1,082**: todo "GB/s de pesos" do `bench`
  está **8,2% otimista**, e as linhas de banda do README herdam isso (12,030/34,2 ms = 351,8 GB/s =
  o "352 GB/s = 56%" do `README:199`; o honesto é 325 GB/s = **51%** da roofline de 633 GB/s).
- **Veredito**: **achado** (review F4), P1 — é uma correção de uma linha no código (fora do meu
  escopo) que muda números publicados.

## 16. O `MtpHead` aloca KV fora do orçamento?

- **Referência**: `include/rdna4/mtp.cuh:365-372`; `src/main.hip:918-923`.
- **Hipótese a falsificar**: `--mtp` cabe em qualquer configuração que `info` aprove.
- **Comando**: leitura de `alloc_kv` + `grep -rn "kv_bytes_\|d_ck_" include/rdna4/mtp.cuh`.
- **Resultado**: **falsificado**: são **duas** caches (`d_ck_`, `d_cv_`) de
  `max_ctx * NKV * kv_row_bytes(tipo)`. Para `NKV=4, HD=256, f16` isso é 4 KiB por token ⇒
  **0,25 GiB a 64K** e **0,5 GiB a 128K**, e nada disso entra em
  `required_bytes = file_bytes + kv_cache_bytes + 1 GiB` (`device.h:84-87`). O "1 GiB de overhead"
  pode ou não cobrir; a tabela de "cabe/não cabe" (`docs/quants-precisao.md:242-249`) não diz se foi
  medida com `--mtp`. Em 48K f16 (1,30 GiB livres) ainda cabe; a 64K f16 (0,30 GiB livres) o modo
  `--mtp` só pode piorar — e `docs/mtp.md:269` ainda descreve a cache como **uma** (8 MiB a 4K),
  quando o código aloca duas (16 MiB a 4K) — contradição entre documento e código.
- **Veredito**: **achado** (review F15), P2. Verificado por leitura.

## 17. Sobrou código morto com armadilha depois dos merges?

- **Referência**: `docs/auditoria-qualidade.md` §E8; `include/rdna4/device.h:76-87`;
  `include/rdna4/attn.cuh:110-111`; `include/rdna4/kv.h:288-295`.
- **Hipótese a falsificar**: o código morto apontado pela auditoria foi resolvido ou é inofensivo.
- **Comando**: `grep -rn "required_bytes\|kv_cache_bytes" src include tests scripts`;
  `grep -n kbase include/rdna4/attn.cuh`; leitura de `kv_fill_kernel`.
- **Resultado**: três itens:
  (a) **`required_bytes()` (as duas sobrecargas, `device.h:76-87`) tem 0 chamadores** — e a
  sobrecarga de um tipo só ainda contém a fórmula **pré-M4** (`ctx * 32768 * bpe(tipo)`), isto é, o
  bug que o M4 corrigiu continua no arquivo, num nome mais natural de chamar que
  `kv_cache_bytes(ctx, k, v)`. `main.hip:918` e `serve.hip:385` usam a função nova.
  (b) `attn.cuh:110-111`: `const char *kbase = (const char *)k + ((std::int64_t)kvh * head_dim) * 0;`
  — morto, e o comentário diz "set per key below", o que nunca acontece.
  (c) `kv.h:288-295`: `amx` é calculado e descartado (`(void)amx`), e — mais sério —
  no ramo `Q4_0` do `kv_fill_kernel` **16 lanes escrevem o `b->d` do mesmo bloco com valores
  diferentes** (cada lane usa o próprio par de elementos para `signed_max`), enquanto os `qs` são
  quantizados com o `id` do próprio lane: o bloco resultante é internamente inconsistente e o
  vencedor da corrida de escrita é indefinido. O comentário (`kv.h:239-241`) promete um padrão
  pseudo-aleatório **determinístico**.
- **Veredito**: **achado** (review F7/F8/F13). (c) é o único com efeito em teste: os dois gates que
  usam `--fill-cache` comparam dois caminhos sobre a *mesma* cache, então o erro cancela — mas
  "determinístico" é falso e um gate futuro que compare *valores* a contexto longo vai falhar de
  forma inexplicável.

## 18. Falsos positivos que eu mesmo descartei (registro do que **não** é achado)

- `attn_split_kernel` com split vazio: o caso `cmax == -INFINITY` (`attn.cuh:373-381`) e o
  `0 * expf(-inf - mm)` do merge estão corretos; não há NaN.
- `attn_merge_kernel` com todos os splits vazios: early return com zeros (`:420-423`) ✅.
- `release()` e idempotência (M1): todos os 18 ponteiros de `ptrs[]` **e** os 16 de `batch_ptrs[]`
  são zerados depois do `hipFree` (`graph.cuh:1341-1378`) ✅ — chamar duas vezes é seguro.
- `check_regression.sh` "gate vazio": a checagem `grep -q rel-L2` é fraca, mas o **binário** é
  forte (`want.size() != got.size()` → FAIL, `kMinCases`, nome/prompt/gen/tamanho de subamostra por
  caso, ids bit-exatos, rel-L2 e norma) — não é um gate vazio.
- `hipMemcpy`/`hipMemset`/`hipMemGetInfo` no motor: **todos** os sítios checam (ou devolvem
  `== hipSuccess`); os únicos sem teste são `(void)hipFree` em caminho de liberação (H6, cosmético).
- `serve`/`run`/`bench`/`ppl`/`info`: o teto de `--ctx-size` está nos cinco e **antes** da checagem
  de dispositivo (`main.hip:258,598,830,1043`, `:1096`), então é testável sem GPU ✅.
- `ev[p.open_ev]` etc. em `phase_prof.cuh`: sem acesso fora de faixa (o `pending` inicial é -1 e o
  primeiro `mark` não empilha par).

---

## Resumo da janela

| # | o que tentei falsificar | veredito |
|---|---|---|
| 1 | `-Wall -Wextra` ligado e árvore limpa | MANTIDO (README desatualizado) |
| 2 | `RD_PHASE` inerte | MANTIDO |
| 3 | base de tempo dos eventos | MANTIDO |
| 4 | `tuning.h` como fonte única | **FALSIFICADO** (literal 16 em `attn.cuh`) |
| 5 | `check-tuning` como gate da fiação | MANTIDO com ressalva (só cobre a tabela) |
| 6 | `check_all.sh` sem gate fantasma | **FALSIFICADO** (PASS com 9 gates) |
| 7 | prova de forja do loader | MANTIDO (reproduzida por `git archive`) |
| 8 | cauda do split-KV (M2) | MANTIDO |
| 9 | E2 (WPB do kernel sem split) | **FALSIFICADO** (continua aberto) |
| 10 | `kAttnSplitWpbWide` em todo contexto | garantia MANTIDA, evidência **em falta** |
| 11 | `q8_0` > `q4_0` a 64K | **FALSIFICADO** como afirmação de velocidade |
| 12 | 12 números do README | 7 ok, 5 com problema |
| 13 | fechar 10 com PPL na GPU | **ABANDONADO** (fila da GPU) |
| 14 | `kv_row_bytes` falha alto | **FALSIFICADO** (0 silencioso) |
| 15 | bytes/token do `bench` | **FALSIFICADO** (8,2% otimista) |
| 16 | KV do MTP no orçamento | **FALSIFICADO** (fora) |
| 17 | código morto inofensivo | **FALSIFICADO** (3 itens, um com corrida de escrita) |
| 18 | 7 falsos positivos descartados | — |
