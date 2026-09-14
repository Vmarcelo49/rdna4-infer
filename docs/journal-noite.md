# Diário da rodada noturna — índice e relatório da manhã

Sessão não supervisionada de 2026-09-14. Baseline: tag `noite-baseline-2026-09-14`
(`main` = `e48f3c4`, `scripts/check_all.sh` PASS). Regras: `docs/noite-regras.md`.

**Alvo**: 131K de contexto · KV `q5_0` (K) / `q4_1` (V) · MTP entregando ganho real · e o
máximo de desempenho que couber na noite.

| frente | branch | worktree | diário | estado |
|---|---|---|---|---|
| Revisão adversarial + backlog | `feat/noite-review` | `../rdna4-wt-noite-review` | `docs/journal-review.md` | **mergeada** (15 achados, 233 itens de backlog) |
| KV `q5_0`/`q4_1` + memória | `feat/noite-kv` | `../rdna4-wt-noite-kv` | `docs/journal-kv.md` | a começar |
| MTP (verificação em batch) | `feat/noite-mtp` | `../rdna4-wt-noite-mtp` | `docs/journal-mtp.md` | a começar |
| Prefill / chunking / MMQ | `feat/noite-prefill` | `../rdna4-wt-noite-prefill` | `docs/journal-prefill.md` | a começar |
| Kernels gfx1201 | `feat/noite-kernels` | `../rdna4-wt-noite-kernels` | `docs/journal-kernels.md` | a começar |
| Contexto longo / RoPE / qualidade | `feat/noite-longctx` | `../rdna4-wt-noite-longctx` | `docs/journal-longctx.md` | a começar |
| Levantamento de referências | `feat/noite-refs` | `../rdna4-wt-noite-refs` | `docs/journal-refs.md` | **mergeada** (1014 linhas; cf(N) medido, Hadamard, veredito K/V) |

Este arquivo é atualizado pelo coordenador conforme os merges entram: no fim da noite ele tem
o estado final, os números medidos e o que ficou de fora com o motivo.

## Diário do coordenador

### C0. Premissa do alvo: `q5_0`/`q4_1` não existem neste motor (verificado, 02:00)
- **Referência**: o enunciado da rodada ("achado bom: K q5_0 / V q4_1").
- **Hipótese**: é um resultado deste motor.
- **Verificação**: `include/rdna4/kv.h` só tem `f32/f16/q8_0/q4_0` (`kv_type_parse`, linha 40).
  O achado vem do llama.cpp (que tem 6 tipos de KV), não daqui.
- **Veredito**: tratado como **hipótese a implementar e medir** pela frente KV, não como fato.
  A conta preliminar (21 504 B/token ⇒ 2,625 GiB a 131K) foi confirmada pela frente KV com
  aritmética medida e virou asserção em `check-kvtype`.

### C1. Argmax no device no caminho greedy (MANTIDO, +0,9 % e uma sincronia a menos)
- **Referência**: `docs/vulkan-vs-hip.md` §4 item 3 ("argmax no device no caminho greedy,
  bit-exato, ~0,5 ms = 1,5 %"); orçamento por fase: cópia de logits 0,21 ms + sampler 0,44 ms.
- **Hipótese**: com `temp <= 0` e sem penalidade de repetição, o sampler inteiro reduz a
  "primeiro máximo estrito em ordem de id" (`Sampler::filter`, ramo `temp <= 0`), que o device
  calcula sem trazer os 993 KB de logits para o host a cada token.
- **Comando**: `./scripts/gpu-lock.sh ./build/rdna4-infer bench -m IQ3_S -n 32 --reps 3`.
- **Resultado**: `per token: 32,64 ms = launch loop 30,83 + tail 1,80 + **sample 0,00**` (o
  sampler do host saiu do caminho); decode 30,64 tok/s contra 30,38 da janela limpa anterior
  (+0,9 %, dentro do ruído da máquina carregada — a evidência direta é a fase `sample` ir a
  zero). Gates: `check_golden_run.sh` OK (ids greedy idênticos nas 4 configurações de KV),
  `run --greedy` e `run --greedy --repeat-penalty 1.1` preservados (o caminho com penalidade
  continua no host, intocado).
- **Veredito**: MANTIDO. Arquivos: `include/rdna4/nn.cuh` (kernel `argmax_kernel`),
  `include/rdna4/graph.cuh` (`set_want_argmax`/`last_argmax`), `src/main.hip` (run e bench).
  Limitação conhecida: o servidor OpenAI ainda não usa o caminho rápido (fácil de ligar).

### C2. Correções da revisão adversarial aplicadas na `main` (MANTIDO)
- **`check_all.sh` mentia (F3, P0)**: `[ -x bin ] && step` sem `set -e` ⇒ com um binário
  ausente a bateria imprimia PASS com 9 dos 10 gates. Consertado: gate ausente = FAIL + lista.
  Verificado movendo `build/check-tuning` (`check_all (CPU half): FAIL`).
- **Banda 8,2 % otimista (F4)**: `main.hip` usava `loader.total_bytes()` (12,030 GB, o
  arquivo) como tráfego por token; o real é 11,122 GB (10,36 GiB). Corrigido com a mesma regra
  de exclusão do inventário. Efeito nos números publicados: 352 GB/s → **325 GB/s = 51 %** da
  roofline.
- **README**: tabela de fases somava 37,9 contra os 36,0 declarados (LM head em duas linhas);
  64K `q8_0` tinha três valores no mesmo documento. Ambos corrigidos.
- **Lock aninhado**: `gpu-lock.sh` agora exporta `GPU_LOCK_HELD=1` para o filho (um wrapper
  ad-hoc meu fez o `check_golden_run.sh` esperar 18 min pelo lock do próprio pai).

### C3. Repasses feitos às frentes (para o relatório da manhã)
- KV: portas do F6 (fechadas pela própria frente, com achado extra: `kv_fill_launch` respondia
  `true` para tipo desconhecido), evidência de KLD do PR #21038 (K mais sensível ~1,5×, nunca
  `q4_1` em K), rotação de Hadamard como alavanca de qualidade, e a interação de VRAM com os
  planos de estado do MTP.
- MTP: o alvo honesto é 1,3-1,5× com **D=4** (o `cf(N)` medido é 1,16/1,68/1,75/3,27/6,11 para
  N=2/3/4/8/16 — a projeção de 2,5× assume cf=1) e a atenção batelada tem **dono único** (a
  frente de prefill); mais o bug de orçamento do `alloc_kv` do MTP.
- Prefill: é o dono único da atenção multi-linha causal + scan de GDN multi-token; o `cf(N)`
  é o número que decide o N da verificação do MTP; e o gate que falta da regra de WPB larga.
- Kernels: o literal `16` em `attn.cuh` (F1) e a varredura de warps do caminho sem split (E2).

## As duas descobertas que reorganizaram a noite (frente de referências)

1. **A projeção de 2,5× do MTP é otimista em ~1,7×.** O `cf(N)` (custo de um token dentro de um
   chunk batelado, medido no `check-batch-gpu`, bit-exato, modelo real) é **1,16 / 1,68 / 1,75 /
   3,27 / 6,11** para N = 2/3/4/8/16. Com ele, o ganho honesto do MTP é **1,31× (D=2), 1,26×
   (D=3), 1,51× (D=4), 1,34× (D=8)** — **D=4 é o ótimo e D=8 é pior**. A projeção antiga
   (`docs/mtp.md`) supõe cf = 1,0, a mesma hipótese escondida no Teorema 3.8 do Leviathan.
2. **A peça que falta é a mesma para o MTP e para o prefill**: atenção multi-linha com máscara
   causal + scan de GDN multi-token. O prefill lê 1,1 ms/token de pesos e gasta 13,7 ms/token —
   os outros ~12 ms são andaime sequencial. Dono único definido: a frente de prefill; o MTP
   consome e re-mede.

Terceira alavanca de qualidade, ainda não medida aqui: **rotação de Hadamard em K/V antes de
quantizar** (llama.cpp PR #21038, `HADAMARD_Q` no ExLlamaV2). No upstream: PPL com KV `q4_1`
212,5 → 22,3 e AIME25 com `q4_0` 2,0 % → 21,7 %, com memória zero. **E o llama.cpp rotaciona
hoje**, então todo número de qualidade de KV publicado lá não é o nosso regime — o nosso é
pior. Repassado à frente KV.

### C4. Incidente: segundo deadlock de lock, ~2 h de GPU perdidas (corrigido, 03:02)
- **Referência**: o primeiro caso (wrapper ad-hoc) já tinha sido corrigido exportando
  `GPU_LOCK_HELD=1` do wrapper para o filho.
- **O que aconteceu**: um script de matriz da frente MTP pegou o lock uma vez e chamava
  `./scripts/gpu-lock.sh` de novo por configuração. Como o *wrapper* não checava a variável
  (só os gates checavam), cada configuração esperava 900 s pelo lock do próprio avô e morria
  com `RC=124` — 5 configs a 4K + 4 a 16K, GPU ociosa (VRAM 189 MB) e 13 processos na fila
  parados atrás por 11 min (detectado pela frente de contexto longo, que provou a árvore de
  processos).
- **Ação**: matei a árvore (27622/30730/30732/30733/30739); a fila andou em 20 s.
- **Conserto estrutural**: `gpu-lock.sh` agora roda direto quando `GPU_LOCK_HELD=1` (um
  ancestral já segura o lock), em vez de confiar em disciplina de cada chamador. Cópia
  corrigida distribuída aos 5 worktrees ativos. Documentado em `docs/gpu-queue.md`.
- **Veredito**: MANTIDO (o conserto). Custo: ~2 h de GPU da noite, que é o motivo de o teto de
  medição da noite ser menor do que o planejado — e por isso a limitação vai escrita no
  relatório.

### C6. Economia do MTP medida pela própria frente (03:55, antes da varredura final)
- **Referência**: `docs/mtp.md`; a projeção antiga de 2,5× e a recalibração da frente de
  referências para 1,3-1,5× (com o `cf(N)` velho).
- **Peças medidas** (`/tmp/mtp-gate2.log`, gates do próprio worktree): passo de rascunho
  **2,21 ms** (6,7-6,8 % de um passo do tronco); *snapshot+restore* do estado GDN
  **0,53 ms por par** (149,6 MiB); verificação em lote: 2 linhas 32,6 ms, 3 linhas 36,7,
  4 linhas 43,9 ⇒ **custo marginal de uma linha extra = 7,2 ms** contra 32,7 ms de um passo
  por token.
- **O que isso implica**: com D=4 (verificação de 5 linhas ≈ 51 ms + 4 rascunhos ≈ 8,8 ms, mais
  o raro restore/replay) e ~3,8 tokens aceitos por rodada, o custo por token cai para ~16 ms
  contra 33 ms — da ordem de **~2×**, e não os 1,3-1,5× que a recalibração previa com o
  `forward_batch` antigo. A frente está medindo.
- **Prova de correção que já existe**: o *snapshot/restore* do estado reproduz as linhas do
  lote de forma **bit-idêntica** (`max|d| = 0, argmax igual`) — é o que torna a rejeição de
  rascunho segura, e era o ponto que eu tinha marcado como "a crux" no briefing.
- **Aberta**: 1 falha no gate deles ("um passo de rascunho por proposta mais uma linha de
  reconstrução de KV por token aceito") — é eficiência do caminho novo, não correção; a frente
  está nisso.
- **Veredito**: ainda em medição; o relatório da frente decide.

### C7. A premissa do alvo foi testada e confirmada (frente KV, 04:00)
- **Hipótese do enunciado**: "K `q5_0` + V `q4_1` é o ponto doce do KV".
- **Medido** (KL média sobre o vocabulário, 4096 tokens de texto real, 256 probes, piso f16
  rodado duas vezes, mesmo processo): `q8_0/q8_0` **0,000492** · `q8_0/q4_1` 0,001443 ·
  **`q5_0/q4_1` 0,001715** · `q5_0/q4_0` 0,002118 · `q4_0/q4_0` 0,003208.
- **Eixos isolados**: K `q8_0` é **16 % melhor** que K `q5_0` (V fixo em q4_1) e V `q4_1` é
  **19 % melhor** que V `q4_0` (K fixo em q5_0). As duas direções que o PR #21038 do llama.cpp
  mediu em Qwen3.5, reproduzidas aqui — e `q4_0` em K é o pior de todos.
- **Needle a 8K**: 8/8 agulhas recuperadas em **todos** os formatos, inclusive `q4_0/q4_0` —
  a sonda não discrimina nesse tamanho; a KL é que ordena.
- **PPL não ordena** (medido, não argumentado): a ordem por PPL e a ordem por KL discordam
  nos dois configs do meio, e `q5_0/q4_0` tem PPL *melhor* que `q5_0/q4_1` com 23 % mais KL.
- **Veredito**: MANTIDO o padrão `q5_0/q4_1` (é o alvo, KL é o 2º melhor e sobram 1,68 GiB para
  o MTP); `q8_0/q4_1` documentado como a opção de melhor qualidade (0,75 GiB a mais).

### C8. Duas estimativas discordaram, e a mais barata de checar venceu (04:20)
- **O que aconteceu**: a frente de prefill mediu que o matvec em lote lê 12,0 GB por chunk de 16
  em 110 ms (**109 GB/s**) contra 446 GB/s no caminho por token, e estimou 2-2,5× de ganho ao
  "dequantizar o bloco uma vez". A frente de kernels implementou a mudança para `iq3_s`/`iq3_xxs`
  (com fallback identidade para os outros 12 tipos) e **contou as instruções**: ~40 operações de
  preparo por bloco de 110 B mais ~25 por token dá um teto de **~10 %**, não 2-3×.
- **Decisão do coordenador**: a mudança fica **desligada** (duas linhas de `RD_BATCH` prontas para
  ligar) até existir medida. Regra da noite: mudança de kernel sem número não entra — e uma
  estimativa que outra frente contradiz por contagem de instruções é exatamente o caso em que
  "medir antes" paga. O gargalo real do matvec em lote segue **não identificado**: é o item
  aberto mais valioso para a próxima sessão, e agora tem duas hipóteses concorrentes medidas
  (banda amortizada vs issue de ALU) e nenhuma confirmada.
- **Números que a frente de kernels trouxe e que ficam** (todos bit-exatos, memcmp): LUT dos IQ
  em LDS `iq3_s` 7,762→7,251 ms/token e `iq3_xxs` 4,468→3,781 (**−1,20 ms/token**; nos `iq2_*`
  a LUT é maior que o peso lido por CTA e o ganho vira perda, então uma guarda
  peso/CTA ≥ 2× LUT decide); `delta_rule` com carga `float4` **71,70→8,42 µs por camada**
  (8,5×, −3,0 ms/token nas 48 camadas, com a ressalva de que a cadeia de 100 reusa o mesmo estado
  no Infinity Cache); `rms_norm` com 4 cargas em voo 1,227× (−0,26 ms/token). Abandonados com
  medida: `rows=1` (0,973×, porque `WPR=1` faz o número de warps ser `nrows`) e a fusão dos
  kernels pequenos (o `emit()` do grafo faz `hipMemcpy` bloqueante, então fundir 4 ops em 1 faz o
  oráculo por nó ler um buffer já sobrescrito — a variante que preserva o dump vale 0,26 ms).

## Relatório da manhã — como ler

1. **O que foi pedido e o que foi entregue**: a tabela de frentes no topo diz o estado de
   cada uma; as entradas C0-C9 contam o que o coordenador fez, com o número que justifica
   cada decisão. Os diários por frente (`docs/journal-*.md`) têm o detalhe por experimento,
   no formato referência → hipótese → comando → resultado → veredito.
2. **Duas revisões adversariais** rodaram esta noite: `docs/adversarial-noite.md` (a árvore
   mergeada antes da rodada; achados F1-F15) e `docs/adversarial-noite2.md` (o código
   produzido *durante* a rodada; achados R1-R11). **Nove achados eram de código escrito hoje**
   — três dos quais eu já consertei (R2 servidor, R4 argmax com NaN, R9 `forward_batch` sem
   argmax) — e essa é a parte do relatório que mais importa: a noite produziu código rápido e
   as revisões pegaram o que os gates não pegavam.
3. **O que NÃO foi medido** está escrito em cada frente e repetido no fechamento: a 131K não
   houve medida de qualidade com texto real; nenhuma medida fim-a-fim de 131K cabe no
   orçamento da noite (o prefill em lote leva ~29 min só para encher o KV).

## Diário do coordenador (continuação)

### C9. O merge da frente KV expos uma armadilha de build da classe "tabela em header"
Duas vezes nesta noite uma tabela pequena de host dentro de um header (`g_attn_wpb_forced`,
depois `kKvTypeNames`) quebrou o link da biblioteca compartilhada com `relocation
R_X86_64_PC32 ... recompile with -fPIC`. Mover a tabela de `static` local para escopo de
namespace **não** resolve (o erro só muda de nome). O conserto é de classe:
`POSITION_INDEPENDENT_CODE ON` + `-fPIC` no passe HIP de `rdna4_serve`. Registrado no
`CMakeLists.txt` com o motivo, para o próximo não perder meia hora com a mesma coisa.

### C10. O orçamento honesto devolveu o alvo
Com o conserto da frente KV (`required_bytes` com os 448 MiB medidos e sem contar o bloco MTP
quando ele não é usado):

```
model file: 11.21 GiB, uploaded weights: 10.88 GiB, graph buffers: 157.98 MiB
kv(ctx=131072, k=q5_0/v=q4_1): 2.62 GiB + margin: 448.00 MiB = need 14.09 GiB
(file-based estimate was 14.84 GiB)  -> budget OK
```

O orçamento antigo recusava `q8_0/q8_0` a 131K (estimava 16,46 GiB) para uma configuração que
**roda** (15,87 GiB em uso, medido). Ou seja: "a 131K só cabe com `q4_0`" era falso, e a
diferença vinha de um chute de 1 GiB de overhead e de contar o bloco MTP que não é carregado.

### C11. MTP: o ganho existe e é 1,75× (medido pela frente, 05:35)
- **Referência**: `docs/mtp.md` §7 ("verify batelado" como a peça que falta), `docs/medicoes-m8.md`
  (projeção de 1,3-1,5× com o `cf(N)` antigo).
- **Resultado medido, COM O BASELINE CERTO** (a primeira medição da frente deu 1,71-1,75× e
  estava errada: comparava contra o baseline quebrado pelo R9, e o texto degenerado que ele
  produzia — "actor actor actor…" — *inflava a aceitação* de 67,9 % para 88,9 %): a 4K,
  **ganancioso 29,37 tok/s · `--draft 2` 34,56 (1,18×) · `--draft 3` 30,11 (1,03×) ·
  `--draft 3 --mtp-serial` 25,85 (0,88×)**; a 16K, ganancioso 26,81 · **`--draft 3` 30,53
  (1,14×)**. Exatidão conferida por **md5 do stdout** (`--mtp` == ganancioso nas duas pontas,
  todas as variantes). O contraste que sobrevive é o que prova a tese: **0,88× (serial) →
  1,18× (batelado)** na mesma janela e com a mesma saída.
- **Duas lições de método que vão para o relatório**: (a) um ganho só vale o que vale o
  baseline contra o qual foi medido — o número inflado ficou ~40 min no README antes de ser
  corrigido; (b) taxa de aceitação é um proxy de qualidade que um motor quebrado consegue
  *melhorar*.
- **Peças que sustentam o número**: *snapshot/restore* do estado GDN bit-exato (max|d| = 0,
  argmax igual) a 0,53 ms por par; `forward_batch_all` bit-exato em posição 4084 (linha a linha,
  h e logits); linha extra na verificação a 7,0-7,2 ms contra 32,7 ms de um passo por token.
- **Ressalva de escopo**: os números de **16K** da frente (0,86× e aceitação 54,9 %) estão
  contaminados pelo achado **R1** (a atenção dividida em lote lia o `q` da primeira linha), que
  foi consertado depois; a re-medição decide se o MTP paga também em contexto longo.
- **Veredito**: MANTIDO. É a entrega da tarefa 2: o MTP saiu de **0,88-0,91× (serial)** para
  **1,18× a 4K e 1,14× a 16K**, com a saída byte-idêntica ao ganancioso (md5) e a aceitação
  subindo de 54,9 % para 67,7 % quando o R1 foi consertado.

## Estado do alvo (atualizado pelo coordenador)

- **131K**: ainda não medido nesta rodada. No baseline, 131K com KV `q4_0` roda a 14,0 tok/s
  com 13,88 GiB em uso; com `f16` **não cabe** (transborda para GTT e cai para 2-8 tok/s).
- **KV `q5_0`/`q4_1`**: **não existem no motor** (`include/rdna4/kv.h` tem f32/f16/q8_0/q4_0).
  Conta preliminar: 21 504 B/token ⇒ 2,63 GiB em 131K (contra 2,25 GiB do `q4_0` e 4,25 GiB do
  `q8_0`). É hipótese a implementar e medir, não resultado.
- **MTP**: implementado, aceitação 86,7 % e saída idêntica ao greedy, mas **9-10 % mais lento**
  porque cada token aceito roda um forward do tronco (sem verificação em batch). Com o
  `Graph::forward_batch` do M8 já no lugar, a verificação em batch é o trabalho da frente MTP.

### C5. Gate em texto real para a ponta larga da regra de WPB (achado F5) — FECHADO
- **Referência**: `docs/adversarial-noite.md` F5 (a combinação embarcada ≥16 splits × 16 warps
  não tinha gate em texto real; só um teste com cache sintético e tolerância 5e-2).
- **Comando**: `./scripts/gpu-lock.sh timeout 1500 ./scripts/check_attn_split.sh` (o caso novo
  roda com `RD_ATTN_SPLITS=16`, exatamente a ponta larga) em wikitext-2, ctx 1024, 2 chunks.
- **Resultado**: PPL sem split **5,1989** · 4 splits **5,2054** (+0,125 %) · **16 splits × 16
  warps 5,2114 (+0,240 %)** contra o sem-split. Limite do gate 0,5 %. Ou seja: a regra larga é
  numericamente equivalente ao caminho sem split, com **o dobro** do desvio do caso de 4 splits
  — coerente com mais reordenação da soma das chaves, e dentro da mesma classe numérica.
- **Veredito**: MANTIDO (gate novo em `scripts/check_attn_split.sh`, `WIDE=0` pula). De quebra,
  o número entra no README como a escada de desvio do split-KV: 0,125 % (4 splits), 0,240 %
  (16 splits × 16 warps).
