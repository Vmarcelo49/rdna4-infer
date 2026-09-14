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
