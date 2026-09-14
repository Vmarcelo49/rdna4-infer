# Diário da rodada noturna — índice e relatório da manhã

Sessão não supervisionada de 2026-09-14. Baseline: tag `noite-baseline-2026-09-14`
(`main` = `e48f3c4`, `scripts/check_all.sh` PASS). Regras: `docs/noite-regras.md`.

**Alvo**: 131K de contexto · KV `q5_0` (K) / `q4_1` (V) · MTP entregando ganho real · e o
máximo de desempenho que couber na noite.

| frente | branch | worktree | diário | estado |
|---|---|---|---|---|
| Revisão adversarial + backlog | `feat/noite-review` | `../rdna4-wt-noite-review` | `docs/journal-review.md` | a começar |
| KV `q5_0`/`q4_1` + memória | `feat/noite-kv` | `../rdna4-wt-noite-kv` | `docs/journal-kv.md` | a começar |
| MTP (verificação em batch) | `feat/noite-mtp` | `../rdna4-wt-noite-mtp` | `docs/journal-mtp.md` | a começar |
| Prefill / chunking / MMQ | `feat/noite-prefill` | `../rdna4-wt-noite-prefill` | `docs/journal-prefill.md` | a começar |
| Kernels gfx1201 | `feat/noite-kernels` | `../rdna4-wt-noite-kernels` | `docs/journal-kernels.md` | a começar |
| Contexto longo / RoPE / qualidade | `feat/noite-longctx` | `../rdna4-wt-noite-longctx` | `docs/journal-longctx.md` | a começar |
| Levantamento de referências | `feat/noite-refs` | `../rdna4-wt-noite-refs` | `docs/journal-refs.md` | a começar |

Este arquivo é atualizado pelo coordenador conforme os merges entram: no fim da noite ele tem
o estado final, os números medidos e o que ficou de fora com o motivo.

## Estado do alvo (atualizado pelo coordenador)

- **131K**: ainda não medido nesta rodada. No baseline, 131K com KV `q4_0` roda a 14,0 tok/s
  com 13,88 GiB em uso; com `f16` **não cabe** (transborda para GTT e cai para 2-8 tok/s).
- **KV `q5_0`/`q4_1`**: **não existem no motor** (`include/rdna4/kv.h` tem f32/f16/q8_0/q4_0).
  Conta preliminar: 21 504 B/token ⇒ 2,63 GiB em 131K (contra 2,25 GiB do `q4_0` e 4,25 GiB do
  `q8_0`). É hipótese a implementar e medir, não resultado.
- **MTP**: implementado, aceitação 86,7 % e saída idêntica ao greedy, mas **9-10 % mais lento**
  porque cada token aceito roda um forward do tronco (sem verificação em batch). Com o
  `Graph::forward_batch` do M8 já no lugar, a verificação em batch é o trabalho da frente MTP.
