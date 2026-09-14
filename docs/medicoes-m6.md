# rdna4-infer — M6 measurement: does batched prefill pay off?

M5 left one large gap: prefill runs at 28.8 tok/s (per-token path) against
llama.cpp's **440 tok/s** batched on the same machine and model. M6 was planned as
"share one weight pass across N tokens". This is the measurement that decides how
much that is actually worth, taken *before* restructuring the graph for it.

## What was built (and kept)

`include/rdna4/matvec.cuh`: `matvec_kernel_batch<T, ROWS, WPR, ILP, N>` plus
`matvec_launch_batch()` (N ∈ {2,4,8,16}, same per-type shape/ILP as the GEMV path).
For a given token the k-walk, ILP slots, per-slot sum and both reduction stages are
the same operations in the same order as `matvec_kernel_gen`, so each output element
is **bit-identical to N separate GEMV calls** — verified, not assumed:

```
tests/check_matmul_gpu.hip:  check-matmul-gpu: OK (bit-exact in every configuration tested)
```

Two methodology fixes were needed to trust the numbers, both worth recording:

- **The working set must exceed L2.** Timing one 20-30 MB tensor repeatedly keeps it
  in the 64 MB L2 and measures L2 bandwidth; the real model streams 12 GB from DRAM.
  The test now loads up to 16 tensors of the same shape (140-874 MB) per case.
  This also **corrects the M2 bandwidth table**: those per-type GB/s (iq3_s 557-753)
  were L2 numbers. DRAM-realistic N=1 values are 143-552 GB/s.
- **The warm-up must sync.** A `while (wall < 300 ms) { fn(); }` loop without a sync
  lets the command queue absorb thousands of iterations (the wall clock then measures
  launch throughput); the first M6 run queued ~20 minutes of GPU work.

## The scaling curve (IQ3_S, DRAM-realistic working sets)

Speedup of one batched call vs N separate GEMV calls, per quant type:

> **Correção da coluna "share of model" (tarefa 4 do lote paralelo).** A coluna foi
> calculada sobre `loader.total_bytes()` — o arquivo inteiro — em vez do tráfego real por
> token, que exclui `token_embd.weight` (lê-se uma linha, não a matriz) e o bloco MTP
> `blk.64.*`, que a v1 não executa. O efeito é grande onde o tensor excluído domina:
> `q3_k` é **3,6 %** do tráfego por token, não 7,9 %, e `q6_k` é **0,04 %**, não 2,9 %;
> os tipos `iq*` estavam subestimados. `docs/quants-inventario.md` §3.3 reproduz, por
> aritmética, exatamente esta coluna a partir de `total_bytes()` (as 14 linhas batem com
> 1 casa decimal) e traz a tabela corrigida. As *velocidades* desta seção não mudam — o
> que mudava era a leitura de "onde 1 byte de peso dói mais".

| type | share of model | N=1 gemv tok/s | N=1 GB/s | N=2 | N=4 | N=8 | N=16 | best |
|---|---|---|---|---|---|---|---|---|
| iq3_s | 30.9 % | 729 | 416 | 2.05× | 3.21× | 3.53× | **4.35×** | 4.35× |
| iq4_xs | 21.6 % | 2430 | 416 | 1.72× | 2.41× | **2.62×** | 2.56× | 2.62× |
| iq3_xxs | 15.5 % | 992 | 315 | 2.41× | 3.90× | 4.31× | **5.14×** | 5.14× |
| q5_k | 8.2 % | 678 | 552 | 2.02× | 3.33× | **3.50×** | 2.95× | 3.50× |
| q3_k | 7.9 % | 282 | 143 | 2.78× | 4.04× | 3.16× | **5.45×** | 5.45× |
| iq2_s | 4.8 % | 960 | 358 | 2.56× | 3.36× | 3.45× | **4.13×** | 4.13× |
| q6_k | 2.9 % | 90516 (L2) | 363 | 1.68× | 2.64× | 2.84× | **4.31×** | 4.31× |
| iq2_xs | 2.4 % | 1030 | 247 | 2.80× | 3.65× | 4.14× | **4.42×** | 4.42× |
| iq2_xxs | 2.2 % | 1429 | 275 | 2.46× | 3.28× | 4.12× | **4.86×** | 4.86× |
| q4_k | 1.9 % | 3355 | 442 | 2.18× | 3.05× | **3.34×** | 2.72× | 3.34× |
| q2_k | 1.0 % | 6335 | 487 | 1.24× | 1.61× | 1.26× | **1.80×** | 1.80× |
| q8_0 | 0.3 % | 13110 | 51 | 2.03× | **3.15×** | 1.23× | 1.38× | 3.15× |
| iq1_s | 0.3 % | 9831 | 319 | 1.48× | 2.02× | 2.38× | **2.61×** | 2.61× |
| iq4_nl | 0.0 % | 45332 | 124 | 4.21× | 3.77× | **13.51×** | 9.84× | 13.51× |

So weight sharing is worth **~2-5×** per type (not N×). Some types *regress* at N=16
(q4_k 3.34 → 2.72, q8_0 3.15 → 1.38): register pressure. The reason the win is
sub-linear is that these `vec_dot` bodies are **issue-bound** — the ALU work per
weight byte grows with N, so the per-token cost stops falling once the SIMD/issue
units saturate. This is exactly what M2 saw when it measured the IQ3_S body: 452
instructions per vec_dot, 55-62 % of them byte-emulation.

## What that projects to for the model

Applying each type's measured N=1 bandwidth and best-N speedup to the model's actual
byte composition (11.2 GiB per token, measured tensor inventory):

| | per token | implied rate |
|---|---|---|
| matvec, GEMV (today) | 35.9 ms | 27.9 tok/s |
| matvec, batched | 8.97 ms | — |
| + per-token scaffolding (norms, attention, GDN, quantization: 6.9 ms measured in M5) | 15.9 ms | **~63 tok/s** |

Two conclusions:

1. **Batching the matvec is worth ~2.2× prefill** (28.8 → ~63 tok/s), not the 16× the
   gap to llama.cpp suggests. That is a real win (a 512-token prompt: 17.8 s → 8 s)
   but it requires the layer-major restructure.
2. **The per-token scaffolding alone caps prefill at ~145 tok/s** even with an
   infinitely fast matvec. Reaching llama.cpp's 440 tok/s therefore requires batching
   *everything* (norms, attention, the GDN recurrence, activation quantization) **and**
   a fundamentally cheaper inner loop (MMQ-style tiled kernel with shared-memory
   weight staging and wide int8 dots — the route M2 deliberately did not take, since
   it trades bit-exactness for speed).

## Status of the M6 plan as written

- Step 1 (batched kernel + bit-exact test) — **done**, and it answered the feasibility
  question before any graph work: `matvec_kernel_batch` is correct and bit-exact, so
  the graph can switch to it without revalidating the arithmetic.
- The "≥ 150 tok/s" target is **not reachable** by weight sharing alone (projection
  ~63 tok/s, structural ceiling ~145 tok/s). The restructure that was planned for it
  is a large change; the decision of whether to spend it belongs to the user, with
  these numbers in hand.
