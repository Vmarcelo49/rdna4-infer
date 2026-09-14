// Measurement tool (worktree rdna4-wt-medicoes-gpu) — where the milliseconds of
// one decode token actually go, measured with HIP events inside the real graph.
//
//   bench-phases-gpu <file.gguf> [--ctx N] [--kv-k T] [--kv-v T] [--pos N]
//                    [--tokens N] [--level 1|2] [--count-only] [--mem] [--splits N]
//                    [--no-clean]
//
// What it measures, and how
//   1. `--mem`      : DRAM read peak and L2/Infinity-Cache read peak of THIS card
//                     (the roofline denominator). Run before the model is loaded.
//   2. dispatch floor: a chain of empty kernels, timed with one event pair, and
//                     the same chain with an event between every kernel — the
//                     difference per kernel is the cost of one `hipEventRecord`,
//                     which is what the phase profiler adds per mark.
//   3. clean pass   : N tokens with the profiler off -> ms/token (the number the
//                     other benches report, reproduced inside this tool).
//   4. phase pass   : the same N tokens with Graph's opt-in phase profiler on
//                     (RD_PHASE marks in graph.cuh -> include/rdna4/phase_prof.cuh).
//                     The device timeline is partitioned into named buckets; the
//                     table prints ms/token, the share, the marks/token and the
//                     event overhead that belongs to that bucket, so the reader
//                     can subtract it (the corrected column is printed too).
//   5. --count-only : no events at all, only the mark counters -> the exact number
//                     of launches per phase bucket (e.g. matvec = one per proj()).
//   6. KV traffic    : K/V bytes read by the attention per token at this context,
//                     with the 6x GQA duplication of the real kernel, and the
//                     effective rate implied by the measured attention bucket.
//
// Every timing number here is taken after a warm-up that keeps the GPU busy for
// >= 300 ms with a sync inside the loop (the DPM trap: SCLK drops to 9-16 MHz
// between kernels and a cold measurement is 1.5-2x wrong).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rdna4/attn.cuh"
#include "rdna4/gdn.cuh"
#include "rdna4/nn.cuh"
#include "rdna4/graph.cuh"
#include "rdna4/loader.h"
#include "rdna4/model.h"
#include "rdna4/phase_prof.cuh"

namespace {

double now_ms() {
  return (double)std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
             .count() /
         1000.0;
}

const char *kUsage =
    "usage: bench-phases-gpu <file.gguf> [--ctx N] [--kv-k T] [--kv-v T] [--pos N]\n"
    "                        [--tokens N] [--level 1|2] [--count-only] [--mem] [--gdn]\n"
    "                        [--no-clean]\n";

// ---------------------------------------------------------------------------
// Memory roofline: grid-stride read of a buffer that is much larger than L2
// (8 MiB) and Infinity Cache (64 MB) -> DRAM; and of a 4 MiB buffer -> L2/IC.
// ---------------------------------------------------------------------------
// Four independent accumulators (and 4 in-flight float4 per iteration) so the
// loop is bandwidth-bound and not latency-bound: with one accumulator the L2
// measurement of a 4 MiB buffer comes out ~3x too low (measured).
__global__ void read_reduce_kernel(const float4 *__restrict__ p, std::int64_t n4,
                                   float *__restrict__ out) {
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  const std::int64_t tid = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const std::int64_t stride = (std::int64_t)gridDim.x * blockDim.x;
  std::int64_t i = tid;
  for (; i + 3 * stride < n4; i += 4 * stride) {
    const float4 v0 = p[i], v1 = p[i + stride], v2 = p[i + 2 * stride], v3 = p[i + 3 * stride];
    a0 += v0.x + v0.y + v0.z + v0.w;
    a1 += v1.x + v1.y + v1.z + v1.w;
    a2 += v2.x + v2.y + v2.z + v2.w;
    a3 += v3.x + v3.y + v3.z + v3.w;
  }
  for (; i < n4; i += stride) {
    const float4 v = p[i];
    a0 += v.x + v.y + v.z + v.w;
  }
  const float acc = a0 + a1 + a2 + a3;
  if (acc == 1234567.0f) out[blockIdx.x] = acc;  // never true: keeps the loads alive
}

__global__ void empty_kernel() {}

// Time `reps` repetitions of a lambda that queues GPU work, with a sync per rep
// so the host cannot run ahead; returns ms per rep. Warm-up first (>= 300 ms).
template <typename F>
double time_reps(int reps, double warm_ms, F &&f) {
  // Warm-up by GPU-busy TIME, not by iteration count: the DPM state of this card
  // only settles after ~300 ms of continuous work (SCLK drops to 9-16 MHz
  // between kernels and a cold measurement is 1.5-2x wrong).
  const double w0 = now_ms();
  double last = w0;
  while (last - w0 < warm_ms) {
    f();
    (void)hipDeviceSynchronize();
    last = now_ms();
  }
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    const double t0 = now_ms();
    f();
    (void)hipDeviceSynchronize();
    best = std::min(best, now_ms() - t0);
  }
  return best;
}

struct MemResult {
  double dram_gbs = 0, dram_ms = 0;
  double l2_gbs = 0, l2_ms = 0;
  double bytes_dram = 0, bytes_l2 = 0;
};

MemResult mem_bench() {
  MemResult r;
  const std::size_t dram_bytes = (std::size_t)3 * 1024 * 1024 * 1024;  // 3 GiB >> 64 MB IC
  const std::size_t ic_bytes = (std::size_t)32 * 1024 * 1024;          // 32 MiB: IC, not L2
  const std::size_t l2_bytes = (std::size_t)4 * 1024 * 1024;           // 4 MiB: L2
  float *out = nullptr;
  if (hipMalloc(&out, 4096 * sizeof(float)) != hipSuccess) return r;
  const int blocks = 4096, threads = 256;
  auto measure = [&](std::size_t bytes, int passes, int reps, double warm, double *ms_out) {
    void *buf = nullptr;
    if (hipMalloc(&buf, bytes) != hipSuccess) {
      std::printf("  (hipMalloc of %.0f MB failed)\n", (double)bytes / 1e6);
      return 0.0;
    }
    (void)hipMemset(buf, 0, bytes);
    const std::int64_t n4 = (std::int64_t)(bytes / sizeof(float4));
    auto pass = [&] {
      for (int k = 0; k < passes; ++k) {
        read_reduce_kernel<<<blocks, threads>>>((const float4 *)buf, n4, out);
      }
    };
    *ms_out = time_reps(reps, warm, pass);
    const double gbs = (double)bytes * (double)passes / (*ms_out / 1000.0) / 1e9;
    (void)hipFree(buf);
    return gbs;
  };
  // DRAM: 3 GiB per pass (warm-up = 2 passes at ~600 GB/s ~ 10 ms each, so the
  // loop below also does enough repetitions to keep the GPU busy > 300 ms).
  r.dram_gbs = measure(dram_bytes, 1, 12, 300.0, &r.dram_ms);
  r.bytes_dram = (double)dram_bytes;
  // Infinity Cache: 32 MiB re-read 16x (> 8 MiB L2, < 64 MB IC).
  r.l2_gbs = measure(ic_bytes, 16, 12, 300.0, &r.l2_ms);
  r.bytes_l2 = (double)ic_bytes * 16.0;
  (void)hipFree(out);
  return r;
}

// Dispatch floor + cost of one hipEventRecord. Both chains contain the same
// number of kernels, so the difference is exactly the event cost.
struct DispatchResult {
  double empty_us = 0;      // us per empty kernel (back-to-back, one event pair)
  double ev_kernel_us = 0;  // same chain with one event between every kernel
  double ev_us = 0;         // derived: cost of one hipEventRecord
};

DispatchResult dispatch_bench() {
  DispatchResult r;
  const int n = 2000, reps = 40;
  // (a) one pair of events around the whole chain
  hipEvent_t a{}, b{};
  (void)hipEventCreate(&a);
  (void)hipEventCreate(&b);
  auto chain_plain = [&] {
    for (int i = 0; i < n; ++i) empty_kernel<<<1, 32>>>();
  };
  {  // warm-up: 200 chains (each ~2000 x 2.3 us = 4.6 ms)
    for (int k = 0; k < 80; ++k) {
      chain_plain();
      (void)hipDeviceSynchronize();
    }
  }
  double best_ms = 1e30;
  for (int k = 0; k < reps; ++k) {
    (void)hipEventRecord(a, nullptr);
    chain_plain();
    (void)hipEventRecord(b, nullptr);
    (void)hipDeviceSynchronize();
    float ms = 0;
    (void)hipEventElapsedTime(&ms, a, b);
    best_ms = std::min(best_ms, (double)ms);
  }
  r.empty_us = best_ms * 1000.0 / n;
  // (b) an event between every kernel
  std::vector<hipEvent_t> ev((std::size_t)n + 1);
  for (auto &e : ev) (void)hipEventCreate(&e);
  auto chain_ev = [&] {
    for (int i = 0; i < n; ++i) {
      (void)hipEventRecord(ev[(std::size_t)i], nullptr);
      empty_kernel<<<1, 32>>>();
    }
    (void)hipEventRecord(ev[(std::size_t)n], nullptr);
  };
  for (int k = 0; k < 40; ++k) {
    chain_ev();
    (void)hipDeviceSynchronize();
  }
  best_ms = 1e30;
  for (int k = 0; k < reps; ++k) {
    (void)hipEventRecord(ev[0], nullptr);
    for (int i = 1; i < n; ++i) {
      empty_kernel<<<1, 32>>>();
      (void)hipEventRecord(ev[(std::size_t)i], nullptr);
    }
    empty_kernel<<<1, 32>>>();
    (void)hipEventRecord(ev[(std::size_t)n], nullptr);
    (void)hipDeviceSynchronize();
    float ms = 0;
    (void)hipEventElapsedTime(&ms, ev[0], ev[(std::size_t)n]);
    best_ms = std::min(best_ms, (double)ms);
  }
  r.ev_kernel_us = best_ms * 1000.0 / n;  // per (kernel + event) pair
  r.ev_us = r.ev_kernel_us - r.empty_us;
  for (auto &e : ev) (void)hipEventDestroy(e);
  (void)hipEventDestroy(a);
  (void)hipEventDestroy(b);
  return r;
}

// One argmax step on the host, exactly like the CLI's greedy path.
std::int32_t argmax_of(const std::vector<float> &v) {
  std::int32_t best = 0;
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) best = (std::int32_t)i;
  }
  return best;
}

struct RunResult {
  double ms_per_token = 0;  // best single token in the pass
  double median_ms = 0;
  double mean_ms = 0;
  std::size_t tokens = 0;
  int splits = 0;
  std::vector<std::int32_t> ids;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fputs(kUsage, stderr);
    return 2;
  }
  const char *model = argv[1];
  int ctx = 4096, tokens = 24, level = 1, pos = -1, splits = -1;
  bool count_only = false, do_mem = false, do_clean = true, do_gdn = false;
  rdna4::KvType kv_k = rdna4::KvType::F16, kv_v = rdna4::KvType::F16;
  for (int i = 2; i < argc; ++i) {
    const char *a = argv[i];
    auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : nullptr; };
    if (!std::strcmp(a, "--ctx")) { const char *v = next(); ctx = v ? std::atoi(v) : ctx; }
    else if (!std::strcmp(a, "--pos")) { const char *v = next(); pos = v ? std::atoi(v) : pos; }
    else if (!std::strcmp(a, "--tokens")) { const char *v = next(); tokens = v ? std::atoi(v) : tokens; }
    else if (!std::strcmp(a, "--level")) { const char *v = next(); level = v ? std::atoi(v) : level; }
    else if (!std::strcmp(a, "--splits")) { const char *v = next(); splits = v ? std::atoi(v) : splits; }
    else if (!std::strcmp(a, "--count-only")) count_only = true;
    else if (!std::strcmp(a, "--mem")) do_mem = true;
    else if (!std::strcmp(a, "--gdn")) do_gdn = true;
    else if (!std::strcmp(a, "--no-clean")) do_clean = false;
    else if (!std::strcmp(a, "--kv-k")) {
      const char *v = next();
      const char *e = v ? rdna4::kv_type_parse(v, &kv_k) : "missing value";
      if (e) { std::fprintf(stderr, "%s\n", e); return 2; }
    } else if (!std::strcmp(a, "--kv-v")) {
      const char *v = next();
      const char *e = v ? rdna4::kv_type_parse(v, &kv_v) : "missing value";
      if (e) { std::fprintf(stderr, "%s\n", e); return 2; }
    } else {
      std::fprintf(stderr, "unknown argument %s\n%s", a, kUsage);
      return 2;
    }
  }
  if (pos < 0) pos = 0;

  std::printf("== bench-phases-gpu (worktree medicoes-gpu)\n");
  std::printf("model: %s\nctx %d, kv %s/%s, decode positions %d..%d, tokens %d, level %d%s\n", model,
              ctx, rdna4::kv_type_name(kv_k), rdna4::kv_type_name(kv_v), pos, pos + tokens - 1,
              tokens, level, count_only ? ", count-only" : "");

  // ---------------------------------------------------------------- roofline
  if (do_mem) {
    std::printf("\n== memoria: pico de leitura desta placa\n");
    const MemResult m = mem_bench();
    if (m.dram_gbs > 0) {
      std::printf("  DRAM  read: %.1f MB in %.3f ms -> %.1f GB/s (3 GiB buffer, 5 reps, best)\n",
                  m.bytes_dram / 1e6, m.dram_ms, m.dram_gbs);
    }
    if (m.l2_gbs > 0) {
      std::printf("  Infinity Cache (32 MiB x16): %.1f MB in %.3f ms -> %.1f GB/s\n",
                  m.bytes_l2 / 1e6, m.l2_ms, m.l2_gbs);
    }
  }

  // ------------------------------------------------------------- dispatch/event
  const DispatchResult d = dispatch_bench();
  std::printf("\n== despacho de kernel e custo de evento (2000 kernels vazios x40)\n");
  std::printf("  kernel vazio enfileirado: %.3f us/kernel\n", d.empty_us);
  std::printf("  kernel vazio + 1 evento : %.3f us/(kernel+evento)\n", d.ev_kernel_us);
  std::printf("  => custo de um hipEventRecord: %.3f us\n", d.ev_us);

  // ------------------------------------------------- GDN isolado (fora do grafo)
  // O bucket "gdn_delta" do grafo mede 4,1 ms/token nas 48 camadas recorrentes
  // (~85 us por camada) para 3,1 MB de estado por camada. Este bloco mede o
  // MESMO kernel sozinho, com as formas reais, para confirmar que o numero nao
  // e artefato de medicao dentro do grafo.
  if (do_gdn) {
    const int nvh = 48, nkh = 16, S = 128;   // qwen3.8-27B: group_count 16, time_step_rank 48, state 128
    const int dim = nvh * S;
    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dg = nullptr, *db = nullptr, *dstate = nullptr,
         *dout = nullptr, *dconv = nullptr, *dcw = nullptr, *dcs = nullptr;
    const std::size_t fb = (std::size_t)dim * sizeof(float);
    const std::size_t sb = (std::size_t)nvh * S * S * sizeof(float);
    const int chan = 2 * nkh * S + dim;
    if (hipMalloc(&dq, fb) == hipSuccess && hipMalloc(&dk, fb) == hipSuccess &&
        hipMalloc(&dv, fb) == hipSuccess && hipMalloc(&dg, fb) == hipSuccess &&
        hipMalloc(&db, fb) == hipSuccess && hipMalloc(&dstate, sb) == hipSuccess &&
        hipMalloc(&dout, fb) == hipSuccess &&
        hipMalloc(&dconv, (std::size_t)chan * sizeof(float)) == hipSuccess &&
        hipMalloc(&dcw, (std::size_t)chan * 4 * sizeof(float)) == hipSuccess &&
        hipMalloc(&dcs, (std::size_t)3 * chan * sizeof(float)) == hipSuccess) {
      (void)hipMemset(dstate, 0, sb);
      rdna4::fill_random_kernel<<<(unsigned)((sb / 4 + 255) / 256), 256>>>(
          (float *)dstate, (std::int64_t)(sb / 4), 7u);
      std::printf("\n== GDN isolado (fora do grafo, formas reais: nvh %d, nkh %d, S %d)\n", nvh, nkh,
                  S);
      std::printf("  estado por camada: %.2f MB (lido+escrito 2x por camada)\n",
                  (double)sb / 1e6);
    } else {
      std::printf("\n--gdn: hipMalloc falhou\n");
    }
    // Micro-benchmarks dos kernels "pequenos", em CADEIA. Medir um lancamento
    // com hipDeviceSynchronize() depois mede o round-trip host->GPU (~21 us
    // nesta placa: um kernel de 48 elementos "custa" o mesmo que um de 6144),
    // entao a cadeia de N lanca o mesmo kernel N vezes e divide por N; o numero
    // ja inclui o despacho (2,5 us), que e o que o grafo tambem paga.
    {
      float *x = nullptr, *y = nullptr, *w = nullptr, *z = nullptr;
      rdna4::block_q8_1 *qb = nullptr;
      const int E = 5120;
      const int nq8 = E / 32;
      bool ok = hipMalloc(&x, (std::size_t)E * 4) == hipSuccess &&
                hipMalloc(&y, (std::size_t)E * 4) == hipSuccess &&
                hipMalloc(&w, (std::size_t)E * 4) == hipSuccess &&
                hipMalloc(&z, (std::size_t)E * 4) == hipSuccess &&
                hipMalloc(&qb, (std::size_t)nq8 * sizeof(rdna4::block_q8_1)) == hipSuccess;
      if (ok) {
        (void)hipMemset(x, 0, (std::size_t)E * 4);
        (void)hipMemset(w, 0, (std::size_t)E * 4);
        hipEvent_t ea{}, eb{};
        (void)hipEventCreate(&ea);
        (void)hipEventCreate(&eb);
        const int CH = 100;
        auto cadeia = [&](const char *nome, auto &&fn) {
          const double w0 = now_ms();
          while (now_ms() - w0 < 300.0) {
            for (int i = 0; i < CH; ++i) fn();
            (void)hipDeviceSynchronize();
          }
          double melhor = 1e30;
          for (int r = 0; r < 20; ++r) {
            (void)hipEventRecord(ea, nullptr);
            for (int i = 0; i < CH; ++i) fn();
            (void)hipEventRecord(eb, nullptr);
            (void)hipDeviceSynchronize();
            float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, ea, eb);
            melhor = std::min(melhor, (double)ms);
          }
          std::printf("  %-34s %8.2f us/lancamento (cadeia de %d)\n", nome, melhor * 1000.0 / CH,
                      CH);
        };
        cadeia("kernel vazio (piso de despacho)", [&] { empty_kernel<<<1, 32>>>(); });
        cadeia("rms_norm(1 x 5120)", [&] {
          (void)rdna4::rms_norm_launch(x, w, y, 1, E, 1e-6f);
        });
        cadeia("rms_norm(48 x 128)", [&] {
          (void)rdna4::rms_norm_launch(x, w, y, 48, 128, 1e-6f);
        });
        cadeia("l2_norm(16 x 128)", [&] {
          (void)rdna4::l2_norm_launch(x, y, 16, 128, 1e-6f);
        });
        cadeia("unary(48, sigmoid)", [&] {
          (void)rdna4::unary_launch(x, y, 48, rdna4::UnOp::Sigmoid);
        });
        cadeia("mul(48)", [&] { (void)rdna4::mul_launch(x, z, y, 48); });
        cadeia("add(48)", [&] { (void)rdna4::add_launch(x, z, y, 48); });
        cadeia("silu(6144)", [&] { (void)rdna4::unary_launch(x, y, 6144, rdna4::UnOp::Silu); });
        cadeia("mul(6144)", [&] { (void)rdna4::mul_launch(x, z, y, 6144); });
        cadeia("quantize_q8_1(5120)", [&] {
          rdna4::quantize_q8_1_kernel<<<(nq8 + 3) / 4, 128>>>(x, qb, nq8);
        });
        cadeia("rope(1 x 24 x 256, n_rot 64)", [&] {
          (void)rdna4::rope_launch(y, 1, 24, 256, 64, 1e7f, (const int *)z);
        });
        cadeia("deinterleave_q_gate(24 x 256)", [&] {
          (void)rdna4::deinterleave_q_gate_launch(x, y, z, 24, 256);
        });
        cadeia("kv_store_row(f16, 256)", [&] {
          (void)rdna4::kv_store_row_launch(rdna4::KvType::F16, x, (void *)y, 256);
        });
        cadeia("kv_store_row(q4_0, 256)", [&] {
          (void)rdna4::kv_store_row_launch(rdna4::KvType::Q4_0, x, (void *)y, 256);
        });
        // GDN com os buffers do TAMANHO CERTO (o estado do delta_rule tem
        // nvh*S*S = 786k floats; passar um buffer pequeno aqui da memory fault).
        float *gq = nullptr, *gstate = nullptr, *gout = nullptr, *cin = nullptr, *cout2 = nullptr;
        const std::size_t gb1 = (std::size_t)48 * 128 * 4;                 // 6144 floats
        const std::size_t gst = (std::size_t)48 * 128 * 128 * 4;           // 786432 floats
        const std::size_t cch = 10240;
        if (hipMalloc(&gq, 5 * gb1) == hipSuccess && hipMalloc(&gstate, gst) == hipSuccess &&
            hipMalloc(&gout, gb1) == hipSuccess && hipMalloc(&cin, cch * 4) == hipSuccess &&
            hipMalloc(&cout2, (std::size_t)cch * 4 * 4 + cch * 4 + 3 * cch * 4) == hipSuccess) {
          (void)hipMemset(gstate, 0, gst);
          cadeia("delta_rule(48 cabecas, S 128)", [&] {
            (void)rdna4::delta_rule_launch((const float *)gq, (const float *)gq + 6144,
                                           (const float *)gq + 12288, (const float *)gq + 18432,
                                           (const float *)gq + 24576, gstate, gout, 48, 16, 128);
          });
          cadeia("conv1d(4 taps, 10240 canais)", [&] {
            (void)rdna4::conv1d_state_launch((const float *)cin, (const float *)cout2, cout2 + cch,
                                             cout2 + cch + cch * 4, 10240, 4);
          });
        } else {
          std::printf("  micro GDN: hipMalloc falhou\n");
        }
        if (gq) (void)hipFree(gq);
        if (gstate) (void)hipFree(gstate);
        if (gout) (void)hipFree(gout);
        if (cin) (void)hipFree(cin);
        if (cout2) (void)hipFree(cout2);
        (void)hipEventDestroy(ea);
        (void)hipEventDestroy(eb);
      } else {
        std::printf("  micro: hipMalloc falhou\n");
      }
      if (x) (void)hipFree(x);
      if (y) (void)hipFree(y);
      if (w) (void)hipFree(w);
      if (z) (void)hipFree(z);
      if (qb) (void)hipFree(qb);
    }

    if (dq) (void)hipFree(dq);
    if (dk) (void)hipFree(dk);
    if (dv) (void)hipFree(dv);
    if (dg) (void)hipFree(dg);
    if (db) (void)hipFree(db);
    if (dstate) (void)hipFree(dstate);
    if (dout) (void)hipFree(dout);
    if (dconv) (void)hipFree(dconv);
    if (dcw) (void)hipFree(dcw);
    if (dcs) (void)hipFree(dcs);
  }

  // ------------------------------------------------------------------- model
  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(model, err)) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }
  rdna4::Qwen35Config cfg;
  if (!rdna4::parse_qwen35_config(ld.meta(), cfg, err)) {
    std::fprintf(stderr, "config: %s\n", err.c_str());
    return 1;
  }
  rdna4::Graph g(ld, cfg);
  const double t_init0 = now_ms();
  if (!g.init(ctx, kv_k, kv_v, err)) {
    std::fprintf(stderr, "graph init: %s\n", err.c_str());
    return 1;
  }
  const double t_init = now_ms() - t_init0;
  if (splits >= 0) g.set_attn_splits(splits);
  const int E = g.n_embd(), NH = g.n_head(), NKV = g.n_head_kv(), HD = g.head_dim();
  std::printf("\n== modelo\n");
  std::printf("  %d camadas (%d attn + %d GDN), embd %d, vocab %d, head %d/de %d (GQA %dx)\n",
              g.n_layer(), g.n_full_attn(), g.n_layer() - g.n_full_attn(), E, g.n_vocab(), NH, NKV,
              NKV ? NH / NKV : 0);
  std::printf("  init (upload dos pesos + caches): %.1f ms\n", t_init);

  // bytes per token: the trunk weights minus the two things decode does not read
  std::uint64_t tok_embd_bytes = 0, mtp_bytes = 0;
  for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
    const auto &t = ld.meta().tensors[i];
    const bool is_mtp = t.name.rfind("blk." + std::to_string(cfg.block_count - 1) + ".", 0) == 0 ||
                        t.name.rfind("nextn.", 0) == 0;
    if (t.name == "token_embd.weight") tok_embd_bytes = ld.tensor_bytes(i);
    else if (is_mtp) mtp_bytes += ld.tensor_bytes(i);
  }
  const std::uint64_t read_bytes = ld.total_bytes() - tok_embd_bytes - mtp_bytes;
  std::printf("  pesos: total %.3f GB, lidos por token %.3f GB (sem token_embd %.3f GB e MTP "
              "%.3f GB)\n",
              ld.total_bytes() / 1e9, read_bytes / 1e9, tok_embd_bytes / 1e9, mtp_bytes / 1e9);

  // KV bytes read by the attention in ONE token at this context (all keys are
  // attended). `unique` = one read per KV head; `issued` = what the kernel
  // actually issues, because every query head reads its own KV head row (GQA).
  const double row_k = (double)rdna4::kv_row_bytes(kv_k, HD);
  const double row_v = (double)rdna4::kv_row_bytes(kv_v, HD);
  const double keys = (double)(pos + 1);  // all measured tokens decode at the same position
  const double kv_unique = keys * NKV * (row_k + row_v) * g.n_full_attn();
  const double kv_issued = kv_unique * (NKV ? (double)NH / NKV : 1.0);
  std::printf("  KV na atencao (posicao %d, %d chaves): %.1f MB unicos, %.1f MB emitidos "
              "(GQA %dx), por token\n",
              pos, pos + 1, kv_unique / 1e6, kv_issued / 1e6, NKV ? NH / NKV : 0);

  std::vector<float> hidden, logits;
  std::int32_t id = 9419;  // "The"
  g.set_want_hidden(false);

  // Warm-up: keep the GPU busy for >= 300 ms with a sync inside the loop (the
  // DPM trap: SCLK drops to 9-16 MHz between kernels). The profiler is OFF here,
  // so no mark and no event is recorded during the warm-up.
  auto warm_up = [&]() {
    if (pos > 0 && !g.debug_fill_caches(1234u, err)) return false;
    const double w0 = now_ms();
    double last = w0;
    while (last - w0 < 300.0) {
      if (!g.forward_tokens({id}, pos, hidden, logits, err)) return false;
      id = argmax_of(logits);
      (void)hipDeviceSynchronize();
      last = now_ms();
    }
    return true;
  };

  // One measured pass: `n` decode steps, each one forward_tokens call (exactly
  // like the CLI/bench loop), no sync inside, then a single drain+finish().
  auto decode_n = [&](int n, rdna4::PhaseProf *prof, RunResult &out) {
    if (!warm_up()) {
      std::fprintf(stderr, "warm-up: %s\n", err.c_str());
      return false;
    }
    if (prof) g.set_phase_prof(prof);
    std::vector<double> per_tok;
    per_tok.reserve((std::size_t)n);
    int done = 0;
    for (int i = 0; i < n; ++i) {
      const double t0 = now_ms();
      if (!g.forward_tokens({id}, pos, hidden, logits, err)) {
        std::fprintf(stderr, "forward: %s\n", err.c_str());
        return false;
      }
      id = argmax_of(logits);
      per_tok.push_back(now_ms() - t0);
      ++done;
    }
    if (prof) {
      (void)hipDeviceSynchronize();  // drain: every recorded event has completed
      prof->finish();
      prof->set_tokens((std::int64_t)done);
    }
    g.set_phase_prof(nullptr);
    // best / median / mean: the benches report a best-of-N, so the same
    // statistic (best) is reported first, with the mean next to it.
    std::vector<double> sorted = per_tok;
    std::sort(sorted.begin(), sorted.end());
    out.ms_per_token = sorted.front();
    out.median_ms = sorted[sorted.size() / 2];
    out.mean_ms = 0;
    for (double v : per_tok) out.mean_ms += v;
    out.mean_ms /= (double)per_tok.size();
    out.tokens = (std::size_t)done;
    out.splits = g.debug_attn_splits();
    return true;
  };

  if (pos > 0) std::printf("  cache sintetico preenchido a cada warm-up (custo, nao qualidade)\n");

  // -------------------------------------------------------------- clean pass
  RunResult clean;
  if (do_clean) {
    if (!decode_n(tokens, nullptr, clean)) return 1;
    std::printf("\n== passada limpa (sem instrumentacao): melhor %.3f ms/token (%.2f tok/s), "
                "mediana %.3f, media %.3f, %zu tokens, %d splits\n",
                clean.ms_per_token, 1000.0 / clean.ms_per_token, clean.median_ms, clean.mean_ms,
                clean.tokens, clean.splits);
  }

  // -------------------------------------------------------------- count pass
  if (count_only) {
    rdna4::PhaseProf cp;
    cp.level = level;
    cp.count_only = true;  // no events at all: counting must not perturb the run
    RunResult rr;
    if (!decode_n(tokens, &cp, rr)) return 1;
    std::printf("\n== contagem de marcas (count-only, sem eventos): %.3f ms/token\n",
                rr.ms_per_token);
    std::printf("  %-24s %12s\n", "bucket", "marks/token");
    for (const auto &b : cp.buckets) {
      std::printf("  %-24s %12.1f\n", b.name.c_str(),
                  cp.tokens ? (double)b.marks / (double)cp.tokens : 0.0);
    }
    std::printf("  TOTAL de marcas/token: %.1f (as marcas de nivel 2 sao 1 por lancamento nas "
                "projecoes)\n",
                cp.tokens ? (double)cp.marks_tot() / (double)cp.tokens : 0.0);
    return 0;
  }

  // -------------------------------------------------------------- phase pass
  // NOTE: the profiler is armed by decode_n AFTER the warm-up, so the warm-up
  // tokens (which exist only to reach the DPM plateau) contribute no marks.
  rdna4::PhaseProf prof;
  prof.level = level;
  prof.ev_us = d.ev_us;
  RunResult inst;
  if (!decode_n(tokens, &prof, inst)) return 1;
  std::printf("\n== passada instrumentada (nivel %d): melhor %.3f ms/token (%.2f tok/s), mediana "
              "%.3f, media %.3f, %d splits\n",
              level, inst.ms_per_token, 1000.0 / inst.ms_per_token, inst.median_ms, inst.mean_ms,
              inst.splits);
  std::printf("  soma dos buckets: %.3f ms/token; inflacao da instrumentacao: %+.1f%%\n",
              prof.total_ms() / (double)prof.tokens,
              100.0 * (inst.ms_per_token / clean.ms_per_token - 1.0));
  prof.print("buckets por fase");
  std::printf("\n  (ms/token = tempo do kernel da fase + gap de despacho ate a proxima fase;\n"
              "   ev_us/tok = eventos que caem nesse bucket x %.3f us — subtraia para o tempo "
              "corrigido)\n",
              d.ev_us);

  // --------------------------------------------------------------- KV traffic
  std::printf("\n== trafego de KV vs tempo medido da atencao\n");
  double attn_ms = 0, matvec_ms = 0, quant_ms = 0, head_ms = 0;
  for (const auto &b : prof.buckets) {
    const double per_tok = prof.tokens ? b.ms / (double)prof.tokens : 0.0;
    if (b.name == "attention") attn_ms = per_tok;
    if (b.name == "matvec") matvec_ms = per_tok;
    if (b.name == "act_quant") quant_ms = per_tok;
    if (b.name == "head:matvec" || b.name == "head") head_ms = per_tok;
  }
  if (attn_ms > 0) {
    std::printf("  atencao medida: %.3f ms/token (%d splits)\n", attn_ms, inst.splits);
    std::printf("  bytes unicos: %.1f MB -> %.1f GB/s efetivos\n", kv_unique / 1e6,
                kv_unique / (attn_ms / 1000.0) / 1e9);
    std::printf("  bytes emitidos (GQA %dx): %.1f MB -> %.1f GB/s efetivos\n", NKV ? NH / NKV : 0,
                kv_issued / 1e6, kv_issued / (attn_ms / 1000.0) / 1e9);
  }
  if (matvec_ms > 0) {
    std::printf("  matvec (trunk, sem head): %.3f ms/token -> %.1f GB/s sobre %.3f GB lidos\n",
                matvec_ms, read_bytes / (matvec_ms / 1000.0) / 1e9, read_bytes / 1e9);
  }
  std::uint64_t out_bytes = 0;
  for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
    if (ld.meta().tensors[i].name == "output.weight") out_bytes = ld.tensor_bytes(i);
  }
  if (head_ms > 0 && out_bytes > 0) {
    std::printf("  head (output.weight %.3f GB): %.3f ms/token -> %.1f GB/s\n",
                out_bytes / 1e9, head_ms, out_bytes / (head_ms / 1000.0) / 1e9);
  }
  return 0;
}
