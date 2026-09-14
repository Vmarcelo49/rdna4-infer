// M-noite/longctx — RoPE against llama.cpp's real ggml at LONG positions.
//
// Why this exists (docs/journal-longctx.md §2): the two RoPE gates we already had
// are blind to long context.
//   * tests/check_rope_gpu.hip compares our kernel against a *hand-written*
//     reference at positions 3..34 only (its own formula, same op order as the
//     kernel): it cannot see a wrong OP ORDER, and it cannot see anything that
//     only shows up at large |theta|;
//   * check-graph-gpu compares the assembled graph against a captured dump whose
//     prompt is 8 tokens long (positions 0..7).
// This test calls the *actual reference implementation* — ggml's
// ggml_rope_multi()/ggml_rope_ext() on the CPU backend, the same functions
// llama.cpp's qwen35 graph calls (src/models/qwen35.cpp:587-592) — with the
// runtime parameters llama.cpp actually resolves for this GGUF:
//
//   mode        = GGML_ROPE_TYPE_IMROPE   (llama-model.cpp:3041, LLM_ARCH_QWEN35)
//   sections    = {11, 11, 10, 0}         (qwen35.rope.dimension_sections)
//   n_rot       = 64                      (qwen35.rope.dimension_count)
//   freq_base   = 1e7                     (qwen35.rope.freq_base)
//   freq_scale  = 1.0                     (no rope.scaling.factor -> 1.0, llama-model.cpp:1330)
//   ext_factor  = 0.0                     (scaling type LINEAR != YARN, llama-context.cpp:172)
//   attn_factor = 1.0                     (rope_attn_factor, llama-hparams.h:149)
//   positions   = the same value in all 4 M-RoPE channels, which is what
//                 llama.cpp does for a *token* batch (llama-batch.cpp:780-788:
//                 "if the current batch is text, we need to broadcast the same
//                 position across all RoPE sections")
//
// What it reports, per position:
//   ours vs ggml   max|delta| and rel-L2 of the rotated tensor  (fidelity)
//   ref  vs ideal  how far the *reference itself* is from the exact (double)
//                  rotation -- at 131K the phase pos*freq_base^(-2p/n_rot) is
//                  ~1e5 rad, float ulp(theta) ~ 0.008, so cos/sin carry ~1e-2
//                  absolute error no matter who computes them
//   theta deltas   the angle computed by our kernel (powf per pair) vs the angle
//                  ggml accumulates (theta *= theta_scale, n_rot/2 times), in
//                  ulps of theta: this is the *cause* of the tensor difference
//   neox control   ggml NEOX-n_dims=64 (a plain split-half rope) must reproduce
//                  the IMROPE+sections call when the 4 channels are equal
//   yarn x4        what the reference would look like if a rope.scaling were
//                  applied (freq_scale 0.25 / ext_factor 1) -- the size of the
//                  P0 error we would have if we were missing scaling
//
// usage: check-rope-long-gpu [--threads N]
#include "rdna4/attn.cuh"

#include <hip/hip_runtime.h>

#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kHeadDim = 256;
constexpr int kHeads = 24;
constexpr int kHeadsKv = 4;
constexpr int kRot = 64;
constexpr float kBase = 10000000.0f;
constexpr int kNctxOrig = 262144;  // qwen35.context_length in the GGUF
constexpr float kFreqScale = 1.0f;
constexpr float kExtFactor = 0.0f;
constexpr float kAttnFactor = 1.0f;
constexpr float kBetaFast = 32.0f;
constexpr float kBetaSlow = 1.0f;

// Positions to test. Short ones catch a formula bug; the long ones are the point
// of the night (131072 = the target context, 262144 = the model's n_ctx_train).
const int kPos[] = {0, 1, 3, 34, 512, 4096, 16384, 32768, 65536, 131071, 262143};
constexpr int kNTok = (int)(sizeof(kPos) / sizeof(kPos[0]));

float frand(unsigned &s) {
  s = s * 1664525u + 1013904223u;
  return ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 2.0f;
}

double rel_l2(const std::vector<float> &a, const std::vector<float> &b) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - b[i];
    num += d * d;
    den += (double)a[i] * a[i];
  }
  return std::sqrt(num / (den > 0 ? den : 1.0));
}

double max_abs(const std::vector<float> &a, const std::vector<float> &b, int *where) {
  double worst = 0;
  int at = -1;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::fabs((double)a[i] - b[i]);
    if (d > worst) { worst = d; at = (int)i; }
  }
  if (where) *where = at;
  return worst;
}

// ---------------------------------------------------------------------------
// The reference: run one rope variant through ggml's own CPU implementation.
// x layout matches the engine's rope_kernel: [n_tokens, n_heads, head_dim], and
// ggml's is [head_dim, n_heads, n_tokens] (ne[0] fastest), so the same buffer
// serves both.
// ---------------------------------------------------------------------------
std::vector<float> ggml_rope(const std::vector<float> &x, int mode, const int sections[4],
                             float freq_scale, float ext_factor, float attn_factor, int threads) {
  const std::size_t n = x.size();
  struct ggml_init_params ip = {/*.mem_size=*/64u * 1024u * 1024u, /*.mem_buffer=*/nullptr,
                                /*.no_alloc=*/false};
  struct ggml_context *ctx = ggml_init(ip);
  if (!ctx) { std::fprintf(stderr, "ggml_init failed\n"); std::exit(1); }

  struct ggml_tensor *a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kHeads, kNTok);
  std::memcpy(a->data, x.data(), n * sizeof(float));

  // pos: n_tokens * 4 for M-RoPE (4 position rows t/h/w/e), n_tokens otherwise.
  // llama.cpp builds them with the same value in all 4 rows for token batches
  // (llama-batch.cpp:780-788).
  const bool mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;
  struct ggml_tensor *b =
      ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (std::int64_t)kNTok * (mrope ? 4 : 1));
  std::int32_t *pd = (std::int32_t *)b->data;
  for (int j = 0; j < (mrope ? 4 : 1); ++j)
    for (int t = 0; t < kNTok; ++t) pd[j * kNTok + t] = kPos[t];

  struct ggml_tensor *res;
  if (mrope) {
    int sec[GGML_MROPE_SECTIONS] = {sections[0], sections[1], sections[2], sections[3]};
    res = ggml_rope_multi(ctx, a, b, nullptr, kRot, sec, mode, kNctxOrig, kBase, freq_scale,
                          ext_factor, attn_factor, kBetaFast, kBetaSlow);
  } else {
    res = ggml_rope_ext(ctx, a, b, nullptr, kRot, mode, kNctxOrig, kBase, freq_scale, ext_factor,
                        attn_factor, kBetaFast, kBetaSlow);
  }

  struct ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, res);
  const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, threads);
  if (st != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "ggml compute failed\n"); std::exit(1); }

  std::vector<float> out(n);
  std::memcpy(out.data(), res->data, n * sizeof(float));
  ggml_free(ctx);
  return out;
}

void ours_rope(std::vector<float> &x, std::vector<int> &pos) {
  const std::size_t n = x.size();
  float *d_x = nullptr;
  int *d_pos = nullptr;
  hipMalloc(&d_x, n * sizeof(float));
  hipMalloc(&d_pos, pos.size() * sizeof(int));
  hipMemcpy(d_x, x.data(), n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_pos, pos.data(), pos.size() * sizeof(int), hipMemcpyHostToDevice);
  if (!rdna4::rope_launch(d_x, (int)pos.size(), kHeads, kHeadDim, kRot, kBase, d_pos)) {
    std::fprintf(stderr, "rope_launch failed\n");
    std::exit(1);
  }
  hipMemcpy(x.data(), d_x, n * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(d_x);
  hipFree(d_pos);
}

// The exact rotation, in double: theta = pos * freq_base^(-2p/n_rot) application on
// split-half pairs. This is the definition both engines are approximating in fp32.
std::vector<float> ideal_rope(const std::vector<float> &x) {
  std::vector<float> out = x;
  for (int t = 0; t < kNTok; ++t) {
    for (int h = 0; h < kHeads; ++h) {
      float *base = out.data() + ((std::int64_t)t * kHeads + h) * kHeadDim;
      for (int p = 0; p < kRot / 2; ++p) {
        const double theta = (double)kPos[t] * std::pow((double)kBase, -2.0 * (double)p / (double)kRot);
        const double c = std::cos(theta), s = std::sin(theta);
        const double x0 = x[((std::int64_t)t * kHeads + h) * kHeadDim + p];
        const double x1 = x[((std::int64_t)t * kHeads + h) * kHeadDim + p + kRot / 2];
        base[p] = (float)(x0 * c - x1 * s);
        base[p + kRot / 2] = (float)(x0 * s + x1 * c);
      }
    }
  }
  return out;
}

// Angles, computed exactly the way each side computes them.
//   ours: (float)pos * powf(base, -2p/n_rot)          (attn.cuh:39)
//   ggml: theta = (float)pos; theta *= theta_scale  p times, theta_scale =
//         powf(base, -2.0f/n_rot)                      (ggml-cpu/ops.cpp:5964,6040-6052)
struct AngleCmp {
  double max_dtheta = 0, max_dcos = 0, max_dsin = 0, max_ulp = 0;
  int worst_t = -1, worst_p = -1;
};

AngleCmp angles() {
  AngleCmp r;
  const float theta_scale = std::pow(kBase, -2.0f / (float)kRot);
  for (int t = 0; t < kNTok; ++t) {
    float acc = (float)kPos[t];
    for (int p = 0; p < kRot / 2; ++p) {
      const float ours = (float)kPos[t] * std::pow(kBase, -2.0f * (float)p / (float)kRot);
      if (p > 0) acc *= theta_scale;
      const double dt = std::fabs((double)ours - (double)acc);
      const double dcos = std::fabs(std::cos((double)ours) - std::cos((double)acc));
      const double dsin = std::fabs(std::sin((double)ours) - std::sin((double)acc));
      // ulp of the angle magnitude (fp32)
      const double a = std::fabs((double)acc);
      const double ulp = a == 0 ? 1e-45 : std::ldexp(1.0, (int)std::floor(std::log2(a)) - 23);
      if (dt / ulp > r.max_ulp) { r.max_ulp = dt / ulp; r.worst_t = t; r.worst_p = p; }
      r.max_dtheta = std::max(r.max_dtheta, dt);
      r.max_dcos = std::max(r.max_dcos, dcos);
      r.max_dsin = std::max(r.max_dsin, dsin);
    }
  }
  return r;
}
}  // namespace

int main(int argc, char **argv) {
  int threads = 4;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--threads") == 0) threads = std::atoi(argv[i + 1]);

  const std::size_t n = (std::size_t)kNTok * kHeads * kHeadDim;
  std::vector<float> x(n);
  unsigned seed = 987654321u;
  for (auto &f : x) f = frand(seed);

  std::vector<int> pos(kNTok);
  for (int t = 0; t < kNTok; ++t) pos[t] = kPos[t];

  const int sections[4] = {11, 11, 10, 0};

  std::printf("check-rope-long-gpu: ggml reference = ggml_rope_multi(IMROPE, sections "
              "{11,11,10,0}, n_rot 64, base 1e7, freq_scale 1.0, ext_factor 0.0)\n");
  std::printf("positions:");
  for (int t = 0; t < kNTok; ++t) std::printf(" %d", kPos[t]);
  std::printf("\n\n");

  std::vector<float> ref = ggml_rope(x, GGML_ROPE_TYPE_IMROPE, sections, kFreqScale, kExtFactor,
                                     kAttnFactor, threads);
  std::vector<float> got = x;
  ours_rope(got, pos);
  std::vector<float> ideal = ideal_rope(x);
  const std::vector<float> neox = ggml_rope(x, GGML_ROPE_TYPE_NEOX, sections, kFreqScale,
                                            kExtFactor, kAttnFactor, threads);
  const std::vector<float> yarn4 = ggml_rope(x, GGML_ROPE_TYPE_IMROPE, sections, 0.25f, 1.0f,
                                             0.5f, threads);  // as if a x4 YaRN were configured
  const std::vector<float> lin4 = ggml_rope(x, GGML_ROPE_TYPE_IMROPE, sections, 0.25f, 0.0f,
                                            1.0f, threads);  // as if a x4 linear were configured

  // per-position slice indices
  auto slice = [&](const std::vector<float> &v, int t) {
    std::vector<float> s((std::size_t)kHeads * kHeadDim);
    std::memcpy(s.data(), v.data() + (std::size_t)t * kHeads * kHeadDim, s.size() * sizeof(float));
    return s;
  };
  auto slice_all = [&](const std::vector<float> &v) { return v; };

  std::printf("%9s %12s %11s %11s %11s %11s %11s %11s\n", "pos", "ours-ref", "relL2", "relL2(ref",
              "relL2(ours", "relL2(neox", "relL2(yarn4", "relL2(lin4");
  std::printf("%9s %12s %11s %11s %11s %11s %11s %11s\n", "", "max|d|", "(ours,ref)", ",ideal)",
              ",ideal)", "ctrl,ref)", ",ref)", ",ref)");

  double worst_rel = 0, worst_max = 0;
  int worst_pos = -1;
  int failures = 0;
  for (int t = 0; t < kNTok; ++t) {
    const std::vector<float> r = slice(ref, t), o = slice(got, t), id = slice(ideal, t);
    const std::vector<float> nx = slice(neox, t), ya = slice(yarn4, t), li = slice(lin4, t);
    int where = -1;
    const double mx = max_abs(r, o, &where);
    const double rel = rel_l2(r, o);
    std::printf("%9d %12.4e %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e\n", kPos[t], mx, rel,
                rel_l2(r, id), rel_l2(o, id), rel_l2(nx, r), rel_l2(ya, r), rel_l2(li, r));
    if (rel > worst_rel) { worst_rel = rel; worst_pos = kPos[t]; }
    worst_max = std::max(worst_max, mx);
  }

  const AngleCmp ac = angles();
  std::printf("\nangle check (ours = pos*powf per pair, ggml = pos *= theta_scale per pair):\n");
  std::printf("  max |dtheta| = %.4e   max |dcos| = %.4e   max |dsin| = %.4e   max %.2f ulp"
              "  (worst at pos %d, pair %d)\n",
              ac.max_dtheta, ac.max_dcos, ac.max_dsin, ac.max_ulp,
              ac.worst_t >= 0 ? kPos[ac.worst_t] : -1, ac.worst_p);
  std::printf("  float ulp of theta at 131071 (pair 1) = %.4e; 1 ulp of the angle is the whole\n"
              "  budget: cos/sin of a ~1e5 rad phase cannot be closer than that in fp32.\n",
              std::ldexp(1.0, (int)std::floor(std::log2((double)((float)131071 * 0.6043f))) - 23));

  std::printf("\nworst position: %d  rel-L2 %.3e  max|d| %.3e\n", worst_pos, worst_rel, worst_max);
  std::printf("control: ggml IMROPE+sections vs ggml NEOX-64 rel-L2 = %.3e (must be ~0: with the\n"
              "         4 M-RoPE channels equal, IMROPE degenerates to split-half over n_rot)\n",
              rel_l2(ref, neox));

  // Formula gate. A wrong formula (e.g. ignoring the M-RoPE sections, or a
  // missing/wrong scaling) is an O(0.1..1) difference -- the yarn4/lin4 columns
  // above measure exactly that size. What is left between two *correct* fp32
  // implementations is the phase rounding: theta ~ pos * freq_base^(-2p/n_rot)
  // reaches 1e5 rad at 131K, so one ulp of theta is 7.8e-3 rad and cos/sin
  // inherit it. The measured law is rel-L2 ~= 4e-9 * pos (the "measured law"
  // line below); the gate allows 10x that, which fails loudly if a real
  // disagreement (scaling, pairing, sections) ever appears.
  double law_num = 0, law_den = 0;
  for (int t = 0; t < kNTok; ++t) {
    if (kPos[t] < 1) continue;
    const double rel = rel_l2(slice(ref, t), slice(got, t));
    const double limit = std::max(1e-7, 4e-8 * (double)kPos[t]);
    law_num += rel;
    law_den += (double)kPos[t];
    if (!(rel < limit)) {
      std::printf("FAIL at pos %d: rel-L2 %.3e >= %.3e\n", kPos[t], rel, limit);
      ++failures;
    }
  }
  std::printf("measured law: mean(rel-L2/pos) = %.3e  (gate: rel-L2 < max(1e-7, 4e-8*pos))\n",
              law_num / law_den);
  if (rel_l2(ref, neox) > 1e-6) {
    std::printf("FAIL: the IMROPE/sections call is NOT equivalent to a plain split-half rope "
                "(rel-L2 %.3e)\n", rel_l2(ref, neox));
    ++failures;
  }
  std::printf("check-rope-long-gpu: %s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
