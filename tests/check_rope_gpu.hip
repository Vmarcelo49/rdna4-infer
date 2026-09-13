// M3 step 2a — RoPE + causal attention kernels vs a direct CPU implementation.
//
// Validates the kernels in include/rdna4/attn.cuh against the documented
// reference semantics (llama.cpp qwen35 + ggml_rope_cache_init):
//   angle(pair p, position t) = pos[t] * freq_base^(-2p/n_rot), split-half pairs
//   (element p rotates with element p + n_rot/2 — ggml's rotate_pairs with
//   n_offset = n_dims/2 is what NEOX/MROPE/IMROPE use),
//   applied to the first n_rot dims of each head.
// The attention check is a straightforward softmax(q·k/sqrt(d))·v reference.
//
// usage: check-rope-gpu
#include "rdna4/attn.cuh"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
constexpr int kTokens = 8;
constexpr int kHeadDim = 256;
constexpr int kHead = 24;
constexpr int kHeadKv = 4;
constexpr int kRot = 64;
constexpr float kBase = 10000000.0f;  // qwen35 rope.freq_base

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
}  // namespace

int main() {
  int failures = 0;
  unsigned seed = 12345;

  // ---------------- RoPE ----------------
  {
    const int n = kTokens * kHead * kHeadDim;
    std::vector<float> x(n), ref;
    for (int i = 0; i < n; ++i) x[i] = frand(seed);
    ref = x;
    std::vector<int> pos(kTokens);
    for (int t = 0; t < kTokens; ++t) pos[t] = t + 3;  // non-zero start position
    for (int t = 0; t < kTokens; ++t) {
      for (int h = 0; h < kHead; ++h) {
        float *base = ref.data() + ((std::int64_t)t * kHead + h) * kHeadDim;
        for (int p = 0; p < kRot / 2; ++p) {
          const float theta = (float)pos[t] * std::pow(kBase, -2.0f * (float)p / (float)kRot);
          const float c = std::cos(theta), s = std::sin(theta);
          const float x0 = base[p], x1 = base[p + kRot / 2];
          base[p] = x0 * c - x1 * s;
          base[p + kRot / 2] = x0 * s + x1 * c;
        }
      }
    }
    float *d_x = nullptr;
    int *d_pos = nullptr;
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_pos, kTokens * sizeof(int));
    hipMemcpy(d_x, x.data(), n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_pos, pos.data(), kTokens * sizeof(int), hipMemcpyHostToDevice);
    const bool ok = rdna4::rope_launch(d_x, kTokens, kHead, kHeadDim, kRot, kBase, d_pos);
    std::vector<float> got(n);
    hipMemcpy(got.data(), d_x, n * sizeof(float), hipMemcpyDeviceToHost);
    const double rel = rel_l2(ref, got);
    std::printf("rope          rel-L2=%.3e  %s\n", rel, (ok && rel < 1e-6) ? "OK" : "MISMATCH");
    if (!ok || rel >= 1e-6) ++failures;
    hipFree(d_x);
    hipFree(d_pos);
  }

  // ---------------- causal attention ----------------
  {
    const std::int64_t nq = (std::int64_t)kTokens * kHead * kHeadDim;
    const std::int64_t nkv = (std::int64_t)kTokens * kHeadKv * kHeadDim;
    std::vector<float> q(nq), k(nkv), v(nkv);
    for (auto &f : q) f = frand(seed);
    for (auto &f : k) f = frand(seed);
    for (auto &f : v) f = frand(seed);

    const float scale = 1.0f / std::sqrt((float)kHeadDim);
    std::vector<float> ref(nq, 0.0f);
    for (int t = 0; t < kTokens; ++t) {
      for (int h = 0; h < kHead; ++h) {
        // GQA as the reference does it: query head h reads kv head h % kHeadKv
        // (ggml flash attention repeats the kv heads into the query heads)
        const int kvh = h % kHeadKv;
        std::vector<double> sc(t + 1);
        double m = -1e30;
        for (int j = 0; j <= t; ++j) {
          double d = 0;
          for (int i = 0; i < kHeadDim; ++i) {
            d += (double)q[((std::int64_t)t * kHead + h) * kHeadDim + i] *
                 k[((std::int64_t)j * kHeadKv + kvh) * kHeadDim + i];
          }
          sc[j] = d * scale;
          m = std::max(m, sc[j]);
        }
        double sum = 0;
        for (int j = 0; j <= t; ++j) { sc[j] = std::exp(sc[j] - m); sum += sc[j]; }
        for (int i = 0; i < kHeadDim; ++i) {
          double acc = 0;
          for (int j = 0; j <= t; ++j) {
            acc += (sc[j] / sum) * v[((std::int64_t)j * kHeadKv + kvh) * kHeadDim + i];
          }
          ref[((std::int64_t)t * kHead + h) * kHeadDim + i] = (float)acc;
        }
      }
    }

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    hipMalloc(&d_q, nq * sizeof(float));
    hipMalloc(&d_k, nkv * sizeof(float));
    hipMalloc(&d_v, nkv * sizeof(float));
    hipMalloc(&d_o, nq * sizeof(float));
    hipMemcpy(d_q, q.data(), nq * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_k, k.data(), nkv * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_v, v.data(), nkv * sizeof(float), hipMemcpyHostToDevice);
    // one launch per query token, against the cache rows 0..t
    std::vector<float> got(nq, 0.0f);
    bool ok = true;
    for (int t = 0; t < kTokens; ++t) {
      // q is the current token's slice: attn_launch takes [n_head, head_dim]
      ok = ok && rdna4::attn_launch(d_q + (std::int64_t)t * kHead * kHeadDim, d_k, d_v, d_o, t,
                                    kHead, kHeadKv, kHeadDim, scale);
      hipMemcpy(got.data() + (std::int64_t)t * kHead * kHeadDim, d_o, (std::size_t)kHead * kHeadDim * sizeof(float),
                hipMemcpyDeviceToHost);
    }
    const double rel = rel_l2(ref, got);
    std::printf("attn (causal) rel-L2=%.3e  %s\n", rel, (ok && rel < 1e-5) ? "OK" : "MISMATCH");
    if (!ok || rel >= 1e-5) ++failures;
    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_o);
  }

  std::printf("check-rope-gpu: %s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
