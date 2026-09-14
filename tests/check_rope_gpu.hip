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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
// 32 tokens (> kAttnWarpsPerBlock = 8) so every query has to walk several keys
// per warp and the cross-warp merge is exercised with 8 non-empty slices, plus a
// tail that is not a multiple of the warp count (32 % 8 == 0, but t = 13, 21, 29
// inside the loop are not).
constexpr int kTokens = 32;
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
        // GQA: contiguous grouping — query head h reads kv head
        // h / (n_head/n_head_kv) (HF's repeat_kv; verified against the graph
        // oracle in check-graph-gpu)
        const int kvh = h / (kHead / kHeadKv);
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
      // q and out are the current token's slice: attn_launch works on ONE token
      ok = ok && rdna4::attn_launch(d_q + (std::int64_t)t * kHead * kHeadDim, d_k, d_v,
                                    d_o + (std::int64_t)t * kHead * kHeadDim, t, kHead, kHeadKv,
                                    kHeadDim, scale, rdna4::KvType::F32, rdna4::KvType::F32);
      hipMemcpy(got.data() + (std::int64_t)t * kHead * kHeadDim,
                d_o + (std::int64_t)t * kHead * kHeadDim,
                (std::size_t)kHead * kHeadDim * sizeof(float), hipMemcpyDeviceToHost);
    }
    const double rel = rel_l2(ref, got);
    std::printf("attn (causal) rel-L2=%.3e  %s\n", rel, (ok && rel < 1e-5) ? "OK" : "MISMATCH");

    // --- single-key case over every KV cache type: with one key the softmax is
    // 1, so the output must be the quantized-and-dequantized V of the query
    // head's kv head. This exercises kv_store_row_launch + kv_load for each type
    // (and catches a wrong cache row pointer immediately).
    {
      const int ctx = 8;  // cache rows, only row 0 is attended
      std::vector<float> kc((std::size_t)ctx * kHeadKv * kHeadDim);
      std::vector<float> vc((std::size_t)ctx * kHeadKv * kHeadDim);
      for (auto &f : kc) f = frand(seed);
      for (auto &f : vc) f = frand(seed);
      float *d_ksrc = nullptr, *d_vsrc = nullptr, *d_q1 = nullptr, *d_o1 = nullptr;
      hipMalloc(&d_ksrc, (std::size_t)kHeadKv * kHeadDim * sizeof(float));
      hipMalloc(&d_vsrc, (std::size_t)kHeadKv * kHeadDim * sizeof(float));
      hipMalloc(&d_q1, (std::size_t)kHead * kHeadDim * sizeof(float));
      hipMalloc(&d_o1, (std::size_t)kHead * kHeadDim * sizeof(float));
      hipMemcpy(d_q1, q.data(), (std::size_t)kHead * kHeadDim * sizeof(float),
                hipMemcpyHostToDevice);
      hipMemcpy(d_vsrc, vc.data(), (std::size_t)kHeadKv * kHeadDim * sizeof(float),
                hipMemcpyHostToDevice);
      hipMemcpy(d_ksrc, kc.data(), (std::size_t)kHeadKv * kHeadDim * sizeof(float),
                hipMemcpyHostToDevice);

      // Q5_0/Q4_1 are the night's KV formats (frente KV). They are in this list
      // for the same reason the other four are: the host mirror below is written
      // from the ggml reference (quantize_row_q5_0_ref / quantize_row_q4_1_ref and
      // dequantize_row_q5_0 / dequantize_row_q4_1) and NOT from the kernels, and
      // the attention check right after proves that the shipping kernel returns
      // exactly what those bytes decode to, at every offset of the row.
      const rdna4::KvType types[] = {rdna4::KvType::F32,  rdna4::KvType::F16,
                                     rdna4::KvType::Q8_0, rdna4::KvType::Q4_0,
                                     rdna4::KvType::Q5_0, rdna4::KvType::Q4_1};
      for (rdna4::KvType ty : types) {
        const std::uint64_t row = rdna4::kv_row_bytes(ty, kHeadDim);
        std::vector<std::uint8_t> kcache((std::size_t)kHeadKv * row, 0);
        std::vector<std::uint8_t> vcache((std::size_t)kHeadKv * row, 0);
        void *d_kcache = nullptr, *d_vcache = nullptr;
        hipMalloc(&d_kcache, kcache.size());
        hipMalloc(&d_vcache, vcache.size());
        hipMemset(d_kcache, 0, kcache.size());
        hipMemset(d_vcache, 0, vcache.size());
        bool ok1 = true;
        for (int h = 0; h < kHeadKv; ++h) {
          ok1 = ok1 && rdna4::kv_store_row_launch(ty, d_ksrc + (std::size_t)h * kHeadDim,
                                                  (std::uint8_t *)d_kcache + h * row, kHeadDim);
          ok1 = ok1 && rdna4::kv_store_row_launch(ty, d_vsrc + (std::size_t)h * kHeadDim,
                                                  (std::uint8_t *)d_vcache + h * row, kHeadDim);
        }
        ok1 = ok1 && rdna4::attn_launch(d_q1, d_kcache, d_vcache, d_o1, 0, kHead, kHeadKv, kHeadDim,
                                        scale, ty, ty);
        std::vector<float> got1((std::size_t)kHead * kHeadDim);
        hipMemcpy(got1.data(), d_o1, got1.size() * sizeof(float), hipMemcpyDeviceToHost);
        // read the quantized rows back: the expectation is "the kernel returns
        // exactly what was stored", whatever the storage type did to the values
        hipMemcpy(kcache.data(), d_kcache, kcache.size(), hipMemcpyDeviceToHost);
        hipMemcpy(vcache.data(), d_vcache, vcache.size(), hipMemcpyDeviceToHost);
        hipDeviceSynchronize();

        // Byte-exactness of the stored rows against ggml's reference
        // quantizers (host mirror of quantize_row_q8_0_ref /
        // quantize_row_q4_0_ref / GGML_FP32_TO_FP16). This is what backs the
        // "the cache holds what llama.cpp would store" claim; the attention
        // check below then only has to prove the loader reads it back.
        {
          int bad_bytes = 0;
          double worst_f16 = 0.0;
          for (int h = 0; h < kHeadKv; ++h) {
            const std::uint8_t *rowp = vcache.data() + (std::size_t)h * row;
            const float *src = vc.data() + (std::size_t)h * kHeadDim;
            for (int blk = 0; blk < kHeadDim / 32; ++blk) {
              const float *xb = src + blk * 32;
              if (ty == rdna4::KvType::F32) {
                for (int j = 0; j < 32; ++j) {
                  float v = 0.0f;
                  std::memcpy(&v, rowp + ((std::size_t)blk * 32 + j) * 4, 4);
                  if (v != xb[j]) ++bad_bytes;
                }
              } else if (ty == rdna4::KvType::F16) {
                for (int j = 0; j < 32; ++j) {
                  std::uint16_t h16 = 0;
                  std::memcpy(&h16, rowp + ((std::size_t)blk * 32 + j) * 2, 2);
                  const double back = rdna4::fp16_to_float(h16);
                  worst_f16 = std::max(worst_f16, std::fabs(back - (double)xb[j]) /
                                                        std::max(1e-9, (double)std::fabs(xb[j])));
                }
              } else if (ty == rdna4::KvType::Q8_0) {
                float amax = 0.0f;
                for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xb[j]));
                const float d = amax / 127.0f;
                const float id = d != 0.0f ? 1.0f / d : 0.0f;
                const rdna4::block_q8_0 *b =
                    (const rdna4::block_q8_0 *)rowp + blk;
                if (b->d != rdna4::float_to_fp16(d)) ++bad_bytes;
                for (int j = 0; j < 32; ++j) {
                  if (b->qs[j] != (std::int8_t)std::round(xb[j] * id)) ++bad_bytes;
                }
              } else if (ty == rdna4::KvType::Q5_0) {
                // ggml-quants.c:187 quantize_row_q5_0_ref
                float amax = 0.0f, vmax = 0.0f;
                for (int j = 0; j < 32; ++j) {
                  if (amax < std::fabs(xb[j])) {
                    amax = std::fabs(xb[j]);
                    vmax = xb[j];
                  }
                }
                const float d = vmax / -16.0f;
                const float id = d != 0.0f ? 1.0f / d : 0.0f;
                const rdna4::block_q5_0 *b = (const rdna4::block_q5_0 *)rowp + blk;
                if (b->d != rdna4::float_to_fp16(d)) ++bad_bytes;
                std::uint32_t qh = 0;
                for (int j = 0; j < 16; ++j) {
                  const int q0 = (int)(std::int8_t)(xb[j] * id + 16.5f);
                  const int q1 = (int)(std::int8_t)(xb[j + 16] * id + 16.5f);
                  const std::uint8_t xi0 = (std::uint8_t)(q0 > 31 ? 31 : (q0 < 0 ? 0 : q0));
                  const std::uint8_t xi1 = (std::uint8_t)(q1 > 31 ? 31 : (q1 < 0 ? 0 : q1));
                  if (b->qs[j] != (std::uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4))) ++bad_bytes;
                  qh |= (std::uint32_t)((xi0 & 0x10u) >> 4) << (j + 0);
                  qh |= (std::uint32_t)((xi1 & 0x10u) >> 4) << (j + 16);
                }
                for (int i = 0; i < 4; ++i) {
                  if (b->qh[i] != (std::uint8_t)((qh >> (8 * i)) & 0xFFu)) ++bad_bytes;
                }
              } else if (ty == rdna4::KvType::Q4_1) {
                // ggml-quants.c:150 quantize_row_q4_1_ref
                float vmin = xb[0], vmax = xb[0];
                for (int j = 1; j < 32; ++j) {
                  vmin = std::min(vmin, xb[j]);
                  vmax = std::max(vmax, xb[j]);
                }
                const float d = (vmax - vmin) / 15.0f;
                const float id = d != 0.0f ? 1.0f / d : 0.0f;
                const rdna4::block_q4_1 *b = (const rdna4::block_q4_1 *)rowp + blk;
                if (b->d != rdna4::float_to_fp16(d) || b->m != rdna4::float_to_fp16(vmin)) {
                  ++bad_bytes;
                }
                for (int j = 0; j < 16; ++j) {
                  const int q0 = (int)(std::int8_t)((xb[j] - vmin) * id + 0.5f);
                  const int q1 = (int)(std::int8_t)((xb[j + 16] - vmin) * id + 0.5f);
                  const std::uint8_t xi0 = (std::uint8_t)(q0 > 15 ? 15 : (q0 < 0 ? 0 : q0));
                  const std::uint8_t xi1 = (std::uint8_t)(q1 > 15 ? 15 : (q1 < 0 ? 0 : q1));
                  if (b->qs[j] != (std::uint8_t)(xi0 | (xi1 << 4))) ++bad_bytes;
                }
              } else {
                float amax = 0.0f, vmax = 0.0f;
                for (int j = 0; j < 32; ++j) {
                  if (amax < std::fabs(xb[j])) {
                    amax = std::fabs(xb[j]);
                    vmax = xb[j];
                  }
                }
                const float d = vmax / -8.0f;
                const float id = d != 0.0f ? 1.0f / d : 0.0f;
                const rdna4::block_q4_0 *b = (const rdna4::block_q4_0 *)rowp + blk;
                if (b->d != rdna4::float_to_fp16(d)) ++bad_bytes;
                for (int j = 0; j < 16; ++j) {
                  const int q0 = (int)(std::int8_t)(xb[j] * id + 8.5f);
                  const int q1 = (int)(std::int8_t)(xb[j + 16] * id + 8.5f);
                  const std::uint8_t xi0 = (std::uint8_t)(q0 > 15 ? 15 : (q0 < 0 ? 0 : q0));
                  const std::uint8_t xi1 = (std::uint8_t)(q1 > 15 ? 15 : (q1 < 0 ? 0 : q1));
                  const std::uint8_t want = (std::uint8_t)(xi0 | (xi1 << 4));
                  if (b->qs[j] != want) ++bad_bytes;
                }
              }
            }
          }
          const bool ok3 = (ty == rdna4::KvType::F16) ? (worst_f16 <= 5e-4) : (bad_bytes == 0);
          std::printf("kv store  (%-4s) %s  %s\n", rdna4::kv_type_name(ty),
                      (ty == rdna4::KvType::F16)
                          ? ("max relative fp16 error = " + std::to_string(worst_f16)).c_str()
                          : ("bytes differing from the ggml reference: " +
                             std::to_string(bad_bytes)).c_str(),
                      ok3 ? "OK" : "MISMATCH");
          if (!ok3) ++failures;
        }

        // The kernel must return exactly what the cache holds (the quantized
        // values, dequantized): this checks every offset of the row layout.
        double worst = 0.0;
        for (int h = 0; h < kHead; ++h) {
          const int kvh = h / (kHead / kHeadKv);
          for (int i = 0; i < kHeadDim; ++i) {
            const double got_v = (double)got1[(std::size_t)h * kHeadDim + i];
            double exp_v = 0.0;
            if (ty == rdna4::KvType::F32) {
              float f32v = 0.0f;
              std::memcpy(&f32v, vcache.data() + (std::uint64_t)kvh * row + i * 4, 4);
              exp_v = (double)f32v;
            } else if (ty == rdna4::KvType::F16) {
              std::uint16_t h16 = 0;
              std::memcpy(&h16, vcache.data() + (std::uint64_t)kvh * row + i * 2, 2);
              exp_v = rdna4::fp16_to_float(h16);
            } else if (ty == rdna4::KvType::Q8_0) {
              const rdna4::block_q8_0 *b =
                  (const rdna4::block_q8_0 *)(vcache.data() + (std::uint64_t)kvh * row) + (i / 32);
              exp_v = rdna4::fp16_to_float(b->d) * (float)b->qs[i & 31];
            } else if (ty == rdna4::KvType::Q5_0) {
              const rdna4::block_q5_0 *b = (const rdna4::block_q5_0 *)(vcache.data() +
                                                                       (std::uint64_t)kvh * row) +
                                           (i / 32);
              const int j = i & 31;
              const std::uint8_t byte = b->qs[j & 15];
              const int nib = (j < 16) ? (byte & 0x0F) : (byte >> 4);
              const int hi = (b->qh[j >> 3] >> (j & 7)) & 1;
              exp_v = rdna4::fp16_to_float(b->d) * (double)((nib | (hi << 4)) - 16);
            } else if (ty == rdna4::KvType::Q4_1) {
              const rdna4::block_q4_1 *b = (const rdna4::block_q4_1 *)(vcache.data() +
                                                                       (std::uint64_t)kvh * row) +
                                           (i / 32);
              const int j = i & 31;
              const std::uint8_t byte = b->qs[j & 15];
              const int qq = (j < 16) ? (byte & 0x0F) : (byte >> 4);
              exp_v = rdna4::fp16_to_float(b->d) * (double)qq + rdna4::fp16_to_float(b->m);
            } else {
              const rdna4::block_q4_0 *b =
                  (const rdna4::block_q4_0 *)(vcache.data() + (std::uint64_t)kvh * row) + (i / 32);
              const int j = i & 31;
              const std::uint8_t byte = b->qs[j & 15];
              const int qq = (j < 16) ? (byte & 0x0F) : (byte >> 4);
              exp_v = rdna4::fp16_to_float(b->d) * (double)(qq - 8);
            }
            worst = std::max(worst, std::fabs(got_v - exp_v));
          }
        }
        // tolerance: the stored quantized values are what they are; the kernel
        // must return exactly those (float rounding of ~1e-7 relative)
        const bool ok2 = ok1 && worst < 1e-4;
        std::printf("attn (1 key, %-4s) max|out - cache| = %.3e  %s\n", rdna4::kv_type_name(ty),
                    worst, ok2 ? "OK" : "MISMATCH");
        if (!ok2) ++failures;
        hipFree(d_kcache);
        hipFree(d_vcache);
      }
      hipFree(d_ksrc); hipFree(d_vsrc); hipFree(d_q1); hipFree(d_o1);
    }

    if (!ok || rel >= 1e-5) ++failures;
    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_o);
  }

  std::printf("check-rope-gpu: %s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
