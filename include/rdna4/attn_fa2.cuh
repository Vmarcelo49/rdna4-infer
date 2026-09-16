#pragma once
// FA2-style fused-GQA prefill attention (H5, medido 16/09).
//
// UMA CTA por (KV-head x tile de QT=16 queries) = 96 linhas (16 pos x 6
// q-heads, razao GQA 6:1 deste modelo). Cada tile KT64 de chaves e'
// dequantizado UMA vez para a LDS (K e V lado a lado) e compartilhado pelas
// 6 cabecas; softmax online em registradores fp32; skip causal por bloco;
// Q relido da global (sem persistencia na v1); acumuladores fp32; SEM WMMA.
//
// Transcricao verificada do prototipo medido em tests/bench_mmq_wmma_gpu.hip
// (secao H5): 2.72-2.79x do split-batch em T=512/K=2K-8K f16, rel-L2 <=7.5e-7
// e argmax 384/384-3072/3072 vs o batch de producao. Gate NUMERICO (a ordem
// das somas muda), nunca bit-exato -- mesma classe da tolerancia GEMM que
// ja embarca (kTolChunkRelL2 cobre o prefill ponta a ponta; PPL neutro).
//
// v1 = f16/f16 APENAS (q8_0 media 1.12-1.33x, nao paga um segundo caminho;
// T=128 e variante split-over-keys ficaram na mesa). Despacho fail-closed:
// fora da admissao (f16, n_q==512, keys 2048..8192, HD256/grupo6) o
// chamador usa o caminho incumbente.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

#include "rdna4/fp16.h"
#include "rdna4/kv.h"

namespace rdna4 {

template <KvType KT, KvType VT, int KTT, int HP, bool F32L, int QT>
__global__ void attn_fa2_prefill_kernel(const float *__restrict__ q, const void *__restrict__ k,
                                        const void *__restrict__ v, float *__restrict__ out,
                                        const int *__restrict__ pos, int n_head, int n_head_kv,
                                        int head_dim, float dscale, int n_q, int kmax,
                                        int keys_total) {
  extern __shared__ uint4 lds4[];  // [2*KTT][256] halves: K tile, then V tile
  const int kvh = blockIdx.x;
  const int qt = blockIdx.y;
  const int w = threadIdx.x >> 5;  // query slot 0..7 inside the tile
  const int lane = threadIdx.x & 31;
  const int tid = threadIdx.x;
  const int grp = n_head / n_head_kv;
  if (head_dim != 256 || grp != 6) return;  // prototype: this model's shape only
  constexpr int DPW = 8;
  const int qq = qt * QT + w;
  const bool active = (qq < n_q);
  const int mypos = active ? pos[qq] : -1;

  const std::uint64_t krowb = kv_row_bytes(KT, head_dim);
  const std::uint64_t vrowb = kv_row_bytes(VT, head_dim);
  const int ntiles = (kmax + 1 + KTT - 1) / KTT;
  const int vbase4 = KTT * 32;  // uint4 offset of the V tile (= KTT*256 halves)

  const int npass = (grp + HP - 1) / HP;
  for (int c = 0; c < npass; ++c) {
    const int hh0 = c * HP;
    int nh = grp - hh0;
    if (nh > HP) nh = HP;
    float qv[HP][8];
    float acc[HP][8];
    float m[HP], l[HP];
    if (active) {
      for (int hh = 0; hh < nh; ++hh) {
        const float *qp = q + ((std::int64_t)qq * n_head + kvh * grp + hh0 + hh) * head_dim;
        for (int i = 0; i < DPW; ++i) {
          qv[hh][i] = qp[lane * DPW + i];
          acc[hh][i] = 0.0f;
        }
        m[hh] = -INFINITY;
        l[hh] = 0.0f;
      }
    }
    for (int tl = 0; tl < ntiles; ++tl) {
      const int j0 = tl * KTT;
      int nload = keys_total - j0;
      if (nload > KTT) nload = KTT;
      // -- coop K tile: one uint4 (8 dims) per thread per iter; a warp covers
      // one full row contiguously, all 8 warps cover 8 distinct rows.
      // F32L=false (f16 KV): bit-copy halves K<->LDS (exact roundtrip).
      // Coop stride is the CTA width (QT*32 threads).
      for (int cc = tid; cc < nload * 32; cc += QT * 32) {
        const int j = cc / 32;
        const int l32 = cc % 32;
        const char *row = (const char *)k + ((std::int64_t)(j0 + j) * n_head_kv + kvh) * krowb;
        if constexpr (!F32L) {
          lds4[j * 32 + l32] = ((const uint4 *)row)[l32];
        } else {
          float tmp[8];
          kv_load8<KT>(row, l32, tmp);
          float *dst = (float *)lds4 + j * 256 + l32 * 8;
          for (int i = 0; i < 8; ++i) dst[i] = tmp[i];
        }
      }
      // -- coop V tile (same mapping, second half of LDS) --
      for (int cc = tid; cc < nload * 32; cc += QT * 32) {
        const int j = cc / 32;
        const int l32 = cc % 32;
        const char *row = (const char *)v + ((std::int64_t)(j0 + j) * n_head_kv + kvh) * vrowb;
        if constexpr (!F32L) {
          lds4[vbase4 + j * 32 + l32] = ((const uint4 *)row)[l32];
        } else {
          float tmp[8];
          kv_load8<VT>(row, l32, tmp);
          float *dst = (float *)lds4 + (std::size_t)vbase4 * 8 + j * 256 + l32 * 8;
          for (int i = 0; i < 8; ++i) dst[i] = tmp[i];
        }
      }
      __syncthreads();
      if (active) {
        for (int j = 0; j < KTT; ++j) {
          const int jk = j0 + j;
          if (jk > mypos) break;  // causal: keys ascend, later tiles skipped the
                                  // same way; syncs are outside this loop
          float kk[8], vv[8];
          if constexpr (!F32L) {
            const uint4 ku = lds4[j * 32 + lane];
            const uint4 vu = lds4[vbase4 + j * 32 + lane];
            const std::uint16_t kh[8] = {
                (std::uint16_t)(ku.x & 0xFFFF), (std::uint16_t)(ku.x >> 16),
                (std::uint16_t)(ku.y & 0xFFFF), (std::uint16_t)(ku.y >> 16),
                (std::uint16_t)(ku.z & 0xFFFF), (std::uint16_t)(ku.z >> 16),
                (std::uint16_t)(ku.w & 0xFFFF), (std::uint16_t)(ku.w >> 16)};
            const std::uint16_t vh[8] = {
                (std::uint16_t)(vu.x & 0xFFFF), (std::uint16_t)(vu.x >> 16),
                (std::uint16_t)(vu.y & 0xFFFF), (std::uint16_t)(vu.y >> 16),
                (std::uint16_t)(vu.z & 0xFFFF), (std::uint16_t)(vu.z >> 16),
                (std::uint16_t)(vu.w & 0xFFFF), (std::uint16_t)(vu.w >> 16)};
            for (int i = 0; i < 8; ++i) {
              kk[i] = fp16_to_float(kh[i]);
              vv[i] = fp16_to_float(vh[i]);
            }
          } else {
            const float *Lkf = (const float *)lds4;
            const float *Lvf = Lkf + (std::size_t)vbase4 * 8;
            for (int i = 0; i < 8; ++i) {
              kk[i] = Lkf[j * 256 + lane * 8 + i];
              vv[i] = Lvf[j * 256 + lane * 8 + i];
            }
          }
          for (int hh = 0; hh < nh; ++hh) {
            float dot = 0.0f;
            for (int i = 0; i < DPW; ++i) dot = fmaf(qv[hh][i], kk[i], dot);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
              dot += __shfl_xor_sync(0xffffffffull, dot, off);
            const float score = dot * dscale;
            if (score > m[hh]) {
              const float corr = (m[hh] == -INFINITY) ? 0.0f : expf(m[hh] - score);
              l[hh] *= corr;
              for (int i = 0; i < DPW; ++i) acc[hh][i] *= corr;
              m[hh] = score;
            }
            const float p = (m[hh] == -INFINITY) ? 0.0f : expf(score - m[hh]);
            l[hh] += p;
            for (int i = 0; i < DPW; ++i) acc[hh][i] = fmaf(p, vv[i], acc[hh][i]);
          }
        }
      }
      __syncthreads();
    }
    if (active) {
      for (int hh = 0; hh < nh; ++hh) {
        const float inv = (l[hh] > 0.0f) ? 1.0f / l[hh] : 0.0f;
        float *op = out + ((std::int64_t)qq * n_head + kvh * grp + hh0 + hh) * head_dim;
        for (int i = 0; i < DPW; ++i) op[lane * DPW + i] = acc[hh][i] * inv;
      }
    }
  }
}

// v1: f16/f16, KT64/QT16 APENAS (a config medida 2.72x). Outras instanciacoes
// do template acima existem para a bancada; a producao nao as despacha.
inline bool fa2_prefill_launch(const float *d_q, const void *d_k, const void *d_v, float *d_o,
                               const int *d_pos, int n_q, int kmax, int keys_total, int n_head,
                               int n_head_kv, int head_dim, float scale,
                               hipStream_t stream = nullptr) {
  if (head_dim != 256 || n_head_kv <= 0 || n_head / n_head_kv != 6 || n_head % n_head_kv != 0 ||
      n_q < 1 || kmax < 0 || keys_total <= 0) {
    return false;
  }
  const dim3 grid((unsigned)n_head_kv, (unsigned)((n_q + 16 - 1) / 16));
  const std::size_t smem = (std::size_t)2 * 64 * 256 * sizeof(std::uint16_t);  // 64 KiB
  attn_fa2_prefill_kernel<KvType::F16, KvType::F16, 64, 6, false, 16>
      <<<grid, (unsigned)16 * 32, smem, stream>>>(d_q, d_k, d_v, d_o, d_pos, n_head, n_head_kv,
                                                  head_dim, scale, n_q, kmax, keys_total);
  return hipGetLastError() == hipSuccess;
}

// Admissao fail-closed (v1): f16/f16, chunk cheio de 512 (forma medida;
// caudas <512 e T=16 usam o incumbente -- sem starvation), chaves
// 2048..8192 (fora disso, territorio nao medido), HD256/grupo6.
inline bool fa2_prefill_usable(KvType kt, KvType vt, int n_q, int pos0, int n_head, int n_head_kv,
                               int head_dim) {
  static const int on = [] {
    const char *e = std::getenv("RD_GFX12_FA2_PREFILL");
    return !(e && std::atoi(e) == 0);
  }();
  if (!on) return false;
  if (kt != KvType::F16 || vt != KvType::F16) return false;
  if (n_q != 512) return false;
  const int keys_total = pos0 + n_q;
  if (keys_total < 2048 || keys_total > 8192) return false;
  if (head_dim != 256 || n_head_kv <= 0 || n_head / n_head_kv != 6 || n_head % n_head_kv != 0) {
    return false;
  }
  return true;
}

}  // namespace rdna4
