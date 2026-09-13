// M2 — CPU dequant oracle bridge (C++ TU, links llama.cpp's libggml-base.so).
//
// Isolates the llama.cpp headers (ggml-quants.h / ggml-common.h) from the HIP
// translation unit, exposing C-linkage entry points used by the GPU test.
//
// Two entry points:
//   rdna4_cpu_dequant_row()  — reference dequant (llama.cpp's dequantize_row_*)
//   rdna4_audit_layout()     — field-offset/size audit of OUR block structs
//                              (rdna4/quants.h) against llama.cpp's originals.
//                              Guards the silent-corruption class of bug where
//                              a struct keeps its size but reorders fields
//                              (this actually happened for block_q5_K: qh/qs).
#include "rdna4/quants.h"

#include "ggml-quants.h"

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

int g_mismatches = 0;

void report(char *buf, int bufsize, const char *fmt, ...) {
  char line[256];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(line, sizeof(line), fmt, ap);
  va_end(ap);
  ++g_mismatches;
  if (buf && bufsize > 0) {
    std::strncat(buf, line, (std::size_t)bufsize - std::strlen(buf) - 1);
    std::strncat(buf, "\n", (std::size_t)bufsize - std::strlen(buf) - 1);
  }
}

#define AUDIT(T, THEIR, field)                                                        \
  do {                                                                                \
    if (sizeof(rdna4::T) != sizeof(THEIR)) {                                          \
      report(buf, bufsize, "%-14s sizeof: ours=%zu theirs=%zu", #T, sizeof(rdna4::T), \
             sizeof(THEIR));                                                          \
    }                                                                                 \
    if (offsetof(rdna4::T, field) != offsetof(THEIR, field)) {                        \
      report(buf, bufsize, "%-14s offsetof(%s): ours=%zu theirs=%zu", #T, #field,     \
             offsetof(rdna4::T, field), offsetof(THEIR, field));                      \
    }                                                                                 \
  } while (0)

}  // namespace


extern "C" bool rdna4_cpu_dequant_row(int dt, const void *src, float *dst, std::int64_t nelem) {
  switch (dt) {
    // dt values are rdna4::DType ordinals (see include/rdna4/dtype.h).
    case 1:  // Q8_0
      dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(src), dst, nelem);
      return true;
    case 2:  // Q2_K
      dequantize_row_q2_K(reinterpret_cast<const block_q2_K *>(src), dst, nelem);
      return true;
    case 3:  // Q3_K
      dequantize_row_q3_K(reinterpret_cast<const block_q3_K *>(src), dst, nelem);
      return true;
    case 4:  // Q4_K
      dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(src), dst, nelem);
      return true;
    case 5:  // Q5_K
      dequantize_row_q5_K(reinterpret_cast<const block_q5_K *>(src), dst, nelem);
      return true;
    case 6:  // Q6_K
      dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(src), dst, nelem);
      return true;
    case 7:  // IQ2_XXS
      dequantize_row_iq2_xxs(reinterpret_cast<const block_iq2_xxs *>(src), dst, nelem);
      return true;
    case 8:  // IQ2_XS
      dequantize_row_iq2_xs(reinterpret_cast<const block_iq2_xs *>(src), dst, nelem);
      return true;
    case 9:  // IQ3_XXS
      dequantize_row_iq3_xxs(reinterpret_cast<const block_iq3_xxs *>(src), dst, nelem);
      return true;
    case 10:  // IQ1_S
      dequantize_row_iq1_s(reinterpret_cast<const block_iq1_s *>(src), dst, nelem);
      return true;
    case 11:  // IQ4_NL
      dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl *>(src), dst, nelem);
      return true;
    case 12:  // IQ3_S
      dequantize_row_iq3_s(reinterpret_cast<const block_iq3_s *>(src), dst, nelem);
      return true;
    case 13:  // IQ2_S
      dequantize_row_iq2_s(reinterpret_cast<const block_iq2_s *>(src), dst, nelem);
      return true;
    case 14:  // IQ4_XS
      dequantize_row_iq4_xs(reinterpret_cast<const block_iq4_xs *>(src), dst, nelem);
      return true;
    default:
      return false;
  }
}

// Independent reference for the activation quantization: llama.cpp's own
// quantize_row_q8_1_ref (ggml-quants.c), exported by libggml-base. Without
// this, the matvec test fed the CPU side the GPU's own activation bytes, so no
// test could ever have caught a bug in quantize_q8_1_block.
extern "C" void rdna4_cpu_quantize_q8_1(const float *x, void *vy, std::int64_t nelem) {
  quantize_row_q8_1_ref(x, reinterpret_cast<block_q8_1 *>(vy), nelem);
}

// Compares every field offset (and size) of our block structs against
// llama.cpp's originals. Returns the number of mismatches; a human-readable
// description of each is appended to `buf`.
extern "C" int rdna4_audit_layout(char *buf, int bufsize) {
  g_mismatches = 0;
  if (buf && bufsize > 0) buf[0] = '\0';

  AUDIT(block_q8_0, ::block_q8_0, d);
  AUDIT(block_q8_0, ::block_q8_0, qs);

  AUDIT(block_q2_K, ::block_q2_K, scales);
  AUDIT(block_q2_K, ::block_q2_K, qs);
  AUDIT(block_q2_K, ::block_q2_K, dm);

  AUDIT(block_q3_K, ::block_q3_K, hmask);
  AUDIT(block_q3_K, ::block_q3_K, qs);
  AUDIT(block_q3_K, ::block_q3_K, scales);
  AUDIT(block_q3_K, ::block_q3_K, d);

  AUDIT(block_q4_K, ::block_q4_K, dm);
  AUDIT(block_q4_K, ::block_q4_K, scales);
  AUDIT(block_q4_K, ::block_q4_K, qs);

  AUDIT(block_q5_K, ::block_q5_K, dm);
  AUDIT(block_q5_K, ::block_q5_K, scales);
  AUDIT(block_q5_K, ::block_q5_K, qh);
  AUDIT(block_q5_K, ::block_q5_K, qs);

  AUDIT(block_q6_K, ::block_q6_K, ql);
  AUDIT(block_q6_K, ::block_q6_K, qh);
  AUDIT(block_q6_K, ::block_q6_K, scales);
  AUDIT(block_q6_K, ::block_q6_K, d);

  AUDIT(block_iq2_xxs, ::block_iq2_xxs, d);
  AUDIT(block_iq2_xxs, ::block_iq2_xxs, qs);

  AUDIT(block_iq2_xs, ::block_iq2_xs, d);
  AUDIT(block_iq2_xs, ::block_iq2_xs, qs);
  AUDIT(block_iq2_xs, ::block_iq2_xs, scales);

  AUDIT(block_iq2_s, ::block_iq2_s, d);
  AUDIT(block_iq2_s, ::block_iq2_s, qs);
  AUDIT(block_iq2_s, ::block_iq2_s, qh);
  AUDIT(block_iq2_s, ::block_iq2_s, scales);

  AUDIT(block_iq3_xxs, ::block_iq3_xxs, d);
  AUDIT(block_iq3_xxs, ::block_iq3_xxs, qs);

  AUDIT(block_iq3_s, ::block_iq3_s, d);
  AUDIT(block_iq3_s, ::block_iq3_s, qs);
  AUDIT(block_iq3_s, ::block_iq3_s, qh);
  AUDIT(block_iq3_s, ::block_iq3_s, signs);
  AUDIT(block_iq3_s, ::block_iq3_s, scales);

  AUDIT(block_iq1_s, ::block_iq1_s, d);
  AUDIT(block_iq1_s, ::block_iq1_s, qs);
  AUDIT(block_iq1_s, ::block_iq1_s, qh);

  AUDIT(block_iq4_nl, ::block_iq4_nl, d);
  AUDIT(block_iq4_nl, ::block_iq4_nl, qs);

  // Activation block: our uint32 ds must sit exactly where llama.cpp's
  // anonymous union {struct{d,s} data; half2 ds;} puts it.
  if (sizeof(rdna4::block_q8_1) != sizeof(::block_q8_1)) {
    report(buf, bufsize, "%-14s sizeof: ours=%zu theirs=%zu", "block_q8_1",
           sizeof(rdna4::block_q8_1), sizeof(::block_q8_1));
  }
  if (offsetof(rdna4::block_q8_1, ds) != offsetof(::block_q8_1, ds) ||
      offsetof(rdna4::block_q8_1, qs) != offsetof(::block_q8_1, qs)) {
    report(buf, bufsize, "%-14s offsetof: ds/qs mismatch", "block_q8_1");
  }

  AUDIT(block_iq4_xs, ::block_iq4_xs, d);
  AUDIT(block_iq4_xs, ::block_iq4_xs, scales_h);
  AUDIT(block_iq4_xs, ::block_iq4_xs, scales_l);
  AUDIT(block_iq4_xs, ::block_iq4_xs, qs);

  return g_mismatches;
}
