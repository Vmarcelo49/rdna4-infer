#pragma once
// Device-side fp16 -> float conversion (HIP, no ROCm header dependency).
// Handles normal and subnormal fp16 values exactly.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace rdna4 {

// Convert a 16-bit IEEE754 fp16 to float32.
//
// On device this uses the hardware conversion (V_CVT_F32_F16): the software
// bit-twiddling form costs ~9 instructions and every vec_dot calls this twice
// per 32 elements (research finding: ~4% of the IQ kernels). The host path
// keeps the portable implementation, which is what the test references use.
__host__ __device__ __forceinline__ float fp16_to_float(uint16_t h) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __half2float(__ushort_as_half(h));
#else
  unsigned sign = (h >> 15) & 1u;
  unsigned e    = (h >> 10) & 0x1fu;
  unsigned m    = h & 0x3ffu;
  if (e == 0u) {
    // Subnormal: sign * 2^(-14) * (m / 1024)
    float v = static_cast<float>(m) * 5.9604644775390625e-8f; // 2^-24
    return sign ? -v : v;
  }
  // Normal: bias 15 -> float bias 127 => exp = e + 112
  union { unsigned u; float f; } conv;
  conv.u = (sign << 31) | ((e + 112u) << 23) | (m << 13);
  return conv.f;
#endif
}


// Convert a float32 to fp16 bits, round-to-nearest-even.
__host__ __device__ __forceinline__ uint16_t float_to_fp16(float f) {
    union { float f; unsigned u; } conv;
    conv.f = f;
    const unsigned x = conv.u;
    const unsigned sign = (x >> 16) & 0x8000u;
    unsigned mant = x & 0x007fffffu;
    const int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;          // underflow -> +-0
        mant |= 0x00800000u;                            // implicit bit
        const unsigned shift = (unsigned)(14 - exp);
        unsigned half_mant = mant >> shift;
        const unsigned round_bit = 1u << (shift - 1);
        if ((mant & round_bit) && ((mant & (round_bit - 1)) || (half_mant & 1u))) ++half_mant;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);   // overflow -> inf
    unsigned half = sign | ((unsigned)exp << 10) | (mant >> 13);
    if ((mant & 0x1fffu) > 0x1000u || ((mant & 0x1fffu) == 0x1000u && (half & 1u))) ++half;
    return (uint16_t)half;
}

}  // namespace rdna4
