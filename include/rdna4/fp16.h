#pragma once
// Device-side fp16 -> float conversion (HIP, no ROCm header dependency).
// Handles normal and subnormal fp16 values exactly.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace rdna4 {

// Convert a 16-bit IEEE754 fp16 (little-endian bit layout) to float32.
__device__ __forceinline__ float fp16_to_float(uint16_t h) {
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
}

}  // namespace rdna4
