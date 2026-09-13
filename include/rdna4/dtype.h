#pragma once
// rdna4-infer internal tensor dtype (PLAN.md M1 step 4).
//
// DType is the whitelist of tensor data types the engine supports: exactly the
// union of quant types present in the Qwen3.8-27B UD files (15 types,
// docs/exploracao-qwen38.md). The loader maps the raw ggml_type ids stored in
// GGUF tensor entries (TensorInfo::dtype) through dtype_from_ggml() and
// rejects anything outside the whitelist ("resto é erro").
//
// F16/Q4_0 are deliberately NOT in the whitelist: they exist only as engine-side
// KV cache types (device.h), never as GGUF tensor types in the UD files.
//
// ggml_type ids follow ggml/include/ggml.h (llama.cpp rev 790cf51, .ref/).

#include <cstdint>
#include <optional>

namespace rdna4 {

enum class DType : std::uint32_t {
  F32,       // ggml 0
  Q8_0,      // ggml 8
  Q2_K,      // ggml 10
  Q3_K,      // ggml 11
  Q4_K,      // ggml 12
  Q5_K,      // ggml 13
  Q6_K,      // ggml 14
  IQ2_XXS,   // ggml 16
  IQ2_XS,    // ggml 17
  IQ3_XXS,   // ggml 18
  IQ1_S,     // ggml 19
  IQ4_NL,    // ggml 20
  IQ3_S,     // ggml 21
  IQ2_S,     // ggml 22
  IQ4_XS,    // ggml 23
};

// Raw ggml_type id (GGUF tensor entry) -> internal dtype.
// nullopt for every id outside the UD whitelist, including ids of ggml types
// the engine does not support (F16, Q4_0, TQ1_0, ...).
inline std::optional<DType> dtype_from_ggml(std::uint32_t ggml_type) {
  switch (ggml_type) {
    case 0:  return DType::F32;       // GGML_TYPE_F32
    case 8:  return DType::Q8_0;      // GGML_TYPE_Q8_0
    case 10: return DType::Q2_K;      // GGML_TYPE_Q2_K
    case 11: return DType::Q3_K;      // GGML_TYPE_Q3_K
    case 12: return DType::Q4_K;      // GGML_TYPE_Q4_K
    case 13: return DType::Q5_K;      // GGML_TYPE_Q5_K
    case 14: return DType::Q6_K;      // GGML_TYPE_Q6_K
    case 16: return DType::IQ2_XXS;   // GGML_TYPE_IQ2_XXS
    case 17: return DType::IQ2_XS;    // GGML_TYPE_IQ2_XS
    case 18: return DType::IQ3_XXS;   // GGML_TYPE_IQ3_XXS
    case 19: return DType::IQ1_S;     // GGML_TYPE_IQ1_S
    case 20: return DType::IQ4_NL;    // GGML_TYPE_IQ4_NL
    case 21: return DType::IQ3_S;     // GGML_TYPE_IQ3_S
    case 22: return DType::IQ2_S;     // GGML_TYPE_IQ2_S
    case 23: return DType::IQ4_XS;    // GGML_TYPE_IQ4_XS
    default: return std::nullopt;
  }
}

// Block layout per type: elements per block, bytes per block.
// Verified against the UD files (scripts/check_geometry.py: every tensor size
// fits its on-disk span, alignment holds, last tensor ends exactly at EOF) and
// cross-checked with the llama.cpp b10902 reference dump (llama-gguf r).
// The .ref snapshot (790cf51) is mid-refactor and self-inconsistent for
// IQ1_S (struct 66B vs its own static_assert 50B); it is NOT trusted for
// file format. The on-disk IQ1_S block is 50B (1.5625 bpw).
inline std::uint32_t dtype_block_elems(DType t) {
  switch (t) {
    case DType::F32: return 1;
    case DType::Q8_0: return 32;
    case DType::IQ4_NL: return 32;
    default: return 256;  // all K-quants and the remaining IQ types
  }
}

inline std::uint32_t dtype_block_bytes(DType t) {
  switch (t) {
    case DType::F32: return 4;      // 4.0 bpw
    case DType::Q8_0: return 34;    // 8.5
    case DType::Q2_K: return 84;    // 2.625
    case DType::Q3_K: return 110;   // 3.4375
    case DType::Q4_K: return 144;   // 4.5
    case DType::Q5_K: return 176;   // 5.5
    case DType::Q6_K: return 210;   // 6.5625
    case DType::IQ2_XXS: return 66; // 2.0625
    case DType::IQ2_XS: return 74;  // 2.3125
    case DType::IQ3_XXS: return 98; // 3.0625
    case DType::IQ1_S: return 50;   // 1.5625
    case DType::IQ4_NL: return 18;  // 4.5 (32-elem blocks)
    case DType::IQ3_S: return 110;  // 3.4375
    case DType::IQ2_S: return 82;   // 2.5625
    case DType::IQ4_XS: return 136; // 4.25
  }
  return 0;
}

// Byte size of a tensor with `nelem` elements of type `t` (ceil to whole
// blocks, per ggml_row_size).
inline std::uint64_t tensor_bytes(DType t, std::uint64_t nelem) {
  const std::uint64_t be = dtype_block_elems(t);
  const std::uint64_t blocks = (nelem + be - 1) / be;
  return blocks * dtype_block_bytes(t);
}

inline const char *dtype_name(DType t) {
  switch (t) {
    case DType::F32: return "f32";
    case DType::Q8_0: return "q8_0";
    case DType::Q2_K: return "q2_k";
    case DType::Q3_K: return "q3_k";
    case DType::Q4_K: return "q4_k";
    case DType::Q5_K: return "q5_k";
    case DType::Q6_K: return "q6_k";
    case DType::IQ2_XXS: return "iq2_xxs";
    case DType::IQ2_XS: return "iq2_xs";
    case DType::IQ3_XXS: return "iq3_xxs";
    case DType::IQ1_S: return "iq1_s";
    case DType::IQ4_NL: return "iq4_nl";
    case DType::IQ3_S: return "iq3_s";
    case DType::IQ2_S: return "iq2_s";
    case DType::IQ4_XS: return "iq4_xs";
  }
  return "?";
}

}  // namespace rdna4
