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
