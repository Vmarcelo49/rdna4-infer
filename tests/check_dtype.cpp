// M1 step 1 acceptance check: every tensor in a UD file must map 1:1 onto the
// 15-type DType whitelist (PLAN.md M1 step 4, "rest é erro").
//
// CPU-only (no GPU/HIP needed). Usage: check-dtype <model.gguf>
// Exit: 0 = ok, 1 = read/whitelist failure, 2 = usage.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "rdna4/dtype.h"
#include "rdna4/gguf.h"

namespace {

constexpr rdna4::DType kAll[] = {
    rdna4::DType::F32, rdna4::DType::Q8_0, rdna4::DType::Q2_K, rdna4::DType::Q3_K,
    rdna4::DType::Q4_K, rdna4::DType::Q5_K, rdna4::DType::Q6_K, rdna4::DType::IQ2_XXS,
    rdna4::DType::IQ2_XS, rdna4::DType::IQ3_XXS, rdna4::DType::IQ1_S, rdna4::DType::IQ4_NL,
    rdna4::DType::IQ3_S, rdna4::DType::IQ2_S, rdna4::DType::IQ4_XS,
};
constexpr std::size_t kCount = sizeof(kAll) / sizeof(kAll[0]);

}  // namespace

int main(int argc, char **argv) {
  bool require_all = false;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--require-all") == 0) require_all = true;
  }
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-dtype <model.gguf> [--require-all]\n");
    return 2;
  }
  rdna4::gguf::File f;
  std::string err;
  if (!rdna4::gguf::read(argv[1], f, err)) {
    std::fprintf(stderr, "check-dtype: gguf read failed: %s\n", err.c_str());
    return 1;
  }
  if (rdna4::gguf::kv_str(f, "general.architecture") != "qwen35") {
    std::fprintf(stderr, "check-dtype: general.architecture != qwen35\n");
    return 1;
  }

  std::uint64_t count[kCount] = {};
  for (const auto &t : f.tensors) {
    const auto dt = rdna4::dtype_from_ggml(t.dtype);
    if (!dt) {
      std::fprintf(stderr, "check-dtype: tensor '%s': ggml_type id %u (%s) outside whitelist\n",
                   t.name.c_str(), t.dtype, rdna4::gguf::ggml_type_name(t.dtype));
      return 1;
    }
    for (std::size_t i = 0; i < kCount; ++i) {
      if (kAll[i] == *dt) {
        ++count[i];
      }
    }
  }

  std::size_t present = 0;
  std::string missing;
  for (std::size_t i = 0; i < kCount; ++i) {
    if (count[i] > 0) {
      ++present;
      std::printf("  %-8s %llu\n", rdna4::dtype_name(kAll[i]),
                  (unsigned long long)count[i]);
    } else {
      if (!missing.empty()) missing += ", ";
      missing += rdna4::dtype_name(kAll[i]);
    }
  }
  std::printf("tensors: %llu, types present: %zu/%zu\n",
              (unsigned long long)f.tensors.size(), present, kCount);
  // Report absent types explicitly: a dtype-mapping regression that turned a
  // type into "absent" used to look like success here (review finding L5).
  if (!missing.empty()) {
    std::printf("  absent: %s\n", missing.c_str());
  }
  // --require-all enforces the PLAN claim "15/15 tipos presentes no IQ3_S";
  // it is opt-in because files legitimately lack some types (IQ4_XS has no
  // iq2_xxs / iq1_s).
  if (require_all && present != kCount) {
    std::fprintf(stderr, "check-dtype: FAILED (%zu/%zu types present, --require-all)\n", present,
                 kCount);
    return 1;
  }
  std::printf("check-dtype: OK\n");
  return 0;
}
