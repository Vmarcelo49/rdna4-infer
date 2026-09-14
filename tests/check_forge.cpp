// Loader hardening probe (CPU-only, audit finding M7).
//
// Opens the file it is given, and if the loader accepts it, reads the named
// tensor back and prints its first bytes. On a forged header whose tensor offset
// wraps, the pre-fix loader accepted the file and returned *header* bytes as
// weights, which is what this tool makes visible: the head of the load comes
// back as ASCII from the KV section instead of the model's data.
//
//   check-forge <file.gguf> <tensor-name>
//
// Exit status: 0 when the loader rejected the file (the expected result for a
// forged header on the hardened loader), 1 when the file was accepted, 2 on a
// usage error. The accepted case is not a failure of this tool: it is the
// evidence, and scripts/check_hardening.sh reads the printed head bytes.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "rdna4/loader.h"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: check-forge <file.gguf> <tensor-name>\n");
    return 2;
  }
  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    std::printf("rejected: %s\n", err.c_str());
    return 0;
  }
  const auto *t = ld.find(argv[2]);
  if (!t) {
    std::printf("accepted (tensor %s not found)\n", argv[2]);
    return 1;
  }
  const std::size_t i = static_cast<std::size_t>(t - ld.meta().tensors.data());
  std::vector<std::uint8_t> raw;
  if (!ld.load_tensor(i, raw, err)) {
    std::printf("accepted, load failed: %s\n", err.c_str());
    return 1;
  }
  std::printf("accepted: %s offset=%llu loaded %llu bytes, head:", argv[2],
              (unsigned long long)t->offset, (unsigned long long)raw.size());
  for (std::size_t k = 0; k < 16 && k < raw.size(); ++k) {
    std::printf(" %02x", raw[k]);
  }
  std::printf("  ascii:");
  for (std::size_t k = 0; k < 16 && k < raw.size(); ++k) {
    std::printf("%c", raw[k] >= 32 && raw[k] < 127 ? static_cast<char>(raw[k]) : '.');
  }
  if (raw.size() >= sizeof(float)) {
    float f = 0.0f;
    std::memcpy(&f, raw.data(), sizeof(float));
    std::printf("  f32[0]=%.6g\n", static_cast<double>(f));
  } else {
    std::printf("\n");
  }
  return 1;
}
