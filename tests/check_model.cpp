// CPU-only qwen35 config + layer-layout validator (PLAN.md M1 items 2-3).
// No tensor data is loaded, so this also runs on tiny synthetic GGUF files
// (see tests/gen_fake_qwen35.py) to exercise the fail-fast paths.
//
//   check-model <file.gguf>            -> validate, exit 0 on OK
//   check-model <file.gguf> --expect-fail
//                                        -> exit 0 only if validation fails
#include "rdna4/loader.h"
#include "rdna4/model.h"

#include <cstdio>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-model <file.gguf> [--expect-fail]\n");
    return 2;
  }
  bool expect_fail = false;
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "--expect-fail") {
      expect_fail = true;
    }
  }
  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    if (expect_fail) {
      std::printf("check-model: open failed as expected (%s)\n", err.c_str());
      return 0;
    }
    std::fprintf(stderr, "check-model: open failed: %s\n", err.c_str());
    return 1;
  }
  rdna4::Qwen35Config cfg;
  bool ok = rdna4::parse_qwen35_config(ld.meta(), cfg, err);
  if (ok) {
    ok = rdna4::validate_qwen35_layout(ld, cfg, err);
  }
  if (ok == expect_fail) {
    std::fprintf(stderr, "check-model: UNEXPECTED: %s\n", err.c_str());
    return 1;
  }
  if (ok) {
    const unsigned main_blocks = cfg.block_count - 1;
    std::printf("check-model: OK  %u blocks (%u full-attn, %u GDN, 1 MTP) "
                "emb=%llu ffn=%llu ctx=%llu\n",
                cfg.block_count, main_blocks / cfg.full_attention_interval,
                main_blocks - main_blocks / cfg.full_attention_interval,
                (unsigned long long)cfg.embedding_length,
                (unsigned long long)cfg.feed_forward_length,
                (unsigned long long)cfg.context_length);
  } else {
    std::printf("check-model: rejected as expected: %s\n", err.c_str());
  }
  return 0;
}
