// Pins the engine's end-of-generation set against llama.cpp (PLAN.md M4).
//
// The CLI must stop on exactly the tokens llama.cpp's `llama_vocab_is_eog`
// reports — not merely on `tokenizer.ggml.eos_token_id`, which would keep
// generating after `<|endoftext|>`. The reference set comes from
// tests/oracle_eog.cpp (libllama) and is committed as tests/golden/eog_ids.txt,
// so a tokenizer change that silently narrows the stop set fails here.
//
// usage: check-eog <file.gguf> <golden_eog.txt>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "rdna4/loader.h"
#include "rdna4/tokenizer.h"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: check-eog <file.gguf> <golden_eog.txt>\n");
    return 2;
  }
  rdna4::GgufLoader loader;
  std::string err;
  if (!loader.open(argv[1], err)) {
    std::fprintf(stderr, "gguf load failed: %s\n", err.c_str());
    return 1;
  }
  rdna4::Tokenizer tk;
  if (!tk.init(loader.meta(), err)) {
    std::fprintf(stderr, "tokenizer init failed: %s\n", err.c_str());
    return 1;
  }

  std::ifstream f(argv[2]);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", argv[2]);
    return 1;
  }
  std::set<std::int32_t> ref;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    ref.insert((std::int32_t)std::strtol(line.c_str(), nullptr, 10));
  }
  if (ref.empty()) {
    std::fprintf(stderr, "FAIL: %s holds no eog ids (empty oracle)\n", argv[2]);
    return 1;
  }

  std::set<std::int32_t> ours;
  for (std::int32_t id = 0; id < (std::int32_t)tk.n_vocab(); ++id) {
    if (tk.is_eog(id)) ours.insert(id);
  }

  int failures = 0;
  std::printf("eog tokens: reference %zu, ours %zu\n", ref.size(), ours.size());
  for (std::int32_t id : ref) {
    if (!ours.count(id)) {
      std::printf("FAIL: %d ('%s') is EOG in llama.cpp but not here\n", id,
                  tk.token_text(id).c_str());
      ++failures;
    }
  }
  for (std::int32_t id : ours) {
    if (!ref.count(id)) {
      std::printf("FAIL: %d ('%s') is EOG here but not in llama.cpp\n", id,
                  tk.token_text(id).c_str());
      ++failures;
    }
  }
  if (failures == 0) {
    std::printf("ids:");
    for (std::int32_t id : ours) std::printf(" %d", id);
    std::printf("\ncheck-eog: OK\n");
  } else {
    std::printf("check-eog: FAILED (%d difference(s))\n", failures);
  }
  return failures ? 1 : 0;
}
