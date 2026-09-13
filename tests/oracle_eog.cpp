// Test tool — llama.cpp's own end-of-generation token set for this vocab.
//
// `llama_vocab_is_eog()` is what makes llama-cli stop, and it is *not* just the
// EOS token: llama.cpp also marks every entry whose text looks like a control
// end-of-turn token (`<|im_end|>`, `<|endoftext|>`, `</s>`, ...). For the qwen35
// vocab that is 248046 (`<|im_end|>`) *and* 248044 (`<|endoftext|>`), so an
// engine that only checks EOS keeps generating where the reference stops
// (review M4, MAJOR). This tool prints the reference set so a test can pin it.
//
// usage: oracle-eog <file.gguf>
// Output: one line per EOG token: "<id> <piece>"
#include <llama.h>

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: oracle-eog <file.gguf>\n");
    return 2;
  }
  llama_backend_init();
  llama_model_params mparams = llama_model_default_params();
  mparams.vocab_only = true;
  llama_model *model = llama_model_load_from_file(argv[1], mparams);
  if (!model) {
    std::fprintf(stderr, "load failed\n");
    return 1;
  }
  const llama_vocab *vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);
  int n = 0;
  for (int id = 0; id < n_vocab; ++id) {
    if (!llama_vocab_is_eog(vocab, id)) continue;
    char buf[256] = {};
    const int len = llama_token_to_piece(vocab, id, buf, (int)sizeof(buf) - 1, 0, true);
    std::string piece = len > 0 ? std::string(buf, (std::size_t)len) : std::string();
    std::printf("%d %s\n", id, piece.c_str());
    ++n;
  }
  std::printf("# %d eog token(s) of %d\n", n, n_vocab);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
