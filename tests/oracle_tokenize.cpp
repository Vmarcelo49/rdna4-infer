// Test tool — llama.cpp's own tokenization for a corpus of strings.
//
// The engine's BPE tokenizer (include/rdna4/tokenizer.h) must produce exactly
// these ids: tokenization is not a place where "close" is acceptable, because a
// single different id changes the prompt the model sees.
//
// usage: oracle-tokenize <file.gguf> [corpus.txt]
//   corpus.txt: one test string per line, with \n, \t and \\ escaped (see
//               tests/tokenizer_corpus.txt). With no file the built-in list is
//               used.
// Output: "<line index>\t<id> <id> ..." per line, plus a trailing "# ok".
#include <llama.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

std::string unescape(const std::string &in) {
  std::string out;
  for (std::size_t i = 0; i < in.size(); ++i) {
    if (in[i] == '\\' && i + 1 < in.size()) {
      const char c = in[++i];
      if (c == 'n') out.push_back('\n');
      else if (c == 't') out.push_back('\t');
      else if (c == 'r') out.push_back('\r');
      else if (c == '\\') out.push_back('\\');
      else { out.push_back('\\'); out.push_back(c); }
    } else {
      out.push_back(in[i]);
    }
  }
  return out;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: oracle-tokenize <file.gguf> [corpus.txt]\n");
    return 2;
  }

  std::vector<std::string> corpus = {
      "Hello world, this is a test.",
      "The capital of France is",
      "def fibonacci(n):",
      "1 2 3 4 5 6 7 8 9",
      "Hello",
      " don't stop believin'",
      "café naïve résumé",
      "日本語のテキスト",
      "emoji: 🚀🔥",
      "  leading and trailing  ",
      "line1\nline2\n\nline3",
      "tab\tseparated\tvalues",
      "x = (a + b) * c / d[0] - 1.5e-3",
      "int main(void) { return 0; }",
      "<|im_start|>user\nHello<|im_end|>\n",
      "<|endoftext|>",
      "MiXeD CaSe WOULDn't",
      "12345678901234567890",
      "https://example.com/path?q=1",
      "a",
      " ",
      "  ",
      "\n",
      "αβγδε ΑΒΓΔΕ",
      "Ünïcödé with combĩning",
  };
  if (argc >= 3) {
    std::FILE *fp = std::fopen(argv[2], "r");
    if (!fp) {
      std::fprintf(stderr, "cannot open corpus %s\n", argv[2]);
      return 1;
    }
    corpus.clear();
    char buf[65536];
    while (std::fgets(buf, sizeof(buf), fp)) {
      std::string line(buf);
      while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
      if (line.empty()) continue;
      corpus.push_back(unescape(line));
    }
    std::fclose(fp);
  }

  llama_backend_init();
  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 0;
  mparams.vocab_only = true;
  llama_model *model = llama_model_load_from_file(argv[1], mparams);
  if (!model) {
    std::fprintf(stderr, "load failed\n");
    return 1;
  }
  const llama_vocab *vocab = llama_model_get_vocab(model);

  std::printf("# n_vocab %d\n", llama_vocab_n_tokens(vocab));
  for (std::size_t i = 0; i < corpus.size(); ++i) {
    const std::string &text = corpus[i];
    // llama_tokenize returns the number of tokens, or a *negative* value
    // -N when the buffer was too small, where N is the required size.
    int32_t need =
        llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), nullptr, 0, false, true);
    if (need < 0) need = -need;
    if (need == 0) need = 1;
    std::vector<llama_token> ids((std::size_t)need);
    const int32_t got =
        llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), ids.data(), need, false, true);
    if (got < 0) {
      std::fprintf(stderr, "tokenize failed (2nd pass) for line %zu (%d)\n", i, (int)got);
      return 1;
    }
    ids.resize((std::size_t)got);
    std::printf("%zu\t", i);
    for (std::size_t k = 0; k < ids.size(); ++k) {
      std::printf("%s%d", k ? " " : "", (int)ids[k]);
    }
    std::printf("\n");
  }
  std::printf("# ok\n");

  llama_model_free(model);
  llama_backend_free();
  return 0;
}
