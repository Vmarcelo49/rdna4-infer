// M4 — the BPE tokenizer against llama.cpp's own tokenizer.
//
// Tokenization is exact or it is wrong: one different id changes the prompt the
// model sees. This test runs the same corpus through our Tokenizer and through
// llama.cpp (tests/oracle_tokenize.cpp, linked against libllama) and requires
// the id sequences to be identical, then round-trips decode(encode(text)).
//
// usage: check-tokenizer <file.gguf> <oracle_tokenize_output.txt> [corpus.txt]
//   (regenerate the oracle with: oracle-tokenize <gguf> [corpus.txt])
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "rdna4/loader.h"
#include "rdna4/tokenizer.h"

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

// Same built-in corpus as the oracle tool (used when no corpus file is given).
std::vector<std::string> default_corpus() {
  return {
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
}

std::vector<std::string> read_corpus(const char *path) {
  std::vector<std::string> out;
  std::ifstream f(path);
  if (!f) return out;
  std::string line;
  while (std::getline(f, line)) {
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    if (line.empty()) continue;
    out.push_back(unescape(line));
  }
  return out;
}

struct Oracle {
  int n_vocab = 0;
  std::vector<std::vector<std::int32_t>> ids;
};

Oracle read_oracle(const char *path) {
  Oracle o;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("# n_vocab ", 0) == 0) {
      o.n_vocab = std::atoi(line.c_str() + 10);
      continue;
    }
    if (line.rfind("#", 0) == 0) continue;
    const std::size_t tab = line.find('\t');
    if (tab == std::string::npos) continue;
    const std::size_t idx = (std::size_t)std::atol(line.c_str());
    std::vector<std::int32_t> ids;
    const char *p = line.c_str() + tab + 1;
    while (*p) {
      char *end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) {
        ++p;
        continue;
      }
      ids.push_back((std::int32_t)v);
      p = end;
    }
    if (o.ids.size() <= idx) o.ids.resize(idx + 1);
    o.ids[idx] = std::move(ids);
  }
  return o;
}

std::string join(const std::vector<std::int32_t> &ids, std::size_t limit = 24) {
  std::string s;
  for (std::size_t i = 0; i < ids.size() && i < limit; ++i) {
    s += (i ? " " : "") + std::to_string(ids[i]);
  }
  if (ids.size() > limit) s += " ...";
  return s;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: check-tokenizer <file.gguf> <oracle_output.txt> [corpus.txt]\n");
    return 2;
  }

  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }
  rdna4::Tokenizer tok;
  if (!tok.init(ld.meta(), err)) {
    std::fprintf(stderr, "tokenizer init: %s\n", err.c_str());
    return 1;
  }

  const std::vector<std::string> corpus =
      argc >= 4 ? read_corpus(argv[3]) : default_corpus();
  const Oracle oracle = read_oracle(argv[2]);

  int failures = 0;
  std::printf("vocab: ours %zu, llama.cpp %d; corpus %zu strings\n", tok.n_vocab(), oracle.n_vocab,
              corpus.size());
  if (oracle.n_vocab != 0 && (std::size_t)oracle.n_vocab != tok.n_vocab()) {
    std::printf("FAIL vocab size mismatch\n");
    ++failures;
  }

  int encoded_mismatch = 0;
  int roundtrip_mismatch = 0;
  for (std::size_t i = 0; i < corpus.size(); ++i) {
    const std::vector<std::int32_t> got = tok.encode(corpus[i]);
    // A missing oracle line is a failure, not a skip: silently comparing
    // nothing is how a broken oracle passes as "0 mismatches".
    if (i >= oracle.ids.size() || oracle.ids[i].empty()) {
      std::printf("MISSING ORACLE line %zu (corpus has %zu, oracle has %zu)\n", i, corpus.size(),
                  oracle.ids.size());
      ++encoded_mismatch;
      continue;
    }
    {
      const std::vector<std::int32_t> &want = oracle.ids[i];
      if (got != want) {
        if (encoded_mismatch < 5) {
          std::printf("ENCODE MISMATCH line %zu: %s\n  ours: %s\n  ref : %s\n", i,
                      corpus[i].c_str(), join(got).c_str(), join(want).c_str());
        }
        ++encoded_mismatch;
      }
    }
    // decode(encode(x)) == x for text without special tokens that we would
    // re-encode differently (the corpus has none of those except the explicit
    // special-token strings, which round-trip as their own text).
    const std::string back = tok.decode(got);
    if (back != corpus[i]) {
      if (roundtrip_mismatch < 5) {
        std::printf("ROUNDTRIP MISMATCH line %zu:\n  in : %s\n  out: %s\n", i, corpus[i].c_str(),
                    back.c_str());
      }
      ++roundtrip_mismatch;
    }
  }
  std::printf("encode: %zu strings, %d mismatches (bit-exact vs llama.cpp)\n", corpus.size(),
              encoded_mismatch);
  std::printf("roundtrip decode(encode(x)) == x: %d mismatches\n", roundtrip_mismatch);
  failures += encoded_mismatch + roundtrip_mismatch;

  // special tokens: id lookups used by the chat path
  std::printf("bos=%d eos=%d pad=%d add_bos=%d add_eos=%d\n", tok.bos_id(), tok.eos_id(),
              tok.pad_id(), (int)tok.add_bos(), (int)tok.add_eos());
  for (const char *name : {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}) {
    const std::vector<std::int32_t> ids = tok.encode(name, true);
    const std::vector<std::int32_t> ids_nospecial = tok.encode(name, false);
    std::printf("  %-16s parse_special=%d -> %s | parse_special=0 -> %s\n", name,
                (int)!ids.empty(), join(ids, 8).c_str(), join(ids_nospecial, 8).c_str());
    if (ids.size() != 1) {
      std::printf("FAIL %s should be a single special token with parse_special=1\n", name);
      ++failures;
    }
  }

  std::printf("check-tokenizer: %s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
