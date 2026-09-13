// Incremental output helpers for the CLI's `run` command (PLAN.md M4).
//
// The engine emits one token at a time, but a UTF-8 character can span two
// tokens: byte-encoded BPE pieces routinely split a multibyte character (e.g.
// "é" is almost always one token, but a rare CJK codepoint is three bytes that
// can land in three tokens). Writing a piece straight to stdout would emit a
// half-formed character; the fix is to hold the incomplete tail back until the
// rest arrives, which is what `utf8_keep_len` computes.
//
// Header-only and free of HIP so tests/check_stream.cpp can exercise it on CPU.
#pragma once

#include <cstddef>
#include <string>

namespace rdna4 {

// Length of the longest prefix of `s` that ends on a UTF-8 sequence boundary.
// Bytes belonging to a trailing *incomplete* sequence are excluded (they are
// not emitted yet); everything else is included.
//
// Malformed input is handled without reading out of bounds: a run of 4+ stray
// continuation bytes cannot be resolved to a lead byte, so the whole string is
// returned rather than guessing (the alternative would stall the stream
// forever). Invalid UTF-8 cannot be produced by the tokenizer's byte decoder.
inline std::size_t utf8_keep_len(const std::string &s) {
  if (s.empty()) return 0;
  std::size_t i = s.size();
  std::size_t cont = 0;
  // walk back over continuation bytes (0x80..0xBF), at most 3 of them
  while (i > 0 && cont < 3) {
    const unsigned char c = static_cast<unsigned char>(s[i - 1]);
    if (c < 0x80 || c >= 0xC0) break;
    --i;
    ++cont;
  }
  if (i == 0) return s.size();  // only continuation bytes: nothing to trim against
  const unsigned char c = static_cast<unsigned char>(s[i - 1]);
  const std::size_t need = c < 0x80 ? 1 : (c >= 0xF0 ? 4 : (c >= 0xE0 ? 3 : 2));
  const std::size_t have = 1 + (s.size() - i);
  return have >= need ? s.size() : i - 1;
}

}  // namespace rdna4
