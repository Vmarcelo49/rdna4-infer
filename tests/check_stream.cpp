// CPU test for the CLI's incremental UTF-8 output (PLAN.md M4).
//
// The engine streams one token at a time and a multibyte character can be split
// across two tokens, so the emit loop must hold the incomplete tail back. This
// test pins the boundary function and then replays real byte streams through the
// same "emit the new complete prefix" loop the CLI uses, asserting that the
// concatenation of everything emitted is byte-identical to the input (nothing
// dropped, nothing duplicated, nothing half-written).
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

#include "rdna4/stream.h"

namespace {

int failures = 0;

void expect_eq(std::size_t got, std::size_t want, const char *what) {
  if (got != want) {
    std::printf("FAIL: %s: got %zu, want %zu\n", what, got, want);
    ++failures;
  }
}

// Explicit bytes, so the test does not depend on the source file's encoding.
std::string bytes(std::initializer_list<int> b) {
  std::string s;
  for (int v : b) s.push_back((char)(unsigned char)v);
  return s;
}

// The CLI's loop, reduced to its essential state: `emitted` holds the bytes
// already written and must always stay a prefix of the full text.
struct Emitter {
  std::string emitted;
  bool prefix_violation = false;
  void feed(const std::string &all) {
    if (all.compare(0, emitted.size(), emitted) != 0) prefix_violation = true;
    const std::size_t keep = rdna4::utf8_keep_len(all);
    if (keep > emitted.size()) emitted.assign(all, 0, keep);
  }
};

void boundary_cases() {
  // empty / ascii
  expect_eq(rdna4::utf8_keep_len(""), 0, "empty");
  expect_eq(rdna4::utf8_keep_len("abc"), 3, "ascii");
  // complete 2-byte (é = C3 A9), 3-byte (€ = E2 82 AC), 4-byte (😀 = F0 9F 98 80)
  expect_eq(rdna4::utf8_keep_len(bytes({0xC3, 0xA9})), 2, "2-byte complete");
  expect_eq(rdna4::utf8_keep_len(bytes({0xE2, 0x82, 0xAC})), 3, "3-byte complete");
  expect_eq(rdna4::utf8_keep_len(bytes({0xF0, 0x9F, 0x98, 0x80})), 4, "4-byte complete");
  // truncated tails: the lead byte must be held back too
  expect_eq(rdna4::utf8_keep_len(bytes({0xC3})), 0, "2-byte lead only");
  expect_eq(rdna4::utf8_keep_len(bytes({0xE2, 0x82})), 0, "3-byte lead + 1");
  expect_eq(rdna4::utf8_keep_len(bytes({0xF0, 0x9F, 0x98})), 0, "4-byte lead + 2");
  // ascii before a truncated tail: only the ascii is emitted
  expect_eq(rdna4::utf8_keep_len(bytes({'a', 0xC3})), 1, "ascii + 2-byte lead");
  expect_eq(rdna4::utf8_keep_len(bytes({'a', 'b', 0xE2, 0x82})), 2, "ascii + 3-byte partial");
  expect_eq(rdna4::utf8_keep_len(bytes({'a', 0xF0, 0x9F, 0x98})), 1, "ascii + 4-byte partial");
  // complete char before a truncated one
  expect_eq(rdna4::utf8_keep_len(bytes({0xC3, 0xA9, 0xE2, 0x82})), 2, "2-byte then partial 3-byte");
  // several complete chars: everything is emittable
  expect_eq(rdna4::utf8_keep_len(bytes({0xC3, 0xA9, 0xE2, 0x82, 0xAC, 'x'})), 6,
            "2+3-byte + ascii");
  // malformed: a 4-continuation run cannot be resolved to a lead byte -> emit all
  // (documented degradation; the tokenizer's byte decoder cannot produce it)
  expect_eq(rdna4::utf8_keep_len(bytes({0x80, 0x80, 0x80, 0x80})), 4, "stray continuations");
  expect_eq(rdna4::utf8_keep_len(bytes({0x80, 0x80})), 2, "two stray continuations");
}

// Replays each text one byte at a time (the worst case for a split) and also in
// a few random-sized chunks, checking the concatenated output.
void replay_case(const std::string &text, const char *label) {
  for (int mode = 0; mode < 3; ++mode) {
    Emitter e;
    std::string acc;  // what the CLI would have written, in order
    std::string prev;
    if (mode == 0) {
      for (std::size_t i = 1; i <= text.size(); ++i) {
        e.feed(text.substr(0, i));
        acc.append(e.emitted, prev.size(), e.emitted.size() - prev.size());
        prev = e.emitted;
      }
    } else if (mode == 1) {
      const std::size_t chunk = 2;
      for (std::size_t i = chunk; i < text.size() + chunk; i += chunk) {
        e.feed(text.substr(0, i < text.size() ? i : text.size()));
        acc.append(e.emitted, prev.size(), e.emitted.size() - prev.size());
        prev = e.emitted;
      }
    } else {
      e.feed(text);
      acc = e.emitted;
      prev = e.emitted;
    }
    // final flush (the CLI writes the remaining tail once the loop ends)
    if (text.size() > prev.size()) acc.append(text, prev.size(), std::string::npos);
    if (e.prefix_violation) {
      std::printf("FAIL: %s mode %d: emitted bytes were not a prefix of the text\n", label, mode);
      ++failures;
    }
    if (acc != text) {
      std::printf("FAIL: %s mode %d: replayed %zu bytes, input %zu, %s\n", label, mode, acc.size(),
                  text.size(), acc == text ? "equal" : "DIFFERENT");
      ++failures;
    }
  }
}

}  // namespace

int main() {
  boundary_cases();
  replay_case("Paris.\n", "ascii");
  replay_case("caf\xC3\xA9 \xE2\x82\xAC \xF0\x9F\x98\x80", "mixed multibyte");
  replay_case(std::string("\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E"), "CJK");
  replay_case(std::string("\xF0\x9F\x98\x80\xF0\x9F\x98\x80"), "two 4-byte emoji");
  replay_case(std::string("a\xC3\xA9") + std::string(40, 'z') + "\xE2\x82\xAC", "long ascii tail");

  if (failures == 0) {
    std::printf("check-stream: OK\n");
    return 0;
  }
  std::printf("check-stream: %d failure(s)\n", failures);
  return 1;
}
