// BPE tokenizer for the qwen35 vocab (PLAN.md M4).
//
// The GGUF carries the whole vocabulary (`tokenizer.ggml.tokens`), the merge
// ranks (`tokenizer.ggml.merges`), the token attributes
// (`tokenizer.ggml.token_type`) and `tokenizer.ggml.pre = "qwen35"`, which
// selects llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN35 pre-tokenizer regex. The
// pipeline mirrors llama.cpp's llm_tokenizer_bpe:
//
//   text -> split on special tokens (CONTROL/UNKNOWN only when parse_special,
//           USER_DEFINED always) -> per fragment:
//           pre-tokenize with the qwen35 regex -> GPT-2 byte-encode each word ->
//           BPE merges by rank -> token ids
//
// `decode` is the reverse for the byte-encoded pieces (token text -> bytes).
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "rdna4/gguf.h"

namespace rdna4 {

// Token attributes (tokenizer.ggml.token_type).
enum TokenAttr : std::int32_t {
  kTokenNormal = 1,
  kTokenUnknown = 2,
  kTokenControl = 3,
  kTokenUserDefined = 4,
  kTokenUnused = 5,
  kTokenByte = 6,
};

class Tokenizer {
 public:
  // Reads the vocabulary out of an already-parsed GGUF. Fails (with err) if the
  // vocab is not a GPT-2-style BPE with pre = qwen35, or if any required array
  // is missing.
  bool init(const gguf::File &f, std::string &err);

  // Encodes text. `parse_special` enables the CONTROL/UNKNOWN tokens as text
  // (USER_DEFINED ones are always matched, as llama.cpp does).
  std::vector<std::int32_t> encode(const std::string &text, bool parse_special = true) const;

  // Piece of one token, byte-decoded (GPT-2 byte encoding undone).
  std::string token_to_piece(std::int32_t id) const;

  // Concatenated pieces; `skip_special` drops CONTROL tokens (the usual
  // "printable text" behaviour).
  std::string decode(const std::vector<std::int32_t> &ids, bool skip_special = false) const;

  std::size_t n_vocab() const { return tokens_.size(); }
  std::int32_t bos_id() const { return bos_; }
  std::int32_t eos_id() const { return eos_; }
  std::int32_t pad_id() const { return pad_; }
  bool add_bos() const { return add_bos_; }
  bool add_eos() const { return add_eos_; }
  bool is_special(std::int32_t id) const {
    return id >= 0 && (std::size_t)id < types_.size() &&
           (types_[id] == kTokenControl || types_[id] == kTokenUnknown);
  }

  // Text of one token, un-byte-decoded (the raw vocab entry).
  const std::string &token_text(std::int32_t id) const { return tokens_[id]; }

 private:
  // BPE merges one pre-tokenized, byte-encoded word into ids.
  void bpe(const std::string &word, std::vector<std::int32_t> &out) const;

  std::vector<std::string> tokens_;
  std::vector<std::int32_t> types_;
  std::unordered_map<std::string, std::int32_t> token_to_id_;
  std::unordered_map<std::string, std::int32_t> merge_rank_;  // "left right" -> rank
  // Special tokens sorted longest-first, for the partition pass.
  std::vector<std::pair<std::string, std::int32_t>> specials_;
  std::int32_t bos_ = -1, eos_ = -1, pad_ = -1;
  bool add_bos_ = false, add_eos_ = false;
};

}  // namespace rdna4
