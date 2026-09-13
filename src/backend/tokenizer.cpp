// BPE tokenizer implementation (see include/rdna4/tokenizer.h).
#include "rdna4/tokenizer.h"

#include <algorithm>
#include <cstring>

#include "rdna4/unicode.h"

namespace rdna4 {
namespace {

std::int32_t kv_i32(const gguf::File &f, const char *key, std::int32_t fallback) {
  auto it = f.kv.find(key);
  if (it == f.kv.end()) return fallback;
  const gguf::Value &v = it->second;
  switch (v.type) {
    case gguf::U8: return (std::int32_t)v.u;
    case gguf::I8: return (std::int32_t)(std::int8_t)v.u;
    case gguf::U16: return (std::int32_t)(std::uint16_t)v.u;
    case gguf::I16: return (std::int32_t)(std::int16_t)v.u;
    case gguf::U32: return (std::int32_t)v.u;
    case gguf::I32: return (std::int32_t)(std::int32_t)v.u;
    case gguf::U64: return (std::int32_t)v.u;
    case gguf::I64: return (std::int32_t)(std::int64_t)v.u;
    default: return fallback;
  }
}

bool kv_bool(const gguf::File &f, const char *key, bool fallback) {
  auto it = f.kv.find(key);
  if (it == f.kv.end()) return fallback;
  if (it->second.type == gguf::BOOL) return it->second.u != 0;
  if (it->second.type == gguf::U32 || it->second.type == gguf::I32) return it->second.u != 0;
  return fallback;
}

const gguf::Value *kv(const gguf::File &f, const char *key) {
  auto it = f.kv.find(key);
  return it == f.kv.end() ? nullptr : &it->second;
}

std::string kv_text(const gguf::File &f, const char *key, const char *fallback) {
  const gguf::Value *v = kv(f, key);
  return (v && v->type == gguf::STR) ? v->s : std::string(fallback);
}

}  // namespace

bool Tokenizer::init(const gguf::File &f, std::string &err) {
  const std::string model = kv_text(f, "tokenizer.ggml.model", "");
  if (model != "gpt2") {
    err = "tokenizer.ggml.model must be \"gpt2\", got \"" + model + "\"";
    return false;
  }
  const std::string pre = kv_text(f, "tokenizer.ggml.pre", "");
  if (pre != "qwen35") {
    err = "tokenizer.ggml.pre must be \"qwen35\", got \"" + pre + "\"";
    return false;
  }

  const gguf::Value *toks = kv(f, "tokenizer.ggml.tokens");
  if (!toks || toks->type != gguf::ARRAY || toks->arr_type != gguf::STR) {
    err = "tokenizer.ggml.tokens missing or not an array of strings";
    return false;
  }
  const gguf::Value *merges = kv(f, "tokenizer.ggml.merges");
  if (!merges || merges->type != gguf::ARRAY || merges->arr_type != gguf::STR) {
    err = "tokenizer.ggml.merges missing or not an array of strings";
    return false;
  }

  tokens_.clear();
  tokens_.reserve(toks->arr.size());
  token_to_id_.clear();
  token_to_id_.reserve(toks->arr.size() * 2);
  for (std::size_t i = 0; i < toks->arr.size(); ++i) {
    tokens_.push_back(toks->arr[i].s);
    token_to_id_.emplace(tokens_.back(), (std::int32_t)i);
  }

  // token_type is optional (llama.cpp defaults everything to NORMAL); a missing
  // array means "no special tokens", which would break the chat prompt, so
  // require it when the file has one and treat absent as all-NORMAL.
  types_.assign(tokens_.size(), kTokenNormal);
  if (const gguf::Value *tt = kv(f, "tokenizer.ggml.token_type")) {
    if (tt->type != gguf::ARRAY || tt->arr.size() != tokens_.size()) {
      err = "tokenizer.ggml.token_type must be an array with one entry per token";
      return false;
    }
    for (std::size_t i = 0; i < tt->arr.size(); ++i) {
      types_[i] = (std::int32_t)tt->arr[i].u;
    }
  }

  merge_rank_.clear();
  merge_rank_.reserve(merges->arr.size() * 2);
  for (std::size_t i = 0; i < merges->arr.size(); ++i) {
    // "left right": the key is the pair joined by a space, as llama.cpp stores
    // it in its bpe_ranks cache.
    merge_rank_.emplace(merges->arr[i].s, (std::int32_t)i);
  }

  specials_.clear();
  for (std::size_t i = 0; i < tokens_.size(); ++i) {
    const bool user_defined = types_[i] == kTokenUserDefined;
    const bool control = types_[i] == kTokenControl || types_[i] == kTokenUnknown;
    if ((user_defined || control) && !tokens_[i].empty()) {
      specials_.emplace_back(tokens_[i], (std::int32_t)i);
    }
  }
  // Longest match first so that e.g. "<|im_start|>" wins over "<|im";
  // ties resolve to the smaller id, as llama.cpp's cache order does.
  std::sort(specials_.begin(), specials_.end(),
            [](const std::pair<std::string, std::int32_t> &a,
               const std::pair<std::string, std::int32_t> &b) {
              if (a.first.size() != b.first.size()) return a.first.size() > b.first.size();
              return a.second < b.second;
            });

  bos_ = kv_i32(f, "tokenizer.ggml.bos_token_id", -1);
  eos_ = kv_i32(f, "tokenizer.ggml.eos_token_id", -1);
  pad_ = kv_i32(f, "tokenizer.ggml.padding_token_id", -1);
  add_bos_ = kv_bool(f, "tokenizer.ggml.add_bos_token", false);
  add_eos_ = kv_bool(f, "tokenizer.ggml.add_eos_token", false);

  // End-of-generation set, built the way llama.cpp builds it (llama-vocab.cpp
  // "maintain a list of tokens that cause end-of-generation"): the FIM ids, then
  // every token whose *text* is in the control-looking EOG list, then the EOS
  // (and EOT/EOM) ids from the metadata. For qwen35 that adds 248044
  // (`<|endoftext|>`) next to 248046 (`<|im_end|>`), and a generation that ends
  // on the former must stop like the reference does.
  eog_.assign(tokens_.size(), false);
  auto add_eog = [&](std::int32_t id) {
    if (id >= 0 && (std::size_t)id < eog_.size()) eog_[(std::size_t)id] = true;
  };
  static const char *kEogTexts[] = {
      "<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>",
      "<|calls|>", "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>",
      "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>", "<end_of_utterance>", "<eos>",
      "<turn|>", "<|tool_response>", "<\xEF\xBD\x9Cend\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>",
      "[e~[",
  };
  for (const char *txt : kEogTexts) {
    auto it = token_to_id_.find(txt);
    if (it != token_to_id_.end()) add_eog(it->second);
  }
  // The FIM ids are *not* in this GGUF's metadata: llama.cpp auto-detects them by
  // text too (llama-vocab.cpp "find FIM_PAD token: ...", :2769-2810) and then
  // inserts them into the EOG set, which is where 248063/248064/248065 come
  // from. llama.cpp keeps only the first text match it encounters while scanning
  // an unordered_map; every candidate here has at most one match in this vocab,
  // so adding all matches is identical. (A vocab carrying two spellings of the
  // same FIM token would make us add both — documented divergence.)
  static const char *kFimTexts[] = {
      "<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]",
      "<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>",
      "<|file_sep|>",
  };
  for (const char *txt : kFimTexts) {
    auto it = token_to_id_.find(txt);
    if (it != token_to_id_.end()) add_eog(it->second);
  }
  add_eog(kv_i32(f, "tokenizer.ggml.fim_pad_token_id", -1));
  add_eog(kv_i32(f, "tokenizer.ggml.fim_rep_token_id", -1));
  add_eog(kv_i32(f, "tokenizer.ggml.fim_sep_token_id", -1));
  add_eog(eos_);
  add_eog(kv_i32(f, "tokenizer.ggml.eot_token_id", -1));
  add_eog(kv_i32(f, "tokenizer.ggml.eom_token_id", -1));

  if (tokens_.empty()) {
    err = "empty vocabulary";
    return false;
  }
  return true;
}

void Tokenizer::bpe(const std::string &word, std::vector<std::int32_t> &out) const {
  if (word.empty()) return;

  // Split into UTF-8 symbols (the byte-encoded word is valid UTF-8).
  std::vector<std::string> symbols;
  symbols.reserve(word.size());
  for (std::size_t off = 0; off < word.size();) {
    const std::size_t n = unicode_len_utf8(word[off]);
    symbols.push_back(word.substr(off, n));
    off += n;
  }

  // if the whole word is already a token, take it (llama.cpp's
  // `tokenizer_ignore_merges` path is off by default, but the explicit
  // text_to_token check below covers the same case through the merge loop).
  while (symbols.size() > 1) {
    // find the adjacent pair with the lowest merge rank
    std::int32_t best_rank = -1;
    std::size_t best = 0;
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      auto it = merge_rank_.find(symbols[i] + " " + symbols[i + 1]);
      if (it == merge_rank_.end()) continue;
      if (best_rank < 0 || it->second < best_rank) {
        best_rank = it->second;
        best = i;
      }
    }
    if (best_rank < 0) break;
    symbols[best] += symbols[best + 1];
    symbols.erase(symbols.begin() + (long)best + 1);
  }

  for (const std::string &s : symbols) {
    auto it = token_to_id_.find(s);
    if (it != token_to_id_.end()) {
      out.push_back(it->second);
      continue;
    }
    // Unknown symbol: fall back to per-byte tokens (llama.cpp does the same);
    // a byte with no token is a hard vocabulary problem, so report it by
    // emitting nothing for that byte and letting the caller compare ids.
    for (char c : s) {
      auto bit = token_to_id_.find(std::string(1, c));
      if (bit != token_to_id_.end()) out.push_back(bit->second);
    }
  }
}

std::vector<std::int32_t> Tokenizer::encode(const std::string &text, bool parse_special) const {
  std::vector<std::int32_t> out;
  if (text.empty()) return out;

  std::size_t pos = 0;
  while (pos < text.size()) {
    // Find the earliest special-token match at or after `pos`.
    std::size_t best_at = std::string::npos;
    std::int32_t best_id = -1;
    for (const auto &sp : specials_) {
      if (!parse_special && types_[sp.second] != kTokenUserDefined) continue;
      const std::size_t at = text.find(sp.first, pos);
      if (at == std::string::npos) continue;
      if (best_at == std::string::npos || at < best_at ||
          (at == best_at && sp.first.size() > tokens_[best_id].size())) {
        best_at = at;
        best_id = sp.second;
      }
    }
    if (best_at == std::string::npos) {
      const std::string fragment = text.substr(pos);
      for (const std::string &word : unicode_regex_split(fragment)) bpe(word, out);
      break;
    }
    if (best_at > pos) {
      const std::string fragment = text.substr(pos, best_at - pos);
      for (const std::string &word : unicode_regex_split(fragment)) bpe(word, out);
    }
    out.push_back(best_id);
    pos = best_at + tokens_[best_id].size();
  }

  if (add_bos_ && bos_ >= 0) out.insert(out.begin(), bos_);
  if (add_eos_ && eos_ >= 0) out.push_back(eos_);
  return out;
}

namespace {
// "<0xXX>" placeholder helper (BYTE tokens): value of one hex digit, -1 if not.
int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}
}  // namespace

std::string Tokenizer::token_to_piece(std::int32_t id) const {
  if (id < 0 || (std::size_t)id >= tokens_.size()) return std::string();
  const std::string &text = tokens_[id];
  // A BYTE token's text is the placeholder "<0xXX>"; llama.cpp emits the raw
  // byte itself (llama-vocab.cpp token_to_piece). Not reachable in the qwen35
  // vocab (it has no BYTE tokens) but cheap to get right.
  if ((std::size_t)id < types_.size() && types_[id] == kTokenByte && text.size() == 6 &&
      text.compare(0, 3, "<0x") == 0 && text[5] == '>') {
    const int hi = hex_digit(text[3]);
    const int lo = hex_digit(text[4]);
    if (hi >= 0 && lo >= 0) return std::string(1, (char)((hi << 4) | lo));
  }
  std::string out;
  out.reserve(text.size());
  for (std::size_t off = 0; off < text.size();) {
    const std::size_t n = unicode_len_utf8(text[off]);
    if (n == 0 || off + n > text.size()) {
      out.append(text, off, std::string::npos);  // malformed tail: keep it verbatim
      break;
    }
    const std::string cpt = text.substr(off, n);
    std::uint8_t byte = 0;
    if (unicode_utf8_to_byte_safe(cpt, &byte)) {
      out.push_back((char)byte);
    } else {
      // not part of the GPT-2 byte encoding: emit the codepoint as-is instead of
      // throwing out of std::unordered_map::at (review M4)
      out.append(cpt);
    }
    off += n;
  }
  return out;
}

std::string Tokenizer::decode(const std::vector<std::int32_t> &ids, bool skip_special) const {
  std::string out;
  for (std::int32_t id : ids) {
    if (skip_special && is_special(id)) continue;
    out += token_to_piece(id);
  }
  return out;
}

}  // namespace rdna4
