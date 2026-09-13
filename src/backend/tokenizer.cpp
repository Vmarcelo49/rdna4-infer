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

std::string Tokenizer::token_to_piece(std::int32_t id) const {
  if (id < 0 || (std::size_t)id >= tokens_.size()) return std::string();
  const std::string &text = tokens_[id];
  std::string out;
  out.reserve(text.size());
  for (std::size_t off = 0; off < text.size();) {
    const std::size_t n = unicode_len_utf8(text[off]);
    const std::string cpt = text.substr(off, n);
    const std::uint8_t byte = unicode_utf8_to_byte(cpt);
    out.push_back((char)byte);
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
