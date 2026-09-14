#include "rdna4/server_json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rdna4 {
namespace server {
namespace json {

namespace {

constexpr int kMaxDepth = 64;

bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// Appends one codepoint as UTF-8. Lone surrogates become U+FFFD (the only
// lossless-ish choice: JSON allows the escape, UTF-8 does not allow the
// codepoint).
void append_utf8(std::string &out, unsigned cp) {
  if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD;
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

int hex_val(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

class Parser {
 public:
  Parser(const std::string &s, std::string &err) : s_(s), err_(err) {}

  bool run(Value &out) {
    skip_ws();
    if (!value(out, 0)) return false;
    skip_ws();
    if (i_ != s_.size()) return fail("trailing characters");
    return true;
  }

 private:
  bool fail(const char *what) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s at offset %zu", what, i_);
    err_ = buf;
    return false;
  }
  bool failf(const char *what, const std::string &extra) {
    err_ = std::string(what) + " '" + extra + "' at offset " + std::to_string(i_);
    return false;
  }

  void skip_ws() {
    while (i_ < s_.size() && is_ws(s_[i_])) ++i_;
  }

  bool literal(const char *lit) {
    const std::size_t n = std::strlen(lit);
    if (s_.compare(i_, n, lit) != 0) return fail("invalid literal");
    i_ += n;
    return true;
  }

  bool value(Value &out, int depth) {
    if (depth > kMaxDepth) return fail("nesting too deep");
    if (i_ >= s_.size()) return fail("unexpected end of input");
    switch (s_[i_]) {
      case '{': return object(out, depth);
      case '[': return array(out, depth);
      case '"': {
        std::string str;
        if (!string(str)) return false;
        out = Value::string(str);
        return true;
      }
      case 't':
        if (!literal("true")) return false;
        out = Value::boolean(true);
        return true;
      case 'f':
        if (!literal("false")) return false;
        out = Value::boolean(false);
        return true;
      case 'n':
        if (!literal("null")) return false;
        out = Value::null();
        return true;
      default: return number(out);
    }
  }

  bool object(Value &out, int depth) {
    ++i_;  // '{'
    out = Value::object();
    skip_ws();
    if (i_ < s_.size() && s_[i_] == '}') {
      ++i_;
      return true;
    }
    for (;;) {
      skip_ws();
      if (i_ >= s_.size() || s_[i_] != '"') return fail("expected a member name");
      std::string key;
      if (!string(key)) return false;
      skip_ws();
      if (i_ >= s_.size() || s_[i_] != ':') return fail("expected ':'");
      ++i_;
      skip_ws();
      Value v;
      if (!value(v, depth + 1)) return false;
      out.set(key, std::move(v));
      skip_ws();
      if (i_ >= s_.size()) return fail("unexpected end of input in object");
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == '}') {
        ++i_;
        return true;
      }
      return fail("expected ',' or '}'");
    }
  }

  bool array(Value &out, int depth) {
    ++i_;  // '['
    out = Value::array();
    skip_ws();
    if (i_ < s_.size() && s_[i_] == ']') {
      ++i_;
      return true;
    }
    for (;;) {
      skip_ws();
      Value v;
      if (!value(v, depth + 1)) return false;
      out.push(std::move(v));
      skip_ws();
      if (i_ >= s_.size()) return fail("unexpected end of input in array");
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == ']') {
        ++i_;
        return true;
      }
      return fail("expected ',' or ']'");
    }
  }

  // Lenient on purpose: a raw control byte inside a string is accepted as-is.
  // Every sane client escapes them, and accepting one costs nothing while a
  // hard rejection would turn a slightly-sloppy harness into a 400.
  bool string(std::string &out) {
    ++i_;  // '"'
    out.clear();
    while (i_ < s_.size()) {
      const char c = s_[i_++];
      if (c == '"') return true;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (i_ >= s_.size()) return fail("unterminated escape");
      const char e = s_[i_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          unsigned cp = 0;
          if (!hex4(&cp)) return false;
          if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 1 < s_.size() && s_[i_] == '\\' &&
              s_[i_ + 1] == 'u') {
            const std::size_t save = i_;
            i_ += 2;
            unsigned lo = 0;
            if (!hex4(&lo)) return false;
            if (lo >= 0xDC00 && lo <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else {
              i_ = save;  // not a low surrogate: emit the high one as U+FFFD
            }
          }
          append_utf8(out, cp);
          break;
        }
        default: return fail("invalid escape");
      }
    }
    return fail("unterminated string");
  }

  bool hex4(unsigned *out) {
    if (i_ + 4 > s_.size()) return fail("truncated \\u escape");
    unsigned v = 0;
    for (int k = 0; k < 4; ++k) {
      const int h = hex_val(s_[i_ + (std::size_t)k]);
      if (h < 0) return fail("invalid \\u escape");
      v = (v << 4) | (unsigned)h;
    }
    i_ += 4;
    *out = v;
    return true;
  }

  bool number(Value &out) {
    const std::size_t start = i_;
    if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
    std::size_t digits = 0;
    while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') {
      ++i_;
      ++digits;
    }
    if (digits == 0) return fail("expected a value");
    if (i_ < s_.size() && s_[i_] == '.') {
      ++i_;
      std::size_t frac = 0;
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') {
        ++i_;
        ++frac;
      }
      if (frac == 0) return fail("expected a digit after '.'");
    }
    if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      ++i_;
      if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
      std::size_t exp = 0;
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') {
        ++i_;
        ++exp;
      }
      if (exp == 0) return fail("expected a digit in the exponent");
    }
    const std::string tok = s_.substr(start, i_ - start);
    out = Value::number(std::strtod(tok.c_str(), nullptr));
    return true;
  }

  const std::string &s_;
  std::string &err_;
  std::size_t i_ = 0;
};

}  // namespace

Value Value::boolean(bool b) {
  Value v;
  v.type_ = Type::Bool;
  v.bool_ = b;
  return v;
}

Value Value::number(double d) {
  Value v;
  v.type_ = Type::Number;
  v.num_ = d;
  return v;
}

Value Value::integer(long long i) { return number(static_cast<double>(i)); }

Value Value::string(const std::string &s) {
  Value v;
  v.type_ = Type::String;
  v.str_ = s;
  return v;
}

Value Value::array() {
  Value v;
  v.type_ = Type::Array;
  return v;
}

Value Value::object() {
  Value v;
  v.type_ = Type::Object;
  return v;
}

const std::string &Value::empty() {
  static const std::string kEmpty;
  return kEmpty;
}

long long Value::as_int(long long def) const {
  if (type_ != Type::Number || !std::isfinite(num_)) return def;
  if (num_ >= 9.2233720368547758e18 || num_ <= -9.2233720368547758e18) return def;
  return static_cast<long long>(num_);
}

std::size_t Value::size() const {
  if (type_ == Type::Array) return arr_.size();
  if (type_ == Type::Object) return obj_.size();
  return 0;
}

const Value *Value::at(std::size_t i) const {
  if (type_ != Type::Array || i >= arr_.size()) return nullptr;
  return &arr_[i];
}

const Value *Value::member(const char *key) const {
  if (type_ != Type::Object || key == nullptr) return nullptr;
  for (const Member &m : obj_) {
    if (m.first == key) return &m.second;
  }
  return nullptr;
}

void Value::push(Value v) {
  if (type_ != Type::Array) return;
  arr_.push_back(std::move(v));
}

void Value::set(const std::string &key, Value v) {
  if (type_ != Type::Object) {
    type_ = Type::Object;
    obj_.clear();
  }
  for (Member &m : obj_) {
    if (m.first == key) {
      m.second = std::move(v);
      return;
    }
  }
  obj_.emplace_back(key, std::move(v));
}

bool Value::get_string(const char *key, std::string *out) const {
  const Value *v = member(key);
  if (v == nullptr || !v->is_string()) return false;
  *out = v->str_;
  return true;
}

bool Value::get_number(const char *key, double *out) const {
  const Value *v = member(key);
  if (v == nullptr || !v->is_number()) return false;
  *out = v->num_;
  return true;
}

bool Value::get_int(const char *key, long long *out) const {
  const Value *v = member(key);
  if (v == nullptr || !v->is_number()) return false;
  *out = v->as_int();
  return true;
}

bool Value::get_bool(const char *key, bool *out) const {
  const Value *v = member(key);
  if (v == nullptr || !v->is_bool()) return false;
  *out = v->bool_;
  return true;
}

std::string quote(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 2);
  out.push_back('"');
  for (const char c : s) {
    const unsigned char u = static_cast<unsigned char>(c);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (u < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", u);
          out += buf;
        } else {
          out.push_back(c);  // UTF-8 bytes pass through unescaped
        }
    }
  }
  out.push_back('"');
  return out;
}

void Value::dump_to(std::string &out) const {
  switch (type_) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += bool_ ? "true" : "false"; break;
    case Type::Number: {
      if (!std::isfinite(num_)) {
        out += "null";  // JSON has no NaN/Infinity
        break;
      }
      char buf[40];
      if (num_ == std::floor(num_) && std::fabs(num_) < 9.007199254740992e15) {
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(num_));
      } else {
        std::snprintf(buf, sizeof(buf), "%.17g", num_);
      }
      out += buf;
      break;
    }
    case Type::String: out += quote(str_); break;
    case Type::Array: {
      out.push_back('[');
      for (std::size_t i = 0; i < arr_.size(); ++i) {
        if (i != 0) out.push_back(',');
        arr_[i].dump_to(out);
      }
      out.push_back(']');
      break;
    }
    case Type::Object: {
      out.push_back('{');
      for (std::size_t i = 0; i < obj_.size(); ++i) {
        if (i != 0) out.push_back(',');
        out += quote(obj_[i].first);
        out.push_back(':');
        obj_[i].second.dump_to(out);
      }
      out.push_back('}');
      break;
    }
  }
}

std::string Value::dump() const {
  std::string out;
  dump_to(out);
  return out;
}

bool parse(const std::string &text, Value &out, std::string &err) {
  err.clear();
  Parser p(text, err);
  out = Value::null();
  if (!p.run(out)) {
    if (err.empty()) err = "invalid JSON";
    return false;
  }
  return true;
}

}  // namespace json
}  // namespace server
}  // namespace rdna4
