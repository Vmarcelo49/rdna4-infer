// Minimal JSON value/parser/serializer for the OpenAI-compatible server
// (`serve`, see docs/servidor-openai.md).
//
// Why not a library: third_party/ is empty and SPEC.md keeps the dependency set
// at "ROCm and nothing else". The server needs exactly two things — parse a
// request body and build a response body — so this is a ~350-line recursive
// descent parser instead of a vendored header.
//
// Deliberate choices:
//   - no exceptions (the repo builds with the default flags but every module
//     reports errors through `std::string &err`; keeping that here means the
//     request handler has a single failure path: HTTP 400);
//   - object members keep insertion order (a response body is easier to read
//     and test when `id`..`usage` come out in the order they were written);
//   - numbers are stored as double, and dumped without a decimal point when
//     they are integral (so `"completion_tokens": 12` never becomes 12.0);
//   - strings are byte strings: UTF-8 passes through untouched and `\uXXXX`
//     escapes (including surrogate pairs) are encoded to UTF-8 on parse.
#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace rdna4 {
namespace server {
namespace json {

class Value;

using Array = std::vector<Value>;
using Member = std::pair<std::string, Value>;
using Object = std::vector<Member>;

// A JSON value. Default-constructed is `null`, like nlohmann's.
class Value {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Value() = default;

  static Value null() { return Value(); }
  static Value boolean(bool b);
  static Value number(double d);
  static Value integer(long long i);
  static Value string(const std::string &s);
  static Value array();
  static Value object();

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  // Accessors with a fallback: a handler can read a field without checking the
  // type first, which keeps the request parsing short and total.
  bool as_bool(bool def = false) const { return type_ == Type::Bool ? bool_ : def; }
  double as_number(double def = 0.0) const { return type_ == Type::Number ? num_ : def; }
  // Truncates toward zero; returns `def` for a non-number or a non-finite value.
  long long as_int(long long def = 0) const;
  const std::string &as_string() const { return type_ == Type::String ? str_ : empty(); }

  // Array access. `size()` is only non-zero for arrays/objects.
  std::size_t size() const;
  const Value *at(std::size_t i) const;  // nullptr when out of range / not an array
  const Value *member(const char *key) const;  // nullptr when absent / not an object
  bool has(const char *key) const { return member(key) != nullptr; }

  // Builders (in place).
  void push(Value v);
  void set(const std::string &key, Value v);  // replaces an existing member

  // Text accessors used by the server for the numeric/boolean request fields.
  bool get_string(const char *key, std::string *out) const;
  bool get_number(const char *key, double *out) const;
  bool get_int(const char *key, long long *out) const;
  bool get_bool(const char *key, bool *out) const;

  // Serializes to compact JSON (what an HTTP body wants: no pretty printing).
  std::string dump() const;

 private:
  static const std::string &empty();
  void dump_to(std::string &out) const;

  Type type_ = Type::Null;
  bool bool_ = false;
  double num_ = 0.0;
  std::string str_;
  Array arr_;
  Object obj_;
};

// Parses a whole document. Returns false with a human-readable `err`
// ("unexpected character at offset 7") on malformed input, on trailing garbage
// or on nesting deeper than kMaxDepth (a hostile body must not blow the stack).
bool parse(const std::string &text, Value &out, std::string &err);

// Escapes `s` as a JSON string, quotes included (used by dump and by the SSE
// writer, which builds `data: {...}` lines by hand only for the fixed shapes).
std::string quote(const std::string &s);

}  // namespace json
}  // namespace server
}  // namespace rdna4
