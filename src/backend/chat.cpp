// qwen35 chat rendering + output parsing (see include/rdna4/chat.h).
//
// Render mirrors llama.cpp's common_chat_templates_apply over the qwen35 Jinja
// template (tools via common_chat_tools_to_json_oaicompat + `tool|tojson`,
// assistant tool_calls via the template's <tool_call> blocks). Parse mirrors
// common_chat_parse with the qwen3-coder PEG parser, which upstream selects
// for this template family.
#include "rdna4/chat.h"

#include <cstdint>
#include <cstdio>

namespace rdna4 {
namespace {

// The template's reasoning_instructions texts, verbatim.
constexpr const char *kEffortXhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";
constexpr const char *kEffortLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
    "conclusion without unnecessary elaboration.";

// The template's tool-format instructions, verbatim (the text after <tools>).
constexpr const char *kToolsSuffix =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
    "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
    "Reminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML "
    "tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for "
    "your function call in natural language BEFORE the function call, but NOT after\n- If "
    "there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n</IMPORTANT>";

constexpr const char *kThinkOpen = "<think>";
constexpr const char *kThinkClose = "</think>";
constexpr const char *kToolCallOpen = "<tool_call>";
constexpr const char *kToolCallClose = "</tool_call>";
constexpr const char *kFuncClose = "</function>";
constexpr const char *kParamClose = "</parameter>";

bool is_ws(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// Jinja's `|trim` strips whitespace from both ends.
std::string trim(const std::string &s) {
  std::size_t b = 0, e = s.size();
  while (b < e && is_ws(s[b])) ++b;
  while (e > b && is_ws(s[e - 1])) --e;
  return s.substr(b, e - b);
}

// PEG space(): leading whitespace run (mirrors the parser's space rule).
std::string lstrip_space(const std::string &s) {
  std::size_t b = 0;
  while (b < s.size() && is_ws(s[b])) ++b;
  return s.substr(b);
}

bool is_system_role(const std::string &role) {
  return role == "system" || role == "developer";
}

bool is_all_ws(const std::string &s) {
  for (char c : s) {
    if (!is_ws(c)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Minimal order-preserving JSON value (just enough for tool parameters and
// tool-call arguments: objects, arrays, strings, numbers, booleans, null).
// Emission mirrors minja's `tojson` filter (common/jinja/value.cpp): ", " /
// ": " separators, insertion-ordered keys, ensure_ascii=false (raw UTF-8),
// C-style short escapes plus \u00xx for other control characters.
// ---------------------------------------------------------------------------

struct JVal {
  enum class Type { Null, Bool, Int, Double, String, Array, Object, Raw };
  Type type = Type::Null;
  bool boolean = false;
  std::int64_t integer = 0;
  double number = 0.0;
  std::string literal;  // doubles: original text, emitted verbatim
  std::string str;
  std::vector<JVal> items;
  std::vector<std::pair<std::string, JVal>> fields;  // insertion-ordered
  // Raw: pre-rendered text emitted verbatim (only for still-streaming partial
  // argument values, which are deliberately unterminated so clients can append
  // deltas, mirroring the PEG mapper's args_buffer prefixes).
};

struct JParser {
  const std::string &s;
  std::size_t pos = 0;
  std::string err;

  explicit JParser(const std::string &s) : s(s) {}

  bool fail(const std::string &msg) {
    err = msg + " at byte " + std::to_string(pos);
    return false;
  }

  void skip_ws() {
    while (pos < s.size() && is_ws(s[pos])) ++pos;
  }

  bool expect(char c, const char *what) {
    if (pos >= s.size() || s[pos] != c) return fail(std::string("expected ") + what);
    ++pos;
    return true;
  }

  bool parse_value(JVal &out) {
    skip_ws();
    if (pos >= s.size()) return fail("unexpected end of JSON");
    const char c = s[pos];
    if (c == '{') return parse_object(out);
    if (c == '[') return parse_array(out);
    if (c == '"') {
      out.type = JVal::Type::String;
      return parse_string(out.str);
    }
    if (c == 't') return parse_lit("true", out, JVal::Type::Bool, true);
    if (c == 'f') return parse_lit("false", out, JVal::Type::Bool, false);
    if (c == 'n') {
      if (s.compare(pos, 4, "null") != 0) return fail("bad literal");
      pos += 4;
      out.type = JVal::Type::Null;
      return true;
    }
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number(out);
    return fail("unexpected character");
  }

  bool parse_lit(const char *word, JVal &out, JVal::Type t, bool b) {
    if (s.compare(pos, 4, word, 0, 4) != 0 && s.compare(pos, 5, word, 0, 5) != 0) {
      return fail("bad literal");
    }
    const std::size_t n = std::string(word).size();
    if (s.compare(pos, n, word) != 0) return fail("bad literal");
    pos += n;
    out.type = t;
    out.boolean = b;
    return true;
  }

  static void append_utf8(std::string &out, unsigned cp) {
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

  bool parse_hex4(unsigned &cp) {
    if (pos + 4 > s.size()) return fail("bad \\u escape");
    cp = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = s[pos++];
      cp <<= 4;
      if (c >= '0' && c <= '9') cp |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f') cp |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') cp |= static_cast<unsigned>(c - 'A' + 10);
      else return fail("bad \\u escape");
    }
    return true;
  }

  bool parse_string(std::string &out) {
    // s[pos] == '"'
    ++pos;
    out.clear();
    while (true) {
      if (pos >= s.size()) return fail("unterminated string");
      const char c = s[pos++];
      if (c == '"') return true;
      if (c == '\\') {
        if (pos >= s.size()) return fail("unterminated escape");
        const char e = s[pos++];
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
            if (!parse_hex4(cp)) return false;
            if (cp >= 0xD800 && cp <= 0xDBFF && pos + 2 <= s.size() && s[pos] == '\\' &&
                s[pos + 1] == 'u') {
              pos += 2;
              unsigned lo = 0;
              if (!parse_hex4(lo)) return false;
              if (lo >= 0xDC00 && lo <= 0xDFFF) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              // else: unpaired high surrogate; emit replacement for both parts
            }
            if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD;
            append_utf8(out, cp);
            break;
          }
          default: return fail("bad escape");
        }
      } else {
        out.push_back(c);
      }
    }
  }

  bool parse_number(JVal &out) {
    const std::size_t start = pos;
    if (pos < s.size() && s[pos] == '-') ++pos;
    if (pos >= s.size()) return fail("bad number");
    if (s[pos] == '0') {
      ++pos;
    } else if (s[pos] >= '1' && s[pos] <= '9') {
      while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
    } else {
      return fail("bad number");
    }
    bool is_double = false;
    if (pos < s.size() && s[pos] == '.') {
      is_double = true;
      ++pos;
      if (pos >= s.size() || s[pos] < '0' || s[pos] > '9') return fail("bad number");
      while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
    }
    if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
      is_double = true;
      ++pos;
      if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
      if (pos >= s.size() || s[pos] < '0' || s[pos] > '9') return fail("bad number");
      while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
    }
    const std::string tok = s.substr(start, pos - start);
    if (!is_double) {
      try {
        out.type = JVal::Type::Int;
        out.integer = std::stoll(tok);
        return true;
      } catch (...) {
        // overflow: fall through to double
      }
    }
    out.type = JVal::Type::Double;
    out.literal = tok;
    try {
      out.number = std::stod(tok);
    } catch (...) {
      return fail("bad number");
    }
    return true;
  }

  bool parse_object(JVal &out) {
    // s[pos] == '{'
    ++pos;
    out.type = JVal::Type::Object;
    skip_ws();
    if (pos < s.size() && s[pos] == '}') {
      ++pos;
      return true;
    }
    while (true) {
      skip_ws();
      if (pos >= s.size() || s[pos] != '"') return fail("expected object key");
      std::string key;
      if (!parse_string(key)) return false;
      skip_ws();
      if (!expect(':', "':'")) return false;
      JVal val;
      if (!parse_value(val)) return false;
      out.fields.emplace_back(key, val);
      skip_ws();
      if (pos >= s.size()) return fail("unterminated object");
      if (s[pos] == ',') {
        ++pos;
        continue;
      }
      if (s[pos] == '}') {
        ++pos;
        return true;
      }
      return fail("expected ',' or '}'");
    }
  }

  bool parse_array(JVal &out) {
    // s[pos] == '['
    ++pos;
    out.type = JVal::Type::Array;
    skip_ws();
    if (pos < s.size() && s[pos] == ']') {
      ++pos;
      return true;
    }
    while (true) {
      JVal val;
      if (!parse_value(val)) return false;
      out.items.push_back(val);
      skip_ws();
      if (pos >= s.size()) return fail("unterminated array");
      if (s[pos] == ',') {
        ++pos;
        continue;
      }
      if (s[pos] == ']') {
        ++pos;
        return true;
      }
      return fail("expected ',' or ']'");
    }
  }
};

bool json_parse(const std::string &text, JVal &out, std::string &err) {
  JParser p(text);
  if (!p.parse_value(out)) {
    err = p.err;
    return false;
  }
  p.skip_ws();
  if (p.pos != text.size()) {
    err = "trailing characters at byte " + std::to_string(p.pos);
    return false;
  }
  return true;
}

void json_escape_into(const std::string &in, std::string &out) {
  // minja value_to_json string escaping (ensure_ascii=false).
  for (char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
}

void json_emit(const JVal &v, const char *item_sep, const char *key_sep, std::string &out) {
  if (v.type == JVal::Type::Raw) {
    out += v.str;  // pre-rendered partial text, verbatim
    return;
  }
  switch (v.type) {
    case JVal::Type::Null: out += "null"; break;
    case JVal::Type::Bool: out += v.boolean ? "true" : "false"; break;
    case JVal::Type::Int: out += std::to_string(v.integer); break;
    case JVal::Type::Double: out += v.literal; break;  // verbatim (see header note)
    case JVal::Type::String:
      out.push_back('"');
      json_escape_into(v.str, out);
      out.push_back('"');
      break;
    case JVal::Type::Array:
      out.push_back('[');
      for (std::size_t i = 0; i < v.items.size(); ++i) {
        if (i) out += item_sep;
        json_emit(v.items[i], item_sep, key_sep, out);
      }
      out.push_back(']');
      break;
    case JVal::Type::Object:
      out.push_back('{');
      for (std::size_t i = 0; i < v.fields.size(); ++i) {
        if (i) out += item_sep;
        out.push_back('"');
        json_escape_into(v.fields[i].first, out);
        out.push_back('"');
        out += key_sep;
        json_emit(v.fields[i].second, item_sep, key_sep, out);
      }
      out.push_back('}');
      break;
  }
}

// minja `tojson` (template `|tojson`): ", " / ": " separators.
std::string json_minja(const JVal &v) {
  std::string out;
  json_emit(v, ", ", ": ", out);
  return out;
}

// Canonical compact form for parsed tool-call arguments (llama.cpp's PEG
// mapper builds {"k":v,...} with no spaces, i.e. nlohmann-dump style).
std::string json_compact(const JVal &v) {
  std::string out;
  json_emit(v, ",", ":", out);
  return out;
}

// --- rendering ---------------------------------------------------------------

// Renders one tool as {"type": "function", "function": {...}} exactly the way
// common_chat_tools_to_json_oaicompat builds it and minja `tojson` prints it.
bool render_tool_json(const ChatTool &tool, std::string &out, std::string &err) {
  if (tool.name.empty()) {
    err = "tool is missing a name";
    return false;
  }
  JVal params;
  if (!json_parse(tool.parameters.empty() ? "{}" : tool.parameters, params, err)) {
    err = "tool \"" + tool.name + "\" has invalid parameters JSON: " + err;
    return false;
  }
  if (params.type != JVal::Type::Object) {
    err = "tool \"" + tool.name + "\" parameters must be a JSON object";
    return false;
  }
  out += "{\"type\": \"function\", \"function\": {\"name\": \"";
  json_escape_into(tool.name, out);
  out += "\", \"description\": \"";
  json_escape_into(tool.description, out);
  out += "\", \"parameters\": ";
  out += json_minja(params);
  out += "}}";
  return true;
}

// Renders the <tool_call> blocks of one assistant message. `content_trimmed`
// selects the "\n\n" vs "" separator before the first block, mirroring the
// template's `{% if content|trim %}` branch.
bool render_tool_calls(const std::vector<ChatToolCall> &calls, const std::string &content_trimmed,
                       std::string &out, std::string &err) {
  for (std::size_t i = 0; i < calls.size(); ++i) {
    const ChatToolCall &tc = calls[i];
    if (tc.name.empty()) {
      err = "Tool call is missing a function name.";
      return false;
    }
    if (i == 0) {
      out += content_trimmed.empty() ? "<tool_call>\n" : "\n\n<tool_call>\n";
    } else {
      out += "\n<tool_call>\n";
    }
    out += "<function=" + tc.name + ">\n";
    JVal args;
    const std::string args_text = tc.arguments.empty() ? "{}" : tc.arguments;
    if (!json_parse(args_text, args, err)) {
      err = "Tool call arguments for function \"" + tc.name + "\" are not valid JSON: " + err;
      return false;
    }
    if (args.type == JVal::Type::String) {
      if (!trim(args.str).empty()) {
        err = "Tool call arguments for function \"" + tc.name +
              "\" were passed as a JSON string. Parse them into an object before calling "
              "chat_render.";
        return false;
      }
      // empty string: no parameters, like the template's no-op branch
    } else if (args.type == JVal::Type::Null) {
      // absent arguments: no parameters
    } else if (args.type != JVal::Type::Object) {
      err = "Tool call arguments for function \"" + tc.name +
            "\" must be an object/mapping or a JSON string.";
      return false;
    } else {
      for (const auto &kv : args.fields) {
        out += "<parameter=" + kv.first + ">\n";
        if (kv.second.type == JVal::Type::String) {
          out += kv.second.str;  // strings render raw (`|string`)
        } else {
          out += json_minja(kv.second);  // everything else renders as JSON (`|tojson`)
        }
        out += "\n</parameter>\n";
      }
    }
    out += "</function>\n</tool_call>";
  }
  return true;
}

// --- parsing -----------------------------------------------------------------

// Strips a trailing partial end-tag opener (a non-empty proper prefix of
// `</think>` or `<tool_call>`) left by a streaming prefix, mirroring how the
// PEG parser's until/end rules treat a truncated terminator.
std::string strip_partial_tail(const std::string &s, bool think_too) {
  if (think_too) {
    for (std::size_t len = 1; len < 8 && len <= s.size(); ++len) {
      // proper prefix of "</think>" (8 chars)
      if (std::string(kThinkClose).compare(0, len, s, s.size() - len, len) == 0) {
        return s.substr(0, s.size() - len);
      }
    }
  }
  for (std::size_t len = 1; len < 11 && len <= s.size(); ++len) {
    // proper prefix of "<tool_call>" (11 chars)
    if (std::string(kToolCallOpen).compare(0, len, s, s.size() - len, len) == 0) {
      return s.substr(0, s.size() - len);
    }
  }
  return s;
}

// Renders a still-streaming parameter value: complete JSON keeps its form,
// an open container or string stays verbatim, anything else becomes an
// unterminated JSON string. Mirrors the PEG mapper's args_buffer prefixes so
// streaming clients can append deltas.
std::string emit_partial_value(const std::string &raw) {
  if (!raw.empty() && (raw[0] == '{' || raw[0] == '[' || raw[0] == '"')) return raw;
  JVal val;
  std::string jerr;
  if (json_parse(raw, val, jerr)) return json_compact(val);
  std::string out = "\"";
  json_escape_into(raw, out);
  return out;  // deliberately unterminated
}

bool starts_with_at(const std::string &s, std::size_t pos, const char *tok) {
  const std::string t(tok);
  return s.compare(pos, t.size(), t) == 0;
}

// Parses one "<parameter=name>\nVALUE\n</parameter>\n" into (name, typed JSON
// value). On success `pos` is past the closer. In partial mode a missing
// closer consumes to end-of-input instead of failing.
bool parse_parameter(const std::string &s, std::size_t &pos, bool partial,
                     std::pair<std::string, JVal> &field, std::string &err) {
  // s[pos..] starts with "<parameter="
  const std::size_t entry = pos;
  pos += 11;  // strlen("<parameter=")
  const std::size_t name_end = s.find('>', pos);
  if (name_end == std::string::npos) {
    if (partial) {
      pos = entry;  // truncated opener: nothing usable yet; rewind so the caller stops
      return false;
    }
    err = "unterminated <parameter= opener";
    return false;
  }
  const std::string name = s.substr(pos, name_end - pos);
  pos = name_end + 1;
  if (pos >= s.size() || s[pos] != '\n') {
    if (partial && pos >= s.size()) {
      pos = entry;  // truncated opener line: nothing usable yet
      return false;
    }
    err = "expected newline after <parameter=" + name + ">";
    return false;
  }
  ++pos;
  const std::size_t val_start = pos;
  const std::size_t close = s.find("\n</parameter>", pos);
  std::string raw;
  bool value_partial = false;
  if (close == std::string::npos) {
    if (!partial) {
      err = "unterminated <parameter=" + name + "> (missing \\n</parameter>)";
      return false;
    }
    raw = s.substr(val_start);
    pos = s.size();
    value_partial = true;
  } else {
    raw = s.substr(val_start, close - val_start);
    pos = close + 1 + 12;  // past "\n</parameter>"
    if (pos < s.size() && s[pos] == '\n') ++pos;
  }
  if (value_partial) {
    JVal open;
    open.type = JVal::Type::Raw;
    open.str = emit_partial_value(raw);
    field = {name, open};
    return true;
  }
  JVal val;
  std::string jerr;
  if (json_parse(raw, val, jerr)) {
    // A bare JSON value keeps its type (numbers stay numbers, ...). NOTE: a
    // string-typed parameter holding e.g. "123" types as a number here;
    // llama.cpp disambiguates through the tool schema, which this
    // schema-less parser does not see (see header note).
    field = {name, val};
  } else {
    JVal str;
    str.type = JVal::Type::String;
    str.str = raw;
    field = {name, str};
  }
  return true;
}

// Parses "<tool_call>\n<function=name>\n" + params + "</function>\n</tool_call>"
// at pos. On success appends to out and advances pos past the block.
// In partial mode a block truncated before a usable name rewinds pos to the
// block start and reports "nothing yet" (false with pos unchanged); a block
// with a complete name but truncated body yields the prefix parsed so far.
bool parse_tool_call(const std::string &s, std::size_t &pos, bool partial, ChatToolCall &out,
                     std::string &err) {
  const std::size_t entry = pos;
  // s[pos..] starts with "<tool_call>"
  pos += 11;  // strlen("<tool_call>")
  if (pos < s.size() && s[pos] == '\n') ++pos;
  const std::string func_open = "<function=";
  if (s.compare(pos, func_open.size(), func_open) != 0) {
    if (partial && pos >= s.size()) {
      pos = entry;  // truncated right after the opener; yield nothing yet
      return false;
    }
    err = "expected <function=name> after <tool_call>";
    return false;
  }
  pos += func_open.size();
  const std::size_t name_end = s.find('>', pos);
  if (name_end == std::string::npos) {
    if (partial) {
      pos = entry;  // truncated name; yield nothing yet
      return false;
    }
    err = "unterminated <function= opener";
    return false;
  }
  const std::string name = s.substr(pos, name_end - pos);
  if (name.empty()) {
    err = "Tool call is missing a function name.";
    return false;
  }
  pos = name_end + 1;
  if (pos < s.size() && s[pos] == '\n') {
    ++pos;
  } else if (pos >= s.size()) {
    if (partial) {
      out.name = name;  // name complete, body still streaming
      out.arguments = "{";
      return true;
    }
    err = "expected newline after <function=" + name + ">";
    return false;
  } else {
    err = "expected newline after <function=" + name + ">";
    return false;
  }
  JVal args;
  args.type = JVal::Type::Object;
  while (true) {
    if (s.compare(pos, 11, "</function>") == 0) {
      pos += 11;
      if (pos < s.size() && s[pos] == '\n') ++pos;
      break;
    }
    if (s.compare(pos, 11, "<parameter=") == 0) {
      std::pair<std::string, JVal> field;
      if (!parse_parameter(s, pos, partial, field, err)) {
        if (partial) {
          err.clear();  // truncated opener: keep the parameters parsed so far
          break;
        }
        return false;
      }
      args.fields.push_back(field);
      continue;
    }
    if (partial && pos >= s.size()) break;  // truncated body: keep prefix
    err = "expected <parameter=...>> or </function> in tool call \"" + name + "\"";
    return false;
  }
  // In partial mode a missing </function> just ends the block at end-of-input.
  const std::string closer = "</tool_call>";
  bool closed = false;
  if (s.compare(pos, closer.size(), closer) == 0) {
    pos += closer.size();
    closed = true;
  } else if (!partial) {
    err = "unterminated <tool_call> for function \"" + name + "\" (missing </tool_call>)";
    return false;
  }
  out.name = name;
  out.arguments = json_compact(args);
  if (partial && !closed && !out.arguments.empty()) {
    // Still streaming: drop the object closer so clients can append deltas,
    // mirroring the PEG mapper's args_buffer (braces close at </tool_call>).
    out.arguments.pop_back();
  }
  return true;
}

}  // namespace

const char *chat_reasoning_instructions(const std::string &effort) {
  if (effort == "xhigh" || effort == "high") return kEffortXhigh;
  if (effort == "low") return kEffortLow;
  return "";  // medium, and anything else the template rejects
}

bool chat_render(const std::vector<ChatMessage> &messages, const ChatOptions &opts,
                 std::string &out, std::string &err) {
  out.clear();
  if (messages.empty()) {
    err = "no messages provided";
    return false;
  }
  // The template only looks at reasoning_effort inside its thinking block, so
  // with thinking disabled an unknown effort is never validated there (and must
  // not be rejected here either — review M4).
  if (opts.enable_thinking && opts.reasoning_effort != "xhigh" &&
      opts.reasoning_effort != "high" && opts.reasoning_effort != "medium" &&
      opts.reasoning_effort != "low") {
    err = "unexpected reasoning effort " + opts.reasoning_effort +
          " (supported: xhigh, medium, low)";
    return false;
  }

  // --- merged system message (only the leading system/developer messages) ---
  std::string merged_system;
  std::size_t num_sys = 0;
  for (const ChatMessage &m : messages) {
    if (!is_system_role(m.role)) break;
    const std::string content = trim(m.content);
    if (!content.empty()) {
      if (!merged_system.empty()) merged_system += "\n";
      merged_system += content;
    }
    ++num_sys;
  }

  const char *instructions = opts.enable_thinking ? chat_reasoning_instructions(opts.reasoning_effort)
                                                  : "";
  const bool has_instr = instructions[0] != '\0';

  if (!opts.tools.empty()) {
    // The template's `{% if tools ... %}` branch: one system block with the
    // tool list, the format instructions, and the merged system text.
    out += "<|im_start|>system\n";
    if (has_instr) {
      out += instructions;
      out += "\n\n";
    }
    out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const ChatTool &t : opts.tools) {
      std::string tool_json;
      if (!render_tool_json(t, tool_json, err)) return false;
      out += "\n" + tool_json;
    }
    out += "\n</tools>";
    out += kToolsSuffix;
    if (!merged_system.empty()) {
      out += "\n\n" + merged_system;
    }
    out += "<|im_end|>\n";
  } else if (!merged_system.empty()) {
    out += "<|im_start|>system\n";
    if (has_instr) {
      out += instructions;
      out += "\n\n";
    }
    out += merged_system;
    out += "<|im_end|>\n";
  } else if (has_instr) {
    out += "<|im_start|>system\n";
    out += instructions;
    out += "<|im_end|>\n";
  }

  // --- the conversation ------------------------------------------------
  for (std::size_t i = num_sys; i < messages.size(); ++i) {
    const ChatMessage &m = messages[i];
    const std::string content = trim(m.content);
    if (is_system_role(m.role)) {
      err = "system message must be at the beginning";
      return false;
    }
    if (m.role == "user") {
      out += "<|im_start|>" + m.role + "\n" + content + "<|im_end|>\n";
    } else if (m.role == "assistant") {
      // preserve_thinking is undefined in the template's condition, so the
      // thinking block is always emitted (with the reasoning content, which may
      // be empty).
      out += "<|im_start|>" + m.role + "\n<think>\n" + trim(m.reasoning_content) +
             "\n</think>\n\n" + content;
      if (!render_tool_calls(m.tool_calls, content, out, err)) return false;
      out += "<|im_end|>\n";
    } else if (m.role == "tool") {
      // The template opens a user turn for the first tool message of a group and
      // closes it after the last one.
      const bool prev_is_tool = i > num_sys && messages[i - 1].role == "tool";
      const bool next_is_tool = i + 1 < messages.size() && messages[i + 1].role == "tool";
      if (!prev_is_tool) out += "<|im_start|>user";
      out += "\n<tool_response>\n" + content + "\n</tool_response>";
      if (!next_is_tool) out += "<|im_end|>\n";
    } else {
      err = "unexpected message role: " + m.role;
      return false;
    }
  }

  if (opts.add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    if (opts.enable_thinking) {
      out += "<think>\n";
    } else {
      out += "<think>\n\n</think>\n\n";
    }
  }
  return true;
}

bool chat_parse(const std::string &generated, const ChatParseOptions &opts, ChatParsedMessage &out,
                std::string &err) {
  out = ChatParsedMessage();
  std::string rest = generated;

  if (opts.enable_thinking) {
    // Thinking ends at the first </think>; without one, at the first
    // <tool_call>; without either, the whole turn is reasoning (this is what
    // the PEG optional-reasoning rule produces for plain content). Leading
    // framing whitespace is not part of the reasoning (the parser's space
    // rule); trailing whitespace is kept verbatim.
    const std::size_t think_end = rest.find(kThinkClose);
    if (think_end != std::string::npos) {
      const std::string reasoning = lstrip_space(rest.substr(0, think_end));
      if (!is_all_ws(reasoning)) out.reasoning_content = reasoning;
      rest = lstrip_space(rest.substr(think_end + 8));  // strlen("</think>")
    } else {
      const std::size_t call_at = rest.find(kToolCallOpen);
      std::string reasoning = call_at == std::string::npos ? rest : rest.substr(0, call_at);
      // The reasoning rule ends at </think> or <tool_call>, so a streaming
      // prefix of either terminator is framing, not reasoning.
      if (opts.partial) reasoning = strip_partial_tail(reasoning, /*think_too=*/true);
      reasoning = lstrip_space(reasoning);
      if (!is_all_ws(reasoning)) out.reasoning_content = reasoning;
      rest = call_at == std::string::npos ? "" : rest.substr(call_at);
    }
  }

  // Content runs up to the first <tool_call>. Leading framing whitespace is
  // stripped (the parser's space rule); trailing whitespace is kept verbatim.
  // A partial </think> inside content is literal text (only a partial
  // <tool_call> opener is framing there, matching the content rule's
  // until_one_of set).
  std::size_t pos = 0;
  const std::size_t call_at = rest.find(kToolCallOpen);
  if (call_at != std::string::npos) {
    out.content = lstrip_space(rest.substr(0, call_at));
    pos = call_at;
  } else {
    std::string tail = rest;
    if (opts.partial) tail = strip_partial_tail(tail, /*think_too=*/false);
    out.content = lstrip_space(tail);
    return true;
  }

  // Tool calls. In partial mode a truncated block yields what parsed so far.
  while (pos < rest.size()) {
    // Skip the "\n" separator the renderer puts between blocks (and any stray
    // whitespace); anything else before the next block is malformed.
    std::size_t b = pos;
    while (b < rest.size() && is_ws(rest[b])) ++b;
    if (b >= rest.size()) break;
    if (!starts_with_at(rest, b, kToolCallOpen)) {
      if (opts.partial) break;  // trailing partial opener: stop, keep parsed prefix
      std::string frag = rest.substr(b, 32);
      err = "unexpected text after tool call: \"" + frag + "\"";
      return false;
    }
    // Whitespace between blocks is only framing: keep content as-is (it was
    // fixed before the first block) and continue after it.
    pos = b;
    ChatToolCall tc;
    const std::size_t before = pos;
    if (!parse_tool_call(rest, pos, opts.partial, tc, err)) {
      // Partial mode only swallows truncation at the block start (pos rewound
      // to `before` by the callee); anything else is malformed input.
      if (!(opts.partial && pos == before)) return false;
      break;
    }
    if (tc.name.empty()) break;  // unreachable; defensive
    out.tool_calls.push_back(tc);
  }
  return true;
}

}  // namespace rdna4
