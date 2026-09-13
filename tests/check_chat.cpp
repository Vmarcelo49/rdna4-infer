// M4 — `--chat` prompt rendering against llama.cpp's Jinja rendering.
//
// The engine renders the qwen35 chat format directly (SPEC.md §2 rules out other
// models, so a Jinja engine is out of scope); this test is what makes that
// legitimate: the same 13 conversations go through llama.cpp's own template
// engine (tests/oracle_chat.cpp, minja + the metadata template) and through
// chat_render(), and the prompts must be byte-identical.
//
// usage: check-chat <golden_renders.txt>
//   regenerate with: oracle-chat <file.gguf> > tests/golden/chat_renders.txt
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "rdna4/chat.h"

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

std::string escape(const std::string &in) {
  std::string out;
  for (char c : in) {
    switch (c) {
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\\': out += "\\\\"; break;
      default: out.push_back(c);
    }
  }
  return out;
}

// The same case matrix as tests/oracle_chat.cpp, in the same order.
struct Case {
  const char *name;
  std::vector<rdna4::ChatMessage> messages;
  bool add_assistant;
  bool thinking;
};

std::vector<Case> make_cases() {
  using rdna4::ChatMessage;
  std::vector<Case> cases;
  auto msg = [](const char *role, const char *content) {
    ChatMessage m;
    m.role = role;
    m.content = content;
    return m;
  };

  cases.push_back({"user-only", {msg("user", "Hello")}, true, true});
  cases.push_back({"user-only-no-gen", {msg("user", "Hello")}, false, true});
  cases.push_back({"user-only-no-thinking", {msg("user", "Hello")}, true, false});
  cases.push_back(
      {"system+user", {msg("system", "You are a helpful assistant."), msg("user", "Hello")}, true, true});
  cases.push_back({"developer+user", {msg("developer", "Be terse."), msg("user", "Hi")}, true, true});
  cases.push_back({"two-systems",
                   {msg("system", "First."), msg("system", "Second."), msg("user", "Hi")}, true, true});
  cases.push_back({"multi-turn",
                   {msg("user", "What is 2+2?"), msg("assistant", "4"), msg("user", "And 3+3?")}, true,
                   true});
  cases.push_back({"empty-system", {msg("system", ""), msg("user", "Hi")}, true, true});
  cases.push_back({"long-user",
                   {msg("user", "Hello world, this is a test. Please reply briefly.")}, true, true});
  cases.push_back({"unicode", {msg("user", "Olá, café ☕ 日本語")}, true, true});
  cases.push_back({"newlines", {msg("user", "line1\nline2\n\nline3")}, true, true});
  {
    ChatMessage a = msg("assistant", "The answer is 4.");
    a.reasoning_content = "Let me think about it.";
    cases.push_back({"assistant-thinking", {msg("user", "What is 2+2?"), a}, true, true});
  }
  {
    ChatMessage t = msg("tool", "{\"result\": 4}");
    cases.push_back({"tool-after-assistant",
                     {msg("user", "What is 2+2?"), msg("assistant", ""), t}, true, true});
  }
  return cases;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-chat <golden_renders.txt>\n");
    return 2;
  }
  std::ifstream f(argv[1]);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 1;
  }

  std::vector<std::string> ref;
  std::vector<std::string> ref_names;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("# case ", 0) == 0) {
      const std::size_t p = line.find(' ', 7);
      ref_names.push_back(p == std::string::npos ? line : line.substr(7, p - 7));
      continue;
    }
    if (line.rfind("#", 0) == 0 || line.empty()) continue;
    ref.push_back(unescape(line));
  }

  const std::vector<Case> cases = make_cases();
  int failures = 0;
  std::printf("cases: ours %zu, reference %zu\n", cases.size(), ref.size());
  if (ref.size() != cases.size()) {
    std::printf("FAIL case count mismatch\n");
    ++failures;
  }
  for (std::size_t i = 0; i < cases.size(); ++i) {
    rdna4::ChatOptions opts;
    opts.add_generation_prompt = cases[i].add_assistant;
    opts.enable_thinking = cases[i].thinking;
    std::string out, err;
    const bool ok = rdna4::chat_render(cases[i].messages, opts, out, err);
    if (!ok) {
      std::printf("FAIL case %zu (%s): %s\n", i, cases[i].name, err.c_str());
      ++failures;
      continue;
    }
    if (i >= ref.size()) {
      std::printf("FAIL case %zu (%s): no reference\n", i, cases[i].name);
      ++failures;
      continue;
    }
    const bool same = out == ref[i];
    std::printf("%-5s case %2zu %-24s (%zu bytes)\n", same ? "ok" : "FAIL", i, cases[i].name,
                out.size());
    if (!same) {
      std::printf("  ours: %s\n  ref : %s\n", escape(out).c_str(), escape(ref[i]).c_str());
      ++failures;
    }
  }

  // error paths the template raises on
  {
    std::string out, err;
    rdna4::ChatOptions o;
    const bool ok = rdna4::chat_render({}, o, out, err);
    std::printf("%-5s empty conversation rejected (%s)\n", !ok ? "ok" : "FAIL", err.c_str());
    if (ok) ++failures;

    std::vector<rdna4::ChatMessage> m = {{"user", "hi"}, {"system", "late"}};
    const bool ok2 = rdna4::chat_render(m, o, out, err);
    std::printf("%-5s system after the first turn rejected (%s)\n", !ok2 ? "ok" : "FAIL", err.c_str());
    if (ok2) ++failures;

    std::vector<rdna4::ChatMessage> m2 = {{"robot", "hi"}};
    const bool ok3 = rdna4::chat_render(m2, o, out, err);
    std::printf("%-5s unknown role rejected (%s)\n", !ok3 ? "ok" : "FAIL", err.c_str());
    if (ok3) ++failures;
  }

  std::printf("check-chat: %s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
