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

// The same case matrix as tests/oracle_chat.cpp, in the same order, plus the
// tools cases (20+), which mirror /tmp/oracle_tools.cpp's scripted
// conversations rendered through llama.cpp's Jinja engine on the qwen35
// (Qwen3.8-family) template. Tool-call outputs are validated twice: the
// render half against the golden file, the parse-back half (chat_parse) by
// exact in-code expectations below.
struct Case {
  const char *name;
  std::vector<rdna4::ChatMessage> messages;
  bool add_assistant;
  bool thinking;
  const char *effort = "xhigh";  // reasoning_effort; the oracle's "-" row is xhigh
  std::vector<rdna4::ChatTool> tools;
};

std::vector<Case> make_cases() {
  using rdna4::ChatMessage;
  using rdna4::ChatTool;
  using rdna4::ChatToolCall;
  std::vector<Case> cases;
  auto msg = [](const char *role, const char *content) {
    ChatMessage m;
    m.role = role;
    m.content = content;
    return m;
  };
  auto tool = [](const char *name, const char *desc, const char *params) {
    ChatTool t;
    t.name = name;
    t.description = desc;
    t.parameters = params;
    return t;
  };
  auto call = [](const char *name, const char *args) {
    ChatToolCall c;
    c.name = name;
    c.arguments = args;
    return c;
  };
  const ChatTool kWeather =
      tool("get_weather", "Get the weather for a city",
           "{\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, "
           "\"units\": {\"type\": \"string\"}}, \"required\": [\"city\"]}");
  const ChatTool kAdd =
      tool("add", "Add two numbers",
           "{\"type\": \"object\", \"properties\": {\"a\": {\"type\": \"number\"}, "
           "\"b\": {\"type\": \"number\"}}, \"required\": [\"a\", \"b\"]}");
  const ChatTool kCode =
      tool("run_code", "Run code", "{\"type\": \"object\", \"properties\": {\"code\": "
                                   "{\"type\": \"string\"}}}");
  const ChatTool kNoop =
      tool("noop", "Does nothing", "{\"type\": \"object\", \"properties\": {}}");

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
  // reasoning_effort: every value the template accepts (including the
  // `high` -> `xhigh` alias and `medium`, which emits no instruction at all),
  // with and without a user system message to prepend to.
  cases.push_back({"effort-xhigh", {msg("user", "Hi")}, true, true, "xhigh"});
  cases.push_back({"effort-high-alias", {msg("user", "Hi")}, true, true, "high"});
  cases.push_back({"effort-medium", {msg("user", "Hi")}, true, true, "medium"});
  cases.push_back({"effort-low", {msg("user", "Hi")}, true, true, "low"});
  cases.push_back({"effort-medium-system",
                   {msg("system", "You are terse."), msg("user", "Hi")}, true, true, "medium"});
  cases.push_back({"effort-low-system",
                   {msg("system", "You are terse."), msg("user", "Hi")}, true, true, "low"});
  cases.push_back({"effort-low-no-thinking", {msg("user", "Hi")}, true, false, "low"});

  // --- tool-calling renders (validated against the Jinja oracle) ---
  cases.push_back({"tools-user", {msg("user", "What is the weather in Paris?")}, true, true,
                   "xhigh", {kWeather}});
  cases.push_back({"tools-system-user",
                   {msg("system", "You are terse."), msg("user", "What is the weather in Paris?")},
                   true, true, "xhigh", {kWeather}});
  {
    ChatMessage a = msg("assistant", "");
    a.reasoning_content = "I should check the weather.";
    a.tool_calls = {call("get_weather", "{\"city\": \"Paris\", \"units\": \"metric\"}")};
    cases.push_back({"tools-single-call",
                     {msg("user", "Weather in Paris?"), a, msg("tool", "{\"temp\": 21}")}, true,
                     true, "xhigh", {kWeather}});
  }
  {
    ChatMessage a = msg("assistant", "Let me look that up.");
    a.reasoning_content = "Two lookups needed.";
    a.tool_calls = {call("get_weather", "{\"city\": \"Paris\"}"),
                    call("get_weather", "{\"city\": \"Rome\", \"units\": \"metric\"}")};
    cases.push_back({"tools-parallel", {msg("user", "Weather in Paris and Rome?"), a}, true, true,
                     "xhigh", {kWeather}});
  }
  {
    ChatMessage a = msg("assistant", "");
    a.tool_calls = {call("add", "{\"a\": 2, \"b\": 3.5}")};
    cases.push_back(
        {"tools-nonstr-args", {msg("user", "Add 2 and 3.5"), a}, true, true, "xhigh", {kAdd}});
  }
  {
    ChatMessage a = msg("assistant", "");
    a.reasoning_content = "Running it.";
    a.tool_calls = {call("run_code", "{\"code\": \"def hello():\\n    print(\\\"Hi\\\")\\n\\nhello()\"}")};
    cases.push_back({"tools-code-arg", {msg("user", "Run it"), a}, true, true, "xhigh", {kCode}});
  }
  {
    ChatMessage a = msg("assistant", "");
    a.tool_calls = {call("noop", "{}")};
    cases.push_back(
        {"tools-empty-args", {msg("user", "Do nothing"), a}, true, true, "xhigh", {kNoop}});
  }
  cases.push_back({"tools-no-thinking", {msg("user", "Weather in Paris?")}, true, false, "xhigh",
                   {kWeather}});
  cases.push_back(
      {"tools-low", {msg("user", "Weather in Paris?")}, true, true, "low", {kWeather}});
  cases.push_back(
      {"tools-all", {msg("user", "Hi")}, true, true, "xhigh", {kWeather, kAdd, kCode, kNoop}});
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
  std::vector<std::string> ref_effort;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("# case ", 0) == 0) {
      const std::size_t p = line.find(' ', 7);
      ref_names.push_back(p == std::string::npos ? line : line.substr(7, p - 7));
      // effort=<x|medium|low|-> records the reasoning_effort kwarg the oracle
      // passed; a case we render with a different effort would otherwise compare
      // against the wrong reference silently
      const std::string k = "effort=";
      const std::size_t e = line.find(k);
      ref_effort.push_back(e == std::string::npos ? std::string("-")
                                                  : line.substr(e + k.size()));
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
    opts.reasoning_effort = cases[i].effort;
    opts.tools = cases[i].tools;
    // refuse to compare against a reference rendered with another effort: the
    // failure would look like a rendering bug instead of a stale golden
    if (i < ref_effort.size()) {
      const std::string want = ref_effort[i] == "-" ? std::string("xhigh") : ref_effort[i];
      if (want != cases[i].effort) {
        std::printf("FAIL case %zu (%s): golden was rendered with effort=%s, test uses %s\n", i,
                    cases[i].name, ref_effort[i].c_str(), cases[i].effort);
        ++failures;
        continue;
      }
    }
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

    // the template raises for an effort outside ('xhigh','medium','low') after
    // mapping 'high' to 'xhigh'
    o.reasoning_effort = "ultra";
    const bool ok4 = rdna4::chat_render({{"user", "hi"}}, o, out, err);
    std::printf("%-5s unknown reasoning effort rejected (%s)\n", !ok4 ? "ok" : "FAIL", err.c_str());
    if (ok4) ++failures;

    // tools render errors mirror the template's raise_exception paths
    {
      rdna4::ChatTool bad;
      bad.name = "";
      rdna4::ChatOptions to;
      to.tools = {bad};
      const bool okb = rdna4::chat_render({{"user", "hi"}}, to, out, err);
      std::printf("%-5s nameless tool rejected (%s)\n", !okb ? "ok" : "FAIL", err.c_str());
      if (okb) ++failures;
    }
    {
      rdna4::ChatTool bad;
      bad.name = "f";
      bad.parameters = "{oops";
      rdna4::ChatOptions to;
      to.tools = {bad};
      const bool okb = rdna4::chat_render({{"user", "hi"}}, to, out, err);
      std::printf("%-5s bad tool parameters rejected (%s)\n", !okb ? "ok" : "FAIL", err.c_str());
      if (okb) ++failures;
    }
    {
      rdna4::ChatMessage a;
      a.role = "assistant";
      rdna4::ChatToolCall c;
      c.name = "";
      a.tool_calls = {c};
      rdna4::ChatOptions ao;  // fresh opts: the "ultra" effort above must not leak in
      const bool okb = rdna4::chat_render({a}, ao, out, err);
      std::printf("%-5s nameless tool call rejected (%s)\n", !okb ? "ok" : "FAIL", err.c_str());
      if (okb) ++failures;
    }
    {
      rdna4::ChatMessage a;
      a.role = "assistant";
      rdna4::ChatToolCall c;
      c.name = "f";
      c.arguments = "\"just a string\"";
      a.tool_calls = {c};
      rdna4::ChatOptions ao;
      const bool okb = rdna4::chat_render({a}, ao, out, err);
      std::printf("%-5s string tool arguments rejected (%s)\n", !okb ? "ok" : "FAIL", err.c_str());
      if (okb) ++failures;
    }
  }

  // parse-back (chat_parse): exact expectations mirroring llama.cpp's
  // common_chat_parse with the qwen3-coder PEG parser on the same inputs
  // (verified against the /tmp differential oracle during development).
  {
    struct PCall {
      const char *name;
      const char *args;
    };
    struct PCase {
      const char *name;
      const char *input;
      bool thinking;
      bool partial;
      const char *reasoning;
      const char *content;
      std::vector<PCall> calls;
    };
    const std::vector<PCase> pcases = {
        {"think-content", "I need to think.\n</think>\n\nHello there", true, false,
         "I need to think.\n", "Hello there", {}},
        {"content-only", "Just an answer.", true, false, "Just an answer.", "", {}},
        {"single-call",
         "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
         false, false, "", "", {{"get_weather", "{\"city\":\"Paris\"}"}}},
        {"think-single-call",
         "Checking now.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=units>\nmetric\n</parameter>\n</function>\n</tool_call>",
         true, false, "Checking now.\n", "",
         {{"get_weather", "{\"city\":\"Paris\",\"units\":\"metric\"}"}}},
        {"content-single-call",
         "Let me look that up.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
         false, false, "", "Let me look that up.\n",
         {{"get_weather", "{\"city\":\"Paris\"}"}}},
        {"parallel-calls",
         "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n<parameter=units>\nmetric\n</parameter>\n</function>\n</tool_call>",
         false, false, "", "",
         {{"get_weather", "{\"city\":\"Paris\"}"},
          {"get_weather", "{\"city\":\"Rome\",\"units\":\"metric\"}"}}},
        {"numeric-args",
         "<tool_call>\n<function=add>\n<parameter=a>\n2\n</parameter>\n<parameter=b>\n3.5\n</parameter>\n</function>\n</tool_call>",
         false, false, "", "", {{"add", "{\"a\":2,\"b\":3.5}"}}},
        {"code-arg",
         "<tool_call>\n<function=run_code>\n<parameter=code>\ndef hello():\n    print(\"Hi\")\n\nhello()\n</parameter>\n</function>\n</tool_call>",
         false, false, "", "",
         {{"run_code", "{\"code\":\"def hello():\\n    print(\\\"Hi\\\")\\n\\nhello()\"}"}}},
        {"trailing-nl-value",
         "<tool_call>\n<function=run_code>\n<parameter=code>\nline1\nline2\n\n</parameter>\n</function>\n</tool_call>",
         false, false, "", "", {{"run_code", "{\"code\":\"line1\\nline2\\n\"}"}}},
        {"noop-call", "<tool_call>\n<function=noop>\n</function>\n</tool_call>", false, false, "",
         "", {{"noop", "{}"}}},
        {"empty-think", "</think>\n\nHi", true, false, "", "Hi", {}},
        {"nothink-thinktext", "<think>\nHi\n</think>\n\nYo", false, false, "",
         "<think>\nHi\n</think>\n\nYo", {}},
        {"think-then-call-no-close",
         "Plans here. <tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
         true, false, "Plans here. ", "", {{"get_weather", "{\"city\":\"Paris\"}"}}},
        // streaming prefixes (partial=true): unterminated blocks yield prefixes
        {"partial-think", "I need to th", true, true, "I need to th", "", {}},
        {"partial-think-close", "I need to think.\n</thi", true, true, "I need to think.\n", "",
         {}},
        {"partial-tool", "Checking.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
         true, true, "Checking.\n", "", {{"get_weather", "{\"city\":\"Par"}}},
        {"partial-second-call",
         "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRo",
         false, true, "", "",
         {{"get_weather", "{\"city\":\"Paris\"}"}, {"get_weather", "{\"city\":\"Ro"}}},
        {"partial-call-opener", "abc<tool_cal", true, true, "abc", "", {}},
        {"partial-call-opener-nothink", "content here <tool_cal", false, true, "",
         "content here ", {}},
        {"partial-think-partial-close2", "abc</think>def</th", true, true, "abc", "def</th", {}},
    };
    for (const PCase &c : pcases) {
      rdna4::ChatParseOptions po;
      po.enable_thinking = c.thinking;
      po.partial = c.partial;
      rdna4::ChatParsedMessage got;
      std::string perr;
      const bool ok = rdna4::chat_parse(c.input, po, got, perr);
      bool same = ok && got.reasoning_content == c.reasoning && got.content == c.content &&
                  got.tool_calls.size() == c.calls.size();
      if (same) {
        for (std::size_t i = 0; i < c.calls.size(); ++i) {
          if (got.tool_calls[i].name != c.calls[i].name ||
              got.tool_calls[i].arguments != c.calls[i].args) {
            same = false;
            break;
          }
        }
      }
      std::printf("%-5s parse %-28s\n", same ? "ok" : "FAIL", c.name);
      if (!same) {
        std::printf("  reasoning: [%s] want [%s]\n  content: [%s] want [%s]\n",
                    escape(got.reasoning_content).c_str(), escape(c.reasoning).c_str(),
                    escape(got.content).c_str(), escape(c.content).c_str());
        for (std::size_t i = 0; i < got.tool_calls.size(); ++i) {
          std::printf("  got call [%s %s]\n", got.tool_calls[i].name.c_str(),
                      escape(got.tool_calls[i].arguments).c_str());
        }
        ++failures;
      }
    }
    // strict-mode malformed turns are errors, not silent truncations
    const char *bad_inputs[] = {
        "<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n",  // no </tool_call>
        "<tool_call>\n<function=f>\n<parameter=x>\n1\n",  // no </parameter>
        "<tool_call>\nGARBAGE",                          // no <function=>
        "<tool_call>\n<function=>\n</function>\n</tool_call>",  // empty name
        "content <tool_call> trailing junk, no block",          // junk after content
    };
    for (const char *in : bad_inputs) {
      rdna4::ChatParseOptions po;
      po.enable_thinking = false;
      rdna4::ChatParsedMessage got;
      std::string perr;
      const bool ok = rdna4::chat_parse(in, po, got, perr);
      std::printf("%-5s parse rejects [%.40s] (%s)\n", !ok ? "ok" : "FAIL", in, perr.c_str());
      if (ok) ++failures;
    }
    // message-level round trip (mirrors llama.cpp's expect_reconstruction):
    // parse a continuation, re-render the assistant turn, and require the
    // generation prompt + continuation back (modulo the closing turn tag).
    {
      struct RCase {
        const char *continuation;
        bool thinking;
        const char *gen_prompt;
      };
      const RCase rcases[] = {
          {"Checking now.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
           true, "<|im_start|>assistant\n<think>\n"},
          // NOTE: render trims message content and always joins it to the
          // first tool call with "\n\n" (the template's `{% if content|trim %}`
          // branch), so a content+tool-call turn is text-stable only when the
          // content already ends with exactly "\n\n" — same normalization
          // boundary as llama.cpp's expect_reconstruction, which is asserted
          // only on stable shapes (call-only / content-only turns).
          {"Let me look that up.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
           false, "<|im_start|>assistant\n<think>\n\n</think>\n\n"},
      };
      for (const RCase &c : rcases) {
        rdna4::ChatParseOptions po;
        po.enable_thinking = c.thinking;
        rdna4::ChatParsedMessage pm;
        std::string perr, rerr, rerender;
        bool ok = rdna4::chat_parse(c.continuation, po, pm, perr);
        rdna4::ChatMessage am;
        am.role = "assistant";
        am.content = pm.content;
        am.reasoning_content = pm.reasoning_content;
        am.tool_calls = pm.tool_calls;
        rdna4::ChatOptions ro;
        ro.add_generation_prompt = false;
        ro.enable_thinking = c.thinking;
        if (ok) ok = rdna4::chat_render({am}, ro, rerender, rerr);
        // With thinking on and no user system message, render prepends the
        // reasoning-effort system block; account for it in the expectation.
        std::string sys;
        if (c.thinking) {
          sys = std::string("<|im_start|>system\n") +
                rdna4::chat_reasoning_instructions("xhigh") + "<|im_end|>\n";
        }
        const std::string want = sys + std::string(c.gen_prompt) + c.continuation + "<|im_end|>\n";
        const bool same = ok && rerender == want;
        std::printf("%-5s round-trip %-12s\n", same ? "ok" : "FAIL",
                    c.thinking ? "thinking" : "no-thinking");
        if (!same) {
          std::printf("  got : %s\n  want: %s\n", escape(rerender).c_str(), escape(want).c_str());
          ++failures;
        }
      }
    }
  }

  std::printf("check-chat: %s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
