// Test tool — llama.cpp's own chat-template rendering (minja, the real Jinja
// engine), for a fixed matrix of conversations.
//
// `--chat` must produce exactly the prompt the model was trained on, and the
// template is a 10 KB Jinja file. Rather than reimplementing Jinja, this engine
// implements the qwen35 format directly (SPEC.md §2 rules out other models); this
// tool is the reference it is validated against.
//
// usage: oracle-chat <file.gguf>
// Output: one record per case:
//   # case <n> <name> add_assistant=<0|1> thinking=<0|1> effort=<xhigh|medium|low|->
//   <prompt with \n \t \\ escaped>
//
// `effort=-` means the template kwarg was not passed (the template's own
// default, xhigh); the other rows pin `chat_reasoning_instructions()` for every
// value the template accepts, including the `high` -> `xhigh` alias.
#include "chat.h"

#include <llama.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

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

struct Case {
  const char *name;
  std::vector<common_chat_msg> messages;
  bool add_assistant;
  bool thinking;
  const char *effort = nullptr;  // reasoning_effort kwarg; null = template default
};

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  auto msg = [](const char *role, const char *content) {
    common_chat_msg m;
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
    common_chat_msg a = msg("assistant", "The answer is 4.");
    a.reasoning_content = "Let me think about it.";
    cases.push_back({"assistant-thinking", {msg("user", "What is 2+2?"), a}, true, true});
  }
  {
    common_chat_msg t = msg("tool", "{\"result\": 4}");
    cases.push_back({"tool-after-assistant",
                     {msg("user", "What is 2+2?"), msg("assistant", ""), t}, true, true});
  }
  // reasoning_effort: every value the template accepts, with and without a user
  // system message (the template emits the instruction either as its own system
  // message or prepended to the user's, and emits nothing for `medium`).
  cases.push_back({"effort-xhigh", {msg("user", "Hi")}, true, true, "xhigh"});
  cases.push_back({"effort-high-alias", {msg("user", "Hi")}, true, true, "high"});
  cases.push_back({"effort-medium", {msg("user", "Hi")}, true, true, "medium"});
  cases.push_back({"effort-low", {msg("user", "Hi")}, true, true, "low"});
  cases.push_back({"effort-medium-system",
                   {msg("system", "You are terse."), msg("user", "Hi")}, true, true, "medium"});
  cases.push_back({"effort-low-system",
                   {msg("system", "You are terse."), msg("user", "Hi")}, true, true, "low"});
  cases.push_back({"effort-low-no-thinking", {msg("user", "Hi")}, true, false, "low"});
  return cases;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: oracle-chat <file.gguf>\n");
    return 2;
  }
  llama_backend_init();
  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 0;
  mparams.vocab_only = true;
  llama_model *model = llama_model_load_from_file(argv[1], mparams);
  if (!model) {
    std::fprintf(stderr, "load failed\n");
    return 1;
  }

  common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
  const std::vector<Case> cases = make_cases();
  std::size_t n = 0;
  for (const Case &c : cases) {
    common_chat_templates_inputs in;
    in.messages = c.messages;
    in.add_generation_prompt = c.add_assistant;
    in.enable_thinking = c.thinking;
    in.use_jinja = true;
    if (c.effort != nullptr) {
      // chat_template_kwargs values are JSON, so a string needs its quotes
      in.chat_template_kwargs["reasoning_effort"] = std::string("\"") + c.effort + "\"";
    }
    const common_chat_params params = common_chat_templates_apply(tmpls.get(), in);
    std::printf("# case %zu %s add_assistant=%d thinking=%d effort=%s\n", n, c.name,
                (int)c.add_assistant, (int)c.thinking, c.effort ? c.effort : "-");
    std::printf("%s\n", escape(params.prompt).c_str());
    ++n;
  }
  std::printf("# ok %zu cases\n", n);

  llama_model_free(model);
  llama_backend_free();
  return 0;
}
