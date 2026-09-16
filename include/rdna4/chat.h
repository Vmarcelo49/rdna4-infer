// Chat prompt rendering for qwen35 (PLAN.md M4, `--chat`).
//
// The GGUF carries a ~10 KB Jinja template (tokenizer.chat_template). SPEC.md §2
// rules out other models, so instead of implementing a Jinja engine this renders
// the qwen35 format directly, written from that template's structure and
// validated token for token against llama.cpp's Jinja rendering
// (tests/oracle_chat.cpp → tests/golden/chat_renders.txt).
//
// What the template does (quoted structure):
//   - leading system/developer messages are merged (joined with "\n") into one
//     system message, each trimmed and empty ones skipped;
//   - when thinking is enabled (the default) a `reasoning_effort` instruction
//     (default "xhigh") is emitted as its own system message when there is no
//     user system message, or prepended to it as `instructions + "\n\n" + system`;
//   - tools (OpenAI-style function tools): a `# Tools` system block carrying
//     each tool serialized as {"type": "function", "function": {...}} JSON,
//     followed by the <tool_call> format instructions and the merged system
//     text, if any. Mirrors llama.cpp's common_chat_tools_to_json_oaicompat +
//     the template's `{% if tools ... %}` branch (minja `tojson`: ", " / ": "
//     separators, insertion-ordered keys).
//   - user:      <|im_start|>user\n{content}<|im_end|>\n
//   - assistant: <|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n
//     plus one <tool_call>\n<function={name}>\n<parameter={p}>\n{v}\n</parameter>\n...
//     </function>\n</tool_call> block per entry of message.tool_calls (string
//     argument values render raw, every other JSON value renders as JSON).
//   - tool:      <|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n
//   - generation prompt: <|im_start|>assistant\n<think>\n, or ...\n<think>\n\n</think>\n\n
//     when thinking is disabled.
//
// Parse-back (llama.cpp: common_chat_parse with the qwen3-coder PEG parser,
// which the upstream code selects for this template family — see
// common_chat_try_specialized_template: "Qwen3-Coder XML tool calls, also used
// by ... Qwen3.5 ..."):
//   chat_parse() takes the raw model continuation produced after the generation
//   prompt and splits it into (reasoning_content, content, tool_calls) with the
//   same boundary rules the PEG parser applies: thinking ends at the first
//   </think> (or at the first <tool_call>); the rest is content up to the first
//   <tool_call> followed by <tool_call> blocks. `partial=true` mirrors
//   is_partial=true for streaming: unterminated blocks yield prefix results
//   instead of errors. Parsed argument objects are canonical compact JSON
//   ({"city": "Paris"} renders back from {"city":"Paris"}); a parameter value
//   that is not valid JSON is kept as a string. NOTE: without the tool schemas
//   a bare scalar such as 123 or true always types as JSON (llama.cpp types it
//   through the parameter schema, so a string-typed "123" would stay a string
//   there); any function name is accepted (llama.cpp constrains names through
//   the generation grammar, which needs the tool list at sampler level).
#pragma once

#include <string>
#include <vector>

namespace rdna4 {

struct ChatMessage {
  std::string role;               // system | developer | user | assistant | tool
  std::string content;
  std::string reasoning_content;  // assistant only
  std::vector<struct ChatToolCall> tool_calls;  // assistant only
};

// A function tool definition. `parameters` is a JSON object text (the tool's
// JSON-Schemaish parameters object, e.g. {"type": "object", "properties": {...}};
// "{}" for a parameter-less tool). Rendered into the # Tools system block as
// {"type": "function", "function": {"name", "description", "parameters"}}.
struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters = "{}";
};

// One assistant tool call. `arguments` is a JSON object text mapping parameter
// names to values, e.g. {"city": "Paris", "n": 3} ("{}" for no arguments).
// String values render raw inside <parameter> blocks; every other JSON value
// renders as JSON — mirroring the template's
// `args_value | string if string else tojson` rule.
struct ChatToolCall {
  std::string name;
  std::string arguments = "{}";
};

struct ChatOptions {
  bool add_generation_prompt = true;
  bool enable_thinking = true;
  std::string reasoning_effort = "xhigh";  // xhigh | medium | low (high == xhigh)
  std::vector<ChatTool> tools;             // empty == no # Tools system block
};

// Renders the prompt. Returns false with `err` set for an invalid conversation
// (unknown role, system message after the first turn), mirroring the template's
// raise_exception paths.
bool chat_render(const std::vector<ChatMessage> &messages, const ChatOptions &opts,
                 std::string &out, std::string &err);

// The instruction the template inserts when thinking is enabled. Exposed so the
// test can assert the exact text without duplicating it.
const char *chat_reasoning_instructions(const std::string &effort);

struct ChatParsedMessage {
  std::string reasoning_content;  // empty unless thinking was extracted
  std::string content;
  std::vector<ChatToolCall> tool_calls;  // arguments: canonical compact JSON
};

struct ChatParseOptions {
  bool enable_thinking = true;  // false: no <think> handling, content is literal
  bool partial = false;         // true: streaming prefix; unterminated blocks
                                // yield prefix results instead of errors
};

// Parses a raw model continuation (the text generated after the generation
// prompt) back into reasoning / content / tool calls. Returns false with `err`
// set when the text is not a well-formed assistant turn (strict mode only;
// partial mode never fails on truncation, only on garbage before it).
bool chat_parse(const std::string &generated, const ChatParseOptions &opts,
                ChatParsedMessage &out, std::string &err);

}  // namespace rdna4
