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
//   - user:      <|im_start|>user\n{content}<|im_end|>\n
//   - assistant: <|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>\n
//   - tool:      <|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n
//   - generation prompt: <|im_start|>assistant\n<think>\n, or ...\n<think>\n\n</think>\n\n
//     when thinking is disabled.
#pragma once

#include <string>
#include <vector>

namespace rdna4 {

struct ChatMessage {
  std::string role;               // system | developer | user | assistant | tool
  std::string content;
  std::string reasoning_content;  // assistant only
};

struct ChatOptions {
  bool add_generation_prompt = true;
  bool enable_thinking = true;
  std::string reasoning_effort = "xhigh";  // xhigh | medium | low (high == xhigh)
};

// Renders the prompt. Returns false with `err` set for an invalid conversation
// (unknown role, system message after the first turn), mirroring the template's
// raise_exception paths.
bool chat_render(const std::vector<ChatMessage> &messages, const ChatOptions &opts,
                 std::string &out, std::string &err);

// The instruction the template inserts when thinking is enabled. Exposed so the
// test can assert the exact text without duplicating it.
const char *chat_reasoning_instructions(const std::string &effort);

}  // namespace rdna4
