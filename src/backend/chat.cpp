// qwen35 chat rendering (see include/rdna4/chat.h).
#include "rdna4/chat.h"

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

bool is_system_role(const std::string &role) {
  return role == "system" || role == "developer";
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
  if (opts.reasoning_effort != "xhigh" && opts.reasoning_effort != "high" &&
      opts.reasoning_effort != "medium" && opts.reasoning_effort != "low") {
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

  if (!merged_system.empty()) {
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

}  // namespace rdna4
