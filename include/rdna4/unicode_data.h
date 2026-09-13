// Declarations for the vendored Unicode tables (see unicode_data.cpp).
#pragma once

#include <cstdint>
#include <initializer_list>
#include <unordered_set>
#include <utility>

inline constexpr uint32_t kMaxCodepoints = 0x110000;

// (codepoint_start, flags) ranges; the flags end at the next start.
extern const std::initializer_list<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
extern const std::unordered_set<uint32_t> unicode_set_whitespace;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_lowercase;
