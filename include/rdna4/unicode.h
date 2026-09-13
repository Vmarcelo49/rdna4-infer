// Minimal Unicode helpers for the qwen35 BPE tokenizer (PLAN.md M4).
//
// The struct and the declarations mirror llama.cpp's src/unicode.h (MIT); the
// implementations live in src/backend/unicode.cpp and the generated tables in
// src/backend/unicode_data.cpp, both vendored from the same revision
// (df03399b8) restricted to what this vocab's pre-tokenizer uses.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct unicode_cpt_flags {
    enum {
        UNDEFINED       = 0x0001,
        NUMBER          = 0x0002,  // regex: \p{N}
        LETTER          = 0x0004,  // regex: \p{L}
        SEPARATOR       = 0x0008,  // regex: \p{Z}
        ACCENT_MARK     = 0x0010,  // regex: \p{M}
        PUNCTUATION     = 0x0020,  // regex: \p{P}
        SYMBOL          = 0x0040,  // regex: \p{S}
        CONTROL         = 0x0080,  // regex: \p{C}
        MASK_CATEGORIES = 0x00FF,
        WHITESPACE      = 0x0100,
        LOWERCASE       = 0x0200,
        UPPERCASE       = 0x0400,
        NFD             = 0x0800,
    };

    uint16_t is_undefined   : 1;
    uint16_t is_number      : 1;
    uint16_t is_letter      : 1;
    uint16_t is_separator   : 1;
    uint16_t is_accent_mark : 1;
    uint16_t is_punctuation : 1;
    uint16_t is_symbol      : 1;
    uint16_t is_control     : 1;
    uint16_t is_whitespace  : 1;
    uint16_t is_lowercase   : 1;
    uint16_t is_uppercase   : 1;
    uint16_t is_nfd         : 1;

    inline unicode_cpt_flags(const uint16_t flags = 0) {
        static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                      "the vendored flag decoding assumes little endian");
        *reinterpret_cast<uint16_t *>(this) = flags;
    }

    inline uint16_t as_uint() const {
        return *reinterpret_cast<const uint16_t *>(this);
    }

    inline uint16_t category_flag() const { return as_uint() & MASK_CATEGORIES; }
};

size_t unicode_len_utf8(char src);

std::string unicode_cpt_to_utf8(uint32_t cpt);
uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset);
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8);

unicode_cpt_flags unicode_cpt_flags_from_cpt(uint32_t cpt);
unicode_cpt_flags unicode_cpt_flags_from_utf8(const std::string & utf8);

std::string unicode_byte_to_utf8(uint8_t byte);
uint8_t unicode_utf8_to_byte(const std::string & utf8);
// Non-throwing variant of unicode_utf8_to_byte (added here, not upstream): the
// upstream version is `map.at()`, which throws std::out_of_range on a codepoint
// that is not part of the GPT-2 byte encoding. A tokenizer decoding an
// unexpected vocab (a BYTE token, a special token spelled with a codepoint
// outside the map) would then abort through an uncaught exception instead of
// emitting text (review M4). Returns false when there is no mapping.
bool unicode_utf8_to_byte_safe(const std::string & utf8, uint8_t * out);

uint32_t unicode_tolower(uint32_t cpt);
bool unicode_cpt_is_han(uint32_t cpt);

// Pre-tokenize `text` with the qwen35 regex and byte-encode (GPT-2) the words.
std::vector<std::string> unicode_regex_split(const std::string & text);
