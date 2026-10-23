#include "../src/unicode.h"

#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

int main() {
    bool success = true;
    const std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> normalization = {
        {{}, {}},
        {{0x65, 0x301}, {0xE9}},
        {{0xE9}, {0xE9}},
        {{0x212B}, {0xC5}},
        {{0x61, 0x315, 0x300}, {0xE0, 0x315}},
        {{0x61, 0x305, 0x301}, {0x61, 0x305, 0x301}},
        {{0x301, 0x327}, {0x327, 0x301}},
        {{0x1100, 0x1161, 0x11A8}, {0xAC01}},
        {{0xAC01}, {0xAC01}},
        {{0xFB01, 0xFF21}, {0xFB01, 0xFF21}},
        {{0x958}, {0x915, 0x93C}},
        {{0x1D15E}, {0x1D157, 0x1D165}},
    };
    for (const auto & test : normalization) {
        if (unicode_cpts_normalize_nfc(test.first) != test.second || unicode_cpts_normalize_nfc(test.second) != test.second) {
            fprintf(stderr, "unexpected NFC normalization\n");
            success = false;
            break;
        }
    }

    const struct {
        std::string text;
        std::vector<uint32_t> expected;
    } utf8_tests[] = {
        { "\x7f",                 { 0x7f } },
        { "\xc2\x80",             { 0x80 } },
        { "\xdf\xbf",             { 0x7ff } },
        { "\xe0\xa0\x80",         { 0x800 } },
        { "\xef\xbf\xbf",         { 0xffff } },
        { "\xf0\x90\x80\x80",     { 0x10000 } },
        { "\xf4\x8f\xbf\xbf",     { 0x10ffff } },
        // Invalid-input cases preserve the existing recovery behavior: one U+FFFD per byte.
        { "\xc0\xa0",             { 0xfffd, 0xfffd } },
        { "\xc1\xbf",             { 0xfffd, 0xfffd } },
        { "\xe0\x80\xa0",         { 0xfffd, 0xfffd, 0xfffd } },
        { "\xe0\x9f\xbf",         { 0xfffd, 0xfffd, 0xfffd } },
        { "\xf0\x80\x80\xa0",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf0\x8f\xbf\xbf",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf4\x90\x80\x80",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf5\x80\x80\x80",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xf7\xbf\xbf\xbf",     { 0xfffd, 0xfffd, 0xfffd, 0xfffd } },
        { "\xe2\x80",             { 0xfffd, 0xfffd } },
    };

    for (const auto & test : utf8_tests) {
        auto expected = test.expected;
        expected.insert(expected.begin(), 'a');
        expected.push_back('b');
        if (unicode_cpts_from_utf8("a" + test.text + "b") != expected) {
            fprintf(stderr, "unexpected UTF-8 decoding for bytes:");
            for (const unsigned char byte : test.text) {
                fprintf(stderr, " %02x", byte);
            }
            fprintf(stderr, "\n");
            success = false;
        }
    }

    {
        const std::vector<std::string> regex_exprs = {
            "[~][A-Za-z]+| ?[\\p{S}]+|\\s+",
        };
        const std::vector<std::string> expected = { " ~", "foo" };
        const auto actual = unicode_regex_split(" ~foo", regex_exprs, false);

        if (actual != expected) {
            fprintf(stderr, "unexpected split:");
            for (const auto & piece : actual) {
                fprintf(stderr, " [%s]", piece.c_str());
            }
            fprintf(stderr, "\n");
            success = false;
        }
    }

    {
        const std::vector<std::string> regex_exprs = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
            "[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };

        const std::string input = "ab\u200ccd ef\u200dgh cafe\u0301";
        const std::vector<std::string> expected = {
            "ab\u200ccd",
            " ef\u200dgh",
            " cafe\u0301",
        };

        try {
            const auto actual = unicode_regex_split(input, regex_exprs, false);

            if (actual != expected) {
                fprintf(stderr, "unexpected K2-Horizon split:");
                for (const auto & piece : actual) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }
                fprintf(stderr, "\n");
                return 1;
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "K2-Horizon regex split threw exception: %s\n", e.what());
            return 1;
        }
    }


    {
        const std::vector<std::string> llama3_regex = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
            "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };

        const std::vector<std::string> k2_horizon_regex = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
            "[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };

        const std::vector<std::string> inputs = {
            "Hello, world!",
            "can't won't we're they'll",
            "123 1234 1234567",
            "Hello!!!\\nNext line",
            "alpha beta gamma",
            " café résumé",
        };

        for (const auto & input : inputs) {
            const auto llama3 = unicode_regex_split(input, llama3_regex, false);
            const auto k2 = unicode_regex_split(input, k2_horizon_regex, false);

            if (llama3 != k2) {
                fprintf(stderr, "K2-Horizon diverged from Llama 3 for ordinary input: %s\n", input.c_str());

                fprintf(stderr, "Llama 3:");
                for (const auto & piece : llama3) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }

                fprintf(stderr, "\nK2-Horizon:");
                for (const auto & piece : k2) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }

                fprintf(stderr, "\n");
                return 1;
            }
        }
    }


    {
        const std::vector<std::string> k2_horizon_regex = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
            "[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };

        auto check_k2 = [&] (
                const char * name,
                const std::string & input,
                const std::vector<std::string> & expected) {
            const auto actual = unicode_regex_split(input, k2_horizon_regex, false);

            if (actual != expected) {
                fprintf(stderr, "K2-Horizon reference mismatch for %s\n", name);

                fprintf(stderr, "expected:");
                for (const auto & piece : expected) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }

                fprintf(stderr, "\nactual:");
                for (const auto & piece : actual) {
                    fprintf(stderr, " [%s]", piece.c_str());
                }

                fprintf(stderr, "\n");
                return false;
            }

            return true;
        };

        if (!check_k2(
                "ZWNJ",
                "ab\u200ccd",
                { "ab\u200ccd" })) {
            return 1;
        }

        if (!check_k2(
                "ZWJ",
                "ef\u200dgh",
                { "ef\u200dgh" })) {
            return 1;
        }

        if (!check_k2(
                "combined ZWNJ/ZWJ",
                "ab\u200ccd ef\u200dgh",
                { "ab\u200ccd", " ef\u200dgh" })) {
            return 1;
        }

        // The reference tokenizer NFC-normalizes cafe + combining acute
        // to the composed form before pre-tokenization.
        if (!check_k2(
                "NFC accent",
                "caf\u00e9",
                { "caf\u00e9" })) {
            return 1;
        }

        if (!check_k2(
                "Persian ZWNJ",
                "\u0645\u06cc\u200c\u0631\u0648\u0645",
                { "\u0645\u06cc\u200c\u0631\u0648\u0645" })) {
            return 1;
        }

        if (!check_k2(
                "Devanagari ZWJ",
                "\u0915\u094d\u200d\u0937",
                { "\u0915\u094d\u200d\u0937" })) {
            return 1;
        }

        if (!check_k2(
                "contractions",
                "can't won't we're they'll",
                { "can", "'t", " won", "'t", " we", "'re", " they", "'ll" })) {
            return 1;
        }

        if (!check_k2(
                "Unicode contraction",
                "'\u017fa",
                { "'\u017f", "a" })) {
            return 1;
        }

        if (!check_k2(
                "empty input",
                "",
                { })) {
            return 1;
        }

        if (!check_k2(
                "numbers",
                "123 1234 1234567",
                { "123", " ", "123", "4", " ", "123", "456", "7" })) {
            return 1;
        }

        if (!check_k2(
                "punctuation/newline",
                "Hello!!!\nNext line",
                { "Hello", "!!!\n", "Next", " line" })) {
            return 1;
        }
    }

    return success ? 0 : 1;
}
