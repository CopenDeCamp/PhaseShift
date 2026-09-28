#include <xgrammar/xgrammar.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    // Default named_grammars argument: must compile and run under C++20/libc++.
    try {
        auto grammar = xgrammar::Grammar::FromLark("start: \"YES\"");
        check(grammar.SerializeJSON().size() > 0, "default FromLark produces a grammar");
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "default FromLark threw: %s\n", exc.what());
        ++failures;
    }

    // Explicit named grammars regression: the overload still accepts them.
    try {
        std::vector<xgrammar::NamedGrammar> named;
        named.push_back(xgrammar::NamedGrammar{"answer", std::string("start: \"NO\"")});
        auto grammar =
            xgrammar::Grammar::FromLark("start: @answer", std::nullopt, named);
        check(grammar.SerializeJSON().size() > 0, "explicit named FromLark produces a grammar");
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "explicit named FromLark threw: %s\n", exc.what());
        ++failures;
    }

    // Empty explicit vector must remain valid.
    try {
        std::vector<xgrammar::NamedGrammar> empty;
        auto grammar = xgrammar::Grammar::FromLark("start: \"YES\"", std::nullopt, empty);
        check(grammar.SerializeJSON().size() > 0, "empty explicit named FromLark");
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "empty explicit FromLark threw: %s\n", exc.what());
        ++failures;
    }

    if (failures == 0) {
        std::printf("xgrammar-cxx20: passed\n");
    }
    return failures == 0 ? 0 : 1;
}
