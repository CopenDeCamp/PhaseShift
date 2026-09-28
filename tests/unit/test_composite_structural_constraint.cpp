#include <phaseshift/models/qwen35/runtime/token_constraint.h>

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
        ++failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

struct TestTokenizer {
    std::vector<std::string> encoded;
    std::unordered_map<char, int32_t> char_to_id;
    int32_t stop = -1;
};

TestTokenizer make_tokenizer() {
    TestTokenizer tokenizer;
    tokenizer.encoded.push_back("\n");
    tokenizer.char_to_id['\n'] = 0;
    for (int c = 32; c < 127; ++c) {
        int32_t id = static_cast<int32_t>(tokenizer.encoded.size());
        tokenizer.encoded.push_back(std::string(1, static_cast<char>(c)));
        tokenizer.char_to_id[static_cast<char>(c)] = id;
    }
    tokenizer.stop = static_cast<int32_t>(tokenizer.encoded.size());
    tokenizer.encoded.push_back("<stop>");
    return tokenizer;
}

xgrammar::TokenizerInfo make_tokenizer_info(const TestTokenizer& tokenizer) {
    return xgrammar::TokenizerInfo(
        tokenizer.encoded, xgrammar::VocabType::RAW,
        static_cast<int>(tokenizer.encoded.size()),
        std::vector<int32_t>{tokenizer.stop});
}

using State = ps::qwen35::runtime::TokenConstraintState;

bool accept_text(State& state, const TestTokenizer& tokenizer, const std::string& text) {
    for (char c : text) {
        auto it = tokenizer.char_to_id.find(c);
        if (it == tokenizer.char_to_id.end()) {
            return false;
        }
        if (!state.accept_token(it->second)) {
            return false;
        }
    }
    return true;
}

bool accept_stop(State& state, const TestTokenizer& tokenizer) {
    return state.accept_token(tokenizer.stop);
}

bool next_allows(ps::qwen35::runtime::TokenConstraintCompiler& compiler, State& state,
                 const TestTokenizer& tokenizer, char c) {
    std::vector<uint32_t> mask(compiler.mask_words(), 0u);
    if (!state.fill_next_mask(mask.data(), compiler.mask_words()).ok()) {
        return false;
    }
    auto it = tokenizer.char_to_id.find(c);
    if (it == tokenizer.char_to_id.end()) {
        return false;
    }
    return ps::qwen35::runtime::constraint_token_allowed(
        mask.data(), compiler.vocab_size(), it->second);
}

std::string string_literal(const std::string& raw) {
    std::string out = "\"";
    for (char c : raw) {
        if (c == '\n') {
            out += "\\n";
        } else if (c == '"') {
            out += "\\\"";
        } else if (c == '\\') {
            out += "\\\\";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
}

std::string strict_tag(const std::string& name, const std::string& schema) {
    std::string content = "{\"type\":\"qwen_xml_parameter\",\"json_schema\":" + schema +
                          ",\"any_order\":true}";
    std::string begin = "<tool_call>\n<function=" + name + ">\n";
    std::string end = "\n</function>\n</tool_call>";
    return "{\"type\":\"tag\",\"begin\":" + string_literal(begin) + ",\"content\":" +
           content + ",\"end\":" + string_literal(end) + "}";
}

std::string tags_with_separator(const std::vector<std::string>& tags, bool parallel) {
    std::string result = "{\"type\":\"tags_with_separator\",\"tags\":[";
    for (size_t i = 0; i < tags.size(); ++i) {
        if (i != 0) result += ",";
        result += tags[i];
    }
    result += "],\"separator\":\"\\n\",\"at_least_one\":true,\"stop_after_first\":";
    result += parallel ? "false}" : "true}";
    return result;
}

std::string composite(const std::string& grammar, const std::string& tool_format) {
    return "{\"type\":\"structural_tag\",\"format\":{\"type\":\"or\",\"elements\":[{\"type\":"
           "\"grammar\",\"grammar\":" +
           string_literal(grammar) + "}," + tool_format + "]}}";
}

const std::string kCitySchema =
    "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
    "\"required\":[\"city\"],\"additionalProperties\":false}";

const char* kYesGrammar = R"GBNF(root ::= "YES")GBNF";

// Representative LocalAI-generated response grammar (json_schema enum).
const char* kLocalAIGrammar = R"GBNF(freestring ::= (
			[^\x00] |
			"\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
		  )* space
root ::= root-0
root-0 ::= "{" space "\"answer\"" space ":" space root-0-answer "}" space
root-0-answer ::= "\"yes\"" | "\"no\""
space ::= " "?
)GBNF";

std::string city_call(const std::string& name) {
    return "<tool_call>\n<function=" + name + ">\n<parameter=city>\nOsaka\n</parameter>\n"
           "</function>\n</tool_call>";
}

std::unique_ptr<State> make_composite(ps::qwen35::runtime::TokenConstraintCompiler& compiler,
                                      const std::string& grammar,
                                      const std::string& tool_format) {
    auto result = compiler.create_structural_tag_state(composite(grammar, tool_format));
    if (!result.ok()) {
        return nullptr;
    }
    return std::move(result.release());
}

void test_composition(ps::qwen35::runtime::TokenConstraintCompiler& compiler,
                      const TestTokenizer& tokenizer) {
    const std::string weather = strict_tag("get_weather", kCitySchema);
    const std::string tool_only = tags_with_separator({weather}, false);
    const std::string tool_parallel = tags_with_separator({weather}, true);

    // Structured branch: the simple grammar is accepted.
    auto structured = make_composite(compiler, kYesGrammar, tool_only);
    CHECK(structured != nullptr);
    if (structured != nullptr) {
        CHECK(accept_text(*structured, tokenizer, "YES"));
        CHECK(accept_stop(*structured, tokenizer));
        CHECK(structured->is_terminated());
    }

    // Tool branch is accepted with the same constraint.
    auto tool = make_composite(compiler, kYesGrammar, tool_only);
    CHECK(tool != nullptr);
    if (tool != nullptr) {
        CHECK(accept_text(*tool, tokenizer, city_call("get_weather")));
        CHECK(accept_stop(*tool, tokenizer));
        CHECK(tool->is_terminated());
    }

    // Plain unstructured text is rejected (STOP 2).
    auto plain = make_composite(compiler, kYesGrammar, tool_only);
    CHECK(plain != nullptr);
    if (plain != nullptr) {
        CHECK(!accept_text(*plain, tokenizer, "HELLO"));
    }

    // auto parallel=false: one call then no second call.
    auto single = make_composite(compiler, kYesGrammar, tool_only);
    CHECK(single != nullptr);
    if (single != nullptr) {
        CHECK(accept_text(*single, tokenizer, city_call("get_weather")));
        CHECK(!next_allows(compiler, *single, tokenizer, '\n'));
    }

    // auto parallel=true: a second call remains reachable.
    auto many = make_composite(compiler, kYesGrammar, tool_parallel);
    CHECK(many != nullptr);
    if (many != nullptr) {
        CHECK(accept_text(*many, tokenizer, city_call("get_weather")));
        CHECK(next_allows(compiler, *many, tokenizer, '\n'));
        CHECK(accept_text(*many, tokenizer, "\n" + city_call("get_weather")));
        CHECK(many->is_terminated() == false);
    }

    // Strict argument rejection is preserved inside the composite.
    auto strict = make_composite(compiler, kYesGrammar, tool_only);
    CHECK(strict != nullptr);
    if (strict != nullptr) {
        CHECK(accept_text(*strict, tokenizer,
                          "<tool_call>\n<function=get_weather>\n<parameter="));
        CHECK(next_allows(compiler, *strict, tokenizer, 'c'));
        CHECK(!next_allows(compiler, *strict, tokenizer, 'z'));
    }

    // Real LocalAI grammar branch and tool branch coexist.
    auto real = make_composite(compiler, kLocalAIGrammar, tool_only);
    CHECK(real != nullptr);
    if (real != nullptr) {
        CHECK(accept_text(*real, tokenizer, "{\"answer\":\"yes\"}"));
        CHECK(accept_stop(*real, tokenizer));
        CHECK(real->is_terminated());
    }
    auto real_tool = make_composite(compiler, kLocalAIGrammar, tool_only);
    CHECK(real_tool != nullptr);
    if (real_tool != nullptr) {
        CHECK(accept_text(*real_tool, tokenizer, city_call("get_weather")));
        CHECK(accept_stop(*real_tool, tokenizer));
    }

    // The real grammar rejects an unrelated plain word.
    auto real_plain = make_composite(compiler, kLocalAIGrammar, tool_only);
    CHECK(real_plain != nullptr);
    if (real_plain != nullptr) {
        CHECK(!accept_text(*real_plain, tokenizer, "hello"));
    }
}

void test_compile_failures(ps::qwen35::runtime::TokenConstraintCompiler& compiler,
                           const TestTokenizer& tokenizer) {
    (void)tokenizer;
    CHECK(compiler.create_structural_tag_state(
              "{\"type\":\"structural_tag\",\"format\":{\"type\":\"or\",\"elements\":["
              "{\"type\":\"grammar\",\"grammar\":\"root ::= [\"}]}}")
              .ok() == false);
    CHECK(make_composite(compiler, kYesGrammar,
                         tags_with_separator({strict_tag("get_weather", kCitySchema)}, false)) !=
          nullptr);
}

}  // namespace

int main() {
    TestTokenizer tokenizer = make_tokenizer();
    const std::string serialized = make_tokenizer_info(tokenizer).SerializeJSON();
    auto compiler_result = ps::qwen35::runtime::TokenConstraintCompiler::create(
        serialized, static_cast<uint32_t>(tokenizer.encoded.size()), tokenizer.stop);
    CHECK(compiler_result.ok());
    if (compiler_result.ok()) {
        auto& compiler = *compiler_result.value();
        test_composition(compiler, tokenizer);
        test_compile_failures(compiler, tokenizer);
    }

    if (failures == 0) {
        std::printf("composite-structural-constraint: passed\n");
    }
    return failures == 0 ? 0 : 1;
}
