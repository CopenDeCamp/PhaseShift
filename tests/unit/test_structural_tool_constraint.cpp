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
        } else {
            out += c;
        }
    }
    out += "\"";
    return out;
}

std::string strict_tag(const std::string& name, const std::string& schema,
                       bool any_order) {
    std::string content = "{\"type\":\"qwen_xml_parameter\",\"json_schema\":" + schema;
    if (any_order) {
        content += ",\"any_order\":true";
    }
    content += "}";
    std::string begin = "<tool_call>\n<function=" + name + ">\n";
    std::string end = "\n</function>\n</tool_call>";
    return "{\"type\":\"tag\",\"begin\":" + string_literal(begin) + ",\"content\":" +
           content + ",\"end\":" + string_literal(end) + "}";
}

std::string loose_tag(const std::string& name) {
    std::string begin = "<tool_call>\n<function=" + name + ">\n";
    std::string end = "\n</function>\n</tool_call>";
    return "{\"type\":\"tag\",\"begin\":" + string_literal(begin) +
           ",\"content\":{\"type\":\"any_text\"},\"end\":" + string_literal(end) + "}";
}

std::string wrap(const std::string& format) {
    return "{\"type\":\"structural_tag\",\"format\":" + format + "}";
}

// Gate 11B: one Structural Tag expresses the reasoning phase followed by a
// constrained final phase. The generation prompt pre-fills "<think>\n", so the
// envelope begins with an empty begin and free-form reasoning, closes on
// "</think>\n\n", then applies the existing final format unchanged.
std::string reasoning_envelope(const std::string& final_format) {
    return "{\"type\":\"sequence\",\"elements\":["
           "{\"type\":\"tag\",\"begin\":\"\",\"content\":{\"type\":\"any_text\"},"
           "\"end\":\"</think>\\n\\n\"},"
           + final_format + "]}";
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

std::string triggered_tags(const std::vector<std::string>& tags, bool parallel) {
    (void)tags;
    (void)parallel;
    return std::string();
}

const std::string kCitySchema =
    "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
    "\"required\":[\"city\"],\"additionalProperties\":false}";

const std::string kWeatherSchema =
    "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},"
    "\"unit\":{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"]}},"
    "\"required\":[\"city\",\"unit\"],\"additionalProperties\":false}";

std::string city_call(const std::string& name, const std::string& value) {
    return "<tool_call>\n<function=" + name + ">\n<parameter=city>\n" + value +
           "\n</parameter>\n</function>\n</tool_call>";
}

std::unique_ptr<State> make_gbnf(ps::qwen35::runtime::TokenConstraintCompiler& compiler,
                                 const std::string& gbnf) {
    auto result = compiler.create_gbnf_state(gbnf);
    if (!result.ok()) {
        return nullptr;
    }
    return std::move(result.release());
}

std::unique_ptr<State> make_tag(ps::qwen35::runtime::TokenConstraintCompiler& compiler,
                                const std::string& structural_tag) {
    auto result = compiler.create_structural_tag_state(wrap(structural_tag));
    if (!result.ok()) {
        return nullptr;
    }
    return std::move(result.release());
}

void test_named_strict() {
    TestTokenizer tokenizer = make_tokenizer();
    const std::string serialized = make_tokenizer_info(tokenizer).SerializeJSON();
    auto compiler_result = ps::qwen35::runtime::TokenConstraintCompiler::create(
        serialized, static_cast<uint32_t>(tokenizer.encoded.size()), tokenizer.stop);
    CHECK(compiler_result.ok());
    if (!compiler_result.ok()) {
        return;
    }
    auto& compiler = *compiler_result.value();

    auto state = make_tag(compiler, strict_tag("get_weather", kCitySchema, true));
    CHECK(state != nullptr);
    if (state == nullptr) {
        return;
    }

    CHECK(accept_text(*state, tokenizer, "<tool_call>\n<function="));
    CHECK(next_allows(compiler, *state, tokenizer, 'g'));
    CHECK(!next_allows(compiler, *state, tokenizer, 'd'));
    CHECK(accept_text(*state, tokenizer, "get_weather>\n<parameter="));
    CHECK(next_allows(compiler, *state, tokenizer, 'c'));
    CHECK(!next_allows(compiler, *state, tokenizer, 'z'));
    CHECK(accept_text(*state, tokenizer,
                      "city>\nOsaka\n</parameter>\n</function>\n</tool_call>"));
    CHECK(accept_stop(*state, tokenizer));
    CHECK(state->is_terminated());

    // The named choice cannot reach another function even after a separator.
    auto rejected = make_tag(compiler, strict_tag("get_weather", kCitySchema, true));
    CHECK(rejected != nullptr);
    if (rejected != nullptr) {
        CHECK(!accept_text(*rejected, tokenizer, city_call("delete_everything", "1")));
    }
}

void test_strict_arguments() {
    TestTokenizer tokenizer = make_tokenizer();
    const std::string serialized = make_tokenizer_info(tokenizer).SerializeJSON();
    auto compiler_result = ps::qwen35::runtime::TokenConstraintCompiler::create(
        serialized, static_cast<uint32_t>(tokenizer.encoded.size()), tokenizer.stop);
    if (!compiler_result.ok()) {
        CHECK(false);
        return;
    }
    auto& compiler = *compiler_result.value();

    // any_order: both orderings are accepted.
    auto forward = make_tag(compiler, strict_tag("get_weather", kWeatherSchema, true));
    CHECK(forward != nullptr);
    if (forward != nullptr) {
        CHECK(accept_text(*forward, tokenizer,
                          "<tool_call>\n<function=get_weather>\n<parameter=city>\n"
                          "Osaka\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n"
                          "</function>\n</tool_call>"));
        CHECK(accept_stop(*forward, tokenizer));
        CHECK(forward->is_terminated());
    }

    auto reverse = make_tag(compiler, strict_tag("get_weather", kWeatherSchema, true));
    CHECK(reverse != nullptr);
    if (reverse != nullptr) {
        CHECK(accept_text(*reverse, tokenizer,
                          "<tool_call>\n<function=get_weather>\n<parameter=unit>\n"
                          "fahrenheit\n</parameter>\n<parameter=city>\nOsaka\n</parameter>\n"
                          "</function>\n</tool_call>"));
        CHECK(accept_stop(*reverse, tokenizer));
        CHECK(reverse->is_terminated());
    }

    // Invalid enum value is rejected.
    auto invalid_enum = make_tag(compiler, strict_tag("get_weather", kWeatherSchema, true));
    CHECK(invalid_enum != nullptr);
    if (invalid_enum != nullptr) {
        CHECK(accept_text(*invalid_enum, tokenizer,
                          "<tool_call>\n<function=get_weather>\n<parameter=city>\n"
                          "Osaka\n</parameter>\n<parameter=unit>\n"));
        CHECK(next_allows(compiler, *invalid_enum, tokenizer, 'c'));
        CHECK(!next_allows(compiler, *invalid_enum, tokenizer, 'x'));
    }

    // Extra parameter is rejected with additionalProperties=false.
    auto extra = make_tag(compiler, strict_tag("get_weather", kWeatherSchema, true));
    CHECK(extra != nullptr);
    if (extra != nullptr) {
        CHECK(accept_text(*extra, tokenizer,
                          "<tool_call>\n<function=get_weather>\n<parameter="));
        CHECK(next_allows(compiler, *extra, tokenizer, 'c'));
        CHECK(next_allows(compiler, *extra, tokenizer, 'u'));
        CHECK(!next_allows(compiler, *extra, tokenizer, 'z'));
    }

    // Missing required property: closing the call after city only is rejected.
    auto missing = make_tag(compiler, strict_tag("get_weather", kWeatherSchema, true));
    CHECK(missing != nullptr);
    if (missing != nullptr) {
        CHECK(!accept_text(*missing, tokenizer,
                           "<tool_call>\n<function=get_weather>\n<parameter=city>\n"
                           "Osaka\n</parameter>\n</function>\n</tool_call>"));
    }
}

void test_required_and_parallel() {
    TestTokenizer tokenizer = make_tokenizer();
    const std::string serialized = make_tokenizer_info(tokenizer).SerializeJSON();
    auto compiler_result = ps::qwen35::runtime::TokenConstraintCompiler::create(
        serialized, static_cast<uint32_t>(tokenizer.encoded.size()), tokenizer.stop);
    if (!compiler_result.ok()) {
        CHECK(false);
        return;
    }
    auto& compiler = *compiler_result.value();

    const std::string weather = strict_tag("get_weather", kCitySchema, true);
    const std::string time = strict_tag("get_time", kCitySchema, true);

    // parallel=false: one call then stop.
    auto one = make_tag(compiler, tags_with_separator({weather, time}, false));
    CHECK(one != nullptr);
    if (one != nullptr) {
        CHECK(accept_text(*one, tokenizer, city_call("get_weather", "Osaka")));
        CHECK(!next_allows(compiler, *one, tokenizer, '\n'));
        CHECK(accept_stop(*one, tokenizer));
        CHECK(one->is_terminated());
    }

    // parallel=true: a second call remains reachable.
    auto many = make_tag(compiler, tags_with_separator({weather, time}, true));
    CHECK(many != nullptr);
    if (many != nullptr) {
        CHECK(accept_text(*many, tokenizer, city_call("get_weather", "Osaka")));
        CHECK(next_allows(compiler, *many, tokenizer, '\n'));
        CHECK(accept_text(*many, tokenizer, "\n" + city_call("get_time", "Tokyo")));
        CHECK(accept_stop(*many, tokenizer));
        CHECK(many->is_terminated());
    }

    // at_least_one: zero calls cannot terminate.
    auto zero = make_tag(compiler, tags_with_separator({weather, time}, true));
    CHECK(zero != nullptr);
    if (zero != nullptr) {
        CHECK(!next_allows(compiler, *zero, tokenizer, '\n'));
        CHECK(next_allows(compiler, *zero, tokenizer, '<'));
    }
}

void test_mixed_strict_and_loose() {
    TestTokenizer tokenizer = make_tokenizer();
    const std::string serialized = make_tokenizer_info(tokenizer).SerializeJSON();
    auto compiler_result = ps::qwen35::runtime::TokenConstraintCompiler::create(
        serialized, static_cast<uint32_t>(tokenizer.encoded.size()), tokenizer.stop);
    if (!compiler_result.ok()) {
        CHECK(false);
        return;
    }
    auto& compiler = *compiler_result.value();

    const std::string strict_weather = strict_tag("get_weather", kCitySchema, true);
    const std::string loose_time = loose_tag("get_time");

    auto strict_state = make_tag(compiler, tags_with_separator(
                                              {strict_weather, loose_time}, false));
    CHECK(strict_state != nullptr);
    if (strict_state != nullptr) {
        CHECK(accept_text(*strict_state, tokenizer, city_call("get_weather", "Osaka")));
    }

    auto loose_state = make_tag(compiler, tags_with_separator(
                                             {strict_weather, loose_time}, false));
    CHECK(loose_state != nullptr);
    if (loose_state != nullptr) {
        CHECK(accept_text(*loose_state, tokenizer,
                          "<tool_call>\n<function=get_time>\n<parameter=anything>\n"
                          "whatever text here\n</parameter>\n</function>\n</tool_call>"));
    }
}

void test_reasoning_envelope() {
    TestTokenizer tokenizer = make_tokenizer();
    const std::string serialized = make_tokenizer_info(tokenizer).SerializeJSON();
    auto compiler_result = ps::qwen35::runtime::TokenConstraintCompiler::create(
        serialized, static_cast<uint32_t>(tokenizer.encoded.size()), tokenizer.stop);
    CHECK(compiler_result.ok());
    if (!compiler_result.ok()) {
        return;
    }
    auto& compiler = *compiler_result.value();

    const std::string strict_weather = strict_tag("get_weather", kCitySchema, true);

    // The empty-begin envelope compiles; arbitrary reasoning is accepted, the
    // delimiter transitions, and the strict tool format is enforced after it.
    auto state = make_tag(compiler, reasoning_envelope(strict_weather));
    CHECK(state != nullptr);
    if (state != nullptr) {
        CHECK(accept_text(*state, tokenizer,
                          "The user asks about the weather in Osaka.\n"
                          "I should call get_weather.\n"));
        CHECK(!state->is_terminated());
        CHECK(accept_text(*state, tokenizer, "</think>\n\n"));
        CHECK(accept_text(*state, tokenizer, city_call("get_weather", "Osaka")));
        CHECK(accept_stop(*state, tokenizer));
        CHECK(state->is_terminated());
    }

    // Without the closing delimiter, final-format-looking bytes remain reasoning
    // and never transition into the constraint.
    auto no_delim = make_tag(compiler, reasoning_envelope(strict_weather));
    CHECK(no_delim != nullptr);
    if (no_delim != nullptr) {
        CHECK(accept_text(*no_delim, tokenizer, city_call("get_weather", "Osaka")));
        CHECK(!no_delim->is_terminated());
    }

    // A wrong delimiter does not transition either.
    auto wrong = make_tag(compiler, reasoning_envelope(strict_weather));
    CHECK(wrong != nullptr);
    if (wrong != nullptr) {
        CHECK(accept_text(*wrong, tokenizer, "reasoning\n</think>\nWRONG SUFFIX"));
        CHECK(!wrong->is_terminated());
    }

    // Strict argument validation still applies in the final phase.
    const std::string strict_both = strict_tag("get_weather", kWeatherSchema, true);
    auto missing = make_tag(compiler, reasoning_envelope(strict_both));
    CHECK(missing != nullptr);
    if (missing != nullptr) {
        CHECK(accept_text(*missing, tokenizer, "reasoning\n</think>\n\n"));
        CHECK(!accept_text(*missing, tokenizer, city_call("get_weather", "Osaka")));
    }

    // A grammar final format is constrained only after the envelope.
    const std::string grammar_format =
        "{\"type\":\"grammar\",\"grammar\":\"root ::= \\\"YES\\\"\"}";
    auto ok = make_tag(compiler, reasoning_envelope(grammar_format));
    CHECK(ok != nullptr);
    if (ok != nullptr) {
        CHECK(accept_text(*ok, tokenizer, "reasoning\n</think>\n\n"));
        CHECK(next_allows(compiler, *ok, tokenizer, 'Y'));
        CHECK(!next_allows(compiler, *ok, tokenizer, 'N'));
        CHECK(accept_text(*ok, tokenizer, "YES"));
        CHECK(accept_stop(*ok, tokenizer));
        CHECK(ok->is_terminated());
    }

    // One envelope around an or() of a grammar branch and a pure tool branch:
    // both outcomes are reachable.
    const std::string or_format =
        "{\"type\":\"or\",\"elements\":[" + grammar_format + "," + strict_weather + "]}";
    auto text_branch = make_tag(compiler, reasoning_envelope(or_format));
    CHECK(text_branch != nullptr);
    if (text_branch != nullptr) {
        CHECK(accept_text(*text_branch, tokenizer, "reasoning\n</think>\n\nYES"));
        CHECK(accept_stop(*text_branch, tokenizer));
        CHECK(text_branch->is_terminated());
    }
    auto tool_branch = make_tag(compiler, reasoning_envelope(or_format));
    CHECK(tool_branch != nullptr);
    if (tool_branch != nullptr) {
        CHECK(accept_text(*tool_branch, tokenizer, "reasoning\n</think>\n\n"));
        CHECK(accept_text(*tool_branch, tokenizer, city_call("get_weather", "Osaka")));
        CHECK(accept_stop(*tool_branch, tokenizer));
        CHECK(tool_branch->is_terminated());
    }
}

void test_compile_failures() {
    TestTokenizer tokenizer = make_tokenizer();
    const std::string serialized = make_tokenizer_info(tokenizer).SerializeJSON();
    auto compiler_result = ps::qwen35::runtime::TokenConstraintCompiler::create(
        serialized, static_cast<uint32_t>(tokenizer.encoded.size()), tokenizer.stop);
    if (!compiler_result.ok()) {
        CHECK(false);
        return;
    }
    auto& compiler = *compiler_result.value();

    CHECK(compiler.create_structural_tag_state("not json").ok() == false);
    CHECK(compiler.create_structural_tag_state("{}").ok() == false);
    CHECK(make_tag(compiler, strict_tag("get_weather", kCitySchema, true)) != nullptr);
}

}  // namespace

int main() {
    test_named_strict();
    test_strict_arguments();
    test_required_and_parallel();
    test_mixed_strict_and_loose();
    test_reasoning_envelope();
    test_compile_failures();

    if (failures == 0) {
        std::printf("structural-tool-constraint: passed\n");
    }
    return failures == 0 ? 0 : 1;
}
