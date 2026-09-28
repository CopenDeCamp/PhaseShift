cmake_minimum_required(VERSION 3.18)

add_library(phaseshift_xgrammar STATIC
    cpp/compiled_grammar.cc
    cpp/config.cc
    cpp/earley_parser.cc
    cpp/fsm.cc
    cpp/fsm_builder.cc
    cpp/grammar.cc
    cpp/grammar_builder.cc
    cpp/grammar_compiler.cc
    cpp/grammar_functor.cc
    cpp/grammar_matcher.cc
    cpp/grammar_parser.cc
    cpp/grammar_printer.cc
    cpp/json_schema_converter.cc
    cpp/json_schema_converter_ext.cc
    cpp/lark_converter.cc
    cpp/regex_converter.cc
    cpp/structural_tag.cc
    cpp/testing.cc
    cpp/tokenizer_info.cc
    cpp/support/logging.cc
    cpp/support/recursion_guard.cc
)

target_compile_features(phaseshift_xgrammar PRIVATE cxx_std_20)
target_compile_definitions(phaseshift_xgrammar PUBLIC XGRAMMAR_ENABLE_CPPTRACE=0)
target_include_directories(phaseshift_xgrammar PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(phaseshift_xgrammar SYSTEM PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/picojson"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include"
)
set_target_properties(phaseshift_xgrammar PROPERTIES POSITION_INDEPENDENT_CODE ON)
