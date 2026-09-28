#!/usr/bin/env python3
"""Gate 11A: reasoning transport, policy, parsing and history codec units."""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import Checker, COMMON_DIR, REPO_ROOT, load_processor  # noqa: E402

for _extra in (str(REPO_ROOT / "src" / "apps" / "server"),
               str(REPO_ROOT / "src" / "apps" / "server" / "backend")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from phaseshift_chat import codec  # noqa: E402
import message_codec  # noqa: E402


def policy_cases(checker):
    cases = [
        ("empty", {}, (False, "")),
        ("none", {"reasoning_effort": "none"}, (False, "none")),
        ("minimal", {"reasoning_effort": "minimal"}, (True, "minimal")),
        ("low", {"reasoning_effort": "low"}, (True, "low")),
        ("medium", {"reasoning_effort": "medium"}, (True, "medium")),
        ("high", {"reasoning_effort": "high"}, (True, "high")),
        ("xhigh", {"reasoning_effort": "xhigh"}, (True, "xhigh")),
        ("max", {"reasoning_effort": "max"}, (True, "max")),
        ("enable-true", {"enable_thinking": "true"}, (True, "")),
        ("enable-false", {"enable_thinking": "false"}, (False, "")),
        ("enable-wins",
         {"enable_thinking": "false", "reasoning_effort": "high"}, (False, "high")),
        ("enable-wins-max",
         {"enable_thinking": "false", "reasoning_effort": "max"}, (False, "max")),
        ("blob-fallback",
         {"chat_template_kwargs": '{"enable_thinking": true, "reasoning_effort": "low"}'},
         (True, "low")),
        ("blob-fallback-xhigh",
         {"chat_template_kwargs": '{"reasoning_effort": "xhigh"}'},
         (True, "xhigh")),
    ]
    for name, metadata, expected in cases:
        got = codec.resolve_reasoning_policy(metadata)
        checker.check(f"policy-{name}", got == expected, f"{got} != {expected}")
    try:
        codec.resolve_reasoning_policy({"reasoning_effort": "banana"})
        checker.check("policy-invalid-rejected", False, "banana accepted")
    except codec.ReasoningEffortError:
        checker.check("policy-invalid-rejected", True, "")
    try:
        codec.resolve_reasoning_policy({"enable_thinking": "maybe"})
        checker.check("policy-bad-bool-rejected", False, "maybe accepted")
    except codec.ReasoningEffortError:
        checker.check("policy-bad-bool-rejected", True, "")


def split_cases(checker):
    cases = [
        ("thinking-answer", "reason\n</think>\n\nAnswer", ("reason", "Answer")),
        ("empty-thinking", "\n\n</think>\n\nAnswer", ("", "Answer")),
        ("thinking-only", "reason without close", ("reason without close", "")),
        ("with-open", "<think>\nreason\n</think>\n\nAnswer", ("reason", "Answer")),
        ("unicode", "考える\n</think>\n\n答え", ("考える", "答え")),
        ("markup-in-answer", "r\n</think>\n\nuse <think> inline", ("r", "use <think> inline")),
    ]
    for name, text, expected in cases:
        got = codec.split_reasoning_final(text)
        checker.check(f"split-{name}", got == expected, f"{got} != {expected}")

    full = "reason</think>\n\nAnswer"
    seen_r, seen_c = "", ""
    for index in range(1, len(full) + 1):
        reasoning, content = codec.split_reasoning_partial(full[:index])
        checker.check(f"partial-monotonic-{index}",
                      reasoning.startswith(seen_r) and content.startswith(seen_c),
                      f"regressed at {index}: {reasoning!r} {content!r}")
        seen_r, seen_c = reasoning, content
    checker.check("partial-final-consistent",
                  codec.split_reasoning_final(full)[1] == content,
                  f"{codec.split_reasoning_final(full)} vs {content!r}")


def history_cases(checker):
    messages = [
        types.SimpleNamespace(role="user", content="q", tool_calls="", name="",
                              tool_call_id="", reasoning_content=""),
        types.SimpleNamespace(role="assistant", content="a", tool_calls="",
                              name="", tool_call_id="",
                              reasoning_content="thinking text"),
        types.SimpleNamespace(role="user", content="next", tool_calls="", name="",
                              tool_call_id="", reasoning_content=""),
    ]
    hf = message_codec.proto_messages_to_hf(messages)
    checker.check("history-reasoning-kept",
                  hf[1].get("reasoning_content") == "thinking text", repr(hf))
    checker.check("history-content-kept", hf[1].get("content") == "a", repr(hf))
    checker.check("history-no-merge",
                  "thinking text" not in hf[1].get("content", ""), repr(hf))

    both = [
        types.SimpleNamespace(
            role="assistant", content="", tool_calls=(
                '[{"type":"function","function":{"name":"f","arguments":"{}"}}]'),
            name="", tool_call_id="", reasoning_content="why"),
    ]
    hf2 = message_codec.proto_messages_to_hf(both)
    checker.check("history-reasoning-and-tools",
                  hf2[0].get("reasoning_content") == "why"
                  and bool(hf2[0].get("tool_calls")), repr(hf2))


def template_cases(checker, processor):
    checker.check("thinking-available", codec.thinking_available(processor), "")

    messages = [{"role": "user", "content": "hello"}]
    off = codec.prompt_ids_with_cache_boundary(
        processor, messages, None, False)
    on = codec.prompt_ids_with_cache_boundary(
        processor, messages, None, True)
    checker.check("thinking-render-differs", off.ids != on.ids, "")
    off_text = processor.decode(off.ids, skip_special_tokens=False)
    on_text = processor.decode(on.ids, skip_special_tokens=False)
    checker.check("thinking-off-closed",
                  "</think>" in off_text and off_text.rstrip().endswith("</think>"),
                  repr(off_text[-40:]))
    checker.check("thinking-on-open",
                  on_text.rstrip().endswith("<think>"), repr(on_text[-40:]))
    checker.check("boundary-off-exact", off.cache_boundary > 0, str(off))
    checker.check("boundary-on-exact", on.cache_boundary > 0, str(on))
    # The stable history (add_generation_prompt=False) is identical for both
    # modes; only the generation prompt tail differs.
    checker.check("boundary-stable-same", off.cache_boundary == on.cache_boundary,
                  f"{off.cache_boundary} vs {on.cache_boundary}")
    checker.check("full-prompt-differs", len(off.ids) != len(on.ids),
                  f"{len(off.ids)} vs {len(on.ids)}")


def main() -> int:
    checker = Checker("reasoning-transport")
    try:
        policy_cases(checker)
        split_cases(checker)
        history_cases(checker)
        template_cases(checker, load_processor())
    except Exception as exc:  # noqa: BLE001
        checker.check("exception", False, repr(exc))
    return checker.done()


if __name__ == "__main__":
    sys.exit(main())
