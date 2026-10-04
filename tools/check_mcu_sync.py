#!/usr/bin/env python3
import re
import sys
from pathlib import Path

BLOCK_COMMENTS = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENTS = re.compile(r"//[^\n]*")
DOUBLE_STRINGS = re.compile(r'"(?:[^"\\]|\\.)*"')
SINGLE_STRINGS = re.compile(r"'(?:[^'\\]|\\.)*'")

RULES = [
    (
        re.compile(r"\bhipDeviceSynchronize\s*\("),
        "hipDeviceSynchronize blocks while the persistent mcu wave is alive",
    ),
    (
        re.compile(r"\bhipMemcpy\s*\([^;]*hipMemcpyDeviceToHost"),
        "sync device->host hipMemcpy blocks while the persistent mcu wave is alive",
    ),
    (
        re.compile(r"\bhipMemcpyWithStream\s*\([^;]*hipMemcpyDeviceToHost"),
        "hipMemcpyWithStream to a pageable dst blocks while the persistent wave is alive",
    ),
]


def blank(match):
    return re.sub(r"[^\n]", " ", match.group(0))


def masked(text):
    text = BLOCK_COMMENTS.sub(blank, text)
    text = LINE_COMMENTS.sub(blank, text)
    text = DOUBLE_STRINGS.sub(blank, text)
    return SINGLE_STRINGS.sub(blank, text)


def statements(text):
    start = 0
    for index, char in enumerate(text):
        if char == ";":
            yield start, text[start:index]
            start = index + 1
    yield start, text[start:]


def scan(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    clean = masked(raw)
    hits = []
    for start, statement in statements(clean):
        for pattern, message in RULES:
            if pattern.search(statement):
                line = raw.count("\n", 0, start) + 1
                prompt = re.sub(r"\s+", " ", statement).strip()
                if len(prompt) > 100:
                    prompt = prompt[:100] + "..."
                hits.append((line, message, prompt))
    return hits


def main(argv):
    files = []
    for arg in argv[1:]:
        root = Path(arg)
        if root.is_dir():
            files.extend(sorted(root.rglob("*.hip")))
            files.extend(sorted(root.rglob("*.cpp")))
            files.extend(sorted(root.rglob("*.h")))
        elif root.is_file():
            files.append(root)
    total = 0
    for path in files:
        for line, message, prompt in scan(path):
            print("MCU-SYNC {}:{}".format(path, line))
            print("  {}".format(message))
            print("  {}".format(prompt))
            total += 1
    if total:
        print("mcu sync check: {} violation(s)".format(total))
        return 1
    print("mcu sync check: clean ({} files)".format(len(files)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
