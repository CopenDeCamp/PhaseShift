#!/usr/bin/env bash
# Build the PhaseShift-bundled LocalAI runtime from a LocalAI v4.10.0 checkout.
#
# The LocalAI source tree is never vendored into this repository. This helper
# copies the requested checkout into a temporary worktree, applies the
# PhaseShift fail-closed patch series in order, builds the official `local-ai`
# target, and copies the resulting binary to the output path.
#
# The source checkout is never modified.
#
# Requirements: go (>= 1.26), make, git, curl and unzip (used by the LocalAI
# Makefile to fetch protoc), and network access for Go modules / build tools.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PATCH_SERIES=(
    "${REPO_ROOT}/vendor/localai/patches/0001-chat-response-format-fail-closed.patch"
    "${REPO_ROOT}/vendor/localai/patches/0002-responses-structured-transport.patch"
    "${REPO_ROOT}/vendor/localai/patches/0003-tool-policy-transport.patch"
    "${REPO_ROOT}/vendor/localai/patches/0004-structured-tools-composition-transport.patch"
    "${REPO_ROOT}/vendor/localai/patches/0005-responses-reasoning-transport.patch"
    "${REPO_ROOT}/vendor/localai/patches/0006-responses-reasoning-tools-stream.patch"
)
EXPECTED_TAG="v4.10.0"
EXPECTED_COMMIT="7ad0cbf259f0c7bf9920fe2438fc3630ecd6c672"

SOURCE_DIR=""
OUTPUT="${REPO_ROOT}/build/local-ai"
KEEP_TEMP=0

usage() {
    cat <<'EOF'
Usage: tools/build_localai_runtime.sh --source-dir PATH [--output PATH] [--keep-temp]

  --source-dir PATH  LocalAI v4.10.0 source checkout (git repo or source tree)
  --output PATH      Where to write the patched local-ai binary
                     (default: build/local-ai)
  --keep-temp        Do not remove the temporary build directory
  -h, --help         Show this help

The source directory must be LocalAI v4.10.0. A checkout whose HEAD is not the
v4.10.0 tag/commit is rejected; the patch is not forced onto another revision.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --source-dir) SOURCE_DIR="${2:-}"; shift 2 ;;
        --output) OUTPUT="${2:-}"; shift 2 ;;
        --keep-temp) KEEP_TEMP=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "${SOURCE_DIR}" ]; then
    echo "error: --source-dir is required" >&2
    usage >&2
    exit 2
fi
if [ ! -d "${SOURCE_DIR}" ]; then
    echo "error: source directory not found: ${SOURCE_DIR}" >&2
    exit 2
fi
for patch in "${PATCH_SERIES[@]}"; do
    if [ ! -f "${patch}" ]; then
        echo "error: patch not found: ${patch}" >&2
        exit 2
    fi
done

command -v go >/dev/null 2>&1 || {
    echo "error: go not found on PATH (Go >= 1.26 is required; see docs/user/build.md)" >&2
    exit 2
}
command -v make >/dev/null 2>&1 || {
    echo "error: make not found on PATH (see docs/user/build.md)" >&2
    exit 2
}

SOURCE_ABS="$(cd "${SOURCE_DIR}" && pwd)"

if [ ! -f "${SOURCE_ABS}/Makefile" ] || [ ! -d "${SOURCE_ABS}/core/http/endpoints/openai" ]; then
    echo "error: ${SOURCE_ABS} does not look like a LocalAI source tree" >&2
    exit 2
fi

validate_revision() {
    local source="$1"
    if git -C "${source}" rev-parse --git-dir >/dev/null 2>&1; then
        local head
        head="$(git -C "${source}" rev-parse HEAD 2>/dev/null || true)"
        if [ "${head}" = "${EXPECTED_COMMIT}" ]; then
            echo "source revision: ${EXPECTED_TAG} (${EXPECTED_COMMIT})"
            return 0
        fi
        local tag
        tag="$(git -C "${source}" describe --tags --exact-match 2>/dev/null || true)"
        if [ "${tag}" = "${EXPECTED_TAG}" ]; then
            echo "source revision: ${EXPECTED_TAG} (${head})"
            return 0
        fi
        echo "error: source HEAD is not LocalAI ${EXPECTED_TAG} (${EXPECTED_COMMIT}): ${head} (${tag:-no exact tag})" >&2
        return 1
    fi
    echo "error: ${source} is not a git checkout; cannot confirm LocalAI ${EXPECTED_TAG}" >&2
    echo "hint: clone the tag first: git clone --depth 1 --branch ${EXPECTED_TAG} https://github.com/mudler/LocalAI.git" >&2
    return 1
}

validate_revision "${SOURCE_ABS}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/phaseshift-localai-build-XXXXXX")"
cleanup() {
    if [ "${KEEP_TEMP}" -eq 1 ]; then
        echo "temporary build directory kept at ${TMP_DIR}"
        return
    fi
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

BUILD_SRC="${TMP_DIR}/LocalAI"
echo "copying source to ${BUILD_SRC}"
cp -a "${SOURCE_ABS}/." "${BUILD_SRC}/"

for patch in "${PATCH_SERIES[@]}"; do
    if git -C "${BUILD_SRC}" apply --check "${patch}" 2>/dev/null; then
        echo "applying patch: $(basename "${patch}")"
        git -C "${BUILD_SRC}" apply "${patch}"
    elif git -C "${BUILD_SRC}" apply --reverse --check "${patch}" 2>/dev/null; then
        echo "patch already applied: $(basename "${patch}")"
    else
        echo "error: patch does not apply cleanly: $(basename "${patch}")" >&2
        git -C "${BUILD_SRC}" apply --check "${patch}" >&2 || true
        exit 1
    fi
done

mkdir -p "${BUILD_SRC}/core/http/react-ui/dist"
if [ ! -f "${BUILD_SRC}/core/http/react-ui/dist/index.html" ]; then
    printf '<!doctype html><title>LocalAI</title>\n' \
        > "${BUILD_SRC}/core/http/react-ui/dist/index.html"
fi

echo "building local-ai (this downloads Go modules and protoc)"
(
    cd "${BUILD_SRC}"
    export GOTOOLCHAIN="${GOTOOLCHAIN:-local}"
    make build
)

BUILT="${BUILD_SRC}/local-ai"
if [ ! -x "${BUILT}" ]; then
    echo "error: build did not produce ${BUILT}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"
cp "${BUILT}" "${OUTPUT}"
chmod +x "${OUTPUT}"

echo "patched LocalAI runtime: ${OUTPUT}"
"${OUTPUT}" --version 2>/dev/null | head -1 || true
