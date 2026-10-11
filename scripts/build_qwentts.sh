#!/usr/bin/env bash
# Build qwentts.cpp (tts-server, qwen-tts and its C ABI test) at the pinned
# commit with LingkodAI's patches from patches/qwentts/ applied.
#
# Usage:
#   scripts/build_qwentts.sh                        CUDA build for the Colab GPUs
#                                                   (sm_75 T4, sm_80 A100, sm_89 L4)
#   scripts/build_qwentts.sh --cuda-arch native     CUDA build for this machine's GPU only (faster)
#   scripts/build_qwentts.sh --cpu                  CPU-only compile check (laptop), no model loads
#   ... --cache-dir DIR                             reuse a cached build from DIR, or cache this one there
#
# --cache-dir keys a cached build on everything that shapes the binaries: the
# pinned commits, the patch files, the mode and CUDA arch (for native, this
# GPU's compute capability), the compiler versions, the extra cmake arguments
# and the build path (the binaries carry an absolute RUNPATH). A key match
# restores the binaries and shared libraries and runs the same checks as a
# fresh build; any other change builds from source and caches the result.
#
# Environment:
#   QWENTTS_DIR   clone and build location (default: build/qwentts.cpp under the
#                 repo root, which .gitignore already covers)
#   CMAKE         cmake binary (default: cmake on PATH)
#   NVCC          CUDA compiler for the CUDA build (default: nvcc on PATH, else
#                 /usr/local/cuda/bin/nvcc)
#   QWENTTS_CMAKE_ARGS
#                 extra cmake arguments, space separated, e.g. -DGGML_NATIVE=OFF
#                 for a binary that must run on other CPUs (docker/Dockerfile.gpu)
#
# The CUDA build goes to $QWENTTS_DIR/build, the path qwentts.cpp's own
# tests/debug-clone-cossim.py expects. The CPU check goes to
# $QWENTTS_DIR/build-cpu. Both write BUILD_INFO.txt next to the binaries.

set -euo pipefail

QWENTTS_URL="https://github.com/ServeurpersoCom/qwentts.cpp.git"
QWENTTS_COMMIT="0cbde9b5d21aa2142efa5d62e14b97f197ac6f6d"  # Track A's commit
GGML_COMMIT="0af0d7d5f66a6976b259b292cb4e7dc60457aa45"     # submodule pinned by QWENTTS_COMMIT
PATCHES=(
    "0001-icl-non-streaming-layout.patch"
    "0002-talker-kv-length.patch"
)
DEFAULT_CUDA_ARCH="75;80;89"
TARGETS=(tts-server qwen-tts test-abi-c)

die() {
    echo "[build_qwentts] FATAL: $*" >&2
    exit 1
}

MODE="cuda"
CUDA_ARCH="$DEFAULT_CUDA_ARCH"
CACHE_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) MODE="cpu"; shift ;;
        --cuda-arch)
            [[ $# -ge 2 ]] || die "--cuda-arch needs a value, e.g. '75;80;89' or native"
            CUDA_ARCH="$2"; shift 2 ;;
        --cache-dir)
            [[ $# -ge 2 ]] || die "--cache-dir needs a directory"
            CACHE_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$REPO_ROOT/patches/qwentts"
QWENTTS_DIR="${QWENTTS_DIR:-$REPO_ROOT/build/qwentts.cpp}"
CMAKE="${CMAKE:-cmake}"
if [[ "$MODE" == "cuda" ]]; then
    BUILD_DIR="$QWENTTS_DIR/build"
else
    BUILD_DIR="$QWENTTS_DIR/build-cpu"
fi

first_line() { "$@" 2>&1 | head -n 1; }

# ---------------------------------------------------------------------------
# Paths and versions first, before any work.
# ---------------------------------------------------------------------------
command -v git >/dev/null || die "git not found"
command -v "$CMAKE" >/dev/null || die "cmake not found (set CMAKE=/path/to/cmake)"
command -v c++ >/dev/null || die "c++ compiler not found"

echo "== build_qwentts.sh =="
echo "date (UTC):     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo root:      $REPO_ROOT"
echo "patch dir:      $PATCH_DIR"
echo "qwentts dir:    $QWENTTS_DIR"
echo "build dir:      $BUILD_DIR"
echo "mode:           $MODE"
echo "qwentts commit: $QWENTTS_COMMIT"
echo "ggml commit:    $GGML_COMMIT"
echo "bash:           ${BASH_VERSION}"
echo "git:            $(first_line git --version)"
echo "cmake:          $(first_line "$CMAKE" --version)"
echo "c++:            $(first_line c++ --version)"
echo "cpus:           $(getconf _NPROCESSORS_ONLN)"
echo "extra cmake:    ${QWENTTS_CMAKE_ARGS:-(none)}"
echo "cache dir:      ${CACHE_DIR:-(none)}"

NVCC_PATH=""
GPU_LINE=""
if [[ "$MODE" == "cuda" ]]; then
    NVCC_PATH="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
    [[ -x "$NVCC_PATH" ]] || die "nvcc not found at '$NVCC_PATH' (set NVCC, or use --cpu for a compile check)"
    echo "nvcc:           $NVCC_PATH, $("$NVCC_PATH" --version | grep -i release)"
    echo "cuda arch:      $CUDA_ARCH"
    if command -v nvidia-smi >/dev/null; then
        GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | head -n 1)"
        echo "gpu:            $GPU_LINE"
        # A binary built without this GPU's architecture fails at the first
        # kernel launch, long after the build. Refuse it here instead.
        if [[ "$CUDA_ARCH" != "native" ]]; then
            GPU_SM="$(echo "$GPU_LINE" | awk -F', ' '{print $2}' | tr -d '.')"
            [[ ";$CUDA_ARCH;" == *";$GPU_SM;"* ]] \
                || die "this GPU is sm_$GPU_SM, not in --cuda-arch '$CUDA_ARCH'. Add it or pass --cuda-arch native."
        fi
    else
        echo "gpu:            none visible (nvidia-smi not found), building anyway"
        [[ "$CUDA_ARCH" != "native" ]] || die "--cuda-arch native needs a visible GPU"
    fi
fi
echo

# ---------------------------------------------------------------------------
# Patch set: exactly the files listed above, nothing more.
# ---------------------------------------------------------------------------
[[ -d "$PATCH_DIR" ]] || die "patch dir missing: $PATCH_DIR"
mapfile -t ON_DISK < <(find "$PATCH_DIR" -maxdepth 1 -type f -name '*.patch' -printf '%f\n' | sort)
EXPECTED="$(printf '%s\n' "${PATCHES[@]}" | sort)"
[[ "$(printf '%s\n' "${ON_DISK[@]}")" == "$EXPECTED" ]] \
    || die "patch files on disk (${ON_DISK[*]}) do not match the list in this script (${PATCHES[*]})"
PATCH_SUMS=()
for p in "${PATCHES[@]}"; do
    PATCH_SUMS+=("$(sha256sum "$PATCH_DIR/$p" | awk '{print $1}')  $p")
done

# Checks that need no model: the C ABI smoke test, both patched flags present
# in the binaries (an unpatched build would lack them), and every shared
# library resolvable.
check_binaries() {
    "$BUILD_DIR/test-abi-c" || die "test-abi-c failed"
    local bin usage flag
    for bin in tts-server qwen-tts; do
        usage="$("$BUILD_DIR/$bin" 2>&1 || true)"
        for flag in --icl-non-streaming --talker-ctx; do
            grep -q -- "$flag" <<<"$usage" || die "$bin usage lacks $flag: the patches did not reach this binary"
        done
        if ldd "$BUILD_DIR/$bin" | grep -q "not found"; then
            ldd "$BUILD_DIR/$bin" | grep "not found" >&2
            die "$bin has unresolved shared libraries (above)"
        fi
    done
}

# ---------------------------------------------------------------------------
# Build cache: restore a matching build instead of compiling.
# ---------------------------------------------------------------------------
CACHE_FILE=""
if [[ -n "$CACHE_DIR" ]]; then
    KEY_TEXT="$(
        echo "qwentts $QWENTTS_COMMIT"
        echo "ggml $GGML_COMMIT"
        printf 'patch %s\n' "${PATCH_SUMS[@]}"
        echo "mode $MODE"
        echo "build dir $BUILD_DIR"
        echo "extra cmake ${QWENTTS_CMAKE_ARGS:-}"
        echo "cmake $(first_line "$CMAKE" --version)"
        echo "c++ $(first_line c++ --version)"
        if [[ "$MODE" == "cuda" ]]; then
            echo "cuda arch $CUDA_ARCH"
            echo "nvcc $("$NVCC_PATH" --version | grep -i release)"
            # native builds for this GPU only, so the GPU is part of the key.
            [[ "$CUDA_ARCH" != "native" ]] || echo "gpu compute cap $(echo "$GPU_LINE" | awk -F', ' '{print $2}')"
        fi
    )"
    KEY="$(sha256sum <<<"$KEY_TEXT" | cut -c1-16)"
    CACHE_FILE="$CACHE_DIR/qwentts-$MODE-$KEY.tar.gz"
    echo "[build_qwentts] cache key $KEY:"
    while IFS= read -r line; do echo "    $line"; done <<<"$KEY_TEXT"
    if [[ -f "$CACHE_FILE" ]]; then
        echo "[build_qwentts] restoring $CACHE_FILE into $BUILD_DIR"
        mkdir -p "$BUILD_DIR"
        tar -xzf "$CACHE_FILE" -C "$BUILD_DIR"
        check_binaries
        echo "== BUILD_INFO ($BUILD_DIR/BUILD_INFO.txt, from the cached build) =="
        cat "$BUILD_DIR/BUILD_INFO.txt"
        echo "[build_qwentts] OK (restored from cache, not rebuilt)"
        exit 0
    fi
    echo "[build_qwentts] no cached build for this key; building from source"
fi
echo

# ---------------------------------------------------------------------------
# Source: clone, or reuse an existing clone of the same upstream.
# ---------------------------------------------------------------------------
if [[ -e "$QWENTTS_DIR" ]]; then
    [[ -d "$QWENTTS_DIR/.git" ]] || die "$QWENTTS_DIR exists but is not a git clone; move it away first"
    ORIGIN="$(git -C "$QWENTTS_DIR" remote get-url origin)"
    [[ "$ORIGIN" == "$QWENTTS_URL" ]] || die "$QWENTTS_DIR is a clone of '$ORIGIN', expected '$QWENTTS_URL'"
    echo "[build_qwentts] reusing clone at $QWENTTS_DIR, resetting tracked files to the pinned commit"
    git -C "$QWENTTS_DIR" cat-file -e "${QWENTTS_COMMIT}^{commit}" 2>/dev/null \
        || git -C "$QWENTTS_DIR" fetch origin
else
    echo "[build_qwentts] cloning $QWENTTS_URL into $QWENTTS_DIR"
    mkdir -p "$(dirname "$QWENTTS_DIR")"
    git clone "$QWENTTS_URL" "$QWENTTS_DIR"
fi
# Resetting undoes any previously applied patches, so they always apply to
# the pristine commit. Only tracked files change; build dirs are untracked.
git -C "$QWENTTS_DIR" checkout --quiet --force --detach "$QWENTTS_COMMIT"
git -C "$QWENTTS_DIR" reset --quiet --hard "$QWENTTS_COMMIT"
git -C "$QWENTTS_DIR" submodule update --init --recursive
GOT_GGML="$(git -C "$QWENTTS_DIR/ggml" rev-parse HEAD)"
[[ "$GOT_GGML" == "$GGML_COMMIT" ]] || die "ggml submodule is at $GOT_GGML, expected $GGML_COMMIT"

for p in "${PATCHES[@]}"; do
    git -C "$QWENTTS_DIR" apply --check "$PATCH_DIR/$p" || die "patch does not apply cleanly: $p"
    git -C "$QWENTTS_DIR" apply "$PATCH_DIR/$p"
    echo "[build_qwentts] applied $p (sha256 $(sha256sum "$PATCH_DIR/$p" | awk '{print $1}'))"
done
echo

# ---------------------------------------------------------------------------
# Configure and build.
# ---------------------------------------------------------------------------
CMAKE_ARGS=(-S "$QWENTTS_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release)
if [[ "$MODE" == "cuda" ]]; then
    CMAKE_ARGS+=(-DGGML_CUDA=ON "-DCMAKE_CUDA_COMPILER=$NVCC_PATH" "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH")
else
    CMAKE_ARGS+=(-DGGML_CUDA=OFF)
fi
if [[ -n "${QWENTTS_CMAKE_ARGS:-}" ]]; then
    read -r -a EXTRA_CMAKE <<<"$QWENTTS_CMAKE_ARGS"
    CMAKE_ARGS+=("${EXTRA_CMAKE[@]}")
fi
echo "[build_qwentts] $CMAKE ${CMAKE_ARGS[*]}"
"$CMAKE" "${CMAKE_ARGS[@]}"
"$CMAKE" --build "$BUILD_DIR" --config Release -j "$(getconf _NPROCESSORS_ONLN)" --target "${TARGETS[@]}"
echo

check_binaries
VERSION_LINE="$("$BUILD_DIR/tts-server" 2>&1 | head -n 1 || true)"

INFO="$BUILD_DIR/BUILD_INFO.txt"
{
    echo "built (UTC):    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "mode:           $MODE"
    if [[ "$MODE" == "cuda" ]]; then
        echo "cuda arch:      $CUDA_ARCH"
    fi
    echo "binary version: $VERSION_LINE (upstream hash; the patches below are applied on top)"
    echo "extra cmake:    ${QWENTTS_CMAKE_ARGS:-(none)}"
    echo "qwentts commit: $QWENTTS_COMMIT"
    echo "ggml commit:    $GOT_GGML"
    echo "patches (sha256):"
    printf '  %s\n' "${PATCH_SUMS[@]}"
    echo "cmake:          $(first_line "$CMAKE" --version)"
    echo "c++:            $(first_line c++ --version)"
    if [[ -n "$NVCC_PATH" ]]; then
        echo "nvcc:           $("$NVCC_PATH" --version | grep -i release)"
    fi
    echo "tts-server:     $BUILD_DIR/tts-server"
    echo "qwen-tts:       $BUILD_DIR/qwen-tts"
} >"$INFO"
echo "== BUILD_INFO ($INFO) =="
cat "$INFO"

if [[ -n "$CACHE_FILE" ]]; then
    # Only what runs: the three binaries, every shared library, BUILD_INFO.
    # Written beside the target and renamed, so a cut-off copy never matches.
    mkdir -p "$CACHE_DIR"
    mapfile -t CACHED < <(cd "$BUILD_DIR" && {
        printf '%s\n' "${TARGETS[@]}" BUILD_INFO.txt
        # Versioned names (libggml.so.0) are symlinks; tar keeps them as links.
        find . \( -type f -o -type l \) \( -name '*.so' -o -name '*.so.*' \) -printf '%P\n'
    } | sort)
    tar -czf "$CACHE_FILE.part" -C "$BUILD_DIR" "${CACHED[@]}"
    mv "$CACHE_FILE.part" "$CACHE_FILE"
    echo "[build_qwentts] cached ${#CACHED[@]} files to $CACHE_FILE ($(du -h "$CACHE_FILE" | cut -f1))"
fi
echo "[build_qwentts] OK"
