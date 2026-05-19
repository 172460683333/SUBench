#!/bin/bash
set -euo pipefail

REPO="172460683333/SUBench"
TAG="data-v1"
BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"
DATA_DIR="$(cd "$(dirname "$0")" && pwd)/data"

PACKAGES=(
    "benchmarks.tar.gz"
    "Qwen3-Coder-EP.tar.gz"
    "trace-8ep-1k.tar.gz"
    "trace-8ep-in16k.tar.gz"
    "trace-16ep-1k.tar.gz"
    "trace-16ep-in16k.tar.gz"
    "trace-32ep-1k.tar.gz"
    "trace-32ep-in16k.tar.gz"
    "trace-48ep-16k.tar.gz"
    "trace-48ep-1k.tar.gz"
    "trace-48ep-in16k.tar.gz"
    "trace-64ep-in16k.tar.gz"
    "trace-64ep-in1k.tar.gz"
)

usage() {
    echo "Usage: $0 [--all | --benchmarks | --traces | --nsys | PACKAGE_NAME...]"
    echo ""
    echo "Options:"
    echo "  --all          Download all data packages"
    echo "  --benchmarks   Download benchmark results only"
    echo "  --traces       Download all trace sqlite files"
    echo "  --nsys         Download nsys profiling files"
    echo "  PACKAGE_NAME   Download specific package(s), e.g. trace-8ep-1k.tar.gz"
    echo ""
    echo "Available packages:"
    for pkg in "${PACKAGES[@]}"; do
        echo "  $pkg"
    done
}

download_and_extract() {
    local pkg="$1"
    local url="${BASE_URL}/${pkg}"
    echo ">>> Downloading ${pkg} ..."
    curl -L --fail --progress-bar -o "/tmp/${pkg}" "${url}"
    echo ">>> Extracting ${pkg} to ${DATA_DIR}/ ..."
    tar xzf "/tmp/${pkg}" -C "${DATA_DIR}/"
    rm -f "/tmp/${pkg}"
    echo ">>> Done: ${pkg}"
    echo ""
}

if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

mkdir -p "${DATA_DIR}"

selected=()

for arg in "$@"; do
    case "$arg" in
        --all)
            selected=("${PACKAGES[@]}")
            ;;
        --benchmarks)
            selected+=("benchmarks.tar.gz")
            ;;
        --traces)
            for pkg in "${PACKAGES[@]}"; do
                [[ "$pkg" == trace-* ]] && selected+=("$pkg")
            done
            ;;
        --nsys)
            selected+=("Qwen3-Coder-EP.tar.gz")
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            selected+=("$arg")
            ;;
    esac
done

echo "Will download ${#selected[@]} package(s) to ${DATA_DIR}/"
echo ""

for pkg in "${selected[@]}"; do
    download_and_extract "$pkg"
done

echo "All downloads complete!"
