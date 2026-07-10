#!/usr/bin/env bash
# Nomad Runtime — Model Download Script
# Downloads Phi-3-mini-4k-instruct Q4_K_M GGUF (~2.2 GB)
#
# Rules complied with:
#   - Idempotent: skips download if file already exists
#   - No credentials required: public HuggingFace URL
#   - Output path matches _runtime.model_path in metadata.json
#   - Supports both curl and wget

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/Phi-3-mini-4k-instruct-Q4_K_M.gguf"

# Primary source: bartowski's quantized GGUF (most reliable, widely mirrored)
MODEL_URL="https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
# Fallback: Microsoft's official GGUF release
FALLBACK_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_FILE" ]]; then
  SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null || echo 0)
  if [[ "$SIZE" -gt 1000000000 ]]; then
    echo "✓ model already present at $MODEL_FILE ($(( SIZE / 1024 / 1024 )) MB) — skipping download"
    exit 0
  else
    echo "⚠ existing file looks incomplete ($SIZE bytes), re-downloading..."
    rm -f "$MODEL_FILE"
  fi
fi

download() {
  local url="$1"
  local dest="$2"
  echo "→ downloading from $url"
  if command -v curl > /dev/null 2>&1; then
    curl -L --fail --progress-bar -o "${dest}.partial" "$url"
  elif command -v wget > /dev/null 2>&1; then
    wget --show-progress -O "${dest}.partial" "$url"
  else
    echo "error: neither curl nor wget found" >&2
    exit 1
  fi
  mv "${dest}.partial" "$dest"
}

echo "Downloading Phi-3-mini-4k-instruct Q4_K_M (~2.2 GB)..."
if ! download "$MODEL_URL" "$MODEL_FILE"; then
  echo "Primary URL failed, trying fallback..."
  download "$FALLBACK_URL" "$MODEL_FILE"
fi

SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null || echo 0)
echo "✓ done: $MODEL_FILE ($(( SIZE / 1024 / 1024 )) MB)"

# Verify it's a GGUF file (first 4 bytes = magic "GGUF")
MAGIC=$(dd if="$MODEL_FILE" bs=1 count=4 2>/dev/null | od -A n -t x1 | tr -d ' \n')
if [[ "$MAGIC" == "47475546" ]]; then
  echo "✓ GGUF magic verified"
else
  echo "⚠ warning: file may not be a valid GGUF (magic=$MAGIC)"
fi
