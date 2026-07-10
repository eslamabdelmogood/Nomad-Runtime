#!/usr/bin/env bash
# Run the ADTC profiler against this submission
# Requires: llama-bench on PATH (from llama.cpp build)
# Install profiler: pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ ! -f "model/Phi-3-mini-4k-instruct-Q4_K_M.gguf" ]]; then
  echo "Run ./download_model.sh first"
  exit 1
fi

mkdir -p profiler-results

echo "Running ADTC profiler (participant mode)..."
adtc-profiler run \
  --submission "$HERE" \
  --mode participant \
  --output "$HERE/profiler-results/submission.json" \
  --skip-accuracy \
  --seed 42

echo ""
echo "Results written to profiler-results/submission.json"
echo ""
cat profiler-results/submission.json | python3 -m json.tool | head -40
