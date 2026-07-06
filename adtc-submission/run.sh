#!/usr/bin/env bash
# Nomad Runtime — one-command startup for judges
# Usage: bash run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 1. Download model if not present
if [[ ! -f "model/Phi-3-mini-4k-instruct-Q4_K_M.gguf" ]]; then
  echo "Model not found — running download_model.sh..."
  bash download_model.sh
fi

# 2. Check Python deps
python3 -c "import flask, psutil" 2>/dev/null || {
  echo "Installing Python dependencies..."
  pip install -r requirements.txt --break-system-packages -q
}

# 3. Set ADTC submission mode (disables cloud Stallion)
export ADTC_SUBMISSION=true
export LLAMA_CPP_MODEL_DIR="$HERE/model"

# 4. Start the API server
echo ""
echo "Starting Nomad Runtime API on http://localhost:8765"
echo "Dashboard: http://localhost:8765/dashboard"
echo ""
exec python3 src/nomad_runtime/api/server.py
