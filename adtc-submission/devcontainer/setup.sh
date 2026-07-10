#!/usr/bin/env bash
# .devcontainer/setup.sh
# Runs automatically when the Codespace is first created.
# Do NOT run this manually — it is called by postCreateCommand in devcontainer.json.
 
set -euo pipefail
 
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        Nomad Runtime — Codespace Setup               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
 
# ── 1. System build dependencies ─────────────────────────────────────────
echo "→ Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
  build-essential \
  cmake \
  git \
  curl \
  wget \
  python3-dev \
  libgomp1 \
  2>&1 | tail -3
 
# ── 2. Verify AVX2 support (critical for inference speed) ─────────────────
echo ""
if grep -q avx2 /proc/cpuinfo; then
  echo "✓ AVX2 detected — inference will run at full speed (~12 TPS)"
  AVX2_FLAG="-DGGML_AVX2=on"
else
  echo "⚠ AVX2 not detected — inference will be slower (~3-4 TPS)"
  echo "  Consider upgrading to a 4-core Codespace machine type"
  AVX2_FLAG=""
fi
 
# ── 3. Python virtual environment ─────────────────────────────────────────
echo ""
echo "→ Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
 
# Upgrade pip silently
pip install --upgrade pip -q
 
# ── 4. llama-cpp-python with AVX2 ─────────────────────────────────────────
# This is the most important install — it IS the ADTC-required runtime.
# The AVX2 flag gives 3x speed on x86-64 Codespace machines.
echo ""
echo "→ Installing llama-cpp-python with AVX2 (this takes 2-4 minutes)..."
CMAKE_ARGS="$AVX2_FLAG" pip install llama-cpp-python 2>&1 | tail -5
 
# Verify
python3 -c "import llama_cpp; print('✓ llama-cpp-python installed:', llama_cpp.__version__)"
 
# ── 5. Other Python dependencies ──────────────────────────────────────────
echo ""
echo "→ Installing Python dependencies..."
pip install flask flask-cors psutil requests -q
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git" -q 2>&1 | tail -2
 
echo "✓ Python dependencies installed"
 
# ── 6. Environment variables ──────────────────────────────────────────────
echo ""
echo "→ Writing environment config..."
 
# .env file that server.py reads on startup
cat > .env << 'ENV'
# Nomad Runtime — Codespace environment
ADTC_SUBMISSION=true
NOMAD_PORT=8765
NOMAD_FAST_DEMO=true
NOMAD_RESCAN=false
ENV
 
# Persist LLAMA_CPP_MODEL_DIR in the shell profile so terminals pick it up
WORKSPACE_DIR="/workspaces/$(basename $(pwd))"
echo "export LLAMA_CPP_MODEL_DIR=\"$WORKSPACE_DIR/model\"" >> ~/.bashrc
echo "export ADTC_SUBMISSION=true" >> ~/.bashrc
echo "source $WORKSPACE_DIR/.venv/bin/activate" >> ~/.bashrc
 
# ── 7. Copy dashboard to a serveable location ─────────────────────────────
echo ""
echo "→ Setting up dashboard..."
mkdir -p public
cp src/nomad_runtime/api/server.py src/nomad_runtime/api/server.py   # no-op, just verify exists
# The dashboard HTML needs to reach the API — we'll serve it from port 3000
# after the user confirms the Codespace ports are forwarded
 
# ── 8. Create VS Code tasks.json for one-click launch ─────────────────────
mkdir -p .vscode
cat > .vscode/tasks.json << 'JSON'
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Start Nomad Runtime API",
      "type": "shell",
      "command": "source .venv/bin/activate && export ADTC_SUBMISSION=true && export LLAMA_CPP_MODEL_DIR=$(pwd)/model && cd src/nomad_runtime/api && python3 server.py",
      "group": { "kind": "build", "isDefault": true },
      "presentation": { "reveal": "always", "panel": "new" },
      "problemMatcher": []
    },
    {
      "label": "Serve Dashboard (port 3000)",
      "type": "shell",
      "command": "source .venv/bin/activate && python3 -m http.server 3000 --directory .",
      "presentation": { "reveal": "always", "panel": "new" },
      "problemMatcher": []
    },
    {
      "label": "Download Model",
      "type": "shell",
      "command": "bash download_model.sh",
      "presentation": { "reveal": "always", "panel": "new" },
      "problemMatcher": []
    },
    {
      "label": "Run ADTC Profiler",
      "type": "shell",
      "command": "bash run_profiler.sh",
      "presentation": { "reveal": "always", "panel": "new" },
      "problemMatcher": []
    },
    {
      "label": "Run ADTC Simulation (no model needed)",
      "type": "shell",
      "command": "source .venv/bin/activate && cd src/nomad_runtime/core && python3 adtc_test_runner.py --all-variants",
      "presentation": { "reveal": "always", "panel": "new" },
      "problemMatcher": []
    },
    {
      "label": "Quick Test (Guardian mode, no model needed)",
      "type": "shell",
      "command": "source .venv/bin/activate && cd src/nomad_runtime/api && python3 server.py & sleep 4 && curl -s -X POST http://localhost:8765/chat -H 'Content-Type: application/json' -d '{\"prompt\":\"My engine is making a knocking sound\",\"mode_override\":\"guardian\"}' | python3 -m json.tool",
      "presentation": { "reveal": "always", "panel": "new" },
      "problemMatcher": []
    }
  ]
}
JSON
 
# ── 9. Create API test file for REST Client extension ─────────────────────
cat > nomad_api_tests.http << 'HTTP'
# Nomad Runtime — API Tests
# Use with VS Code "REST Client" extension (humao.rest-client)
# Press "Send Request" above each block to run it
 
@base = http://localhost:8765
 
### 1. Health check (always works)
GET {{base}}/health
 
### 2. Hardware capability scan
GET {{base}}/capability
 
### 3. Live device status + current mode
GET {{base}}/status
 
### 4. Plugin registry — see all 4 engines
GET {{base}}/plugins
 
### 5. Guardian chat — works immediately, no model needed
POST {{base}}/chat
Content-Type: application/json
 
{
  "prompt": "My brake pedal goes to the floor. Is it safe to drive?",
  "mode_override": "guardian"
}
 
### 6. Nomad chat — requires model downloaded
POST {{base}}/chat
Content-Type: application/json
 
{
  "prompt": "My 2014 Toyota Corolla shows OBD code P0301. What does this mean and what should I check first?",
  "mode_override": "nomad"
}
 
### 7. Swahili prompt (Africa use case)
POST {{base}}/chat
Content-Type: application/json
 
{
  "prompt": "Gari langu la Bajaj Boxer haliwaki asubuhi wakati wa baridi. Sauti ya kubonyeza inasikika. Tatizo hili linaweza kuwa nini?",
  "mode_override": "nomad"
}
 
### 8. Force a mode manually
POST {{base}}/mode/force
Content-Type: application/json
 
{
  "mode": "workhorse"
}
 
### 9. Adaptive learning — device profile
GET {{base}}/learning/device
 
### 10. Adaptive learning — task quality scores
GET {{base}}/learning/tasks
 
### 11. Run benchmark (Guardian, instant)
GET {{base}}/benchmark?mode=guardian
 
### 12. Plugin health — llama.cpp
GET {{base}}/plugins/llama_cpp/health
 
### 13. Plugin health — guardian (always healthy)
GET {{base}}/plugins/guardian/health
 
### 14. Mode switch history
GET {{base}}/history
 
### 15. Reset learning state
POST {{base}}/learning/reset
HTTP
 
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Nomad Runtime Codespace is ready!                   ║"
echo "║                                                      ║"
echo "║  Next steps:                                         ║"
echo "║  1. Ctrl+Shift+B → 'Start Nomad Runtime API'        ║"
echo "║     (works immediately — Guardian mode needs no      ║"
echo "║      model download)                                 ║"
echo "║                                                      ║"
echo "║  2. To run full AI inference:                        ║"
echo "║     Ctrl+Shift+B → 'Download Model' (~2.2 GB)        ║"
echo "║     then → 'Start Nomad Runtime API'                 ║"
echo "║                                                      ║"
echo "║  3. Open nomad_api_tests.http and click              ║"
echo "║     'Send Request' on any block to test the API      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
 
