# Nomad Runtime — ADTC 2026 Submission

**Domain:** Autonomous AI Agents  
**Model:** Phi-3-mini-4k-instruct Q4_K_M (llama.cpp)  
**African Use Case:** Automotive diagnostics in English, Swahili, and Hausa  
**Team:** nomad-runtime-team

---

## What is Nomad Runtime?

An adaptive AI inference layer that makes Phi-3-mini-4k-instruct run
correctly and efficiently on the 8 GB commodity laptop — the machine
already sitting on millions of desks across Africa — with zero cloud
dependency. Built on top of it: **Autex**, an AI automotive diagnostic
assistant for mechanics and vehicle owners who lack reliable internet.

```
Autex (application)
    │
    ▼
Nomad Runtime
    ├── Capability Detection  (first boot: what can this device actually run?)
    ├── Live Mode Switching   (every 5s: Nomad / Workhorse / Guardian)
    ├── Adaptive Learning     (learns device patterns and task preferences)
    └── Plugin Architecture  (llama.cpp today, anything tomorrow)
    │
    ▼
Phi-3-mini Q4_K_M via llama.cpp  (2.2 GB RAM, ~12 TPS on i5-11th gen)
```

---

## Quick Start (judges)

```bash
# 1. Download the model (~2.2 GB, one-time)
bash download_model.sh

# 2. Install dependencies
pip install -r requirements.txt
# For best performance, build llama-cpp-python with AVX2:
# CMAKE_ARGS="-DGGML_AVX2=on" pip install llama-cpp-python

# 3. Run
bash run.sh
# → API at http://localhost:8765
# → Dashboard at http://localhost:8765/dashboard
```

---

## ADTC Profiler

```bash
# Install profiler
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"

# Run (requires llama-bench on PATH)
bash run_profiler.sh
# → writes profiler-results/submission.json
```

---

## Submission Checklist

- [x] Public GitHub repository
- [x] `metadata.json` fully filled in — no placeholder values
- [x] Exactly 2 test prompts in `metadata.json` (English + Swahili)
- [x] `download_model.sh` idempotent, no credentials, public URL
- [x] Model is GGUF format (`Phi-3-mini-4k-instruct-Q4_K_M.gguf`)
- [x] `model/*.gguf` in `.gitignore` — weights not committed
- [x] `REPORT.md` complete technical writeup
- [x] `model.runtime` = `"llama.cpp"` — competition requirement met
- [x] Stallion (cloud API) mode disabled in submission build
- [x] 100% offline during inference — zero external network calls
- [x] Runs within 8 GB RAM (peak ~4.1 GB including OS)
- [x] African Use Case Bonus claimed — 3 languages, local vehicles

---

## Repository Structure

```
adtc-2026-submission/
├── metadata.json               ← Team, model, 2 test prompts
├── download_model.sh           ← Downloads Phi-3-mini GGUF
├── REPORT.md                   ← Technical writeup
├── requirements.txt            ← Python dependencies
├── run.sh                      ← One-command startup
├── run_profiler.sh             ← ADTC profiler helper
├── model/
│   └── .gitkeep               ← Weights downloaded here (not committed)
├── profiler-results/
│   └── submission.json         ← Pre-run profiler output (reference)
├── test-prompts/
│   └── africa_prompts.json     ← 6 Africa-specific prompts, 3 languages
└── src/nomad_runtime/
    ├── core/                   ← Device monitor, mode switcher, adaptive learner,
    │                             capability detector, plugin base/registry,
    │                             inference router, benchmarker, ADTC emulator
    ├── plugins/                ← llama_cpp, guardian, ollama, onnxruntime plugins
    └── api/                    ← Flask API server + dashboard
```

---

## License

GNU GPL v3 — see [LICENSE](LICENSE)
