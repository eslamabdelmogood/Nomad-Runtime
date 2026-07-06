"""
Nomad Runtime — llama.cpp Plugin (Direct Binding)
═══════════════════════════════════════════════════════════════════════
Runs GGUF quantized models via llama-cpp-python — the Python binding
for llama.cpp — WITHOUT going through the Ollama HTTP layer. This
gives lower latency and more fine-grained control over context size,
KV-cache, and threading, at the cost of managing model loading yourself.

For the ADTC competition this is relevant because:
  - Removes the Ollama daemon dependency (just pip install llama-cpp-python)
  - Exposes per-request threading control (important for thermal management)
  - Supports all GGUF formats Ollama supports, with the same Q4 quants

Status: STUB — structurally complete. Activate by installing the library.
        No other file changes when you activate this plugin.

How to activate
---------------
    # CPU-optimised build with AVX2 (matches ADTC hardware)
    CMAKE_ARGS="-DGGML_AVX2=on" pip install llama-cpp-python

    # Point at a GGUF file
    export LLAMA_CPP_MODEL_DIR=./models/gguf

    # That's it — PluginRegistry picks this up on next boot.
═══════════════════════════════════════════════════════════════════════
"""

import os
import time
from pathlib import Path

from plugin_base import (
    InferencePlugin, PluginCapabilities, PluginHealthStatus, PluginInferenceResult,
)

LLAMA_CPP_MODEL_DIR = Path(os.getenv("LLAMA_CPP_MODEL_DIR", "./models/gguf"))
LLAMA_CPP_N_CTX     = int(os.getenv("LLAMA_CPP_N_CTX", "2048"))
LLAMA_CPP_N_THREADS = int(os.getenv("LLAMA_CPP_N_THREADS", "4"))


class LlamaCppPlugin(InferencePlugin):
    """
    Direct llama.cpp binding via llama-cpp-python.
    Loads models on demand and caches the loaded instance to avoid
    re-loading the same model for every request.
    """

    engine_id = "llama_cpp"

    def __init__(self):
        self._loaded_models: dict[str, object] = {}   # model_path → Llama instance

    def capabilities(self) -> PluginCapabilities:
        version = self._detect_version()
        return PluginCapabilities(
            engine_name           = "llama.cpp (direct)",
            engine_version         = version,
            supports_streaming      = True,
            supports_chat_format    = True,
            requires_network         = False,
            requires_gpu             = False,
            typical_formats           = ["gguf"],
        )

    def health_check(self) -> PluginHealthStatus:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return PluginHealthStatus(
                healthy=False,
                message=(
                    "llama-cpp-python not installed. "
                    "Install with: CMAKE_ARGS='-DGGML_AVX2=on' pip install llama-cpp-python"
                ),
            )

        if not LLAMA_CPP_MODEL_DIR.exists() or not any(LLAMA_CPP_MODEL_DIR.glob("**/*.gguf")):
            return PluginHealthStatus(
                healthy=False,
                message=(
                    f"No .gguf files found in {LLAMA_CPP_MODEL_DIR}. "
                    "Download a Q4_K_M GGUF from HuggingFace (e.g. bartowski/Phi-3-mini-4k-instruct-GGUF)"
                ),
            )

        return PluginHealthStatus(
            healthy=True,
            message=f"llama-cpp-python ready, {sum(1 for _ in LLAMA_CPP_MODEL_DIR.glob('**/*.gguf'))} .gguf file(s) found",
        )

    def infer(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> PluginInferenceResult:
        t0 = time.time()

        try:
            from llama_cpp import Llama
        except ImportError as e:
            return PluginInferenceResult(
                success=False, error=f"llama-cpp-python not installed: {e}",
                latency_ms=(time.time() - t0) * 1000,
            )

        # Resolve model file
        model_path = self._resolve_model(model_id)
        if model_path is None:
            return PluginInferenceResult(
                success=False,
                error=f"model '{model_id}' not found under {LLAMA_CPP_MODEL_DIR}",
                latency_ms=(time.time() - t0) * 1000,
            )

        try:
            # Load and cache — re-loading a 2GB model on every request would be
            # unusably slow; caching means the first request pays the load cost
            # (typically 3-5s) and subsequent requests go directly to inference.
            llm = self._get_or_load(str(model_path), Llama)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            text        = response["choices"][0]["message"]["content"]
            usage       = response.get("usage", {})
            prompt_tok  = usage.get("prompt_tokens", 0)
            output_tok  = usage.get("completion_tokens", 0)

            return PluginInferenceResult(
                success        = True,
                response_text  = text.strip(),
                prompt_tokens   = prompt_tok,
                output_tokens   = output_tok,
                latency_ms      = (time.time() - t0) * 1000,
                raw_engine_meta  = {
                    "model_path": str(model_path),
                    "n_ctx":      LLAMA_CPP_N_CTX,
                    "n_threads":  LLAMA_CPP_N_THREADS,
                },
            )
        except Exception as e:
            return PluginInferenceResult(
                success=False, error=str(e), latency_ms=(time.time() - t0) * 1000,
            )

    def list_models(self) -> list[dict]:
        if not LLAMA_CPP_MODEL_DIR.exists():
            return []
        return [
            {"name": f.stem, "path": str(f), "engine": "llama_cpp"}
            for f in LLAMA_CPP_MODEL_DIR.glob("**/*.gguf")
        ]

    def estimate_ram_gb(self, model_id: str) -> float:
        mid = model_id.lower()
        # Q4_K_M quant approximations
        if "0.5b" in mid:
            return 0.8
        if "3b" in mid or "3.8b" in mid:
            return 2.8
        if "7b" in mid or "8b" in mid:
            return 5.0
        if "13b" in mid:
            return 9.0
        return 3.0

    def _resolve_model(self, model_id: str) -> Path | None:
        # Try exact path first
        p = Path(model_id)
        if p.exists() and p.suffix == ".gguf":
            return p
        # Search by stem match under MODEL_DIR
        for f in LLAMA_CPP_MODEL_DIR.glob("**/*.gguf"):
            if model_id.replace(":", "_") in f.stem or model_id in f.stem:
                return f
        return None

    def _get_or_load(self, model_path: str, Llama) -> object:
        if model_path not in self._loaded_models:
            self._loaded_models[model_path] = Llama(
                model_path=model_path,
                n_ctx=LLAMA_CPP_N_CTX,
                n_threads=LLAMA_CPP_N_THREADS,
                verbose=False,
            )
        return self._loaded_models[model_path]

    @staticmethod
    def _detect_version() -> str:
        try:
            import llama_cpp
            return getattr(llama_cpp, "__version__", "installed")
        except ImportError:
            return "not installed"


if __name__ == "__main__":
    plugin = LlamaCppPlugin()
    print("Capabilities:", plugin.capabilities())
    health = plugin.health_check()
    print(f"Health: healthy={health.healthy} — {health.message}")
