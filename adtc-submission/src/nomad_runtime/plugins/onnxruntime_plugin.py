"""
Nomad Runtime — ONNX Runtime Plugin
═══════════════════════════════════════════════════════════════════════
Demonstrates how a new inference engine is added to Nomad Runtime:
one file, zero changes to the router, registry, API, or dashboard.

ONNX Runtime is a cross-platform accelerator from Microsoft that
supports INT8/FP16 quantized models on CPU, GPU, and NPU. It's
particularly relevant for the ADTC target hardware because Intel's
AI Boost NPU (present in Core Ultra CPUs) is exposed via ONNX Runtime's
OpenVINO EP — meaning this plugin could give 2–4× speedup over pure
CPU inference on newer laptops, for free.

Status: STUB — the structural contract is complete and tested.
        Activate by installing onnxruntime and pointing MODEL_MAPPING
        at real .onnx model files. No other file in the codebase changes.

How to activate
---------------
1. Install the runtime:
       pip install onnxruntime          # CPU-only
       pip install onnxruntime-gpu      # GPU/NPU support
       pip install optimum[onnxruntime] # HuggingFace export helper

2. Export a model to ONNX:
       optimum-cli export onnx \
           --model microsoft/phi-3-mini-4k-instruct \
           ./models/phi3-mini-onnx

3. Update ONNX_MODEL_DIR env var to point at your models:
       export ONNX_MODEL_DIR=./models

4. Done — PluginRegistry discovers this file automatically on next boot.
═══════════════════════════════════════════════════════════════════════
"""

import os
import time
from pathlib import Path

from plugin_base import (
    InferencePlugin, PluginCapabilities, PluginHealthStatus, PluginInferenceResult,
)

# Runtime and tokenizer are imported lazily (inside infer/health_check)
# so the plugin loads cleanly even when onnxruntime is not installed —
# the registry just marks it as unhealthy and moves on.
ONNX_MODEL_DIR = Path(os.getenv("ONNX_MODEL_DIR", "./models/onnx"))


class ONNXRuntimePlugin(InferencePlugin):
    """
    Runs GGUF-exported or optimum-exported ONNX language models via
    Microsoft's onnxruntime library, with optional NPU acceleration
    through the OpenVINO Execution Provider on Intel Core Ultra hardware.
    """

    engine_id = "onnxruntime"

    def capabilities(self) -> PluginCapabilities:
        version = self._detect_version()
        return PluginCapabilities(
            engine_name           = "ONNX Runtime",
            engine_version         = version,
            supports_streaming      = False,
            supports_chat_format    = True,
            requires_network         = False,
            requires_gpu             = False,    # works CPU-only; GPU/NPU optional
            typical_formats           = ["onnx"],
        )

    def health_check(self) -> PluginHealthStatus:
        # Check 1: is onnxruntime installed?
        try:
            import onnxruntime as ort
        except ImportError:
            return PluginHealthStatus(
                healthy=False,
                message="onnxruntime not installed. Run: pip install onnxruntime",
            )

        # Check 2: is the model directory populated?
        if not ONNX_MODEL_DIR.exists() or not any(ONNX_MODEL_DIR.glob("**/*.onnx")):
            return PluginHealthStatus(
                healthy=False,
                message=(
                    f"No .onnx models found in {ONNX_MODEL_DIR}. "
                    "Export a model with: optimum-cli export onnx --model <hf_id> <output_dir>"
                ),
            )

        # Check 3: can we list execution providers (basic runtime sanity check)?
        try:
            providers = ort.get_available_providers()
            npu = "OpenVINOExecutionProvider" in providers
            gpu = "CUDAExecutionProvider" in providers
            accel = ("NPU via OpenVINO" if npu else ("GPU via CUDA" if gpu else "CPU only"))
            return PluginHealthStatus(
                healthy=True,
                message=f"onnxruntime {ort.__version__} ready. Acceleration: {accel}. "
                        f"Providers: {providers}",
            )
        except Exception as e:
            return PluginHealthStatus(healthy=False, message=f"onnxruntime runtime error: {e}")

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
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as e:
            return PluginInferenceResult(
                success=False,
                error=f"onnxruntime or transformers not installed: {e}",
                latency_ms=(time.time() - t0) * 1000,
            )

        # Resolve model path: model_id is either a subdirectory name under
        # ONNX_MODEL_DIR, or a full path.
        model_path = Path(model_id) if Path(model_id).exists() else ONNX_MODEL_DIR / model_id
        if not model_path.exists():
            return PluginInferenceResult(
                success=False,
                error=f"model path not found: {model_path}",
                latency_ms=(time.time() - t0) * 1000,
            )

        try:
            # Pick the best available EP automatically
            providers = ort.get_available_providers()
            ep = (
                "OpenVINOExecutionProvider" if "OpenVINOExecutionProvider" in providers else
                "CUDAExecutionProvider"      if "CUDAExecutionProvider"     in providers else
                "CPUExecutionProvider"
            )

            tokenizer  = AutoTokenizer.from_pretrained(str(model_path))
            session    = ort.InferenceSession(
                str(model_path / "model.onnx"), providers=[ep],
            )

            # Build prompt text
            full_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt
            inputs = tokenizer(full_prompt, return_tensors="np")

            # Run inference
            outputs = session.run(None, dict(inputs))
            output_ids = outputs[0][0]
            response_text = tokenizer.decode(output_ids[inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True)

            return PluginInferenceResult(
                success        = True,
                response_text  = response_text.strip(),
                prompt_tokens   = int(inputs["input_ids"].shape[1]),
                output_tokens   = len(output_ids) - int(inputs["input_ids"].shape[1]),
                latency_ms      = (time.time() - t0) * 1000,
                raw_engine_meta  = {"execution_provider": ep, "model_path": str(model_path)},
            )

        except Exception as e:
            return PluginInferenceResult(
                success=False, error=str(e), latency_ms=(time.time() - t0) * 1000,
            )

    def list_models(self) -> list[dict]:
        models = []
        if ONNX_MODEL_DIR.exists():
            for d in ONNX_MODEL_DIR.iterdir():
                if d.is_dir() and any(d.glob("*.onnx")):
                    models.append({"name": d.name, "engine": "onnxruntime"})
        return models

    def estimate_ram_gb(self, model_id: str) -> float:
        # ONNX INT8 models are roughly 30-40% smaller than F16 GGUF equivalents.
        mid = model_id.lower()
        if "0.5b" in mid or "mini" in mid:
            return 0.7
        if "3b" in mid:
            return 2.5
        if "7b" in mid or "8b" in mid:
            return 4.5
        return 2.0

    @staticmethod
    def _detect_version() -> str:
        try:
            import onnxruntime as ort
            return ort.__version__
        except ImportError:
            return "not installed"


if __name__ == "__main__":
    plugin = ONNXRuntimePlugin()
    print("Capabilities:", plugin.capabilities())
    health = plugin.health_check()
    print(f"Health: healthy={health.healthy} — {health.message}")
