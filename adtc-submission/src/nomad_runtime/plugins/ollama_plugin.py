"""
Nomad Runtime — Ollama Plugin
═══════════════════════════════════════════════════════════════════════
The reference InferencePlugin implementation. This is the exact logic
that used to be hardcoded inside InferenceRouter._ollama_infer() —
moved here unchanged in behaviour, but now structurally just one
interchangeable plugin among many rather than a special case the
router knows about by name.

If Ollama disappeared tomorrow, deleting this single file (and
registering a replacement plugin for whatever replaced it) is the
entire migration — InferenceRouter, PluginRegistry, the API server,
and the dashboard all stay exactly as they are.
═══════════════════════════════════════════════════════════════════════
"""

import json
import os
import time
import urllib.error
import urllib.request

from plugin_base import InferencePlugin, PluginCapabilities, PluginHealthStatus, PluginInferenceResult


OLLAMA_BASE    = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_TIMEOUT = 120   # seconds


class OllamaPlugin(InferencePlugin):
    """Talks to a local Ollama daemon over its HTTP API."""

    engine_id = "ollama"

    def capabilities(self) -> PluginCapabilities:
        return PluginCapabilities(
            engine_name          = "Ollama",
            engine_version        = self._detect_version(),
            supports_streaming     = True,    # Ollama supports it; this plugin uses stream=False for simplicity
            supports_chat_format   = True,
            requires_network        = False,   # local daemon, no internet needed
            requires_gpu            = False,   # runs fine CPU-only, which is the whole point for ADTC
            typical_formats          = ["gguf"],
        )

    def health_check(self) -> PluginHealthStatus:
        """Fast check: is the Ollama daemon reachable at all."""
        try:
            req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return PluginHealthStatus(healthy=True, message="Ollama daemon reachable")
                return PluginHealthStatus(healthy=False, message=f"Ollama returned HTTP {resp.status}")
        except urllib.error.URLError as e:
            return PluginHealthStatus(
                healthy=False,
                message=f"Ollama unreachable at {OLLAMA_BASE} ({e}). Run: ollama serve",
            )
        except Exception as e:
            return PluginHealthStatus(healthy=False, message=f"health check failed: {e}")

    def infer(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> PluginInferenceResult:
        t0 = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model":    model_id,
            "messages": messages,
            "stream":   False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            return PluginInferenceResult(
                success=False,
                error=f"Ollama unreachable: {e}. Is ollama running? Try: ollama serve",
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return PluginInferenceResult(
                success=False, error=str(e), latency_ms=(time.time() - t0) * 1000,
            )

        latency_ms   = (time.time() - t0) * 1000
        response_txt = data.get("message", {}).get("content", "")

        return PluginInferenceResult(
            success        = True,
            response_text  = response_txt,
            prompt_tokens   = data.get("prompt_eval_count", 0),
            output_tokens   = data.get("eval_count", 0),
            latency_ms      = latency_ms,
            raw_engine_meta  = {
                "total_duration_ns": data.get("total_duration"),
                "load_duration_ns":   data.get("load_duration"),
            },
        )

    def list_models(self) -> list[dict]:
        try:
            req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return data.get("models", [])
        except Exception:
            return []

    def estimate_ram_gb(self, model_id: str) -> float:
        """
        Best-effort estimate from Ollama's own /api/tags size field when
        the model is already pulled; falls back to a conservative guess
        by parameter-count heuristics in the model_id string otherwise.
        """
        for m in self.list_models():
            if m.get("name", "").startswith(model_id.split(":")[0]):
                size_bytes = m.get("size", 0)
                if size_bytes:
                    return round(size_bytes / 1024**3, 2)

        # Fallback heuristic from common naming conventions
        mid = model_id.lower()
        if "0.5b" in mid or "mini" in mid:
            return 1.0
        if "3b" in mid:
            return 3.8
        if "7b" in mid or "8b" in mid:
            return 6.5
        return 2.5   # unknown — conservative mid estimate

    def _detect_version(self) -> str:
        try:
            req = urllib.request.Request(f"{OLLAMA_BASE}/api/version")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                return data.get("version", "unknown")
        except Exception:
            return "unknown (daemon unreachable)"


# ═══════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    plugin = OllamaPlugin()
    print("Capabilities:", plugin.capabilities())

    health = plugin.health_check()
    print(f"\nHealth: healthy={health.healthy} — {health.message}")

    if health.healthy:
        models = plugin.list_models()
        print(f"\nModels available: {[m['name'] for m in models]}")
        if models:
            result = plugin.infer(model_id=models[0]["name"], prompt="Say hello in 5 words")
            print(f"\nTest inference: success={result.success}")
            print(f"  Response: {result.response_text}")
    else:
        print("\nSkipping inference test — Ollama daemon not reachable")
