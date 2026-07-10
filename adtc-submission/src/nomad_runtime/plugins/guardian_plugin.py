"""
Nomad Runtime — Guardian Plugin
═══════════════════════════════════════════════════════════════════════
Deterministic, rule-based fallback. No model, no network, no GPU —
just keyword matching. This is deliberately implemented as a plugin
too, not a special case in the router, to prove the InferencePlugin
contract isn't secretly ML-specific: any "thing that turns a prompt
into a response" can be a plugin, including a zero-dependency rules
engine that always succeeds even when every real model backend is
down. This is what keeps Autex responding instead of crashing when
the device is in Guardian mode.
═══════════════════════════════════════════════════════════════════════
"""

import time

from plugin_base import InferencePlugin, PluginCapabilities, PluginHealthStatus, PluginInferenceResult


class GuardianPlugin(InferencePlugin):
    """Always-available deterministic fallback — no external engine required."""

    engine_id = "guardian"

    RESPONSES = {
        "keywords": {
            "overheat":   "WARNING: Engine overheating detected. Stop the vehicle and allow it to cool. Check coolant level.",
            "misfire":    "ALERT: Engine misfire pattern detected. Reduce speed and seek service soon.",
            "oil":        "NOTICE: Oil-related anomaly detected. Check oil level and pressure when safe.",
            "brake":      "WARNING: Brake system alert. Test brakes cautiously and seek immediate service.",
            "battery":    "NOTICE: Battery voltage low. Check alternator and battery connections.",
        },
        "default": (
            "Device resources are critically low. "
            "Nomad Runtime is operating in Guardian mode. "
            "Basic monitoring continues. Please free up memory or reduce CPU load."
        ),
    }

    def capabilities(self) -> PluginCapabilities:
        return PluginCapabilities(
            engine_name          = "Guardian (deterministic rules)",
            engine_version         = "1.0.0",
            supports_streaming      = False,
            supports_chat_format    = False,
            requires_network         = False,
            requires_gpu             = False,
            typical_formats           = [],   # no model file format — pure code
        )

    def health_check(self) -> PluginHealthStatus:
        # Always healthy — this is the point of Guardian mode: it has no
        # external dependency that could be down.
        return PluginHealthStatus(healthy=True, message="always available, no external dependency")

    def infer(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> PluginInferenceResult:
        t0 = time.time()
        pl = prompt.lower()
        text = self.RESPONSES["default"]
        matched_keyword = None
        for kw, msg in self.RESPONSES["keywords"].items():
            if kw in pl:
                text = msg
                matched_keyword = kw
                break

        return PluginInferenceResult(
            success        = True,
            response_text  = text,
            prompt_tokens   = len(prompt.split()),
            output_tokens   = len(text.split()),
            latency_ms      = (time.time() - t0) * 1000,
            raw_engine_meta  = {"matched_keyword": matched_keyword},
        )

    def estimate_ram_gb(self, model_id: str) -> float:
        return 0.0   # genuinely zero — no model loaded


if __name__ == "__main__":
    plugin = GuardianPlugin()
    print("Capabilities:", plugin.capabilities())
    print("Health:", plugin.health_check())
    result = plugin.infer(model_id="none", prompt="Engine overheat detected on sensor 3")
    print(f"\nResponse: {result.response_text}")
    print(f"Latency: {result.latency_ms:.2f}ms")
