"""
Nomad Runtime — Plugin Contract
═══════════════════════════════════════════════════════════════════════
This is the entire surface a new inference engine has to implement to
become a Nomad Runtime backend. Ollama, llama.cpp, TensorRT, ExecuTorch,
ONNX Runtime, a cloud API — all of these are plugins, not special cases
hardcoded into the router.

Why this exists
----------------
Without this contract, adding a new engine means editing InferenceRouter
itself: a new `if cfg["backend"] == "newengine"` branch, threading a new
config shape through MODEL_REGISTRY, and risking the existing Ollama path
every time. That doesn't scale — "after five years, a new engine will
appear" is a near-certainty in this space (today it's llama.cpp/Ollama,
tomorrow it might be a wasm runtime or a vendor-specific NPU SDK).

With this contract, a new engine is a single new file that subclasses
InferencePlugin, dropped into the plugins/ directory. The PluginRegistry
discovers it automatically. The router never changes.

Design principles
------------------
1. Plugins are responsible for their OWN engine's wire protocol (HTTP,
   gRPC, native bindings — whatever). The contract only cares about the
   plugin's *outputs*, not how it gets there.
2. Plugins declare what they can do (capabilities()) so the registry and
   capability detector can reason about fit *before* attempting a call —
   consistent with the "don't waste time trying things that won't work"
   principle from capability detection.
3. Plugins must fail gracefully and return a structured error, never
   raise an uncaught exception into the router — one broken plugin must
   never take down inference for every other mode.
4. health_check() lets the registry skip plugins whose engine isn't
   even running (e.g. Ollama installed but `ollama serve` not started)
   without paying the cost of a real inference attempt first.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
#  Shared data shapes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PluginCapabilities:
    """
    What a plugin can do, declared up front. Used by PluginRegistry to
    pick a plugin for a given model identifier and by the dashboard to
    show what engines are available, without making a real inference call.
    """
    engine_name:        str                  # "ollama", "tensorrt", "onnxruntime", ...
    engine_version:      str = "unknown"
    supports_streaming:   bool = False
    supports_chat_format: bool = True         # vs. raw completion-only
    requires_network:     bool = False        # True for cloud-backed engines
    requires_gpu:         bool = False
    typical_formats:      list[str] = field(default_factory=list)   # e.g. ["gguf"], ["onnx"], ["engine"]


@dataclass
class PluginInferenceResult:
    """
    The plugin-level result shape. InferenceRouter adapts this into the
    existing InferenceResult dataclass so callers (the API server, the
    benchmarker) don't need to change regardless of which plugin ran.
    """
    success:        bool
    response_text:  str  = ""
    prompt_tokens:  int  = 0
    output_tokens:  int  = 0
    latency_ms:     float = 0.0
    error:          Optional[str] = None
    raw_engine_meta: dict = field(default_factory=dict)   # engine-specific extras, passthrough


@dataclass
class PluginHealthStatus:
    healthy:  bool
    message:  str = ""
    checked_at: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════
#  The contract itself
# ═══════════════════════════════════════════════════════════════════════════

class InferencePlugin(ABC):
    """
    Base class every inference engine plugin must subclass.

    Minimum required overrides: capabilities(), health_check(), infer().
    Everything else has a sensible default.

    A new engine (TensorRT, ExecuTorch, ONNX Runtime, a future engine
    nobody has invented yet) is added by creating ONE new file that
    subclasses this and dropping it in plugins/ — the router and registry
    require zero changes.
    """

    # Override in subclasses — must be a short, stable, lowercase identifier
    # used to address this plugin from MODEL_REGISTRY/config (e.g. "ollama").
    engine_id: str = "base"

    @abstractmethod
    def capabilities(self) -> PluginCapabilities:
        """Declare what this plugin's engine can do."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> PluginHealthStatus:
        """
        Cheap, fast check of whether the underlying engine is reachable
        and ready (e.g. is the Ollama daemon running, is the TensorRT
        runtime initialised). Should NOT attempt a real inference call —
        keep this fast enough to run before every request if needed.
        """
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> PluginInferenceResult:
        """
        Run inference. Must NEVER raise — catch everything internally and
        return a PluginInferenceResult with success=False and a populated
        error string. The router treats an uncaught exception here as a
        plugin bug, not a normal inference failure, and will disable the
        plugin for the remainder of the session to protect other modes.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Optional overrides — sensible defaults provided                     #
    # ------------------------------------------------------------------ #

    def list_models(self) -> list[dict]:
        """
        Models currently available/loaded for this engine, if discoverable.
        Default: empty list (not every engine can introspect this cheaply).
        """
        return []

    def estimate_ram_gb(self, model_id: str) -> float:
        """
        Best-effort RAM estimate for a given model on this engine.
        Used by capability detection / mode switching to avoid attempting
        a model the device's hardware ceiling rules out. Default: unknown.
        """
        return 0.0

    def describe(self) -> str:
        """Human-readable one-liner for dashboards/logs."""
        caps = self.capabilities()
        return f"{caps.engine_name} v{caps.engine_version}"


# ═══════════════════════════════════════════════════════════════════════════
#  Plugin-level exception — distinguishes "plugin is broken" from
#  "inference failed normally" (e.g. engine not running)
# ═══════════════════════════════════════════════════════════════════════════

class PluginFault(Exception):
    """
    Raised by PluginRegistry (never by well-behaved plugins themselves)
    when a plugin's infer() raises an uncaught exception, violating the
    contract. Signals the registry should quarantine that plugin.
    """
    def __init__(self, engine_id: str, original: Exception):
        self.engine_id = engine_id
        self.original  = original
        super().__init__(f"Plugin '{engine_id}' violated its contract: {original}")
