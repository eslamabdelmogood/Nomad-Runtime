"""
Nomad Runtime — Plugin Registry
═══════════════════════════════════════════════════════════════════════
Discovers InferencePlugin subclasses in the plugins/ directory at
startup, instantiates them, and exposes a single lookup surface the
InferenceRouter uses to dispatch by engine_id.

This is the piece that makes the "drop a new file, don't touch the
router" promise real. Adding TensorRT support means:

    1. Create plugins/tensorrt_plugin.py subclassing InferencePlugin
    2. Set engine_id = "tensorrt"
    3. Done — PluginRegistry finds it automatically on next boot.

Quarantine
----------
If a plugin's infer() raises an uncaught exception (violating the
contract — well-behaved plugins should never do this, but we don't
trust that blindly), the registry quarantines it: marks it unhealthy
and stops routing to it for the rest of the process lifetime, so one
broken third-party plugin can't take down every other mode. This
mirrors why Guardian mode exists at the mode-switcher level — defence
in depth against a single failure point.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from plugin_base import InferencePlugin, PluginHealthStatus, PluginFault


PLUGINS_DIR = Path(__file__).parent / "plugins"


@dataclass
class PluginRecord:
    """Internal bookkeeping the registry keeps per loaded plugin."""
    plugin:           InferencePlugin
    engine_id:         str
    source_file:        str
    loaded_at:          float
    quarantined:        bool = False
    quarantine_reason:   str = ""
    last_health:         Optional[PluginHealthStatus] = None
    last_health_check_ts: float = 0.0
    call_count:           int = 0
    error_count:           int = 0


class PluginRegistry:
    """
    Discovers and manages all InferencePlugin implementations.

    Usage:
        registry = PluginRegistry()
        registry.discover()                      # scan plugins/ dir
        plugin = registry.get("ollama")           # lookup by engine_id
        result = registry.safe_infer("ollama", model_id="phi3:mini", prompt="...")
    """

    HEALTH_CHECK_CACHE_S = 10   # don't re-check health more than once per N seconds

    def __init__(self, plugins_dir: Path = PLUGINS_DIR):
        self._plugins_dir = plugins_dir
        self._records: dict[str, PluginRecord] = {}

    # ------------------------------------------------------------------ #
    #  Discovery                                                           #
    # ------------------------------------------------------------------ #

    def discover(self) -> list[str]:
        """
        Scans plugins_dir for *.py files, imports each, and registers any
        InferencePlugin subclass found. Returns the list of engine_ids
        successfully loaded. A broken plugin file (syntax error, import
        error) is logged and skipped — it must never prevent the other
        plugins, or the runtime itself, from starting.
        """
        loaded = []
        if not self._plugins_dir.exists():
            print(f"[plugin-registry] plugins dir not found: {self._plugins_dir}")
            return loaded

        # Ensure the plugins dir is importable
        sys.path.insert(0, str(self._plugins_dir.parent))

        for py_file in sorted(self._plugins_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue   # skip __init__.py and private helper files
            try:
                engine_ids = self._load_file(py_file)
                loaded.extend(engine_ids)
            except Exception as e:
                print(f"[plugin-registry] FAILED to load {py_file.name}: {e}")
                # Deliberately swallow — one bad plugin file must not stop
                # discovery of the others, nor crash the runtime at boot.
                continue

        return loaded

    def _load_file(self, py_file: Path) -> list[str]:
        module_name = f"nomad_plugins.{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not build import spec for {py_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        found = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, InferencePlugin)
                and obj is not InferencePlugin
                and obj.__module__ == module_name
            ):
                engine_id = self._instantiate_and_register(obj, py_file)
                if engine_id:
                    found.append(engine_id)
        return found

    def _instantiate_and_register(self, plugin_cls: type, source_file: Path) -> Optional[str]:
        try:
            instance = plugin_cls()
        except Exception as e:
            print(f"[plugin-registry] FAILED to instantiate {plugin_cls.__name__}: {e}")
            return None

        engine_id = getattr(instance, "engine_id", None) or plugin_cls.__name__.lower()

        if engine_id in self._records:
            print(
                f"[plugin-registry] WARNING: duplicate engine_id '{engine_id}' "
                f"from {source_file.name} — keeping first-loaded, skipping this one"
            )
            return None

        self._records[engine_id] = PluginRecord(
            plugin       = instance,
            engine_id    = engine_id,
            source_file  = source_file.name,
            loaded_at    = time.time(),
        )
        caps = instance.capabilities()
        print(f"[plugin-registry] loaded '{engine_id}' from {source_file.name} ({caps.engine_name} v{caps.engine_version})")
        return engine_id

    def register_instance(self, plugin: InferencePlugin) -> str:
        """
        Manually register an already-constructed plugin instance —
        useful for tests, or built-in plugins that ship inline rather
        than as a discoverable file.
        """
        engine_id = plugin.engine_id
        self._records[engine_id] = PluginRecord(
            plugin=plugin, engine_id=engine_id,
            source_file="<manual>", loaded_at=time.time(),
        )
        return engine_id

    # ------------------------------------------------------------------ #
    #  Lookup                                                              #
    # ------------------------------------------------------------------ #

    def get(self, engine_id: str) -> Optional[InferencePlugin]:
        rec = self._records.get(engine_id)
        return rec.plugin if rec else None

    def list_engines(self) -> list[dict]:
        """Dashboard/API-friendly summary of every loaded plugin."""
        out = []
        for rec in self._records.values():
            caps = rec.plugin.capabilities()
            out.append({
                "engine_id":         rec.engine_id,
                "engine_name":       caps.engine_name,
                "engine_version":    caps.engine_version,
                "source_file":       rec.source_file,
                "quarantined":       rec.quarantined,
                "quarantine_reason": rec.quarantine_reason,
                "requires_network":  caps.requires_network,
                "requires_gpu":      caps.requires_gpu,
                "supports_streaming": caps.supports_streaming,
                "typical_formats":    caps.typical_formats,
                "call_count":         rec.call_count,
                "error_count":        rec.error_count,
                "healthy":            rec.last_health.healthy if rec.last_health else None,
                "health_message":     rec.last_health.message if rec.last_health else "not yet checked",
            })
        return out

    # ------------------------------------------------------------------ #
    #  Health                                                              #
    # ------------------------------------------------------------------ #

    def check_health(self, engine_id: str, force: bool = False) -> PluginHealthStatus:
        rec = self._records.get(engine_id)
        if rec is None:
            return PluginHealthStatus(healthy=False, message=f"no plugin registered for '{engine_id}'")

        if rec.quarantined:
            return PluginHealthStatus(healthy=False, message=f"quarantined: {rec.quarantine_reason}")

        now = time.time()
        if not force and rec.last_health and (now - rec.last_health_check_ts) < self.HEALTH_CHECK_CACHE_S:
            return rec.last_health

        try:
            status = rec.plugin.health_check()
        except Exception as e:
            # A plugin whose health_check() itself raises is treated the
            # same as an unhealthy engine, not a registry crash.
            status = PluginHealthStatus(healthy=False, message=f"health_check() raised: {e}")

        rec.last_health = status
        rec.last_health_check_ts = now
        return status

    # ------------------------------------------------------------------ #
    #  Safe dispatch                                                       #
    # ------------------------------------------------------------------ #

    def safe_infer(
        self,
        engine_id: str,
        model_id: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ):
        """
        Dispatches to the named plugin's infer(), with quarantine
        protection: if the plugin raises (contract violation), it's
        quarantined and a structured failure is returned instead of
        propagating the exception up into the router/API layer.
        """
        from plugin_base import PluginInferenceResult

        rec = self._records.get(engine_id)
        if rec is None:
            return PluginInferenceResult(
                success=False, error=f"no plugin registered for engine '{engine_id}'",
            )

        if rec.quarantined:
            return PluginInferenceResult(
                success=False,
                error=f"plugin '{engine_id}' is quarantined: {rec.quarantine_reason}",
            )

        rec.call_count += 1
        try:
            result = rec.plugin.infer(
                model_id=model_id, prompt=prompt, system_prompt=system_prompt,
                max_tokens=max_tokens, temperature=temperature,
            )
            if not result.success:
                rec.error_count += 1
            return result
        except Exception as e:
            # Contract violation: infer() must never raise. Quarantine
            # the plugin so this failure mode can't recur for the rest
            # of the process, then report it as a normal failed result
            # rather than crashing the caller.
            rec.error_count += 1
            rec.quarantined = True
            rec.quarantine_reason = f"infer() raised uncaught exception: {e}"
            tb = traceback.format_exc(limit=3)
            print(f"[plugin-registry] QUARANTINED '{engine_id}': {e}\n{tb}")
            return PluginInferenceResult(
                success=False,
                error=f"plugin '{engine_id}' crashed and was quarantined: {e}",
            )

    def reset_quarantine(self, engine_id: str) -> bool:
        """Manually clear a quarantine — e.g. after fixing/restarting the underlying engine."""
        rec = self._records.get(engine_id)
        if rec is None:
            return False
        rec.quarantined = False
        rec.quarantine_reason = ""
        return True


# ═══════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    registry = PluginRegistry()
    loaded = registry.discover()
    print(f"\nDiscovered {len(loaded)} plugin(s): {loaded}\n")

    for engine_id in loaded:
        health = registry.check_health(engine_id, force=True)
        print(f"  {engine_id}: healthy={health.healthy} — {health.message}")

    print("\n── Full engine list ──")
    for e in registry.list_engines():
        print(f"  {e}")
