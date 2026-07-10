"""
Nomad Runtime — Benchmarker
Measures what the ADTC judges care about:
  • RAM usage (peak and baseline)
  • Latency (time-to-first-token approximation)
  • Throughput (tokens per second)
  • Model size on disk
  • Offline capability score
  • ADTC scoring formula projection
"""

import os
import sys
import time
import json
import psutil
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional

from mode_switcher import Mode
from inference_router import InferenceRouter


# ── ADTC scoring constants ────────────────────────────────────────────────
TPS_REFERENCE    = 15.0   # ADTC target (provisional)
RAM_BUDGET_GB    = 7.0    # ADTC Seff budget
THERMAL_PENALTY  = 10     # points deducted if throttled


TEST_PROMPTS = [
    ("short",    "What does a knocking sound from the engine usually indicate?"),
    ("medium",   "My car shows OBD code P0301. What does it mean and how serious is it?"),
    ("long",     (
        "My 2014 Toyota Etios has been running rough for two days. "
        "The check engine light is on. OBD scan shows P0301 (cylinder 1 misfire) "
        "and P0171 (system too lean). Coolant temp is normal. "
        "What are the most likely causes and what should I check first?"
    )),
]

AUTEX_SYSTEM_PROMPT = (
    "You are Autex, an AI automotive diagnostic assistant. "
    "Answer clearly, in 2–4 sentences unless a longer explanation is needed. "
    "Prioritise safety."
)


@dataclass
class SingleRunResult:
    prompt_label:   str
    prompt_chars:   int
    output_chars:   int
    prompt_tokens:  int
    output_tokens:  int
    latency_ms:     float
    tps:            float      # tokens per second (output only)
    ram_before_gb:  float
    ram_peak_gb:    float
    error:          Optional[str] = None


@dataclass
class BenchmarkReport:
    mode:               str
    model:              str
    backend:            str
    runs:               list[SingleRunResult] = field(default_factory=list)
    # Aggregates
    avg_latency_ms:     float = 0.0
    median_latency_ms:  float = 0.0
    avg_tps:            float = 0.0
    peak_ram_gb:        float = 0.0
    # ADTC projections
    adtc_sperf:         float = 0.0   # 0–100
    adtc_seff:          float = 0.0   # 0–100
    adtc_total_est:     float = 0.0   # ignores Sacc (needs judge)
    offline_capable:    bool  = False
    thermal_ok:         bool  = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["runs"] = [asdict(r) for r in self.runs]
        return d

    def summary(self) -> str:
        lines = [
            f"┌─ Benchmark: {self.mode.upper()} ({self.model}) ─────────────",
            f"│  Backend          : {self.backend}",
            f"│  Avg latency      : {self.avg_latency_ms:.0f} ms",
            f"│  Median latency   : {self.median_latency_ms:.0f} ms",
            f"│  Avg TPS          : {self.avg_tps:.1f} tok/s  (ADTC target: {TPS_REFERENCE})",
            f"│  Peak RAM used    : {self.peak_ram_gb:.2f} GB  (budget: {RAM_BUDGET_GB} GB)",
            f"│  Offline capable  : {'✓ YES' if self.offline_capable else '✗ NO'}",
            f"│  Thermal OK       : {'✓ YES' if self.thermal_ok else '⚠ THROTTLED (-10 pts)'}",
            f"├─ ADTC score projection ─────────────────────────────",
            f"│  Sperf (30%)      : {self.adtc_sperf:.1f}/100",
            f"│  Seff  (20%)      : {self.adtc_seff:.1f}/100",
            f"│  Total est.*      : {self.adtc_total_est:.1f}  (* Sacc excluded — needs judge)",
            f"└─────────────────────────────────────────────────────",
        ]
        for r in self.runs:
            status = "✓" if not r.error else "✗"
            lines.append(
                f"   {status} [{r.prompt_label:6s}] {r.latency_ms:6.0f}ms  "
                f"{r.tps:5.1f} TPS  RAM Δ: {r.ram_peak_gb - r.ram_before_gb:+.2f} GB"
                + (f"  ERROR: {r.error}" if r.error else "")
            )
        return "\n".join(lines)


class Benchmarker:
    """
    Run the test suite against a given mode and produce a BenchmarkReport.
    """

    def __init__(self):
        self.router = InferenceRouter()

    def run(
        self,
        mode: Mode,
        prompts: list[tuple[str, str]] = TEST_PROMPTS,
        system_prompt: str = AUTEX_SYSTEM_PROMPT,
        check_thermal: bool = True,
    ) -> BenchmarkReport:
        cfg = self.router.MODEL_REGISTRY if hasattr(self.router, "MODEL_REGISTRY") else {}

        from inference_router import MODEL_REGISTRY
        model_cfg = MODEL_REGISTRY[mode]

        report = BenchmarkReport(
            mode            = mode.value,
            model           = model_cfg["model"] or "none",
            backend         = model_cfg["engine_id"],   # v2: engine_id replaced backend
            offline_capable = mode in (Mode.NOMAD, Mode.WORKHORSE, Mode.GUARDIAN),
        )

        latencies = []
        tps_list  = []
        peak_rams = []

        for label, prompt in prompts:
            run = self._single_run(mode, label, prompt, system_prompt)
            report.runs.append(run)
            if not run.error:
                latencies.append(run.latency_ms)
                tps_list.append(run.tps)
                peak_rams.append(run.ram_peak_gb)

        if latencies:
            report.avg_latency_ms    = statistics.mean(latencies)
            report.median_latency_ms = statistics.median(latencies)
            report.avg_tps           = statistics.mean(tps_list)
            report.peak_ram_gb       = max(peak_rams) if peak_rams else 0.0

        # ADTC projections
        report.adtc_sperf = min(100.0, 100 * (report.avg_tps / TPS_REFERENCE))
        report.adtc_seff  = max(0.0, 100 * ((RAM_BUDGET_GB - report.peak_ram_gb) / RAM_BUDGET_GB))

        # Thermal check
        if check_thermal:
            report.thermal_ok = self._thermal_ok()

        thermal_pen = 0 if report.thermal_ok else THERMAL_PENALTY
        # Sacc = 0 (unknown) → total = 0.30*Sperf + 0.20*Seff − Pthermal
        report.adtc_total_est = (
            0.30 * report.adtc_sperf
            + 0.20 * report.adtc_seff
            - thermal_pen
        )

        return report

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _single_run(
        self,
        mode: Mode,
        label: str,
        prompt: str,
        system_prompt: str,
    ) -> SingleRunResult:
        proc       = psutil.Process(os.getpid())
        ram_before = psutil.virtual_memory().used / 1024**3

        result = self.router.infer(
            mode          = mode,
            prompt        = prompt,
            system_prompt = system_prompt,
        )

        ram_peak = psutil.virtual_memory().used / 1024**3

        # Approximate TPS from character count if token count missing
        out_tokens = result.output_tokens or max(1, len(result.response_text.split()))
        tps = (out_tokens / (result.latency_ms / 1000)) if result.latency_ms > 0 else 0

        return SingleRunResult(
            prompt_label  = label,
            prompt_chars  = len(prompt),
            output_chars  = len(result.response_text),
            prompt_tokens = result.prompt_tokens,
            output_tokens = out_tokens,
            latency_ms    = result.latency_ms,
            tps           = tps,
            ram_before_gb = round(ram_before, 2),
            ram_peak_gb   = round(ram_peak, 2),
            error         = result.error,
        )

    @staticmethod
    def _thermal_ok() -> bool:
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return True   # can't measure → assume OK
            for readings in temps.values():
                for t in readings:
                    if t.current >= 85:
                        return False
            return True
        except Exception:
            return True


# ------------------------------------------------------------------ #
#  CLI runner                                                         #
# ------------------------------------------------------------------ #

def run_all_benchmarks(modes: list[Mode] = None):
    if modes is None:
        modes = [Mode.GUARDIAN, Mode.NOMAD]   # safe defaults (no model needed for Guardian)

    bench = Benchmarker()
    results = {}

    for mode in modes:
        print(f"\nRunning benchmark: {mode.value.upper()} ...")
        report = bench.run(mode)
        results[mode.value] = report
        print(report.summary())

    # Save JSON
    out_path = "/home/claude/nomad-runtime/docs/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved → {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nomad Runtime Benchmarker")
    parser.add_argument(
        "--modes", nargs="+",
        choices=["guardian", "nomad", "workhorse", "stallion"],
        default=["guardian"],
        help="Modes to benchmark (guardian always works, others need Ollama)"
    )
    args = parser.parse_args()

    mode_map = {
        "guardian":  Mode.GUARDIAN,
        "nomad":     Mode.NOMAD,
        "workhorse": Mode.WORKHORSE,
        "stallion":  Mode.STALLION,
    }
    selected = [mode_map[m] for m in args.modes]
    run_all_benchmarks(selected)
