"""
Nomad Runtime — ADTC Simulation Test Runner
═══════════════════════════════════════════════════════════════════════
Runs the full hardware-emulated benchmark against the ADTC Standard
Laptop profile and prints a judge-ready report with all scored results.

Usage:
    python adtc_test_runner.py                      # default: i5 11th gen
    python adtc_test_runner.py --cpu i5_12th         # specify variant
    python adtc_test_runner.py --all-variants        # compare all CPUs
    python adtc_test_runner.py --json report.json    # save JSON output
═══════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adtc_emulator import (
    ADTCHardwareEmulator, ADTCSimulationReport, ModeResult,
    TEST_PROMPTS, MODEL_PROFILES,
    ADTC_TPS_REFERENCE, ADTC_RAM_BUDGET_GB, ADTC_THERMAL_LIMIT_C,
)

# ── Terminal colour helpers (graceful fallback if no ANSI) ────────────────
try:
    import os
    _COLOUR = os.isatty(sys.stdout.fileno())
except Exception:
    _COLOUR = False

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text

def green(t):  return _c("92", t)
def yellow(t): return _c("93", t)
def cyan(t):   return _c("96", t)
def bold(t):   return _c("1",  t)
def red(t):    return _c("91", t)
def dim(t):    return _c("2",  t)


# ═══════════════════════════════════════════════════════════════════════════
#  Report renderer
# ═══════════════════════════════════════════════════════════════════════════

def render_report(report: ADTCSimulationReport) -> str:
    lines = []
    W = 72   # line width

    def rule(char="═"):
        return char * W

    def header(text):
        pad = (W - len(text) - 2) // 2
        return f"{'═'*pad} {text} {'═'*(W - pad - len(text) - 2)}"

    def row(label, value, unit="", highlight=False):
        label_w = 32
        value_s = f"{value}{(' '+unit) if unit else ''}"
        line = f"  {label:<{label_w}} {value_s}"
        return green(line) if highlight else line

    # ── Header ─────────────────────────────────────────────────────────
    lines += [
        "",
        bold(rule()),
        bold(header("NOMAD RUNTIME — ADTC SIMULATION REPORT")),
        bold(rule()),
        "",
        bold("  Hardware Profile"),
        rule("─"),
    ]
    hw = report.hardware_profile
    lines += [
        row("CPU",           hw["cpu"]),
        row("Cores",         hw["cpu_cores"]),
        row("RAM (total)",   f"{hw['ram_total_gb']} GB DDR4"),
        row("RAM (OS base)", f"{hw['ram_os_gb']} GB  (Ubuntu 22.04 LTS idle)"),
        row("AVX2",          "✓ Yes"),
        row("GPU",           hw["gpu"]),
        row("Storage",       hw["storage"]),
        row("OS",            hw["os"]),
        row("Benchmark seed","42  (reproducible)"),
        "",
    ]

    # ── Per-mode results ───────────────────────────────────────────────
    lines += [
        bold("  Mode Results  —  all four Nomad Runtime modes"),
        rule("─"),
    ]

    mode_order = ["guardian", "nomad", "workhorse", "stallion"]
    mode_emoji = {
        "guardian":  "🛡 ",
        "nomad":     "📡",
        "workhorse": "⚙ ",
        "stallion":  "🚀",
    }

    for mode_key in mode_order:
        if mode_key not in report.modes:
            continue
        m: ModeResult = report.modes[mode_key]
        is_winner = (mode_key == report.winner)
        tag = green("  ★ BEST LOCAL") if is_winner else ""

        lines.append("")
        lines.append(bold(f"  {mode_emoji[mode_key]} {m.mode.upper()}  —  {m.model_name}{tag}"))
        lines.append(f"  {dim('Engine: ' + m.engine)}")
        lines.append("")

        # Performance table
        lines.append(dim("  ┌─ Performance ─────────────────────────────────"))
        lines.append(row("  │  Avg TPS", f"{m.avg_tps:.1f}", f"tok/s  (target: {ADTC_TPS_REFERENCE})"))
        lines.append(row("  │  Latency p50 / p95", f"{m.p50_latency_ms:.0f} ms / {m.p95_latency_ms:.0f} ms"))
        lines.append(row("  │  Peak RAM used", f"{m.peak_ram_gb:.2f} GB  (budget: {ADTC_RAM_BUDGET_GB} GB)"))
        lines.append(row("  │  Avg CPU load", f"{m.avg_cpu_pct:.0f}%"))
        lines.append(row("  │  Peak temperature",
            f"{m.peak_temp_c:.1f}°C  {'⚠ THROTTLED' if m.thermal_throttled else '✓ OK'}  (limit: {ADTC_THERMAL_LIMIT_C}°C)"))
        lines.append(row("  │  Offline capable",
            f"{'✓ YES' if m.offline_capable else '✗ NO (cloud)'}"))

        # Score table
        lines.append(dim("  ├─ ADTC Scores ─────────────────────────────────"))
        lines.append(row("  │  Sacc  (50%)",
            f"{m.sacc:.1f}/100", "← judge accuracy"))
        lines.append(row("  │  Sperf (30%)",
            f"{m.sperf:.1f}/100",
            f"← {m.avg_tps:.1f} TPS / {ADTC_TPS_REFERENCE} ref"))
        lines.append(row("  │  Seff  (20%)",
            f"{m.seff:.1f}/100",
            f"← {m.peak_ram_gb:.2f} GB peak RAM"))
        thermal_str = f"-{m.pthermal} pts" if m.pthermal else "0 (no penalty)"
        lines.append(row("  │  Pthermal penalty",
            thermal_str,
            "← throttle / temp > 85°C"))
        lines.append(dim("  ├────────────────────────────────────────────────"))

        score_str = f"{m.stotal:.1f}/100"
        score_line = f"  {'│':1s}  {'STOTAL':32s} {score_str}"
        lines.append(bold(green(score_line) if is_winner else bold(score_line)))
        lines.append(dim("  └────────────────────────────────────────────────"))

        # Per-prompt breakdown
        lines.append(f"  {dim('Per-prompt detail:')}")
        lines.append(
            f"  {'Prompt':14s} {'TPS':>7s} {'Lat ms':>8s} {'Temp°C':>7s} "
            f"{'RAM GB':>7s} {'Quality':>8s}"
        )
        lines.append(f"  {'─'*14} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*8}")
        for r in m.runs:
            lines.append(
                f"  {r.prompt_label:14s} {r.tps:>7.1f} {r.latency_ms:>8.0f} "
                f"{r.temp_c:>7.1f} {r.ram_used_gb:>7.2f} {r.quality_score:>8.1f}"
            )

    # ── Score comparison table ─────────────────────────────────────────
    lines += [
        "",
        bold(rule()),
        bold(header("SCORE COMPARISON")),
        bold(rule()),
        "",
        f"  {'Mode':12s} {'Sacc':>7s} {'Sperf':>7s} {'Seff':>7s} {'Pthermal':>10s} {'STOTAL':>8s} {'Offline':>8s}",
        f"  {'─'*12} {'─'*7} {'─'*7} {'─'*7} {'─'*10} {'─'*8} {'─'*8}",
    ]
    for mode_key in mode_order:
        if mode_key not in report.modes:
            continue
        m = report.modes[mode_key]
        is_winner = mode_key == report.winner
        penalty_str = f"-{m.pthermal}" if m.pthermal else "  0"
        offline_str = "✓" if m.offline_capable else "✗ (net)"
        row_str = (
            f"  {m.mode:12s} {m.sacc:>7.1f} {m.sperf:>7.1f} {m.seff:>7.1f} "
            f"{penalty_str:>10s} {m.stotal:>8.1f} {offline_str:>8s}"
        )
        lines.append(bold(green(row_str)) if is_winner else row_str)

    lines.append("")

    # Formula reminder
    lines += [
        dim("  STOTAL = 0.50·Sacc + 0.30·Sperf + 0.20·Seff − Pthermal"),
        dim(f"  Sperf  = 100 × (TPS / {ADTC_TPS_REFERENCE})"),
        dim(f"  Seff   = 100 × (({ADTC_RAM_BUDGET_GB} GB − Peak RAM) / {ADTC_RAM_BUDGET_GB} GB)"),
        "",
    ]

    # ── Adaptive simulation ────────────────────────────────────────────
    if report.adaptive_simulation:
        adap = report.adaptive_simulation
        lines += [
            bold(rule()),
            bold(header("ADAPTIVE MODE-SWITCHING SIMULATION")),
            bold(rule()),
            "",
            f"  Session duration : {adap['duration_minutes']} minutes",
            f"  Mode transitions : {adap['total_mode_changes']}",
            f"  Summary          : {adap['summary']}",
            "",
            f"  {'T (min)':>7s}  {'Mode':12s}  {'Scenario'}",
            f"  {'─'*7}  {'─'*12}  {'─'*44}",
        ]
        for event in adap["timeline"]:
            lines.append(
                f"  {event['t_min']:>7.0f}  {event['mode_chosen']:12s}  {event['scenario']}"
            )
            lines.append(
                f"  {'':>7s}  {dim(event['reason'])}"
            )
            lines.append("")

    # ── Recommendation ─────────────────────────────────────────────────
    winner_result = report.modes.get(report.winner)
    lines += [
        bold(rule()),
        bold(header("RECOMMENDATION")),
        bold(rule()),
        "",
    ]
    if winner_result:
        lines += [
            green(bold(f"  Best local mode:  {report.winner.upper()}")),
            green(bold(f"  Model:            {winner_result.model_name}")),
            green(bold(f"  Score:            {winner_result.stotal:.1f}/100")),
            "",
            f"  {report.winner.upper()} achieves the best balance of accuracy, speed,",
            f"  and RAM efficiency on the ADTC Standard Laptop (8 GB, no GPU).",
            "",
            f"  • Sacc  {winner_result.sacc:.1f}/100  — quality sufficient for automotive diagnosis",
            f"  • Sperf {winner_result.sperf:.1f}/100  — {winner_result.avg_tps:.1f} TPS vs {ADTC_TPS_REFERENCE} target",
            f"  • Seff  {winner_result.seff:.1f}/100  — {winner_result.peak_ram_gb:.2f} GB peak, {ADTC_RAM_BUDGET_GB - winner_result.peak_ram_gb:.2f} GB headroom",
            f"  • Thermal: {'⚠ throttled' if winner_result.thermal_throttled else '✓ below 85°C, no penalty'}",
            "",
            "  Nomad Runtime adapts the mode at runtime based on live device",
            "  conditions and learned device/task profiles, always staying within",
            "  the hardware capability ceiling detected on first boot.",
        ]

    lines.append("")
    lines.append(bold(rule()))
    lines.append("")

    return "\n".join(lines)


def render_variant_comparison(all_results: dict[str, ADTCSimulationReport]) -> str:
    """Print a side-by-side comparison of all CPU variants."""
    lines = [
        "",
        bold("═" * 72),
        bold("  CPU VARIANT COMPARISON  (all ADTC-class CPUs)"),
        bold("═" * 72),
        "",
        f"  {'CPU Variant':20s}  {'Best Mode':10s}  {'STOTAL':>8s}  {'Avg TPS':>8s}  {'Peak RAM':>9s}",
        f"  {'─'*20}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*9}",
    ]
    for variant, report in sorted(all_results.items()):
        best = report.modes.get(report.winner)
        if best:
            lines.append(
                f"  {variant:20s}  {report.winner:10s}  {best.stotal:>8.1f}  "
                f"{best.avg_tps:>8.1f}  {best.peak_ram_gb:>9.2f} GB"
            )
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Nomad Runtime ADTC Simulation Test Runner"
    )
    parser.add_argument(
        "--cpu", default="i5_11th",
        choices=["i5_10th", "i5_11th", "i5_12th", "ryzen5_3600", "ryzen5_5600"],
        help="CPU variant to simulate (default: i5_11th)",
    )
    parser.add_argument(
        "--all-variants", action="store_true",
        help="Run and compare all CPU variants",
    )
    parser.add_argument(
        "--json", metavar="FILE",
        help="Also save full results to a JSON file",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    t0 = time.time()

    if args.all_variants:
        variants = ["i5_10th", "i5_11th", "i5_12th", "ryzen5_3600", "ryzen5_5600"]
        all_results = {}
        for variant in variants:
            emu = ADTCHardwareEmulator(seed=args.seed, cpu_variant=variant)
            all_results[variant] = emu.run_full_simulation()

        # Print main report for reference variant
        main_report = all_results["i5_11th"]
        print(render_report(main_report))
        print(render_variant_comparison(all_results))

        if args.json:
            out = {v: r.to_dict() for v, r in all_results.items()}
            Path(args.json).write_text(json.dumps(out, indent=2))
            print(f"  Results saved → {args.json}")
    else:
        emu    = ADTCHardwareEmulator(seed=args.seed, cpu_variant=args.cpu)
        report = emu.run_full_simulation()
        print(render_report(report))

        if args.json:
            Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
            print(f"  Results saved → {args.json}")

    elapsed = time.time() - t0
    print(dim(f"  Simulation completed in {elapsed:.2f}s\n"))


if __name__ == "__main__":
    main()
