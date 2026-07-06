"""
Nomad Runtime — Mode Switcher
Decides the best inference mode based on a DeviceSnapshot.

Modes (in descending capability order):
  STALLION  — cloud API, best quality, requires internet
  WORKHORSE — medium local model (~3–4 GB RAM), balanced
  NOMAD     — tiny local model (~0.8 GB RAM), offline-first
  GUARDIAN  — emergency fallback, minimal processing

Hysteresis prevents flickering between modes.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from device_monitor import DeviceSnapshot


class Mode(str, Enum):
    STALLION  = "stallion"
    WORKHORSE = "workhorse"
    NOMAD     = "nomad"
    GUARDIAN  = "guardian"


@dataclass
class ModeDecision:
    mode: Mode
    reason: str
    score: int          # 0–100 health score of the decision
    timestamp: float
    snap_summary: dict  # key metrics at decision time


class Thresholds:
    """
    All thresholds in one place — easy to tune for the ADTC laptop.
    Budget is 7 GB RAM total; we leave ~1 GB headroom for the OS.
    """
    # RAM limits (GB)
    RAM_GUARDIAN_TRIGGER  = 6.8   # above this → GUARDIAN (almost out)
    RAM_WORKHORSE_MIN     = 3.5   # need this free to run Workhorse safely
    RAM_NOMAD_MIN         = 1.0   # need this free to run even Nomad

    # CPU limits (%)
    CPU_HIGH              = 85    # above this → drop a tier
    CPU_CRITICAL          = 95    # above this → GUARDIAN

    # Temperature (°C)
    TEMP_WARN             = 80    # above this → drop a tier
    TEMP_THROTTLE         = 85    # ADTC thermal penalty threshold

    # Battery (%)
    BAT_LOW               = 20    # prefer lower-power mode
    BAT_CRITICAL          = 10    # force NOMAD

    # Hysteresis — minimum seconds between mode changes
    HYSTERESIS_UP_S       = 30    # wait before upgrading mode
    HYSTERESIS_DOWN_S     = 10    # downgrade faster (safety)


class ModeSwitcher:
    """
    Stateful mode selector with hysteresis.

    Usage:
        switcher = ModeSwitcher()
        decision = switcher.decide(snapshot)
        print(decision.mode)
    """

    def __init__(self, capability_report=None):
        """
        capability_report: Optional CapabilityReport from CapabilityDetector.
        When provided, mode decisions are hard-capped at the highest tier
        the hardware can physically run — e.g. a device whose one-time scan
        showed it can't run Workhorse will never be offered Workhorse here,
        no matter how favourable the live RAM/CPU snapshot looks. This is
        deliberately a ceiling, not a floor: live conditions can still push
        the decision *down* from the capability ceiling (e.g. RAM is
        temporarily low), just never *above* it.
        """
        self._current_mode: Mode = Mode.NOMAD      # safe default
        self._last_change_ts: float = 0.0
        self._last_proposed: Optional[Mode] = None
        self._proposed_since: float = 0.0
        self.history: list[ModeDecision] = []
        self._capability_ceiling = self._compute_ceiling(capability_report)

    @staticmethod
    def _compute_ceiling(capability_report) -> Optional[Mode]:
        """Highest local mode the hardware can run, per capability detection."""
        if capability_report is None:
            return None
        viable_local = {
            t.mode for t in capability_report.tiers
            if t.viable and t.mode in (Mode.NOMAD.value, Mode.WORKHORSE.value)
        }
        if Mode.WORKHORSE.value in viable_local:
            return Mode.WORKHORSE
        if Mode.NOMAD.value in viable_local:
            return Mode.NOMAD
        return Mode.GUARDIAN   # hardware can't reliably run any local model

    def set_capability_ceiling(self, capability_report) -> None:
        """Update the ceiling after the fact — e.g. once detection finishes."""
        self._capability_ceiling = self._compute_ceiling(capability_report)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def current_mode(self) -> Mode:
        return self._current_mode

    def decide(self, snap: DeviceSnapshot) -> ModeDecision:
        """
        Evaluate the snapshot and return a ModeDecision.
        Applies hysteresis: won't switch until the proposed mode has been
        stable for the appropriate hold time.
        """
        proposed, reason, score = self._evaluate(snap)

        # Capability ceiling: never propose a mode the hardware can't
        # physically run, regardless of how good live conditions look.
        if self._capability_ceiling is not None:
            capped, ceiling_reason = self._apply_ceiling(proposed)
            if capped != proposed:
                proposed = capped
                reason   = ceiling_reason

        now = time.time()

        # --- Hysteresis logic ---
        if proposed != self._last_proposed:
            # New proposal — start the timer
            self._last_proposed = proposed
            self._proposed_since = now

        hold = self._hold_time(proposed)
        time_in_proposal = now - self._proposed_since

        if proposed != self._current_mode and time_in_proposal >= hold:
            # Commit the switch
            self._current_mode = proposed
            self._last_change_ts = now

        decision = ModeDecision(
            mode         = self._current_mode,
            reason       = reason,
            score        = score,
            timestamp    = now,
            snap_summary = self._summarize(snap),
        )
        self.history.append(decision)
        return decision

    def force_mode(self, mode: Mode, reason: str = "manual override") -> ModeDecision:
        """Bypass hysteresis — for manual control via the API."""
        self._current_mode   = mode
        self._last_proposed  = mode
        self._last_change_ts = time.time()
        decision = ModeDecision(
            mode         = mode,
            reason       = reason,
            score        = 100,
            timestamp    = time.time(),
            snap_summary = {},
        )
        self.history.append(decision)
        return decision

    # ------------------------------------------------------------------ #
    #  Core evaluation logic                                               #
    # ------------------------------------------------------------------ #

    def _evaluate(self, snap: DeviceSnapshot) -> tuple[Mode, str, int]:
        """
        Returns (proposed_mode, reason_string, health_score).
        Evaluated top-down: worst conditions win.
        """
        t  = Thresholds
        reasons = []
        score   = 100

        # ── 1. Guardian triggers ────────────────────────────────────────
        if snap.ram_free_gb < (snap.ram_total_gb - t.RAM_GUARDIAN_TRIGGER):
            return Mode.GUARDIAN, "RAM critically low (<0.2 GB free)", 5

        if snap.cpu_percent >= t.CPU_CRITICAL:
            return Mode.GUARDIAN, f"CPU critical: {snap.cpu_percent:.0f}%", 10

        if snap.cpu_temp_c and snap.cpu_temp_c >= t.TEMP_THROTTLE:
            return Mode.GUARDIAN, f"Temperature throttle: {snap.cpu_temp_c}°C", 10

        # ── 2. Battery critical ─────────────────────────────────────────
        if snap.battery_percent is not None and snap.battery_percent <= t.BAT_CRITICAL:
            if not snap.battery_plugged:
                return Mode.NOMAD, f"Battery critical: {snap.battery_percent}%", 20

        # ── 3. No network → can't use Stallion ─────────────────────────
        if not snap.network_online:
            reasons.append("offline")
            score -= 10

        # ── 4. RAM check ────────────────────────────────────────────────
        if snap.ram_free_gb < t.RAM_NOMAD_MIN:
            return Mode.GUARDIAN, f"Insufficient RAM for any model ({snap.ram_free_gb:.1f} GB free)", 5

        ram_ok_for_workhorse = snap.ram_free_gb >= t.RAM_WORKHORSE_MIN

        # ── 5. CPU pressure ─────────────────────────────────────────────
        cpu_high = snap.cpu_percent >= t.CPU_HIGH
        if cpu_high:
            reasons.append(f"CPU high ({snap.cpu_percent:.0f}%)")
            score -= 20

        # ── 6. Temperature warning ──────────────────────────────────────
        temp_warn = snap.cpu_temp_c and snap.cpu_temp_c >= t.TEMP_WARN
        if temp_warn:
            reasons.append(f"temp warn ({snap.cpu_temp_c}°C)")
            score -= 15

        # ── 7. Battery low ──────────────────────────────────────────────
        bat_low = (
            snap.battery_percent is not None
            and snap.battery_percent <= t.BAT_LOW
            and not snap.battery_plugged
        )
        if bat_low:
            reasons.append(f"battery low ({snap.battery_percent}%)")
            score -= 10

        # ── 8. Mode selection ───────────────────────────────────────────

        # STALLION: online, RAM ok, CPU ok, temp ok
        if (
            snap.network_online
            and snap.network_speed == "fast"
            and ram_ok_for_workhorse
            and not cpu_high
            and not temp_warn
            and not bat_low
        ):
            return Mode.STALLION, "All systems nominal, fast internet", min(score, 100)

        # WORKHORSE: RAM ok but degraded conditions or slow/no internet
        if ram_ok_for_workhorse and not cpu_high and not temp_warn:
            reason = "Workhorse: " + (
                ", ".join(reasons) if reasons else "slow internet or mild constraints"
            )
            return Mode.WORKHORSE, reason, max(score, 40)

        # NOMAD: last local option
        reason = "Nomad: " + (", ".join(reasons) if reasons else "constrained device")
        return Mode.NOMAD, reason, max(score, 20)

    def _hold_time(self, proposed: Mode) -> float:
        """Upgrade is slower than downgrade."""
        if self._mode_rank(proposed) > self._mode_rank(self._current_mode):
            return Thresholds.HYSTERESIS_UP_S    # upgrading — be cautious
        return Thresholds.HYSTERESIS_DOWN_S      # downgrading — be fast

    def _apply_ceiling(self, proposed: Mode) -> tuple[Mode, str]:
        """
        Caps `proposed` at self._capability_ceiling if it exceeds it.
        Stallion is never capped here — capability detection only governs
        local-model viability; Stallion's gate is live network status,
        handled separately by _evaluate().
        """
        ceiling = self._capability_ceiling
        if proposed == Mode.STALLION or ceiling is None:
            return proposed, ""
        if self._mode_rank(proposed) > self._mode_rank(ceiling):
            return ceiling, (
                f"Capped at {ceiling.value.upper()} — capability scan found this "
                f"device cannot reliably run {proposed.value.upper()}"
            )
        return proposed, ""

    @staticmethod
    def _mode_rank(mode: Mode) -> int:
        return {Mode.GUARDIAN: 0, Mode.NOMAD: 1, Mode.WORKHORSE: 2, Mode.STALLION: 3}[mode]

    @staticmethod
    def _summarize(snap: DeviceSnapshot) -> dict:
        return {
            "cpu":     f"{snap.cpu_percent:.0f}%",
            "ram_free": f"{snap.ram_free_gb:.1f} GB",
            "temp":    f"{snap.cpu_temp_c}°C" if snap.cpu_temp_c else "n/a",
            "net":     snap.network_speed if snap.network_online else "offline",
            "battery": f"{snap.battery_percent}%" if snap.battery_percent else "n/a",
        }


# ------------------------------------------------------------------ #
#  Quick self-test                                                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    from device_monitor import DeviceMonitor

    monitor  = DeviceMonitor()
    switcher = ModeSwitcher()

    print("=== Mode Decision ===")
    snap     = monitor.snapshot()
    decision = switcher.decide(snap)

    print(f"  Mode   : {decision.mode.value.upper()}")
    print(f"  Reason : {decision.reason}")
    print(f"  Score  : {decision.score}/100")
    print(f"  Device : {decision.snap_summary}")
