"""
Nomad Runtime — Adaptive Learner
═══════════════════════════════════════════════════════════════════════
Learns from device behaviour and task history to make smarter mode
decisions than the static rule engine alone.

Two learning subsystems:

1. DeviceProfiler
   Tracks RAM, CPU, and temperature across time windows.
   Detects persistent patterns (e.g. "this machine is always memory-
   constrained") and produces a DeviceProfile that biases mode selection.

2. TaskClassifier
   Learns which task types (OBD, sound, maintenance, code, medical…)
   historically needed higher-quality responses and which were fine with
   a small model.  Produces a quality-demand signal that can upgrade the
   mode the rule engine would otherwise choose.

3. AdaptiveLearner  (orchestrator)
   Combines both signals with the static rule-engine decision.
   Persists knowledge to a JSON file so learning survives restarts.
   Exposes explain() to make the reasoning transparent for the dashboard.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from mode_switcher import Mode


# ── Storage path ──────────────────────────────────────────────────────────
DEFAULT_DB = Path(os.getenv("NOMAD_LEARN_DB", "/tmp/nomad_adaptive.json"))


# ═══════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeviceProfile:
    """Summarises what the device is typically like."""
    sessions_seen:    int   = 0
    ram_chronic_low:  bool  = False   # RAM free routinely < 3 GB
    cpu_chronic_high: bool  = False   # CPU routinely > 70 %
    temp_chronic_hot: bool  = False   # temp routinely > 78 °C
    ram_mean_free_gb: float = 4.0
    cpu_mean_pct:     float = 20.0
    temp_mean_c:      float = 50.0
    confidence:       float = 0.0     # 0–1; grows with sessions_seen


@dataclass
class TaskProfile:
    """What we've learned about a task category."""
    name:             str
    count:            int   = 0
    quality_needed:   float = 0.5    # 0–1; high = needs big model
    avg_satisfaction: float = 0.5    # proxy: did the response have enough tokens?
    last_mode:        str   = "nomad"


@dataclass
class LearnerDecision:
    """The adaptive learner's recommendation (not necessarily final mode)."""
    recommended_mode: Mode
    rule_mode:        Mode          # what the static rules said
    device_bias:      str           # "upgrade" | "downgrade" | "neutral"
    task_bias:        str           # "upgrade" | "downgrade" | "neutral"
    confidence:       float         # 0–1
    explanation:      str           # human-readable reasoning


# ═══════════════════════════════════════════════════════════════════════════
#  Device Profiler
# ═══════════════════════════════════════════════════════════════════════════

class DeviceProfiler:
    """
    Maintains a rolling window of hardware snapshots and detects
    persistent device patterns.

    Algorithm
    ---------
    Keeps the last MAX_WINDOW readings.  Every LEARN_EVERY readings it
    recomputes rolling means and updates chronic-condition flags using
    an exponential moving average (α = 0.15) so the profile adapts
    gradually rather than flipping on every anomaly.
    """

    MAX_WINDOW  = 500    # readings kept in memory
    LEARN_EVERY = 20     # recompute profile every N readings
    EMA_ALPHA   = 0.15   # EMA weight for new observations

    # Thresholds for "chronic" classification
    CHRONIC_RAM_FREE_GB  = 3.0   # below this → chronic low
    CHRONIC_CPU_PCT      = 70.0  # above this → chronic high
    CHRONIC_TEMP_C       = 78.0  # above this → chronic hot

    # Minimum raw samples before trusting the profile at all
    MIN_SAMPLES_FOR_BIAS = 30

    def __init__(self, profile: Optional[DeviceProfile] = None, fast_demo: bool = False):
        """
        fast_demo: if True, lowers LEARN_EVERY and MIN_SAMPLES_FOR_BIAS so the
        profile becomes confident within ~1 minute (5s poll interval) instead
        of several minutes. Intended for live demos / judging sessions, not
        production deployments where slower, steadier learning is preferable.
        """
        self._profile  = profile or DeviceProfile()
        self._window: deque[dict] = deque(maxlen=self.MAX_WINDOW)
        self._tick     = 0
        if fast_demo:
            self.LEARN_EVERY          = 4
            self.MIN_SAMPLES_FOR_BIAS = 8

    @property
    def profile(self) -> DeviceProfile:
        return self._profile

    def record(self, snap_dict: dict) -> None:
        """Feed a raw device snapshot (as dict from DeviceSnapshot.to_dict())."""
        ram  = snap_dict.get("ram_free_gb", 4.0)
        cpu  = snap_dict.get("cpu_percent",  0.0)
        temp = snap_dict.get("cpu_temp_c") or 50.0

        self._window.append({"ram": ram, "cpu": cpu, "temp": temp, "ts": time.time()})
        self._tick += 1

        if self._tick % self.LEARN_EVERY == 0:
            self._recompute()

    def _recompute(self) -> None:
        # Floor must stay below LEARN_EVERY or fast_demo configurations
        # (LEARN_EVERY=4) would silently skip their first recompute cycle,
        # as a hardcoded floor of 5 used to do.
        min_window = min(self.LEARN_EVERY, 4)
        if len(self._window) < min_window:
            return

        rams  = [r["ram"]  for r in self._window]
        cpus  = [r["cpu"]  for r in self._window]
        temps = [r["temp"] for r in self._window]

        mean_ram  = sum(rams)  / len(rams)
        mean_cpu  = sum(cpus)  / len(cpus)
        mean_temp = sum(temps) / len(temps)

        a = self.EMA_ALPHA
        p = self._profile
        p.ram_mean_free_gb = a * mean_ram  + (1 - a) * p.ram_mean_free_gb
        p.cpu_mean_pct     = a * mean_cpu  + (1 - a) * p.cpu_mean_pct
        p.temp_mean_c      = a * mean_temp + (1 - a) * p.temp_mean_c

        p.ram_chronic_low  = p.ram_mean_free_gb < self.CHRONIC_RAM_FREE_GB
        p.cpu_chronic_high = p.cpu_mean_pct     > self.CHRONIC_CPU_PCT
        p.temp_chronic_hot = p.temp_mean_c      > self.CHRONIC_TEMP_C

        p.sessions_seen   += 1
        # Confidence blends "how many recompute cycles" with "how many raw
        # samples" so it rises smoothly even before 50 recompute cycles.
        cycle_conf  = min(1.0, p.sessions_seen / 50)
        sample_conf = min(1.0, len(self._window) / 100)
        p.confidence = max(cycle_conf, sample_conf * 0.6)

    def device_bias(self) -> tuple[str, str]:
        """
        Returns (bias, reason).
        bias: "downgrade" | "upgrade" | "neutral"
        """
        p = self._profile
        # Require a minimum amount of raw observations (not just recompute
        # cycles) before trusting the profile — avoids early false neutrality.
        if len(self._window) < self.MIN_SAMPLES_FOR_BIAS or p.confidence < 0.05:
            return "neutral", f"not enough history yet ({len(self._window)} samples)"

        problems = []
        if p.ram_chronic_low:
            problems.append(f"RAM typically only {p.ram_mean_free_gb:.1f} GB free")
        if p.cpu_chronic_high:
            problems.append(f"CPU typically at {p.cpu_mean_pct:.0f}%")
        if p.temp_chronic_hot:
            problems.append(f"temperature typically {p.temp_mean_c:.0f}°C")

        if len(problems) >= 2:
            return "downgrade", "chronic constraints: " + "; ".join(problems)
        if problems:
            return "downgrade", "mild chronic constraint: " + problems[0]
        return "neutral", "device is healthy"


# ═══════════════════════════════════════════════════════════════════════════
#  Task Classifier
# ═══════════════════════════════════════════════════════════════════════════

# Keyword → category mapping.  Order matters: first match wins.
_TASK_PATTERNS: list[tuple[str, list[str]]] = [
    ("obd_code",       [r"\bP[0-9]{4}\b", r"\bDTC\b", r"OBD", r"fault code"]),
    ("engine_sound",   [r"knock", r"squeal", r"tick", r"rattle", r"clunk",
                        r"noise", r"sound", r"vibrat"]),
    ("temperature",    [r"overheat", r"coolant", r"temp", r"radiator", r"thermostat"]),
    ("fuel_system",    [r"fuel", r"injector", r"trim", r"LTFT", r"STFT", r"lean", r"rich"]),
    ("electrical",     [r"battery", r"alternator", r"starter", r"fuse", r"relay",
                        r"voltage", r"electrical"]),
    ("maintenance",    [r"oil", r"filter", r"brake", r"tyre", r"tire", r"service",
                        r"schedule", r"replace", r"worn"]),
    ("complex_diag",   [r"multiple", r"several codes", r"intermittent",
                        r"rough idle", r"stall", r"won't start"]),
    ("general_query",  [r"what is", r"how does", r"explain", r"tell me"]),
]

# Prior quality-need per category (0–1).  Hand-tuned; learning adjusts these.
_CATEGORY_PRIORS: dict[str, float] = {
    "obd_code":      0.65,   # structured + specific → Workhorse helps
    "engine_sound":  0.70,   # nuanced pattern matching → more model helps
    "temperature":   0.55,
    "fuel_system":   0.60,
    "electrical":    0.55,
    "maintenance":   0.40,   # well-structured FAQ → Nomad fine
    "complex_diag":  0.85,   # multiple symptoms → Workhorse/Stallion
    "general_query": 0.30,   # simple → Nomad fine
    "unknown":       0.50,
}

# Minimum output tokens we consider a "satisfying" response per category
_MIN_TOKENS: dict[str, int] = {
    "obd_code":     60,
    "engine_sound": 80,
    "complex_diag": 120,
    "general_query": 30,
}


class TaskClassifier:
    """
    Classifies prompts into task categories and tracks quality-demand
    learned from response satisfaction signals.
    """

    EMA_ALPHA = 0.20   # how fast satisfaction updates quality_needed

    def __init__(self, task_profiles: Optional[dict[str, dict]] = None):
        self._profiles: dict[str, TaskProfile] = {}

        # Seed from priors
        for cat, prior in _CATEGORY_PRIORS.items():
            self._profiles[cat] = TaskProfile(name=cat, quality_needed=prior)

        # Restore persisted learning if available
        if task_profiles:
            for cat, data in task_profiles.items():
                if cat in self._profiles:
                    p = self._profiles[cat]
                    p.count            = data.get("count", 0)
                    p.quality_needed   = data.get("quality_needed", p.quality_needed)
                    p.avg_satisfaction = data.get("avg_satisfaction", 0.5)
                    p.last_mode        = data.get("last_mode", "nomad")

    def classify(self, prompt: str) -> str:
        """Return the task category for a prompt."""
        pl = prompt.lower()
        for cat, patterns in _TASK_PATTERNS:
            for pat in patterns:
                if re.search(pat, pl, re.IGNORECASE):
                    return cat
        return "unknown"

    def quality_demand(self, category: str) -> float:
        """Returns 0–1 quality-need for the category."""
        return self._profiles.get(category, self._profiles["unknown"]).quality_needed

    def task_bias(self, category: str) -> tuple[str, str]:
        """
        Returns (bias, reason).
        bias: "upgrade" | "downgrade" | "neutral"
        """
        qd = self.quality_demand(category)
        p  = self._profiles.get(category)
        count_note = f" ({p.count} samples)" if p and p.count > 0 else " (prior)"

        if qd >= 0.72:
            return "upgrade",   f"{category} tasks need high quality{count_note}"
        if qd <= 0.35:
            return "downgrade", f"{category} tasks work fine with small model{count_note}"
        return "neutral", f"{category} quality demand is moderate{count_note}"

    def record_outcome(
        self,
        category:      str,
        mode_used:     str,
        output_tokens: int,
        latency_ms:    float,
    ) -> None:
        """
        Update the quality-need estimate for a category based on the outcome.

        Satisfaction heuristic
        ----------------------
        A response is "satisfying" if it hits the minimum token count
        for the category.  Long responses → the task needed substance.
        Very fast responses with few tokens → probably fine with small model.
        """
        if category not in self._profiles:
            self._profiles[category] = TaskProfile(name=category,
                                                    quality_needed=0.5)
        p = self._profiles[category]
        p.count    += 1
        p.last_mode = mode_used

        min_tok = _MIN_TOKENS.get(category, 50)
        # Satisfaction: 0–1 based on whether response was long enough
        satisfaction = min(1.0, output_tokens / max(1, min_tok))
        a = self.EMA_ALPHA
        p.avg_satisfaction = a * satisfaction + (1 - a) * p.avg_satisfaction

        # If satisfaction is low and the model was small, raise quality_needed
        # If satisfaction is high and the model was small, keep or lower
        mode_rank = {"guardian": 0, "nomad": 1, "workhorse": 2, "stallion": 3}
        rank = mode_rank.get(mode_used, 1)

        if satisfaction < 0.6 and rank <= 1:
            # Small model wasn't enough → raise bar
            p.quality_needed = min(0.95, p.quality_needed + 0.05)
        elif satisfaction >= 0.8 and rank <= 1:
            # Small model was fine → lower bar slightly
            p.quality_needed = max(0.10, p.quality_needed - 0.02)
        elif satisfaction >= 0.8 and rank >= 2:
            # Big model + satisfied → could maybe use smaller next time
            p.quality_needed = max(0.10, p.quality_needed - 0.01)

    def to_dict(self) -> dict:
        return {cat: asdict(p) for cat, p in self._profiles.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  Adaptive Learner  (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveLearner:
    """
    Combines DeviceProfiler + TaskClassifier with the static rule engine
    to produce an adaptive mode recommendation.

    Persistence
    -----------
    Knowledge is saved to a JSON file on every record_outcome() call
    so learning survives process restarts.

    Mode upgrade/downgrade logic
    ----------------------------
    Rule mode → biases applied → final recommendation

    If device says downgrade AND task says upgrade → stay on rule mode
    If both agree → apply the shift
    One signal alone → apply with half weight (may or may not shift)

    Mode ranking: guardian(0) < nomad(1) < workhorse(2) < stallion(3)
    """

    _RANK = {Mode.GUARDIAN: 0, Mode.NOMAD: 1, Mode.WORKHORSE: 2, Mode.STALLION: 3}
    _BY_RANK = {0: Mode.GUARDIAN, 1: Mode.NOMAD, 2: Mode.WORKHORSE, 3: Mode.STALLION}

    def __init__(self, db_path: Path = DEFAULT_DB, fast_demo: bool = False):
        self._db = db_path
        data = self._load()
        self.profiler   = DeviceProfiler(
            DeviceProfile(**data["device_profile"]) if data.get("device_profile") else None,
            fast_demo = fast_demo,
        )
        self.classifier = TaskClassifier(data.get("task_profiles"))
        self._history: list[dict] = data.get("history", [])

    # ── Public API ─────────────────────────────────────────────────────

    def recommend(
        self,
        rule_mode:    Mode,
        snap_dict:    dict,
        prompt:       str,
    ) -> LearnerDecision:
        """
        Given the static rule engine's decision, return an adaptive
        recommendation that may upgrade or downgrade the mode.

        Note: this no longer feeds the profiler itself — the caller
        (typically a background monitor loop) is expected to call
        self.profiler.record() on its own polling cadence, so device
        learning continues even when individual chat calls use
        mode_override and skip recommend() entirely.
        """
        # Get signals
        dev_bias, dev_reason  = self.profiler.device_bias()
        category              = self.classifier.classify(prompt)
        task_bias, task_reason = self.classifier.task_bias(category)

        # Combine
        delta     = self._compute_delta(dev_bias, task_bias)
        rule_rank = self._RANK[rule_mode]
        new_rank  = max(0, min(3, rule_rank + delta))
        final     = self._BY_RANK[new_rank]

        # Confidence: product of device-profile confidence and task sample size
        task_count = self.classifier._profiles.get(category, TaskProfile(name=category)).count
        task_conf  = min(1.0, task_count / 20)
        dev_conf   = self.profiler.profile.confidence
        confidence = 0.5 * dev_conf + 0.5 * task_conf

        explanation = self._explain(
            rule_mode, final, dev_bias, dev_reason,
            task_bias, task_reason, category, delta, confidence,
        )

        return LearnerDecision(
            recommended_mode = final,
            rule_mode        = rule_mode,
            device_bias      = dev_bias,
            task_bias        = task_bias,
            confidence       = round(confidence, 3),
            explanation      = explanation,
        )

    def record_outcome(
        self,
        prompt:        str,
        mode_used:     Mode,
        output_tokens: int,
        latency_ms:    float,
    ) -> None:
        """
        Call after each inference to update task learning.
        Also appends to history and persists to disk.
        """
        category = self.classifier.classify(prompt)
        self.classifier.record_outcome(
            category, mode_used.value, output_tokens, latency_ms
        )
        self._history.append({
            "ts":       time.time(),
            "prompt_snippet": prompt[:80],
            "category": category,
            "mode":     mode_used.value,
            "tokens":   output_tokens,
            "latency":  round(latency_ms, 1),
        })
        # Keep last 200 entries
        if len(self._history) > 200:
            self._history = self._history[-200:]
        self._save()

    def device_summary(self) -> dict:
        """Human-readable device learning summary for the dashboard."""
        p = self.profiler.profile
        return {
            "sessions_seen":     p.sessions_seen,
            "confidence_pct":    round(p.confidence * 100, 1),
            "ram_mean_free_gb":  round(p.ram_mean_free_gb, 2),
            "cpu_mean_pct":      round(p.cpu_mean_pct, 1),
            "temp_mean_c":       round(p.temp_mean_c, 1),
            "ram_chronic_low":   p.ram_chronic_low,
            "cpu_chronic_high":  p.cpu_chronic_high,
            "temp_chronic_hot":  p.temp_chronic_hot,
            "device_bias":       self.profiler.device_bias()[0],
            "device_bias_reason": self.profiler.device_bias()[1],
        }

    def task_summary(self) -> list[dict]:
        """Ranked task profiles for the dashboard."""
        profiles = []
        for cat, p in self.classifier._profiles.items():
            if p.count == 0:
                continue
            profiles.append({
                "category":       cat,
                "count":          p.count,
                "quality_needed": round(p.quality_needed, 3),
                "avg_satisfaction": round(p.avg_satisfaction, 3),
                "last_mode":      p.last_mode,
                "bias":           self.classifier.task_bias(cat)[0],
            })
        return sorted(profiles, key=lambda x: -x["count"])

    def history(self, n: int = 30) -> list[dict]:
        return self._history[-n:]

    # ── Internals ──────────────────────────────────────────────────────

    def _compute_delta(self, dev_bias: str, task_bias: str) -> int:
        """
        Returns integer rank delta (−1, 0, +1) to apply to the rule mode.
        """
        d = {"upgrade": +1, "downgrade": -1, "neutral": 0}
        dv = d[dev_bias]
        tv = d[task_bias]

        if dv == tv:
            return dv          # both agree → full shift
        if dv == 0:
            return tv          # only task signal
        if tv == 0:
            return dv          # only device signal
        # Conflict (one up, one down) → stay put
        return 0

    @staticmethod
    def _explain(
        rule_mode, final, dev_bias, dev_reason,
        task_bias, task_reason, category, delta, confidence,
    ) -> str:
        parts = [f"Rule engine → {rule_mode.value.upper()}"]
        parts.append(f"Task type: {category}")
        parts.append(f"Device profile: {dev_bias} ({dev_reason})")
        parts.append(f"Task learning: {task_bias} ({task_reason})")
        if delta > 0:
            parts.append(f"Adaptive upgrade → {final.value.upper()}")
        elif delta < 0:
            parts.append(f"Adaptive downgrade → {final.value.upper()}")
        else:
            parts.append(f"No adaptive shift → {final.value.upper()}")
        parts.append(f"Learning confidence: {confidence*100:.0f}%")
        return " | ".join(parts)

    # ── Persistence ────────────────────────────────────────────────────

    def _load(self) -> dict:
        try:
            if self._db.exists():
                with open(self._db) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save(self) -> None:
        data = {
            "device_profile": asdict(self.profiler.profile),
            "task_profiles":  self.classifier.to_dict(),
            "history":        self._history,
            "saved_at":       time.time(),
        }
        try:
            with open(self._db, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Self-test / simulation
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    random.seed(42)
    db = Path("/tmp/nomad_test_adaptive.json")
    if db.exists():
        db.unlink()

    learner = AdaptiveLearner(db_path=db)

    print("╔══ Adaptive Learner Simulation ══════════════════════════════╗")
    print("║  Simulating a memory-constrained device with mixed tasks    ║")
    print("╚═════════════════════════════════════════════════════════════╝\n")

    # Simulate a device that is chronically low on RAM
    low_ram_snap = {
        "ram_free_gb": 1.8,   # consistently low
        "cpu_percent": 55.0,
        "cpu_temp_c":  62.0,
        "network_online": True,
        "network_speed": "fast",
    }

    # Phase 1: feed 60 low-RAM snapshots so the profiler learns
    print("Phase 1 — training device profile (60 low-RAM observations)...")
    for i in range(60):
        snap = dict(low_ram_snap)
        snap["ram_free_gb"] += random.uniform(-0.3, 0.3)
        learner.profiler.record(snap)

    # Phase 2: simulate tasks that need quality (complex_diag / engine_sound)
    print("Phase 2 — simulating 20 complex diagnostic tasks (Nomad, short responses)...")
    for i in range(20):
        prompt = random.choice([
            "My car has P0301 and P0171 and rough idle",
            "Multiple misfires, won't start in the morning",
            "Engine knock and low oil pressure light intermittently",
        ])
        decision = learner.recommend(Mode.NOMAD, low_ram_snap, prompt)
        # Simulate Nomad giving a short (unsatisfying) response
        learner.record_outcome(prompt, Mode.NOMAD, output_tokens=35, latency_ms=800)

    print("Phase 3 — simulating 10 simple maintenance tasks (Nomad, good responses)...")
    for i in range(10):
        prompt = random.choice([
            "When should I change my oil?",
            "How often to replace brake pads?",
        ])
        learner.record_outcome(prompt, Mode.NOMAD, output_tokens=90, latency_ms=600)

    # Now ask for recommendations
    print("\n══ Recommendations after learning ══\n")

    test_cases = [
        (Mode.NOMAD,     low_ram_snap, "My car has P0301 and P0171, rough idle, stalls"),
        (Mode.WORKHORSE, low_ram_snap, "When should I change my engine oil?"),
        (Mode.STALLION,  low_ram_snap, "Ticking noise from engine, worse when cold"),
        (Mode.NOMAD,     low_ram_snap, "Multiple warning lights, car won't start"),
    ]

    for rule_mode, snap, prompt in test_cases:
        dec = learner.recommend(rule_mode, snap, prompt)
        arrow = "→" if dec.recommended_mode == rule_mode else "⟹ ADAPTED"
        print(f"  Prompt : \"{prompt[:55]}...\"" if len(prompt)>55 else f"  Prompt : \"{prompt}\"")
        print(f"  Rule   : {rule_mode.value:10s}  {arrow}  Adaptive: {dec.recommended_mode.value.upper()}")
        print(f"  Reason : {dec.explanation}")
        print(f"  Conf   : {dec.confidence*100:.0f}%\n")

    print("── Device summary ──────────────────────────────────────────")
    for k, v in learner.device_summary().items():
        print(f"  {k:28s}: {v}")

    print("\n── Task learning summary ───────────────────────────────────")
    for t in learner.task_summary():
        bar = "█" * int(t["quality_needed"] * 20)
        print(f"  {t['category']:15s} [{bar:<20s}] {t['quality_needed']:.2f}  ({t['count']} samples)  bias={t['bias']}")
