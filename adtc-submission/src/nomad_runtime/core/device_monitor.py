"""
Nomad Runtime — Device Monitor
Polls RAM, CPU, temperature, and network state.
Returns a DeviceSnapshot every tick.
"""

import psutil
import socket
import time
import urllib.request
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class DeviceSnapshot:
    timestamp: float
    cpu_percent: float          # 0–100
    ram_used_gb: float          # GB actually used
    ram_total_gb: float         # GB physical
    ram_free_gb: float          # GB available
    ram_percent: float          # 0–100
    cpu_temp_c: Optional[float] # None if sensor unavailable
    network_online: bool        # basic internet reachability
    network_speed: str          # "none" | "slow" | "fast"
    battery_percent: Optional[float]   # None if desktop / no battery
    battery_plugged: Optional[bool]

    def to_dict(self) -> dict:
        return asdict(self)


class DeviceMonitor:
    """
    Lightweight hardware monitor for Nomad Runtime.
    Call .snapshot() to get a fresh DeviceSnapshot.
    Call .start_loop(callback, interval) for continuous polling.
    """

    PING_HOST = "8.8.8.8"
    PING_PORT = 53
    PING_TIMEOUT = 1          # seconds — kept short so snapshot() stays fast when offline
    SPEED_TEST_URL = "https://httpbin.org/get"
    SPEED_FAST_THRESH_S = 1.5 # round-trip under this = "fast"

    def __init__(self):
        self._running = False
        # First call to psutil.cpu_percent(interval=None) always returns 0.0
        # as a baseline reading; prime it once here so the first real
        # snapshot() already returns a meaningful (non-blocking) value.
        psutil.cpu_percent(interval=None)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> DeviceSnapshot:
        """Collect all metrics and return a snapshot."""
        cpu   = self._cpu()
        ram   = self._ram()
        temp  = self._temperature()
        net   = self._network()
        bat   = self._battery()

        return DeviceSnapshot(
            timestamp        = time.time(),
            cpu_percent      = cpu,
            ram_used_gb      = ram["used_gb"],
            ram_total_gb     = ram["total_gb"],
            ram_free_gb      = ram["free_gb"],
            ram_percent      = ram["percent"],
            cpu_temp_c       = temp,
            network_online   = net["online"],
            network_speed    = net["speed"],
            battery_percent  = bat["percent"],
            battery_plugged  = bat["plugged"],
        )

    def start_loop(self, callback, interval: float = 5.0):
        """
        Run snapshot() every `interval` seconds and pass the result to callback.
        Blocking — run in a thread.
        """
        self._running = True
        while self._running:
            snap = self.snapshot()
            callback(snap)
            time.sleep(interval)

    def stop_loop(self):
        self._running = False

    # ------------------------------------------------------------------ #
    #  Internal collectors                                                 #
    # ------------------------------------------------------------------ #

    def _cpu(self) -> float:
        # Non-blocking: returns usage since the previous call (primed in
        # __init__ / previous snapshot). Avoids the 0.5s stall the blocking
        # form incurs, which matters for fast polling loops and the
        # adaptive learner's convergence speed.
        return psutil.cpu_percent(interval=None)

    def _ram(self) -> dict:
        mem = psutil.virtual_memory()
        gb  = 1024 ** 3
        return {
            "used_gb":  round(mem.used  / gb, 2),
            "total_gb": round(mem.total / gb, 2),
            "free_gb":  round(mem.available / gb, 2),
            "percent":  mem.percent,
        }

    def _temperature(self) -> Optional[float]:
        """
        Returns the highest CPU core temperature, or None.
        Works on Linux with lm-sensors; returns None on macOS/Windows or
        when sensors are absent (common in VMs and containers).
        """
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return None
            # Prefer coretemp or k10temp (AMD)
            for key in ("coretemp", "k10temp", "cpu_thermal"):
                if key in temps:
                    readings = [t.current for t in temps[key]]
                    return round(max(readings), 1) if readings else None
            # Fallback: first available sensor
            first = next(iter(temps.values()))
            readings = [t.current for t in first]
            return round(max(readings), 1) if readings else None
        except Exception:
            return None

    def _network(self) -> dict:
        """
        Fast TCP probe to check internet reachability.
        Speed is classified by a lightweight HTTPS round-trip.
        """
        online = self._tcp_probe()
        if not online:
            return {"online": False, "speed": "none"}

        speed = self._classify_speed()
        return {"online": True, "speed": speed}

    def _tcp_probe(self) -> bool:
        try:
            s = socket.create_connection(
                (self.PING_HOST, self.PING_PORT),
                timeout=self.PING_TIMEOUT
            )
            s.close()
            return True
        except OSError:
            return False

    def _classify_speed(self) -> str:
        try:
            t0 = time.time()
            urllib.request.urlopen(self.SPEED_TEST_URL, timeout=3)
            elapsed = time.time() - t0
            return "fast" if elapsed < self.SPEED_FAST_THRESH_S else "slow"
        except Exception:
            return "slow"

    def _battery(self) -> dict:
        try:
            bat = psutil.sensors_battery()
            if bat is None:
                return {"percent": None, "plugged": None}
            return {
                "percent": round(bat.percent, 1),
                "plugged": bat.power_plugged,
            }
        except Exception:
            return {"percent": None, "plugged": None}


# ------------------------------------------------------------------ #
#  Quick self-test                                                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    monitor = DeviceMonitor()
    snap = monitor.snapshot()
    print("=== Device Snapshot ===")
    for k, v in snap.to_dict().items():
        print(f"  {k:22s}: {v}")
