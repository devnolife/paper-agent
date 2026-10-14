"""GPU power sampling with ``nvidia-smi`` for local-model energy per call.

A background thread reads ``nvidia-smi --query-gpu=index,power.draw,memory.used,utilization.gpu -lms <ms>`` and keeps
timestamped samples. Energy of a call on GPU *g* is the trapezoidal integral of power over the call's wall-clock
window: *gross* (all power drawn by the board) and *marginal* (minus the board's idle baseline measured before the
model was loaded). On a shared machine other workloads on the same GPU add to both numbers; the sampler records
utilisation before each model so this can be reported.
"""
from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

Sample = Tuple[float, int, float, float, float]       # t, gpu index, watts, memory MiB, utilisation %


def available() -> bool:
    return shutil.which("nvidia-smi") is not None


def snapshot() -> Dict[int, Tuple[float, float, float]]:
    """{gpu: (watts, memory MiB, utilisation %)} right now; {} without nvidia-smi."""
    if not available():
        return {}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,power.draw,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    res = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                res[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                continue
    return res


def parse_line(line: str, t: float) -> List[Sample]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        return []
    try:
        return [(t, int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))]
    except ValueError:
        return []


def integrate(samples: List[Tuple[float, float]], t0: float, t1: float) -> Optional[float]:
    """Trapezoidal integral (joules) of (t, watts) samples over [t0, t1], interpolating at the window edges."""
    pts = sorted(samples)
    if t1 <= t0 or len(pts) < 2 or pts[0][0] > t0 + 2.0 or pts[-1][0] < t1 - 2.0:
        return None

    def at(t: float) -> float:
        if t <= pts[0][0]:
            return pts[0][1]
        for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                return wa if tb == ta else wa + (wb - wa) * (t - ta) / (tb - ta)
        return pts[-1][1]

    inner = [(t, w) for t, w in pts if t0 < t < t1]
    seq = [(t0, at(t0))] + inner + [(t1, at(t1))]
    return sum((tb - ta) * (wa + wb) / 2.0 for (ta, wa), (tb, wb) in zip(seq, seq[1:]))


class GpuSampler:
    def __init__(self, interval_ms: int = 250, keep_seconds: float = 6 * 3600):
        self.interval_ms = interval_ms
        self.keep_seconds = keep_seconds
        self._samples: List[Sample] = []
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if not available() or self._proc is not None:
            return self._proc is not None
        self._proc = subprocess.Popen(["nvidia-smi", "--query-gpu=index,power.draw,memory.used,utilization.gpu",
                                       "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._read, name="gpu-sampler", daemon=True)
        self._thread.start()
        return True

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            rows = parse_line(line, time.time())
            if rows:
                with self._lock:
                    self._samples.extend(rows)
                    cutoff = time.time() - self.keep_seconds
                    if self._samples and self._samples[0][0] < cutoff:
                        self._samples = [s for s in self._samples if s[0] >= cutoff]

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
            self._proc = None

    def series(self, gpu: int, t0: float, t1: float, pad: float = 2.0) -> List[Tuple[float, float]]:
        with self._lock:
            return [(t, w) for t, g, w, _m, _u in self._samples if g == gpu and t0 - pad <= t <= t1 + pad]

    def energy_j(self, gpus: List[int], t0: float, t1: float, baseline_w: Optional[Dict[int, float]] = None
                 ) -> Tuple[Optional[float], Optional[float]]:
        """(gross, marginal) joules over [t0, t1] summed over ``gpus``; None when samples are missing."""
        gross = marginal = 0.0
        for g in gpus:
            e = integrate(self.series(g, t0, t1), t0, t1)
            if e is None:
                return None, None
            gross += e
            marginal += max(0.0, e - (baseline_w or {}).get(g, 0.0) * (t1 - t0))
        return gross, marginal

    def window(self, gpu: int, t0: float, t1: float) -> dict:
        """Gross joules, mean watts and mean utilisation of one GPU over [t0, t1] (shared GPUs: attribute later)."""
        with self._lock:
            pts = [(t, w, u) for t, g, w, _m, u in self._samples if g == gpu and t0 - 2.0 <= t <= t1 + 2.0]
        e = integrate([(t, w) for t, w, _ in pts], t0, t1)
        inner = [(w, u) for t, w, u in pts if t0 <= t <= t1] or [(w, u) for _t, w, u in pts]
        return {"gross_j": None if e is None else round(e, 1),
                "mean_w": round(statistics.mean(w for w, _ in inner), 1) if inner else None,
                "mean_util": round(statistics.mean(u for _, u in inner), 1) if inner else None}

    def baseline(self, gpu: int, seconds: float = 8.0) -> Optional[float]:
        """Median power of ``gpu`` over the next ``seconds`` (call while the model under test is not loaded)."""
        t0 = time.time()
        time.sleep(seconds)
        pts = [w for _t, w in self.series(gpu, t0, time.time(), pad=0)]
        return statistics.median(pts) if pts else None

    def utilisation(self, gpu: int, seconds: float = 3.0) -> Optional[float]:
        now = time.time()
        with self._lock:
            pts = [u for t, g, _w, _m, u in self._samples if g == gpu and t >= now - seconds]
        return statistics.mean(pts) if pts else None


def model_gpus(before: Dict[int, Tuple[float, float, float]], after: Dict[int, Tuple[float, float, float]],
               min_mib: float = 500.0) -> List[int]:
    """GPUs whose memory grew by at least ``min_mib`` between two snapshots (where a model was just loaded)."""
    return sorted(g for g in after if g in before and after[g][1] - before[g][1] >= min_mib)


def ollama_gpus(size_vram: Optional[int]) -> List[int]:
    """GPUs of the Ollama runner process whose GPU memory best matches ``size_vram`` (bytes, from ``/api/ps``);
    used when a model was already loaded (no memory delta to observe). [] when it cannot be told."""
    if not available() or not size_vram:
        return []
    try:
        idx = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    uuid_to_idx = {}
    for line in idx.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            uuid_to_idx[parts[1]] = int(parts[0])
    per_pid: Dict[str, Tuple[float, set]] = {}
    for line in apps.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4 or "ollama" not in parts[3].lower():
            continue
        try:
            mem = float(parts[2])
        except ValueError:
            continue
        tot, gpus = per_pid.get(parts[0], (0.0, set()))
        gpus.add(uuid_to_idx.get(parts[1], -1))
        per_pid[parts[0]] = (tot + mem, gpus)
    want = size_vram / 2 ** 20
    best = min(per_pid.values(), key=lambda v: abs(v[0] - want), default=None)
    if best is None or abs(best[0] - want) > 0.25 * want:
        return []
    return sorted(g for g in best[1] if g >= 0)
