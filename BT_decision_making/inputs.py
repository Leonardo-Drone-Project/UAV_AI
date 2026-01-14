"""
Stub inputs for Behaviour Tree testing so no YOLO needed rn

We provide deterministic scenarios so you can demonstrate BT behaviour:
- scenario 1: appear -> track -> lost -> search -> appear -> track
- scenario 2: flicker -> debounce should prevent spam state switching
- scenario 3: long lost -> FAILSAFE
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Inputs:
    target_detected: bool
    target_conf: float
    target_offset_px: Tuple[float, float]  # (dx, dy) in pixels


def _interp_offset(t: float) -> Tuple[float, float]:
    # Simple “moving target” drift
    dx = 120.0 - 30.0 * (t % 4.0)
    dy = 30.0 * ((t * 0.5) % 2.0 - 0.5)
    return (dx, dy)


def get_inputs(now_s: float, scenario: int = 1) -> Inputs:
    t = now_s  # absolute time works fine; logic below uses modulo windows

    if scenario == 1:
        # 0–2s: no target
        # 2–8s: target visible (good confidence)
        # 8–12s: lost
        # 12–18s: visible again
        phase = t % 18.0
        if 2.0 <= phase < 8.0 or 12.0 <= phase < 18.0:
            return Inputs(True, 0.85, _interp_offset(phase))
        return Inputs(False, 0.0, (0.0, 0.0))

    if scenario == 2:
        # Flicker: alternating detected / not detected every 0.2s for 6s,
        # then stable detection.
        phase = t % 10.0
        if phase < 6.0:
            flicker = int((phase / 0.2)) % 2 == 0
            conf = 0.75 if flicker else 0.0
            return Inputs(flicker, conf, _interp_offset(phase) if flicker else (0.0, 0.0))
        else:
            return Inputs(True, 0.85, _interp_offset(phase))

    if scenario == 3:
        # Target appears briefly, then disappears long enough to trigger FAILSAFE.
        phase = t % 20.0
        if 1.0 <= phase < 3.0:
            return Inputs(True, 0.9, _interp_offset(phase))
        return Inputs(False, 0.0, (0.0, 0.0))

    # default
    return Inputs(False, 0.0, (0.0, 0.0))
