from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any, Dict


@dataclass
class PerceptionSample:
    ts: float
    detected: bool
    conf: float
    cls: str
    offset_px: Tuple[int, int]
    offset_norm: Tuple[float, float]
    held: bool


class PerceptionClient:
    def __init__(self, json_path: Path, max_age_s: float = 0.5):
        self.json_path = Path(json_path)
        self.max_age_s = float(max_age_s)

    def _stale(self) -> PerceptionSample:
        return PerceptionSample(
            ts=0.0,
            detected=False,
            conf=0.0,
            cls="",
            offset_px=(0, 0),
            offset_norm=(0.0, 0.0),
            held=False,
        )

    def read(self) -> Optional[PerceptionSample]:
        if not self.json_path.exists():
            return None

        now = time.time()

        # Fast stale guard using file mtime
        try:
            mtime = float(self.json_path.stat().st_mtime)
        except Exception:
            return None

        if (now - mtime) > self.max_age_s:
            return self._stale()

        # Parse JSON
        try:
            payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        result: Dict[str, Any] = payload.get("result", payload)

        detected = bool(result.get("detected", False))
        conf = float(result.get("confidence", 0.0))
        cls = str(result.get("class") or "")

        off_px = result.get("offset_px", (0, 0))
        try:
            dx_px = int(off_px[0])
            dy_px = int(off_px[1])
        except Exception:
            dx_px, dy_px = 0, 0

        off_n = result.get("offset_norm", (0.0, 0.0))
        try:
            dx_n = float(off_n[0])
            dy_n = float(off_n[1])
        except Exception:
            dx_n, dy_n = 0.0, 0.0

        held = bool(payload.get("held", False))
        ts = float(result.get("timestamp", payload.get("timestamp", 0.0)) or 0.0)

        # Second stale guard using JSON timestamp
        if ts > 0.0 and (now - ts) > self.max_age_s:
            return self._stale()

        return PerceptionSample(
            ts=ts,
            detected=detected,
            conf=conf,
            cls=cls,
            offset_px=(dx_px, dy_px),
            offset_norm=(dx_n, dy_n),
            held=held,
        )
