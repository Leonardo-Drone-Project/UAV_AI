from dataclasses import dataclass, field
import time


@dataclass
class Blackboard:
    # State machine
    state: str = "SEARCH"  # SEARCH, TRACK, LOST, FAILSAFE

    # Timing
    start_time_s: float = field(default_factory=lambda: time.time())
    last_seen_s: float = -1.0

    # Debounce
    detect_stable_since_s: float = -1.0

    # Counters / misc
    tick: int = 0

    def now(self) -> float:
        return time.time()
