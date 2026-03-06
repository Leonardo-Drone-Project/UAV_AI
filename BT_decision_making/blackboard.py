from dataclasses import dataclass, field
from typing import Tuple
import time


@dataclass
class Blackboard:
    # Mission mode
    mode: str = "IDLE"
    prev_mode: str = "IDLE"

    # Target sub-state
    target_state: str = "SEARCH"  # SEARCH, TRACK, LOST

    # Timing
    start_time_s: float = field(default_factory=lambda: time.time())
    mode_enter_s: float = field(default_factory=lambda: time.time())

    # Target timing
    last_seen_s: float = -1.0
    detect_stable_since_s: float = -1.0

    # Last known target offset
    target_last_offset_px: Tuple[float, float] = (0.0, 0.0)

    # Mission timing
    comms_lost_since_s: float = -1.0
    download_start_s: float = -1.0
    request_missing_start_s: float = -1.0
    search_start_s: float = -1.0
    search_last_known_start_s: float = -1.0
    verify_start_s: float = -1.0
    verify_good_since_s: float = -1.0
    report_start_s: float = -1.0

    # Misc
    tick: int = 0

    def now(self) -> float:
        return time.time()

    def reset_target_timers(self) -> None:
        self.target_state = "SEARCH"
        self.last_seen_s = -1.0
        self.detect_stable_since_s = -1.0

    def reset_verify_report_timers(self) -> None:
        self.verify_start_s = -1.0
        self.verify_good_since_s = -1.0
        self.report_start_s = -1.0

    def reset_all(self) -> None:
        self.reset_target_timers()
        self.reset_verify_report_timers()
        self.target_last_offset_px = (0.0, 0.0)

    def enter_mode(self, new_mode: str) -> None:
        if new_mode == self.mode:
            return

        now = self.now()

        self.prev_mode = self.mode
        self.mode = new_mode
        self.mode_enter_s = now

        self.download_start_s = now if new_mode == "DOWNLOAD_MISSION" else -1.0
        self.request_missing_start_s = now if new_mode == "REQUEST_MISSING_DATA" else -1.0

        if new_mode == "SEARCH_AREA":
            self.search_start_s = now
            self.reset_target_timers()
        else:
            self.search_start_s = -1.0

        if new_mode == "SEARCH_LAST_KNOWN_LOCATION":
            self.search_last_known_start_s = now
            self.reset_target_timers()
        else:
            self.search_last_known_start_s = -1.0

        self.comms_lost_since_s = now if new_mode == "LOST_COMMUNICATIONS" else -1.0

        if new_mode == "VERIFY_TARGET":
            self.verify_start_s = now
            self.verify_good_since_s = -1.0
        else:
            self.verify_start_s = -1.0
            self.verify_good_since_s = -1.0

        self.report_start_s = now if new_mode == "TARGET_REPORT" else -1.0

        if new_mode == "IDLE":
            self.reset_all()