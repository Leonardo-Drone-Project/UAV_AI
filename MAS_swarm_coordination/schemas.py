from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple


@dataclass
class DroneStatusMessage:
    timestamp_s: float
    drone_id: str

    x_m: float
    y_m: float
    z_m: float

    battery_pct: float
    comms_ok: bool
    nav_ok: bool
    available: bool
    direct_control_enabled: bool

    target_detected: bool
    target_x_m: Optional[float]
    target_y_m: Optional[float]
    target_confidence: float

    tracking_locked: bool
    last_track_update_s: float

    handover_ack: bool
    handover_reject: bool

    last_update_s: float
    last_heartbeat_s: float
    heartbeat_seq: int
    missed_heartbeats: int

    def validate(self) -> None:
        if not self.drone_id:
            raise ValueError("drone_id must be provided")

        if not (0.0 <= self.battery_pct <= 100.0):
            raise ValueError("battery_pct must be between 0 and 100")

        if not (0.0 <= self.target_confidence <= 1.0):
            raise ValueError("target_confidence must be between 0 and 1")

        if self.target_detected and (self.target_x_m is None or self.target_y_m is None):
            raise ValueError("target_detected is True but target position is missing")

        if self.heartbeat_seq < 0:
            raise ValueError("heartbeat_seq must be non-negative")

        if self.missed_heartbeats < 0:
            raise ValueError("missed_heartbeats must be non-negative")

        if self.handover_ack and self.handover_reject:
            raise ValueError("handover_ack and handover_reject cannot both be True")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class SwarmCommandMessage:
    timestamp_s: float
    drone_id: str
    parent_id: Optional[str]
    role: str
    task: str
    priority_index: int

    target_owner_id: Optional[str]
    target_xy_m: Optional[Tuple[float, float]]

    handover_state: str
    swarm_degraded: bool
    swarm_failure: bool

    hold_position: bool
    deconfliction_active: bool
    collision_risk: bool
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)