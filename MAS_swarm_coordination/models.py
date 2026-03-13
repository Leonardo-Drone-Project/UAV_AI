from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DroneState:
    drone_id: str

    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0

    battery_pct: float = 100.0
    comms_ok: bool = True
    nav_ok: bool = True
    available: bool = True
    direct_control_enabled: bool = False

    target_detected: bool = False
    target_x_m: Optional[float] = None
    target_y_m: Optional[float] = None
    target_confidence: float = 0.0

    tracking_locked: bool = False
    last_track_update_s: float = 0.0

    handover_ack: bool = False
    handover_reject: bool = False

    last_update_s: float = 0.0
    last_heartbeat_s: float = 0.0
    heartbeat_seq: int = 0
    missed_heartbeats: int = 0

    def position_xyz(self) -> Tuple[float, float, float]:
        return (self.x_m, self.y_m, self.z_m)

    def position_xy(self) -> Tuple[float, float]:
        return (self.x_m, self.y_m)


@dataclass
class SwarmDecision:
    timestamp_s: float = 0.0

    parent_id: Optional[str] = None
    priority_list: List[str] = field(default_factory=list)
    assigned_roles: Dict[str, str] = field(default_factory=dict)
    task_assignments: Dict[str, str] = field(default_factory=dict)

    stale_drone_ids: List[str] = field(default_factory=list)
    heartbeat_lost_drone_ids: List[str] = field(default_factory=list)

    deconfliction_active: bool = False
    collision_risk: bool = False
    deconfliction_pairs: List[Tuple[str, str]] = field(default_factory=list)

    swarm_coordinated: bool = False
    roles_assigned: bool = False
    priority_list_sent: bool = False
    role_election_failed: bool = False

    swarm_degraded: bool = False
    swarm_failure: bool = False
    degraded_reasons: List[str] = field(default_factory=list)
    failure_reason: str = ""

    parent_lost: bool = False
    parent_reassigned: bool = False

    converge_complete: bool = False
    target_known: bool = False
    target_xy_m: Optional[Tuple[float, float]] = None

    target_owner_id: Optional[str] = None
    target_tracking_drone_id: Optional[str] = None
    pending_target_owner_id: Optional[str] = None
    handover_state: str = "IDLE"
    target_handover_required: bool = False
    target_handover_complete: bool = False

    target_owner_lock_active: bool = False



