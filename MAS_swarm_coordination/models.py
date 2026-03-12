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

    last_update_s: float = 0.0

    def position_xyz(self) -> Tuple[float, float, float]:
        return (self.x_m, self.y_m, self.z_m)


@dataclass
class SwarmDecision:
    timestamp_s: float = 0.0

    parent_id: Optional[str] = None
    priority_list: List[str] = field(default_factory=list)
    assigned_roles: Dict[str, str] = field(default_factory=dict)
    task_assignments: Dict[str, str] = field(default_factory=dict)

    stale_drone_ids: List[str] = field(default_factory=list)

    swarm_coordinated: bool = False
    roles_assigned: bool = False
    priority_list_sent: bool = False
    role_election_failed: bool = False

    parent_lost: bool = False
    parent_reassigned: bool = False

    converge_complete: bool = False
    target_known: bool = False
    target_xy_m: Optional[Tuple[float, float]] = None