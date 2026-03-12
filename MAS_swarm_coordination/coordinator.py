from typing import Dict, Iterable, Optional

from .config import SwarmConfig
from .models import DroneState, SwarmDecision
from .utils import (
    choose_parent_and_priority,
    distance_xy_m,
    drone_is_healthy_for_swarm,
    estimate_target_xy,
)


class SwarmCoordinator:
    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config or SwarmConfig()
        self._drones: Dict[str, DroneState] = {}

        self.active_parent_id: Optional[str] = None
        self.last_parent_id: Optional[str] = None

    def update_drone(self, drone: DroneState) -> None:
        self._drones[drone.drone_id] = drone

    def update_many(self, drones: Iterable[DroneState]) -> None:
        for drone in drones:
            self.update_drone(drone)

    def get_drones(self) -> Dict[str, DroneState]:
        return dict(self._drones)

    def _current_parent_is_lost(self) -> bool:
        if self.active_parent_id is None:
            return False

        parent = self._drones.get(self.active_parent_id)
        if parent is None:
            return True

        return not drone_is_healthy_for_swarm(parent)

    def step(self, timestamp_s: float) -> SwarmDecision:
        drones = list(self._drones.values())

        parent_lost = self._current_parent_is_lost()

        parent_id, priority_list, roles = choose_parent_and_priority(drones, self.config)

        parent_reassigned = False
        if parent_id is not None:
            if self.active_parent_id is not None and parent_id != self.active_parent_id:
                parent_reassigned = True
            if parent_lost and self.active_parent_id is not None and parent_id != self.active_parent_id:
                parent_reassigned = True

        self.last_parent_id = self.active_parent_id
        self.active_parent_id = parent_id

        target_xy = estimate_target_xy(drones)
        target_known = target_xy is not None

        converge_complete = False
        if target_known and parent_id is not None:
            children_ids = [drone_id for drone_id, role in roles.items() if role == "child"]
            if children_ids:
                converge_complete = True
                for child_id in children_ids:
                    child = self._drones[child_id]
                    child_xy = (child.x_m, child.y_m)
                    if distance_xy_m(child_xy, target_xy) > self.config.converge_radius_m:
                        converge_complete = False
                        break

        swarm_ready = parent_id is not None and len(priority_list) >= 1

        return SwarmDecision(
            timestamp_s=float(timestamp_s),
            parent_id=parent_id,
            priority_list=priority_list,
            assigned_roles=roles,
            swarm_coordinated=swarm_ready,
            roles_assigned=swarm_ready,
            priority_list_sent=swarm_ready,
            role_election_failed=not swarm_ready,
            parent_lost=parent_lost,
            parent_reassigned=parent_reassigned,
            converge_complete=converge_complete,
            target_known=target_known,
            target_xy_m=target_xy,
        )