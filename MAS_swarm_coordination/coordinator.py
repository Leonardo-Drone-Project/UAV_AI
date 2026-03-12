from typing import Dict, Iterable, Optional

from .config import SwarmConfig
from .models import DroneState, SwarmDecision
from .utils import (
    build_task_assignments,
    choose_parent_and_priority,
    distance_xy_m,
    drone_is_fresh,
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

    def _current_parent_is_lost(self, now_s: float) -> bool:
        if self.active_parent_id is None:
            return False

        parent = self._drones.get(self.active_parent_id)
        if parent is None:
            return True

        return not drone_is_healthy_for_swarm(parent, now_s, self.config.parent_loss_timeout_s)

    def step(self, timestamp_s: float) -> SwarmDecision:
        now_s = float(timestamp_s)
        drones = list(self._drones.values())

        stale_drone_ids = [
            d.drone_id for d in drones
            if not drone_is_fresh(d, now_s, self.config.parent_loss_timeout_s)
        ]

        parent_lost = self._current_parent_is_lost(now_s)

        prev_parent_id = self.active_parent_id

        parent_id, priority_list, roles = choose_parent_and_priority(
            drones=drones,
            config=self.config,
            now_s=now_s,
        )

        parent_reassigned = False
        if prev_parent_id is not None and parent_id is not None and parent_id != prev_parent_id:
            parent_reassigned = True
        if parent_lost and prev_parent_id is not None and parent_id != prev_parent_id:
            parent_reassigned = True

        self.last_parent_id = self.active_parent_id
        self.active_parent_id = parent_id

        target_xy = estimate_target_xy(drones)
        target_known = target_xy is not None

        task_assignments = build_task_assignments(
            parent_id=parent_id,
            priority_list=priority_list,
            target_known=target_known,
        )

        converge_complete = False
        if target_known and parent_id is not None:
            children_ids = [drone_id for drone_id, role in roles.items() if role == "child"]
            if children_ids:
                converge_complete = True
                for child_id in children_ids:
                    child = self._drones[child_id]

                    if not drone_is_healthy_for_swarm(child, now_s, self.config.parent_loss_timeout_s):
                        converge_complete = False
                        break

                    child_xy = (child.x_m, child.y_m)
                    if distance_xy_m(child_xy, target_xy) > self.config.converge_radius_m:
                        converge_complete = False
                        break

        swarm_ready = parent_id is not None and len(priority_list) >= 1

        return SwarmDecision(
            timestamp_s=now_s,
            parent_id=parent_id,
            priority_list=priority_list,
            assigned_roles=roles,
            task_assignments=task_assignments,
            stale_drone_ids=stale_drone_ids,
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