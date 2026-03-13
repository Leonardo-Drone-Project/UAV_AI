from typing import Dict, Iterable, Optional, Set, Tuple

from .config import SwarmConfig
from .models import DroneState, SwarmDecision
from .utils import (
    build_task_assignments,
    choose_parent_and_priority,
    choose_target_candidate,
    detect_deconfliction,
    distance_xy_m,
    drone_heartbeat_ok,
    drone_is_fresh,
    drone_is_healthy_for_swarm,
    drone_is_valid_for_tracking,
    estimate_target_xy,
    resolve_deconfliction_holds,
)


class SwarmCoordinator:
    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config or SwarmConfig()
        self._drones: Dict[str, DroneState] = {}

        self.active_parent_id: Optional[str] = None
        self.last_parent_id: Optional[str] = None

        self.active_target_owner_id: Optional[str] = None
        self.last_target_owner_id: Optional[str] = None

        self.pending_target_owner_id: Optional[str] = None
        self.handover_state: str = "IDLE"
        self.handover_request_start_s: Optional[float] = None

        self.target_owner_lock_until_s: float = 0.0
        self.parent_reassignment_lock_until_s: float = 0.0

        self._active_deconfliction_pairs: Set[Tuple[str, str]] = set()

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

        return not drone_is_healthy_for_swarm(parent, now_s, self.config)

    def _reset_handover(self) -> None:
        self.pending_target_owner_id = None
        self.handover_request_start_s = None
        self.handover_state = "IDLE"

    def _update_handover_state(
        self,
        now_s: float,
        desired_owner_id: Optional[str],
        target_known: bool,
        blocked_candidate_ids: Optional[Set[str]] = None,
    ) -> Tuple[bool, bool, str]:
        blocked_candidate_ids = blocked_candidate_ids or set()
        target_handover_required = False
        target_handover_complete = False

        if not target_known or desired_owner_id is None:
            current_owner = self._drones.get(self.active_target_owner_id) if self.active_target_owner_id else None
            if current_owner is None or not drone_is_valid_for_tracking(current_owner, now_s, self.config):
                self.active_target_owner_id = None
            self._reset_handover()
            return False, False, "IDLE"

        current_owner = self._drones.get(self.active_target_owner_id) if self.active_target_owner_id else None
        current_owner_valid = (
            current_owner is not None
            and drone_is_valid_for_tracking(current_owner, now_s, self.config)
        )

        if self.active_target_owner_id is not None and not current_owner_valid:
            self.active_target_owner_id = None
            self.target_owner_lock_until_s = 0.0
            current_owner = None
            current_owner_valid = False

        if self.active_target_owner_id is None:
            self.active_target_owner_id = desired_owner_id
            self._reset_handover()
            return False, False, "IDLE"

        if current_owner_valid and now_s < self.target_owner_lock_until_s:
            self._reset_handover()
            return False, False, "IDLE"

        if current_owner_valid and now_s < self.parent_reassignment_lock_until_s:
            self._reset_handover()
            return False, False, "IDLE"

        if desired_owner_id == self.active_target_owner_id:
            self._reset_handover()
            return False, False, "IDLE"

        target_handover_required = True

        if self.pending_target_owner_id != desired_owner_id or self.handover_state not in {"REQUESTED", "ACCEPTED"}:
            self.pending_target_owner_id = desired_owner_id
            self.handover_request_start_s = now_s
            self.handover_state = "REQUESTED"

        pending = self._drones.get(self.pending_target_owner_id) if self.pending_target_owner_id else None

        if pending is None:
            self.pending_target_owner_id = None
            self.handover_request_start_s = None
            self.handover_state = "FAILED"
            return False, False, "FAILED"

        if pending.drone_id in blocked_candidate_ids:
            self.pending_target_owner_id = None
            self.handover_request_start_s = None
            self.handover_state = "FAILED"
            return False, False, "FAILED"

        if not drone_is_valid_for_tracking(pending, now_s, self.config):
            self.pending_target_owner_id = None
            self.handover_request_start_s = None
            self.handover_state = "FAILED"
            return False, False, "FAILED"

        if pending.handover_reject:
            self.pending_target_owner_id = None
            self.handover_request_start_s = None
            self.handover_state = "FAILED"
            return False, False, "FAILED"

        if self.handover_request_start_s is not None:
            if (now_s - self.handover_request_start_s) > self.config.handover_accept_timeout_s:
                self.pending_target_owner_id = None
                self.handover_request_start_s = None
                self.handover_state = "TIMED_OUT"
                return False, False, "TIMED_OUT"

        if self.handover_state == "REQUESTED" and pending.handover_ack:
            self.handover_state = "ACCEPTED"

        if self.handover_state == "ACCEPTED":
            self.last_target_owner_id = self.active_target_owner_id
            self.active_target_owner_id = pending.drone_id
            self.pending_target_owner_id = None
            self.handover_request_start_s = None
            self.handover_state = "ACTIVE"
            self.target_owner_lock_until_s = now_s + self.config.target_owner_stability_lock_s
            target_handover_complete = True
            return True, True, "ACTIVE"

        return True, False, self.handover_state

    def step(self, timestamp_s: float) -> SwarmDecision:
        now_s = float(timestamp_s)
        drones = list(self._drones.values())

        stale_drone_ids = [
            d.drone_id for d in drones
            if not drone_is_fresh(d, now_s, self.config.drone_stale_timeout_s)
        ]

        heartbeat_lost_drone_ids = [
            d.drone_id for d in drones
            if not drone_heartbeat_ok(d, now_s, self.config)
        ]

        parent_lost = self._current_parent_is_lost(now_s)

        prev_parent_id = self.active_parent_id
        prev_target_owner_id = self.active_target_owner_id

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

        if parent_reassigned:
            self.parent_reassignment_lock_until_s = now_s + self.config.parent_reassignment_stability_lock_s

        target_xy = estimate_target_xy(drones)
        target_known = target_xy is not None

        deconfliction_active, deconfliction_pairs, collision_risk, next_active_pairs = detect_deconfliction(
            drones_by_id=self._drones,
            roles=roles,
            now_s=now_s,
            config=self.config,
            previous_active_pairs=self._active_deconfliction_pairs,
        )
        self._active_deconfliction_pairs = next_active_pairs

        blocked_candidate_ids: Set[str] = set()
        for a_id, b_id in deconfliction_pairs:
            blocked_candidate_ids.add(a_id)
            blocked_candidate_ids.add(b_id)

        desired_target_owner_id = choose_target_candidate(
            drones_by_id=self._drones,
            roles=roles,
            target_xy=target_xy,
            current_target_owner_id=self.active_target_owner_id,
            now_s=now_s,
            config=self.config,
            blocked_candidate_ids=blocked_candidate_ids,
        )

        target_handover_required, target_handover_complete, handover_state_out = self._update_handover_state(
            now_s=now_s,
            desired_owner_id=desired_target_owner_id,
            target_known=target_known,
            blocked_candidate_ids=blocked_candidate_ids,
        )

        hold_ids = resolve_deconfliction_holds(
            conflict_pairs=deconfliction_pairs,
            roles=roles,
            priority_list=priority_list,
            target_owner_id=self.active_target_owner_id,
        )

        task_assignments = build_task_assignments(
            parent_id=parent_id,
            priority_list=priority_list,
            target_known=target_known,
            target_owner_id=self.active_target_owner_id,
            hold_ids=hold_ids,
        )

        converge_complete = False
        if target_known and parent_id is not None:
            children_ids = [drone_id for drone_id, role in roles.items() if role == "child"]
            if children_ids:
                converge_complete = True
                for child_id in children_ids:
                    child = self._drones[child_id]

                    if not drone_is_healthy_for_swarm(child, now_s, self.config):
                        converge_complete = False
                        break

                    if task_assignments.get(child_id) == "hold_position_deconflict":
                        converge_complete = False
                        break

                    child_xy = (child.x_m, child.y_m)
                    if distance_xy_m(child_xy, target_xy) > self.config.converge_radius_m:
                        converge_complete = False
                        break

        healthy_count = sum(1 for d in drones if drone_is_healthy_for_swarm(d, now_s, self.config))

        swarm_failure = False
        failure_reason = ""
        if healthy_count < self.config.min_drones_for_swarm:
            swarm_failure = True
            failure_reason = "insufficient_healthy_drones"
        elif parent_id is None:
            swarm_failure = True
            failure_reason = "no_valid_parent"

        degraded_reasons = []
        if stale_drone_ids:
            degraded_reasons.append("stale_drones")
        if heartbeat_lost_drone_ids:
            degraded_reasons.append("heartbeat_loss")
        if deconfliction_active:
            degraded_reasons.append("deconfliction_active")
        if target_known and self.active_target_owner_id is None:
            degraded_reasons.append("no_target_owner")
        if handover_state_out in {"REQUESTED", "ACCEPTED"}:
            degraded_reasons.append("handover_in_progress")

        swarm_degraded = (not swarm_failure) and bool(degraded_reasons)
        swarm_ready = (not swarm_failure) and parent_id is not None and len(priority_list) >= 1

        decision = SwarmDecision(
            timestamp_s=now_s,
            parent_id=parent_id,
            priority_list=priority_list,
            assigned_roles=roles,
            task_assignments=task_assignments,
            stale_drone_ids=stale_drone_ids,
            heartbeat_lost_drone_ids=heartbeat_lost_drone_ids,
            deconfliction_active=deconfliction_active,
            collision_risk=collision_risk,
            deconfliction_pairs=deconfliction_pairs,
            swarm_coordinated=swarm_ready,
            roles_assigned=swarm_ready,
            priority_list_sent=swarm_ready,
            role_election_failed=not swarm_ready,
            swarm_degraded=swarm_degraded,
            swarm_failure=swarm_failure,
            degraded_reasons=degraded_reasons,
            failure_reason=failure_reason,
            parent_lost=parent_lost,
            parent_reassigned=parent_reassigned,
            converge_complete=converge_complete,
            target_known=target_known,
            target_xy_m=target_xy,
            target_owner_id=self.active_target_owner_id,
            target_tracking_drone_id=self.active_target_owner_id,
            pending_target_owner_id=self.pending_target_owner_id,
            handover_state=handover_state_out,
            target_handover_required=target_handover_required,
            target_handover_complete=target_handover_complete,
            target_owner_lock_active=(now_s < self.target_owner_lock_until_s),
        )

        if self.handover_state == "ACTIVE":
            self.handover_state = "IDLE"

        return decision