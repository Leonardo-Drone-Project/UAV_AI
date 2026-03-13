from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Tuple

from .adapters import decision_to_command_messages
from .config import SwarmConfig
from .coordinator import SwarmCoordinator
from .models import DroneState, SwarmDecision
from .schemas import SwarmCommandMessage


class SwarmCoordinationInterface:
    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config or SwarmConfig()
        self.coordinator = SwarmCoordinator(self.config)

    def update_drone(self, drone: DroneState) -> None:
        self.coordinator.update_drone(drone)

    def update_many(self, drones: Iterable[DroneState]) -> None:
        self.coordinator.update_many(drones)

    def step(self, timestamp_s: float) -> Tuple[SwarmDecision, Dict[str, bool]]:
        decision = self.coordinator.step(timestamp_s=timestamp_s)
        bt_flags = self.derive_bt_inputs(decision)
        return decision, bt_flags

    def step_with_commands(
        self,
        timestamp_s: float,
    ) -> Tuple[SwarmDecision, Dict[str, bool], List[SwarmCommandMessage]]:
        decision, bt_flags = self.step(timestamp_s=timestamp_s)
        commands = decision_to_command_messages(decision)
        return decision, bt_flags, commands

    @staticmethod
    def derive_bt_inputs(decision: SwarmDecision) -> Dict[str, bool]:
        return {
            "swarm_coordinated": bool(decision.swarm_coordinated),
            "roles_assigned": bool(decision.roles_assigned),
            "priority_list_sent": bool(decision.priority_list_sent),
            "role_election_failed": bool(decision.role_election_failed),
            "swarm_degraded": bool(decision.swarm_degraded),
            "swarm_failure": bool(decision.swarm_failure),
            "parent_lost": bool(decision.parent_lost),
            "parent_reassigned": bool(decision.parent_reassigned),
            "converge_complete": bool(decision.converge_complete),
            "target_handover_required": bool(decision.target_handover_required),
            "target_handover_complete": bool(decision.target_handover_complete),
            "deconfliction_active": bool(decision.deconfliction_active),
            "collision_risk": bool(decision.collision_risk),
        }

    @staticmethod
    def decision_to_dict(decision: SwarmDecision) -> Dict[str, object]:
        return asdict(decision)