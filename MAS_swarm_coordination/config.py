from dataclasses import dataclass


@dataclass
class SwarmConfig:
    min_drones_for_swarm: int = 2

    battery_weight: float = 0.45
    nav_weight: float = 0.25
    comms_weight: float = 0.20
    target_weight: float = 0.10

    parent_loss_timeout_s: float = 2.0
    converge_radius_m: float = 3.0

    prefer_child_for_target_tracking: bool = True
    target_handover_distance_margin_m: float = 0.75