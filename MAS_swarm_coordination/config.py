from dataclasses import dataclass


@dataclass
class SwarmConfig:
    min_drones_for_swarm: int = 2

    battery_weight: float = 0.45
    nav_weight: float = 0.25
    comms_weight: float = 0.20
    target_weight: float = 0.10

    drone_stale_timeout_s: float = 2.0
    heartbeat_timeout_s: float = 1.5
    max_missed_heartbeats: int = 3

    converge_radius_m: float = 3.0

    prefer_child_for_target_tracking: bool = True
    target_handover_distance_margin_m: float = 0.75
    handover_accept_timeout_s: float = 2.0

    min_tracking_battery_pct: float = 20.0
    min_target_confidence: float = 0.50
    track_lock_timeout_s: float = 1.5

    target_owner_stability_lock_s: float = 2.0
    parent_reassignment_stability_lock_s: float = 1.5

    min_separation_m: float = 2.5
    clear_separation_m: float = 3.25