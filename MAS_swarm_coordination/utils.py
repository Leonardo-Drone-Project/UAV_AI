import math
from typing import Dict, List, Optional, Tuple

from .config import SwarmConfig
from .models import DroneState


def distance_xy_m(a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> float:
    dx = float(a_xy[0]) - float(b_xy[0])
    dy = float(a_xy[1]) - float(b_xy[1])
    return math.hypot(dx, dy)


def drone_is_fresh(drone: DroneState, now_s: float, timeout_s: float) -> bool:
    return (float(now_s) - float(drone.last_update_s)) <= float(timeout_s)


def drone_is_healthy_for_swarm(drone: DroneState, now_s: float, timeout_s: float) -> bool:
    return (
        drone.available
        and drone.comms_ok
        and drone.nav_ok
        and not drone.direct_control_enabled
        and drone.battery_pct > 0.0
        and drone_is_fresh(drone, now_s, timeout_s)
    )


def score_drone_for_parent(drone: DroneState, config: SwarmConfig) -> float:
    battery_score = max(0.0, min(1.0, drone.battery_pct / 100.0))
    nav_score = 1.0 if drone.nav_ok else 0.0
    comms_score = 1.0 if drone.comms_ok else 0.0
    target_score = 1.0 if drone.target_detected else 0.0

    return (
        config.battery_weight * battery_score
        + config.nav_weight * nav_score
        + config.comms_weight * comms_score
        + config.target_weight * target_score
    )


def choose_parent_and_priority(
    drones: List[DroneState],
    config: SwarmConfig,
    now_s: float,
) -> Tuple[Optional[str], List[str], Dict[str, str]]:
    healthy = [
        d for d in drones
        if drone_is_healthy_for_swarm(d, now_s, config.parent_loss_timeout_s)
    ]

    if len(healthy) < config.min_drones_for_swarm:
        return None, [], {}

    ordered = sorted(
        healthy,
        key=lambda d: score_drone_for_parent(d, config),
        reverse=True,
    )

    parent = ordered[0]
    children = ordered[1:]

    roles: Dict[str, str] = {parent.drone_id: "parent"}
    for child in children:
        roles[child.drone_id] = "child"

    priority_list = [d.drone_id for d in children]
    return parent.drone_id, priority_list, roles


def estimate_target_xy(drones: List[DroneState]) -> Optional[Tuple[float, float]]:
    xs: List[float] = []
    ys: List[float] = []

    for drone in drones:
        if drone.target_detected and drone.target_x_m is not None and drone.target_y_m is not None:
            xs.append(float(drone.target_x_m))
            ys.append(float(drone.target_y_m))

    if not xs:
        return None

    return (sum(xs) / len(xs), sum(ys) / len(ys))


def build_task_assignments(
    parent_id: Optional[str],
    priority_list: List[str],
    target_known: bool,
) -> Dict[str, str]:
    if parent_id is None:
        return {}

    tasks: Dict[str, str] = {}

    if target_known:
        tasks[parent_id] = "coordinate_and_report_target"

        if priority_list:
            tasks[priority_list[0]] = "track_or_converge_target"

        for child_id in priority_list[1:]:
            tasks[child_id] = "converge_target"
    else:
        tasks[parent_id] = "coordinate_swarm"

        if priority_list:
            tasks[priority_list[0]] = "search_primary"

        for child_id in priority_list[1:]:
            tasks[child_id] = "search_support"

    return tasks