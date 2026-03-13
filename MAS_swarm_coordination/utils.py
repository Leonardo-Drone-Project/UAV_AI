import math
from typing import Dict, List, Optional, Set, Tuple

from .config import SwarmConfig
from .models import DroneState


def distance_xy_m(a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> float:
    dx = float(a_xy[0]) - float(b_xy[0])
    dy = float(a_xy[1]) - float(b_xy[1])
    return math.hypot(dx, dy)


def drone_is_fresh(drone: DroneState, now_s: float, timeout_s: float) -> bool:
    return (float(now_s) - float(drone.last_update_s)) <= float(timeout_s)


def drone_heartbeat_ok(drone: DroneState, now_s: float, config: SwarmConfig) -> bool:
    heartbeat_fresh = (float(now_s) - float(drone.last_heartbeat_s)) <= float(config.heartbeat_timeout_s)
    heartbeat_count_ok = int(drone.missed_heartbeats) <= int(config.max_missed_heartbeats)
    return heartbeat_fresh and heartbeat_count_ok


def drone_is_healthy_for_swarm(drone: DroneState, now_s: float, config: SwarmConfig) -> bool:
    return (
        drone.available
        and drone.comms_ok
        and drone.nav_ok
        and not drone.direct_control_enabled
        and drone.battery_pct > 0.0
        and drone_is_fresh(drone, now_s, config.drone_stale_timeout_s)
        and drone_heartbeat_ok(drone, now_s, config)
    )


def drone_is_valid_for_tracking(drone: DroneState, now_s: float, config: SwarmConfig) -> bool:
    if not drone_is_healthy_for_swarm(drone, now_s, config):
        return False

    if drone.battery_pct < config.min_tracking_battery_pct:
        return False

    if drone.target_detected and drone.target_confidence >= config.min_target_confidence:
        return True

    if drone.tracking_locked and (now_s - float(drone.last_track_update_s)) <= config.track_lock_timeout_s:
        return True

    return False


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
    healthy = [d for d in drones if drone_is_healthy_for_swarm(d, now_s, config)]

    if len(healthy) < config.min_drones_for_swarm:
        return None, [], {}

    ordered = sorted(healthy, key=lambda d: score_drone_for_parent(d, config), reverse=True)

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
    ws: List[float] = []

    for drone in drones:
        if drone.target_detected and drone.target_x_m is not None and drone.target_y_m is not None:
            weight = max(0.05, float(drone.target_confidence))
            xs.append(float(drone.target_x_m))
            ys.append(float(drone.target_y_m))
            ws.append(weight)

    if not xs:
        return None

    w_sum = sum(ws)
    x_est = sum(x * w for x, w in zip(xs, ws)) / w_sum
    y_est = sum(y * w for y, w in zip(ys, ws)) / w_sum
    return (x_est, y_est)


def _healthy_role_drones(
    drones_by_id: Dict[str, DroneState],
    roles: Dict[str, str],
    now_s: float,
    config: SwarmConfig,
) -> List[DroneState]:
    out: List[DroneState] = []
    for drone_id in roles:
        drone = drones_by_id.get(drone_id)
        if drone is None:
            continue
        if drone_is_healthy_for_swarm(drone, now_s, config):
            out.append(drone)
    return out


def choose_target_candidate(
    drones_by_id: Dict[str, DroneState],
    roles: Dict[str, str],
    target_xy: Optional[Tuple[float, float]],
    now_s: float,
    config: SwarmConfig,
) -> Optional[str]:
    if target_xy is None or not roles:
        return None

    healthy = _healthy_role_drones(drones_by_id, roles, now_s, config)
    if not healthy:
        return None

    tracking_candidates = [d for d in healthy if drone_is_valid_for_tracking(d, now_s, config)]
    if not tracking_candidates:
        tracking_candidates = [d for d in healthy if d.battery_pct >= config.min_tracking_battery_pct]
    if not tracking_candidates:
        return None

    candidate_pool = tracking_candidates
    if config.prefer_child_for_target_tracking:
        children = [d for d in candidate_pool if roles.get(d.drone_id) == "child"]
        if children:
            candidate_pool = children

    best_candidate = min(candidate_pool, key=lambda d: distance_xy_m(d.position_xy(), target_xy))
    return best_candidate.drone_id


def detect_deconfliction(
    drones_by_id: Dict[str, DroneState],
    roles: Dict[str, str],
    now_s: float,
    config: SwarmConfig,
    previous_active_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[bool, List[Tuple[str, str]], bool, Set[Tuple[str, str]]]:
    previous_active_pairs = previous_active_pairs or set()

    healthy = _healthy_role_drones(drones_by_id, roles, now_s, config)
    if len(healthy) < 2:
        return False, [], False, set()

    active_pairs: Set[Tuple[str, str]] = set()

    for i in range(len(healthy)):
        for j in range(i + 1, len(healthy)):
            a = healthy[i]
            b = healthy[j]
            pair = tuple(sorted((a.drone_id, b.drone_id)))
            dist = distance_xy_m(a.position_xy(), b.position_xy())

            if pair in previous_active_pairs:
                if dist < config.clear_separation_m:
                    active_pairs.add(pair)
            else:
                if dist < config.min_separation_m:
                    active_pairs.add(pair)

    active = len(active_pairs) > 0
    collision_risk = active
    return active, sorted(list(active_pairs)), collision_risk, active_pairs


def _priority_index(drone_id: str, priority_list: List[str]) -> int:
    if drone_id in priority_list:
        return priority_list.index(drone_id)
    return 10_000


def resolve_deconfliction_holds(
    conflict_pairs: List[Tuple[str, str]],
    roles: Dict[str, str],
    priority_list: List[str],
    target_owner_id: Optional[str],
) -> Set[str]:
    hold_ids: Set[str] = set()

    for a_id, b_id in conflict_pairs:
        if target_owner_id is not None:
            if a_id == target_owner_id and b_id != target_owner_id:
                hold_ids.add(b_id)
                continue
            if b_id == target_owner_id and a_id != target_owner_id:
                hold_ids.add(a_id)
                continue

        a_role = roles.get(a_id, "")
        b_role = roles.get(b_id, "")

        if a_role == "parent" and b_role != "parent":
            hold_ids.add(b_id)
            continue
        if b_role == "parent" and a_role != "parent":
            hold_ids.add(a_id)
            continue

        a_idx = _priority_index(a_id, priority_list)
        b_idx = _priority_index(b_id, priority_list)

        if a_idx <= b_idx:
            hold_ids.add(b_id)
        else:
            hold_ids.add(a_id)

    return hold_ids


def build_task_assignments(
    parent_id: Optional[str],
    priority_list: List[str],
    target_known: bool,
    target_owner_id: Optional[str],
    hold_ids: Optional[Set[str]] = None,
) -> Dict[str, str]:
    if parent_id is None:
        return {}

    hold_ids = hold_ids or set()
    tasks: Dict[str, str] = {}

    if target_known:
        if target_owner_id == parent_id:
            tasks[parent_id] = "coordinate_report_and_track_target"
        else:
            tasks[parent_id] = "coordinate_and_report_target"

        for child_id in priority_list:
            if child_id == target_owner_id:
                tasks[child_id] = "track_target"
            else:
                tasks[child_id] = "converge_target"
    else:
        tasks[parent_id] = "coordinate_swarm"

        if priority_list:
            tasks[priority_list[0]] = "search_primary"

        for child_id in priority_list[1:]:
            tasks[child_id] = "search_support"

    for drone_id in hold_ids:
        if drone_id in tasks:
            tasks[drone_id] = "hold_position_deconflict"

    return tasks