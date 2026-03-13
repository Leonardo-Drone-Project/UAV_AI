import json
from typing import Any, Dict, List, Optional, Tuple

from .models import DroneState, SwarmDecision
from .schemas import DroneStatusMessage, SwarmCommandMessage


def _first(payload: Dict[str, Any], keys, default=None):
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False

    return default


def _timestamp_s(payload: Dict[str, Any]) -> float:
    if "timestamp_s" in payload:
        return float(payload["timestamp_s"])
    if "timestamp" in payload:
        return float(payload["timestamp"])
    if "time_s" in payload:
        return float(payload["time_s"])
    if "time_ms" in payload:
        return float(payload["time_ms"]) / 1000.0
    if "time_boot_ms" in payload:
        return float(payload["time_boot_ms"]) / 1000.0
    return 0.0


def _xyz_from_payload(payload: Dict[str, Any]) -> Tuple[float, float, float]:
    vec = _first(payload, ["position_xyz_m", "position", "pos_xyz_m"], default=None)

    if isinstance(vec, dict):
        return (
            float(_first(vec, ["x", "east_m", "pos_x"], default=0.0)),
            float(_first(vec, ["y", "north_m", "pos_y"], default=0.0)),
            float(_first(vec, ["z", "up_m", "pos_z"], default=0.0)),
        )

    if isinstance(vec, (list, tuple)) and len(vec) >= 3:
        return (float(vec[0]), float(vec[1]), float(vec[2]))

    return (
        float(_first(payload, ["x_m", "x", "east_m", "pos_x"], default=0.0)),
        float(_first(payload, ["y_m", "y", "north_m", "pos_y"], default=0.0)),
        float(_first(payload, ["z_m", "z", "up_m", "pos_z"], default=0.0)),
    )


def _target_xy_from_payload(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    target = _first(payload, ["target_xy_m", "target_position", "target"], default=None)

    if isinstance(target, dict):
        tx = _first(target, ["x", "target_x_m", "tx"], default=None)
        ty = _first(target, ["y", "target_y_m", "ty"], default=None)
        return (
            None if tx is None else float(tx),
            None if ty is None else float(ty),
        )

    if isinstance(target, (list, tuple)) and len(target) >= 2:
        return (float(target[0]), float(target[1]))

    tx = _first(payload, ["target_x_m", "tx", "target_x"], default=None)
    ty = _first(payload, ["target_y_m", "ty", "target_y"], default=None)

    return (
        None if tx is None else float(tx),
        None if ty is None else float(ty),
    )


def payload_to_status_message(payload: Dict[str, Any]) -> DroneStatusMessage:
    drone_id = _first(payload, ["drone_id", "id", "agent_id", "uav_id"], default=None)
    if drone_id is None:
        raise ValueError("Payload must contain a drone_id, id, agent_id, or uav_id field")

    x_m, y_m, z_m = _xyz_from_payload(payload)
    target_x_m, target_y_m = _target_xy_from_payload(payload)

    explicit_target_detected = _first(
        payload,
        ["target_detected", "has_target", "target_seen"],
        default=None,
    )

    if explicit_target_detected is None:
        target_detected = (target_x_m is not None) and (target_y_m is not None)
    else:
        target_detected = _to_bool(explicit_target_detected, default=False)

    timestamp_s = _timestamp_s(payload)
    last_update_s = float(_first(payload, ["last_update_s"], default=timestamp_s))
    last_heartbeat_s = float(_first(payload, ["last_heartbeat_s"], default=timestamp_s))
    last_track_update_s = float(_first(payload, ["last_track_update_s"], default=timestamp_s))

    msg = DroneStatusMessage(
        timestamp_s=timestamp_s,
        drone_id=str(drone_id),
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        battery_pct=float(_first(payload, ["battery_pct", "battery", "battery_percent"], default=100.0)),
        comms_ok=_to_bool(_first(payload, ["comms_ok", "link_ok", "communications_ok"], default=True), default=True),
        nav_ok=_to_bool(_first(payload, ["nav_ok", "ekf_ok", "navigation_ok"], default=True), default=True),
        available=_to_bool(_first(payload, ["available", "is_available"], default=True), default=True),
        direct_control_enabled=_to_bool(
            _first(payload, ["direct_control_enabled", "manual_override", "operator_control"], default=False),
            default=False,
        ),
        target_detected=target_detected,
        target_x_m=target_x_m,
        target_y_m=target_y_m,
        target_confidence=float(_first(payload, ["target_confidence", "target_conf", "confidence"], default=0.0)),
        tracking_locked=_to_bool(_first(payload, ["tracking_locked", "track_locked"], default=False), default=False),
        last_track_update_s=last_track_update_s,
        handover_ack=_to_bool(_first(payload, ["handover_ack", "handover_accept"], default=False), default=False),
        handover_reject=_to_bool(_first(payload, ["handover_reject"], default=False), default=False),
        last_update_s=last_update_s,
        last_heartbeat_s=last_heartbeat_s,
        heartbeat_seq=int(_first(payload, ["heartbeat_seq", "hb_seq"], default=0)),
        missed_heartbeats=int(_first(payload, ["missed_heartbeats", "hb_missed"], default=0)),
    )
    msg.validate()
    return msg


def drone_from_payload(payload: Dict[str, Any]) -> DroneState:
    msg = payload_to_status_message(payload)

    return DroneState(
        drone_id=msg.drone_id,
        x_m=msg.x_m,
        y_m=msg.y_m,
        z_m=msg.z_m,
        battery_pct=msg.battery_pct,
        comms_ok=msg.comms_ok,
        nav_ok=msg.nav_ok,
        available=msg.available,
        direct_control_enabled=msg.direct_control_enabled,
        target_detected=msg.target_detected,
        target_x_m=msg.target_x_m,
        target_y_m=msg.target_y_m,
        target_confidence=msg.target_confidence,
        tracking_locked=msg.tracking_locked,
        last_track_update_s=msg.last_track_update_s,
        handover_ack=msg.handover_ack,
        handover_reject=msg.handover_reject,
        last_update_s=msg.last_update_s,
        last_heartbeat_s=msg.last_heartbeat_s,
        heartbeat_seq=msg.heartbeat_seq,
        missed_heartbeats=msg.missed_heartbeats,
    )


def decision_to_command_messages(decision: SwarmDecision) -> List[SwarmCommandMessage]:
    commands: List[SwarmCommandMessage] = []

    for drone_id, role in decision.assigned_roles.items():
        task = decision.task_assignments.get(drone_id, "idle")
        priority_index = decision.priority_list.index(drone_id) if drone_id in decision.priority_list else -1
        hold_position = task == "hold_position_deconflict"

        reason = "mission"
        if decision.swarm_failure:
            reason = "swarm_failure"
        elif hold_position:
            reason = "deconfliction"
        elif decision.handover_state in {"REQUESTED", "ACCEPTED"} and drone_id == decision.pending_target_owner_id:
            reason = "handover_pending"
        elif drone_id == decision.target_owner_id:
            reason = "target_owner"
        elif drone_id == decision.parent_id:
            reason = "parent"

        commands.append(
            SwarmCommandMessage(
                timestamp_s=decision.timestamp_s,
                drone_id=drone_id,
                parent_id=decision.parent_id,
                role=role,
                task=task,
                priority_index=priority_index,
                target_owner_id=decision.target_owner_id,
                target_xy_m=decision.target_xy_m,
                handover_state=decision.handover_state,
                swarm_degraded=decision.swarm_degraded,
                swarm_failure=decision.swarm_failure,
                hold_position=hold_position,
                deconfliction_active=decision.deconfliction_active,
                collision_risk=decision.collision_risk,
                reason=reason,
            )
        )

    return commands