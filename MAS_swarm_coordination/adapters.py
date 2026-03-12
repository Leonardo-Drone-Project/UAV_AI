from typing import Any, Dict, Optional, Tuple

from .models import DroneState


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


def drone_from_payload(payload: Dict[str, Any]) -> DroneState:
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

    return DroneState(
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
        last_update_s=float(_first(payload, ["last_update_s"], default=timestamp_s)),
    )