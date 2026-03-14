import pytest

from MAS_swarm_coordination.interface import SwarmCoordinationInterface
from MAS_swarm_coordination.models import DroneState


def make_drone(
    drone_id: str,
    *,
    x_m: float = 0.0,
    y_m: float = 0.0,
    z_m: float = 10.0,
    battery_pct: float = 100.0,
    comms_ok: bool = True,
    nav_ok: bool = True,
    available: bool = True,
    direct_control_enabled: bool = False,
    target_detected: bool = False,
    target_x_m: float | None = None,
    target_y_m: float | None = None,
    target_confidence: float = 0.0,
    tracking_locked: bool = False,
    last_track_update_s: float = 0.0,
    handover_ack: bool = False,
    handover_reject: bool = False,
    last_update_s: float = 0.0,
    last_heartbeat_s: float = 0.0,
    heartbeat_seq: int = 0,
    missed_heartbeats: int = 0,
) -> DroneState:
    if target_detected and target_x_m is None:
        target_x_m = 20.0
    if target_detected and target_y_m is None:
        target_y_m = 8.0

    return DroneState(
        drone_id=drone_id,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        battery_pct=battery_pct,
        comms_ok=comms_ok,
        nav_ok=nav_ok,
        available=available,
        direct_control_enabled=direct_control_enabled,
        target_detected=target_detected,
        target_x_m=target_x_m,
        target_y_m=target_y_m,
        target_confidence=target_confidence,
        tracking_locked=tracking_locked,
        last_track_update_s=last_track_update_s,
        handover_ack=handover_ack,
        handover_reject=handover_reject,
        last_update_s=last_update_s,
        last_heartbeat_s=last_heartbeat_s,
        heartbeat_seq=heartbeat_seq,
        missed_heartbeats=missed_heartbeats,
    )


def test_best_healthy_drone_becomes_parent():
    swarm = SwarmCoordinationInterface()

    drones = [
        make_drone("drone_1", battery_pct=95.0, last_update_s=0.0, last_heartbeat_s=0.0),
        make_drone("drone_2", battery_pct=85.0, last_update_s=0.0, last_heartbeat_s=0.0),
        make_drone("drone_3", battery_pct=75.0, last_update_s=0.0, last_heartbeat_s=0.0),
    ]

    swarm.update_many(drones)
    decision, bt_flags = swarm.step(timestamp_s=0.0)

    assert decision.parent_id == "drone_1"
    assert decision.assigned_roles["drone_1"] == "parent"
    assert decision.priority_list == ["drone_2", "drone_3"]
    assert bt_flags["swarm_coordinated"] is True
    assert bt_flags["roles_assigned"] is True
    assert bt_flags["priority_list_sent"] is True


def test_direct_control_drone_is_excluded_from_parent_selection():
    swarm = SwarmCoordinationInterface()

    drones = [
        make_drone(
            "drone_1",
            battery_pct=99.0,
            direct_control_enabled=True,
            last_update_s=0.0,
            last_heartbeat_s=0.0,
        ),
        make_drone("drone_2", battery_pct=88.0, last_update_s=0.0, last_heartbeat_s=0.0),
        make_drone("drone_3", battery_pct=82.0, last_update_s=0.0, last_heartbeat_s=0.0),
    ]

    swarm.update_many(drones)
    decision, _ = swarm.step(timestamp_s=0.0)

    assert decision.parent_id == "drone_2"
    assert "drone_1" not in decision.assigned_roles
    assert decision.priority_list == ["drone_3"]


def test_stale_or_unhealthy_drone_is_not_selected_as_parent():
    swarm = SwarmCoordinationInterface()

    drones = [
        make_drone(
            "drone_1",
            battery_pct=99.0,
            nav_ok=False,
            last_update_s=-10.0,
            last_heartbeat_s=-10.0,
        ),
        make_drone("drone_2", battery_pct=87.0, last_update_s=5.0, last_heartbeat_s=5.0),
        make_drone("drone_3", battery_pct=80.0, last_update_s=5.0, last_heartbeat_s=5.0),
    ]

    swarm.update_many(drones)
    decision, _ = swarm.step(timestamp_s=5.0)

    assert decision.parent_id == "drone_2"
    assert "drone_1" not in decision.assigned_roles
    assert decision.swarm_failure is False