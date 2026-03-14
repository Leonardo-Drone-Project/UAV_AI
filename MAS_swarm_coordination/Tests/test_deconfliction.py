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


def test_close_drones_trigger_deconfliction_and_block_initial_target_owner():
    swarm = SwarmCoordinationInterface()

    drones = [
        make_drone(
            "drone_1",
            battery_pct=95.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.55,
            last_update_s=0.0,
            last_heartbeat_s=0.0,
        ),
        make_drone(
            "drone_2",
            x_m=20.0,
            y_m=8.0,
            battery_pct=88.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.95,
            tracking_locked=True,
            last_track_update_s=0.0,
            last_update_s=0.0,
            last_heartbeat_s=0.0,
        ),
        make_drone(
            "drone_3",
            x_m=20.9,
            y_m=8.1,
            battery_pct=82.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.70,
            last_update_s=0.0,
            last_heartbeat_s=0.0,
        ),
    ]

    swarm.update_many(drones)
    decision, bt_flags = swarm.step(timestamp_s=0.0)

    assert decision.parent_id == "drone_1"
    assert decision.target_owner_id is None
    assert decision.deconfliction_active is True
    assert decision.collision_risk is True
    assert ("drone_2", "drone_3") in decision.deconfliction_pairs

    assert decision.task_assignments["drone_1"] == "coordinate_and_report_target"
    assert decision.task_assignments["drone_2"] == "converge_target"
    assert decision.task_assignments["drone_3"] == "hold_position_deconflict"

    assert bt_flags["deconfliction_active"] is True
    assert bt_flags["collision_risk"] is True


def test_deconfliction_clears_and_target_owner_is_selected_after_spacing_increases():
    swarm = SwarmCoordinationInterface()

    drones_t0 = [
        make_drone(
            "drone_1",
            battery_pct=95.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.55,
            last_update_s=0.0,
            last_heartbeat_s=0.0,
        ),
        make_drone(
            "drone_2",
            x_m=20.0,
            y_m=8.0,
            battery_pct=88.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.95,
            tracking_locked=True,
            last_track_update_s=0.0,
            last_update_s=0.0,
            last_heartbeat_s=0.0,
        ),
        make_drone(
            "drone_3",
            x_m=20.9,
            y_m=8.1,
            battery_pct=82.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.70,
            last_update_s=0.0,
            last_heartbeat_s=0.0,
        ),
    ]

    swarm.update_many(drones_t0)
    decision_t0, _ = swarm.step(timestamp_s=0.0)

    assert decision_t0.parent_id == "drone_1"
    assert decision_t0.target_owner_id is None
    assert decision_t0.deconfliction_active is True

    drones_t1 = [
        make_drone(
            "drone_1",
            battery_pct=95.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.55,
            last_update_s=1.0,
            last_heartbeat_s=1.0,
        ),
        make_drone(
            "drone_2",
            x_m=20.0,
            y_m=8.0,
            battery_pct=88.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.95,
            tracking_locked=True,
            last_track_update_s=1.0,
            last_update_s=1.0,
            last_heartbeat_s=1.0,
        ),
        make_drone(
            "drone_3",
            x_m=25.0,
            y_m=8.1,
            battery_pct=82.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.70,
            last_update_s=1.0,
            last_heartbeat_s=1.0,
        ),
    ]

    swarm.update_many(drones_t1)
    decision_t1, bt_flags_t1 = swarm.step(timestamp_s=1.0)

    assert decision_t1.parent_id == "drone_1"
    assert decision_t1.target_owner_id == "drone_2"
    assert decision_t1.deconfliction_active is False
    assert decision_t1.collision_risk is False
    assert decision_t1.deconfliction_pairs == []

    assert decision_t1.task_assignments["drone_1"] == "coordinate_and_report_target"
    assert decision_t1.task_assignments["drone_2"] == "track_target"
    assert decision_t1.task_assignments["drone_3"] == "converge_target"

    assert bt_flags_t1["deconfliction_active"] is False