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


def build_initial_swarm() -> list[DroneState]:
    return [
        make_drone(
            "drone_1",
            x_m=0.0,
            y_m=0.0,
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
            x_m=17.0,
            y_m=8.0,
            battery_pct=88.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.70,
            last_update_s=0.0,
            last_heartbeat_s=0.0,
        ),
        make_drone(
            "drone_3",
            x_m=20.4,
            y_m=8.1,
            battery_pct=82.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.86,
            tracking_locked=True,
            last_track_update_s=0.0,
            last_update_s=0.0,
            last_heartbeat_s=0.0,
        ),
    ]


def test_initial_target_owner_selection():
    swarm = SwarmCoordinationInterface()
    swarm.update_many(build_initial_swarm())

    decision, bt_flags = swarm.step(timestamp_s=0.0)

    assert decision.target_owner_id == "drone_3"
    assert decision.handover_state == "IDLE"
    assert decision.target_handover_required is False
    assert decision.target_handover_complete is False
    assert bt_flags["target_handover_required"] is False
    assert bt_flags["target_handover_complete"] is False


def test_handover_request_and_completion():
    swarm = SwarmCoordinationInterface()
    drones = build_initial_swarm()
    swarm.update_many(drones)
    swarm.step(timestamp_s=0.0)

    drones[0].last_update_s = 1.0
    drones[0].last_heartbeat_s = 1.0

    drones[1].x_m = 20.0
    drones[1].y_m = 8.0
    drones[1].target_confidence = 0.92
    drones[1].handover_ack = True
    drones[1].last_update_s = 1.0
    drones[1].last_heartbeat_s = 1.0

    drones[2].x_m = 24.0
    drones[2].y_m = 8.1
    drones[2].tracking_locked = False
    drones[2].last_track_update_s = 1.0
    drones[2].last_update_s = 1.0
    drones[2].last_heartbeat_s = 1.0

    swarm.update_many(drones)
    decision, bt_flags = swarm.step(timestamp_s=1.0)

    assert decision.target_owner_id == "drone_2"
    assert decision.handover_state == "ACTIVE"
    assert decision.target_handover_required is True
    assert decision.target_handover_complete is True
    assert decision.target_owner_lock_active is True
    assert bt_flags["target_handover_required"] is True
    assert bt_flags["target_handover_complete"] is True


def test_handover_timeout_when_no_ack_is_received():
    swarm = SwarmCoordinationInterface()
    drones = build_initial_swarm()
    swarm.update_many(drones)
    swarm.step(timestamp_s=0.0)

    drones[0].last_update_s = 1.0
    drones[0].last_heartbeat_s = 1.0

    drones[1].x_m = 20.0
    drones[1].y_m = 8.0
    drones[1].target_confidence = 0.92
    drones[1].handover_ack = False
    drones[1].last_update_s = 1.0
    drones[1].last_heartbeat_s = 1.0

    drones[2].x_m = 24.0
    drones[2].y_m = 8.1
    drones[2].tracking_locked = False
    drones[2].last_track_update_s = 1.0
    drones[2].last_update_s = 1.0
    drones[2].last_heartbeat_s = 1.0

    swarm.update_many(drones)
    decision_request, _ = swarm.step(timestamp_s=1.0)

    assert decision_request.handover_state == "REQUESTED"
    assert decision_request.target_handover_required is True
    assert decision_request.target_handover_complete is False
    assert decision_request.pending_target_owner_id == "drone_2"

    drones[0].last_update_s = 3.2
    drones[0].last_heartbeat_s = 3.2
    drones[1].last_update_s = 3.2
    drones[1].last_heartbeat_s = 3.2
    drones[2].last_update_s = 3.2
    drones[2].last_heartbeat_s = 3.2

    swarm.update_many(drones)
    decision_timeout, bt_flags = swarm.step(timestamp_s=3.2)

    assert decision_timeout.target_owner_id == "drone_3"
    assert decision_timeout.pending_target_owner_id is None
    assert decision_timeout.handover_state == "TIMED_OUT"
    assert decision_timeout.target_handover_required is False
    assert decision_timeout.target_handover_complete is False
    assert bt_flags["target_handover_required"] is False
    assert bt_flags["target_handover_complete"] is False


def test_owner_lock_blocks_immediate_rehandover():
    swarm = SwarmCoordinationInterface()
    drones = build_initial_swarm()
    swarm.update_many(drones)
    swarm.step(timestamp_s=0.0)

    drones[0].last_update_s = 1.0
    drones[0].last_heartbeat_s = 1.0

    drones[1].x_m = 20.0
    drones[1].y_m = 8.0
    drones[1].target_confidence = 0.92
    drones[1].handover_ack = True
    drones[1].last_update_s = 1.0
    drones[1].last_heartbeat_s = 1.0

    drones[2].x_m = 24.0
    drones[2].y_m = 8.1
    drones[2].tracking_locked = False
    drones[2].last_track_update_s = 1.0
    drones[2].last_update_s = 1.0
    drones[2].last_heartbeat_s = 1.0

    swarm.update_many(drones)
    decision_handover, _ = swarm.step(timestamp_s=1.0)

    assert decision_handover.target_owner_id == "drone_2"
    assert decision_handover.target_owner_lock_active is True

    drones[0].last_update_s = 1.5
    drones[0].last_heartbeat_s = 1.5

    drones[1].x_m = 20.3
    drones[1].y_m = 8.0
    drones[1].handover_ack = False
    drones[1].tracking_locked = True
    drones[1].last_track_update_s = 1.5
    drones[1].last_update_s = 1.5
    drones[1].last_heartbeat_s = 1.5

    drones[2].x_m = 20.05
    drones[2].y_m = 8.0
    drones[2].target_confidence = 0.98
    drones[2].handover_ack = True
    drones[2].last_update_s = 1.5
    drones[2].last_heartbeat_s = 1.5

    swarm.update_many(drones)
    decision_locked, bt_flags = swarm.step(timestamp_s=1.5)

    assert decision_locked.target_owner_id == "drone_2"
    assert decision_locked.handover_state == "IDLE"
    assert decision_locked.target_handover_required is False
    assert decision_locked.target_handover_complete is False
    assert bt_flags["target_handover_required"] is False
    assert bt_flags["target_handover_complete"] is False