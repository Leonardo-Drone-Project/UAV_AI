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


def test_heartbeat_loss_marks_drone_invalid():
    swarm = SwarmCoordinationInterface()

    drones = [
        make_drone(
            "drone_1",
            battery_pct=99.0,
            last_update_s=2.0,
            last_heartbeat_s=0.0,
            target_detected=True,
            target_confidence=0.8,
        ),
        make_drone(
            "drone_2",
            battery_pct=88.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.7,
            last_update_s=2.0,
            last_heartbeat_s=2.0,
        ),
        make_drone("drone_3", battery_pct=80.0, last_update_s=2.0, last_heartbeat_s=2.0),
    ]

    swarm.update_many(drones)
    decision, bt_flags = swarm.step(timestamp_s=2.0)

    assert "drone_1" in decision.heartbeat_lost_drone_ids
    assert decision.parent_id == "drone_2"
    assert bt_flags["swarm_failure"] is False


def test_parent_timeout_causes_reassignment():
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
            battery_pct=88.0,
            x_m=20.0,
            y_m=8.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.9,
            last_update_s=0.0,
            last_heartbeat_s=0.0,
        ),
        make_drone("drone_3", battery_pct=82.0, x_m=24.0, y_m=8.0, last_update_s=0.0, last_heartbeat_s=0.0),
    ]

    swarm.update_many(drones_t0)
    decision_t0, _ = swarm.step(timestamp_s=0.0)

    assert decision_t0.parent_id == "drone_1"

    drones_t3 = [
        make_drone(
            "drone_2",
            battery_pct=88.0,
            x_m=20.0,
            y_m=8.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.9,
            tracking_locked=True,
            last_track_update_s=3.0,
            last_update_s=3.0,
            last_heartbeat_s=3.0,
        ),
        make_drone("drone_3", battery_pct=82.0, x_m=24.0, y_m=8.0, last_update_s=3.0, last_heartbeat_s=3.0),
    ]

    swarm.update_many(drones_t3)
    decision_t3, bt_flags_t3 = swarm.step(timestamp_s=3.0)

    assert decision_t3.parent_id == "drone_2"
    assert decision_t3.parent_lost is True
    assert decision_t3.parent_reassigned is True
    assert bt_flags_t3["parent_lost"] is True
    assert bt_flags_t3["parent_reassigned"] is True


def test_parent_reassignment_lock_blocks_immediate_handover_churn():
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
            battery_pct=88.0,
            x_m=20.0,
            y_m=8.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.90,
            tracking_locked=True,
            last_track_update_s=0.0,
            last_update_s=0.0,
            last_heartbeat_s=0.0,
        ),
        make_drone(
            "drone_3",
            battery_pct=82.0,
            x_m=25.0,
            y_m=8.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.95,
            tracking_locked=False,
            last_update_s=0.0,
            last_heartbeat_s=0.0,
        ),
    ]

    swarm.update_many(drones_t0)
    decision_t0, _ = swarm.step(timestamp_s=0.0)

    assert decision_t0.parent_id == "drone_1"
    assert decision_t0.target_owner_id == "drone_2"

    drones_t3 = [
        make_drone(
            "drone_2",
            battery_pct=88.0,
            x_m=20.0,
            y_m=8.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.90,
            tracking_locked=True,
            last_track_update_s=3.0,
            last_update_s=3.0,
            last_heartbeat_s=3.0,
        ),
        make_drone(
            "drone_3",
            battery_pct=82.0,
            x_m=19.9,
            y_m=8.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.99,
            handover_ack=True,
            last_update_s=3.0,
            last_heartbeat_s=3.0,
        ),
    ]

    swarm.update_many(drones_t3)
    decision_t3, _ = swarm.step(timestamp_s=3.0)

    assert decision_t3.parent_id == "drone_2"
    assert decision_t3.parent_reassigned is True
    assert decision_t3.target_owner_id == "drone_2"

    drones_t32 = [
        make_drone(
            "drone_2",
            battery_pct=88.0,
            x_m=20.1,
            y_m=8.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.90,
            tracking_locked=True,
            last_track_update_s=3.2,
            last_update_s=3.2,
            last_heartbeat_s=3.2,
        ),
        make_drone(
            "drone_3",
            battery_pct=82.0,
            x_m=19.8,
            y_m=8.0,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
            target_confidence=0.99,
            handover_ack=True,
            last_update_s=3.2,
            last_heartbeat_s=3.2,
        ),
    ]

    swarm.update_many(drones_t32)
    decision_t32, bt_flags_t32 = swarm.step(timestamp_s=3.2)

    assert decision_t32.target_owner_id == "drone_2"
    assert decision_t32.handover_state == "IDLE"
    assert decision_t32.target_handover_required is False
    assert decision_t32.target_handover_complete is False
    assert bt_flags_t32["target_handover_required"] is False