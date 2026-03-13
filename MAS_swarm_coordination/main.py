from .adapters import decision_to_command_messages, drone_from_payload
from .interface import SwarmCoordinationInterface


def print_decision(title: str, decision, bt_flags) -> None:
    print(title)
    print(f"Parent: {decision.parent_id}")
    print(f"Priority list: {decision.priority_list}")
    print(f"Assigned roles: {decision.assigned_roles}")
    print(f"Task assignments: {decision.task_assignments}")
    print(f"Target owner: {decision.target_owner_id}")
    print(f"Pending target owner: {decision.pending_target_owner_id}")
    print(f"Handover state: {decision.handover_state}")
    print(f"Target handover required: {decision.target_handover_required}")
    print(f"Target handover complete: {decision.target_handover_complete}")
    print(f"Target owner lock active: {decision.target_owner_lock_active}")
    print(f"Stale drones: {decision.stale_drone_ids}")
    print(f"Heartbeat lost drones: {decision.heartbeat_lost_drone_ids}")
    print(f"Deconfliction active: {decision.deconfliction_active}")
    print(f"Collision risk: {decision.collision_risk}")
    print(f"Deconfliction pairs: {decision.deconfliction_pairs}")
    print(f"Swarm degraded: {decision.swarm_degraded}")
    print(f"Swarm failure: {decision.swarm_failure}")
    print(f"Degraded reasons: {decision.degraded_reasons}")
    print(f"Failure reason: {decision.failure_reason}")
    print(f"Parent lost: {decision.parent_lost}")
    print(f"Parent reassigned: {decision.parent_reassigned}")
    print(f"Converge complete: {decision.converge_complete}")
    print(f"BT flags: {bt_flags}")
    print("Command messages:")
    for cmd in decision_to_command_messages(decision):
        print(
            f"  {cmd.drone_id} | role={cmd.role} | task={cmd.task} | "
            f"parent={cmd.parent_id} | target_owner={cmd.target_owner_id} | "
            f"handover={cmd.handover_state} | hold={cmd.hold_position} | "
            f"degraded={cmd.swarm_degraded} | failure={cmd.swarm_failure} | "
            f"reason={cmd.reason}"
        )
    print()


def main() -> None:
    swarm = SwarmCoordinationInterface()

    payloads = [
        {
            "drone_id": "drone_1",
            "timestamp_s": 0.0,
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 10.0,
            "battery_pct": 95.0,
            "comms_ok": True,
            "nav_ok": True,
            "target_detected": True,
            "target_x_m": 20.0,
            "target_y_m": 8.0,
            "target_confidence": 0.55,
            "tracking_locked": False,
            "last_heartbeat_s": 0.0,
            "last_update_s": 0.0,
        },
        {
            "drone_id": "drone_2",
            "timestamp_s": 0.0,
            "x_m": 17.0,
            "y_m": 8.0,
            "z_m": 10.0,
            "battery_pct": 88.0,
            "comms_ok": True,
            "nav_ok": True,
            "target_detected": True,
            "target_x_m": 20.0,
            "target_y_m": 8.0,
            "target_confidence": 0.70,
            "tracking_locked": False,
            "last_heartbeat_s": 0.0,
            "last_update_s": 0.0,
        },
        {
            "drone_id": "drone_3",
            "timestamp_s": 0.0,
            "x_m": 20.4,
            "y_m": 8.1,
            "z_m": 10.0,
            "battery_pct": 82.0,
            "comms_ok": True,
            "nav_ok": True,
            "target_detected": True,
            "target_x_m": 20.0,
            "target_y_m": 8.0,
            "target_confidence": 0.86,
            "tracking_locked": True,
            "last_track_update_s": 0.0,
            "last_heartbeat_s": 0.0,
            "last_update_s": 0.0,
        },
    ]

    swarm.update_many([drone_from_payload(p) for p in payloads])
    decision, bt_flags = swarm.step(timestamp_s=0.0)
    print_decision("Initial decision", decision, bt_flags)

    # Clean handover request. No deconfliction. No parent loss.
    payloads[0]["timestamp_s"] = 1.0
    payloads[0]["last_heartbeat_s"] = 1.0
    payloads[0]["last_update_s"] = 1.0

    payloads[1]["timestamp_s"] = 1.0
    payloads[1]["x_m"] = 20.0
    payloads[1]["y_m"] = 8.0
    payloads[1]["target_confidence"] = 0.92
    payloads[1]["handover_ack"] = True
    payloads[1]["last_heartbeat_s"] = 1.0
    payloads[1]["last_update_s"] = 1.0

    payloads[2]["timestamp_s"] = 1.0
    payloads[2]["x_m"] = 24.0
    payloads[2]["y_m"] = 8.1
    payloads[2]["tracking_locked"] = False
    payloads[2]["last_track_update_s"] = 1.0
    payloads[2]["last_heartbeat_s"] = 1.0
    payloads[2]["last_update_s"] = 1.0

    swarm.update_many([drone_from_payload(p) for p in payloads])
    decision, bt_flags = swarm.step(timestamp_s=1.0)
    print_decision("After clean handover", decision, bt_flags)

    # Deconfliction scenario after owner lock.
    payloads[0]["timestamp_s"] = 3.0
    payloads[0]["last_heartbeat_s"] = 3.0
    payloads[0]["last_update_s"] = 3.0

    payloads[1]["timestamp_s"] = 3.0
    payloads[1]["x_m"] = 20.0
    payloads[1]["y_m"] = 8.0
    payloads[1]["tracking_locked"] = True
    payloads[1]["last_track_update_s"] = 3.0
    payloads[1]["handover_ack"] = False
    payloads[1]["last_heartbeat_s"] = 3.0
    payloads[1]["last_update_s"] = 3.0

    payloads[2]["timestamp_s"] = 3.0
    payloads[2]["x_m"] = 20.9
    payloads[2]["y_m"] = 8.1
    payloads[2]["last_heartbeat_s"] = 3.0
    payloads[2]["last_update_s"] = 3.0

    swarm.update_many([drone_from_payload(p) for p in payloads])
    decision, bt_flags = swarm.step(timestamp_s=3.0)
    print_decision("After deconfliction scenario", decision, bt_flags)

    # Parent timeout later, separated from handover.
    payloads[1]["timestamp_s"] = 6.0
    payloads[1]["last_heartbeat_s"] = 6.0
    payloads[1]["last_update_s"] = 6.0

    payloads[2]["timestamp_s"] = 6.0
    payloads[2]["last_heartbeat_s"] = 6.0
    payloads[2]["last_update_s"] = 6.0

    swarm.update_many([drone_from_payload(p) for p in payloads[1:]])
    decision, bt_flags = swarm.step(timestamp_s=6.5)
    print_decision("After parent timeout", decision, bt_flags)


if __name__ == "__main__":
    main()