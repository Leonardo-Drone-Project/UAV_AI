from .adapters import drone_from_payload
from .interface import SwarmCoordinationInterface


def print_decision(title: str, decision, bt_flags) -> None:
    print(title)
    print(f"Parent: {decision.parent_id}")
    print(f"Priority list: {decision.priority_list}")
    print(f"Assigned roles: {decision.assigned_roles}")
    print(f"Task assignments: {decision.task_assignments}")
    print(f"Target owner: {decision.target_owner_id}")
    print(f"Target handover required: {decision.target_handover_required}")
    print(f"Target handover complete: {decision.target_handover_complete}")
    print(f"Stale drones: {decision.stale_drone_ids}")
    print(f"Parent lost: {decision.parent_lost}")
    print(f"Parent reassigned: {decision.parent_reassigned}")
    print(f"Converge complete: {decision.converge_complete}")
    print(f"BT flags: {bt_flags}")
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
        },
        {
            "drone_id": "drone_2",
            "timestamp_s": 0.0,
            "x_m": 10.0,
            "y_m": 4.0,
            "z_m": 10.0,
            "battery_pct": 88.0,
            "comms_ok": True,
            "nav_ok": True,
            "target_detected": True,
            "target_x_m": 20.0,
            "target_y_m": 8.0,
        },
        {
            "drone_id": "drone_3",
            "timestamp_s": 0.0,
            "x_m": 14.0,
            "y_m": 5.0,
            "z_m": 10.0,
            "battery_pct": 82.0,
            "comms_ok": True,
            "nav_ok": True,
            "target_detected": False,
        },
    ]

    swarm.update_many([drone_from_payload(p) for p in payloads])
    decision, bt_flags = swarm.step(timestamp_s=0.0)
    print_decision("Initial decision", decision, bt_flags)

    # Make drone_2 clearly better for tracking so handover happens from drone_3 to drone_2
    payloads[1]["timestamp_s"] = 1.0
    payloads[1]["x_m"] = 20.2
    payloads[1]["y_m"] = 8.1

    payloads[2]["timestamp_s"] = 1.0
    payloads[2]["x_m"] = 22.0
    payloads[2]["y_m"] = 8.0
    payloads[2]["target_detected"] = True
    payloads[2]["target_x_m"] = 20.0
    payloads[2]["target_y_m"] = 8.0

    swarm.update_many([drone_from_payload(p) for p in payloads])
    decision, bt_flags = swarm.step(timestamp_s=1.0)
    print_decision("After target handover scenario", decision, bt_flags)

    # Simulate parent timeout by leaving drone_1 stale
    payloads[1]["timestamp_s"] = 4.0
    payloads[2]["timestamp_s"] = 4.0

    swarm.update_many([drone_from_payload(p) for p in payloads[1:]])
    decision, bt_flags = swarm.step(timestamp_s=4.5)
    print_decision("After parent timeout", decision, bt_flags)


if __name__ == "__main__":
    main()