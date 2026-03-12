import time

from .interface import SwarmCoordinationInterface
from .models import DroneState


def main() -> None:
    swarm = SwarmCoordinationInterface()

    drones = [
        DroneState(
            drone_id="drone_1",
            x_m=0.0,
            y_m=0.0,
            z_m=10.0,
            battery_pct=95.0,
            comms_ok=True,
            nav_ok=True,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
        ),
        DroneState(
            drone_id="drone_2",
            x_m=10.0,
            y_m=4.0,
            z_m=10.0,
            battery_pct=88.0,
            comms_ok=True,
            nav_ok=True,
            target_detected=True,
            target_x_m=20.0,
            target_y_m=8.0,
        ),
        DroneState(
            drone_id="drone_3",
            x_m=14.0,
            y_m=5.0,
            z_m=10.0,
            battery_pct=82.0,
            comms_ok=True,
            nav_ok=True,
            target_detected=False,
        ),
    ]

    swarm.update_many(drones)
    decision, bt_flags = swarm.step(timestamp_s=0.0)

    print("Initial decision")
    print(f"Parent: {decision.parent_id}")
    print(f"Priority list: {decision.priority_list}")
    print(f"Assigned roles: {decision.assigned_roles}")
    print(f"BT flags: {bt_flags}")
    print()

    # Simulate child drones converging on the target
    drones[1].x_m, drones[1].y_m = 19.0, 8.5
    drones[2].x_m, drones[2].y_m = 20.5, 7.5
    drones[2].target_detected = True
    drones[2].target_x_m = 20.0
    drones[2].target_y_m = 8.0

    swarm.update_many(drones)
    decision, bt_flags = swarm.step(timestamp_s=1.0)

    print("After converge movement")
    print(f"Parent: {decision.parent_id}")
    print(f"Priority list: {decision.priority_list}")
    print(f"Assigned roles: {decision.assigned_roles}")
    print(f"Converge complete: {decision.converge_complete}")
    print(f"BT flags: {bt_flags}")
    print()

    # Simulate parent loss
    drones[0].comms_ok = False
    swarm.update_many(drones)
    decision, bt_flags = swarm.step(timestamp_s=2.0)

    print("After parent loss")
    print(f"Parent: {decision.parent_id}")
    print(f"Priority list: {decision.priority_list}")
    print(f"Assigned roles: {decision.assigned_roles}")
    print(f"Parent lost: {decision.parent_lost}")
    print(f"Parent reassigned: {decision.parent_reassigned}")
    print(f"BT flags: {bt_flags}")


if __name__ == "__main__":
    main()