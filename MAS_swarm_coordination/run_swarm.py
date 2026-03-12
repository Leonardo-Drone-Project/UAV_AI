import argparse
import csv
import json
from pathlib import Path

from .adapters import drone_from_payload
from .interface import SwarmCoordinationInterface


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay drone state updates through the swarm coordinator")
    parser.add_argument("--input", type=str, required=True, help="Path to newline-delimited JSON replay file")
    parser.add_argument("--log-path", type=str, default="swarm_log.csv", help="CSV output log path")
    parser.add_argument("--print-every", type=int, default=1, help="Print every N processed events")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    log_path = Path(args.log_path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    swarm = SwarmCoordinationInterface()
    event_count = 0
    last_ts = 0.0

    print(f"[INFO] Replaying swarm input from: {input_path}")
    print(f"[INFO] Logging swarm output to: {log_path}")

    with input_path.open("r", encoding="utf-8") as fin, log_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "timestamp_s",
                "drone_id",
                "parent_id",
                "priority_list",
                "assigned_roles",
                "task_assignments",
                "stale_drone_ids",
                "swarm_coordinated",
                "roles_assigned",
                "priority_list_sent",
                "role_election_failed",
                "parent_lost",
                "parent_reassigned",
                "converge_complete",
                "target_known",
                "target_xy_m",
            ]
        )

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            payload = json.loads(line)
            drone = drone_from_payload(payload)
            timestamp_s = float(payload.get("timestamp_s", payload.get("timestamp", drone.last_update_s)))

            swarm.update_drone(drone)
            decision, bt_flags = swarm.step(timestamp_s=timestamp_s)
            last_ts = timestamp_s
            event_count += 1

            writer.writerow(
                [
                    round(timestamp_s, 6),
                    drone.drone_id,
                    decision.parent_id,
                    "|".join(decision.priority_list),
                    json.dumps(decision.assigned_roles),
                    json.dumps(decision.task_assignments),
                    "|".join(decision.stale_drone_ids),
                    int(decision.swarm_coordinated),
                    int(decision.roles_assigned),
                    int(decision.priority_list_sent),
                    int(decision.role_election_failed),
                    int(decision.parent_lost),
                    int(decision.parent_reassigned),
                    int(decision.converge_complete),
                    int(decision.target_known),
                    "" if decision.target_xy_m is None else f"{decision.target_xy_m[0]:.3f},{decision.target_xy_m[1]:.3f}",
                ]
            )

            if event_count % max(1, args.print_every) == 0:
                print(
                    f"[INFO] event={event_count} drone={drone.drone_id} "
                    f"parent={decision.parent_id} "
                    f"reassigned={int(decision.parent_reassigned)} "
                    f"converge={int(decision.converge_complete)} "
                    f"bt_flags={bt_flags}"
                )

    final_decision, final_bt_flags = swarm.step(timestamp_s=last_ts)

    print("\n[INFO] Replay complete")
    print(f"[INFO] Events processed: {event_count}")
    print(f"[INFO] Final parent: {final_decision.parent_id}")
    print(f"[INFO] Final priority list: {final_decision.priority_list}")
    print(f"[INFO] Final roles: {final_decision.assigned_roles}")
    print(f"[INFO] Final tasks: {final_decision.task_assignments}")
    print(f"[INFO] Final BT flags: {final_bt_flags}")


if __name__ == "__main__":
    main()