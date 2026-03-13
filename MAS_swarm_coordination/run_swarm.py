import argparse
import csv
import json
from pathlib import Path

from .adapters import decision_to_command_messages, drone_from_payload
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
                "target_owner_id",
                "pending_target_owner_id",
                "handover_state",
                "target_handover_required",
                "target_handover_complete",
                "stale_drone_ids",
                "heartbeat_lost_drone_ids",
                "deconfliction_active",
                "collision_risk",
                "deconfliction_pairs",
                "swarm_coordinated",
                "swarm_degraded",
                "swarm_failure",
                "degraded_reasons",
                "failure_reason",
                "parent_lost",
                "parent_reassigned",
                "converge_complete",
                "target_known",
                "target_xy_m",
                "command_messages",
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
            commands = [cmd.to_dict() for cmd in decision_to_command_messages(decision)]

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
                    decision.target_owner_id,
                    decision.pending_target_owner_id,
                    decision.handover_state,
                    int(decision.target_handover_required),
                    int(decision.target_handover_complete),
                    "|".join(decision.stale_drone_ids),
                    "|".join(decision.heartbeat_lost_drone_ids),
                    int(decision.deconfliction_active),
                    int(decision.collision_risk),
                    json.dumps(decision.deconfliction_pairs),
                    int(decision.swarm_coordinated),
                    int(decision.swarm_degraded),
                    int(decision.swarm_failure),
                    "|".join(decision.degraded_reasons),
                    decision.failure_reason,
                    int(decision.parent_lost),
                    int(decision.parent_reassigned),
                    int(decision.converge_complete),
                    int(decision.target_known),
                    "" if decision.target_xy_m is None else f"{decision.target_xy_m[0]:.3f},{decision.target_xy_m[1]:.3f}",
                    json.dumps(commands),
                ]
            )

            if event_count % max(1, args.print_every) == 0:
                print(
                    f"[INFO] event={event_count} drone={drone.drone_id} "
                    f"parent={decision.parent_id} "
                    f"target_owner={decision.target_owner_id} "
                    f"handover_state={decision.handover_state} "
                    f"deconflict={int(decision.deconfliction_active)} "
                    f"degraded={int(decision.swarm_degraded)} "
                    f"failure={int(decision.swarm_failure)} "
                    f"bt_flags={bt_flags}"
                )

    final_decision, final_bt_flags = swarm.step(timestamp_s=last_ts)

    print("\n[INFO] Replay complete")
    print(f"[INFO] Events processed: {event_count}")
    print(f"[INFO] Final parent: {final_decision.parent_id}")
    print(f"[INFO] Final priority list: {final_decision.priority_list}")
    print(f"[INFO] Final roles: {final_decision.assigned_roles}")
    print(f"[INFO] Final tasks: {final_decision.task_assignments}")
    print(f"[INFO] Final target owner: {final_decision.target_owner_id}")
    print(f"[INFO] Final handover state: {final_decision.handover_state}")
    print(f"[INFO] Final deconfliction active: {final_decision.deconfliction_active}")
    print(f"[INFO] Final degraded: {final_decision.swarm_degraded}")
    print(f"[INFO] Final failure: {final_decision.swarm_failure}")
    print(f"[INFO] Final BT flags: {final_bt_flags}")


if __name__ == "__main__":
    main()