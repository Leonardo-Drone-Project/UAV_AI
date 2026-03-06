# BT_decision_making/main.py
import argparse
import csv
import time
from pathlib import Path

from BT_decision_making.blackboard import Blackboard
from BT_decision_making.inputs import Inputs
from BT_decision_making.bt import tick_mission
from BT_decision_making.perception_client import PerceptionClient


def apply_demo_autostart(inp: Inputs) -> None:
    # Drives the mission flow into SEARCH_AREA for local AI-side tests.
    inp.mission_upload_received = True
    inp.mission_data_complete = True
    inp.swarm_coordinated = True

    inp.roles_assigned = True
    inp.priority_list_sent = True

    inp.launch_command_received = True
    inp.arm_permission_received = True

    inp.altitude_reached = True
    inp.hover_stable = True
    inp.arrived_search_area = True

    # Set a role for demo logs
    inp.role = "child"


def run_bt(
    *,
    tick_hz: float,
    max_runtime_s: float,
    log_path: str,
    perception_json: str,
    perception_max_age_s: float,
    demo_autostart: bool,
):
    bb = Blackboard()
    dt = 1.0 / tick_hz

    repo_root = Path(__file__).resolve().parents[1]
    default_json = repo_root / "perception" / "outputs" / "latest_detection.json"
    json_path = Path(perception_json).expanduser().resolve() if perception_json else default_json

    percep = PerceptionClient(json_path=json_path, max_age_s=perception_max_age_s)

    print(f"[INFO] tick_hz={tick_hz:.1f}")
    print(f"[INFO] log_path={log_path}")
    print(f"[INFO] perception_json={json_path}")
    print(f"[INFO] demo_autostart={int(demo_autostart)}")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t_s",
                "mission_mode",
                "target_state",
                "transition",
                "role",
                "detected",
                "conf",
                "cls",
                "dx_px",
                "dy_px",
                "dx_norm",
                "dy_norm",
                "yaw_rate_cmd",
                "forward_vel_cmd",
                "hold_position",
                "last_dx_px",
                "last_dy_px",
            ]
        )

        t0 = time.time()

        try:
            while True:
                now = time.time()
                elapsed = now - t0

                if elapsed >= max_runtime_s:
                    print("[INFO] Max runtime reached. Exiting.")
                    break

                inp = Inputs()

                if demo_autostart:
                    apply_demo_autostart(inp)

                s = percep.read()
                if s is not None:
                    inp.target_detected = bool(s.detected)
                    inp.target_conf = float(s.conf)
                    inp.target_class = str(s.cls)
                    inp.target_offset_px = (float(s.offset_px[0]), float(s.offset_px[1]))
                    inp.target_offset_norm = (float(s.offset_norm[0]), float(s.offset_norm[1]))

                out, transition = tick_mission(bb, inp)

                if transition:
                    print(f"[TRANSITION] {transition}")

                if bb.tick % int(max(1, tick_hz)) == 0:
                    print(str(out))

                dx_px, dy_px = inp.target_offset_px
                dx_n, dy_n = inp.target_offset_norm
                ldx, ldy = bb.last_target_offset_px

                writer.writerow(
                    [
                        round(elapsed, 3),
                        bb.mode,
                        bb.target_state,
                        transition,
                        inp.role,
                        int(inp.target_detected),
                        round(inp.target_conf, 3),
                        inp.target_class,
                        round(dx_px, 2),
                        round(dy_px, 2),
                        round(dx_n, 3),
                        round(dy_n, 3),
                        round(out.yaw_rate_cmd, 3),
                        round(out.forward_vel_cmd, 3),
                        int(out.hold_position),
                        round(ldx, 2),
                        round(ldy, 2),
                    ]
                )

                bb.tick += 1
                time.sleep(dt)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tick-hz", type=float, default=10.0)
    p.add_argument("--max-runtime-s", type=float, default=60.0)
    p.add_argument("--log-path", type=str, default="bt_log.csv")
    p.add_argument("--perception-json", type=str, default="")
    p.add_argument("--perception-max-age-s", type=float, default=0.5)
    p.add_argument("--demo-autostart", action="store_true")
    args = p.parse_args()

    run_bt(
        tick_hz=args.tick_hz,
        max_runtime_s=args.max_runtime_s,
        log_path=args.log_path,
        perception_json=args.perception_json,
        perception_max_age_s=args.perception_max_age_s,
        demo_autostart=args.demo_autostart,
    )