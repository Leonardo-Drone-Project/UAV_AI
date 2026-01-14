# BT_decision_making/main.py
import csv
import time
import cv2

from blackboard import Blackboard
from inputs import get_inputs
from bt import tick_bt


def run_bt(
    *,
    scenario: int = 1,
    tick_hz: float = 10.0,
    max_runtime_s: float = 30.0,
    log_path: str = "bt_log.csv",
):
    """
    Runs the BT loop with:
    - ESC to quit immediately
    - max runtime as backup
    - CSV logging of states + transitions
    """
    bb = Blackboard()
    dt = 1.0 / tick_hz

    print(f"Running BT scenario={scenario} at {tick_hz:.1f} Hz (ESC to quit).")
    print(f"Logging to: {log_path}")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_s", "state", "transition",
            "target_detected", "target_conf", "dx_px", "dy_px",
            "yaw_rate_cmd", "forward_vel_cmd", "hold_position"
        ])

        while True:
            now_s = time.time()
            elapsed = now_s - bb.start_time_s

            # Quit conditions
            if elapsed >= max_runtime_s:
                print("Max runtime reached. Exiting.")
                break

            # ESC quit (use waitKey even without a window)
            if (cv2.waitKey(1) & 0xFF) == 27:
                print("ESC pressed. Exiting.")
                break

            # Get stub inputs + tick BT
            inp = get_inputs(now_s, scenario=scenario)
            out, transition = tick_bt(bb, inp)

            # Console output (only print transitions + periodic status)
            if transition:
                print(f"[TRANSITION] {transition}")

            if bb.tick % int(max(1, tick_hz)) == 0:  # ~once per second
                print(str(out))

            # Log row
            dx, dy = inp.target_offset_px
            writer.writerow([
                round(elapsed, 3),
                bb.state,
                transition,
                int(inp.target_detected),
                round(inp.target_conf, 3),
                round(dx, 2),
                round(dy, 2),
                round(out.yaw_rate_cmd, 3),
                round(out.forward_vel_cmd, 3),
                int(out.hold_position),
            ])

            bb.tick += 1
            time.sleep(dt)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Change scenario = 1,2,3 to demonstrate different behaviours
    run_bt(scenario=1, tick_hz=10.0, max_runtime_s=30.0, log_path="bt_log.csv")

