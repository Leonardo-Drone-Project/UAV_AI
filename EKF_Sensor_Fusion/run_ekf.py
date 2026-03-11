import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from .interface import EKFSensorFusionInterface, NavThresholds
from .sensor_adapters import (
    LocalFrameConverter,
    gps_from_payload,
    imu_from_payload,
    realsense_from_payload,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay timestamped sensor data through the EKF")

    p.add_argument("--input", type=str, required=True, help="Path to newline-delimited JSON replay file")
    p.add_argument("--log-path", type=str, default="ekf_log.csv", help="CSV output log path")
    p.add_argument("--print-every", type=int, default=25, help="Print fused state every N events")

    p.add_argument("--ref-lat-deg", type=float, default=None, help="Reference latitude for GPS lat/lon conversion")
    p.add_argument("--ref-lon-deg", type=float, default=None, help="Reference longitude for GPS lat/lon conversion")
    p.add_argument("--ref-alt-m", type=float, default=0.0, help="Reference altitude for GPS lat/lon conversion")

    p.add_argument("--altitude-target-m", type=float, default=None)
    p.add_argument("--search-x-m", type=float, default=None)
    p.add_argument("--search-y-m", type=float, default=None)
    p.add_argument("--base-x-m", type=float, default=None)
    p.add_argument("--base-y-m", type=float, default=None)

    return p


def open_converter(args) -> Optional[LocalFrameConverter]:
    if args.ref_lat_deg is None or args.ref_lon_deg is None:
        return None
    return LocalFrameConverter(
        ref_lat_deg=args.ref_lat_deg,
        ref_lon_deg=args.ref_lon_deg,
        ref_alt_m=args.ref_alt_m,
    )


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input).expanduser().resolve()
    log_path = Path(args.log_path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    converter = open_converter(args)
    ekf = EKFSensorFusionInterface()

    search_xy = None
    if args.search_x_m is not None and args.search_y_m is not None:
        search_xy = (float(args.search_x_m), float(args.search_y_m))

    base_xy = None
    if args.base_x_m is not None and args.base_y_m is not None:
        base_xy = (float(args.base_x_m), float(args.base_y_m))

    event_count = 0
    last_ts = 0.0

    print(f"[INFO] Replaying EKF input from: {input_path}")
    print(f"[INFO] Logging fused output to: {log_path}")

    with input_path.open("r", encoding="utf-8") as fin, log_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "timestamp_s",
                "event_type",
                "x_m",
                "y_m",
                "z_m",
                "vx_mps",
                "vy_mps",
                "vz_mps",
                "roll_rad",
                "pitch_rad",
                "yaw_rad",
                "nav_ok",
                "altitude_reached",
                "hover_stable",
                "arrived_search_area",
                "at_base",
                "var_x",
                "var_y",
                "var_z",
                "var_vx",
                "var_vy",
                "var_vz",
            ]
        )

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            payload = json.loads(line)
            event_type = str(payload.get("type", "")).strip().lower()

            if event_type == "imu":
                meas = imu_from_payload(payload)
                ekf.push_imu(meas)
                last_ts = meas.timestamp_s

            elif event_type == "gps":
                meas = gps_from_payload(payload, converter=converter)
                ekf.push_gps(meas)
                last_ts = meas.timestamp_s

            elif event_type in {"realsense", "vision"}:
                meas = realsense_from_payload(payload)
                ekf.push_realsense(meas)
                last_ts = meas.timestamp_s

            else:
                raise ValueError(f"Unknown event type '{event_type}' on line {line_no}")

            fused = ekf.get_fused_state(timestamp_s=last_ts)
            flags = ekf.derive_bt_navigation_flags(
                thresholds=NavThresholds(),
                altitude_target_m=args.altitude_target_m,
                search_area_xy_m=search_xy,
                base_xy_m=base_xy,
                timestamp_s=last_ts,
            )

            writer.writerow(
                [
                    round(fused.timestamp_s, 6),
                    event_type,
                    round(fused.x_m, 6),
                    round(fused.y_m, 6),
                    round(fused.z_m, 6),
                    round(fused.vx_mps, 6),
                    round(fused.vy_mps, 6),
                    round(fused.vz_mps, 6),
                    round(fused.roll_rad, 6),
                    round(fused.pitch_rad, 6),
                    round(fused.yaw_rad, 6),
                    int(fused.nav_ok),
                    int(flags["altitude_reached"]),
                    int(flags["hover_stable"]),
                    int(flags["arrived_search_area"]),
                    int(flags["at_base"]),
                    round(fused.position_var_xyz[0], 6),
                    round(fused.position_var_xyz[1], 6),
                    round(fused.position_var_xyz[2], 6),
                    round(fused.velocity_var_xyz[0], 6),
                    round(fused.velocity_var_xyz[1], 6),
                    round(fused.velocity_var_xyz[2], 6),
                ]
            )

            event_count += 1
            if event_count % max(1, args.print_every) == 0:
                print(
                    f"[INFO] event={event_count} type={event_type} "
                    f"pos=({fused.x_m:.2f}, {fused.y_m:.2f}, {fused.z_m:.2f}) "
                    f"vel=({fused.vx_mps:.2f}, {fused.vy_mps:.2f}, {fused.vz_mps:.2f}) "
                    f"yaw={fused.yaw_rad:.2f} nav_ok={int(fused.nav_ok)}"
                )

    final_state = ekf.get_fused_state(timestamp_s=last_ts)
    final_flags = ekf.derive_bt_navigation_flags(
        thresholds=NavThresholds(),
        altitude_target_m=args.altitude_target_m,
        search_area_xy_m=search_xy,
        base_xy_m=base_xy,
        timestamp_s=last_ts,
    )

    print("\n[INFO] Replay complete")
    print(f"[INFO] Events processed: {event_count}")
    print(f"[INFO] Final position: {final_state.position_xyz()}")
    print(f"[INFO] Final velocity: {final_state.velocity_xyz()}")
    print(f"[INFO] Final attitude: {final_state.attitude_rpy()}")
    print(f"[INFO] Nav OK: {final_state.nav_ok}")
    print(f"[INFO] Derived BT flags: {final_flags}")


if __name__ == "__main__":
    main()