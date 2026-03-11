import math
import random

from .interface import EKFSensorFusionInterface, NavThresholds
from .measurements import GPSMeasurement, IMUMeasurement, RealSenseMeasurement


def main() -> None:
    ekf = EKFSensorFusionInterface()

    dt = 0.02
    sim_time_s = 12.0

    t = 0.0
    x_true = 0.0
    y_true = 0.0
    z_true = 5.0
    vx_true = 1.0
    vy_true = 0.0
    vz_true = 0.0
    yaw_true = 0.0

    last_gps_t = -999.0
    last_rs_t = -999.0

    print("Starting EKF demo...")

    while t <= sim_time_s:
        # Simple motion model for the demo
        ax_true = 0.0
        ay_true = 0.15 * math.sin(0.5 * t)
        az_true = 0.0

        vx_true += ax_true * dt
        vy_true += ay_true * dt
        vz_true += az_true * dt

        x_true += vx_true * dt
        y_true += vy_true * dt
        z_true += vz_true * dt

        yaw_true = 0.08 * math.sin(0.25 * t)

        imu = IMUMeasurement(
            timestamp_s=t,
            accel_xyz_mps2=(
                ax_true + random.gauss(0.0, 0.08),
                ay_true + random.gauss(0.0, 0.08),
                az_true + random.gauss(0.0, 0.08),
            ),
            gyro_xyz_radps=(
                random.gauss(0.0, 0.01),
                random.gauss(0.0, 0.01),
                0.08 * 0.25 * math.cos(0.25 * t) + random.gauss(0.0, 0.01),
            ),
        )
        ekf.push_imu(imu, dt_s=dt)

        if (t - last_gps_t) >= 0.20:
            gps = GPSMeasurement(
                timestamp_s=t,
                position_xyz_m=(
                    x_true + random.gauss(0.0, 1.2),
                    y_true + random.gauss(0.0, 1.2),
                    z_true + random.gauss(0.0, 0.8),
                ),
            )
            ekf.push_gps(gps)
            last_gps_t = t

        if (t - last_rs_t) >= 0.10:
            rs = RealSenseMeasurement(
                timestamp_s=t,
                position_xyz_m=(
                    x_true + random.gauss(0.0, 0.10),
                    y_true + random.gauss(0.0, 0.10),
                    z_true + random.gauss(0.0, 0.08),
                ),
                velocity_xyz_mps=(
                    vx_true + random.gauss(0.0, 0.05),
                    vy_true + random.gauss(0.0, 0.05),
                    vz_true + random.gauss(0.0, 0.05),
                ),
                yaw_rad=yaw_true + random.gauss(0.0, 0.03),
            )
            ekf.push_realsense(rs)
            last_rs_t = t

        if int(t / dt) % 50 == 0:
            fused = ekf.get_fused_state(timestamp_s=t)
            print(
                f"t={t:5.2f}s | "
                f"pos=({fused.x_m:6.2f}, {fused.y_m:6.2f}, {fused.z_m:5.2f}) | "
                f"vel=({fused.vx_mps:5.2f}, {fused.vy_mps:5.2f}, {fused.vz_mps:5.2f}) | "
                f"yaw={fused.yaw_rad:5.2f} | nav_ok={int(fused.nav_ok)}"
            )

        t += dt

    fused = ekf.get_fused_state(timestamp_s=t)
    flags = ekf.derive_bt_navigation_flags(
        thresholds=NavThresholds(),
        altitude_target_m=5.0,
        search_area_xy_m=(10.0, 0.0),
        base_xy_m=(0.0, 0.0),
        timestamp_s=t,
    )

    print("\nFinal fused state")
    print(f"Position xyz: {fused.position_xyz()}")
    print(f"Velocity xyz: {fused.velocity_xyz()}")
    print(f"Attitude rpy: {fused.attitude_rpy()}")
    print(f"Position variance: {fused.position_var_xyz}")
    print(f"Velocity variance: {fused.velocity_var_xyz}")
    print(f"Nav OK: {fused.nav_ok}")
    print(f"Derived BT flags: {flags}")


if __name__ == "__main__":
    main()