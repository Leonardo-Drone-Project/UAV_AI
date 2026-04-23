import math
import random
from statistics import mean

from EKF_Sensor_Fusion.interface import EKFSensorFusionInterface, NavThresholds
from EKF_Sensor_Fusion.measurements import GPSMeasurement, IMUMeasurement, RealSenseMeasurement


def rmse(values):
    if not values:
        return 0.0
    return math.sqrt(sum(v * v for v in values) / len(values))


def main():
    random.seed(42)

    ekf = EKFSensorFusionInterface()

    dt = 0.02
    sim_time_s = 20.0

    t = 0.0

    # Ground truth state
    x_true = 0.0
    y_true = 0.0
    z_true = 5.0

    vx_true = 1.0
    vy_true = 0.0
    vz_true = 0.0

    yaw_true = 0.0

    last_gps_t = -999.0
    last_rs_t = -999.0

    gps_pos_errors = []
    ekf_pos_errors = []

    gps_x_errors = []
    gps_y_errors = []
    gps_z_errors = []

    ekf_x_errors = []
    ekf_y_errors = []
    ekf_z_errors = []

    nav_ok_count = 0
    total_steps = 0

    altitude_reached_count = 0
    hover_stable_count = 0
    arrived_search_area_count = 0
    at_base_count = 0

    final_flags = {}

    while t <= sim_time_s:
        # Smooth truth trajectory
        ax_true = 0.0
        ay_true = 0.12 * math.sin(0.4 * t)
        az_true = 0.02 * math.sin(0.25 * t)

        vx_true += ax_true * dt
        vy_true += ay_true * dt
        vz_true += az_true * dt

        x_true += vx_true * dt
        y_true += vy_true * dt
        z_true += vz_true * dt

        yaw_true = 0.10 * math.sin(0.2 * t)

        # IMU at high rate
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
                0.10 * 0.2 * math.cos(0.2 * t) + random.gauss(0.0, 0.01),
            ),
        )
        ekf.push_imu(imu, dt_s=dt)

        # GPS at lower rate, noisy
        gps_meas = None
        if (t - last_gps_t) >= 0.20:
            gps_meas = GPSMeasurement(
                timestamp_s=t,
                position_xyz_m=(
                    x_true + random.gauss(0.0, 1.2),
                    y_true + random.gauss(0.0, 1.2),
                    z_true + random.gauss(0.0, 0.8),
                ),
            )
            ekf.push_gps(gps_meas)
            last_gps_t = t

        # RealSense at medium rate, higher accuracy
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

        fused = ekf.get_fused_state(timestamp_s=t)
        flags = ekf.derive_bt_navigation_flags(
            thresholds=NavThresholds(),
            altitude_target_m=5.0,
            search_area_xy_m=(10.0, 0.0),
            base_xy_m=(0.0, 0.0),
            timestamp_s=t,
        )
        final_flags = flags

        # EKF fused error against truth
        ex = fused.x_m - x_true
        ey = fused.y_m - y_true
        ez = fused.z_m - z_true

        ekf_x_errors.append(ex)
        ekf_y_errors.append(ey)
        ekf_z_errors.append(ez)
        ekf_pos_errors.append(math.sqrt(ex * ex + ey * ey + ez * ez))

        # GPS error only when a GPS measurment exists
        if gps_meas is not None:
            gx = gps_meas.position_xyz_m[0] - x_true
            gy = gps_meas.position_xyz_m[1] - y_true
            gz = gps_meas.position_xyz_m[2] - z_true

            gps_x_errors.append(gx)
            gps_y_errors.append(gy)
            gps_z_errors.append(gz)
            gps_pos_errors.append(math.sqrt(gx * gx + gy * gy + gz * gz))

        if fused.nav_ok:
            nav_ok_count += 1
        if flags["altitude_reached"]:
            altitude_reached_count += 1
        if flags["hover_stable"]:
            hover_stable_count += 1
        if flags["arrived_search_area"]:
            arrived_search_area_count += 1
        if flags["at_base"]:
            at_base_count += 1

        total_steps += 1
        t += dt

    ekf_rmse_pos = rmse(ekf_pos_errors)
    gps_rmse_pos = rmse(gps_pos_errors)

    ekf_rmse_x = rmse(ekf_x_errors)
    ekf_rmse_y = rmse(ekf_y_errors)
    ekf_rmse_z = rmse(ekf_z_errors)

    gps_rmse_x = rmse(gps_x_errors)
    gps_rmse_y = rmse(gps_y_errors)
    gps_rmse_z = rmse(gps_z_errors)

    nav_ok_rate = nav_ok_count / total_steps if total_steps else 0.0
    altitude_reached_rate = altitude_reached_count / total_steps if total_steps else 0.0
    hover_stable_rate = hover_stable_count / total_steps if total_steps else 0.0
    arrived_search_area_rate = arrived_search_area_count / total_steps if total_steps else 0.0
    at_base_rate = at_base_count / total_steps if total_steps else 0.0

    improvement_pct = 0.0
    if gps_rmse_pos > 0.0:
        improvement_pct = 100.0 * (gps_rmse_pos - ekf_rmse_pos) / gps_rmse_pos

    print("=== EKF sensor fusion validation ===")
    print(f"Simulation time: {sim_time_s:.2f} s")
    print(f"Time step: {dt:.3f} s")
    print(f"Total steps: {total_steps}")

    print("\nPosition RMSE against ground truth")
    print(f"GPS RMSE total: {gps_rmse_pos:.4f} m")
    print(f"EKF RMSE total: {ekf_rmse_pos:.4f} m")
    print(f"Improvement over GPS: {improvement_pct:.2f} %")

    print("\nAxis-wise RMSE")
    print(f"GPS RMSE x: {gps_rmse_x:.4f} m")
    print(f"GPS RMSE y: {gps_rmse_y:.4f} m")
    print(f"GPS RMSE z: {gps_rmse_z:.4f} m")
    print(f"EKF RMSE x: {ekf_rmse_x:.4f} m")
    print(f"EKF RMSE y: {ekf_rmse_y:.4f} m")
    print(f"EKF RMSE z: {ekf_rmse_z:.4f} m")

    print("\nNavigation health and BT-related flags")
    print(f"nav_ok true rate: {nav_ok_rate:.3f}")
    print(f"altitude_reached true rate: {altitude_reached_rate:.3f}")
    print(f"hover_stable true rate: {hover_stable_rate:.3f}")
    print(f"arrived_search_area true rate: {arrived_search_area_rate:.3f}")
    print(f"at_base true rate: {at_base_rate:.3f}")

    print("\nFinal derived BT flags")
    print(final_flags)

    print("\nValidation checks")
    check_1 = ekf_rmse_pos < gps_rmse_pos
    check_2 = nav_ok_rate > 0.95
    check_3 = altitude_reached_count > 0

    print(f"EKF improves position estimate over GPS: {check_1}")
    print(f"Navigation health remains valid for most of run: {check_2}")
    print(f"Altitude target reached flag is triggered during the run: {check_3}")

    all_passed = check_1 and check_2 and check_3
    print(f"\nOverall EKF validation passed: {all_passed}")


if __name__ == "__main__":
    main()