from dataclasses import dataclass
import numpy as np


@dataclass
class EKFConfig:
    # Frame settings
    frame_name: str = "ENU"
    dt_default_s: float = 0.02
    max_dt_s: float = 0.10

    # IMU handling
    imu_accel_is_linear: bool = True

    # Process noise
    accel_process_std_mps2: float = 0.80
    gyro_process_std_radps: float = 0.15
    attitude_process_std_rad: float = 0.03

    # Measurement noise
    gps_position_std_m: float = 2.50
    realsense_position_std_m: float = 0.25
    realsense_velocity_std_mps: float = 0.20
    realsense_yaw_std_rad: float = 0.08

    # Initial covariance
    initial_position_std_m: float = 5.00
    initial_velocity_std_mps: float = 2.00
    initial_attitude_std_rad: float = 0.30

    # Health thresholds
    nav_max_position_var_m2: float = 9.00
    nav_max_velocity_var_m2ps2: float = 4.00
    nav_max_attitude_var_rad2: float = 0.25
    nav_sensor_timeout_s: float = 1.00

    def initial_covariance(self) -> np.ndarray:
        p = np.zeros((9, 9), dtype=float)

        pos_var = self.initial_position_std_m ** 2
        vel_var = self.initial_velocity_std_mps ** 2
        att_var = self.initial_attitude_std_rad ** 2

        p[0, 0] = pos_var
        p[1, 1] = pos_var
        p[2, 2] = pos_var

        p[3, 3] = vel_var
        p[4, 4] = vel_var
        p[5, 5] = vel_var

        p[6, 6] = att_var
        p[7, 7] = att_var
        p[8, 8] = att_var

        return p