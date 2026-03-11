import math
from typing import Optional, Tuple

import numpy as np

from .config import EKFConfig
from .measurements import GPSMeasurement, IMUMeasurement, RealSenseMeasurement
from .state import StateIndex, wrap_angle, wrap_euler_in_place


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=float,
    )
    ry = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=float,
    )
    rz = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    return rz @ ry @ rx


class ExtendedKalmanFilter:
    def __init__(self, config: Optional[EKFConfig] = None):
        self.config = config or EKFConfig()

        self.x = np.zeros(StateIndex.SIZE, dtype=float)
        self.P = self.config.initial_covariance()

        self.initialized = False

        self.last_predict_ts: Optional[float] = None
        self.last_gps_ts: Optional[float] = None
        self.last_realsense_ts: Optional[float] = None

    def reset(
        self,
        initial_position_xyz_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_velocity_xyz_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_attitude_rpy_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        timestamp_s: float = 0.0,
    ) -> None:
        self.x = np.zeros(StateIndex.SIZE, dtype=float)
        self.P = self.config.initial_covariance()

        self.x[0:3] = np.array(initial_position_xyz_m, dtype=float)
        self.x[3:6] = np.array(initial_velocity_xyz_mps, dtype=float)
        self.x[6:9] = np.array(initial_attitude_rpy_rad, dtype=float)
        wrap_euler_in_place(self.x)

        self.initialized = True
        self.last_predict_ts = timestamp_s
        self.last_gps_ts = None
        self.last_realsense_ts = None

    def predict(self, imu: IMUMeasurement, dt_s: Optional[float] = None) -> None:
        if not imu.valid:
            return

        if not self.initialized:
            self.reset(timestamp_s=float(imu.timestamp_s))

        if dt_s is None:
            if self.last_predict_ts is None:
                dt_s = self.config.dt_default_s
            else:
                dt_s = float(imu.timestamp_s - self.last_predict_ts)

        dt_s = max(1e-3, min(float(dt_s), self.config.max_dt_s))
        self.last_predict_ts = float(imu.timestamp_s)

        accel_body = np.array(imu.accel_xyz_mps2, dtype=float).reshape(3)
        gyro_body = np.array(imu.gyro_xyz_radps, dtype=float).reshape(3)

        roll = float(self.x[StateIndex.ROLL])
        pitch = float(self.x[StateIndex.PITCH])
        yaw = float(self.x[StateIndex.YAW])

        if self.config.imu_accel_is_linear:
            accel_world = euler_to_rotation_matrix(roll, pitch, yaw) @ accel_body
        else:
            accel_world = euler_to_rotation_matrix(roll, pitch, yaw) @ accel_body

        self.x[StateIndex.X:StateIndex.Z + 1] += (
            self.x[StateIndex.VX:StateIndex.VZ + 1] * dt_s
            + 0.5 * accel_world * (dt_s ** 2)
        )
        self.x[StateIndex.VX:StateIndex.VZ + 1] += accel_world * dt_s
        self.x[StateIndex.ROLL:StateIndex.YAW + 1] += gyro_body * dt_s
        wrap_euler_in_place(self.x)

        f = np.eye(StateIndex.SIZE, dtype=float)
        f[StateIndex.X, StateIndex.VX] = dt_s
        f[StateIndex.Y, StateIndex.VY] = dt_s
        f[StateIndex.Z, StateIndex.VZ] = dt_s

        q = np.zeros((StateIndex.SIZE, StateIndex.SIZE), dtype=float)

        accel_var = self.config.accel_process_std_mps2 ** 2
        gyro_var = self.config.gyro_process_std_radps ** 2
        att_var = self.config.attitude_process_std_rad ** 2

        for pos_i, vel_i in [
            (StateIndex.X, StateIndex.VX),
            (StateIndex.Y, StateIndex.VY),
            (StateIndex.Z, StateIndex.VZ),
        ]:
            q[pos_i, pos_i] = 0.25 * (dt_s ** 4) * accel_var
            q[pos_i, vel_i] = 0.5 * (dt_s ** 3) * accel_var
            q[vel_i, pos_i] = 0.5 * (dt_s ** 3) * accel_var
            q[vel_i, vel_i] = (dt_s ** 2) * accel_var

        q[StateIndex.ROLL, StateIndex.ROLL] = (dt_s ** 2) * gyro_var + dt_s * att_var
        q[StateIndex.PITCH, StateIndex.PITCH] = (dt_s ** 2) * gyro_var + dt_s * att_var
        q[StateIndex.YAW, StateIndex.YAW] = (dt_s ** 2) * gyro_var + dt_s * att_var

        self.P = f @ self.P @ f.T + q

    def update_gps(self, gps: GPSMeasurement) -> None:
        if not gps.valid:
            return

        z = np.array(gps.position_xyz_m, dtype=float).reshape(3)

        if not self.initialized:
            self.reset(initial_position_xyz_m=tuple(z.tolist()), timestamp_s=float(gps.timestamp_s))
            self.last_gps_ts = float(gps.timestamp_s)
            return

        h = np.zeros((3, StateIndex.SIZE), dtype=float)
        h[:, StateIndex.X:StateIndex.Z + 1] = np.eye(3, dtype=float)

        r = np.eye(3, dtype=float) * (self.config.gps_position_std_m ** 2)
        self._update_linear(z, h, r)

        self.last_gps_ts = float(gps.timestamp_s)

    def update_realsense(self, rs: RealSenseMeasurement) -> None:
        if not rs.valid:
            return

        if rs.position_xyz_m is not None:
            z = np.array(rs.position_xyz_m, dtype=float).reshape(3)

            if not self.initialized:
                self.reset(initial_position_xyz_m=tuple(z.tolist()), timestamp_s=float(rs.timestamp_s))
                self.last_realsense_ts = float(rs.timestamp_s)
            else:
                h = np.zeros((3, StateIndex.SIZE), dtype=float)
                h[:, StateIndex.X:StateIndex.Z + 1] = np.eye(3, dtype=float)
                r = np.eye(3, dtype=float) * (self.config.realsense_position_std_m ** 2)
                self._update_linear(z, h, r)

        if rs.velocity_xyz_mps is not None and self.initialized:
            z = np.array(rs.velocity_xyz_mps, dtype=float).reshape(3)
            h = np.zeros((3, StateIndex.SIZE), dtype=float)
            h[:, StateIndex.VX:StateIndex.VZ + 1] = np.eye(3, dtype=float)
            r = np.eye(3, dtype=float) * (self.config.realsense_velocity_std_mps ** 2)
            self._update_linear(z, h, r)

        if rs.yaw_rad is not None and self.initialized:
            z = np.array([float(rs.yaw_rad)], dtype=float)
            h = np.zeros((1, StateIndex.SIZE), dtype=float)
            h[0, StateIndex.YAW] = 1.0
            r = np.array([[self.config.realsense_yaw_std_rad ** 2]], dtype=float)
            self._update_yaw(z, h, r)

        self.last_realsense_ts = float(rs.timestamp_s)

    def _update_linear(self, z: np.ndarray, h: np.ndarray, r: np.ndarray) -> None:
        y = z - (h @ self.x)
        s = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.inv(s)

        self.x = self.x + k @ y
        wrap_euler_in_place(self.x)

        i = np.eye(StateIndex.SIZE, dtype=float)
        self.P = (i - k @ h) @ self.P

    def _update_yaw(self, z: np.ndarray, h: np.ndarray, r: np.ndarray) -> None:
        yaw_pred = float(h @ self.x)
        yaw_meas = float(z[0])
        yaw_residual = wrap_angle(yaw_meas - yaw_pred)

        y = np.array([yaw_residual], dtype=float)
        s = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.inv(s)

        self.x = self.x + k @ y
        wrap_euler_in_place(self.x)

        i = np.eye(StateIndex.SIZE, dtype=float)
        self.P = (i - k @ h) @ self.P

    def get_state_vector(self) -> np.ndarray:
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()