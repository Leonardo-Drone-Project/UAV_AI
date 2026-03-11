import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .config import EKFConfig
from .ekf import ExtendedKalmanFilter
from .measurements import GPSMeasurement, IMUMeasurement, RealSenseMeasurement
from .state import FusedState, StateIndex


@dataclass
class NavThresholds:
    altitude_tolerance_m: float = 0.5
    hover_speed_max_mps: float = 0.4
    search_radius_m: float = 2.0
    base_radius_m: float = 2.0


class EKFSensorFusionInterface:
    def __init__(self, config: Optional[EKFConfig] = None):
        self.config = config or EKFConfig()
        self.filter = ExtendedKalmanFilter(self.config)

    def push_imu(self, imu: IMUMeasurement, dt_s: Optional[float] = None) -> None:
        self.filter.predict(imu, dt_s=dt_s)

    def push_gps(self, gps: GPSMeasurement) -> None:
        self.filter.update_gps(gps)

    def push_realsense(self, rs: RealSenseMeasurement) -> None:
        self.filter.update_realsense(rs)

    def get_fused_state(self, timestamp_s: Optional[float] = None) -> FusedState:
        x = self.filter.get_state_vector()
        p = self.filter.get_covariance()

        now_s = float(time.time() if timestamp_s is None else timestamp_s)

        latest_sensor_ts = None
        sensor_times = [
            self.filter.last_predict_ts,
            self.filter.last_gps_ts,
            self.filter.last_realsense_ts,
        ]
        valid_times = [t for t in sensor_times if t is not None]
        if valid_times:
            latest_sensor_ts = max(valid_times)

        pos_var = (
            float(p[StateIndex.X, StateIndex.X]),
            float(p[StateIndex.Y, StateIndex.Y]),
            float(p[StateIndex.Z, StateIndex.Z]),
        )
        vel_var = (
            float(p[StateIndex.VX, StateIndex.VX]),
            float(p[StateIndex.VY, StateIndex.VY]),
            float(p[StateIndex.VZ, StateIndex.VZ]),
        )
        att_var = (
            float(p[StateIndex.ROLL, StateIndex.ROLL]),
            float(p[StateIndex.PITCH, StateIndex.PITCH]),
            float(p[StateIndex.YAW, StateIndex.YAW]),
        )

        nav_ok = self.filter.initialized
        if latest_sensor_ts is None:
            nav_ok = False
        else:
            if (now_s - latest_sensor_ts) > self.config.nav_sensor_timeout_s:
                nav_ok = False

        if max(pos_var) > self.config.nav_max_position_var_m2:
            nav_ok = False
        if max(vel_var) > self.config.nav_max_velocity_var_m2ps2:
            nav_ok = False
        if max(att_var) > self.config.nav_max_attitude_var_rad2:
            nav_ok = False

        return FusedState(
            x_m=float(x[StateIndex.X]),
            y_m=float(x[StateIndex.Y]),
            z_m=float(x[StateIndex.Z]),
            vx_mps=float(x[StateIndex.VX]),
            vy_mps=float(x[StateIndex.VY]),
            vz_mps=float(x[StateIndex.VZ]),
            roll_rad=float(x[StateIndex.ROLL]),
            pitch_rad=float(x[StateIndex.PITCH]),
            yaw_rad=float(x[StateIndex.YAW]),
            timestamp_s=now_s,
            position_var_xyz=pos_var,
            velocity_var_xyz=vel_var,
            attitude_var_rpy=att_var,
            nav_ok=nav_ok,
        )

    def derive_bt_navigation_flags(
        self,
        thresholds: Optional[NavThresholds] = None,
        altitude_target_m: Optional[float] = None,
        search_area_xy_m: Optional[Tuple[float, float]] = None,
        base_xy_m: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, bool]:
        th = thresholds or NavThresholds()
        state = self.get_fused_state()

        speed_mps = math.sqrt(
            state.vx_mps ** 2 + state.vy_mps ** 2 + state.vz_mps ** 2
        )

        flags = {
            "nav_ok": bool(state.nav_ok),
            "altitude_reached": False,
            "hover_stable": bool(state.nav_ok and speed_mps <= th.hover_speed_max_mps),
            "arrived_search_area": False,
            "at_base": False,
        }

        if altitude_target_m is not None:
            flags["altitude_reached"] = abs(state.z_m - altitude_target_m) <= th.altitude_tolerance_m

        if search_area_xy_m is not None:
            dx = state.x_m - float(search_area_xy_m[0])
            dy = state.y_m - float(search_area_xy_m[1])
            flags["arrived_search_area"] = math.hypot(dx, dy) <= th.search_radius_m

        if base_xy_m is not None:
            dx = state.x_m - float(base_xy_m[0])
            dy = state.y_m - float(base_xy_m[1])
            flags["at_base"] = math.hypot(dx, dy) <= th.base_radius_m

        return flags