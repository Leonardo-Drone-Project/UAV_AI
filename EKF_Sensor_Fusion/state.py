from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np


class StateIndex:
    X = 0
    Y = 1
    Z = 2
    VX = 3
    VY = 4
    VZ = 5
    ROLL = 6
    PITCH = 7
    YAW = 8
    SIZE = 9


def wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def wrap_euler_in_place(x: np.ndarray) -> None:
    x[StateIndex.ROLL] = wrap_angle(float(x[StateIndex.ROLL]))
    x[StateIndex.PITCH] = wrap_angle(float(x[StateIndex.PITCH]))
    x[StateIndex.YAW] = wrap_angle(float(x[StateIndex.YAW]))


@dataclass
class FusedState:
    x_m: float
    y_m: float
    z_m: float

    vx_mps: float
    vy_mps: float
    vz_mps: float

    roll_rad: float
    pitch_rad: float
    yaw_rad: float

    timestamp_s: float

    position_var_xyz: Tuple[float, float, float]
    velocity_var_xyz: Tuple[float, float, float]
    attitude_var_rpy: Tuple[float, float, float]

    nav_ok: bool

    def position_xyz(self) -> Tuple[float, float, float]:
        return (self.x_m, self.y_m, self.z_m)

    def velocity_xyz(self) -> Tuple[float, float, float]:
        return (self.vx_mps, self.vy_mps, self.vz_mps)

    def attitude_rpy(self) -> Tuple[float, float, float]:
        return (self.roll_rad, self.pitch_rad, self.yaw_rad)