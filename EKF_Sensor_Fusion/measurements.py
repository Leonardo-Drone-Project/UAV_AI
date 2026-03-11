from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class IMUMeasurement:
    timestamp_s: float
    accel_xyz_mps2: Tuple[float, float, float]
    gyro_xyz_radps: Tuple[float, float, float]
    valid: bool = True


@dataclass
class GPSMeasurement:
    timestamp_s: float
    position_xyz_m: Tuple[float, float, float]
    valid: bool = True


@dataclass
class RealSenseMeasurement:
    timestamp_s: float
    position_xyz_m: Optional[Tuple[float, float, float]] = None
    velocity_xyz_mps: Optional[Tuple[float, float, float]] = None
    yaw_rad: Optional[float] = None
    valid: bool = True