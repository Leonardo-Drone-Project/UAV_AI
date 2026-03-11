import math
from typing import Any, Dict, Optional, Tuple

from .measurements import GPSMeasurement, IMUMeasurement, RealSenseMeasurement


def _first(payload: Dict[str, Any], keys, default=None):
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _timestamp_s(payload: Dict[str, Any]) -> float:
    if "timestamp_s" in payload:
        return float(payload["timestamp_s"])
    if "timestamp" in payload:
        return float(payload["timestamp"])
    if "time_boot_ms" in payload:
        return float(payload["time_boot_ms"]) / 1000.0
    return 0.0


def _triple_from_payload(
    payload: Dict[str, Any],
    tuple_keys,
    scalar_keys,
) -> Optional[Tuple[float, float, float]]:
    vec = _first(payload, tuple_keys, default=None)
    if vec is not None:
        if isinstance(vec, dict):
            return (
                float(vec.get("x", 0.0)),
                float(vec.get("y", 0.0)),
                float(vec.get("z", 0.0)),
            )
        if len(vec) >= 3:
            return (float(vec[0]), float(vec[1]), float(vec[2]))

    x = _first(payload, scalar_keys[0], default=None)
    y = _first(payload, scalar_keys[1], default=None)
    z = _first(payload, scalar_keys[2], default=None)
    if x is None or y is None or z is None:
        return None

    return (float(x), float(y), float(z))


class LocalFrameConverter:
    """
    Simple geodetic to local ENU converter using a small-area approximation.
    Good enough for local mission-scale work.
    """

    def __init__(self, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float = 0.0):
        self.ref_lat_deg = float(ref_lat_deg)
        self.ref_lon_deg = float(ref_lon_deg)
        self.ref_alt_m = float(ref_alt_m)
        self._earth_radius_m = 6378137.0

    def geodetic_to_enu(self, lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
        d_lat = math.radians(float(lat_deg) - self.ref_lat_deg)
        d_lon = math.radians(float(lon_deg) - self.ref_lon_deg)
        ref_lat_rad = math.radians(self.ref_lat_deg)

        east_m = self._earth_radius_m * d_lon * math.cos(ref_lat_rad)
        north_m = self._earth_radius_m * d_lat
        up_m = float(alt_m) - self.ref_alt_m

        return (east_m, north_m, up_m)


def imu_from_payload(payload: Dict[str, Any]) -> IMUMeasurement:
    accel = _triple_from_payload(
        payload,
        tuple_keys=["accel_xyz_mps2", "linear_acceleration", "accel"],
        scalar_keys=[
            ["ax", "accel_x", "linear_acceleration_x"],
            ["ay", "accel_y", "linear_acceleration_y"],
            ["az", "accel_z", "linear_acceleration_z"],
        ],
    )
    gyro = _triple_from_payload(
        payload,
        tuple_keys=["gyro_xyz_radps", "angular_velocity", "gyro"],
        scalar_keys=[
            ["gx", "gyro_x", "angular_velocity_x"],
            ["gy", "gyro_y", "angular_velocity_y"],
            ["gz", "gyro_z", "angular_velocity_z"],
        ],
    )

    if accel is None:
        accel = (0.0, 0.0, 0.0)
    if gyro is None:
        gyro = (0.0, 0.0, 0.0)

    valid = bool(_first(payload, ["valid", "imu_valid"], default=True))

    return IMUMeasurement(
        timestamp_s=_timestamp_s(payload),
        accel_xyz_mps2=accel,
        gyro_xyz_radps=gyro,
        valid=valid,
    )


def gps_from_payload(
    payload: Dict[str, Any],
    converter: Optional[LocalFrameConverter] = None,
) -> GPSMeasurement:
    position_xyz_m = _triple_from_payload(
        payload,
        tuple_keys=["position_xyz_m", "position", "enu_position_m"],
        scalar_keys=[
            ["x", "pos_x", "east_m"],
            ["y", "pos_y", "north_m"],
            ["z", "pos_z", "up_m"],
        ],
    )

    if position_xyz_m is None:
        lat = _first(payload, ["lat_deg", "latitude_deg", "lat"], default=None)
        lon = _first(payload, ["lon_deg", "longitude_deg", "lon"], default=None)
        alt = _first(payload, ["alt_m", "altitude_m", "alt"], default=None)

        if lat is None or lon is None or alt is None:
            raise ValueError("GPS payload must contain local xyz or lat/lon/alt values")

        if converter is None:
            raise ValueError("GPS lat/lon payload needs a LocalFrameConverter")

        position_xyz_m = converter.geodetic_to_enu(float(lat), float(lon), float(alt))

    valid = bool(_first(payload, ["valid", "gps_valid"], default=True))

    return GPSMeasurement(
        timestamp_s=_timestamp_s(payload),
        position_xyz_m=position_xyz_m,
        valid=valid,
    )


def realsense_from_payload(payload: Dict[str, Any]) -> RealSenseMeasurement:
    position_xyz_m = _triple_from_payload(
        payload,
        tuple_keys=["position_xyz_m", "position", "pose_xyz_m"],
        scalar_keys=[
            ["x", "pos_x", "rs_x"],
            ["y", "pos_y", "rs_y"],
            ["z", "pos_z", "rs_z"],
        ],
    )

    velocity_xyz_mps = _triple_from_payload(
        payload,
        tuple_keys=["velocity_xyz_mps", "velocity", "vel_xyz_mps"],
        scalar_keys=[
            ["vx", "vel_x", "rs_vx"],
            ["vy", "vel_y", "rs_vy"],
            ["vz", "vel_z", "rs_vz"],
        ],
    )

    yaw_rad = _first(payload, ["yaw_rad"], default=None)
    if yaw_rad is None:
        yaw_deg = _first(payload, ["yaw_deg"], default=None)
        if yaw_deg is not None:
            yaw_rad = math.radians(float(yaw_deg))

    valid = bool(_first(payload, ["valid", "realsense_valid", "vision_valid"], default=True))

    return RealSenseMeasurement(
        timestamp_s=_timestamp_s(payload),
        position_xyz_m=position_xyz_m,
        velocity_xyz_mps=velocity_xyz_mps,
        yaw_rad=None if yaw_rad is None else float(yaw_rad),
        valid=valid,
    )