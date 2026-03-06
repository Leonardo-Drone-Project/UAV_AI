# BT_decision_making/actions.py
from dataclasses import dataclass


@dataclass
class ActionOutputs:
    mission_mode: str
    target_mode: str
    yaw_rate_cmd: float
    forward_vel_cmd: float
    gimbal_pitch_cmd: float
    hold_position: bool

    def __str__(self) -> str:
        return (
            f"[{self.mission_mode}/{self.target_mode}] "
            f"yaw_rate={self.yaw_rate_cmd:+.3f}, "
            f"fwd_vel={self.forward_vel_cmd:+.3f}, "
            f"gimbal_pitch={self.gimbal_pitch_cmd:+.3f}, "
            f"hold={self.hold_position}"
        )


def hold_action(mission_mode: str, target_mode: str = "-") -> ActionOutputs:
    return ActionOutputs(
        mission_mode=mission_mode,
        target_mode=target_mode,
        yaw_rate_cmd=0.0,
        forward_vel_cmd=0.0,
        gimbal_pitch_cmd=0.0,
        hold_position=True,
    )


def search_action() -> ActionOutputs:
    return ActionOutputs(
        mission_mode="SEARCH_AREA",
        target_mode="SEARCH",
        yaw_rate_cmd=0.25,
        forward_vel_cmd=0.0,
        gimbal_pitch_cmd=0.0,
        hold_position=True,
    )


def track_action(dx_px: float, dy_px: float) -> ActionOutputs:
    k_yaw = 0.002
    yaw_cmd = max(min(-k_yaw * float(dx_px), 0.6), -0.6)
    forward = 0.25 if abs(float(dx_px)) < 40.0 else 0.0
    return ActionOutputs(
        mission_mode="SEARCH_AREA",
        target_mode="TRACK",
        yaw_rate_cmd=yaw_cmd,
        forward_vel_cmd=forward,
        gimbal_pitch_cmd=0.0,
        hold_position=False,
    )


def lost_action() -> ActionOutputs:
    return ActionOutputs(
        mission_mode="SEARCH_AREA",
        target_mode="LOST",
        yaw_rate_cmd=0.45,
        forward_vel_cmd=0.0,
        gimbal_pitch_cmd=0.0,
        hold_position=True,
    )


def search_last_known_action(last_dx_px: float) -> ActionOutputs:
    if abs(float(last_dx_px)) < 1.0:
        yaw_cmd = 0.25
    else:
        k_yaw = 0.0015
        yaw_cmd = max(min(-k_yaw * float(last_dx_px), 0.45), -0.45)

    return ActionOutputs(
        mission_mode="SEARCH_LAST_KNOWN_LOCATION",
        target_mode="LAST_KNOWN",
        yaw_rate_cmd=yaw_cmd,
        forward_vel_cmd=0.0,
        gimbal_pitch_cmd=0.0,
        hold_position=True,
    )