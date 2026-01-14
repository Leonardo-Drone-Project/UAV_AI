from dataclasses import dataclass


@dataclass
class ActionOutputs:
    mode: str
    yaw_rate_cmd: float
    forward_vel_cmd: float
    gimbal_pitch_cmd: float
    hold_position: bool

    def __str__(self) -> str:
        return (
            f"[{self.mode}] yaw_rate={self.yaw_rate_cmd:+.3f}, "
            f"fwd_vel={self.forward_vel_cmd:+.3f}, "
            f"gimbal_pitch={self.gimbal_pitch_cmd:+.3f}, "
            f"hold={self.hold_position}"
        )


def search_action() -> ActionOutputs:
    # Simple scan behaviour (placeholder)
    return ActionOutputs(
        mode="SEARCH",
        yaw_rate_cmd=0.25,       # slow yaw scan
        forward_vel_cmd=0.0,
        gimbal_pitch_cmd=0.0,
        hold_position=True,
    )


def track_action(dx: float, dy: float) -> ActionOutputs:
    # Very simple proportional “keep target centered” placeholder
    k_yaw = 0.002  # px -> yaw rate
    yaw_cmd = max(min(-k_yaw * dx, 0.6), -0.6)

    # Move forward slowly when centered-ish
    forward = 0.25 if abs(dx) < 40.0 else 0.0

    # Optional pitch logic (kept 0 for now)
    return ActionOutputs(
        mode="TRACK",
        yaw_rate_cmd=yaw_cmd,
        forward_vel_cmd=forward,
        gimbal_pitch_cmd=0.0,
        hold_position=False,
    )


def lost_action() -> ActionOutputs:
    # Short, faster scan to reacquire
    return ActionOutputs(
        mode="LOST",
        yaw_rate_cmd=0.45,
        forward_vel_cmd=0.0,
        gimbal_pitch_cmd=0.0,
        hold_position=True,
    )


def failsafe_action() -> ActionOutputs:
    # Hold position (or RTL in a real system)
    return ActionOutputs(
        mode="FAILSAFE",
        yaw_rate_cmd=0.0,
        forward_vel_cmd=0.0,
        gimbal_pitch_cmd=0.0,
        hold_position=True,
    )
