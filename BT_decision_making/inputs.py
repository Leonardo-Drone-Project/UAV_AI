from dataclasses import dataclass
from typing import Tuple


@dataclass
class Inputs:
    # Perception
    target_detected: bool = False
    target_conf: float = 0.0
    target_class: str = ""
    target_offset_px: Tuple[float, float] = (0.0, 0.0)
    target_offset_norm: Tuple[float, float] = (0.0, 0.0)

    # Optional logging / role placeholder
    role: str = "unknown"

    # Mission upload and mission data
    mission_upload_received: bool = False
    mission_data_complete: bool = False

    # Download / missing data handling
    download_timeout: bool = False
    missing_data_required: bool = False
    missing_data_received: bool = False
    request_missing_data_timeout: bool = False

    # Store mission data
    mission_store_ok: bool = True

    # Pre-flight checks
    battery_ok: bool = True
    fc_ok: bool = True
    nav_ok: bool = True
    comms_ok: bool = True
    payload_ok: bool = True
    obc_ok: bool = True
    mission_valid: bool = True

    # Swarm coordination
    swarm_coordinated: bool = False

    # Role election
    roles_assigned: bool = False
    priority_list_sent: bool = False
    role_election_failed: bool = False

    # Launch and arm
    launch_command_received: bool = False
    abort_received: bool = False
    arm_permission_received: bool = False
    arm_denied: bool = False
    arm_timeout: bool = False

    # Takeoff altitude target
    takeoff_altitude_target_set: bool = True

    # Flight progress
    altitude_reached: bool = False
    hover_stable: bool = False
    arrived_search_area: bool = False

    # Target verification and reporting
    target_confirmed: bool = False
    target_rejected: bool = False
    report_sent: bool = False
    report_failed: bool = False

    # Converge and tracking
    converge_complete: bool = False
    direct_control_enabled: bool = False
    direct_control_released: bool = False

    # Parent loss
    parent_lost: bool = False
    parent_reassigned: bool = False

    # Comms
    comms_lost: bool = False
    comms_restored: bool = False
    comms_timeout: bool = False

    # Recall / mission end
    recall_received: bool = False
    mission_complete: bool = False

    # Search timeouts
    search_timeout: bool = False
    search_last_known_timeout: bool = False

    # Return and landing
    at_base: bool = False
    landed: bool = False

    # Safety
    low_battery: bool = False
    low_battery_critical: bool = False
    fault: bool = False
    damage: bool = False

    # Failsafe latch
    failsafe_requested: bool = False
    manual_reset: bool = False