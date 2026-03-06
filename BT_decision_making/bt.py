from typing import Tuple

from .blackboard import Blackboard
from .inputs import Inputs
from .actions import (
    ActionOutputs,
    search_action,
    track_action,
    lost_action,
    hold_action,
    search_last_known_action,
)


def tick_target(
    bb: Blackboard,
    inp: Inputs,
    *,
    conf_threshold: float = 0.7,
    debounce_s: float = 0.6,
    lost_to_search_s: float = 3.0,
) -> Tuple[ActionOutputs, str, bool]:
    now_s = bb.now()
    transition = ""

    detected_good = bool(inp.target_detected and (inp.target_conf >= conf_threshold))

    if detected_good:
        bb.last_seen_s = now_s
        bb.target_last_offset_px = (float(inp.target_offset_px[0]), float(inp.target_offset_px[1]))
        if bb.detect_stable_since_s < 0:
            bb.detect_stable_since_s = now_s
    else:
        bb.detect_stable_since_s = -1.0

    time_since_seen = (now_s - bb.last_seen_s) if bb.last_seen_s > 0 else 1e9
    stable_for = (now_s - bb.detect_stable_since_s) if bb.detect_stable_since_s > 0 else 0.0
    target_detected_stable = bool(detected_good and stable_for >= debounce_s)

    prev_state = bb.target_state

    if bb.target_state == "SEARCH":
        if target_detected_stable:
            bb.target_state = "TRACK"

    elif bb.target_state == "TRACK":
        if not detected_good:
            bb.target_state = "LOST"

    elif bb.target_state == "LOST":
        if target_detected_stable:
            bb.target_state = "TRACK"
        elif time_since_seen >= lost_to_search_s:
            bb.target_state = "SEARCH"

    else:
        bb.target_state = "SEARCH"

    if bb.target_state != prev_state:
        transition = f"{prev_state} -> {bb.target_state}"

    if bb.target_state == "SEARCH":
        out = search_action()
    elif bb.target_state == "TRACK":
        dx, dy = inp.target_offset_px
        out = track_action(dx, dy)
    else:
        out = lost_action()

    return out, transition, target_detected_stable


def tick_mission(
    bb: Blackboard,
    inp: Inputs,
    *,
    download_timeout_s: float = 20.0,
    request_missing_timeout_s: float = 10.0,
    comms_timeout_s: float = 10.0,
    search_timeout_s: float = 120.0,
    search_last_known_timeout_s: float = 20.0,
    arm_timeout_s: float = 10.0,
    verify_timeout_s: float = 6.0,
    verify_conf_threshold: float = 0.80,
    verify_confirm_s: float = 0.60,
    report_auto_sent_s: float = 0.50,
    report_timeout_s: float = 8.0,
) -> Tuple[ActionOutputs, str]:
    now_s = bb.now()
    transition = ""

    # Keep last known offset fresh in all modes
    if inp.target_detected and (inp.target_conf >= 0.25):
        bb.target_last_offset_px = (float(inp.target_offset_px[0]), float(inp.target_offset_px[1]))

    download_timeout_flag = bool(inp.download_timeout)
    arm_timeout_flag = bool(inp.arm_timeout)
    search_timeout_flag = bool(inp.search_timeout)
    comms_timeout_flag = bool(inp.comms_timeout)
    request_missing_timeout_flag = bool(inp.request_missing_data_timeout)
    search_last_known_timeout_flag = bool(inp.search_last_known_timeout)
    manual_reset = bool(inp.manual_reset)
    failsafe_requested = bool(inp.failsafe_requested)

    flight_active_modes = {
        "TAKEOFF_SET_ALTITUDE_TARGET",
        "TAKEOFF_CLIMB",
        "HOVER_STABILISE",
        "FLY_TO_SEARCH_AREA",
        "SEARCH_AREA",
        "VERIFY_TARGET",
        "TARGET_REPORT",
        "CONVERGE_DRONES",
        "TARGET_TRACKING",
        "DIRECT_CONTROL",
        "SEARCH_LAST_KNOWN_LOCATION",
        "PARENT_REASSIGNED",
        "LOST_COMMUNICATIONS",
        "RETURN_TO_BASE",
    }

    # FAILSAFE latch
    if failsafe_requested and bb.mode != "FAILSAFE":
        prev_mode = bb.mode
        bb.enter_mode("FAILSAFE")
        return hold_action("FAILSAFE", bb.target_state), f"{prev_mode} -> FAILSAFE"

    if bb.mode == "FAILSAFE":
        if manual_reset:
            prev_mode = bb.mode
            bb.enter_mode("IDLE")
            transition = f"{prev_mode} -> {bb.mode}"
        return hold_action(bb.mode, bb.target_state), transition

    # Emergency landing only during airborne or active mission phases
    if bb.mode in flight_active_modes and (
        inp.low_battery or inp.low_battery_critical or inp.fault or inp.damage
    ):
        prev_mode = bb.mode
        bb.enter_mode("EMERGENCY_LANDING")
        return hold_action(bb.mode, bb.target_state), f"{prev_mode} -> {bb.mode}"

    # Parent loss routing
    if bb.mode in flight_active_modes and inp.parent_lost and bb.mode != "PARENT_REASSIGNED":
        prev_mode = bb.mode
        bb.enter_mode("PARENT_REASSIGNED")
        return hold_action(bb.mode, bb.target_state), f"{prev_mode} -> {bb.mode}"

    if bb.mode == "PARENT_REASSIGNED":
        if inp.parent_reassigned:
            resume_mode = bb.prev_mode if bb.prev_mode != "PARENT_REASSIGNED" else "SEARCH_AREA"
            prev_mode = bb.mode
            bb.enter_mode(resume_mode)
            return hold_action(bb.mode, bb.target_state), f"{prev_mode} -> {bb.mode}"
        return hold_action(bb.mode, bb.target_state), transition

    # Comms loss routing
    comms_sensitive = {
        "FLY_TO_SEARCH_AREA",
        "SEARCH_AREA",
        "CONVERGE_DRONES",
        "TARGET_TRACKING",
        "DIRECT_CONTROL",
        "SEARCH_LAST_KNOWN_LOCATION",
    }

    if bb.mode in comms_sensitive and inp.comms_lost and bb.mode != "LOST_COMMUNICATIONS":
        prev_mode = bb.mode
        bb.enter_mode("LOST_COMMUNICATIONS")
        return hold_action(bb.mode, bb.target_state), f"{prev_mode} -> {bb.mode}"

    if bb.mode == "LOST_COMMUNICATIONS":
        if inp.comms_restored:
            resume_mode = bb.prev_mode if bb.prev_mode != "LOST_COMMUNICATIONS" else "SEARCH_AREA"
            prev_mode = bb.mode
            bb.enter_mode(resume_mode)
            return hold_action(bb.mode, bb.target_state), f"{prev_mode} -> {bb.mode}"

        timed_out = False
        if bb.comms_lost_since_s > 0 and (now_s - bb.comms_lost_since_s) >= comms_timeout_s:
            timed_out = True
        if comms_timeout_flag:
            timed_out = True

        if timed_out:
            prev_mode = bb.mode
            bb.enter_mode("RETURN_TO_BASE")
            return hold_action(bb.mode, bb.target_state), f"{prev_mode} -> {bb.mode}"

        return hold_action(bb.mode, bb.target_state), transition

    # Global recall routing for airborne phases
    if bb.mode in flight_active_modes and inp.recall_received and bb.mode not in {
        "RETURN_TO_BASE",
        "LANDING",
        "EMERGENCY_LANDING",
    }:
        prev_mode = bb.mode
        bb.enter_mode("RETURN_TO_BASE")
        return hold_action(bb.mode, bb.target_state), f"{prev_mode} -> {bb.mode}"

    prev_mode = bb.mode

    if bb.mode == "IDLE":
        if inp.mission_upload_received:
            bb.enter_mode("DOWNLOAD_MISSION")

    elif bb.mode == "DOWNLOAD_MISSION":
        if inp.mission_data_complete:
            bb.enter_mode("STORE_MISSION_DATA")
        else:
            timed_out = False
            if bb.download_start_s > 0 and (now_s - bb.download_start_s) >= download_timeout_s:
                timed_out = True
            if download_timeout_flag:
                timed_out = True

            if inp.missing_data_required or timed_out:
                bb.enter_mode("REQUEST_MISSING_DATA")

    elif bb.mode == "REQUEST_MISSING_DATA":
        if inp.mission_data_complete or inp.missing_data_received:
            bb.enter_mode("STORE_MISSION_DATA")
        else:
            timed_out = False
            if bb.request_missing_start_s > 0 and (now_s - bb.request_missing_start_s) >= request_missing_timeout_s:
                timed_out = True
            if request_missing_timeout_flag:
                timed_out = True

            if timed_out:
                bb.enter_mode("IDLE")

    elif bb.mode == "STORE_MISSION_DATA":
        if inp.mission_store_ok:
            bb.enter_mode("INITIALISE")
        else:
            bb.enter_mode("REPORT_FAULT")

    elif bb.mode == "INITIALISE":
        checks_ok = (
            inp.battery_ok
            and inp.fc_ok
            and inp.nav_ok
            and inp.comms_ok
            and inp.payload_ok
            and inp.obc_ok
            and inp.mission_valid
        )
        if checks_ok:
            bb.enter_mode("COORDINATE_SWARM")
        else:
            bb.enter_mode("REPORT_FAULT")

    elif bb.mode == "REPORT_FAULT":
        bb.enter_mode("IDLE")

    elif bb.mode == "COORDINATE_SWARM":
        if inp.swarm_coordinated:
            bb.enter_mode("DETERMINE_ROLES")

    elif bb.mode == "DETERMINE_ROLES":
        if inp.role_election_failed:
            bb.enter_mode("IDLE")
        elif inp.roles_assigned and inp.priority_list_sent:
            bb.enter_mode("WAIT_LAUNCH")

    elif bb.mode == "WAIT_LAUNCH":
        if inp.abort_received:
            bb.enter_mode("IDLE")
        elif inp.launch_command_received:
            bb.enter_mode("TAKEOFF_ARM")

    elif bb.mode == "TAKEOFF_ARM":
        timed_out = (now_s - bb.mode_enter_s) >= arm_timeout_s or arm_timeout_flag
        if inp.arm_denied or timed_out:
            bb.enter_mode("REPORT_ARM_REQUEST_FAILURE")
        elif inp.arm_permission_received:
            bb.enter_mode("TAKEOFF_SET_ALTITUDE_TARGET")

    elif bb.mode == "REPORT_ARM_REQUEST_FAILURE":
        bb.enter_mode("IDLE")

    elif bb.mode == "TAKEOFF_SET_ALTITUDE_TARGET":
        if inp.takeoff_altitude_target_set:
            bb.enter_mode("TAKEOFF_CLIMB")

    elif bb.mode == "TAKEOFF_CLIMB":
        if inp.altitude_reached:
            bb.enter_mode("HOVER_STABILISE")

    elif bb.mode == "HOVER_STABILISE":
        if inp.hover_stable:
            bb.enter_mode("FLY_TO_SEARCH_AREA")

    elif bb.mode == "FLY_TO_SEARCH_AREA":
        if inp.arrived_search_area:
            bb.enter_mode("SEARCH_AREA")

    elif bb.mode == "SEARCH_AREA":
        out_t, trans_t, target_detected_stable = tick_target(bb, inp)
        if trans_t:
            transition = f"TARGET {trans_t}"

        timed_out = False
        if bb.search_start_s > 0 and (now_s - bb.search_start_s) >= search_timeout_s:
            timed_out = True
        if search_timeout_flag:
            timed_out = True

        if target_detected_stable:
            bb.enter_mode("VERIFY_TARGET")
        elif timed_out:
            bb.enter_mode("RETURN_TO_BASE")

        if bb.mode != prev_mode:
            mode_transition = f"{prev_mode} -> {bb.mode}"
            transition = f"{transition} | {mode_transition}" if transition else mode_transition

        if bb.mode == "SEARCH_AREA":
            out_t.mission_mode = "SEARCH_AREA"
            out_t.target_mode = bb.target_state
            return out_t, transition

    elif bb.mode == "VERIFY_TARGET":
        # External flags win
        if inp.target_confirmed:
            bb.enter_mode("TARGET_REPORT")
        elif inp.target_rejected or (not inp.target_detected):
            bb.enter_mode("SEARCH_AREA")
        else:
            good_verify = bool(inp.target_detected and (inp.target_conf >= verify_conf_threshold))
            if good_verify:
                if bb.verify_good_since_s < 0:
                    bb.verify_good_since_s = now_s
                if (now_s - bb.verify_good_since_s) >= verify_confirm_s:
                    bb.enter_mode("TARGET_REPORT")
            else:
                bb.verify_good_since_s = -1.0

            if bb.verify_start_s > 0 and (now_s - bb.verify_start_s) >= verify_timeout_s:
                bb.enter_mode("SEARCH_AREA")

    elif bb.mode == "TARGET_REPORT":
        # External flags win
        if inp.report_failed:
            bb.enter_mode("SEARCH_AREA")
        elif inp.report_sent:
            bb.enter_mode("CONVERGE_DRONES")
        else:
            if bb.report_start_s > 0 and (now_s - bb.report_start_s) >= report_auto_sent_s:
                bb.enter_mode("CONVERGE_DRONES")
            elif bb.report_start_s > 0 and (now_s - bb.report_start_s) >= report_timeout_s:
                bb.enter_mode("SEARCH_AREA")

    elif bb.mode == "CONVERGE_DRONES":
        if inp.converge_complete:
            bb.enter_mode("TARGET_TRACKING")

    elif bb.mode == "TARGET_TRACKING":
        if inp.direct_control_enabled:
            bb.enter_mode("DIRECT_CONTROL")
        elif not inp.target_detected:
            bb.enter_mode("SEARCH_LAST_KNOWN_LOCATION")
        elif inp.mission_complete:
            bb.enter_mode("RETURN_TO_BASE")

    elif bb.mode == "SEARCH_LAST_KNOWN_LOCATION":
        _, trans_t, target_detected_stable = tick_target(bb, inp)
        if trans_t:
            transition = f"TARGET {trans_t}"

        timed_out = False
        if bb.search_last_known_start_s > 0 and (now_s - bb.search_last_known_start_s) >= search_last_known_timeout_s:
            timed_out = True
        if search_last_known_timeout_flag:
            timed_out = True

        if target_detected_stable:
            bb.enter_mode("TARGET_TRACKING")
        elif timed_out:
            bb.enter_mode("SEARCH_AREA")

        if bb.mode != prev_mode:
            mode_transition = f"{prev_mode} -> {bb.mode}"
            transition = f"{transition} | {mode_transition}" if transition else mode_transition

        if bb.mode == "SEARCH_LAST_KNOWN_LOCATION":
            out = search_last_known_action(bb.target_last_offset_px[0])
            out.mission_mode = "SEARCH_LAST_KNOWN_LOCATION"
            out.target_mode = "LAST_KNOWN"
            return out, transition

    elif bb.mode == "DIRECT_CONTROL":
        if inp.direct_control_released:
            bb.enter_mode("TARGET_TRACKING")

    elif bb.mode == "RETURN_TO_BASE":
        if inp.at_base:
            bb.enter_mode("LANDING")
        elif inp.low_battery_critical or inp.fault:
            bb.enter_mode("EMERGENCY_LANDING")

    elif bb.mode == "LANDING":
        if inp.landed:
            bb.enter_mode("IDLE")

    elif bb.mode == "EMERGENCY_LANDING":
        if inp.landed:
            bb.enter_mode("IDLE")

    if bb.mode != prev_mode and not transition:
        transition = f"{prev_mode} -> {bb.mode}"

    return hold_action(bb.mode, bb.target_state), transition