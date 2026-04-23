from BT_decision_making.blackboard import Blackboard
from BT_decision_making.inputs import Inputs
from BT_decision_making.bt import tick_mission, tick_target


class FakeClock:
    def __init__(self, start_s: float = 0.0):
        self.t = float(start_s)

    def now(self) -> float:
        return self.t

    def set(self, t: float) -> None:
        self.t = float(t)

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def make_blackboard(start_mode: str = "IDLE", start_time_s: float = 0.0):
    clock = FakeClock(start_time_s)
    bb = Blackboard()
    bb.mode = start_mode
    bb.prev_mode = start_mode
    bb.start_time_s = start_time_s
    bb.mode_enter_s = start_time_s
    bb.now = clock.now
    return bb, clock


def print_result(name: str, passed: bool, expected: str, actual: str, extra: str = ""):
    status = "PASS" if passed else "FAIL"
    line = f"[{status}] {name} | expected={expected} | actual={actual}"
    if extra:
        line += f" | {extra}"
    print(line)


def run_mission_case(
    name: str,
    *,
    start_mode: str,
    inp: Inputs,
    expected_mode: str,
    expected_transition_contains: str | None = None,
):
    bb, clock = make_blackboard(start_mode=start_mode, start_time_s=0.0)
    out, transition = tick_mission(bb, inp)

    passed = bb.mode == expected_mode
    if expected_transition_contains is not None:
        passed = passed and (expected_transition_contains in transition)

    print_result(
        name=name,
        passed=passed,
        expected=expected_mode,
        actual=bb.mode,
        extra=f"transition='{transition}', action_mode='{out.mission_mode}', target_mode='{out.target_mode}'",
    )
    return passed


def run_target_case(
    name: str,
    *,
    start_target_state: str,
    steps: list[tuple[float, Inputs]],
    expected_target_state: str,
):
    bb, clock = make_blackboard(start_mode="SEARCH_AREA", start_time_s=1.0)
    bb.target_state = start_target_state
    bb.last_seen_s = -1.0
    bb.detect_stable_since_s = -1.0

    last_transition = ""
    last_out = None

    for t, inp in steps:
        clock.set(t)
        last_out, last_transition, _ = tick_target(bb, inp)

    passed = bb.target_state == expected_target_state

    print_result(
        name=name,
        passed=passed,
        expected=expected_target_state,
        actual=bb.target_state,
        extra=f"transition='{last_transition}', action_mode='{last_out.mission_mode}', target_mode='{last_out.target_mode}'",
    )
    return passed


def main():
    total = 0
    passed = 0

    def check(result: bool):
        nonlocal total, passed
        total += 1
        if result:
            passed += 1

    print("=== Mission-level BT validation ===")

    check(run_mission_case(
        "Mission upload starts download",
        start_mode="IDLE",
        inp=Inputs(mission_upload_received=True),
        expected_mode="DOWNLOAD_MISSION",
        expected_transition_contains="IDLE -> DOWNLOAD_MISSION",
    ))

    check(run_mission_case(
        "Mission complete data stored",
        start_mode="DOWNLOAD_MISSION",
        inp=Inputs(mission_data_complete=True),
        expected_mode="STORE_MISSION_DATA",
        expected_transition_contains="DOWNLOAD_MISSION -> STORE_MISSION_DATA",
    ))

    check(run_mission_case(
        "Request missing data times out to idle",
        start_mode="REQUEST_MISSING_DATA",
        inp=Inputs(request_missing_data_timeout=True),
        expected_mode="IDLE",
        expected_transition_contains="REQUEST_MISSING_DATA -> IDLE",
    ))

    check(run_mission_case(
        "Initialise with good checks coordinates swarm",
        start_mode="INITIALISE",
        inp=Inputs(
            battery_ok=True,
            fc_ok=True,
            nav_ok=True,
            comms_ok=True,
            payload_ok=True,
            obc_ok=True,
            mission_valid=True,
        ),
        expected_mode="COORDINATE_SWARM",
        expected_transition_contains="INITIALISE -> COORDINATE_SWARM",
    ))

    check(run_mission_case(
        "Swarm coordinated enters determine roles",
        start_mode="COORDINATE_SWARM",
        inp=Inputs(swarm_coordinated=True),
        expected_mode="DETERMINE_ROLES",
        expected_transition_contains="COORDINATE_SWARM -> DETERMINE_ROLES",
    ))

    check(run_mission_case(
        "Roles assigned enters wait launch",
        start_mode="DETERMINE_ROLES",
        inp=Inputs(roles_assigned=True, priority_list_sent=True),
        expected_mode="WAIT_LAUNCH",
        expected_transition_contains="DETERMINE_ROLES -> WAIT_LAUNCH",
    ))

    check(run_mission_case(
        "Launch command enters takeoff arm",
        start_mode="WAIT_LAUNCH",
        inp=Inputs(launch_command_received=True),
        expected_mode="TAKEOFF_ARM",
        expected_transition_contains="WAIT_LAUNCH -> TAKEOFF_ARM",
    ))

    check(run_mission_case(
        "Arm permission enters takeoff target set",
        start_mode="TAKEOFF_ARM",
        inp=Inputs(arm_permission_received=True),
        expected_mode="TAKEOFF_SET_ALTITUDE_TARGET",
        expected_transition_contains="TAKEOFF_ARM -> TAKEOFF_SET_ALTITUDE_TARGET",
    ))

    check(run_mission_case(
        "Altitude reached enters hover stabilise",
        start_mode="TAKEOFF_CLIMB",
        inp=Inputs(altitude_reached=True),
        expected_mode="HOVER_STABILISE",
        expected_transition_contains="TAKEOFF_CLIMB -> HOVER_STABILISE",
    ))

    check(run_mission_case(
        "Hover stable enters fly to search area",
        start_mode="HOVER_STABILISE",
        inp=Inputs(hover_stable=True),
        expected_mode="FLY_TO_SEARCH_AREA",
        expected_transition_contains="HOVER_STABILISE -> FLY_TO_SEARCH_AREA",
    ))

    check(run_mission_case(
        "Arrival enters search area",
        start_mode="FLY_TO_SEARCH_AREA",
        inp=Inputs(arrived_search_area=True),
        expected_mode="SEARCH_AREA",
        expected_transition_contains="FLY_TO_SEARCH_AREA -> SEARCH_AREA",
    ))

    check(run_mission_case(
        "Stable target in search area enters verify target",
        start_mode="SEARCH_AREA",
        inp=Inputs(
            target_detected=True,
            target_conf=0.90,
            target_offset_px=(10.0, 5.0),
        ),
        expected_mode="SEARCH_AREA",
        expected_transition_contains=None,
    ))

    bb, clock = make_blackboard(start_mode="SEARCH_AREA", start_time_s=1.0)
    inp_detect = Inputs(target_detected=True, target_conf=0.90, target_offset_px=(10.0, 5.0))
    clock.set(1.0)
    tick_mission(bb, inp_detect)
    clock.set(1.7)
    out, transition = tick_mission(bb, inp_detect)
    result = bb.mode == "VERIFY_TARGET"
    print_result(
        "Debounced target detection enters verify target",
        result,
        "VERIFY_TARGET",
        bb.mode,
        extra=f"transition='{transition}', action_mode='{out.mission_mode}', target_mode='{out.target_mode}'",
    )
    check(result)

    check(run_mission_case(
        "Verified target enters target report",
        start_mode="VERIFY_TARGET",
        inp=Inputs(target_confirmed=True),
        expected_mode="TARGET_REPORT",
        expected_transition_contains="VERIFY_TARGET -> TARGET_REPORT",
    ))

    check(run_mission_case(
        "Report sent enters converge drones",
        start_mode="TARGET_REPORT",
        inp=Inputs(report_sent=True),
        expected_mode="CONVERGE_DRONES",
        expected_transition_contains="TARGET_REPORT -> CONVERGE_DRONES",
    ))

    check(run_mission_case(
        "Converge complete enters target tracking",
        start_mode="CONVERGE_DRONES",
        inp=Inputs(converge_complete=True),
        expected_mode="TARGET_TRACKING",
        expected_transition_contains="CONVERGE_DRONES -> TARGET_TRACKING",
    ))

    check(run_mission_case(
        "Target lost enters search last known",
        start_mode="TARGET_TRACKING",
        inp=Inputs(target_detected=False),
        expected_mode="SEARCH_LAST_KNOWN_LOCATION",
        expected_transition_contains="TARGET_TRACKING -> SEARCH_LAST_KNOWN_LOCATION",
    ))

    check(run_mission_case(
        "Mission complete returns to base",
        start_mode="TARGET_TRACKING",
        inp=Inputs(target_detected=True, mission_complete=True),
        expected_mode="RETURN_TO_BASE",
        expected_transition_contains="TARGET_TRACKING -> RETURN_TO_BASE",
    ))

    bb, clock = make_blackboard(start_mode="SEARCH_LAST_KNOWN_LOCATION", start_time_s=1.0)
    bb.search_last_known_start_s = 1.0
    inp_reacquire = Inputs(target_detected=True, target_conf=0.90, target_offset_px=(5.0, 0.0))
    clock.set(1.0)
    tick_mission(bb, inp_reacquire)
    clock.set(1.7)
    out, transition = tick_mission(bb, inp_reacquire)
    result = bb.mode == "TARGET_TRACKING"
    print_result(
        "Reacquired target returns to tracking",
        result,
        "TARGET_TRACKING",
        bb.mode,
        extra=f"transition='{transition}', action_mode='{out.mission_mode}', target_mode='{out.target_mode}'",
    )
    check(result)

    check(run_mission_case(
        "Search last known timeout returns search area",
        start_mode="SEARCH_LAST_KNOWN_LOCATION",
        inp=Inputs(search_last_known_timeout=True),
        expected_mode="SEARCH_AREA",
        expected_transition_contains="SEARCH_LAST_KNOWN_LOCATION -> SEARCH_AREA",
    ))

    check(run_mission_case(
        "Parent loss enters reassigned state",
        start_mode="SEARCH_AREA",
        inp=Inputs(parent_lost=True),
        expected_mode="PARENT_REASSIGNED",
        expected_transition_contains="SEARCH_AREA -> PARENT_REASSIGNED",
    ))

    check(run_mission_case(
        "Parent reassigned resumes previous mission",
        start_mode="PARENT_REASSIGNED",
        inp=Inputs(parent_reassigned=True),
        expected_mode="SEARCH_AREA",
        expected_transition_contains="PARENT_REASSIGNED -> SEARCH_AREA",
    ))

    bb, clock = make_blackboard(start_mode="PARENT_REASSIGNED", start_time_s=0.0)
    bb.prev_mode = "SEARCH_AREA"
    out, transition = tick_mission(bb, Inputs(parent_reassigned=True))
    result = bb.mode == "SEARCH_AREA"
    print_result(
        "Parent reassigned restores previous mode",
        result,
        "SEARCH_AREA",
        bb.mode,
        extra=f"transition='{transition}', action_mode='{out.mission_mode}', target_mode='{out.target_mode}'",
    )
    check(result)

    check(run_mission_case(
        "Comms lost enters lost communications",
        start_mode="SEARCH_AREA",
        inp=Inputs(comms_lost=True),
        expected_mode="LOST_COMMUNICATIONS",
        expected_transition_contains="SEARCH_AREA -> LOST_COMMUNICATIONS",
    ))

    bb, clock = make_blackboard(start_mode="LOST_COMMUNICATIONS", start_time_s=0.0)
    bb.prev_mode = "SEARCH_AREA"
    bb.comms_lost_since_s = 0.0
    out, transition = tick_mission(bb, Inputs(comms_restored=True))
    result = bb.mode == "SEARCH_AREA"
    print_result(
        "Comms restored resumes previous mode",
        result,
        "SEARCH_AREA",
        bb.mode,
        extra=f"transition='{transition}', action_mode='{out.mission_mode}', target_mode='{out.target_mode}'",
    )
    check(result)

    check(run_mission_case(
        "Comms timeout returns to base",
        start_mode="LOST_COMMUNICATIONS",
        inp=Inputs(comms_timeout=True),
        expected_mode="RETURN_TO_BASE",
        expected_transition_contains="LOST_COMMUNICATIONS -> RETURN_TO_BASE",
    ))

    check(run_mission_case(
        "Recall returns to base",
        start_mode="SEARCH_AREA",
        inp=Inputs(recall_received=True),
        expected_mode="RETURN_TO_BASE",
        expected_transition_contains="SEARCH_AREA -> RETURN_TO_BASE",
    ))

    check(run_mission_case(
        "Critical fault enters emergency landing",
        start_mode="SEARCH_AREA",
        inp=Inputs(low_battery_critical=True),
        expected_mode="EMERGENCY_LANDING",
        expected_transition_contains="SEARCH_AREA -> EMERGENCY_LANDING",
    ))

    check(run_mission_case(
        "Failsafe request latches failsafe",
        start_mode="SEARCH_AREA",
        inp=Inputs(failsafe_requested=True),
        expected_mode="FAILSAFE",
        expected_transition_contains="SEARCH_AREA -> FAILSAFE",
    ))

    check(run_mission_case(
        "Manual reset exits failsafe",
        start_mode="FAILSAFE",
        inp=Inputs(manual_reset=True),
        expected_mode="IDLE",
        expected_transition_contains="FAILSAFE -> IDLE",
    ))

    check(run_mission_case(
        "At base enters landing",
        start_mode="RETURN_TO_BASE",
        inp=Inputs(at_base=True),
        expected_mode="LANDING",
        expected_transition_contains="RETURN_TO_BASE -> LANDING",
    ))

    check(run_mission_case(
        "Landed returns to idle",
        start_mode="LANDING",
        inp=Inputs(landed=True),
        expected_mode="IDLE",
        expected_transition_contains="LANDING -> IDLE",
    ))

    print("\n=== Target sub-state validation ===")

    check(run_target_case(
        "SEARCH to TRACK after stable detection",
        start_target_state="SEARCH",
        steps=[
            (1.0, Inputs(target_detected=True, target_conf=0.90, target_offset_px=(20.0, 0.0))),
            (1.7, Inputs(target_detected=True, target_conf=0.90, target_offset_px=(20.0, 0.0))),
        ],
        expected_target_state="TRACK",
    ))

    check(run_target_case(
        "TRACK to LOST when detection disappears",
        start_target_state="TRACK",
        steps=[
            (1.0, Inputs(target_detected=False, target_conf=0.0)),
        ],
        expected_target_state="LOST",
    ))

    check(run_target_case(
        "LOST to TRACK when target is reacquired",
        start_target_state="LOST",
        steps=[
            (1.0, Inputs(target_detected=True, target_conf=0.90, target_offset_px=(15.0, 0.0))),
            (1.7, Inputs(target_detected=True, target_conf=0.90, target_offset_px=(15.0, 0.0))),
        ],
        expected_target_state="TRACK",
    ))

    bb, clock = make_blackboard(start_mode="SEARCH_AREA", start_time_s=1.0)
    bb.target_state = "LOST"
    bb.last_seen_s = 1.0
    clock.set(4.2)
    out, transition, _ = tick_target(bb, Inputs(target_detected=False, target_conf=0.0))
    result = bb.target_state == "SEARCH"
    print_result(
        "LOST to SEARCH after timeout",
        result,
        "SEARCH",
        bb.target_state,
        extra=f"transition='{transition}', action_mode='{out.mission_mode}', target_mode='{out.target_mode}'",
    )
    check(result)

    print("\n=== BT validation summary ===")
    print(f"Passed {passed} / {total} tests")

    if passed == total:
        print("All BT validation tests passed.")
    else:
        print("Some BT validation tests failed.")


if __name__ == "__main__":
    main()