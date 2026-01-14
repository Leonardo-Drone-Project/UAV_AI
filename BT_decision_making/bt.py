# BT_decision_making/bt.py
"""
BT decision logic with:
- debouncing (prevents flicker spam)
- LOST timeout
- FAILSAFE timeout
"""
from typing import Tuple
from blackboard import Blackboard
from inputs import Inputs
from actions import (
    ActionOutputs,
    search_action,
    track_action,
    lost_action,
    failsafe_action,
)


def tick_bt(
    bb: Blackboard,
    inp: Inputs,
    *,
    conf_threshold: float = 0.7,
    debounce_s: float = 0.3,
    lost_to_search_s: float = 2.0,
    failsafe_s: float = 12.0,
) -> Tuple[ActionOutputs, str]:
    """
    Returns: (action_outputs, transition_string)
    transition_string is "" if no transition occurred.
    """
    now_s = bb.now()
    transition = ""

    detected_good = inp.target_detected and (inp.target_conf >= conf_threshold)

    # Update last seen / debounce timers
    if detected_good:
        bb.last_seen_s = now_s
        if bb.detect_stable_since_s < 0:
            bb.detect_stable_since_s = now_s
    else:
        bb.detect_stable_since_s = -1.0

    time_since_seen = (now_s - bb.last_seen_s) if bb.last_seen_s > 0 else 1e9
    stable_for = (now_s - bb.detect_stable_since_s) if bb.detect_stable_since_s > 0 else 0.0

    prev_state = bb.state

    # FAILSAFE rule: no target for too long (global)
    if time_since_seen >= failsafe_s:
        bb.state = "FAILSAFE"

    # State transitions
    if bb.state == "SEARCH":
        if detected_good and stable_for >= debounce_s:
            bb.state = "TRACK"

    elif bb.state == "TRACK":
        if not detected_good:
            bb.state = "LOST"

    elif bb.state == "LOST":
        if detected_good and stable_for >= debounce_s:
            bb.state = "TRACK"
        elif time_since_seen >= lost_to_search_s:
            bb.state = "SEARCH"

    elif bb.state == "FAILSAFE":
        # In FAILSAFE, only manual reset should exit (keep it simple)
        pass

    if bb.state != prev_state:
        transition = f"{prev_state} -> {bb.state}"

    # Actions
    if bb.state == "SEARCH":
        out = search_action()
    elif bb.state == "TRACK":
        dx, dy = inp.target_offset_px
        out = track_action(dx, dy)
    elif bb.state == "LOST":
        out = lost_action()
    else:
        out = failsafe_action()

    return out, transition

