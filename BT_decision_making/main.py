import time
from dataclasses import dataclass

from bt_nodes import Action, Condition, Selector, Sequence, Status


@dataclass
class WorldState:
    target_visible: bool = False
    target_last_seen_t: float = 0.0
    have_path: bool = False
    battery_ok: bool = True


class Controller:
    """
    Placeholder actions. Later, connect these to:
    - YOLO detections (target_visible)
    - A* planner (have_path + path)
    - Drone commands (follow/hover)
    """
    def __init__(self, state: WorldState):
        self.s = state
        self.start_t = time.time()

    def now(self) -> float:
        return time.time()

    # -------- Conditions --------
    def c_battery_ok(self) -> bool:
        return self.s.battery_ok

    def c_target_visible(self) -> bool:
        return self.s.target_visible

    def c_target_lost_too_long(self) -> bool:
        # if not seen for > 3 seconds -> lost
        if self.s.target_visible:
            return False
        return (self.now() - self.s.target_last_seen_t) > 3.0

    def c_have_path(self) -> bool:
        return self.s.have_path

    # -------- Actions --------
    def a_search(self) -> Status:
        # Simulate searching: after ~2 seconds, "detect" target
        if (self.now() - self.start_t) > 2.0 and not self.s.target_visible:
            self.s.target_visible = True
            self.s.target_last_seen_t = self.now()
            print("[SEARCH] Target found.")
            return Status.SUCCESS

        print("[SEARCH] Scanning...")
        return Status.RUNNING

    def a_track(self) -> Status:
        # Simulate tracking: keep target visible for a bit, then lose it sometimes
        if self.s.target_visible:
            self.s.target_last_seen_t = self.now()
            print("[TRACK] Tracking target (keeping it centred).")

            # After ~6 seconds from start, simulate losing target once
            if (self.now() - self.start_t) > 6.0:
                self.s.target_visible = False
                print("[TRACK] Lost target.")
                return Status.FAILURE

            return Status.RUNNING

        return Status.FAILURE

    def a_plan_path(self) -> Status:
        # Simulate path planning being quick
        if not self.s.have_path:
            self.s.have_path = True
            print("[PLAN] Path generated.")
        return Status.SUCCESS

    def a_follow_path(self) -> Status:
        # Simulate following path for a short time then finishing
        print("[FOLLOW] Following planned path...")
        time.sleep(0.1)
        # Pretend we reached where we wanted
        self.s.have_path = False
        return Status.SUCCESS

    def a_hover_failsafe(self) -> Status:
        print("[FAILSAFE] Hovering / safe state.")
        return Status.RUNNING


def build_tree(ctrl: Controller):
    # Fail-safe branch if target is lost too long
    failsafe = Sequence("FailsafeSequence", [
        Condition("TargetLostTooLong", ctrl.c_target_lost_too_long),
        Action("HoverFailsafe", ctrl.a_hover_failsafe),
    ])

    # Main engagement: if target visible -> track + plan + follow
    engage = Sequence("EngageSequence", [
        Condition("TargetVisible", ctrl.c_target_visible),
        Action("TrackTarget", ctrl.a_track),
        Action("PlanPath", ctrl.a_plan_path),
        Action("FollowPath", ctrl.a_follow_path),
    ])

    # Search branch: if not visible -> search
    search = Action("Search", ctrl.a_search)

    # Top level: failsafe OR engage OR search
    root = Selector("Root", [
        failsafe,
        engage,
        search,
    ])
    return root


def main():
    state = WorldState(target_visible=False, target_last_seen_t=time.time(), have_path=False, battery_ok=True)
    ctrl = Controller(state)
    tree = build_tree(ctrl)

    print("Running BT loop (Ctrl+C to stop)...\n")
    try:
        while True:
            status = tree.tick()
            # You could log status here if you want
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
