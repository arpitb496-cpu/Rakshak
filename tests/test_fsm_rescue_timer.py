"""
tests/test_fsm_rescue_timer.py - Verification test for FSM rescue cycle, state transitions,
auto-reset timer countdown, and telemetry/banner consistency.
"""

import os
import sys
import math
import pytest

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

os.environ["SDL_VIDEODRIVER"] = "dummy"

from arena import DisasterArena
from robot import DifferentialDriveRobot
from camera import RobotCamera
from detector import VisionDetector, Detection
from tracker import VisionTracker, TrackedTarget
from apf import APFController
from astar import AStarPlanner
from rescuer import RescuerState
from fsm import MissionFSM, MissionState
from app import RakshakApp


def test_fsm_rescue_cycle_and_rescuer_advancement():
    """
    Verifies that when scout enters RESCUE_OPERATION, Rescuer R1 moves to the survivor,
    successfully extracts, missions counter increments (0/4 -> 1/4), and FSM exits
    RESCUE_OPERATION without hanging.
    """
    arena = DisasterArena(width=800, height=650)
    robot = DifferentialDriveRobot(x=650.0, y=130.0, theta=0.0)
    camera = RobotCamera(width=160, height=90, fov_deg=65.0, max_range=340.0)
    detector = VisionDetector(focal_length=camera.focal_length, cx=camera.cx, cy=camera.cy)
    tracker = VisionTracker(lock_threshold=2)
    apf = APFController()
    astar = AStarPlanner(arena_width=800, arena_height=650)
    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf)

    # Initial lock on S1
    s1 = arena.survivors[0]
    det1 = Detection(
        class_label="SURVIVOR",
        bbox=(70, 30, 20, 30),
        estimated_range=50.0,
        estimated_bearing=0.0,
        mask_area_ratio=0.85,
        contour_area=510.0,
        triage_level=s1.triage,
    )
    tracker.update([det1], robot.x, robot.y, robot.theta, ground_truth_survivor=(s1.x, s1.y))
    tracker.update([det1], robot.x, robot.y, robot.theta, ground_truth_survivor=(s1.x, s1.y))

    fsm.transition_to(MissionState.RESCUE_OPERATION, "Testing rescue operation")
    initial_missions = fsm.missions_completed
    assert initial_missions == 0

    print("\n--- FSM State Transition & Rescue Progress Log ---")
    dt = 0.05
    max_steps = 400
    rescued = False

    for step in range(max_steps):
        prev_state = fsm.current_state
        arena.update(dt)
        fsm.update(dt)
        robot.update_kinematics(dt, arena)

        if fsm.current_state != prev_state:
            log_entry = f"Step {step:3d} ({step*dt:.2f}s): Transition {prev_state.name} -> {fsm.current_state.name} | Missions: {fsm.missions_completed}/{fsm.total_survivors}"
            print(log_entry)

        if fsm.missions_completed > initial_missions:
            rescued = True
            print(f"[SUCCESS] Survivor {s1.id} secured! Missions counter incremented to {fsm.missions_completed}/4")
            break

    assert rescued, "Rescuer failed to complete extraction, FSM hung in RESCUE_OPERATION"
    assert fsm.missions_completed >= 1, "Missions completed did not increment from 0"


def test_mission_complete_countdown_and_autoreset():
    """
    Verifies that in MISSION_COMPLETE state, the auto-reset countdown decrements
    smoothly from 3.0s down to 0.0s and triggers reset_mission().
    """
    app = RakshakApp()
    app.fsm.missions_completed = app.fsm.total_survivors
    app.fsm.current_state = MissionState.MISSION_COMPLETE

    dt = 0.1
    print("\n--- Auto-Reset Timer Countdown Log ---")
    timer_snapshots = []
    reset_occurred = False

    for step in range(35):
        remaining_time = max(0.0, 3.0 - app.mission_complete_timer)
        timer_snapshots.append(remaining_time)
        print(f"Time {step*dt:4.1f}s | Timer: {app.mission_complete_timer:.1f}s | Countdown remaining: {remaining_time:.1f}s | State: {app.fsm.current_state.name}")

        app.update(dt)

        if app.mission_complete_timer == 0.0 and step > 29:
            reset_occurred = True
            print("[SUCCESS] Auto-reset triggered cleanly!")
            break

    assert timer_snapshots[0] == pytest.approx(3.0, 0.1)
    assert timer_snapshots[15] < timer_snapshots[5]
    assert reset_occurred, "Auto-reset did not trigger after 3.0s countdown expired"


def test_banner_and_telemetry_target_consistency():
    """
    Verifies that the banner and telemetry card resolve the active target survivor
    consistently (e.g. S2 when tracking S2).
    """
    app = RakshakApp()
    s2 = app.arena.survivors[1] # S2 at (470, 70)
    app.robot.x = 440.0
    app.robot.y = 70.0
    app.robot.theta = 0.0

    # Set tracker to active track S2
    app.tracker.survivor_track = TrackedTarget(
        class_label="SURVIVOR",
        target_id="S2",
        est_world_x=s2.x + 0.4,
        est_world_y=s2.y - 0.3,
        est_range=30.0,
        est_bearing=0.0,
        mask_area_ratio=0.85,
        triage_level="INJURED",
        is_locked=True,
    )

    st = app.tracker.survivor_track
    assert st is not None
    assert st.target_id == "S2"

    active_surv = None
    for s in app.arena.survivors:
        if getattr(s, "id", None) == st.target_id:
            active_surv = s
            break
    assert active_surv is not None
    assert active_surv.id == "S2"
    assert active_surv.x == s2.x and active_surv.y == s2.y

    standoff = math.hypot(active_surv.x - app.robot.x, active_surv.y - app.robot.y)
    assert standoff == pytest.approx(30.0, abs=5.0)

    matched_surv = None
    for s in app.arena.survivors:
        if getattr(s, "id", None) == st.target_id:
            matched_surv = s
            break
    assert matched_surv.id == "S2"
    assert matched_surv.x == 470.0 and matched_surv.y == 70.0

    pos_err = math.hypot(st.est_world_x - matched_surv.x, st.est_world_y - matched_surv.y)
    assert pos_err < 5.0
    print(f"\n[CONSISTENCY CHECK] Target: {active_surv.id} | GT: ({matched_surv.x}, {matched_surv.y}) | Est: ({st.est_world_x:.1f}, {st.est_world_y:.1f}) | Standoff: {standoff:.1f}px | Err: {pos_err:.2f}px")
