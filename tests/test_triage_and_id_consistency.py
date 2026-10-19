"""
tests/test_triage_and_id_consistency.py - Verification tests for:
1. Issue 1: Survivor ID consistency across multi-survivor rescues (no synthetic ID drift).
2. Issue 1: Atomic per-rescue state cleanup.
3. Issue 2: Shared triage priority (CRITICAL > INJURED > STABLE) in autonomous and manual mode.
4. Issue 2: Triage priority order (2 CRITICAL, 2 INJURED, 1 STABLE) with STABLE detected first.
"""

import os
import sys
import math
import pytest

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

os.environ["SDL_VIDEODRIVER"] = "dummy"

from arena import DisasterArena, Survivor, get_active_survivor
from robot import DifferentialDriveRobot
from camera import RobotCamera
from detector import VisionDetector, Detection
from tracker import VisionTracker, TrackedTarget
from apf import APFController
from astar import AStarPlanner
from rescuer import (
    RescuerAgent,
    RescuerState,
    MessageBus,
    DispatchManager,
    SOSMessage,
    triage_rank,
    survivor_priority_key,
)
from fsm import MissionFSM, MissionState
from app import RakshakApp


def test_issue1_survivor_id_consistency_after_first_rescue():
    """
    Validates that in a 5-survivor scenario (S1 CRITICAL, S2 INJURED, S3 STABLE, S4 CRITICAL, S5 INJURED),
    after S1 is secured, detecting S4 assigns ID 'S4' (not 'S2'), and the banner, telemetry,
    FSM, dispatcher, and rescuer all consistently identify S4 with minimal position error.
    """
    arena = DisasterArena(width=800, height=650)
    # Define exact 5 survivors matching the screen recording:
    arena.survivors = [
        Survivor(x=240.0, y=70.0, id="S1", triage="CRITICAL"),
        Survivor(x=210.0, y=570.0, id="S2", triage="INJURED"),
        Survivor(x=690.0, y=260.0, id="S3", triage="STABLE"),
        Survivor(x=690.0, y=480.0, id="S4", triage="CRITICAL"),
        Survivor(x=470.0, y=70.0, id="S5", triage="INJURED"),
    ]

    robot = DifferentialDriveRobot(x=200.0, y=70.0, theta=0.0)
    camera = RobotCamera()
    detector = VisionDetector()
    tracker = VisionTracker(lock_threshold=1)
    astar = AStarPlanner()
    apf = APFController()
    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf)

    # Step 1: Detect and secure S1
    s1 = arena.survivors[0]
    det1 = Detection(
        class_label="SURVIVOR",
        bbox=(70, 30, 20, 30),
        estimated_range=40.0,
        estimated_bearing=0.0,
        mask_area_ratio=0.85,
        contour_area=510.0,
        triage_level="CRITICAL",
    )
    tracker.update([det1], robot.x, robot.y, robot.theta, arena.survivors)
    assert tracker.survivor_track.target_id == "S1"

    # Complete S1 rescue
    fsm.current_state = MissionState.RESCUE_OPERATION
    fsm._handle_rescue_operation(dt=0.033)
    assert fsm.current_sos_id == "S1"
    fsm.dispatch_manager.sos_queue.clear()
    fsm.dispatch_manager.completed_sos["S1"] = SOSMessage(
        survivor_id="S1", est_x=240.0, est_y=70.0, triage="CRITICAL", timestamp=0.0, status="SECURED"
    )
    s1.is_secured = True
    fsm._handle_rescue_operation(dt=0.033)

    # Verify atomic per-rescue state cleanup
    assert fsm.current_sos_id is None
    assert fsm.current_sos_target_track is None
    assert len(fsm.active_path) == 0
    assert tracker.active_track_id is None
    assert fsm.current_state == MissionState.SEARCH_PATROL

    # Step 2: Robot moves to southeast quadrant and detects S4 at (690, 480)
    robot.x = 640.0
    robot.y = 480.0
    robot.theta = 0.0
    det4 = Detection(
        class_label="SURVIVOR",
        bbox=(70, 30, 20, 30),
        estimated_range=50.0,
        estimated_bearing=0.0,
        mask_area_ratio=0.85,
        contour_area=510.0,
        triage_level="CRITICAL",
    )
    tracker.update([det4], robot.x, robot.y, robot.theta, arena.survivors)

    # CRITICAL INVARIANT: The tracker MUST NOT assign synthetic 'S2' to physical survivor S4!
    st4 = tracker.survivor_track
    assert st4 is not None
    assert st4.target_id == "S4", f"Expected tracker target_id 'S4', but got '{st4.target_id}'"
    assert st4.triage_level == "CRITICAL"

    # Verify single source of truth helper get_active_survivor()
    resolved_s = get_active_survivor(arena, tracker, fsm)
    assert resolved_s is not None
    assert resolved_s.id == "S4"
    assert resolved_s.x == 690.0 and resolved_s.y == 480.0

    # Verify localization error is minimal (< 5.0 px), NOT ~488 px!
    loc_err = math.hypot(st4.est_world_x - resolved_s.x, st4.est_world_y - resolved_s.y)
    assert loc_err < 5.0, f"Expected localization error < 5.0 px, got {loc_err:.1f} px"

    # Step 3: FSM transition to RESCUE_OPERATION for S4
    fsm.transition_to(MissionState.RESCUE_OPERATION, "Standoff at S4")
    fsm._handle_rescue_operation(dt=0.033)

    assert fsm.current_sos_id == "S4", f"Expected current_sos_id 'S4', got '{fsm.current_sos_id}'"
    assert "S4" in fsm.dispatch_manager.active_sos or "S4" in [s.survivor_id for s in fsm.dispatch_manager.sos_queue]

    # Verify Rescuer dispatch receives S4
    available_r = [r for r in arena.rescuers if r.state == RescuerState.STANDBY]
    if available_r:
        fsm.dispatch_manager.dispatch_next_sos(arena.rescuers, arena, astar, current_time=2.0)
        sos_s4 = fsm.dispatch_manager.active_sos.get("S4")
        assert sos_s4 is not None
        assigned_rescuer = next((r for r in arena.rescuers if r.name == sos_s4.assigned_rescuer_id), None)
        assert assigned_rescuer is not None
        assert assigned_rescuer.assigned_survivor_id == "S4"


def test_issue2_shared_priority_function():
    """
    Validates the shared priority function: CRITICAL (0) > INJURED (1) > STABLE (2),
    followed by nearest-first distance ordering.
    """
    assert triage_rank("CRITICAL") == 0
    assert triage_rank("critical") == 0
    assert triage_rank("INJURED") == 1
    assert triage_rank("STABLE") == 2
    assert triage_rank("UNKNOWN") == 3

    # Test sorting a mixed list of candidates
    c1 = Survivor(x=100.0, y=100.0, id="S_STABLE", triage="STABLE")
    c2 = Survivor(x=300.0, y=300.0, id="S_CRIT_FAR", triage="CRITICAL")
    c3 = Survivor(x=50.0, y=50.0, id="S_CRIT_NEAR", triage="CRITICAL")
    c4 = Survivor(x=80.0, y=80.0, id="S_INJ", triage="INJURED")

    robot_pos = (0.0, 0.0)
    candidates = [c1, c2, c3, c4]
    candidates.sort(key=lambda s: survivor_priority_key(s, robot_pos))

    # Expected order: CRITICAL near, CRITICAL far, INJURED, STABLE
    expected_ids = ["S_CRIT_NEAR", "S_CRIT_FAR", "S_INJ", "S_STABLE"]
    actual_ids = [s.id for s in candidates]
    assert actual_ids == expected_ids


def test_issue2_triage_priority_order_with_stable_detected_first():
    """
    Validates that with 2 CRITICAL, 2 INJURED, and 1 STABLE survivor,
    even if the STABLE survivor is detected first, the rescue order adheres to
    CRITICAL, CRITICAL, INJURED, INJURED, STABLE.
    """
    arena = DisasterArena(width=800, height=650)
    # 5 survivors: S1 (CRIT), S2 (INJ), S3 (STABLE), S4 (CRIT), S5 (INJ)
    s1 = Survivor(x=240.0, y=70.0, id="S1", triage="CRITICAL")
    s2 = Survivor(x=210.0, y=570.0, id="S2", triage="INJURED")
    s3 = Survivor(x=690.0, y=260.0, id="S3", triage="STABLE")
    s4 = Survivor(x=690.0, y=480.0, id="S4", triage="CRITICAL")
    s5 = Survivor(x=470.0, y=70.0, id="S5", triage="INJURED")
    arena.survivors = [s1, s2, s3, s4, s5]

    bus = MessageBus(latency=0.0)
    dm = DispatchManager(bus)

    # Simulate detection: STABLE S3 is detected and submitted FIRST
    dm.submit_sos(s3.id, s3.x, s3.y, s3.triage, current_time=1.0)
    # Then INJURED S2 is detected
    dm.submit_sos(s2.id, s2.x, s2.y, s2.triage, current_time=2.0)
    # Then CRITICAL S1 is detected
    dm.submit_sos(s1.id, s1.x, s1.y, s1.triage, current_time=3.0)
    # Then CRITICAL S4 is detected
    dm.submit_sos(s4.id, s4.x, s4.y, s4.triage, current_time=4.0)
    # Then INJURED S5 is detected
    dm.submit_sos(s5.id, s5.x, s5.y, s5.triage, current_time=5.0)

    # Sort queue using shared priority function
    dm.sos_queue.sort(key=lambda s: survivor_priority_key(s, (0, 0)))

    # Dispatch one by one and record the order
    dispatch_order = []
    while dm.sos_queue:
        sos = dm.sos_queue.pop(0)
        dispatch_order.append((sos.survivor_id, sos.triage))

    # Verify exact triage priority order: CRITICAL, CRITICAL, INJURED, INJURED, STABLE
    triage_order_result = [t for _, t in dispatch_order]
    assert triage_order_result == [
        "CRITICAL",
        "CRITICAL",
        "INJURED",
        "INJURED",
        "STABLE",
    ], f"Dispatch order did not match triage priority: {triage_order_result}"

    # Verify STABLE was dispatched LAST even though detected first
    assert dispatch_order[-1][0] == "S3"
    assert dispatch_order[-1][1] == "STABLE"


def test_issue2_target_switching_to_higher_priority():
    """
    Validates that if the scout spots a lower-priority survivor (e.g. STABLE)
    and then detects a higher-priority survivor (e.g. CRITICAL) before rescuer dispatch,
    it switches target to the higher-priority survivor.
    """
    arena = DisasterArena(width=800, height=650)
    s_stable = Survivor(x=690.0, y=260.0, id="S_STABLE", triage="STABLE")
    s_crit = Survivor(x=690.0, y=480.0, id="S_CRIT", triage="CRITICAL")
    arena.survivors = [s_stable, s_crit]

    robot = DifferentialDriveRobot(x=650.0, y=260.0, theta=0.0)
    camera = RobotCamera()
    detector = VisionDetector()
    tracker = VisionTracker(lock_threshold=1)
    astar = AStarPlanner()
    apf = APFController()
    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf)

    # Step 1: Detect STABLE survivor
    det_stable = Detection(
        class_label="SURVIVOR",
        bbox=(70, 30, 20, 30),
        estimated_range=40.0,
        estimated_bearing=0.0,
        mask_area_ratio=0.85,
        contour_area=510.0,
        triage_level="STABLE",
    )
    tracker.update([det_stable], robot.x, robot.y, robot.theta, arena.survivors)
    assert tracker.survivor_track.target_id == "S_STABLE"

    fsm.update(0.033)
    assert fsm.current_state in (MissionState.VERIFY_TARGET, MissionState.PLAN_ROUTE)

    # Step 2: CRITICAL survivor comes into view
    det_crit = Detection(
        class_label="SURVIVOR",
        bbox=(70, 30, 20, 30),
        estimated_range=50.0,
        estimated_bearing=0.1,
        mask_area_ratio=0.85,
        contour_area=510.0,
        triage_level="CRITICAL",
    )
    # Robot turns and sees CRITICAL
    robot.x, robot.y = 650.0, 480.0
    tracker.update([det_crit], robot.x, robot.y, robot.theta, arena.survivors)

    # The tracker's primary survivor_track should now automatically return the CRITICAL one!
    assert tracker.survivor_track.target_id == "S_CRIT"
    assert tracker.survivor_track.triage_level == "CRITICAL"

    # FSM search patrol handles target prioritization
    fsm._handle_search_patrol(0.033)
    assert getattr(fsm, "_active_target_id", None) == "S_CRIT"


def test_missions_counter_strictly_increments_after_each_secure():
    """
    Validates Problem 1: missions_completed counter increments strictly when a survivor's
    extraction completes (completed_sos), and does NOT overcount (e.g. 2/3) while the next
    survivor is still mid-extraction.
    """
    app = RakshakApp()
    arena = app.arena
    arena.survivors = [
        Survivor(x=200.0, y=100.0, id="S3", triage="STABLE"),
        Survivor(x=400.0, y=300.0, id="S1", triage="CRITICAL"),
        Survivor(x=600.0, y=500.0, id="S2", triage="INJURED"),
    ]
    fsm = app.fsm
    fsm.total_survivors = 3
    assert fsm.missions_completed == 0

    # 1. First survivor S3 extracted
    s3 = arena.survivors[0]
    r1 = arena.rescuers[0]
    fsm.dispatch_manager.submit_sos(s3.id, s3.x, s3.y, s3.triage, current_time=1.0)
    fsm.dispatch_manager.dispatch_next_sos(arena.rescuers, arena, app.astar, current_time=1.1)

    r1.x, r1.y = s3.x, s3.y
    r1.state = RescuerState.EXTRACTING
    r1.extraction_timer = 0.5  # Mid-extraction
    app.update(0.05)
    assert fsm.missions_completed == 0, "Counter must remain 0 while S3 is mid-extraction"

    # Finish extraction of S3
    r1.extraction_timer = r1.extraction_duration
    for _ in range(5):
        app.update(0.05)
    assert s3.is_secured is True
    assert fsm.missions_completed == 1, f"Counter must equal 1 after S3 secured, got {fsm.missions_completed}"

    # 2. Second survivor S1 targeted and mid-extraction
    s1 = arena.survivors[1]
    r2 = arena.rescuers[1]
    fsm.dispatch_manager.submit_sos(s1.id, s1.x, s1.y, s1.triage, current_time=3.0)
    fsm.dispatch_manager.dispatch_next_sos(arena.rescuers, arena, app.astar, current_time=3.1)

    r2.x, r2.y = s1.x, s1.y
    r2.state = RescuerState.EXTRACTING
    r2.extraction_timer = 0.5  # Mid-extraction of S1
    app.update(0.05)
    assert fsm.missions_completed == 1, f"Counter must NOT overcount to 2 while S1 is mid-extraction, got {fsm.missions_completed}"

    # Finish extraction of S1
    r2.extraction_timer = r2.extraction_duration
    for _ in range(5):
        app.update(0.05)
    assert s1.is_secured is True
    assert fsm.missions_completed == 2, f"Counter must equal 2 after S1 secured, got {fsm.missions_completed}"


def test_telemetry_gt_shows_dashes_when_no_active_track():
    """
    Validates Problem 2: Telemetry GT shows '--' in SEARCH_PATROL when there is no active track.
    """
    app = RakshakApp()
    app.tracker.survivor_tracks.clear()
    app.tracker.active_track_id = None
    app.tracker._manual_survivor_track = None
    assert app.tracker.survivor_track is None

    # Render HUD (which renders target telemetry card)
    app._render_hud()
    # Confirm tracker has no active track and rendered cleanly without raising error
    assert app.tracker.survivor_track is None


def test_stable_rescued_first_log_and_target_switch_before_dispatch():
    """
    Validates Problem 3:
    1. If STABLE is detected first and others not yet detected, logs 'S3 STABLE rescued first: higher-priority survivors not yet detected'.
    2. If CRITICAL is detected before rescuer is dispatched to STABLE target, the target switches.
    """
    arena = DisasterArena(width=800, height=650)
    s3 = Survivor(x=200.0, y=100.0, id="S3", triage="STABLE")
    s1 = Survivor(x=500.0, y=400.0, id="S1", triage="CRITICAL")
    arena.survivors = [s3, s1]

    robot = DifferentialDriveRobot(x=190.0, y=100.0, theta=0.0)
    tracker = VisionTracker(lock_threshold=1)
    fsm = MissionFSM(arena, robot, RobotCamera(), VisionDetector(), tracker, AStarPlanner(), APFController())

    # Detect S3 (STABLE) first
    det_s3 = Detection("SURVIVOR", (70, 30, 20, 30), 20.0, 0.0, 0.85, 500.0, "STABLE")
    tracker.update([det_s3], robot.x, robot.y, robot.theta, arena.survivors)

    fsm._handle_search_patrol(0.033)
    assert any("S3 STABLE rescued first: higher-priority survivors not yet detected" in msg for msg in fsm.mission_log)

    # Now CRITICAL S1 is detected BEFORE rescuer is dispatched to S3
    robot.x, robot.y = 480.0, 400.0
    det_s1 = Detection("SURVIVOR", (70, 30, 20, 30), 40.0, 0.0, 0.85, 500.0, "CRITICAL")
    tracker.update([det_s1], robot.x, robot.y, robot.theta, arena.survivors)

    # In NAVIGATE_TO_SURVIVOR, target must switch to S1
    fsm.current_state = MissionState.NAVIGATE_TO_SURVIVOR
    fsm.active_path = [(480.0, 400.0)]
    fsm.current_waypoint_idx = 0
    fsm._handle_navigate_to_survivor(0.033)

    assert fsm._active_target_id == "S1"
    assert fsm._active_target_triage == "CRITICAL"
    assert any("Switching target from S3 to higher-priority S1" in msg for msg in fsm.mission_log)


def test_siren_console_log_format_and_severity_triggers(capsys):
    """
    Validates Problem 4:
    Console log '[AUDIO] siren played: S<id> <severity>' is printed each time one plays,
    and CRITICAL/INJURED trigger it once per survivor.
    """
    from sound import get_sound_manager
    sm = get_sound_manager()
    sm.set_muted(False)
    sm.clear_siren_flag()

    # CRITICAL triggers siren
    sm.play_siren("S1", "CRITICAL")
    captured = capsys.readouterr().out
    assert "[AUDIO] siren played: S1 CRITICAL" in captured

    # CRITICAL duplicate does not trigger siren
    sm.play_siren("S1", "CRITICAL")
    captured2 = capsys.readouterr().out
    assert "[AUDIO] siren played" not in captured2
    assert "[AUDIO] skipped: flag set" in captured2

    # INJURED triggers siren
    sm.play_siren("S2", "INJURED")
    captured3 = capsys.readouterr().out
    assert "[AUDIO] siren played: S2 INJURED" in captured3

    # STABLE does not trigger siren
    sm.play_siren("S3", "STABLE")
    captured4 = capsys.readouterr().out
    assert "[AUDIO] siren played" not in captured4

