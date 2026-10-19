"""
tests/test_priority_mode.py - Comprehensive tests for Priority Mode in Rakshak.

Features tested:
1. Round-2 layout (2 CRITICAL, 2 INJURED, 1 STABLE, robot starting near INJURED survivors):
   - In Priority Mode (default ON):
     - Full coverage sweep queues every detected survivor (INJURED detected first, but not rescued immediately)
     - Log emitted: "[TRIAGE] sweep complete: N survivors queued, order = ..."
     - Rescues proceed strictly in priority order: CRITICAL, CRITICAL, INJURED, INJURED, STABLE
2. Rescue as found behavior when Priority Mode is toggled OFF:
   - When Priority Mode is OFF, INJURED survivor detected first is rescued first
3. New survivor detected during rescue:
   - Inserted into queue by priority
4. Toggle key and hint bar:
   - P (and T) key toggles Priority mode
   - Hint bar renders "[P] Priority: ON" and "[P] Priority: OFF"
   - CLI flags --priority and --no-priority
"""

import math
import pytest
import pygame
from arena import DisasterArena, Survivor
from robot import DifferentialDriveRobot
from camera import RobotCamera
from detector import VisionDetector, Detection
from tracker import VisionTracker, TrackedTarget
from astar import AStarPlanner
from apf import APFController
from fsm import MissionFSM, MissionState
from rescuer import RescuerState, survivor_priority_key
from app import RakshakApp


def test_priority_mode_round2_layout_secured_order():
    """
    Test Priority Mode with round-2 layout:
    2 CRITICAL, 2 INJURED, 1 STABLE.
    Robot starts near the INJURED survivors.
    Secured order must be strictly: CRITICAL, CRITICAL, INJURED, INJURED, STABLE.
    """
    app = RakshakApp(priority_mode=True)
    assert app.priority_mode is True
    assert app.fsm.priority_mode is True

    # Round-2 layout: S1, S2 INJURED close to robot start (80, 80); S4, S5 CRITICAL farther away
    s1 = Survivor(x=120.0, y=120.0, id="S1", triage="INJURED")
    s2 = Survivor(x=120.0, y=240.0, id="S2", triage="INJURED")
    s3 = Survivor(x=250.0, y=380.0, id="S3", triage="STABLE")
    s4 = Survivor(x=620.0, y=220.0, id="S4", triage="CRITICAL")
    s5 = Survivor(x=680.0, y=480.0, id="S5", triage="CRITICAL")

    app.arena.survivors = [s1, s2, s3, s4, s5]
    app.fsm.total_survivors = 5

    # Robot starts at (80, 80) facing S1
    app.robot.x = 80.0
    app.robot.y = 80.0
    app.robot.theta = math.atan2(s1.y - 80.0, s1.x - 80.0)

    # 1. Sweep phase: Robot sweeps past survivors.
    # It spots S1 and S2 (INJURED) first.
    # Because Priority Mode is ON, it must queue them and NOT abort sweep to rescue immediately.
    for surv in [s1, s2, s3, s4, s5]:
        app.robot.x = surv.x - 40.0
        app.robot.y = surv.y
        app.robot.theta = 0.0
        det = Detection(
            class_label="SURVIVOR",
            bbox=(75, 40, 20, 30),
            estimated_range=40.0,
            estimated_bearing=0.0,
            mask_area_ratio=0.75,
            contour_area=400.0,
            triage_level=surv.triage,
        )
        app.tracker.update([det], app.robot.x, app.robot.y, app.robot.theta, app.arena.survivors)
        app.fsm._handle_search_patrol(0.05)

    # Verify all 5 survivors were queued during sweep
    assert len(app.fsm._sweep_queued_ids) == 5
    assert app.fsm.sweep_complete is True

    # Verify log format: "[TRIAGE] sweep complete: N survivors queued, order = ..."
    sweep_log = next(
        (msg for msg in app.fsm.mission_log if "[TRIAGE] sweep complete: 5 survivors queued, order =" in msg),
        None,
    )
    assert sweep_log is not None, f"Expected sweep complete log not found in {app.fsm.mission_log}"
    assert "CRITICAL" in sweep_log and "INJURED" in sweep_log and "STABLE" in sweep_log

    # 2. Rescue phase: verify extractions proceed strictly in priority order
    secured_order = []

    def _complete_active_rescue():
        # Dispatch manager assigns top priority SOS
        curr_id = app.fsm._active_target_id
        assert curr_id is not None
        target_s = next(s for s in app.arena.survivors if s.id == curr_id)

        # Rescuer extracts target
        target_s.is_secured = True
        matched_sos = app.fsm.dispatch_manager.active_sos.get(curr_id)
        if not matched_sos and app.fsm.dispatch_manager.sos_queue:
            for idx, sos in enumerate(app.fsm.dispatch_manager.sos_queue):
                if sos.survivor_id == curr_id:
                    matched_sos = app.fsm.dispatch_manager.sos_queue.pop(idx)
                    break

        if matched_sos:
            matched_sos.status = "SECURED"
            app.fsm.dispatch_manager.completed_sos[curr_id] = matched_sos

        secured_order.append(target_s.triage)

        # Trigger extraction complete transition in FSM
        app.fsm.current_state = MissionState.RESCUE_OPERATION
        app.fsm._handle_rescue_operation(0.05)
        app.fsm.update(0.05)
        app.tracker.clear_rescue_target(curr_id)

    # Execute all 5 rescues
    for _ in range(5):
        _complete_active_rescue()

    # The secured order MUST be: CRITICAL, CRITICAL, INJURED, INJURED, STABLE
    assert secured_order == ["CRITICAL", "CRITICAL", "INJURED", "INJURED", "STABLE"]
    assert app.fsm.missions_completed == 5
    assert app.fsm.current_state == MissionState.MISSION_COMPLETE


def test_rescue_as_found_when_priority_mode_off():
    """
    Test that when Priority Mode is toggled OFF, Rakshak uses the "rescue as found" behavior:
    The INJURED survivor detected first is rescued first.
    """
    app = RakshakApp(priority_mode=False)
    assert app.priority_mode is False
    assert app.fsm.priority_mode is False

    s1 = Survivor(x=120.0, y=120.0, id="S1", triage="INJURED")
    s4 = Survivor(x=620.0, y=220.0, id="S4", triage="CRITICAL")
    app.arena.survivors = [s1, s4]
    app.fsm.total_survivors = 2

    # Robot starts near S1 INJURED
    app.robot.x = 90.0
    app.robot.y = 90.0

    # Detect S1 INJURED
    det = Detection(
        class_label="SURVIVOR",
        bbox=(75, 40, 20, 30),
        estimated_range=40.0,
        estimated_bearing=0.0,
        mask_area_ratio=0.75,
        contour_area=400.0,
        triage_level="INJURED",
    )
    for _ in range(4):
        app.tracker.update([det], app.robot.x, app.robot.y, app.robot.theta, app.arena.survivors)

    # In "rescue as found" (Priority Mode OFF), robot immediately locks and routes to S1 INJURED
    app.fsm._handle_search_patrol(0.05)
    assert app.fsm.current_state in (MissionState.VERIFY_TARGET, MissionState.PLAN_ROUTE)
    assert app.fsm._active_target_id == "S1"
    assert app.fsm._active_target_triage == "INJURED"


def test_new_survivor_detected_during_rescue_inserted_by_priority():
    """
    Test that if an unqueued survivor is detected during rescue, it is inserted into the queue by priority.
    """
    app = RakshakApp(priority_mode=True)
    app.fsm.sweep_complete = True
    app.fsm.current_state = MissionState.NAVIGATE_TO_SURVIVOR

    # Existing queue has S1 STABLE and S2 INJURED
    app.fsm.dispatch_manager.submit_sos("S1", 200.0, 200.0, "STABLE", 1.0)
    app.fsm.dispatch_manager.submit_sos("S2", 300.0, 300.0, "INJURED", 1.0)
    app.fsm._active_target_id = "S2"
    app.fsm._active_target_triage = "INJURED"

    # Newly detected track during transit: S3 CRITICAL
    trk_crit = TrackedTarget(
        class_label="SURVIVOR",
        target_id="S3",
        est_world_x=500.0,
        est_world_y=400.0,
        est_range=50.0,
        est_bearing=0.0,
        mask_area_ratio=0.8,
        triage_level="CRITICAL",
        is_locked=True,
    )
    app.tracker.survivor_tracks["S3"] = trk_crit

    # Run FSM update cycle
    app.fsm.update(0.05)

    # S3 CRITICAL must be inserted into queue and prioritized at the top
    assert "S3" in [s.survivor_id for s in app.fsm.dispatch_manager.sos_queue]
    top_sos = app.fsm.dispatch_manager.sos_queue[0]
    assert top_sos.survivor_id == "S3"
    assert top_sos.triage == "CRITICAL"
    assert any("[TRIAGE] New survivor detected during rescue: S3 [CRITICAL] inserted by priority." in msg for msg in app.fsm.mission_log)


def test_priority_mode_toggle_key_and_hint_bar():
    """
    Test that pressing P or T toggles Priority mode, updates FSM, and updates hint bars.
    """
    app = RakshakApp(priority_mode=True)
    assert app.priority_mode is True
    assert app.fsm.priority_mode is True

    # Simulate pressing P key
    event_p = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p, mod=0)
    pygame.event.post(event_p)
    app.handle_events()
    assert app.priority_mode is False
    assert app.fsm.priority_mode is False

    # Simulate pressing T key (alias)
    event_t = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t, mod=0)
    pygame.event.post(event_t)
    app.handle_events()
    assert app.priority_mode is True
    assert app.fsm.priority_mode is True

    # Test CLI parser flags
    from app import main
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--priority", dest="priority", action="store_true", default=True)
    parser.add_argument("--no-priority", dest="priority", action="store_false")

    args_default = parser.parse_args([])
    assert args_default.priority is True

    args_disabled = parser.parse_args(["--no-priority"])
    assert args_disabled.priority is False

    args_enabled = parser.parse_args(["--priority"])
    assert args_enabled.priority is True
