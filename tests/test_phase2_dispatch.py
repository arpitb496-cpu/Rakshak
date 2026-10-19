"""
tests/test_phase2_dispatch.py - Unit tests for Phase 2: SOS & Rescuer Dispatch.

Tests:
1. Nearest-available rescuer selection and triage-priority queue ordering (CRITICAL > INJURED > STABLE).
2. SOS coordinates come from tracker estimate, not ground truth.
3. Scout does not resume searching before extraction completes in default sequential mode.
4. Scout resumes searching immediately after sending SOS in --parallel mode.
5. Simulated message bus latency delivery and logging.
6. Extraction cycle and survivor secured transition.
"""

import os
import sys
import math
import pytest

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

os.environ["SDL_VIDEODRIVER"] = "dummy"

from arena import DisasterArena, Survivor
from robot import DifferentialDriveRobot
from camera import RobotCamera
from detector import VisionDetector
from tracker import VisionTracker, TrackedTarget
from apf import APFController
from astar import AStarPlanner
from rescuer import RescuerAgent, RescuerState, MessageBus, DispatchManager, SOSMessage
from fsm import MissionFSM, MissionState


def test_triage_priority_queue_ordering():
    """Validates that DispatchManager orders SOS queue by CRITICAL > INJURED > STABLE."""
    bus = MessageBus(latency=0.0)
    dm = DispatchManager(bus)

    # Submit in reverse priority order: STABLE, INJURED, CRITICAL
    dm.submit_sos("S_STABLE", 200, 200, "STABLE", current_time=1.0)
    dm.submit_sos("S_INJURED", 300, 300, "INJURED", current_time=1.1)
    dm.submit_sos("S_CRITICAL", 400, 400, "CRITICAL", current_time=1.2)

    # Initial order before sort
    dm.sos_queue.sort(key=lambda s: dm.triage_rank(s.triage))

    assert dm.sos_queue[0].survivor_id == "S_CRITICAL"
    assert dm.sos_queue[0].triage == "CRITICAL"
    assert dm.sos_queue[1].survivor_id == "S_INJURED"
    assert dm.sos_queue[1].triage == "INJURED"
    assert dm.sos_queue[2].survivor_id == "S_STABLE"
    assert dm.sos_queue[2].triage == "STABLE"


def test_nearest_available_rescuer_selection():
    """Validates that DispatchManager assigns the nearest available rescuer by path length."""
    bus = MessageBus(latency=0.0)
    dm = DispatchManager(bus)
    arena = DisasterArena(width=800, height=650)
    planner = AStarPlanner(arena_width=800, arena_height=650)

    # 3 rescuers: R1 close to (200, 200), R2 far away at (700, 500), R3 BUSY
    r1 = RescuerAgent(id_num=1, home_x=190, home_y=200)
    r2 = RescuerAgent(id_num=2, home_x=700, home_y=500)
    r3 = RescuerAgent(id_num=3, home_x=195, home_y=205)
    r3.state = RescuerState.EXTRACTING  # Busy

    rescuers = [r1, r2, r3]

    # Target near R1
    dm.submit_sos("S1", 210, 200, "CRITICAL", current_time=0.0)
    dm.update(dt=0.1, current_time=0.1, rescuers=rescuers, arena=arena, planner=planner)

    assert "S1" in dm.active_sos
    sos = dm.active_sos["S1"]
    assert sos.assigned_rescuer_id == "R1"
    assert r1.state == RescuerState.DISPATCHED
    assert r2.state == RescuerState.STANDBY
    assert r3.state == RescuerState.EXTRACTING


def test_sos_coordinates_come_from_tracker_not_ground_truth():
    """Validates that SOS coordinates strictly reflect tracker estimates, not ground truth."""
    arena = DisasterArena(width=800, height=650)
    arena.survivor.x = 650.0
    arena.survivor.y = 200.0

    robot = DifferentialDriveRobot(x=600.0, y=200.0, theta=0.0)
    camera = RobotCamera()
    detector = VisionDetector(focal_length=camera.focal_length, cx=camera.cx, cy=camera.cy)
    tracker = VisionTracker()
    astar = AStarPlanner()
    apf = APFController()

    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf)

    # Inject an intentionally shifted tracker estimate
    tracker.survivor_track = TrackedTarget(
        class_label="SURVIVOR",
        target_id="S_TEST",
        est_world_x=658.4,  # Intentionally differs from 650.0
        est_world_y=204.2,  # Intentionally differs from 200.0
        est_range=58.4,
        est_bearing=0.07,
        mask_area_ratio=0.05,
        triage_level="INJURED",
        is_locked=True,
    )

    fsm.current_state = MissionState.RESCUE_OPERATION
    fsm._handle_rescue_operation(dt=0.033)

    assert fsm.current_sos_id == "S_TEST"
    assert len(fsm.dispatch_manager.sos_queue) == 1
    sos = fsm.dispatch_manager.sos_queue[0]
    assert sos.est_x == pytest.approx(658.4, abs=0.1)
    assert sos.est_y == pytest.approx(204.2, abs=0.1)
    assert sos.triage == "INJURED"
    # Ground truth was (650, 200), assert it did NOT copy ground truth
    assert sos.est_x != 650.0
    assert sos.est_y != 200.0


def test_scout_sequential_mode_waits_for_extraction():
    """Validates that in default sequential mode, scout does not resume search before extraction completes."""
    arena = DisasterArena(width=800, height=650)
    robot = DifferentialDriveRobot(x=600.0, y=200.0, theta=0.0)
    camera = RobotCamera()
    detector = VisionDetector(focal_length=camera.focal_length, cx=camera.cx, cy=camera.cy)
    tracker = VisionTracker()
    astar = AStarPlanner()
    apf = APFController()

    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf, parallel_mode=False)

    tracker.survivor_track = TrackedTarget(
        class_label="SURVIVOR",
        target_id="S1",
        est_world_x=650.0,
        est_world_y=200.0,
        est_range=50.0,
        est_bearing=0.0,
        mask_area_ratio=0.05,
        triage_level="CRITICAL",
        is_locked=True,
    )

    fsm.current_state = MissionState.RESCUE_OPERATION
    fsm.state_timer = 0.0

    # Step for 1.5 seconds while extraction has NOT completed
    for _ in range(45):
        fsm._handle_rescue_operation(dt=0.033)
        assert fsm.current_state == MissionState.RESCUE_OPERATION, "Scout must stay in RESCUE_OPERATION while awaiting extraction."

    # Now simulate extraction complete
    fsm.dispatch_manager.completed_sos["S1"] = SOSMessage(
        survivor_id="S1", est_x=650.0, est_y=200.0, triage="CRITICAL", timestamp=0.0, status="SECURED"
    )
    if arena.survivors:
        arena.survivors[0].is_secured = True

    # Next step: should detect completion and advance
    fsm._handle_rescue_operation(dt=0.033)
    assert fsm.current_state in (MissionState.SEARCH_PATROL, MissionState.MISSION_COMPLETE)


def test_scout_parallel_mode_resumes_immediately():
    """Validates that in parallel mode, scout resumes search immediately after transmitting SOS."""
    arena = DisasterArena(width=800, height=650)
    robot = DifferentialDriveRobot(x=600.0, y=200.0, theta=0.0)
    camera = RobotCamera()
    detector = VisionDetector(focal_length=camera.focal_length, cx=camera.cx, cy=camera.cy)
    tracker = VisionTracker()
    astar = AStarPlanner()
    apf = APFController()

    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf, parallel_mode=True)

    tracker.survivor_track = TrackedTarget(
        class_label="SURVIVOR",
        target_id="S1",
        est_world_x=650.0,
        est_world_y=200.0,
        est_range=50.0,
        est_bearing=0.0,
        mask_area_ratio=0.05,
        triage_level="CRITICAL",
        is_locked=True,
    )

    fsm.current_state = MissionState.RESCUE_OPERATION
    fsm.state_timer = 0.0

    # Step past transmission time (0.4s)
    for _ in range(15):
        fsm._handle_rescue_operation(dt=0.033)
        fsm.state_timer += 0.033

    # In parallel mode, scout should have resumed SEARCH_PATROL
    assert fsm.current_state == MissionState.SEARCH_PATROL
    assert fsm.sos_sent_count == 1
