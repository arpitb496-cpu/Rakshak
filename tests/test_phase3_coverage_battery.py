"""
tests/test_phase3_coverage_battery.py - Unit tests for Phase 3: Coverage Grid, Faster Search, and Battery Model.

Tests:
1. Coverage grid updates correctly and tracks area scanned %.
2. Frontier checkpoint recommendation returns valid accessible points.
3. Battery drains with motion and angular rotation.
4. Low battery (<25%) triggers RETURN_TO_BASE state when survivors remain.
5. Base station recharge restores battery to 100% and resumes SEARCH_PATROL.
6. Checkpoint sweep executes short 120° sweep instead of 360°.
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
from detector import VisionDetector
from tracker import VisionTracker
from apf import APFController
from astar import AStarPlanner
from coverage import CoverageGrid
from fsm import MissionFSM, MissionState


def test_coverage_grid_initialization_and_update():
    """Validates that coverage grid correctly initializes and marks FOV cells."""
    arena = DisasterArena(width=800, height=650)
    cov = CoverageGrid(arena_width=800, arena_height=650, cell_size=20.0, arena=arena)

    initial_pct = cov.percent_scanned
    assert initial_pct == 0.0

    # Robot at center looking east (0 rad)
    cov.update(
        robot_x=400.0,
        robot_y=325.0,
        robot_theta=0.0,
        fov_rad=math.radians(65.0),
        max_range=340.0,
        arena=arena,
    )

    scanned_pct = cov.percent_scanned
    assert scanned_pct > 0.0, "Area scanned percentage should increase after FOV update"
    assert scanned_pct < 100.0


def test_frontier_checkpoint_recommendation():
    """Validates that frontier checkpoint recommendation returns valid accessible points."""
    arena = DisasterArena(width=800, height=650)
    cov = CoverageGrid(arena_width=800, arena_height=650, cell_size=20.0, arena=arena)

    pt = cov.get_unseen_frontier_checkpoint(robot_x=90.0, robot_y=90.0)
    assert pt is not None
    assert 20.0 <= pt[0] <= 780.0
    assert 20.0 <= pt[1] <= 630.0
    # Must be accessible (not inside solid obstacle)
    assert not arena.is_point_colliding(pt[0], pt[1], margin=4.0)


def test_battery_drain_with_motion():
    """Validates that battery drains during movement and turning."""
    arena = DisasterArena(width=800, height=650)
    robot = DifferentialDriveRobot(x=90.0, y=90.0, theta=0.0)
    camera = RobotCamera()
    detector = VisionDetector(focal_length=camera.focal_length, cx=camera.cx, cy=camera.cy)
    tracker = VisionTracker()
    astar = AStarPlanner()
    apf = APFController()

    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf)
    assert fsm.battery == 100.0

    # Move robot forward
    robot.set_twist(85.0, 1.5)
    for _ in range(30):  # 1.0 second
        fsm.update(dt=0.033)

    assert fsm.battery < 100.0
    assert fsm.battery > 85.0, "Battery should drain realistically over 1 second"


def test_low_battery_triggers_return_to_base_and_recharge():
    """Validates that battery < 25% causes return to base and recharge to 100%."""
    arena = DisasterArena(width=800, height=650)
    robot = DifferentialDriveRobot(x=400.0, y=300.0, theta=0.0)
    camera = RobotCamera()
    detector = VisionDetector(focal_length=camera.focal_length, cx=camera.cx, cy=camera.cy)
    tracker = VisionTracker()
    astar = AStarPlanner()
    apf = APFController()

    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf)
    fsm.battery = 24.0  # Force low battery

    fsm.update(dt=0.033)
    assert fsm.current_state == MissionState.RETURN_TO_BASE, "Low battery must trigger RETURN_TO_BASE"

    # Move robot to base station (90, 90) and simulate recharge
    robot.x = 90.0
    robot.y = 90.0
    for _ in range(60):  # 2.0 seconds at recharge rate 50%/s
        fsm._handle_return_to_base(dt=0.033)

    assert fsm.battery == 100.0
    assert fsm.current_state == MissionState.SEARCH_PATROL, "Must resume SEARCH_PATROL after recharge"
    assert fsm.battery_recharges >= 1


def test_short_120_degree_sweep():
    """Validates that checkpoint sweep is ~120 degrees (2.09 rad) and swift."""
    arena = DisasterArena(width=800, height=650)
    robot = DifferentialDriveRobot(x=260.0, y=65.0, theta=0.0)
    camera = RobotCamera()
    detector = VisionDetector(focal_length=camera.focal_length, cx=camera.cx, cy=camera.cy)
    tracker = VisionTracker()
    astar = AStarPlanner()
    apf = APFController()

    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf)
    fsm.patrol_idx = 0
    fsm.is_scanning_360 = True
    fsm.scan_accumulated = 0.0

    # 120° sweep target is 2.0 * pi / 3.0 = ~2.094 rad
    sweep_rad = 2.0 * math.pi / 3.0
    # Step until sweep completes
    steps = 0
    while fsm.is_scanning_360 and steps < 60:
        fsm._handle_search_patrol(dt=0.033)
        steps += 1

    assert not fsm.is_scanning_360, "120° sweep should complete in under 1 second"
    assert steps < 30, f"Expected 120° sweep in ~21 frames (0.7s), took {steps} frames"
