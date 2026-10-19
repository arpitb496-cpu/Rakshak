"""
tests/test_fsm.py - Integration tests for Mission Finite State Machine.
"""

import math
import pytest
from arena import DisasterArena
from robot import DifferentialDriveRobot
from camera import RobotCamera
from detector import VisionDetector
from tracker import VisionTracker
from apf import APFController
from astar import AStarPlanner
from fsm import MissionFSM, MissionState


def test_fsm_initial_state():
    arena = DisasterArena()
    robot = DifferentialDriveRobot()
    camera = RobotCamera()
    detector = VisionDetector()
    tracker = VisionTracker()
    apf = APFController()
    astar = AStarPlanner()

    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf)
    assert fsm.current_state == MissionState.SEARCH_PATROL


def test_fsm_direct_detection_transition():
    arena = DisasterArena()
    # Place robot at (60, 50) facing right along corridor (y=50 is free of rubble)
    robot = DifferentialDriveRobot(x=60.0, y=50.0, theta=0.0)
    # Place survivor right in front of robot in open corridor at (150, 50)
    arena.survivor.x = 150.0
    arena.survivor.y = 50.0

    camera = RobotCamera()
    detector = VisionDetector(focal_length=camera.focal_length, cx=camera.cx, cy=camera.cy)
    tracker = VisionTracker()
    apf = APFController()
    astar = AStarPlanner()

    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf)

    # Run one FSM cycle
    fsm.update(0.05)

    # Robot should spot survivor and transition to VERIFY_TARGET
    assert fsm.current_state in (MissionState.VERIFY_TARGET, MissionState.PLAN_ROUTE, MissionState.NAVIGATE_TO_SURVIVOR)
