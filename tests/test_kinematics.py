"""
tests/test_kinematics.py - Unit tests for DifferentialDriveRobot kinematics.
"""

import math
import numpy as np
import pytest
from robot import DifferentialDriveRobot


def test_robot_initialization():
    robot = DifferentialDriveRobot(x=100.0, y=150.0, theta=0.5)
    assert robot.x == 100.0
    assert robot.y == 150.0
    assert abs(robot.theta - 0.5) < 1e-4
    assert robot.num_sensors == 8


def test_angle_normalization():
    norm = DifferentialDriveRobot.normalize_angle
    assert abs(norm(math.pi * 3) - (-math.pi)) < 1e-4 or abs(norm(math.pi * 3) - math.pi) < 1e-4
    assert abs(norm(0.0)) < 1e-5
    assert abs(norm(-math.pi * 2.5) - (-0.5 * math.pi)) < 1e-4


def test_straight_forward_motion():
    robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0)
    # Move forward at 50 px/s for 1.0 second
    robot.set_twist(50.0, 0.0)
    assert abs(robot.v - 50.0) < 1e-3
    assert abs(robot.omega) < 1e-3
    assert abs(robot.v_left - 50.0) < 1e-3
    assert abs(robot.v_right - 50.0) < 1e-3

    robot.update_kinematics(1.0)
    assert abs(robot.x - 50.0) < 1e-2
    assert abs(robot.y - 0.0) < 1e-2
    assert abs(robot.theta - 0.0) < 1e-2


def test_pure_rotation():
    robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0)
    # Turn at 1.0 rad/s for 1.0 second
    robot.set_twist(0.0, 1.0)
    assert abs(robot.v) < 1e-3
    assert abs(robot.omega - 1.0) < 1e-3

    robot.update_kinematics(1.0)
    # Position should remain practically stationary
    assert abs(robot.x) < 1e-2
    assert abs(robot.y) < 1e-2
    assert abs(robot.theta - 1.0) < 1e-2


def test_speed_clamping():
    robot = DifferentialDriveRobot(max_speed=100.0, max_omega=3.0)
    # Request extreme velocity
    robot.set_twist(500.0, 20.0)
    assert robot.v <= robot.max_speed + 1e-3
    assert abs(robot.omega) <= robot.max_omega + 1e-3
    assert abs(robot.v_left) <= robot.max_speed + 1e-3
    assert abs(robot.v_right) <= robot.max_speed + 1e-3
