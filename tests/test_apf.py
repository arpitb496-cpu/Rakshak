"""
tests/test_apf.py - Unit tests for APF Controller (attractive, repulsive, tangential forces).
"""

import math
import numpy as np
import pytest
from arena import DisasterArena
from robot import DifferentialDriveRobot
from apf import APFController


def test_apf_attractive_force_free_space():
    arena = DisasterArena(width=800, height=650)
    robot = DifferentialDriveRobot(x=100.0, y=100.0, theta=0.0)
    # Clear all sensor readings to max range
    robot.sensor_distances.fill(robot.sensor_range)

    apf = APFController()
    goal_x = 200.0
    goal_y = 100.0

    f_att, f_rep, f_tan, f_net = apf.compute_forces(robot, goal_x, goal_y, arena)

    # Attractive force should point in +x direction
    assert f_att[0] > 0.0
    assert abs(f_att[1]) < 1e-3
    # Repulsive should be virtually zero
    assert np.linalg.norm(f_rep) < 1.0


def test_apf_repulsive_force_obstacle_ahead():
    arena = DisasterArena(width=800, height=650)
    robot = DifferentialDriveRobot(x=100.0, y=100.0, theta=0.0)
    # Simulate front obstacle 25 px ahead
    robot.sensor_distances.fill(robot.sensor_range)
    robot.sensor_distances[0] = 25.0  # Front sensor

    apf = APFController(repulsion_threshold=75.0)
    goal_x = 200.0
    goal_y = 100.0

    f_att, f_rep, f_tan, f_net = apf.compute_forces(robot, goal_x, goal_y, arena)

    # Front sensor ray points in +x. Repulsive force pushes in -x direction
    assert f_rep[0] < 0.0


def test_apf_control_velocity_generation():
    arena = DisasterArena(width=800, height=650)
    robot = DifferentialDriveRobot(x=100.0, y=100.0, theta=0.0)
    robot.sensor_distances.fill(robot.sensor_range)

    apf = APFController()
    v, w = apf.compute_control(robot, 300.0, 100.0, arena)

    assert v > 0.0
    assert abs(w) < 0.2  # Robot already aligned with goal along +x
