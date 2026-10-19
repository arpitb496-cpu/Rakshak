"""
tests/test_sensors.py - Unit tests for Proximity Sensor Ray geometry.
"""

import math
import numpy as np
import pytest
from robot import ProximitySensorRay, DifferentialDriveRobot
from arena import DisasterArena


def test_ray_segment_intersection():
    # Ray along positive x-axis from (0, 0)
    origin = (0.0, 0.0)
    direction = (1.0, 0.0)

    # Vertical segment from (50, -20) to (50, 20)
    seg_start = (50.0, -20.0)
    seg_end = (50.0, 20.0)

    hit = ProximitySensorRay.ray_segment_intersection(origin, direction, seg_start, seg_end, max_dist=100.0)
    assert hit is not None
    dist, pt = hit
    assert abs(dist - 50.0) < 1e-4
    assert abs(pt[0] - 50.0) < 1e-4
    assert abs(pt[1] - 0.0) < 1e-4


def test_ray_segment_miss_parallel():
    origin = (0.0, 0.0)
    direction = (1.0, 0.0)
    # Parallel horizontal segment
    seg_start = (10.0, 10.0)
    seg_end = (50.0, 10.0)

    hit = ProximitySensorRay.ray_segment_intersection(origin, direction, seg_start, seg_end, max_dist=100.0)
    assert hit is None


def test_ray_segment_miss_out_of_range():
    origin = (0.0, 0.0)
    direction = (1.0, 0.0)
    # Beyond max range (at x=150)
    seg_start = (150.0, -20.0)
    seg_end = (150.0, 20.0)

    hit = ProximitySensorRay.ray_segment_intersection(origin, direction, seg_start, seg_end, max_dist=100.0)
    assert hit is None


def test_ray_circle_intersection():
    origin = (0.0, 0.0)
    direction = (1.0, 0.0)
    # Circle at (60, 0) with radius 10 -> front edge at x=50
    circle_center = (60.0, 0.0)
    radius = 10.0

    hit = ProximitySensorRay.ray_circle_intersection(origin, direction, circle_center, radius, max_dist=100.0)
    assert hit is not None
    dist, pt = hit
    assert abs(dist - 50.0) < 1e-4
    assert abs(pt[0] - 50.0) < 1e-4


def test_proximity_sensor_scan_in_arena():
    arena = DisasterArena(width=800, height=650)
    # Place robot near top-left wall: (30, 30), facing right (theta=0)
    robot = DifferentialDriveRobot(x=30.0, y=30.0, theta=0.0)
    distances = robot.scan_proximity_sensors(arena)

    assert len(distances) == 8
    # Rear sensor (angle 180 deg) points toward left wall at x=16
    # From x=30 to x=16 is 14 px
    rear_dist = distances[4]  # Rear sensor index
    assert rear_dist < 20.0
