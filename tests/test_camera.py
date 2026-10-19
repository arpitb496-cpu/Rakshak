"""
tests/test_camera.py - Unit tests for forward-facing robot camera simulation.
"""

import math
import numpy as np
import pytest
from camera import RobotCamera
from robot import DifferentialDriveRobot
from arena import DisasterArena


def test_camera_intrinsics():
    cam = RobotCamera(width=160, height=90, fov_deg=65.0)
    assert cam.width == 160
    assert cam.height == 90
    assert cam.cx == 80.0
    assert cam.cy == 45.0
    # Expected focal length: 80 / tan(32.5 deg) ~ 125.57
    expected_f = 80.0 / math.tan(math.radians(32.5))
    assert abs(cam.focal_length - expected_f) < 1e-2


def test_camera_render_output_shape():
    arena = DisasterArena(width=800, height=650)
    robot = DifferentialDriveRobot(x=100.0, y=100.0, theta=0.0)
    cam = RobotCamera(width=160, height=90, fov_deg=65.0)

    frame = cam.render(robot, arena)
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (90, 160, 3)
    assert frame.dtype == np.uint8
    assert cam.depth_buffer.shape == (160,)


def test_camera_frustum_polygon():
    robot = DifferentialDriveRobot(x=200.0, y=200.0, theta=0.0)
    cam = RobotCamera(width=160, height=90, fov_deg=65.0, max_range=300.0)

    poly = cam.get_frustum_polygon(robot)
    assert len(poly) >= 3
    # Origin should be robot position
    assert poly[0] == (200.0, 200.0)
    # Further vertices should be within max_range distance
    for pt in poly[1:]:
        dist = math.hypot(pt[0] - 200.0, pt[1] - 200.0)
        assert abs(dist - 300.0) < 1e-2
