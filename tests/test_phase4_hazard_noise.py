"""
tests/test_phase4_hazard_noise.py - Test suite for Phase 4: Dynamic Hazard, Sensor Noise, and Standby Rescuers.

Validates:
1. One hazard zone expands slowly, triggering A* and APF re-plans with log "Hazard expanded, re-planning".
2. Camera noise (Gaussian noise + smoke haze patch) toggles on/off.
3. Vision detector functions reliably with noise and classifies triage accurately.
4. Moving rescuers on standby maintain safe separation and zero collisions.
"""

import os
import sys
import math
import numpy as np
import pytest

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

os.environ["SDL_VIDEODRIVER"] = "dummy"

from arena import DisasterArena, HazardZone
from robot import DifferentialDriveRobot
from camera import RobotCamera
from detector import VisionDetector
from tracker import VisionTracker
from apf import APFController
from astar import AStarPlanner
from fsm import MissionFSM, MissionState
from rescuer import RescuerAgent, RescuerState, MessageBus


def test_dynamic_hazard_growth_and_replan():
    """Validates hazard expands dynamically and triggers re-planning in FSM."""
    arena = DisasterArena(width=800, height=650)
    hz = arena.hazard_zones[0]
    initial_radius = hz.radius

    robot = DifferentialDriveRobot(x=90.0, y=90.0, theta=0.0)
    camera = RobotCamera(width=160, height=90, fov_deg=65.0, max_range=340.0)
    detector = VisionDetector(focal_length=camera.focal_length, cx=camera.cx, cy=camera.cy)
    tracker = VisionTracker(lock_threshold=3)
    apf = APFController()
    astar = AStarPlanner(arena_width=800, arena_height=650)
    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf)

    # Simulate 10 seconds of arena updates (growth rate is 0.5 px/s)
    for _ in range(100):
        arena.update(0.1)
        fsm.update(0.1)

    assert hz.radius > initial_radius, f"Expected hazard to expand: {hz.radius} > {initial_radius}"
    assert fsm.hazard_replans_count > 0, "Expected hazard expansion to trigger re-planning"
    assert any("Hazard expanded, re-planning" in msg for msg in fsm.mission_log), (
        "Expected mission log to record 'Hazard expanded, re-planning'"
    )


def test_camera_sensor_noise_and_smoke_haze():
    """Validates Gaussian noise and smoke haze patch generation in RobotCamera."""
    arena = DisasterArena(width=800, height=650)
    robot = DifferentialDriveRobot(x=90.0, y=90.0, theta=0.0)
    camera = RobotCamera(width=160, height=90, fov_deg=65.0, max_range=340.0)

    # 1. Clean frame
    camera.noise_enabled = False
    clean_frame = camera.render(robot, arena)
    assert clean_frame.shape == (90, 160, 3)

    # 2. Noisy frame with smoke haze
    camera.noise_enabled = True
    noisy_frame = camera.render(robot, arena)
    assert noisy_frame.shape == (90, 160, 3)

    # Difference between clean and noisy frames must be substantial due to Gaussian noise and smoke haze
    diff = np.abs(noisy_frame.astype(np.float32) - clean_frame.astype(np.float32))
    assert np.mean(diff) > 5.0, f"Expected substantial noise/smoke difference, got {np.mean(diff)}"


def test_detector_with_and_without_noise():
    """Validates VisionDetector detects survivor both clean and under moderate noise."""
    arena = DisasterArena(width=800, height=650)
    # Clear line of sight to survivor S1 at (700, 130)
    robot = DifferentialDriveRobot(x=640.0, y=130.0, theta=0.0)
    camera = RobotCamera(width=160, height=90, fov_deg=65.0, max_range=340.0)
    detector = VisionDetector(focal_length=camera.focal_length, cx=camera.cx, cy=camera.cy)

    # Clean detection
    camera.noise_enabled = False
    clean_frame = camera.render(robot, arena)
    dets_clean = detector.detect(clean_frame, force_update=True)
    surv_clean = [d for d in dets_clean if d.class_label == "SURVIVOR"]
    assert len(surv_clean) >= 1, "Expected survivor detected without noise"

    # Noisy detection
    camera.noise_enabled = True
    noisy_frame = camera.render(robot, arena)
    dets_noisy = detector.detect(noisy_frame, force_update=True)
    surv_noisy = [d for d in dets_noisy if d.class_label == "SURVIVOR"]
    assert len(surv_noisy) >= 1, "Expected survivor detected with noise and smoke haze"
    assert surv_noisy[0].triage_level in ("CRITICAL", "INJURED", "STABLE")


def test_moving_standby_rescuers_collision_free():
    """Validates standby rescuers moving on patrol orbit remain collision-free with safety margins."""
    arena = DisasterArena(width=800, height=650)
    message_bus = MessageBus()
    scout_pos = (90.0, 90.0)

    for r in arena.rescuers:
        r.moving_standby = True

    dt = 0.05
    for step in range(200):
        arena.update(dt)
        for r in arena.rescuers:
            r.update(dt, arena, scout_pos, arena.rescuers, message_bus, current_time=step * dt)

            # Assert rescuer center is never inside any obstacle
            assert not arena.is_point_colliding(r.x, r.y, margin=0.0), f"Rescuer {r.name} collided with obstacle at ({r.x}, {r.y})"
            # Assert rescuer center is never inside any hazard zone
            assert not arena.is_inside_hazard(r.x, r.y, margin=0.0), f"Rescuer {r.name} entered hazard zone at ({r.x}, {r.y})"
            # Assert rescuer is within arena walls
            assert 16.0 <= r.x <= 784.0 and 16.0 <= r.y <= 634.0, f"Rescuer {r.name} breached arena boundary"

            # Assert safe separation from scout
            d_scout = math.hypot(r.x - scout_pos[0], r.y - scout_pos[1])
            assert d_scout > (r.radius + 15.0), f"Rescuer {r.name} too close to scout ({d_scout:.1f}px)"
