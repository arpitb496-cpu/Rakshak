"""
tests/test_tracker.py - Unit tests for VisionTracker and spatial triangulation.
"""

import math
import pytest
from detector import Detection
from tracker import VisionTracker


def test_tracker_triangulation():
    tracker = VisionTracker(lock_threshold=2)

    # Robot at (100, 200), heading 0.0 (facing +x)
    robot_x = 100.0
    robot_y = 200.0
    robot_theta = 0.0

    # Detection: Survivor at range Z=50 px, bearing 0.0
    det = Detection(
        class_label="SURVIVOR",
        bbox=(70, 30, 20, 30),
        estimated_range=50.0,
        estimated_bearing=0.0,
        mask_area_ratio=0.85,
        contour_area=510.0,
    )

    # Frame 1
    tracker.update([det], robot_x, robot_y, robot_theta, ground_truth_survivor=(150.0, 200.0))
    assert tracker.survivor_track is not None
    assert abs(tracker.survivor_track.est_world_x - 150.0) < 1.0
    assert abs(tracker.survivor_track.est_world_y - 200.0) < 1.0
    assert tracker.last_localization_error is not None
    assert tracker.last_localization_error < 1.0
    assert not tracker.is_survivor_locked()  # Needs 2 hits

    # Frame 2
    tracker.update([det], robot_x, robot_y, robot_theta, ground_truth_survivor=(150.0, 200.0))
    assert tracker.is_survivor_locked()  # Now locked!


def test_tracker_multi_survivor_association():
    tracker = VisionTracker(lock_threshold=2)

    robot_x, robot_y, robot_theta = 100.0, 200.0, 0.0

    # Detection 1: Survivor A at Z=60 px, bearing -0.2 rad
    det_a = Detection(
        class_label="SURVIVOR",
        bbox=(40, 30, 20, 30),
        estimated_range=60.0,
        estimated_bearing=-0.2,
        mask_area_ratio=0.85,
        contour_area=510.0,
        triage_level="CRITICAL",
    )

    # Detection 2: Survivor B at Z=180 px, bearing +0.3 rad (far away from A)
    det_b = Detection(
        class_label="SURVIVOR",
        bbox=(120, 30, 20, 30),
        estimated_range=180.0,
        estimated_bearing=0.3,
        mask_area_ratio=0.85,
        contour_area=510.0,
        triage_level="INJURED",
    )

    # Frame 1: Both detected simultaneously
    tracker.update([det_a, det_b], robot_x, robot_y, robot_theta)
    assert len(tracker.survivor_tracks) == 2

    # Frame 2: Both detected again -> both should lock
    tracker.update([det_a, det_b], robot_x, robot_y, robot_theta)
    assert len(tracker.survivor_tracks) == 2

    tracks = list(tracker.survivor_tracks.values())
    assert all(t.is_locked for t in tracks)
    triages = {t.triage_level for t in tracks}
    assert "CRITICAL" in triages and "INJURED" in triages


def test_two_targets_different_lock_errors():
    """
    Validates that distinct survivor targets record their own distinct error_at_lock
    at the moment each track locks, rather than sharing a single global lock error.
    """
    tracker = VisionTracker(lock_threshold=2)
    robot_x, robot_y, robot_theta = 0.0, 0.0, 0.0

    class DummySurvivor:
        def __init__(self, sid, x, y):
            self.id = sid
            self.x = x
            self.y = y
            self.is_secured = False

    gt_a = DummySurvivor("S1", 100.0, 0.0)
    gt_b = DummySurvivor("S2", 200.0, 0.0)
    gt_list = [gt_a, gt_b]

    # Target A: ~4px error (true range 100, meas 104)
    det_a = Detection(
        class_label="SURVIVOR",
        bbox=(70, 30, 20, 30),
        estimated_range=104.0,
        estimated_bearing=0.0,
        mask_area_ratio=0.85,
        contour_area=510.0,
        triage_level="CRITICAL",
    )

    # Target B: ~18px error (true range 200, meas 218)
    det_b = Detection(
        class_label="SURVIVOR",
        bbox=(30, 80, 20, 30),
        estimated_range=218.0,
        estimated_bearing=0.0,
        mask_area_ratio=0.85,
        contour_area=510.0,
        triage_level="INJURED",
    )

    tracker.update([det_a, det_b], robot_x, robot_y, robot_theta, ground_truth_survivor=gt_list)
    tracker.update([det_a, det_b], robot_x, robot_y, robot_theta, ground_truth_survivor=gt_list)

    track_a = tracker.survivor_tracks.get("S1")
    track_b = tracker.survivor_tracks.get("S2")
    assert track_a is not None and track_a.is_locked
    assert track_b is not None and track_b.is_locked

    err_a = track_a.error_at_lock
    err_b = track_b.error_at_lock

    assert err_a is not None
    assert err_b is not None
    assert abs(err_a - 4.0) < 1.5
    assert abs(err_b - 18.0) < 1.5
    assert abs(err_a - err_b) > 10.0
