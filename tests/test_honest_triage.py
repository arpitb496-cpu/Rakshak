"""
tests/test_honest_triage.py
Verifies Task 1: triage_level comes ONLY from majority vote over perceived
detector labels (8-frame window, >=60% threshold).
Ground truth is never used to set triage_level; only stored in true_triage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tracker import VisionTracker, TrackedTarget
from detector import Detection


def _det(triage="CRITICAL", rng=50.0, bearing=0.0):
    """Build a minimal survivor Detection with required fields."""
    return Detection(
        class_label="SURVIVOR",
        bbox=(70, 30, 20, 20),
        estimated_range=rng,
        estimated_bearing=bearing,
        mask_area_ratio=0.05,
        contour_area=100.0,
        triage_level=triage,
    )


def _feed(tracker, labels, robot=(400, 300, 0.0), gt=None):
    """Feed a list of perceived triage labels one frame at a time."""
    for label in labels:
        tracker.update([_det(label)], *robot, ground_truth_survivor=gt)


def _make_track(triage="CRITICAL", true_triage="", frozen=True):
    return TrackedTarget(
        class_label="SURVIVOR",
        est_world_x=0.0, est_world_y=0.0,
        est_range=50.0, est_bearing=0.0,
        mask_area_ratio=0.05,
        triage_level=triage,
        true_triage=true_triage,
        triage_frozen=frozen,
    )


# ── Test 1: majority vote picks the right label ───────────────────────────────
def test_majority_vote_critical():
    t = VisionTracker(lock_threshold=1)
    # 6 CRITICAL, 2 INJURED → CRITICAL wins (75%)
    _feed(t, ["CRITICAL"] * 6 + ["INJURED"] * 2)
    assert t.survivor_track.triage_level == "CRITICAL"


def test_majority_vote_injured():
    t = VisionTracker(lock_threshold=1)
    _feed(t, ["INJURED"] * 5 + ["CRITICAL"] * 3)
    assert t.survivor_track.triage_level == "INJURED"


# ── Test 2: ground truth does NOT override perceived label ────────────────────
def test_gt_does_not_override_perceived():
    """Detector consistently sees INJURED but GT says CRITICAL → stays INJURED."""
    class FakeSurvivor:
        x, y, id, triage, is_secured = 400.0, 300.0, "S1", "CRITICAL", False

    t = VisionTracker(lock_threshold=1)
    _feed(t, ["INJURED"] * 8, gt=FakeSurvivor())
    trk = t.survivor_track
    assert trk is not None
    assert trk.triage_level == "INJURED", (
        f"Expected INJURED (perceived majority) but got {trk.triage_level}"
    )
    # true_triage must be stored for metrics
    assert trk.true_triage == "CRITICAL"


# ── Test 3: label freezes after lock ─────────────────────────────────────────
def test_label_frozen_after_lock():
    t = VisionTracker(lock_threshold=3, max_lock_variance=1000.0)
    _feed(t, ["STABLE"] * 10)
    trk = t.survivor_track
    assert trk is not None and trk.is_locked
    assert trk.triage_frozen is True
    frozen = trk.triage_level
    # Flood with opposite label — frozen track must not change
    _feed(t, ["CRITICAL"] * 10)
    assert t.survivor_track.triage_level == frozen


# ── Test 4: triage_accuracy() metric ─────────────────────────────────────────
def test_triage_accuracy_perfect():
    t = VisionTracker(lock_threshold=1)
    t.survivor_tracks["S1"] = _make_track("CRITICAL", "CRITICAL")
    t.survivor_tracks["S2"] = _make_track("INJURED",  "INJURED")
    assert t.triage_accuracy() == 100.0


def test_triage_accuracy_partial():
    t = VisionTracker(lock_threshold=1)
    t.survivor_tracks["S1"] = _make_track("CRITICAL", "CRITICAL")
    t.survivor_tracks["S2"] = _make_track("INJURED",  "CRITICAL")  # wrong
    assert t.triage_accuracy() == 50.0


def test_triage_accuracy_no_gt():
    """Tracks without known true_triage are excluded → returns 100.0."""
    t = VisionTracker(lock_threshold=1)
    t.survivor_tracks["S1"] = _make_track("STABLE", "")
    assert t.triage_accuracy() == 100.0


# ── Test 5: no [SEVERITY] spam ───────────────────────────────────────────────
def test_no_severity_spam(capsys):
    class FakeSurvivor:
        x, y, id, triage, is_secured = 400.0, 300.0, "S1", "CRITICAL", False

    t = VisionTracker(lock_threshold=1)
    _feed(t, ["INJURED"] * 5, gt=FakeSurvivor())
    assert "[SEVERITY]" not in capsys.readouterr().out
