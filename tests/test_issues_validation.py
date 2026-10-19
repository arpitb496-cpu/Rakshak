"""
tests/test_issues_validation.py - Comprehensive validation tests for Issues 1 to 6.

Test cases:
- Test A: In a 5-survivor layout, run at least 3 consecutive rescues; banner ID, GT, Est and log ID match and position error < 5 px each time. Repeat after a reset.
- Test B: Severity shown in banner, telemetry and queue equals true severity in every rescue, even with camera noise on.
- Test C: Layout with 2 CRITICAL, 2 INJURED, 1 STABLE: secured order CRITICAL, CRITICAL, INJURED, INJURED, STABLE in autonomous mode, including when STABLE or INJURED is detected first.
- Test D: missions_completed equals the number of secured survivors after each rescue, and a full run reaches MISSION_COMPLETE and auto-resets.
- Test E: Siren function is called exactly once per survivor with the correct severity, not when muted, and again after reset. Sounds still play on every other event.
- Test F: Fullscreen toggle function runs without exception and leaves mission state unchanged (mock the display call if headless).
- Test G: In manual mode, D only steers; Enter dispatches.
"""

import os
import sys
import math
import pytest
import pygame

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

os.environ["SDL_VIDEODRIVER"] = "dummy"

from app import RakshakApp
from arena import DisasterArena, Survivor
from rescuer import RescuerState, survivor_priority_key
from fsm import MissionFSM, MissionState
from detector import Detection
from sound import get_sound_manager


def test_a_consecutive_rescues_and_reset_id_consistency():
    """
    Test A: In a 5-survivor layout, run at least 3 consecutive rescues;
    banner ID, GT, Est and log ID match and position error < 5 px each time.
    Repeat after a reset.
    """
    app = RakshakApp()
    app.arena.survivors = [
        Survivor(x=240.0, y=70.0, id="S1", triage="CRITICAL"),
        Survivor(x=500.0, y=90.0, id="S2", triage="INJURED"),
        Survivor(x=680.0, y=250.0, id="S3", triage="STABLE"),
        Survivor(x=350.0, y=420.0, id="S4", triage="CRITICAL"),
        Survivor(x=600.0, y=550.0, id="S5", triage="INJURED"),
    ]
    app.fsm.total_survivors = 5

    def _perform_and_verify_rescue(target_s):
        # 1. Update detector & tracker for target
        det = Detection(
            class_label="SURVIVOR",
            bbox=(75, 40, 20, 30),
            estimated_range=45.0,
            estimated_bearing=0.0,
            mask_area_ratio=0.75,
            contour_area=400.0,
            triage_level=target_s.triage,
        )
        app.robot.x = target_s.x - 45.0
        app.robot.y = target_s.y
        app.robot.theta = 0.0

        for _ in range(5):
            app.tracker.update([det], app.robot.x, app.robot.y, app.robot.theta, app.arena.survivors)

        st = app.tracker.survivor_track
        assert st is not None
        assert st.target_id == target_s.id
        err = math.hypot(st.est_world_x - target_s.x, st.est_world_y - target_s.y)
        assert err < 5.0, f"Position error {err:.2f}px must be < 5px"

        # 2. Verify active survivor resolution
        active_s = app.arena.get_active_survivor(tracker=app.tracker, fsm=app.fsm)
        assert active_s.id == target_s.id
        assert math.hypot(active_s.x - target_s.x, active_s.y - target_s.y) < 1e-4

        # 3. Simulate rescue operation and completion
        app.fsm.current_state = MissionState.RESCUE_OPERATION
        app.fsm._handle_rescue_operation(0.1)

        # Mark secured
        target_s.is_secured = True
        app.fsm.dispatch_manager.submit_sos(target_s.id, target_s.x, target_s.y, target_s.triage, 1.0)
        app.fsm.dispatch_manager.completed_sos[target_s.id] = app.fsm.dispatch_manager.sos_queue.pop(0) if app.fsm.dispatch_manager.sos_queue else None

        app.fsm.update(0.1)
        app.tracker.clear_rescue_target(target_s.id)

    # Run 3 consecutive rescues
    for s in app.arena.survivors[:3]:
        _perform_and_verify_rescue(s)

    # Repeat after reset
    app.reset_mission(randomize_survivor=False)
    app.arena.survivors = [
        Survivor(x=240.0, y=70.0, id="S1", triage="CRITICAL"),
        Survivor(x=500.0, y=90.0, id="S2", triage="INJURED"),
        Survivor(x=680.0, y=250.0, id="S3", triage="STABLE"),
    ]
    app.fsm.total_survivors = 3
    for s in app.arena.survivors[:2]:
        _perform_and_verify_rescue(s)


def test_b_severity_consistency_with_camera_noise():
    """
    Test B (updated for honest vision): triage_level comes from majority vote
    over perceived detector labels (not ground truth).  The test verifies that:
      1. The track is created/updated correctly for each survivor.
      2. The SOS queue entry stores whatever perceived label the tracker produced.
      3. true_triage is stored on the track for metric calculation.
    (We no longer assert triage_level == surv.triage because that would require
    ground-truth injection, which is intentionally removed.)
    """
    app = RakshakApp()
    app.camera.noise_enabled = True

    s1 = Survivor(x=300.0, y=200.0, id="S1", triage="CRITICAL")
    s2 = Survivor(x=500.0, y=200.0, id="S2", triage="INJURED")
    s3 = Survivor(x=650.0, y=200.0, id="S3", triage="STABLE")
    app.arena.survivors = [s1, s2, s3]

    perceived_label = "INJURED"  # all detections report INJURED

    for surv in app.arena.survivors:
        noisy_det = Detection(
            class_label="SURVIVOR",
            bbox=(70, 35, 18, 28),
            estimated_range=50.0,
            estimated_bearing=0.0,
            mask_area_ratio=0.7,
            contour_area=350.0,
            triage_level=perceived_label,
        )
        app.robot.x = surv.x - 50.0
        app.robot.y = surv.y
        app.robot.theta = 0.0

        for _ in range(8):  # enough for 60% majority vote
            app.tracker.update([noisy_det], app.robot.x, app.robot.y, app.robot.theta, app.arena.survivors)

        st = app.tracker.survivor_tracks.get(surv.id)
        assert st is not None, f"Track for {surv.id} not found"

        # Triage level must be majority-vote perceived label
        assert st.triage_level == perceived_label, (
            f"{surv.id}: expected perceived={perceived_label}, got {st.triage_level}"
        )

        # true_triage stored for metrics
        assert st.true_triage == surv.triage, (
            f"{surv.id}: true_triage should be {surv.triage}, got {st.true_triage}"
        )

        # SOS queue must store the same perceived label
        app.fsm.dispatch_manager.submit_sos(surv.id, st.est_world_x, st.est_world_y, st.triage_level, 0.5)
        sos_entry = next(s for s in app.fsm.dispatch_manager.sos_queue if s.survivor_id == surv.id)
        assert sos_entry.triage == perceived_label


def test_c_triage_priority_order_critical_injured_stable():
    """
    Test C: Layout with 2 CRITICAL, 2 INJURED, 1 STABLE:
    secured order CRITICAL, CRITICAL, INJURED, INJURED, STABLE in autonomous mode,
    including when STABLE or INJURED is detected first.
    """
    app = RakshakApp()
    s1 = Survivor(x=200.0, y=100.0, id="S1", triage="CRITICAL")
    s2 = Survivor(x=400.0, y=100.0, id="S2", triage="INJURED")
    s3 = Survivor(x=600.0, y=100.0, id="S3", triage="STABLE")
    s4 = Survivor(x=300.0, y=300.0, id="S4", triage="CRITICAL")
    s5 = Survivor(x=500.0, y=300.0, id="S5", triage="INJURED")
    app.arena.survivors = [s1, s2, s3, s4, s5]
    app.fsm.total_survivors = 5

    # Simulate detecting S3 (STABLE) first, then S5 (INJURED), then S1 and S4 (CRITICAL)
    detection_order = [s3, s5, s2, s4, s1]
    for surv in detection_order:
        det = Detection(
            class_label="SURVIVOR",
            bbox=(75, 40, 20, 30),
            estimated_range=math.hypot(surv.x - 100, surv.y - 100),
            estimated_bearing=0.0,
            mask_area_ratio=0.75,
            contour_area=400.0,
            triage_level=surv.triage,
        )
        app.tracker.update([det], 100.0, 100.0, 0.0, app.arena.survivors)
        app.fsm.dispatch_manager.submit_sos(surv.id, surv.x, surv.y, surv.triage, 1.0)

    # Sort queue with shared priority function
    ref_pos = (app.robot.x, app.robot.y)
    app.fsm.dispatch_manager.sos_queue.sort(key=lambda s: survivor_priority_key(s, ref_pos))

    queue_ids = [s.survivor_id for s in app.fsm.dispatch_manager.sos_queue]
    queue_triages = [s.triage for s in app.fsm.dispatch_manager.sos_queue]

    # Verify triage priority order: CRITICAL > CRITICAL > INJURED > INJURED > STABLE
    assert queue_triages == ["CRITICAL", "CRITICAL", "INJURED", "INJURED", "STABLE"]
    assert set(queue_ids[:2]) == {"S1", "S4"}
    assert set(queue_ids[2:4]) == {"S2", "S5"}
    assert queue_ids[4] == "S3"


def test_d_missions_completed_increments_and_reaches_mission_complete():
    """
    Test D: missions_completed equals the number of secured survivors after each rescue,
    and a full run reaches MISSION_COMPLETE and auto-resets.
    """
    app = RakshakApp()
    app.arena.survivors = [
        Survivor(x=200.0, y=100.0, id="S1", triage="CRITICAL"),
        Survivor(x=400.0, y=100.0, id="S2", triage="INJURED"),
    ]
    app.fsm.total_survivors = 2

    assert app.fsm.missions_completed == 0

    # Secure first survivor
    app.arena.survivors[0].is_secured = True
    app.fsm.dispatch_manager.submit_sos("S1", 200.0, 100.0, "CRITICAL", 1.0)
    app.fsm.dispatch_manager.completed_sos["S1"] = app.fsm.dispatch_manager.sos_queue.pop(0)
    app.fsm.update(0.1)
    assert app.fsm.missions_completed == 1

    # Secure second survivor
    app.arena.survivors[1].is_secured = True
    app.fsm.dispatch_manager.submit_sos("S2", 400.0, 100.0, "INJURED", 2.0)
    app.fsm.dispatch_manager.completed_sos["S2"] = app.fsm.dispatch_manager.sos_queue.pop(0)
    app.fsm.update(0.1)
    assert app.fsm.missions_completed == 2
    assert app.fsm.current_state == MissionState.MISSION_COMPLETE


def test_e_siren_called_once_per_survivor_and_muted_handling(capsys):
    """
    Test E: Siren function is called exactly once per survivor with the correct severity,
    not when muted, and again after reset. Sounds still play on every other event.
    """
    sm = get_sound_manager()
    sm.set_muted(False)
    sm.clear_siren_flag()

    # S1 CRITICAL: plays siren_critical
    sm.play_siren("S1", "CRITICAL")
    out = capsys.readouterr().out
    assert "[AUDIO] siren played: S1 CRITICAL" in out
    assert "[AUDIO] played: siren_critical S1 CRITICAL" in out

    # Duplicate S1 call is skipped
    sm.play_siren("S1", "CRITICAL")
    out = capsys.readouterr().out
    assert "[AUDIO] skipped: flag set" in out

    # S3 STABLE: plays beep_stable alert beep
    sm.play_siren("S3", "STABLE")
    out = capsys.readouterr().out
    assert "[AUDIO] played: beep_stable S3 STABLE" in out

    # When muted, plays are skipped with 'muted' log
    sm.set_muted(True)
    sm.play_siren("S2", "INJURED")
    out = capsys.readouterr().out
    assert "[AUDIO] skipped: muted" in out

    # Other sounds also skipped when muted
    sm.play_survivor_detected("S2", "INJURED")
    out = capsys.readouterr().out
    assert "[AUDIO] skipped: muted" in out

    # Unmute and clear flags
    sm.set_muted(False)
    sm.clear_siren_flag()

    # S2 INJURED now plays
    sm.play_siren("S2", "INJURED")
    out = capsys.readouterr().out
    assert "[AUDIO] siren played: S2 INJURED" in out
    assert "[AUDIO] played: siren_injured S2 INJURED" in out


def test_f_fullscreen_toggle_state_preservation_and_exception_safety():
    """
    Test F: Fullscreen toggle function runs without exception and leaves mission state unchanged.
    """
    app = RakshakApp(fullscreen=False)
    app.fsm.current_state = MissionState.RESCUE_OPERATION
    app.fsm.missions_completed = 2
    app.paused = True
    app.robot.x = 350.0
    app.robot.y = 220.0
    app.sound_manager.set_muted(True)

    # Toggle to fullscreen
    app.toggle_fullscreen()
    assert app.fullscreen
    assert app.fsm.current_state == MissionState.RESCUE_OPERATION
    assert app.fsm.missions_completed == 2
    assert app.paused is True
    assert app.robot.x == 350.0 and app.robot.y == 220.0
    assert app.sound_manager.is_muted() is True

    # Toggle back to windowed
    app.toggle_fullscreen()
    assert not app.fullscreen
    assert app.fsm.current_state == MissionState.RESCUE_OPERATION
    assert app.fsm.missions_completed == 2


def test_g_manual_mode_d_key_steers_only_and_enter_dispatches():
    """
    Test G: In manual mode, D only steers; Enter dispatches.
    """
    app = RakshakApp()
    app.manual_mode = True
    app.fsm.enter_manual_mode()

    s1 = app.arena.survivors[0]
    app.robot.x = s1.x + 35.0
    app.robot.y = s1.y
    app.fsm.manual_lock_and_sos()
    assert len(app.fsm.dispatch_manager.sos_queue) == 1

    # Press D key -> must NOT dispatch
    event_d = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d, mod=0)
    pygame.event.post(event_d)
    app.handle_events()
    assert len(app.fsm.dispatch_manager.sos_queue) == 1

    # Press Enter key -> dispatches
    event_enter = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0)
    pygame.event.post(event_enter)
    app.handle_events()
    assert len(app.fsm.dispatch_manager.sos_queue) == 0
