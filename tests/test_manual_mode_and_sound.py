"""
tests/test_manual_mode_and_sound.py - Unit tests for Sound subsystem and Manual Mode rescue pipeline.

Tests:
1. Audio subsystem: Sound asset synthesis, mixer init error resilience, and mute toggle.
2. Manual mode: [E] key locks nearby survivor and transmits SOS.
3. Manual mode: [D] key dispatches standby rescuer by triage priority (CRITICAL > INJURED > STABLE).
4. Manual mode: [X] key confirms extraction within perimeter (<= 50px), completes rescue, and increments Missions counter.
5. Mode transition: returning from manual to autonomous mode brings FSM to a clean, un-stuck state.
6. Priority queue dispatch ordering verification in manual mode.
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
from arena import DisasterArena
from robot import DifferentialDriveRobot
from camera import RobotCamera
from detector import VisionDetector
from tracker import VisionTracker
from apf import APFController
from astar import AStarPlanner
from rescuer import RescuerState, MessageBus, DispatchManager
from fsm import MissionFSM, MissionState
from sound import get_sound_manager, SoundManager


def test_sound_manager_synthesis_and_mute():
    """Validates that SoundManager loads assets, handles mute toggle, and does not crash."""
    sm = get_sound_manager()
    assert sm is not None

    # Check that the 5 expected sounds are registered
    expected_sounds = [
        "survivor_detected",
        "rescuer_dispatched",
        "extraction_complete",
        "hazard_warning",
        "mission_complete",
    ]
    for s in expected_sounds:
        assert s in sm.sounds, f"Sound {s} should be registered in SoundManager"

    # Test mute / unmute toggle
    initial_mute = sm.is_muted()
    toggled = sm.toggle_mute()
    assert toggled == (not initial_mute)
    sm.toggle_mute()  # restore
    assert sm.is_muted() == initial_mute

    # Test playback triggers (should be silent / safe and never raise an exception)
    sm.play_survivor_detected()
    sm.play_rescuer_dispatched()
    sm.play_extraction_complete()
    sm.play_hazard_warning(cooldown=0.0)
    sm.play_mission_complete()


def test_manual_mode_lock_and_sos_key_e():
    """
    Validates that in manual mode, pressing E locks a nearby survivor and submits an SOS.
    Also validates that E fails when out of standoff range.
    """
    app = RakshakApp()
    app.manual_mode = True
    app.fsm.enter_manual_mode()

    # Place scout robot far away from any survivor
    app.robot.x = 750.0
    app.robot.y = 50.0
    assert not app.fsm.can_manual_lock()

    # Simulate pressing E when out of range -> should fail
    success, msg = app.fsm.manual_lock_and_sos()
    assert not success
    assert len(app.fsm.dispatch_manager.sos_queue) == 0

    # Now move scout close to survivor S1 (within standoff distance)
    s1 = app.arena.survivors[0]
    app.robot.x = s1.x + 35.0
    app.robot.y = s1.y
    assert app.fsm.can_manual_lock()

    # Simulate key press E via pygame event queue
    initial_locked = app.fsm.locked_survivors_count
    event_e = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e, mod=0)
    pygame.event.post(event_e)
    app.handle_events()

    assert app.fsm.locked_survivors_count == initial_locked + 1
    assert s1.id in app.fsm.sent_sos_ids
    assert len(app.fsm.dispatch_manager.sos_queue) == 1
    queued_sos = app.fsm.dispatch_manager.sos_queue[0]
    assert queued_sos.survivor_id == s1.id
    assert queued_sos.triage == s1.triage


def test_manual_mode_dispatch_key_d():
    """
    Validates that pressing D in manual mode dispatches a free rescuer to the queued SOS.
    """
    app = RakshakApp()
    app.manual_mode = True
    app.fsm.enter_manual_mode()

    s1 = app.arena.survivors[0]
    app.robot.x = s1.x + 40.0
    app.robot.y = s1.y

    # Lock & send SOS
    app.fsm.manual_lock_and_sos()
    assert len(app.fsm.dispatch_manager.sos_queue) == 1
    assert app.fsm.can_manual_dispatch()

    # Simulate pressing D via pygame event queue -> must NOT dispatch (D is steering only)
    event_d = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d, mod=0)
    pygame.event.post(event_d)
    app.handle_events()
    assert len(app.fsm.dispatch_manager.sos_queue) == 1, "D key must not dispatch rescuer"

    # Simulate pressing Enter -> dispatches rescuer
    event_enter = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0)
    pygame.event.post(event_enter)
    app.handle_events()

    # SOS should be dispatched and rescuer assigned
    assert len(app.fsm.dispatch_manager.sos_queue) == 0
    assert s1.id in app.fsm.dispatch_manager.active_sos
    sos = app.fsm.dispatch_manager.active_sos[s1.id]
    assert sos.status == "DISPATCHED"
    assert sos.assigned_rescuer_id is not None

    # Check the assigned rescuer's state
    assigned_rescuer = next((r for r in app.arena.rescuers if r.name == sos.assigned_rescuer_id), None)
    assert assigned_rescuer is not None
    assert assigned_rescuer.state == RescuerState.DISPATCHED
    assert assigned_rescuer.assigned_survivor_id == s1.id


def test_manual_mode_extraction_complete_key_x_and_missions_counter():
    """
    Simulates complete manual rescue flow:
    1. Lock survivor via [E]
    2. Dispatch rescuer via [D]
    3. Rescuer arrives at perimeter (<= 50 px)
    4. Confirm extraction via [X]
    5. Complete extraction and verify missions counter increments!
    """
    app = RakshakApp()
    app.manual_mode = True
    app.fsm.enter_manual_mode()

    s1 = app.arena.survivors[0]
    app.robot.x = s1.x + 40.0
    app.robot.y = s1.y

    initial_missions = app.fsm.missions_completed
    assert initial_missions == 0

    # Step 1: Simulate E key -> lock and send SOS
    event_e = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e, mod=0)
    pygame.event.post(event_e)
    app.handle_events()
    assert len(app.fsm.dispatch_manager.sos_queue) == 1

    # Step 2: Simulate Enter key -> dispatch rescuer
    event_return = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0)
    pygame.event.post(event_return)
    app.handle_events()
    assert len(app.fsm.dispatch_manager.active_sos) == 1

    sos = app.fsm.dispatch_manager.active_sos[s1.id]
    rescuer = next(r for r in app.arena.rescuers if r.name == sos.assigned_rescuer_id)
    assert rescuer.state == RescuerState.DISPATCHED

    # Step 3: Move rescuer to within perimeter (<= 50 px) of survivor
    rescuer.x = s1.x + 30.0
    rescuer.y = s1.y
    app.update(0.05)
    assert app.fsm.can_manual_extract()

    # Step 4: Simulate X key -> confirm / start extraction
    event_x = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x, mod=0)
    pygame.event.post(event_x)
    app.handle_events()
    assert rescuer.state == RescuerState.EXTRACTING

    # Step 5: Advance extraction to completion
    rescuer.extraction_timer = rescuer.extraction_duration
    dt = 0.05
    for _ in range(20):
        app.update(dt)
        if app.fsm.missions_completed > initial_missions:
            break

    assert s1.is_secured
    assert app.fsm.missions_completed == 1, f"Expected 1 mission completed, got {app.fsm.missions_completed}"


def test_manual_mode_triage_priority_dispatch():
    """
    Validates that in manual mode, pressing D dispatches according to triage priority:
    CRITICAL > INJURED > STABLE.
    """
    app = RakshakApp()
    app.manual_mode = True
    app.fsm.enter_manual_mode()

    # Enqueue 3 SOS messages with different priorities
    dm = app.fsm.dispatch_manager
    dm.submit_sos("S_STABLE", 200, 200, "STABLE", current_time=1.0)
    dm.submit_sos("S_CRITICAL", 300, 300, "CRITICAL", current_time=1.1)
    dm.submit_sos("S_INJURED", 400, 400, "INJURED", current_time=1.2)

    # First D press should dispatch CRITICAL
    app.fsm.manual_dispatch_rescuer()
    assert "S_CRITICAL" in dm.active_sos
    assert "S_CRITICAL" not in [s.survivor_id for s in dm.sos_queue]

    # Second D press should dispatch INJURED
    app.fsm.manual_dispatch_rescuer()
    assert "S_INJURED" in dm.active_sos
    assert "S_INJURED" not in [s.survivor_id for s in dm.sos_queue]

    # Third D press should dispatch STABLE
    app.fsm.manual_dispatch_rescuer()
    assert "S_STABLE" in dm.active_sos
    assert len(dm.sos_queue) == 0


def test_manual_to_auto_mode_clean_fsm_state():
    """
    Validates that returning from manual to autonomous mode ([M] key) cleanly restores
    FSM state without getting stuck or leaving half-complete states.
    """
    app = RakshakApp()
    app.manual_mode = True
    app.fsm.enter_manual_mode()
    assert app.fsm.manual_mode is True

    # Case A: Toggle back to auto with no active SOS -> should cleanly enter SEARCH_PATROL
    event_m = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m, mod=0)
    pygame.event.post(event_m)
    app.handle_events()
    assert app.manual_mode is False
    assert app.fsm.manual_mode is False
    assert app.fsm.current_state == MissionState.SEARCH_PATROL

    # Case B: If all survivors are already secured, toggle to auto -> MISSION_COMPLETE
    for s in app.arena.survivors:
        s.is_secured = True
    app.fsm.missions_completed = len(app.arena.survivors)
    app.fsm.enter_manual_mode()
    app.fsm.exit_manual_mode()
    assert app.fsm.current_state == MissionState.MISSION_COMPLETE


def test_siren_procedural_files_duration():
    """Validates that siren_critical.wav and siren_injured.wav exist and are >= 1.5s."""
    import wave
    sm = get_sound_manager()
    for name in ("siren_critical.wav", "siren_injured.wav"):
        p = os.path.join(sm.sounds_dir, name)
        assert os.path.exists(p), f"{name} must exist in {sm.sounds_dir}"
        with wave.open(p, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            assert duration >= 1.5, f"{name} duration must be >= 1.5s, got {duration:.2f}s"


def test_sound_three_detections_in_a_row(capsys):
    """
    Validates that with three survivor detections in a row, the sound function is
    called three times (every event plays its sound every time, no once-flags on beeps).
    """
    sm = get_sound_manager()
    sm.set_muted(False)

    # 3 detections in a row
    sm.play_survivor_detected("S1", "CRITICAL")
    sm.play_survivor_detected("S2", "INJURED")
    sm.play_survivor_detected("S3", "STABLE")

    captured = capsys.readouterr().out
    assert "[AUDIO] played: survivor_detected S1 CRITICAL" in captured
    assert "[AUDIO] played: survivor_detected S2 INJURED" in captured
    assert "[AUDIO] played: survivor_detected S3 STABLE" in captured


def test_siren_called_once_per_survivor_with_correct_severity(capsys):
    """
    Validates:
    - CRITICAL: fast urgent siren played on dedicated channel
    - INJURED: slower siren played once
    - STABLE: short beep only, no siren
    - Per-survivor once flag prevents duplicate sirens until secured or reset
    """
    sm = get_sound_manager()
    sm.set_muted(False)
    sm.clear_siren_flag()

    # CRITICAL: siren plays first time
    sm.play_siren("S1", "CRITICAL")
    out1 = capsys.readouterr().out
    assert "[AUDIO] played: siren_critical S1 CRITICAL" in out1

    # CRITICAL duplicate: should skip with flag set
    sm.play_siren("S1", "CRITICAL")
    out2 = capsys.readouterr().out
    assert "[AUDIO] skipped: flag set" in out2

    # INJURED: siren plays first time
    sm.play_siren("S2", "INJURED")
    out3 = capsys.readouterr().out
    assert "[AUDIO] played: siren_injured S2 INJURED" in out3

    # INJURED duplicate: should skip with flag set
    sm.play_siren("S2", "INJURED")
    out4 = capsys.readouterr().out
    assert "[AUDIO] skipped: flag set" in out4

    # STABLE: no siren
    sm.play_siren("S3", "STABLE")
    out5 = capsys.readouterr().out
    assert "siren" not in out5


def test_sound_nothing_plays_when_muted(capsys):
    """Validates that nothing plays when muted and all sound events log skipped: muted."""
    sm = get_sound_manager()
    sm.set_muted(True)

    sm.play_survivor_detected("S1", "CRITICAL")
    sm.play_siren("S1", "CRITICAL")
    sm.play_rescuer_dispatched("S1", "CRITICAL")
    sm.play_extraction_complete("S1", "CRITICAL")
    sm.play_hazard_warning(cooldown=0.0)
    sm.play_mission_complete()

    captured = capsys.readouterr().out
    assert "[AUDIO] played:" not in captured
    assert captured.count("[AUDIO] skipped: muted") >= 5

    sm.set_muted(False)


def test_siren_flags_reset_after_mission_reset(capsys):
    """Validates that per-survivor siren flags are cleared when mission reset occurs."""
    sm = get_sound_manager()
    sm.set_muted(False)
    sm.clear_siren_flag()

    # Play siren for S1
    sm.play_siren("S1", "CRITICAL")
    assert "[AUDIO] played: siren_critical S1 CRITICAL" in capsys.readouterr().out

    # Second call skipped
    sm.play_siren("S1", "CRITICAL")
    assert "[AUDIO] skipped: flag set" in capsys.readouterr().out

    # Reset via app.reset_mission()
    app = RakshakApp()
    app.reset_mission(randomize_survivor=False)

    # Now siren for S1 should play again
    sm.play_siren("S1", "CRITICAL")
    assert "[AUDIO] played: siren_critical S1 CRITICAL" in capsys.readouterr().out


def test_siren_flag_cleared_after_survivor_secured(capsys):
    """Validates that per-survivor siren flag is cleared after that survivor is secured."""
    sm = get_sound_manager()
    sm.set_muted(False)
    sm.clear_siren_flag()

    # Play siren for S1
    sm.play_siren("S1", "CRITICAL")
    assert "[AUDIO] played: siren_critical S1 CRITICAL" in capsys.readouterr().out

    # Duplicate skipped
    sm.play_siren("S1", "CRITICAL")
    assert "[AUDIO] skipped: flag set" in capsys.readouterr().out

    # Survivor S1 secured -> clear siren flag for S1 only
    sm.clear_siren_flag("S1")

    # Siren for S1 can now play again
    sm.play_siren("S1", "CRITICAL")
    assert "[AUDIO] played: siren_critical S1 CRITICAL" in capsys.readouterr().out


def test_fullscreen_default_and_cli_flag():
    """Validates that RakshakApp starts windowed by default and respects the fullscreen flag."""
    app_windowed = RakshakApp(fullscreen=False)
    assert not app_windowed.fullscreen

    app_fs = RakshakApp(fullscreen=True)
    assert app_fs.fullscreen


def test_fullscreen_toggle_f11_and_alt_enter():
    """Validates that F11 and Alt+Enter toggle fullscreen without interfering with other keybindings."""
    app = RakshakApp(fullscreen=False)
    assert not app.fullscreen

    # Press F key -> toggle to fullscreen
    event_f = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0)
    pygame.event.post(event_f)
    app.handle_events()
    assert app.fullscreen

    # Press F key again -> toggle back to windowed
    pygame.event.post(event_f)
    app.handle_events()
    assert not app.fullscreen

    # Press Cmd+Enter -> toggle to fullscreen
    event_cmd_enter = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=pygame.KMOD_GUI)
    pygame.event.post(event_cmd_enter)
    app.handle_events()
    assert app.fullscreen

    # Press Cmd+Enter again -> toggle back to windowed
    pygame.event.post(event_cmd_enter)
    app.handle_events()
    assert not app.fullscreen

    # Press F11 -> toggle to fullscreen
    event_f11 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F11, mod=0)
    pygame.event.post(event_f11)
    app.handle_events()
    assert app.fullscreen

    # Press F11 again -> toggle back to windowed
    pygame.event.post(event_f11)
    app.handle_events()
    assert not app.fullscreen

    # Press Alt+Enter -> toggle to fullscreen
    event_alt_enter = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=pygame.KMOD_ALT)
    pygame.event.post(event_alt_enter)
    app.handle_events()
    assert app.fullscreen

    # Press Alt+Enter again -> toggle back to windowed
    pygame.event.post(event_alt_enter)
    app.handle_events()
    assert not app.fullscreen


def test_fullscreen_esc_exits_to_windowed_without_quitting():
    """Validates that ESC in fullscreen exits back to windowed mode without terminating the application."""
    app = RakshakApp(fullscreen=True)
    assert app.fullscreen
    assert app.running

    # Press ESC while in fullscreen -> switches to windowed, does NOT quit
    event_esc = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)
    pygame.event.post(event_esc)
    app.handle_events()
    assert not app.fullscreen
    assert app.running

    # Press ESC while in windowed -> does not quit (quit is via window close)
    pygame.event.post(event_esc)
    app.handle_events()
    assert not app.fullscreen
    assert app.running

    # Window close event quits
    event_quit = pygame.event.Event(pygame.QUIT)
    pygame.event.post(event_quit)
    app.handle_events()
    assert not app.running


def test_screen_to_logical_coordinate_mapping():
    """Validates logical coordinate conversion across 1:1, scaled 16:9, and letterboxed 16:10 displays."""
    app = RakshakApp()

    # Case 1: Standard 1:1 windowed (1280x720)
    app._update_scaling(1280, 720)
    assert app.scale == 1.0
    assert app.offset_x == 0 and app.offset_y == 0
    lx, ly = app.screen_to_logical((400, 300))
    assert (lx, ly) == (400, 300)

    # Case 2: 1920x1080 Full HD (1.5x scale, no letterbox)
    app._update_scaling(1920, 1080)
    assert abs(app.scale - 1.5) < 1e-4
    assert app.offset_x == 0 and app.offset_y == 0
    lx, ly = app.screen_to_logical((int(400 * 1.5), int(300 * 1.5)))
    assert (lx, ly) == (400, 300)

    # Case 3: 1920x1200 16:10 display (letterboxed with top/bottom black bars)
    app._update_scaling(1920, 1200)
    # scale = min(1920/1280, 1200/720) = min(1.5, 1.6667) = 1.5
    # scaled height = 720 * 1.5 = 1080; offset_y = (1200 - 1080) // 2 = 60
    assert abs(app.scale - 1.5) < 1e-4
    assert app.offset_x == 0
    assert app.offset_y == 60
    # Screen center (960, 600) should map to logical center (640, 360)
    lx, ly = app.screen_to_logical((960, 600))
    assert (lx, ly) == (640, 360)


def test_fullscreen_toggle_preserves_simulation_state():
    """Validates that toggling fullscreen preserves mission state, pause state, robot pose, and audio."""
    app = RakshakApp(fullscreen=False)

    # Setup specific simulation state
    app.fsm.current_state = MissionState.RESCUE_OPERATION
    app.fsm.missions_completed = 3
    app.robot.x = 245.0
    app.robot.y = 380.0
    app.robot.theta = 1.25
    app.paused = True
    app.sound_manager.set_muted(True)
    app.manual_mode = True

    # Toggle to fullscreen
    app.toggle_fullscreen()
    assert app.fullscreen

    # Verify all state is intact
    assert app.fsm.current_state == MissionState.RESCUE_OPERATION
    assert app.fsm.missions_completed == 3
    assert app.robot.x == 245.0
    assert app.robot.y == 380.0
    assert app.robot.theta == 1.25
    assert app.paused is True
    assert app.sound_manager.is_muted() is True
    assert app.manual_mode is True

    # Toggle back to windowed
    app.toggle_fullscreen()
    assert not app.fullscreen

    # Verify all state remains intact
    assert app.fsm.current_state == MissionState.RESCUE_OPERATION
    assert app.fsm.missions_completed == 3
    assert app.robot.x == 245.0
    assert app.robot.y == 380.0
    assert app.robot.theta == 1.25
    assert app.paused is True
    assert app.sound_manager.is_muted() is True
    assert app.manual_mode is True

