"""
make_demo.py  –  Rakshak scripted demo recorder
================================================
Runs the simulation at 2× speed (headless SDL dummy driver) and injects
synthetic pygame KEYDOWN events at pre-planned sim-time cues.

Every rendered frame is captured via cv2.VideoWriter and saved to demo.mp4
(1280×720, 30 fps ≈ 2 min 45 s of real video).

Usage:
    python make_demo.py
    python make_demo.py --out my_demo.mp4   # custom output name
"""

import os
import sys
import argparse
import time

# ── headless SDL: use pure-software 'dummy' driver (no OpenGL/GPU needed) ───
# Remove or comment these two lines to watch the window live on screen instead.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import cv2
import numpy as np

# ── import the app AFTER env vars ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import RakshakApp
from fsm import MissionState

# ─────────────────────────────────────────────────────────────────────────────
# DEMO SCRIPT
# Each entry: (sim_time_seconds, key_constant, label)
#   sim_time  – when (in simulation seconds) to fire the event
#   key       – pygame.K_* constant
#   label     – printed to console so you know what's happening
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT = [
    # ── SCENE 1: Intro – let Priority sweep start, show state bar ─────────────
    # Nothing to press – just let the scout scan for ~25 s of sim time

    # ── SCENE 2: Overlay toggles – make it visually rich ─────────────────────
    (18.0,  pygame.K_y,  "A* overlay ON"),
    (22.0,  pygame.K_v,  "APF vectors ON"),
    (26.0,  pygame.K_o,  "Coverage grid ON"),
    (30.0,  pygame.K_o,  "Coverage grid OFF"),
    (32.0,  pygame.K_v,  "APF vectors OFF"),
    (34.0,  pygame.K_y,  "A* overlay OFF"),

    # ── SCENE 3: Sensor noise demo ────────────────────────────────────────────
    (36.0,  pygame.K_n,  "Noise ON"),
    (42.0,  pygame.K_n,  "Noise OFF"),

    # ── SCENE 4: Triage dispatch begins (Priority sweep usually done by ~50 s) ─
    # Just watch rescuers deploy in CRITICAL->INJURED->STABLE order

    # ── SCENE 5: Manual mode demo (enter while rescuers are working) ──────────
    (60.0,  pygame.K_m,  "Manual mode ON"),
    (63.0,  pygame.K_e,  "Lock and SOS"),
    # Drive forward 2 s
    (63.0,  None, "DRIVE_START"),   # direct set_twist, not key injection
    (65.5,  None, "DRIVE_STOP"),
    (65.5,  pygame.K_RETURN, "Dispatch rescuer"),
    (68.0,  pygame.K_x,  "Confirm extraction"),
    (70.0,  pygame.K_m,  "Manual mode OFF  (back to auto)"),

    # ── SCENE 6: Priority mode toggle ─────────────────────────────────────────
    (75.0,  pygame.K_p,  "Priority mode OFF"),
    (78.0,  pygame.K_p,  "Priority mode ON"),

    # ── SCENE 7: Mute / unmute demo ───────────────────────────────────────────
    (82.0,  pygame.K_u,  "Mute ON"),
    (85.0,  pygame.K_u,  "Mute OFF"),

    # ── SCENE 8: Pause / resume ───────────────────────────────────────────────
    (90.0,  pygame.K_SPACE, "Pause"),
    (94.0,  pygame.K_SPACE, "Resume"),

    # ── SCENE 9: Watch remaining rescues complete, mission overlay appears ─────
    # (nothing to press; let it finish up to MAX_SIM_TIME)
]

# ── runtime constants ─────────────────────────────────────────────────────────
SIM_SPEED    = 2.0      # 2x faster than real time
TARGET_FPS   = 30       # output video fps
MAX_SIM_TIME = 165.0    # stop after this many sim-seconds (<=3 min video)
OUT_TITLE    = "RAKSHAK // SAR  -  Autonomous Search & Rescue Demo"


def inject_key(key):
    """Post a synthetic KEYDOWN+KEYUP pair into pygame's event queue."""
    pygame.event.post(pygame.event.Event(
        pygame.KEYDOWN,
        key=key, mod=0, unicode="", scancode=0
    ))
    pygame.event.post(pygame.event.Event(
        pygame.KEYUP,
        key=key, mod=0, unicode="", scancode=0
    ))


def add_title_bar(frame, text, sim_t):
    """Burn a slim title bar + sim-clock onto the frame (no external fonts needed)."""
    bar = np.zeros((28, frame.shape[1], 3), dtype=np.uint8)
    bar[:] = (18, 22, 30)
    clock_txt = "  SIM {:6.1f}s  |  {}".format(sim_t, text)
    cv2.putText(bar, clock_txt, (8, 19),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 220, 180), 1, cv2.LINE_AA)
    return np.vstack([bar, frame])


def main(out_path, seed=None):
    pygame.init()
    pygame.font.init()

    # Create the app (windowed 1280x720, Priority mode ON)
    app = RakshakApp(fps=TARGET_FPS, priority_mode=True, seed=seed)

    # Override: write OUR video, not the built-in demo writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    W, H = app.WINDOW_WIDTH, app.WINDOW_HEIGHT
    vw = cv2.VideoWriter(out_path, fourcc, float(TARGET_FPS), (W, H + 28))
    if not vw.isOpened():
        print("[ERROR] Cannot open video writer for {}".format(out_path))
        pygame.quit()
        sys.exit(1)

    clock    = pygame.time.Clock()
    sim_time = 0.0
    dt_fixed = 1.0 / TARGET_FPS          # fixed physics step
    script   = sorted(SCRIPT, key=lambda x: x[0])
    fired    = set()

    print("[DEMO] Recording {}  ({}x{}, {} fps)".format(out_path, W, H + 28, TARGET_FPS))
    print("[DEMO] Max sim time: {:.0f}s  <=3 min video".format(MAX_SIM_TIME))

    app._demo_driving = False
    frame_idx  = 0
    wall_start = time.time()

    while app.running and sim_time <= MAX_SIM_TIME:
        # ── Fire scripted events ──────────────────────────────────────────────
        for idx, (t, key, label) in enumerate(script):
            if idx not in fired and sim_time >= t:
                fired.add(idx)
                print("[{:6.1f}s] >> {}".format(sim_time, label))
                if label == "DRIVE_START":
                    app._demo_driving = True
                elif label == "DRIVE_STOP":
                    app._demo_driving = False
                    app.robot.set_twist(0.0, 0.0)
                elif key is not None:
                    inject_key(key)

        # ── Pump events (picks up both real & injected) ───────────────────────
        app.handle_events()

        # ── Manual drive: direct set_twist (SDL dummy ignores injected arrow keys) ──
        if getattr(app, '_demo_driving', False) and app.manual_mode:
            app.robot.set_twist(85.0, 0.0)  # forward at scout speed

        # ── Step physics SIM_SPEED times per rendered frame ───────────────────
        sub_steps = max(1, int(SIM_SPEED))
        sub_dt    = dt_fixed * SIM_SPEED / sub_steps
        for _ in range(sub_steps):
            app.update(sub_dt)
        sim_time += dt_fixed * SIM_SPEED

        # ── Render one frame ──────────────────────────────────────────────────
        app.render()

        # ── Grab frame from pygame surface -> cv2 -> video ────────────────────
        raw    = pygame.surfarray.array3d(app.screen)         # (W, H, 3) RGB
        bgr    = cv2.cvtColor(np.transpose(raw, (1, 0, 2)),   # -> (H, W, 3) BGR
                              cv2.COLOR_RGB2BGR)
        titled = add_title_bar(bgr, OUT_TITLE, sim_time)
        vw.write(titled)
        frame_idx += 1

        # ── Cap wall-clock speed ───────────────────────────────────────────────
        clock.tick(TARGET_FPS * 4)

        # ── Progress every 5 s of sim time ────────────────────────────────────
        if frame_idx % (TARGET_FPS * 5) == 0:
            wall_elapsed = time.time() - wall_start
            pct = sim_time / MAX_SIM_TIME * 100
            print("[DEMO] {:.0f}s sim / {:.0f}s  ({:.0f}%)  |  {} frames  |  wall {:.0f}s".format(
                sim_time, MAX_SIM_TIME, pct, frame_idx, wall_elapsed))

    # ── Wrap up ───────────────────────────────────────────────────────────────
    vw.release()
    pygame.quit()

    wall_total = time.time() - wall_start
    video_dur  = frame_idx / TARGET_FPS
    size_mb    = os.path.getsize(out_path) / 1e6 if os.path.exists(out_path) else 0
    print("\n[DEMO] Done!  {} frames -> {:.1f}s video  ({:.1f} MB)  rendered in {:.0f}s".format(
        frame_idx, video_dur, size_mb, wall_total))
    print("[DEMO] Output: {}".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rakshak scripted demo recorder")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible layout")
    parser.add_argument("--out", default="demo.mp4",
                        help="Output video filename (default: demo.mp4)")
    args = parser.parse_args()
    main(args.out, seed=args.seed)
