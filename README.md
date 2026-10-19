# Rakshak // Autonomous Search-and-Rescue Robot
### *"See. Locate. Rescue."*

**Hackathon:** Hack in Hills '26 &nbsp;|&nbsp; **Problem Statement:** PS 01 — Autonomous Vision-Guided Robot  
**Demo:** [▶ YouTube Demo (Placeholder)](https://youtu.be/placeholder) &nbsp;|&nbsp; **Screenshots:** *(see docs/screenshots/)*

---

## Problem Statement (PS 01)

> *"Build an autonomous robot that uses an AI-powered vision system to understand its surroundings,
> identify relevant objects and people, and navigate autonomously toward a given objective while
> detecting and avoiding obstacles. Real or simulated environment is allowed."*

Rakshak answers this challenge with a complete 2-D disaster simulation: a scout robot patrols
rubble-filled terrain, detects survivors through a synthetic camera, triages them by severity, and
dispatches up to three rescuers in priority order — all without human intervention.

<!-- Screenshots (placeholders) -->
<!-- ![Rakshak Simulation Overview](docs/screenshots/overview.png) -->
<!-- ![Vision Pipeline & Telemetry](docs/screenshots/vision_telemetry.png) -->

---

## Key Features

| Feature | Details |
|---------|---------|
| **Autonomous FSM** | 6-state mission controller: `SEARCH_PATROL → VERIFY_TARGET → PLAN_ROUTE → NAVIGATE → RESCUE_OPERATION → MISSION_COMPLETE` |
| **Vision pipeline** | Synthetic 160×90 camera → OpenCV HSV segmentation → contour analysis → range/bearing estimation |
| **Honest triage** | 8-frame majority-vote (≥60%) on perceived detector labels; ground truth never used for dispatch |
| **Triage-priority dispatch** | CRITICAL → INJURED → STABLE, with concurrent R1/R2/R3 rescuers |
| **Priority Mode** | Full 90% area sweep before rescue; queue sorted by triage rank then distance |
| **A* planning** | Global path on 800×640 occupancy grid with obstacle inflation |
| **APF avoidance** | Artificial Potential Field with tangential vortex force (no local-minima traps) |
| **Dynamic hazards** | Zones grow +0.5 px/s; robot replans automatically on expansion |
| **Camera noise** | Gaussian pixel noise + drifting smoke haze (toggle `N`) |
| **Fullscreen** | `F`/`F11`; Retina-safe letterbox scaling; mouse coords remapped |
| **Manual override** | `M` → steer with WASD, `E` lock, `Enter` dispatch, `X` extract |
| **95 tests** | Full pytest suite covering FSM, tracker, detector, vision, dispatch, coverage, hazard, manual mode, triage, seed reproducibility |

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                         app.py                             │
│          RakshakApp  (main loop · render · input)          │
└──────┬──────────┬─────────────┬──────────────┬────────────┘
       │          │             │              │
    fsm.py   rescuer.py     sound.py       arena.py
  MissionFSM  DispatchMgr   SoundMgr    DisasterArena
       │          │                          │
    robot.py   camera.py               astar.py / apf.py
    Robot     Camera+noise             Path planners
       │
  detector.py ──► tracker.py ──► coverage.py
  Vision pipeline  Smoothing     Area tracking
```

### How Vision Works

```
Arena frame (800×640)
   │
   ▼ camera.py — rasterise 160×90 first-person buffer every 2 frames
   │
   ▼ detector.py — HSV mask (orange: H 7-25) bounded to triage badge area
                   contour analysis → bbox, range Z, bearing θ
   │
   ▼ tracker.py — nearest-estimate data association (gate 85 px)
                  EMA smoothing (α=0.35)
                  8-frame majority vote ≥60% → perceived triage_level
                  Variance-lock gating (N=3 frames, σ≤20 px) → is_locked=True
                  Label frozen at lock; true_triage stored for metrics only
   │
   ▼ fsm.py — SOS issued; dispatch queue sorted by survivor_priority_key()
              A* plans route; APF smooths execution
```

---

## Install & Run

```bash
# 1. Clone and enter
git clone <repo-url>
cd Rakshak

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python3 app.py                    # default: Priority Mode ON, windowed
python3 app.py --fullscreen       # start fullscreen
python3 app.py --no-priority      # rescue-as-found mode
python3 app.py --seed 42          # reproducible layout
python3 run.py --benchmark        # headless benchmark (20 layouts)
python3 make_demo.py              # record 82-second demo.mp4
```

---

## Key Bindings

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `R` | Reset mission (new random layout) |
| `M` | Toggle Manual / Autonomous |
| `E` | *(Manual)* Lock nearest survivor + issue SOS |
| `WASD` / Arrows | *(Manual)* Steer scout |
| `Enter` | *(Manual)* Dispatch rescuer |
| `X` | *(Manual)* Confirm extraction |
| `U` | Mute / Unmute audio |
| `F` / `F11` | Toggle fullscreen |
| `Esc` | Exit fullscreen |
| `P` / `T` | Toggle Priority Mode |
| `Y` | A* path overlay |
| `V` | APF vectors overlay |
| `O` | Coverage grid overlay |
| `N` | Camera noise + smoke haze |
| `H` | HSV mask camera feed |
| `F1` | FPS counter |

---

## CLI Flags

| Flag | Description |
|------|-------------|
| `--fullscreen` | Start in fullscreen |
| `--no-priority` | Rescue-as-found mode (no sweep) |
| `--seed N` | Fixed random seed for reproducible layout |
| `--lite` | Reduced rendering for low-spec hardware |
| `--benchmark` | Headless 20-layout benchmark |
| `--layouts N` | Number of benchmark layouts (default 20) |
| `--max-frames N` | Max frames per benchmark layout (default 3000) |
| `--demo` | Record 3-layout demo.mp4 |

---

## Results

### Benchmark (20 random layouts, 3000 frames/layout)

Evaluated headlessly via `.venv/bin/python run.py --benchmark` across 20 distinct layouts:

| Metric | Value |
|--------|-------|
| **Reach rate** | **100.0%** (20/20 layouts reached) |
| **Collision count** | **0** (zero obstacle, hazard, wall, or rescuer collisions) |
| **Longest idle time** | **0.20 s** (well below 3.0 s freeze threshold) |
| **Median localization error** | **8.92 px** (at lock time) |
| **Median rescue error** | **2.25 px** (at rescue standoff) |
| **Triage accuracy** | **100.0%** (25/25 frozen survivor targets correctly classified) |
| **Test suite** | **95 tests passing** in pytest |

#### Benchmark Confusion Matrix (Rows = Ground Truth, Columns = Perceived)

```
              CRITICAL   INJURED    STABLE
  CRITICAL          12         0         0
  INJURED            0         8         0
  STABLE             0         0         5
```

> Reproduce at any time:
> ```bash
> .venv/bin/python run.py --benchmark
> ```

### Perception & Accuracy Engineering

1. **Range sanity gating:** Detections with estimated distance $Z > 340$ px or $Z < 10$ px are automatically discarded.
2. **Variance-based lock confirmation:** Position locked only after $\ge 4$ consecutive stable observations with spatial variance $\sigma \le 20$ px.
3. **EMA coordinate smoothing:** Exponential moving average ($lpha = 0.35$) eliminates single-frame jitter.
4. **Honest majority-vote triage:** Ground-truth labels are never consulted for operational decisions. Perceived triage requires $\ge 70\%$ vote consensus across observations before freezing.

---

## Limitations & Future Scope

| Limitation | Technical Detail | Future Scope |
|---|---|---|
| **INJURED vs STABLE long-range confusion** | At distance > 220 px, badge size shrinks to $<4$ px, risking color confusion with torso reflective striping ($H=23$). Resolved via vertical ROI tightening ($0.45h$–$0.85h$) and tuning yellow band to $H \in [26, 38]$. | Implement multi-scale super-resolution or zoom ROI on candidate survivor boxes. |
| **2-D synthetic camera** | Camera rasterizes 2.5-D column projections in 160×90 rather than true 3-D mesh raytracing or RGB-D sensors. | Port sensor stack to ROS 2 Humble + Gazebo simulation with simulated RealSense D435. |
| **Simulation-only environment** | Physics, wheel slip, and sensor latency are ideal 2-D kinematic models. | Deploy on physical differential-drive rover platform with STM32 motor controller. |
| **Single-scout agent** | One scout robot conducts sweeps while 3 autonomous rescuers assist. | Implement multi-agent auction-based dispatch algorithm for collaborative multi-scout swarms. |
| **Fixed arena dimensions** | Disaster site is bounded to 800×650 px grid. | Integrate real-time Occupancy Grid Mapping (Cartographer / SLAM Toolbox) for unbounded environments. |

---

## Test Suite

```bash
.venv/bin/python -m pytest tests/ -v    # run all 95 tests
.venv/bin/python -m pytest tests/ -q    # quiet summary
```

Test files cover: FSM rescue timer, triage/ID consistency, honest vision triage,
manual mode + sound, phase 2 dispatch, phase 3 coverage/battery, phase 4 hazard/noise,
priority mode, random layouts benchmark, seed reproducibility, kinematics, sensors,
camera, A*, APF, detector, tracker.

---

*Built with Python 3.9 · Pygame 2.6 · NumPy 2.0 · OpenCV 5.0*  
*Hack in Hills '26 — "See. Locate. Rescue."*
