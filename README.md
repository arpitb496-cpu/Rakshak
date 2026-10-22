# Rakshak // SAR: Autonomous Search-and-Rescue Robot Simulation

**Hack in Hills '26 · Problem Statement 01: Vision-Based Autonomous Robot · Team Bihari_Coders**

Rakshak is a 2-D autonomous search-and-rescue (SAR) simulation. A scout robot patrols a disaster arena, finds survivors with a camera and a vision pipeline, triages them (CRITICAL / INJURED / STABLE), and dispatches up to three rescuers in triage-priority order. An operator can take over at any time with a manual mode.

- Demo video: **<ADD YOUTUBE LINK HERE>**
- Screenshots: `docs/screenshots/01_autonomous.png`, `02_manual.png`, `03_complete.png`

![Autonomous run](docs/screenshots/01_autonomous.png)

## Problem

After a disaster, the first hours decide who survives, but responders search blind, and the most critical victims are not always found first. Rakshak shows how an AI-vision scout can find victims, judge urgency, and send help in the right order, with safe navigation and zero collisions. It runs in a simulated environment, as allowed by the problem statement.

## Key features

- **Vision pipeline:** a synthetic 160x90 camera is rasterised from the arena; HSV segmentation reads the triage badge on each survivor; a multi-frame tracker smooths and locks the estimate.
- **Honest triage:** the severity label comes only from what the camera perceives (majority vote over the last 8 frames, at least 70% agreement, frozen after lock). Ground truth is used only to measure accuracy.
- **Navigation:** A* global planning with safety inflation, plus Artificial Potential Field (APF) reactive avoidance with a tangential force to avoid local minima; 8-ray proximity sensors; differential-drive kinematics.
- **Dynamic hazards:** hazard zones grow over time and trigger re-planning; rubble can shift.
- **Dispatch:** 3 rescuers (R1-R3) with states STANDBY / EN_ROUTE / EXTRACTING / SECURED, stuck detection, re-planning and SOS release. The assigned rescuer's target is refreshed as the track estimate converges.
- **Priority mode (default):** the scout completes a ~99% area sweep, queues all survivors, and dispatches in triage order. Switch it off for rescue-as-found.
- **Manual override:** steer the scout, lock a survivor, issue SOS, dispatch and confirm extraction from the keyboard.
- **Telemetry HUD:** state, battery, area scanned, camera feed, radar, target telemetry (live error and lock error), rescue team, SOS queue, event log; optional A* / APF / coverage / HSV overlays.
- **Sound events:** detection, sirens per triage level, dispatch, extraction, mission complete (mute with `U`).
- **Mission report:** `mission_report.json` and a final map screenshot are written when a mission completes.
- **Quality:** 97 automated tests (FSM, tracker, detector, dispatch, coverage, hazards, manual mode, triage, seed reproducibility, benchmark metrics).

## How the vision works

```
Arena ──> Synthetic camera (160x90, noise/smoke) ──> HSV detector (triage badge only)
      ──> Tracker (association gate, EMA smoothing, 8-frame majority vote, lock + freeze)
      ──> Mission FSM ──> SOS queue ──> Dispatch manager ──> Rescuers (A* + APF)
```

- Detections beyond the camera range (340 px) or closer than 10 px are discarded.
- A target locks after at least 4 consecutive stable observations (spatial variance at most 20 px). Estimates use exponential smoothing (alpha 0.35) and are frozen once the survivor is secured.
- Track IDs are never reused, so a secured survivor can never be confused with a new one.

## Architecture

```
                         app.py  (RakshakApp: main loop, rendering, input)
      ┌───────────┬───────────┬────────────┬───────────┐
   fsm.py     rescuer.py   sound.py     arena.py     coverage.py
 MissionFSM  DispatchMgr   SoundMgr   DisasterArena  area tracker
      │           │                         │
  robot.py    camera.py  detector.py   astar.py / apf.py
  Robot       Camera     HSV vision    Planners
                          tracker.py
                          Smoothing + lock
```

Mission states: `SEARCH_PATROL → VERIFY_TARGET → PLAN_ROUTE → NAVIGATE_TO_SURVIVOR → RESCUE_OPERATION → MISSION_COMPLETE`.

## Install and run

Requirements: Python 3.9+ (developed and tested on macOS, Apple Silicon).

```bash
git clone https://github.com/arpitb496-cpu/Rakshak.git
cd Rakshak
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 app.py          # or ./rakshak.sh
```

Quick start: the mission starts automatically. Press `M` to take manual control, `R` for a new layout, `SPACE` to pause.

### CLI flags

```bash
python3 app.py                  # Priority mode, windowed
python3 app.py --seed 3         # reproducible layout
python3 app.py --no-priority    # rescue-as-found mode
python3 app.py --fullscreen     # start fullscreen
python3 app.py --lite           # reduced trails for slower machines

python3 run.py --benchmark                               # headless benchmark
python3 run.py --benchmark --priority --max-frames 12000 # full Priority missions
python3 run.py --benchmark --layouts 100 --seed 1000     # 100 held-out layouts
python3 make_demo.py --seed 3 --out demo.mp4             # scripted demo recorder
```

### Key bindings

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `R` | Reset mission (new layout) |
| `M` | Toggle manual / autonomous |
| `WASD` / arrows | (Manual) steer the scout |
| `E` | (Manual) lock nearest survivor and issue SOS |
| `Enter` | (Manual) dispatch nearest available rescuer |
| `X` | (Manual) confirm extraction |
| `P` / `T` | Toggle Priority mode |
| `U` | Mute / unmute |
| `F` / `F11`, `Esc` | Fullscreen on / off |
| `Y` / `V` / `O` | A* path / APF vectors / coverage grid overlays |
| `N` / `H` / `C` | Camera noise and smoke / HSV mask feed / camera frustum |
| `F1` | FPS counter |

## Benchmark results

All numbers were produced headlessly by `run.py` on the final commit (97 pytest tests passing). Parameters were tuned on the **dev layouts** (seeds 0-19); the **held-out layouts** (seeds 1000-1099) were used for evaluation. Layouts are seeded and random, but survivors are placed on a limited set of candidate spots, so layouts are not fully independent.

### Mode 1: Rescue-as-found (`--benchmark`)

The scout locks a survivor from close range (~150 px) and a rescuer is dispatched to the first survivor found.

| Metric | Dev (20 layouts) | Held-out (100 layouts) |
|---|---|---|
| Scout reached a survivor and rescue began | **20/20 (100%)** | **100/100 (100%)** |
| Collisions | **0** | **0** |
| Longest idle time | 0.20 s | 0.20 s |
| Localization error at lock (median / mean / max) | 8.25 / 11.77 / 56.57 px | 8.28 / 9.54 / 53.93 px |
| Rescue standoff error (median) | **2.50 px** | **2.47 px** |
| Triage accuracy | **100% (25/25)** | **96.6% (112/116)** |
| Unmatched locks | 0 | 1 |

Held-out confusion matrix (rows = true, columns = perceived):

```
              CRITICAL   INJURED    STABLE
  CRITICAL          52         0         0
  INJURED            4        30         0
  STABLE             0         0        30
```
All 4 errors are INJURED read as CRITICAL (over-triage); no survivor was ranked below its true urgency in this mode.

### Mode 2: Priority sweep (default app and demo mode)

The scout completes a ~99% area sweep, queues every detected survivor, then dispatches up to 3 rescuers in triage order. Survivors are detected at long range, so estimates are noisier than in Mode 1.

| Metric | Dev (20 layouts) | Held-out (100 layouts) |
|---|---|---|
| Mission reached MISSION_COMPLETE | **20/20 (100%)** | **100/100 (100%)** |
| Survivors physically reached by a rescuer (within 85 px) | **76/84 (90.5%)** | **371/398 (93.2%)** |
| Layouts where every arena survivor was reached | 14/20 (70%) | 75/100 (75%) |
| Survivors never detected | 0 | 10 |
| Collisions | **0** | **0** |
| Median mission time | 62.1 s | 54.2 s |
| Localization error at lock (median / mean / max) | 11.15 / 19.98 / 84.68 px | 11.33 / 19.04 / 82.77 px |
| Rescue standoff error (median) | 15.86 px | 12.97 px |
| Rescues more than 50 px from the estimate target | 14 / 76 | 50 / 361 |
| Triage accuracy | 91.5% (75/82) | 88.5% (339/383) |
| Unmatched locks | 5 | 38 |

"Mission reached MISSION_COMPLETE" means the state machine finished its sweep and queue; it does **not** mean every survivor was reached. The row "Survivors physically reached" is the real rescue rate.

Held-out confusion matrix (rows = true, columns = perceived):

```
              CRITICAL   INJURED    STABLE
  CRITICAL         158         2         2
  INJURED           20       104         2
  STABLE            16         2        77
```

### Reproduce
```bash
python3 run.py --benchmark
python3 run.py --benchmark --layouts 100 --seed 1000
python3 run.py --benchmark --priority --max-frames 12000
python3 run.py --benchmark --priority --max-frames 12000 --layouts 100 --seed 1000
.venv/bin/python -m pytest tests/ -q
```

## Limitations and future scope

| Limitation | Detail | Future scope |
|---|---|---|
| Simulation only | 2-D Pygame world with a synthetic camera (160x90) | ROS 2 with a real wheeled robot and camera |
| Long-range lock error | Early locks from far away can be 50-85 px off; the estimate converges to ~2.5 px at rescue standoff in Mode 1 | Range-adaptive lock gating, stereo/depth cues |
| Triage at long range | In Priority mode accuracy is ~89-92%; INJURED is sometimes read as CRITICAL, and a few CRITICAL survivors are read as lower urgency | Learned detector trained on real data, longer vote windows |
| Duplicate and merged tracks | Two survivors ~90 px apart can merge into one track; a far-range duplicate can send a second rescuer to the same survivor | Multi-hypothesis tracking, appearance cues |
| Noisy early estimates | A rescuer can stop 95-113 px from a survivor when dispatched on an early estimate and the path is blocked | Continuous re-targeting with path-aware standoff |
| Placement variety | Survivors spawn on a limited set of candidate spots; one benchmark layout spawns a survivor inside an obstacle | Continuous random placement with reachability checks |
| Coverage pockets | A few survivors in unscanned corners are never detected (10 of 398 in held-out Priority runs) | Frontier-based exploration of unreachable pockets |

Also planned: a response-time graph per survivor, a web telemetry dashboard, and a `--num-survivors` option.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Main loop, `RakshakApp`, rendering, keyboard input |
| `fsm.py` | `MissionFSM`: mission state logic, priority queue, sweep |
| `rescuer.py` | `DispatchManager`, `RescuerAgent`, triage priority key |
| `detector.py` | HSV camera pipeline and triage badge detection |
| `tracker.py` | Multi-frame tracker, association gate, majority vote, lock |
| `arena.py` | Arena, obstacles, hazard zones, survivor placement |
| `robot.py` | Differential-drive kinematics, proximity sensors |
| `camera.py` | Synthetic camera rasteriser, noise and smoke |
| `astar.py`, `apf.py` | Global planner and reactive avoidance |
| `coverage.py` | Area-scanned tracker and frontier targeting |
| `sound.py` | Sound manager (pygame mixer) |
| `run.py` | CLI: benchmark, demo recorder |
| `make_demo.py` | Headless scripted demo recorder |
| `tests/` | 97 pytest tests |

## Team

Team **Bihari_Coders**: Arpit Bhardwaj, Ritik Soni, Pratyush Singh.