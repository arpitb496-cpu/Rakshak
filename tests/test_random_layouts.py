from typing import Optional
"""
tests/test_random_layouts.py - Headless 20-layout stress test for Rakshak.

Verifies:
a) 20 random layouts x 3000 frames: robot center never overlaps any obstacle, hazard, rescuer, or wall.
b) Survivor is reached in at least 18 of 20 layouts.
c) Robot is never stationary for more than 3 seconds outside MISSION_COMPLETE.
d) Median localization error at lock time is under 30 px.

Prints:
- Reach rate
- Collision count
- Longest idle time
- Median localization error
"""

import os
import sys

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

os.environ["SDL_VIDEODRIVER"] = "dummy"

import math
import statistics
import pytest
from arena import DisasterArena
from robot import DifferentialDriveRobot
from camera import RobotCamera
from detector import VisionDetector
from tracker import VisionTracker
from apf import APFController
from astar import AStarPlanner
from fsm import MissionFSM, MissionState


def run_single_layout(seed: int, max_frames: int = 3000, dt: float = 1.0 / 30.0):
    """Simulates a single randomized layout and checks safety & performance invariants."""
    arena = DisasterArena(width=800, height=650)
    arena.randomize_layout(seed=seed)

    robot = DifferentialDriveRobot(x=90.0, y=90.0, theta=0.0)
    camera = RobotCamera(width=160, height=90, fov_deg=65.0, max_range=340.0)
    detector = VisionDetector(focal_length=camera.focal_length, cx=camera.cx, cy=camera.cy)
    tracker = VisionTracker(lock_threshold=4)
    apf = APFController()
    astar = AStarPlanner(arena_width=800, arena_height=650)
    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf)

    # Initial sensor scan
    robot.scan_proximity_sensors(arena)

    collisions = 0
    reached = False
    longest_idle = 0.0

    for frame in range(max_frames):
        # 1. Collision check: robot center must NEVER overlap obstacle, hazard, rescuer, or wall
        # Center inside obstacle
        if arena.is_point_colliding(robot.x, robot.y, margin=0.0):
            collisions += 1

        # Center inside hazard zone
        if arena.is_inside_hazard(robot.x, robot.y, margin=0.0):
            collisions += 1

        # Center inside boundary wall
        if robot.x < 16.0 or robot.x > (arena.width - 16.0) or robot.y < 16.0 or robot.y > (arena.height - 16.0):
            collisions += 1

        # Center inside friendly rescuer
        for rescuer in arena.rescuers:
            if math.hypot(robot.x - rescuer.x, robot.y - rescuer.y) <= rescuer.radius:
                collisions += 1

        # Simulation updates
        arena.update(dt)
        fsm.update(dt)
        robot.update_kinematics(dt, arena)
        robot.scan_proximity_sensors(arena)

        # Invariant check: never stationary > 3.0s outside MISSION_COMPLETE
        if fsm.idle_timer > longest_idle:
            longest_idle = fsm.idle_timer

        assert fsm.idle_timer < 3.0, (
            f"Layout {seed}: robot was stationary for {fsm.idle_timer:.2f}s at step {frame} "
            f"in state {fsm.current_state.name}"
        )

        # Check reach condition
        if fsm.current_state in (MissionState.RESCUE_OPERATION, MissionState.MISSION_COMPLETE):
            reached = True
            break

    # Accuracy at lock time and rescue time
    lock_err = tracker.error_at_lock
    if reached and tracker.error_at_rescue is None:
        tracker.record_rescue_error((arena.survivor.x, arena.survivor.y))
    rescue_err = tracker.error_at_rescue

    triage_records = dict(getattr(tracker, "frozen_triage_records", {}))
    for tid, trk in tracker.survivor_tracks.items():
        if trk.triage_frozen and trk.true_triage:
            triage_records[tid] = (trk.true_triage, trk.triage_level)

    return {
        "seed": seed,
        "reached": reached,
        "collisions": collisions,
        "longest_idle": longest_idle,
        "lock_err": lock_err,
        "rescue_err": rescue_err,
        "survivor_pos": (arena.survivor.x, arena.survivor.y),
        "triage_records": triage_records,
    }


DEFAULT_BASE_SEED = 0


def execute_benchmark(
    num_layouts: int = 20,
    max_frames: int = 3000,
    seed: Optional[int] = None,
    base_seed: Optional[int] = None,
):
    """
    Executes headless benchmark across multiple randomized disaster layouts.
    Validates safety, reachability, responsiveness, and localization accuracy.
    Layout i uses seed (base_seed + i).
    """
    active_base_seed = seed if seed is not None else (base_seed if base_seed is not None else DEFAULT_BASE_SEED)
    results = []
    total_collisions = 0
    max_idle_overall = 0.0
    lock_errors = []
    rescue_errors = []

    print("\n" + "=" * 78)
    print(f"  RAKSHAK: {num_layouts} RANDOM LAYOUTS BENCHMARK EVALUATION (HEADLESS)")
    print("=" * 78)

    from sound import get_sound_manager
    sm = get_sound_manager()
    old_suppress = getattr(sm, "suppress_logs", False)
    sm.suppress_logs = True
    try:
        for i in range(num_layouts):
            layout_seed = active_base_seed + i
            res = run_single_layout(seed=layout_seed, max_frames=max_frames)
            results.append(res)
            total_collisions += res["collisions"]
            if res["longest_idle"] > max_idle_overall:
                max_idle_overall = res["longest_idle"]

            if res["lock_err"] is not None:
                lock_errors.append(res["lock_err"])
            if res["rescue_err"] is not None:
                rescue_errors.append(res["rescue_err"])

            status = "REACHED" if res["reached"] else "TIMEOUT"
            err_str = f"{res['lock_err']:.2f}px" if res["lock_err"] is not None else "N/A"
            rescue_str = f"{res['rescue_err']:.2f}px" if res["rescue_err"] is not None else "N/A"
            print(
                f"Layout {i:2d}: {status:7s} | "
                f"Collisions: {res['collisions']} | "
                f"Max Idle: {res['longest_idle']:.2f}s | "
                f"Lock Err: {err_str:>8s} | "
                f"Rescue Err: {rescue_str:>8s}"
            )
    finally:
        sm.suppress_logs = old_suppress

    # Aggregates
    reached_count = sum(1 for r in results if r["reached"])
    reach_rate = reached_count / float(num_layouts)
    median_lock_error = statistics.median(lock_errors) if lock_errors else 999.0
    median_rescue_error = statistics.median(rescue_errors) if rescue_errors else 999.0

    print("-" * 78)
    print(f"Reach rate:                {reached_count}/{num_layouts} ({reach_rate * 100:.1f}%)")
    print(f"Collision count:           {total_collisions}")
    print(f"Longest idle time:         {max_idle_overall:.2f}s")
    print(f"Median localization error: {median_lock_error:.2f} px (at lock time)")
    print(f"Median rescue error:       {median_rescue_error:.2f} px (at rescue standoff)")

    # Aggregate triage metrics
    all_triage_records = []
    for res in results:
        for tid, pair in res.get("triage_records", {}).items():
            all_triage_records.append(pair)

    labels = ["CRITICAL", "INJURED", "STABLE"]
    cm = {r: {c: 0 for c in labels} for r in labels}
    correct_triage = 0
    total_triage = 0
    for gt, perc in all_triage_records:
        if gt in labels and perc in labels:
            cm[gt][perc] += 1
            total_triage += 1
            if gt == perc:
                correct_triage += 1

    triage_acc_pct = round(100.0 * correct_triage / total_triage, 1) if total_triage > 0 else 100.0
    print(f"Triage accuracy:           {triage_acc_pct:.1f}% ({correct_triage}/{total_triage})")
    print("\nConfusion Matrix (rows = true, columns = perceived):")
    header = "            " + "  ".join(f"{c:>8s}" for c in labels)
    print(header)
    for r in labels:
        row_str = f"  {r:8s}  " + "  ".join(f"{cm[r][c]:>8d}" for c in labels)
        print(row_str)
    print("=" * 78 + "\n")

    # Required assertions
    # a) Assert robot center never overlaps any obstacle, hazard, rescuer, or wall
    assert total_collisions == 0, f"Expected 0 collisions across {num_layouts} layouts, got {total_collisions}"

    # b) Assert survivor is reached in at least 18 of 20 layouts
    min_required_reach = 18 if num_layouts == 20 else int(math.ceil(num_layouts * 0.9))
    assert reached_count >= min_required_reach, f"Expected >= {min_required_reach} reached, got {reached_count}/{num_layouts}"

    # c) Assert robot is never stationary for more than 3 seconds outside MISSION_COMPLETE
    assert max_idle_overall < 3.0, f"Expected longest idle < 3.0s, got {max_idle_overall:.2f}s"

    # d) Assert median localization error at lock time is under 30 px
    assert median_lock_error < 30.0, f"Expected median lock error < 30.0 px, got {median_lock_error:.2f} px"

    return {
        "reached_count": reached_count,
        "total_layouts": num_layouts,
        "reach_rate": reach_rate,
        "collision_count": total_collisions,
        "longest_idle": max_idle_overall,
        "median_lock_error": median_lock_error,
        "median_rescue_error": median_rescue_error,
        "triage_accuracy": triage_acc_pct,
        "triage_total": total_triage,
        "triage_correct": correct_triage,
        "confusion_matrix": cm,
        "results": results,
    }


def test_20_random_layouts_benchmark():
    execute_benchmark(num_layouts=20, max_frames=3000)


if __name__ == "__main__":
    execute_benchmark(num_layouts=20, max_frames=3000)



def test_benchmark_triage_metrics_and_confusion_matrix(capsys):
    """
    Verifies Task 2:
    - execute_benchmark computes triage accuracy % and 3x3 confusion matrix.
    - [AUDIO] logs are quieted during benchmark execution.
    """
    summary = execute_benchmark(num_layouts=2, max_frames=1500)
    assert "triage_accuracy" in summary
    assert "confusion_matrix" in summary
    assert isinstance(summary["triage_accuracy"], float)
    assert 0.0 <= summary["triage_accuracy"] <= 100.0
    cm = summary["confusion_matrix"]
    for lbl in ["CRITICAL", "INJURED", "STABLE"]:
        assert lbl in cm
        for col in ["CRITICAL", "INJURED", "STABLE"]:
            assert isinstance(cm[lbl][col], int)
            assert cm[lbl][col] >= 0

    captured = capsys.readouterr().out
    assert "[AUDIO]" not in captured
