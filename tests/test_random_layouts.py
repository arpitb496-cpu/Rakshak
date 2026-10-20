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
from rescuer import RescuerState


def run_single_layout(seed: int, max_frames: int = 3000, dt: float = 1.0 / 30.0, priority_mode: bool = False):
    """Simulates a single randomized layout and checks safety & performance invariants."""
    arena = DisasterArena(width=800, height=650)
    arena.randomize_layout(seed=seed)

    robot = DifferentialDriveRobot(x=90.0, y=90.0, theta=0.0)
    camera = RobotCamera(width=160, height=90, fov_deg=65.0, max_range=340.0)
    detector = VisionDetector(focal_length=camera.focal_length, cx=camera.cx, cy=camera.cy)
    tracker = VisionTracker(lock_threshold=4)
    apf = APFController()
    astar = AStarPlanner(arena_width=800, arena_height=650)
    fsm = MissionFSM(arena, robot, camera, detector, tracker, astar, apf, priority_mode=priority_mode)

    # Initial sensor scan
    robot.scan_proximity_sensors(arena)

    collisions = 0
    reached = False
    longest_idle = 0.0
    physically_secured_ids = set()
    rescues_over_85px = 0
    prev_rescuer_states = {r.name: r.state for r in arena.rescuers}

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

        # Monitor physical security at rescuer extraction (standoff)
        for r in arena.rescuers:
            old_st = prev_rescuer_states.get(r.name)
            new_st = r.state
            if old_st != RescuerState.EXTRACTING and new_st == RescuerState.EXTRACTING:
                survs = arena.survivors if (hasattr(arena, "survivors") and arena.survivors) else [arena.survivor]
                close_s = [s for s in survs if math.hypot(s.x - r.x, s.y - r.y) <= 85.0]
                if close_s:
                    for cs in close_s:
                        physically_secured_ids.add(cs.id)
                else:
                    rescues_over_85px += 1
            prev_rescuer_states[r.name] = new_st

        # Invariant check: never stationary > 3.0s outside MISSION_COMPLETE
        if fsm.idle_timer > longest_idle:
            longest_idle = fsm.idle_timer

        assert fsm.idle_timer < 3.0, (
            f"Layout {seed}: robot was stationary for {fsm.idle_timer:.2f}s at step {frame} "
            f"in state {fsm.current_state.name}"
        )

        # Check reach condition
        all_done = (
            hasattr(arena, "survivors")
            and arena.survivors
            and all(s.is_secured for s in arena.survivors)
        )
        if priority_mode:
            if fsm.current_state == MissionState.MISSION_COMPLETE or all_done:
                reached = True
                break
        else:
            if fsm.current_state in (MissionState.RESCUE_OPERATION, MissionState.MISSION_COMPLETE):
                reached = True
                break

    # Accuracy at lock time and rescue time
    lock_err = tracker.error_at_lock
    if reached and tracker.error_at_rescue is None and not priority_mode:
        tracker.record_rescue_error((arena.survivor.x, arena.survivor.y))
    rescue_err = tracker.error_at_rescue

    rescue_errors_dict = dict(getattr(tracker, "track_rescue_errors", {}))
    if hasattr(arena, "survivors"):
        for s in arena.survivors:
            if s.is_secured and s.id in tracker.survivor_tracks and s.id not in rescue_errors_dict:
                trk = tracker.survivor_tracks[s.id]
                rescue_errors_dict[s.id] = float(math.hypot(trk.est_world_x - s.x, trk.est_world_y - s.y))
    if tracker.error_at_rescue is not None and not rescue_errors_dict:
        rescue_errors_dict["scout"] = tracker.error_at_rescue

    if priority_mode and rescue_err is None and rescue_errors_dict:
        rescue_err = statistics.median(rescue_errors_dict.values())

    triage_records = dict(getattr(tracker, "frozen_triage_records", {}))
    for tid, trk in tracker.survivor_tracks.items():
        if trk.triage_frozen and trk.true_triage:
            triage_records[tid] = (trk.true_triage, trk.triage_level)

    lock_errors_dict = dict(tracker.track_lock_errors)
    for tid, trk in tracker.survivor_tracks.items():
        if trk.error_at_lock is not None and tid not in lock_errors_dict:
            lock_errors_dict[tid] = trk.error_at_lock

    all_done = (
        hasattr(arena, "survivors")
        and arena.survivors
        and all(s.is_secured for s in arena.survivors)
    )
    survivors_secured = sum(1 for s in arena.survivors if s.is_secured) if hasattr(arena, "survivors") and arena.survivors else (1 if reached else 0)
    total_survivors = len(arena.survivors) if hasattr(arena, "survivors") and arena.survivors else 1
    survs_list = arena.survivors if (hasattr(arena, "survivors") and arena.survivors) else [arena.survivor]
    all_physically_secured = all(s.id in physically_secured_ids for s in survs_list)

    return {
        "seed": seed,
        "reached": (all_done or fsm.current_state == MissionState.MISSION_COMPLETE) if priority_mode else reached,
        "all_secured": (all_done or fsm.current_state == MissionState.MISSION_COMPLETE),
        "all_physically_secured": all_physically_secured,
        "rescues_over_85px": rescues_over_85px,
        "survivors_secured": survivors_secured,
        "total_survivors": total_survivors,
        "unmatched_locks": len(getattr(tracker, "unmatched_lock_tracks", set())),
        "mission_time": fsm.mission_time,
        "collisions": collisions,
        "longest_idle": longest_idle,
        "lock_err": lock_err,
        "lock_errors": lock_errors_dict,
        "rescue_err": rescue_err,
        "rescue_errors": rescue_errors_dict,
        "survivor_pos": (arena.survivor.x, arena.survivor.y),
        "triage_records": triage_records,
    }


DEFAULT_BASE_SEED = 0


def execute_benchmark(
    num_layouts: int = 20,
    max_frames: int = 3000,
    seed: Optional[int] = None,
    base_seed: Optional[int] = None,
    priority_mode: bool = False,
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
    mission_times = []

    mode_label = "PRIORITY SWEEP MODE" if priority_mode else "HEADLESS"
    print("\n" + "=" * 78)
    print(f"  RAKSHAK: {num_layouts} RANDOM LAYOUTS BENCHMARK EVALUATION ({mode_label})")
    print("=" * 78)

    from sound import get_sound_manager
    sm = get_sound_manager()
    old_suppress = getattr(sm, "suppress_logs", False)
    sm.suppress_logs = True
    try:
        for i in range(num_layouts):
            layout_seed = active_base_seed + i
            res = run_single_layout(seed=layout_seed, max_frames=max_frames, priority_mode=priority_mode)
            results.append(res)
            total_collisions += res["collisions"]
            mission_times.append(res["mission_time"])
            if res["longest_idle"] > max_idle_overall:
                max_idle_overall = res["longest_idle"]

            if "lock_errors" in res and res["lock_errors"]:
                for err in res["lock_errors"].values():
                    lock_errors.append(err)
            elif res["lock_err"] is not None:
                lock_errors.append(res["lock_err"])

            if priority_mode and "rescue_errors" in res and res["rescue_errors"]:
                for err in res["rescue_errors"].values():
                    rescue_errors.append(err)
            elif res["rescue_err"] is not None:
                rescue_errors.append(res["rescue_err"])

            if res["lock_err"] is not None:
                err_str = f"{res['lock_err']:.2f}px"
            elif "lock_errors" in res and res["lock_errors"]:
                err_str = f"{statistics.median(res['lock_errors'].values()):.2f}px"
            else:
                err_str = "N/A"
            rescue_str = f"{res['rescue_err']:.2f}px" if res["rescue_err"] is not None else "N/A"
            if priority_mode:
                status = "ALL_SECURED" if res.get("all_secured", False) else "PARTIAL"
                print(
                    f"Layout {i:2d}: {status:11s} | "
                    f"Secured: {res['survivors_secured']}/{res['total_survivors']} | "
                    f"Time: {res['mission_time']:5.1f}s | "
                    f"Collisions: {res['collisions']} | "
                    f"Lock Err: {err_str:>8s} | "
                    f"Rescue Err: {rescue_str:>8s}"
                )
            else:
                status = "REACHED" if res["reached"] else "TIMEOUT"
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
    all_secured_count = sum(1 for r in results if r.get("all_secured", False))
    all_secured_rate = all_secured_count / float(num_layouts)
    all_physically_secured_count = sum(1 for r in results if r.get("all_physically_secured", False))
    all_physically_secured_rate = all_physically_secured_count / float(num_layouts)
    total_rescues_over_85px = sum(r.get("rescues_over_85px", 0) for r in results)
    rescues_over_50px = sum(1 for e in rescue_errors if e > 50.0)
    median_mission_time = statistics.median(mission_times) if mission_times else 0.0
    median_lock_error = statistics.median(lock_errors) if lock_errors else 999.0
    mean_lock_error = statistics.mean(lock_errors) if lock_errors else 999.0
    max_lock_error = max(lock_errors) if lock_errors else 999.0
    median_rescue_error = statistics.median(rescue_errors) if rescue_errors else 999.0
    total_unmatched_locks = sum(r.get("unmatched_locks", 0) for r in results)

    print("-" * 78)
    if priority_mode:
        print(f"All secured (logical):     {all_secured_count}/{num_layouts} ({all_secured_rate * 100:.1f}%)")
        print(f"All physically secured:    {all_physically_secured_count}/{num_layouts} ({all_physically_secured_rate * 100:.1f}%)")
        print(f"Rescues > 85 px from GT:   {total_rescues_over_85px}")
        print(f"Rescues > 50 px error:     {rescues_over_50px} / {len(rescue_errors)}")
        print(f"Collision count:           {total_collisions}")
        print(f"Median mission time:       {median_mission_time:.1f} s")
        print(f"Median localization error: {median_lock_error:.2f} px (at lock time)")
        print(f"Mean localization error:   {mean_lock_error:.2f} px (at lock time)")
        print(f"Max localization error:    {max_lock_error:.2f} px (at lock time)")
        print(f"Median rescue error:       {median_rescue_error:.2f} px (at rescue standoff)")
        print(f"Unmatched lock count:      {total_unmatched_locks}")
    else:
        print(f"Reach rate:                {reached_count}/{num_layouts} ({reach_rate * 100:.1f}%)")
        print(f"Collision count:           {total_collisions}")
        print(f"Longest idle time:         {max_idle_overall:.2f}s")
        print(f"Median localization error: {median_lock_error:.2f} px (at lock time)")
        print(f"Mean localization error:   {mean_lock_error:.2f} px (at lock time)")
        print(f"Max localization error:    {max_lock_error:.2f} px (at lock time)")
        print(f"Median rescue error:       {median_rescue_error:.2f} px (at rescue standoff)")
        print(f"Unmatched lock count:      {total_unmatched_locks}")

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

    if not priority_mode:
        # b) Assert survivor is reached in at least 18 of 20 layouts
        min_required_reach = 18 if num_layouts == 20 else int(math.ceil(num_layouts * 0.9))
        assert reached_count >= min_required_reach, f"Expected >= {min_required_reach} reached, got {reached_count}/{num_layouts}"

        # c) Assert robot is never stationary for more than 3 seconds outside MISSION_COMPLETE
        assert max_idle_overall < 3.0, f"Expected longest idle < 3.0s, got {max_idle_overall:.2f}s"

        # d) Assert median localization error at lock time is under 30 px
        assert median_lock_error < 30.0, f"Expected median lock error < 30.0 px, got {median_lock_error:.2f} px"

    return {
        "reached_count": all_secured_count if priority_mode else reached_count,
        "all_secured_count": all_secured_count,
        "total_layouts": num_layouts,
        "reach_rate": all_secured_rate if priority_mode else reach_rate,
        "all_secured_rate": all_secured_rate,
        "collision_count": total_collisions,
        "median_mission_time": median_mission_time,
        "longest_idle": max_idle_overall,
        "median_lock_error": median_lock_error,
        "mean_lock_error": mean_lock_error,
        "max_lock_error": max_lock_error,
        "median_rescue_error": median_rescue_error,
        "unmatched_lock_count": total_unmatched_locks,
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


def test_priority_mode_benchmark():
    """
    Verifies priority sweep mode benchmark execution across layouts.
    """
    summary = execute_benchmark(num_layouts=2, max_frames=1200, priority_mode=True)
    assert "all_secured_rate" in summary
    assert "median_mission_time" in summary
    assert summary["collision_count"] == 0
    assert 0.0 <= summary["triage_accuracy"] <= 100.0
