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

    return {
        "seed": seed,
        "reached": reached,
        "collisions": collisions,
        "longest_idle": longest_idle,
        "lock_err": lock_err,
        "rescue_err": rescue_err,
        "survivor_pos": (arena.survivor.x, arena.survivor.y),
    }


def execute_benchmark(num_layouts: int = 20, max_frames: int = 3000):
    """
    Executes headless benchmark across multiple randomized disaster layouts.
    Validates safety, reachability, responsiveness, and localization accuracy.
    """
    results = []
    total_collisions = 0
    max_idle_overall = 0.0
    lock_errors = []
    rescue_errors = []

    print("\n" + "=" * 78)
    print(f"  RAKSHAK: {num_layouts} RANDOM LAYOUTS BENCHMARK EVALUATION (HEADLESS)")
    print("=" * 78)

    for seed in range(num_layouts):
        res = run_single_layout(seed=seed, max_frames=max_frames)
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
            f"Layout {seed:2d}: {status:7s} | "
            f"Collisions: {res['collisions']} | "
            f"Max Idle: {res['longest_idle']:.2f}s | "
            f"Lock Err: {err_str:>8s} | "
            f"Rescue Err: {rescue_str:>8s}"
        )

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
        "results": results,
    }


def test_20_random_layouts_benchmark():
    execute_benchmark(num_layouts=20, max_frames=3000)


if __name__ == "__main__":
    execute_benchmark(num_layouts=20, max_frames=3000)

