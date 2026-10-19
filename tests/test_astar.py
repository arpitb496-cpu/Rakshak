"""
tests/test_astar.py - Unit tests for A* global path planning and smoothing.
"""

import math
import pytest
from arena import DisasterArena
from astar import AStarPlanner


def test_astar_path_planning():
    arena = DisasterArena(width=800, height=650)
    planner = AStarPlanner(arena_width=800, arena_height=650, grid_res=20.0)

    start = (80.0, 80.0)
    goal = (680.0, 140.0)

    path = planner.plan_path(start, goal, arena)

    assert len(path) >= 2
    assert path[0] == start
    assert path[-1] == goal

    # Verify no waypoint is inside any solid obstacle
    for wp in path:
        assert not arena.is_point_colliding(wp[0], wp[1], margin=5.0)


def test_astar_avoids_hazard():
    arena = DisasterArena(width=800, height=650)
    planner = AStarPlanner(arena_width=800, arena_height=650, grid_res=20.0)

    # Route that would go right through Hazard Alpha at (270, 240)
    start = (200.0, 240.0)
    goal = (350.0, 240.0)

    path = planner.plan_path(start, goal, arena)
    assert len(path) >= 2
    # Verify waypoints do not penetrate the center of hazard zone
    for wp in path:
        dist_to_hazard = math.hypot(wp[0] - 270.0, wp[1] - 240.0)
        assert dist_to_hazard > 25.0
