"""
astar.py - Global A* Grid Path Planner with Obstacle Inflation and Smoothing for Rakshak.

Features:
- Discretized 2D grid representation of the disaster arena
- Obstacle inflation matching robot radius and safety clearance margin
- Hazard zones treated as high-cost / forbidden zones
- Rescuer safety zones excluded from route planning
- 8-connected grid search with corner-cutting prevention
- Line-of-sight path shortcutting for smooth natural waypoint navigation
"""

import math
import heapq
from typing import List, Tuple, Optional, Set
import numpy as np


class AStarPlanner:
    """
    Deliberative grid-based A* path planner.
    """

    def __init__(
        self,
        arena_width: int = 800,
        arena_height: int = 650,
        grid_res: float = 20.0,
        robot_radius: float = 16.0,
        safety_margin: float = 5.0,
    ):
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.res = grid_res
        self.inflation_radius = robot_radius + safety_margin

        self.cols = int(math.ceil(arena_width / grid_res))
        self.rows = int(math.ceil(arena_height / grid_res))

        # Static occupancy grid (0: free, 1: solid obstacle, 2: hazard zone)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Converts continuous world coordinates (x, y) to discrete grid indices (col, row)."""
        c = int(np.clip(wx / self.res, 0, self.cols - 1))
        r = int(np.clip(wy / self.res, 0, self.rows - 1))
        return c, r

    def grid_to_world(self, c: int, r: int) -> Tuple[float, float]:
        """Converts discrete grid indices (col, row) to world cell-center coordinates (x, y)."""
        wx = (c + 0.5) * self.res
        wy = (r + 0.5) * self.res
        return wx, wy

    def build_grid(self, arena, ignore_rescuer_id: Optional[int] = None):
        """
        Rasterizes arena obstacles, hazard zones, and rescuer safety zones into grid with inflation.
        """
        self.grid.fill(0)

        # 1. Rasterize solid obstacles with inflation
        for r in range(self.rows):
            for c in range(self.cols):
                wx, wy = self.grid_to_world(c, r)

                # Check collision against obstacles with safety inflation
                if arena.is_point_colliding(wx, wy, margin=self.inflation_radius):
                    self.grid[r, c] = 1
                    continue

                # Check if inside hazard zone (forbidden for planner)
                if arena.is_inside_hazard(wx, wy, margin=15.0):
                    self.grid[r, c] = 2
                    continue

                # Check rescuer safe clearance zones
                for rescuer in arena.rescuers:
                    if ignore_rescuer_id is not None and getattr(rescuer, "id", None) == ignore_rescuer_id:
                        continue
                    dist = math.hypot(wx - rescuer.x, wy - rescuer.y)
                    if dist <= (rescuer.safe_buffer + self.inflation_radius):
                        self.grid[r, c] = 1
                        break

    def plan_path(
        self,
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        arena,
        ignore_rescuer_id: Optional[int] = None,
    ) -> List[Tuple[float, float]]:
        """
        Computes an obstacle-free, smoothed waypoint path from start to goal.
        """
        # Re-rasterize grid to capture dynamic entities
        self.build_grid(arena, ignore_rescuer_id=ignore_rescuer_id)

        start_c, start_r = self.world_to_grid(start_pos[0], start_pos[1])
        goal_c, goal_r = self.world_to_grid(goal_pos[0], goal_pos[1])

        # If start is inside an inflated obstacle, unblock start cell
        self.grid[start_r, start_c] = 0

        # If goal is inside an obstacle (e.g. survivor inside rubble), find closest free neighbor
        if self.grid[goal_r, goal_c] != 0:
            goal_c, goal_r = self._find_nearest_free_cell(goal_c, goal_r)

        # Priority Queue for A*: (f_cost, g_cost, (c, r))
        open_set = []
        h_start = math.hypot(goal_c - start_c, goal_r - start_r)
        heapq.heappush(open_set, (h_start, 0.0, (start_c, start_r)))

        came_from = {}
        g_score = {(start_c, start_r): 0.0}
        visited: Set[Tuple[int, int]] = set()

        # 8 directions: (dc, dr, move_cost)
        neighbors = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, 1.4142),
            (1, -1, 1.4142),
            (-1, 1, 1.4142),
            (-1, -1, 1.4142),
        ]

        found = False

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == (goal_c, goal_r):
                found = True
                break

            cc, cr = current

            for dc, dr, cost in neighbors:
                nc, nr = cc + dc, cr + dr

                # Boundary check
                if nc < 0 or nc >= self.cols or nr < 0 or nr >= self.rows:
                    continue

                # Obstacle or hazard check
                if self.grid[nr, nc] != 0:
                    continue

                # Prevent diagonal corner cutting
                if dc != 0 and dr != 0:
                    if self.grid[cr, nc] != 0 or self.grid[nr, cc] != 0:
                        continue

                tentative_g = current_g + cost

                if (nc, nr) not in g_score or tentative_g < g_score[(nc, nr)]:
                    g_score[(nc, nr)] = tentative_g
                    h = math.hypot(goal_c - nc, goal_r - nr)
                    f = tentative_g + h * 1.05  # Slight heuristic tie-breaker
                    came_from[(nc, nr)] = current
                    heapq.heappush(open_set, (f, tentative_g, (nc, nr)))

        if not found:
            # Fallback direct path if no grid path found
            return [start_pos, goal_pos]

        # Reconstruct path
        path_cells = [(goal_c, goal_r)]
        curr = (goal_c, goal_r)
        while curr in came_from:
            curr = came_from[curr]
            path_cells.append(curr)
        path_cells.reverse()

        # Convert to world coordinates
        world_waypoints = [self.grid_to_world(c, r) for c, r in path_cells]
        # Replace first with exact start and last with exact goal
        world_waypoints[0] = start_pos
        world_waypoints[-1] = goal_pos

        # Apply path shortcutting (smoothing)
        smoothed = self._smooth_path(world_waypoints, arena)
        return smoothed

    def _find_nearest_free_cell(self, tc: int, tr: int) -> Tuple[int, int]:
        """Finds closest non-colliding cell if target is blocked."""
        for radius in range(1, max(self.cols, self.rows)):
            for dc in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    nc, nr = tc + dc, tr + dr
                    if 0 <= nc < self.cols and 0 <= nr < self.rows:
                        if self.grid[nr, nc] == 0:
                            return nc, nr
        return tc, tr

    def _line_of_sight(self, p1: Tuple[float, float], p2: Tuple[float, float], arena) -> bool:
        """Tests if direct line between two points has clear line of sight (no obstacles)."""
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(2, int(dist / (self.res * 0.5)))
        for i in range(1, steps):
            frac = i / steps
            px = p1[0] + frac * (p2[0] - p1[0])
            py = p1[1] + frac * (p2[1] - p1[1])
            c, r = self.world_to_grid(px, py)
            if self.grid[r, c] != 0:
                return False
        return True

    def _smooth_path(
        self, waypoints: List[Tuple[float, float]], arena
    ) -> List[Tuple[float, float]]:
        """Removes redundant intermediate waypoints via line-of-sight checks."""
        if len(waypoints) <= 2:
            return waypoints

        smoothed = [waypoints[0]]
        current_idx = 0

        while current_idx < len(waypoints) - 1:
            # Look ahead as far as possible
            next_idx = len(waypoints) - 1
            while next_idx > current_idx + 1:
                if self._line_of_sight(waypoints[current_idx], waypoints[next_idx], arena):
                    break
                next_idx -= 1
            smoothed.append(waypoints[next_idx])
            current_idx = next_idx

        return smoothed
