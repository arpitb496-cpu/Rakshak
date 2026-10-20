"""
coverage.py - Arena Coverage Grid and Frontier Exploration for Rakshak.

Features:
- Discretized 2D grid tracking areas swept by camera FOV frustum
- Fast line-of-sight cell marking with obstacle occlusion
- Real-time calculation of scanned area percentage
- Frontier exploration waypoint recommendation for unseen areas
- Semi-transparent overlay rendering for tactical arena display
"""

import math
from typing import Tuple, List, Optional
import numpy as np
import pygame


class CoverageGrid:
    """
    Tracks visual coverage of the disaster arena.
    """

    def __init__(self, arena_width: int = 800, arena_height: int = 650, cell_size: float = 20.0, arena=None):
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.cell_size = cell_size

        self.cols = int(math.ceil(arena_width / cell_size))
        self.rows = int(math.ceil(arena_height / cell_size))

        self.grid = np.zeros((self.rows, self.cols), dtype=bool)
        self.accessible_mask = np.ones((self.rows, self.cols), dtype=bool)

        if arena is not None:
            self._init_accessible_mask(arena)
        else:
            self.total_accessible = self.rows * self.cols

    def _init_accessible_mask(self, arena):
        """Identifies cells that are open corridors and not inside solid obstacles or outer walls."""
        for r in range(self.rows):
            for c in range(self.cols):
                cx = (c + 0.5) * self.cell_size
                cy = (r + 0.5) * self.cell_size
                # Perimeter walls check
                if cx < 20.0 or cx > (self.arena_width - 20.0) or cy < 20.0 or cy > (self.arena_height - 20.0):
                    self.accessible_mask[r, c] = False
                elif arena.is_point_colliding(cx, cy, margin=6.0):
                    self.accessible_mask[r, c] = False
                else:
                    self.accessible_mask[r, c] = True

        self.total_accessible = int(np.sum(self.accessible_mask))
        if self.total_accessible == 0:
            self.total_accessible = 1

    def update(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        fov_rad: float,
        max_range: float,
        arena,
    ):
        """
        Marks all accessible cells inside the camera frustum as scanned.
        Includes fast distance and angle filtering plus line-of-sight ray checks.
        """
        half_fov = fov_rad / 2.0
        r_sq = max_range * max_range

        # Bounding box of FOV in grid indices
        min_c = max(0, int((robot_x - max_range) / self.cell_size))
        max_c = min(self.cols - 1, int((robot_x + max_range) / self.cell_size))
        min_r = max(0, int((robot_y - max_range) / self.cell_size))
        max_r = min(self.rows - 1, int((robot_y + max_range) / self.cell_size))

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if not self.accessible_mask[r, c] or self.grid[r, c]:
                    continue

                cx = (c + 0.5) * self.cell_size
                cy = (r + 0.5) * self.cell_size

                dx = cx - robot_x
                dy = cy - robot_y
                dist_sq = dx * dx + dy * dy

                if dist_sq > r_sq or dist_sq < 100.0:
                    continue

                angle_to_cell = math.atan2(dy, dx)
                ang_diff = (angle_to_cell - robot_theta + math.pi) % (2.0 * math.pi) - math.pi

                if abs(ang_diff) <= half_fov:
                    # Line of sight check (midpoint sample)
                    mx = robot_x + 0.5 * dx
                    my = robot_y + 0.5 * dy
                    if not arena.is_point_colliding(mx, my, margin=4.0):
                        self.grid[r, c] = True

    @property
    def percent_scanned(self) -> float:
        """Returns the percentage of accessible arena area scanned [0.0, 100.0]."""
        scanned_count = int(np.sum(self.grid & self.accessible_mask))
        return (scanned_count / float(self.total_accessible)) * 100.0

    def get_largest_unscanned_checkpoint(self, robot_x: float, robot_y: float, arena=None) -> Optional[Tuple[float, float]]:
        """
        Finds the centroid of the largest cluster of unscanned accessible cells.
        Uses 4x4 cell blocks to identify the highest density unscanned region.
        """
        unseen = (~self.grid) & self.accessible_mask
        if not np.any(unseen):
            return None

        block_h, block_w = 4, 4
        best_count = 0
        best_centroid = None

        for r in range(0, self.rows, 2):
            for c in range(0, self.cols, 2):
                sub = unseen[r:r+block_h, c:c+block_w]
                cnt = int(np.sum(sub))
                if cnt > best_count:
                    best_count = cnt
                    sub_unseen = np.argwhere(sub)
                    mr = r + float(np.mean(sub_unseen[:, 0]))
                    mc = c + float(np.mean(sub_unseen[:, 1]))
                    wx = (mc + 0.5) * self.cell_size
                    wy = (mr + 0.5) * self.cell_size
                    if arena is not None and arena.is_point_colliding(wx, wy, margin=14.0):
                        best_d = float("inf")
                        best_cell = None
                        for ur, uc in sub_unseen:
                            cr, cc = r + ur, c + uc
                            c_wx = (cc + 0.5) * self.cell_size
                            c_wy = (cr + 0.5) * self.cell_size
                            if not arena.is_point_colliding(c_wx, c_wy, margin=8.0):
                                d = math.hypot(c_wx - wx, c_wy - wy)
                                if d < best_d:
                                    best_d = d
                                    best_cell = (c_wx, c_wy)
                        if best_cell:
                            wx, wy = best_cell
                    best_centroid = (wx, wy)
        return best_centroid

    def get_unseen_frontier_checkpoint(self, robot_x: float, robot_y: float) -> Optional[Tuple[float, float]]:
        """
        Finds an accessible unseen cell that balances proximity with high unseen density.
        """
        unseen_indices = np.argwhere((~self.grid) & self.accessible_mask)
        if len(unseen_indices) == 0:
            return None

        # Sample candidate clusters
        step = max(1, len(unseen_indices) // 16)
        candidates = unseen_indices[::step]

        best_pt = None
        best_score = float("-inf")

        for r, c in candidates:
            wx = (c + 0.5) * self.cell_size
            wy = (r + 0.5) * self.cell_size
            dist = math.hypot(wx - robot_x, wy - robot_y)

            # Local unseen cluster density (3x3 neighborhood)
            r0 = max(0, r - 2)
            r1 = min(self.rows, r + 3)
            c0 = max(0, c - 2)
            c1 = min(self.cols, c + 3)
            sub = (~self.grid[r0:r1, c0:c1]) & self.accessible_mask[r0:r1, c0:c1]
            density = np.sum(sub)

            # Score: high density rewarded, far distance penalized
            score = (density * 15.0) - (dist * 0.1)
            if score > best_score:
                best_score = score
                best_pt = (wx, wy)

        return best_pt

    def render_overlay(self, screen: pygame.Surface):
        """Draws semi-transparent green/cyan grid tiles for scanned sectors."""
        overlay_surf = pygame.Surface((self.arena_width, self.arena_height), pygame.SRCALPHA)
        color = (40, 220, 160, 42)
        border_col = (50, 240, 180, 75)

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] and self.accessible_mask[r, c]:
                    x = int(c * self.cell_size)
                    y = int(r * self.cell_size)
                    w = int(self.cell_size)
                    pygame.draw.rect(overlay_surf, color, (x, y, w, w))
                    pygame.draw.rect(overlay_surf, border_col, (x, y, w, w), 1)

        screen.blit(overlay_surf, (0, 0))

    def reset(self, arena=None):
        """Resets scanned grid state."""
        self.grid.fill(False)
        if arena is not None:
            self._init_accessible_mask(arena)
