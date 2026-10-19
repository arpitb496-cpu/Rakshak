"""
camera.py - Forward-Facing Robot Camera Simulation for Rakshak.

Renders a first-person perspective view (internal resolution 160x90, scaled up)
from the robot's current pose and heading.
Utilizes vectorized NumPy operations for ray-frustum projection and depth buffering,
avoiding slow per-pixel Python loops.

Output:
- BGR image buffer (shape: [90, 160, 3], dtype: uint8)
- Depth buffer (shape: [160], dtype: float64)
"""

import math
from typing import Tuple, List, Optional
import numpy as np


class RobotCamera:
    """
    Simulated pinhole camera mounted forward on the robot.
    """

    def __init__(
        self,
        width: int = 160,
        height: int = 90,
        fov_deg: float = 65.0,
        max_range: float = 340.0,
    ):
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.fov_rad = math.radians(fov_deg)
        self.max_range = max_range

        # Intrinsic parameters
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        # Focal length: f = (W / 2) / tan(FOV / 2)
        self.focal_length = self.cx / math.tan(self.fov_rad / 2.0)

        # Precompute ray angle offsets for all 160 columns (vectorized)
        col_indices = np.arange(self.width, dtype=np.float64)
        # tan(phi) = (u - cx) / f  ==> phi = arctan((u - cx) / f)
        self.column_angles = np.arctan((col_indices - self.cx) / self.focal_length)
        # Cosine correction to eliminate fisheye/barrel distortion in planar projections
        self.cos_corrections = np.cos(self.column_angles)

        # Pre-allocated BGR background frame
        self._bg_template = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._build_background_template()

        # Depth buffer for column occlusion
        self.depth_buffer = np.full(self.width, self.max_range, dtype=np.float64)

        # Most recent rendered frame
        self.last_frame = np.copy(self._bg_template)

        # Phase 4: Camera Sensor Noise & Smoke Haze
        self.noise_enabled: bool = False
        self.smoke_time: float = 0.0

    def _build_background_template(self):
        """Builds realistic disaster environment backdrop (slate sky + dusty floor gradient)."""
        horizon = int(self.cy)

        # Sky / smoky atmosphere (dark slate blue/gray)
        for y in range(horizon):
            ratio = y / horizon
            b = int(35 + 15 * ratio)
            g = int(30 + 12 * ratio)
            r = int(32 + 10 * ratio)
            self._bg_template[y, :] = [b, g, r]

        # Floor (concrete rubble floor with perspective gradient)
        for y in range(horizon, self.height):
            ratio = (y - horizon) / (self.height - horizon)
            b = int(45 + 35 * ratio)
            g = int(42 + 32 * ratio)
            r = int(40 + 30 * ratio)
            self._bg_template[y, :] = [b, g, r]

        # Subtle horizon haze line
        self._bg_template[horizon - 1: horizon + 1, :] = [50, 45, 45]

    def render(self, robot, arena) -> np.ndarray:
        """
        Renders first-person 160x90 camera frame using vectorized NumPy operations.
        Includes walls, rubble, rescuers, and the target survivor.
        """
        frame = np.copy(self._bg_template)
        self.depth_buffer.fill(self.max_range)

        rx, ry = robot.x, robot.y
        r_theta = robot.theta

        # 1. Vectorized Raycasting for 160 columns against arena segments (Walls & Rubble)
        # Absolute angle for each column ray
        abs_angles = r_theta + self.column_angles
        ray_dx = np.cos(abs_angles)  # shape (160,)
        ray_dy = np.sin(abs_angles)  # shape (160,)

        # Gather all obstacle segments
        segments = arena.get_all_segments()
        if segments:
            # Segments array: shape (N, 4) -> x1, y1, x2, y2
            seg_arr = np.array([[s[0][0], s[0][1], s[1][0], s[1][1]] for s in segments], dtype=np.float64)
            x1 = seg_arr[:, 0]  # (N,)
            y1 = seg_arr[:, 1]
            x2 = seg_arr[:, 2]
            y2 = seg_arr[:, 3]

            sx = x2 - x1  # (N,)
            sy = y2 - y1

            # Broadcast across 160 rays (shape: 160, N)
            dx = ray_dx[:, np.newaxis]
            dy = ray_dy[:, np.newaxis]

            denom = dx * sy - dy * sx
            valid_denom = np.abs(denom) > 1e-9

            qx = x1 - rx
            qy = y1 - ry

            # Solve for t and u
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(valid_denom, (qx * sy - qy * sx) / denom, np.inf)
                u = np.where(valid_denom, (qx * dy - qy * dx) / denom, -1.0)

            # Mask valid hits
            valid_hits = (t > 1.0) & (t < self.max_range) & (u >= 0.0) & (u <= 1.0)
            t_masked = np.where(valid_hits, t, np.inf)

            # Find closest hit per column
            min_col_t = np.min(t_masked, axis=1)  # shape (160,)
            hit_seg_idx = np.argmin(t_masked, axis=1)

            # Render wall/rubble columns
            wall_height_world = 38.0
            for col in range(self.width):
                dist = min_col_t[col]
                if dist < self.max_range:
                    # Perpendicular distance correction to remove fisheye
                    perp_dist = dist * self.cos_corrections[col]
                    self.depth_buffer[col] = perp_dist

                    # Projected height
                    proj_h = int((wall_height_world * self.focal_length) / max(perp_dist, 5.0))
                    v_top = max(0, int(self.cy - proj_h // 2))
                    v_bot = min(self.height, int(self.cy + proj_h // 2))

                    # Shading based on distance
                    shade = max(0.2, min(1.0, 1.0 - (perp_dist / self.max_range) * 0.75))
                    # Distinguish rubble from outer boundary walls
                    hit_seg = segments[hit_seg_idx[col]]
                    base_color = np.array([90, 85, 80], dtype=np.float64)
                    col_bgr = (base_color * shade).astype(np.uint8)

                    frame[v_top:v_bot, col] = col_bgr

        # 2. Render Hazard Zones (translucent red ground glow)
        for hz in arena.hazard_zones:
            dist = math.hypot(hz.x - rx, hz.y - ry)
            if dist < self.max_range:
                angle_to = math.atan2(hz.y - ry, hz.x - rx)
                rel_angle = robot.normalize_angle(angle_to - r_theta)
                if abs(rel_angle) < (self.fov_rad / 2.0 + 0.3):
                    u_center = int(self.cx + self.focal_length * math.tan(rel_angle))
                    proj_w = int((hz.radius * 2.0 * self.focal_length) / max(dist, 10.0))
                    u1 = max(0, u_center - proj_w // 2)
                    u2 = min(self.width, u_center + proj_w // 2)
                    if u2 > u1:
                        # Tint floor red where hazard is closer than walls
                        hazard_cols = (self.depth_buffer[u1:u2] > dist)
                        if np.any(hazard_cols):
                            slice_cols = np.arange(u1, u2)[hazard_cols]
                            v_start = int(self.cy + 10)
                            v_end = min(self.height, int(self.cy + 30))
                            # Add red glow
                            frame[v_start:v_end, slice_cols, 2] = np.clip(
                                frame[v_start:v_end, slice_cols, 2].astype(int) + 70, 0, 255
                            ).astype(np.uint8)

        # 3. Render Rescuers (Blue uniforms with reflective strips)
        # Collect dynamic entities to sort by depth (farthest first)
        entities = []
        for rescuer in arena.rescuers:
            dist = math.hypot(rescuer.x - rx, rescuer.y - ry)
            if dist < self.max_range:
                angle_to = math.atan2(rescuer.y - ry, rescuer.x - rx)
                rel_angle = robot.normalize_angle(angle_to - r_theta)
                if abs(rel_angle) < (self.fov_rad / 2.0 + 0.25):
                    entities.append(("rescuer", dist, rel_angle, rescuer))

        # 4. Render Survivors (High-visibility Rescue Orange with triage tag & pulsating beacon)
        survivor_list = arena.survivors if hasattr(arena, "survivors") and arena.survivors else [arena.survivor]
        for s_obj in survivor_list:
            s_dist = math.hypot(s_obj.x - rx, s_obj.y - ry)
            if s_dist < self.max_range:
                angle_to = math.atan2(s_obj.y - ry, s_obj.x - rx)
                rel_angle = robot.normalize_angle(angle_to - r_theta)
                if abs(rel_angle) < (self.fov_rad / 2.0 + 0.25):
                    entities.append(("survivor", s_dist, rel_angle, s_obj))

        # Sort entities by distance descending (painter's algorithm)
        entities.sort(key=lambda item: item[1], reverse=True)

        for ent_type, dist, rel_angle, ent_obj in entities:
            # Perpendicular distance for projection
            perp_dist = dist * math.cos(rel_angle)
            if perp_dist < 5.0:
                continue

            u_center = int(self.cx + self.focal_length * math.tan(rel_angle))

            if ent_type == "survivor":
                # Real physical dimensions
                real_w = ent_obj.physical_width  # 24.0
                real_h = 36.0
                proj_w = max(4, int((real_w * self.focal_length) / perp_dist))
                proj_h = max(6, int((real_h * self.focal_length) / perp_dist))

                u_min = u_center - proj_w // 2
                u_max = u_center + proj_w // 2

                # Clamp to screen
                c_min = max(0, u_min)
                c_max = min(self.width, u_max)
                if c_max <= c_min:
                    continue

                # Check occlusion against depth buffer: visible if closer than walls
                cols = np.arange(c_min, c_max)
                visible_mask = perp_dist < self.depth_buffer[cols]
                if not np.any(visible_mask):
                    continue  # Blocked by rubble/wall

                vis_cols = cols[visible_mask]

                # Vertical span
                v_center = int(self.cy)
                v_top = max(0, v_center - proj_h // 2)
                v_bot = min(self.height, v_center + proj_h // 2)

                is_done = getattr(ent_obj, "is_rescued", False) or getattr(ent_obj, "is_secured", False)
                if is_done:
                    # Secured/rescued survivor: stabilized green medical gear, beacon offline
                    orange_bgr = np.array([110, 225, 40], dtype=np.uint8)
                    reflective_bgr = np.array([140, 240, 70], dtype=np.uint8)
                else:
                    # Vibrant rescue orange in BGR: (15, 120, 255) with pulsating brightness effect from beacon
                    pulse = 0.85 + 0.15 * math.sin(ent_obj.beacon_phase)
                    orange_bgr = np.array([int(15 * pulse), int(125 * pulse), int(255 * pulse)], dtype=np.uint8)
                    # High-vis fluorescent amber reflective chest stripe
                    reflective_bgr = np.array([30, 205, 255], dtype=np.uint8)

                # Render person silhouette: Head (top ~28%) + Torso (bottom ~72%)
                if proj_w >= 5 and proj_h >= 8:
                    head_h = max(2, int(proj_h * 0.28))
                    head_top = v_top
                    head_bot = min(self.height, v_top + head_h)
                    head_w = max(2, int(proj_w * 0.50))
                    head_min = u_center - head_w // 2
                    head_max = u_center + head_w // 2

                    torso_top = head_bot
                    torso_bot = v_bot

                    for col in vis_cols:
                        # Torso (full visible column)
                        if torso_bot > torso_top:
                            frame[torso_top:torso_bot, col] = orange_bgr
                        # Head (narrower central span)
                        if head_min <= col < head_max and head_bot > head_top:
                            frame[head_top:head_bot, col] = orange_bgr

                    # Reflective stripe across upper chest of torso
                    stripe_top = torso_top + max(1, int((torso_bot - torso_top) * 0.12))
                    stripe_bot = torso_top + max(2, int((torso_bot - torso_top) * 0.30))
                    if stripe_bot > stripe_top and stripe_bot <= self.height:
                        frame[stripe_top:stripe_bot, vis_cols] = reflective_bgr

                    # Triage Identification Tag (Red: CRITICAL, Yellow: INJURED, Green: STABLE)
                    triage_lvl = getattr(ent_obj, "triage", "CRITICAL")
                    if triage_lvl == "CRITICAL":
                        tag_bgr = np.array([20, 20, 235], dtype=np.uint8)   # Bright Red
                    elif triage_lvl == "INJURED":
                        tag_bgr = np.array([20, 230, 240], dtype=np.uint8)  # Bright Yellow
                    else:
                        tag_bgr = np.array([30, 220, 40], dtype=np.uint8)   # Bright Green

                    tag_top = torso_top + max(2, int((torso_bot - torso_top) * 0.42))
                    tag_bot = torso_top + max(3, int((torso_bot - torso_top) * 0.78))
                    tag_col_min = max(0, u_center - max(1, head_w // 2))
                    tag_col_max = min(self.width, u_center + max(1, head_w // 2))

                    if tag_bot > tag_top and tag_bot <= self.height:
                        for col in vis_cols:
                            if tag_col_min <= col < tag_col_max:
                                frame[tag_top:tag_bot, col] = tag_bgr
                else:
                    # Distant low-resolution fallback
                    frame[v_top:v_bot, vis_cols] = orange_bgr
                    triage_lvl = getattr(ent_obj, "triage", "CRITICAL")
                    tag_bgr = np.array([20, 20, 235], dtype=np.uint8) if triage_lvl == "CRITICAL" else (
                        np.array([20, 230, 240], dtype=np.uint8) if triage_lvl == "INJURED" else np.array([30, 220, 40], dtype=np.uint8)
                    )
                    mid_v = (v_top + v_bot) // 2
                    if 0 <= mid_v < self.height:
                        frame[mid_v : mid_v + 1, vis_cols] = tag_bgr

                # Update depth buffer for survivor columns
                self.depth_buffer[vis_cols] = perp_dist

            elif ent_type == "rescuer":
                real_w = 22.0
                real_h = 36.0
                proj_w = max(4, int((real_w * self.focal_length) / perp_dist))
                proj_h = max(6, int((real_h * self.focal_length) / perp_dist))

                u_min = u_center - proj_w // 2
                u_max = u_center + proj_w // 2

                c_min = max(0, u_min)
                c_max = min(self.width, u_max)
                if c_max <= c_min:
                    continue

                cols = np.arange(c_min, c_max)
                visible_mask = perp_dist < self.depth_buffer[cols]
                if not np.any(visible_mask):
                    continue

                vis_cols = cols[visible_mask]

                v_center = int(self.cy)
                v_top = max(0, v_center - proj_h // 2)
                v_bot = min(self.height, v_center + proj_h // 2)

                # Blue rescuer uniform in BGR: (235, 120, 30)
                blue_bgr = np.array([235, 120, 30], dtype=np.uint8)
                frame[v_top:v_bot, vis_cols] = blue_bgr

                # Yellow safety vest stripe
                stripe_top = v_top + int(proj_h * 0.30)
                stripe_bot = v_top + int(proj_h * 0.45)
                if stripe_bot > stripe_top:
                    frame[stripe_top:stripe_bot, vis_cols] = np.array([30, 220, 240], dtype=np.uint8)

                self.depth_buffer[vis_cols] = perp_dist

        # Phase 4: Camera Sensor Noise (Gaussian noise + smoke haze patch)
        if self.noise_enabled:
            # 1. Additive Gaussian noise (sigma ~ 14.0)
            noise = np.random.normal(0, 14.0, frame.shape).astype(np.float32)
            noisy = np.clip(frame.astype(np.float32) + noise, 0, 255)

            # 2. Drifting smoke haze patch
            self.smoke_time += 0.05
            cx_smoke = int(self.width * (0.50 + 0.18 * math.sin(self.smoke_time * 0.7)))
            cy_smoke = int(self.height * (0.52 + 0.12 * math.cos(self.smoke_time * 0.5)))
            Y, X = np.ogrid[:self.height, :self.width]
            dist_sq = ((X - cx_smoke) / 36.0) ** 2 + ((Y - cy_smoke) / 20.0) ** 2
            smoke_alpha = np.clip(0.38 * np.exp(-0.5 * dist_sq), 0.0, 0.38)[:, :, np.newaxis]
            smoke_color = np.array([125, 125, 135], dtype=np.float32)
            frame_blended = noisy * (1.0 - smoke_alpha) + smoke_color * smoke_alpha
            frame = np.clip(frame_blended, 0, 255).astype(np.uint8)

        self.last_frame = frame
        return frame

    def get_frustum_polygon(self, robot) -> List[Tuple[float, float]]:
        """
        Returns the 2D polygon vertices of the camera field-of-view cone in world space.
        Used for top-down tactical map rendering.
        """
        rx, ry = robot.x, robot.y
        half_fov = self.fov_rad / 2.0
        r_range = self.max_range

        left_angle = robot.normalize_angle(robot.theta - half_fov)
        right_angle = robot.normalize_angle(robot.theta + half_fov)

        p_left = (rx + r_range * math.cos(left_angle), ry + r_range * math.sin(left_angle))
        p_right = (rx + r_range * math.cos(right_angle), ry + r_range * math.sin(right_angle))

        # Arc points along the outer perimeter
        arc_points = []
        num_arc = 9
        for k in range(num_arc):
            frac = k / (num_arc - 1)
            ang = left_angle + frac * self.fov_rad
            arc_points.append((rx + r_range * math.cos(ang), ry + r_range * math.sin(ang)))

        return [(rx, ry)] + arc_points
