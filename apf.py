"""
apf.py - Artificial Potential Field (APF) Reactive Obstacle Avoidance for Rakshak.

Combines:
- Attractive force pulling toward the target waypoint
- Repulsive forces pushing away from proximity sensor obstacle hits
- Safe-buffer repulsive forces respecting blue friendly rescuer personnel
- Hazard-zone avoidance repulsion
- Tangential (vortex) steering force to break local minima / symmetric traps
- Differential-drive velocity generation (v_linear, omega_angular)
"""

import math
from typing import Tuple, List, Optional
import numpy as np


class APFController:
    """
    Artificial Potential Field Controller.
    """

    def __init__(
        self,
        k_att: float = 1.2,
        k_rep: float = 2800.0,
        k_tan: float = 1.4,
        repulsion_threshold: float = 75.0,
        rescuer_buffer_dist: float = 55.0,
        max_speed: float = 100.0,
        max_omega: float = 2.8,
    ):
        self.k_att = k_att
        self.k_rep = k_rep
        self.k_tan = k_tan
        self.rep_thresh = repulsion_threshold
        self.rescuer_buffer = rescuer_buffer_dist
        self.max_speed = max_speed
        self.max_omega = max_omega

        # Cached forces for visualization in HUD
        self.f_att: np.ndarray = np.zeros(2)
        self.f_rep: np.ndarray = np.zeros(2)
        self.f_tan: np.ndarray = np.zeros(2)
        self.f_net: np.ndarray = np.zeros(2)

    def compute_forces(
        self,
        robot,
        goal_x: float,
        goal_y: float,
        arena,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes attractive, repulsive, tangential, and resultant force vectors.
        """
        rx, ry = robot.x, robot.y
        r_theta = robot.theta

        # 1. Attractive Force
        dx_g = goal_x - rx
        dy_g = goal_y - ry
        dist_g = math.hypot(dx_g, dy_g)

        if dist_g > 1e-3:
            # Scale attractive force smoothly
            att_mag = min(dist_g * self.k_att, 60.0)
            f_att = np.array([dx_g / dist_g * att_mag, dy_g / dist_g * att_mag])
        else:
            f_att = np.zeros(2)

        # 2. Repulsive Force from 8 Proximity Sensors
        f_rep = np.zeros(2)

        for i, dist in enumerate(robot.sensor_distances):
            if dist < self.rep_thresh:
                # Sensor absolute orientation
                sensor_angle = robot.normalize_angle(r_theta + robot.SENSOR_ANGLES[i])
                ray_dir = np.array([math.cos(sensor_angle), math.sin(sensor_angle)])

                # Repulsion magnitude: k_rep * (1/d - 1/d0) * (1/d^2)
                d_clamped = max(dist, 5.0)
                mag = self.k_rep * (1.0 / d_clamped - 1.0 / self.rep_thresh) / (d_clamped * d_clamped)
                # Push in opposite direction of the sensor ray
                f_rep -= ray_dir * mag

        # 3. Dedicated Rescuer Personnel Safety Buffer Repulsion
        for rescuer in arena.rescuers:
            r_dist = math.hypot(rx - rescuer.x, ry - rescuer.y)
            if r_dist < self.rescuer_buffer:
                away_dir = np.array([rx - rescuer.x, ry - rescuer.y])
                norm = math.hypot(away_dir[0], away_dir[1])
                if norm > 1e-3:
                    away_dir /= norm
                    clearance = max(2.0, r_dist - rescuer.radius)
                    mag = 120.0 * (1.0 / clearance) + 50.0 * (1.0 - r_dist / self.rescuer_buffer)
                    f_rep += away_dir * mag

        # 4. Hazard Zone Repulsion (Strong protective barrier)
        for hz in arena.hazard_zones:
            h_dist = math.hypot(rx - hz.x, ry - hz.y)
            effective_radius = hz.radius + 38.0
            if h_dist < effective_radius:
                away_dir = np.array([rx - hz.x, ry - hz.y])
                norm = math.hypot(away_dir[0], away_dir[1])
                if norm > 1e-3:
                    away_dir /= norm
                    clearance = max(2.0, h_dist - hz.radius)
                    mag = 150.0 * (1.0 / clearance) + 60.0 * (1.0 - h_dist / effective_radius)
                    f_rep += away_dir * mag

        # 5. Tangential (Vortex) Force to eliminate local minima
        f_tan = np.zeros(2)
        norm_rep = np.linalg.norm(f_rep)
        norm_att = np.linalg.norm(f_att)

        if norm_rep > 3.0 and norm_att > 1e-3:
            # Check if repulsive force directly opposes goal heading
            dot_prod = (f_rep[0] * f_att[0] + f_rep[1] * f_att[1]) / (norm_rep * norm_att)
            if dot_prod < -0.2:
                # Perpendicular vectors: (-ry, rx) and (ry, -rx)
                perp_cw = np.array([-f_rep[1], f_rep[0]]) / norm_rep
                perp_ccw = np.array([f_rep[1], -f_rep[0]]) / norm_rep

                # Choose the direction that aligns better with attractive goal heading
                score_cw = np.dot(perp_cw, f_att)
                score_ccw = np.dot(perp_ccw, f_att)

                chosen_perp = perp_cw if score_cw >= score_ccw else perp_ccw
                f_tan = chosen_perp * (norm_rep * self.k_tan)

        # 6. Resultant Force
        f_net = f_att + f_rep + f_tan

        self.f_att = f_att
        self.f_rep = f_rep
        self.f_tan = f_tan
        self.f_net = f_net

        return f_att, f_rep, f_tan, f_net

    def compute_control(
        self,
        robot,
        goal_x: float,
        goal_y: float,
        arena,
        target_standoff: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Calculates differential drive linear velocity (v) and angular velocity (omega).
        """
        dist_to_goal = math.hypot(goal_x - robot.x, goal_y - robot.y)

        # If already at or within designated safe standoff perimeter, stop forward motion
        if target_standoff > 0.0 and dist_to_goal <= target_standoff:
            # Face the target smoothly
            desired_heading = math.atan2(goal_y - robot.y, goal_x - robot.x)
            heading_err = robot.normalize_angle(desired_heading - robot.theta)
            omega = float(np.clip(2.5 * heading_err, -self.max_omega, self.max_omega))
            return 0.0, omega

        f_att, f_rep, f_tan, f_net = self.compute_forces(robot, goal_x, goal_y, arena)

        net_mag = np.linalg.norm(f_net)
        if net_mag < 0.1:
            return 0.0, 0.0

        # Desired heading from net force vector
        desired_heading = math.atan2(f_net[1], f_net[0])
        heading_err = robot.normalize_angle(desired_heading - robot.theta)

        # Proportional angular controller
        omega = float(np.clip(2.8 * heading_err, -self.max_omega, self.max_omega))

        # Forward speed modulated by heading alignment:
        # Full speed when aligned, slows down during sharp turns
        align_factor = max(0.0, math.cos(heading_err))

        # Proximity safety brake: slow down if obstacles are close ahead
        front_clearance = min(
            robot.sensor_distances[0],  # Front
            robot.sensor_distances[1],  # Front-Left
            robot.sensor_distances[7],  # Front-Right
        )
        clearance_factor = min(1.0, max(0.25, front_clearance / 65.0))

        v = float(np.clip(self.max_speed * align_factor * clearance_factor, 0.0, self.max_speed))

        # If turn is very sharp (> 70 degrees), prioritize in-place rotation
        if abs(heading_err) > math.radians(70):
            v *= 0.2

        return v, omega
