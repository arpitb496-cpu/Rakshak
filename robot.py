"""
robot.py - Differential-Drive Kinematics and Proximity Range Sensors for Rakshak.

Implements:
- Differential-drive unicycle kinematics (x, y, theta, v, w, v_left, v_right)
- 8 radial proximity range sensors with ray-vs-segment and ray-vs-circle intersection geometry
- Kinematic constraints (max velocity, max turn rate, wheel separation)
"""

import math
from typing import List, Tuple, Optional, Dict
import numpy as np


class ProximitySensorRay:
    """Ray intersection mathematics."""

    @staticmethod
    def ray_segment_intersection(
        ray_origin: Tuple[float, float],
        ray_dir: Tuple[float, float],
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float],
        max_dist: float,
    ) -> Optional[Tuple[float, Tuple[float, float]]]:
        """
        Calculates intersection distance t and point between ray and line segment.
        Ray: P(t) = ray_origin + t * ray_dir, t in [0, max_dist]
        Segment: S(u) = seg_start + u * (seg_end - seg_start), u in [0, 1]
        """
        ox, oy = ray_origin
        dx, dy = ray_dir
        x1, y1 = seg_start
        x2, y2 = seg_end

        sx = x2 - x1
        sy = y2 - y1

        # 2D cross product of ray_dir and segment vector
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-9:
            return None  # Parallel lines

        # Vector from ray origin to segment start
        qx = x1 - ox
        qy = y1 - oy

        t = (qx * sy - qy * sx) / denom
        u = (qx * dy - qy * dx) / denom

        if t >= 0.0 and t <= max_dist and 0.0 <= u <= 1.0:
            hit_x = ox + t * dx
            hit_y = oy + t * dy
            return t, (hit_x, hit_y)

        return None

    @staticmethod
    def ray_circle_intersection(
        ray_origin: Tuple[float, float],
        ray_dir: Tuple[float, float],
        circle_center: Tuple[float, float],
        circle_radius: float,
        max_dist: float,
    ) -> Optional[Tuple[float, Tuple[float, float]]]:
        """
        Calculates intersection distance t and point between ray and circle.
        """
        ox, oy = ray_origin
        dx, dy = ray_dir
        cx, cy = circle_center

        vx = ox - cx
        vy = oy - cy

        b = 2.0 * (dx * vx + dy * vy)
        c = (vx * vx + vy * vy) - circle_radius * circle_radius

        disc = b * b - 4.0 * c
        if disc < 0:
            return None

        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0

        t = None
        if t1 >= 0.0:
            t = t1
        elif t2 >= 0.0:
            t = t2

        if t is not None and t <= max_dist:
            hit_x = ox + t * dx
            hit_y = oy + t * dy
            return t, (hit_x, hit_y)

        return None


class DifferentialDriveRobot:
    """
    Differential drive robot with 8 proximity sensors.
    """

    # 8 Sensor offset angles relative to robot heading (in radians)
    # Order: Front, Front-Left, Left, Rear-Left, Rear, Rear-Right, Right, Front-Right
    SENSOR_ANGLES = [
        0.0,                    # 0: Front
        math.radians(30),       # 1: Front-Left
        math.radians(75),       # 2: Left
        math.radians(135),      # 3: Rear-Left
        math.radians(180),      # 4: Rear
        math.radians(-135),     # 5: Rear-Right
        math.radians(-75),      # 6: Right
        math.radians(-30),      # 7: Front-Right
    ]

    SENSOR_LABELS = ["F", "FL", "L", "RL", "R", "RR", "Ri", "FR"]

    def __init__(
        self,
        x: float = 80.0,
        y: float = 80.0,
        theta: float = 0.0,
        radius: float = 16.0,
        wheel_base: float = 28.0,
        max_speed: float = 110.0,
        max_omega: float = 3.0,
        sensor_range: float = 125.0,
    ):
        self.x = float(x)
        self.y = float(y)
        self.theta = self.normalize_angle(float(theta))

        self.radius = radius
        self.wheel_base = wheel_base
        self.max_speed = max_speed
        self.max_omega = max_omega
        self.sensor_range = sensor_range

        # Velocities
        self.v_left = 0.0
        self.v_right = 0.0
        self.v = 0.0       # Linear forward velocity (px/s)
        self.omega = 0.0   # Angular velocity (rad/s)

        # Proximity sensor readings
        self.num_sensors = len(self.SENSOR_ANGLES)
        self.sensor_distances: np.ndarray = np.full(self.num_sensors, self.sensor_range, dtype=np.float64)
        self.sensor_hit_points: List[Optional[Tuple[float, float]]] = [None] * self.num_sensors

        # Path history for trajectory rendering
        self.trail: List[Tuple[float, float]] = []
        self.max_trail_points = 250

    @property
    def w(self) -> float:
        """Alias for angular velocity omega."""
        return self.omega

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Wraps angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def set_wheel_speeds(self, v_left: float, v_right: float):
        """Sets individual left and right wheel speeds within motor limits."""
        v_l_clipped = float(np.clip(v_left, -self.max_speed, self.max_speed))
        v_r_clipped = float(np.clip(v_right, -self.max_speed, self.max_speed))

        self.v_left = v_l_clipped
        self.v_right = v_r_clipped
        self.v = (self.v_right + self.v_left) / 2.0
        self.omega = (self.v_right - self.v_left) / self.wheel_base

    def set_twist(self, v_linear: float, omega_angular: float):
        """
        Inverse kinematics: sets linear velocity and angular velocity,
        computing required wheel speeds.
        """
        # Clamp inputs to kinematic maximums
        v_cmd = float(np.clip(v_linear, -self.max_speed, self.max_speed))
        w_cmd = float(np.clip(omega_angular, -self.max_omega, self.max_omega))

        v_l = v_cmd - (w_cmd * self.wheel_base / 2.0)
        v_r = v_cmd + (w_cmd * self.wheel_base / 2.0)

        # Scale down if any wheel exceeds max speed
        peak = max(abs(v_l), abs(v_r))
        if peak > self.max_speed:
            scale = self.max_speed / peak
            v_l *= scale
            v_r *= scale

        self.set_wheel_speeds(v_l, v_r)

    def stop(self):
        """Brings robot to an immediate halt."""
        self.set_wheel_speeds(0.0, 0.0)

    def update_kinematics(self, dt: float, arena=None):
        """
        Integrates differential drive pose using Runge-Kutta / mid-point integration:
        x_dot = v * cos(theta + 0.5 * omega * dt)
        y_dot = v * sin(theta + 0.5 * omega * dt)
        theta_dot = omega
        """
        if abs(self.v) < 1e-4 and abs(self.omega) < 1e-4:
            return

        mid_theta = self.theta + 0.5 * self.omega * dt
        dx = self.v * math.cos(mid_theta) * dt
        dy = self.v * math.sin(mid_theta) * dt
        dtheta = self.omega * dt

        new_x = self.x + dx
        new_y = self.y + dy
        new_theta = self.normalize_angle(self.theta + dtheta)

        # Collision prevention with arena obstacles, walls, hazard zones, and rescuers
        if arena is not None:
            if arena.is_pose_safe(new_x, new_y, self.radius * 0.4):
                self.x = new_x
                self.y = new_y
            elif arena.is_pose_safe(new_x, self.y, self.radius * 0.4):
                self.x = new_x
            elif arena.is_pose_safe(self.x, new_y, self.radius * 0.4):
                self.y = new_y
            # If fully blocked, preserve position but allow rotation
            self.theta = new_theta
        else:
            self.x = new_x
            self.y = new_y
            self.theta = new_theta

        # Add to trail periodically
        if not self.trail or math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > 5.0:
            self.trail.append((self.x, self.y))
            if len(self.trail) > self.max_trail_points:
                self.trail.pop(0)

    def scan_proximity_sensors(self, arena) -> np.ndarray:
        """
        Performs 8-ray casting against arena obstacle segments, walls, and rescuer perimeters.
        Updates self.sensor_distances and self.sensor_hit_points.
        """
        segments = arena.get_all_segments()
        rescuers = arena.rescuers

        ox, oy = self.x, self.y

        for i, angle_offset in enumerate(self.SENSOR_ANGLES):
            ray_angle = self.normalize_angle(self.theta + angle_offset)
            rx = math.cos(ray_angle)
            ry = math.sin(ray_angle)

            closest_t = self.sensor_range
            closest_hit: Optional[Tuple[float, float]] = None

            # 1. Test against line segments (walls & rubble blocks)
            for seg_start, seg_end in segments:
                hit = ProximitySensorRay.ray_segment_intersection(
                    (ox, oy), (rx, ry), seg_start, seg_end, closest_t
                )
                if hit is not None:
                    t, pt = hit
                    if t < closest_t:
                        closest_t = t
                        closest_hit = pt

            # 2. Test against friendly rescuers (so APF and robot avoid them!)
            for rescuer in rescuers:
                hit = ProximitySensorRay.ray_circle_intersection(
                    (ox, oy), (rx, ry), (rescuer.x, rescuer.y), rescuer.radius, closest_t
                )
                if hit is not None:
                    t, pt = hit
                    if t < closest_t:
                        closest_t = t
                        closest_hit = pt

            self.sensor_distances[i] = closest_t
            if closest_hit is not None:
                self.sensor_hit_points[i] = closest_hit
            else:
                self.sensor_hit_points[i] = (ox + closest_t * rx, oy + closest_t * ry)

        return self.sensor_distances

    def get_min_proximity(self) -> float:
        """Returns the minimum clearance distance among all 8 sensors."""
        return float(np.min(self.sensor_distances))
