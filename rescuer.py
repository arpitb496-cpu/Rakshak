"""
rescuer.py - Active Rescuer Agents, Simulated Message Bus, and SOS Dispatch Manager for Rakshak.

Features:
- RescuerAgent with FSM states: STANDBY, DISPATCHED, EXTRACTING, RETURNING
- A* waypoint path navigation with dynamic obstacle/hazard avoidance
- Hard collision avoidance against rubble, hazards, walls, scout, and other rescuers
- Simulated message bus with configurable latency (default 0.3s)
- Triage-priority SOS queue (CRITICAL > INJURED > STABLE, then nearest by path length)
- 3.0s extraction cycle with progress reporting
- Sequential and parallel scout search coordination
"""

import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class RescuerState(Enum):
    STANDBY = "STANDBY"
    DISPATCHED = "DISPATCHED"
    EXTRACTING = "EXTRACTING"
    RETURNING = "RETURNING"


@dataclass
class SOSMessage:
    """SOS request dispatched from scout robot to rescue squad."""
    survivor_id: str
    est_x: float
    est_y: float
    triage: str                     # "CRITICAL", "INJURED", "STABLE"
    timestamp: float
    status: str = "QUEUED"          # "QUEUED", "DISPATCHED", "EXTRACTING", "SECURED", "UNREACHABLE"
    assigned_rescuer_id: Optional[str] = None
    dispatch_time: Optional[float] = None
    arrival_time: Optional[float] = None
    secured_time: Optional[float] = None


@dataclass
class BusMessage:
    """Message routed across simulated communication bus."""
    msg_type: str                   # "SOS", "ACK", "ARRIVED", "SECURED"
    sender: str
    recipient: str
    payload: Dict[str, Any]
    sent_at: float
    deliver_at: float


class MessageBus:
    """
    Simulated RF communication bus with realistic transmission latency.
    """

    def __init__(self, latency: float = 0.3):
        self.latency = latency
        self.pending_messages: List[BusMessage] = []
        self.delivered_history: List[BusMessage] = []

    def send(self, msg_type: str, sender: str, recipient: str, payload: Dict[str, Any], current_time: float):
        """Enqueues a message for delayed transmission."""
        msg = BusMessage(
            msg_type=msg_type,
            sender=sender,
            recipient=recipient,
            payload=payload,
            sent_at=current_time,
            deliver_at=current_time + self.latency,
        )
        self.pending_messages.append(msg)

    def update(self, current_time: float) -> List[BusMessage]:
        """Dispatches messages whose delivery time has arrived."""
        ready: List[BusMessage] = []
        remaining: List[BusMessage] = []
        for msg in self.pending_messages:
            if current_time >= msg.deliver_at:
                ready.append(msg)
                self.delivered_history.append(msg)
            else:
                remaining.append(msg)
        self.pending_messages = remaining
        return ready

    def reset(self):
        self.pending_messages.clear()
        self.delivered_history.clear()


class RescuerAgent:
    """
    Active autonomous search-and-rescue agent.
    Navigates via A* paths to extract survivors and return to base.
    """

    def __init__(
        self,
        id_num: int = 1,
        home_x: float = 0.0,
        home_y: float = 0.0,
        radius: float = 12.0,
        safe_buffer: float = 38.0,
        speed: float = 85.0,
        id: Optional[int] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ):
        actual_id = id if id is not None else id_num
        self.id = actual_id
        self.name = f"R{actual_id}"
        hx = float(x if x is not None else home_x)
        hy = float(y if y is not None else home_y)
        self.x = hx
        self.y = hy
        self.home_x = hx
        self.home_y = hy
        self.theta = 0.0
        self.radius = radius
        self.safe_buffer = safe_buffer
        self.speed = speed
        self.color = (35, 120, 245)
        self.vx = 0.0
        self.vy = 0.0

        # State machine
        self.state = RescuerState.STANDBY
        self.assigned_survivor_id: Optional[str] = None
        self.target_pos: Optional[Tuple[float, float]] = None
        self.path: List[Tuple[float, float]] = []
        self.path_idx: int = 0

        # Phase 4: Moving Standby toggle (P key)
        self.moving_standby: bool = False
        self.patrol_phase: float = 0.0

        # Extraction timer
        self.extraction_timer: float = 0.0
        self.extraction_duration: float = 3.0
        self.stuck_timer: float = 0.0
        self.replan_count: int = 0
        self._next_replan_thresh: float = 2.0

        # Statistics
        self.survivors_rescued: int = 0

    @property
    def extraction_progress(self) -> float:
        """Returns extraction progress fraction in [0.0, 1.0]."""
        if self.state == RescuerState.EXTRACTING:
            return min(1.0, self.extraction_timer / self.extraction_duration)
        return 0.0

    def dispatch(self, survivor_id: str, target_pos: Tuple[float, float], path: List[Tuple[float, float]]):
        """Dispatches rescuer to target survivor location along A* path."""
        self.state = RescuerState.DISPATCHED
        self.assigned_survivor_id = survivor_id
        self.target_pos = target_pos
        self.path = list(path)
        self.path_idx = 0
        self.extraction_timer = 0.0
        self._arrived_msg_sent = False

    def start_extraction(self, current_time: float, message_bus: MessageBus):
        """Starts extraction in manual mode when operator confirms via X key."""
        self.state = RescuerState.EXTRACTING
        self.extraction_timer = 0.0
        self.vx = 0.0
        self.vy = 0.0
        message_bus.send(
            msg_type="ARRIVED",
            sender=self.name,
            recipient="DISPATCHER",
            payload={"rescuer_id": self.name, "survivor_id": self.assigned_survivor_id},
            current_time=current_time,
        )

    def complete_extraction(self):
        """Fast-completes extraction."""
        if self.state == RescuerState.EXTRACTING:
            self.extraction_timer = self.extraction_duration

    def _find_nearest_free_pose(
        self, arena, scout_pos: Optional[Tuple[float, float]], other_rescuers: List["RescuerAgent"]
    ) -> Optional[Tuple[float, float]]:
        """Finds nearest free cell when stuck inside or adjacent to shifted rubble."""
        wall_pad = 18.0
        for r in range(10, 80, 10):
            for step_i in range(16):
                ang = step_i * (2.0 * math.pi / 16.0)
                cx = self.x + r * math.cos(ang)
                cy = self.y + r * math.sin(ang)
                if cx < wall_pad or cx > (arena.width - wall_pad) or cy < wall_pad or cy > (arena.height - wall_pad):
                    continue
                if arena.is_point_colliding(cx, cy, margin=self.radius + 2.0):
                    continue
                if arena.is_inside_hazard(cx, cy, margin=self.radius + 2.0):
                    continue
                if scout_pos is not None and math.hypot(cx - scout_pos[0], cy - scout_pos[1]) < (self.radius + 18.0):
                    continue
                if any(o.id != self.id and math.hypot(cx - o.x, cy - o.y) < 26.0 for o in other_rescuers):
                    continue
                return (cx, cy)
        return None

    def _handle_stuck(
        self,
        dt: float,
        arena,
        scout_pos,
        other_rescuers,
        is_returning: bool,
        message_bus: MessageBus,
        current_time: float,
        planner=None,
        dispatch_manager=None,
    ) -> Optional[str]:
        """
        Handles replanning on rubble shift:
        - Replan A* if stuck > 2s (with safety inflation >= rescuer radius + margin)
        - Push to nearest free cell if still stuck after 3 replans
        - Release SOS back to queue and return home if still stuck after 10s
        - Hard cap on stuck_timer
        """
        if self.stuck_timer > 10.0:
            self.stuck_timer = 10.0

        if self.stuck_timer >= 10.0:
            if not is_returning and self.assigned_survivor_id:
                sid = self.assigned_survivor_id
                if dispatch_manager is not None:
                    dispatch_manager.release_sos(sid)
                message_bus.send(
                    msg_type="RELEASE_SOS",
                    sender=self.name,
                    recipient="DISPATCHER",
                    payload={"rescuer_id": self.name, "survivor_id": sid},
                    current_time=current_time,
                )
                self.state = RescuerState.RETURNING
                self.assigned_survivor_id = None
                self.target_pos = None
                self.replan_count = 0
                self.stuck_timer = 0.0
                self._next_replan_thresh = 2.0
                if planner is not None:
                    self.path = planner.plan_path((self.x, self.y), (self.home_x, self.home_y), arena, ignore_rescuer_id=self.id)
                else:
                    self.path = [(self.home_x, self.home_y)]
                self.path_idx = 0
                return f"RELEASED:{self.name}:{sid}"
            else:
                free_pose = self._find_nearest_free_pose(arena, scout_pos, other_rescuers)
                if free_pose is not None:
                    self.x, self.y = free_pose
                self.stuck_timer = 0.0
                self._next_replan_thresh = 2.0
                self.replan_count = 0
                target = (self.home_x, self.home_y) if is_returning else (self.target_pos if self.target_pos else (self.home_x, self.home_y))
                if planner is not None:
                    self.path = planner.plan_path((self.x, self.y), target, arena, ignore_rescuer_id=self.id)
                else:
                    self.path = [target]
                self.path_idx = 0
                return None

        elif self.stuck_timer >= self._next_replan_thresh:
            self.replan_count += 1
            self._next_replan_thresh += 2.0
            if self.replan_count > 3:
                free_pose = self._find_nearest_free_pose(arena, scout_pos, other_rescuers)
                if free_pose is not None:
                    self.x, self.y = free_pose
            if planner is not None:
                target = self.target_pos if (not is_returning and self.target_pos) else (self.home_x, self.home_y)
                new_path = planner.plan_path((self.x, self.y), target, arena, ignore_rescuer_id=self.id)
                if new_path:
                    self.path = new_path
                    self.path_idx = 0
        return None

    def update(
        self,
        dt: float,
        arena,
        scout_pos: Tuple[float, float],
        other_rescuers: List["RescuerAgent"],
        message_bus: MessageBus,
        current_time: float,
        planner=None,
        manual_mode: bool = False,
        dispatch_manager=None,
    ) -> Optional[str]:
        """
        Executes one physics and navigation step.
        Returns event string if state transition occurred.
        """
        event = None

        if self.state == RescuerState.STANDBY:
            if self.moving_standby:
                self.patrol_phase += dt * 1.5
                patrol_radius = 18.0
                target_x = self.home_x + patrol_radius * math.cos(self.patrol_phase + self.id * 2.0)
                target_y = self.home_y + patrol_radius * math.sin(self.patrol_phase + self.id * 2.0)

                # Collision safety checks
                safe = arena.is_pose_safe(target_x, target_y, radius=self.radius + 4.0)
                if safe and scout_pos is not None:
                    d_scout = math.hypot(target_x - scout_pos[0], target_y - scout_pos[1])
                    if d_scout < (self.safe_buffer + 8.0):
                        safe = False
                if safe and other_rescuers:
                    for other in other_rescuers:
                        if other.id != self.id:
                            d_oth = math.hypot(target_x - other.x, target_y - other.y)
                            if d_oth < (self.radius + other.radius + 12.0):
                                safe = False
                                break

                if safe:
                    dx = target_x - self.x
                    dy = target_y - self.y
                    step = min(math.hypot(dx, dy), 28.0 * dt)
                    ang = math.atan2(dy, dx)
                    self.x += step * math.cos(ang)
                    self.y += step * math.sin(ang)
                    self.vx = step * math.cos(ang) / max(dt, 1e-4)
                    self.vy = step * math.sin(ang) / max(dt, 1e-4)
                else:
                    self.vx = 0.0
                    self.vy = 0.0
            else:
                # Idle at home post
                self.vx = 0.0
                self.vy = 0.0
                d_home = math.hypot(self.x - self.home_x, self.y - self.home_y)
                if d_home > 3.0:
                    ang = math.atan2(self.home_y - self.y, self.home_x - self.x)
                    step = min(d_home, self.speed * dt)
                    self.x += step * math.cos(ang)
                    self.y += step * math.sin(ang)

        elif self.state == RescuerState.DISPATCHED:
            # Navigate along A* path toward target survivor
            event = self._follow_path(
                dt, arena, scout_pos, other_rescuers, is_returning=False,
                message_bus=message_bus, current_time=current_time,
                planner=planner, dispatch_manager=dispatch_manager,
                manual_mode=manual_mode,
            )

        elif self.state == RescuerState.EXTRACTING:
            # Performing field stabilization and extraction
            self.vx = 0.0
            self.vy = 0.0
            self.extraction_timer += dt

            if self.extraction_timer >= self.extraction_duration:
                # FIX 1: Only the assigned survivor may be marked secured by this rescuer
                target_s = None
                survs = getattr(arena, "survivors", None)
                if survs:
                    for s in survs:
                        if s.id == self.assigned_survivor_id:
                            if self.target_pos is None or math.hypot(s.x - self.target_pos[0], s.y - self.target_pos[1]) < 85.0:
                                target_s = s
                                break
                    if target_s is None and self.target_pos is not None:
                        close_to_target = [
                            s for s in survs
                            if not s.is_secured and math.hypot(s.x - self.target_pos[0], s.y - self.target_pos[1]) < 85.0
                        ]
                        if close_to_target:
                            target_s = min(
                                close_to_target,
                                key=lambda s: math.hypot(s.x - self.target_pos[0], s.y - self.target_pos[1]),
                            )
                    if target_s is not None:
                        target_s.is_rescued = True
                        target_s.is_secured = True
                        actual_id = target_s.id
                    else:
                        actual_id = self.assigned_survivor_id
                else:
                    if hasattr(arena, "survivor") and arena.survivor is not None:
                        arena.survivor.is_rescued = True
                        arena.survivor.is_secured = True
                        actual_id = getattr(arena.survivor, "id", self.assigned_survivor_id)
                    else:
                        actual_id = self.assigned_survivor_id

                self.survivors_rescued += 1
                event = f"SECURED:{self.name}:{actual_id}"

                # Emit SECURED message on RF bus
                message_bus.send(
                    msg_type="SECURED",
                    sender=self.name,
                    recipient="DISPATCHER",
                    payload={"rescuer_id": self.name, "survivor_id": actual_id, "track_id": self.assigned_survivor_id},
                    current_time=current_time,
                )

                # Plan path back to home post
                if planner is not None:
                    return_path = planner.plan_path((self.x, self.y), (self.home_x, self.home_y), arena, ignore_rescuer_id=self.id)
                else:
                    return_path = [(self.home_x, self.home_y)]

                self.state = RescuerState.RETURNING
                self.assigned_survivor_id = None
                self.target_pos = None
                self.path = return_path if return_path else [(self.home_x, self.home_y)]
                self.path_idx = 0

        elif self.state == RescuerState.RETURNING:
            # Navigate back to base
            event = self._follow_path(
                dt, arena, scout_pos, other_rescuers, is_returning=True,
                message_bus=message_bus, current_time=current_time,
                planner=planner, dispatch_manager=dispatch_manager,
                manual_mode=manual_mode,
            )

        return event

    def _follow_path(
        self,
        dt: float,
        arena,
        scout_pos: Tuple[float, float],
        other_rescuers: List["RescuerAgent"],
        is_returning: bool,
        message_bus: MessageBus,
        current_time: float,
        planner=None,
        dispatch_manager=None,
        manual_mode: bool = False,
    ) -> Optional[str]:
        """Moves along waypoints while enforcing hard collision checks."""
        event = None

        # If inbound to target survivor and within extraction radius (<= 55px, or <= 75px if held/stuck)
        # FIX 1: Arrival must be judged ONLY against its OWN assigned survivor target, never against any other survivor
        if not is_returning:
            dist_target = math.hypot(self.x - self.target_pos[0], self.y - self.target_pos[1]) if self.target_pos is not None else float("inf")
            dist_assigned = float("inf")
            survs = getattr(arena, "survivors", None)
            if survs:
                assigned_s = None
                for s in survs:
                    if not s.is_secured and s.id == self.assigned_survivor_id:
                        if self.target_pos is None or math.hypot(s.x - self.target_pos[0], s.y - self.target_pos[1]) < 85.0:
                            assigned_s = s
                            break
                if assigned_s is None and self.target_pos is not None:
                    close_to_target = [
                        s for s in survs
                        if not s.is_secured and math.hypot(s.x - self.target_pos[0], s.y - self.target_pos[1]) < 85.0
                    ]
                    if close_to_target:
                        assigned_s = min(
                            close_to_target,
                            key=lambda s: math.hypot(s.x - self.target_pos[0], s.y - self.target_pos[1]),
                        )
                if assigned_s is not None:
                    dist_assigned = math.hypot(self.x - assigned_s.x, self.y - assigned_s.y)
            elif hasattr(arena, "survivor") and arena.survivor is not None:
                if not getattr(arena.survivor, "is_secured", False):
                    dist_assigned = math.hypot(self.x - arena.survivor.x, self.y - arena.survivor.y)

            effective_dist = min(dist_target, dist_assigned)
            if effective_dist <= 55.0 or (self.stuck_timer > 2.0 and effective_dist <= 75.0):
                self.stuck_timer = 0.0
                if manual_mode:
                    self.vx = 0.0
                    self.vy = 0.0
                    if not getattr(self, "_arrived_msg_sent", False):
                        self._arrived_msg_sent = True
                        message_bus.send(
                            msg_type='ARRIVED',
                            sender=self.name,
                            recipient='DISPATCHER',
                            payload={'rescuer_id': self.name, 'survivor_id': self.assigned_survivor_id},
                            current_time=current_time,
                        )
                    return f'AT_PERIMETER:{self.name}:{self.assigned_survivor_id}'
                else:
                    self.state = RescuerState.EXTRACTING
                    self.extraction_timer = 0.0
                    message_bus.send(
                        msg_type='ARRIVED',
                        sender=self.name,
                        recipient='DISPATCHER',
                        payload={'rescuer_id': self.name, 'survivor_id': self.assigned_survivor_id},
                        current_time=current_time,
                    )
                    return f'ARRIVED:{self.name}:{self.assigned_survivor_id}'

        if not self.path or self.path_idx >= len(self.path):
            if is_returning:
                self.state = RescuerState.STANDBY
                self.assigned_survivor_id = None
                self.target_pos = None
                return f"STANDBY:{self.name}"
            else:
                # Arrived at destination
                if manual_mode:
                    self.vx = 0.0
                    self.vy = 0.0
                    if not getattr(self, "_arrived_msg_sent", False):
                        self._arrived_msg_sent = True
                        message_bus.send(
                            msg_type="ARRIVED",
                            sender=self.name,
                            recipient="DISPATCHER",
                            payload={"rescuer_id": self.name, "survivor_id": self.assigned_survivor_id},
                            current_time=current_time,
                        )
                    return f"AT_PERIMETER:{self.name}:{self.assigned_survivor_id}"
                else:
                    self.state = RescuerState.EXTRACTING
                    self.extraction_timer = 0.0
                    message_bus.send(
                        msg_type="ARRIVED",
                        sender=self.name,
                        recipient="DISPATCHER",
                        payload={"rescuer_id": self.name, "survivor_id": self.assigned_survivor_id},
                        current_time=current_time,
                    )
                    return f"ARRIVED:{self.name}:{self.assigned_survivor_id}"

        wp = self.path[self.path_idx]
        # Replan if next waypoint is blocked by a shifted obstacle or hazard
        if (arena.is_point_colliding(wp[0], wp[1], margin=self.radius + 2.0) or
                arena.is_inside_hazard(wp[0], wp[1], margin=self.radius + 2.0)):
            if planner is not None:
                target = self.target_pos if (not is_returning and self.target_pos) else (self.home_x, self.home_y)
                new_p = planner.plan_path((self.x, self.y), target, arena, ignore_rescuer_id=self.id)
                if new_p:
                    self.path = new_p
                    self.path_idx = 0
                    wp = self.path[self.path_idx]
        dx = wp[0] - self.x
        dy = wp[1] - self.y
        dist_wp = math.hypot(dx, dy)

        if dist_wp < 14.0:
            self.path_idx += 1
            if self.path_idx >= len(self.path):
                return self._follow_path(
                    dt, arena, scout_pos, other_rescuers, is_returning, message_bus, current_time,
                    planner=planner, dispatch_manager=dispatch_manager, manual_mode=manual_mode
                )
            wp = self.path[self.path_idx]
            dx = wp[0] - self.x
            dy = wp[1] - self.y
            dist_wp = math.hypot(dx, dy)

        # Move toward waypoint
        angle = math.atan2(dy, dx)
        self.theta = angle
        move_dist = min(dist_wp, self.speed * dt)

        cand_x = self.x + move_dist * math.cos(angle)
        cand_y = self.y + move_dist * math.sin(angle)

        # Hard collision avoidance check
        wall_pad = 18.0
        if cand_x < wall_pad or cand_x > (arena.width - wall_pad) or cand_y < wall_pad or cand_y > (arena.height - wall_pad):
            cand_x = np.clip(cand_x, wall_pad, arena.width - wall_pad)
            cand_y = np.clip(cand_y, wall_pad, arena.height - wall_pad)

        # Obstacle check
        if arena.is_point_colliding(cand_x, cand_y, margin=self.radius + 2.0):
            self.stuck_timer += dt
            return self._handle_stuck(dt, arena, scout_pos, other_rescuers, is_returning, message_bus, current_time, planner, dispatch_manager)

        # Hazard check
        if arena.is_inside_hazard(cand_x, cand_y, margin=self.radius + 2.0):
            self.stuck_timer += dt
            return self._handle_stuck(dt, arena, scout_pos, other_rescuers, is_returning, message_bus, current_time, planner, dispatch_manager)

        # Scout robot dynamic separation (keep >= radius sum + 6px)
        dist_scout = math.hypot(cand_x - scout_pos[0], cand_y - scout_pos[1])
        if dist_scout < (self.radius + 18.0):
            # Attempt tangential detour around stationary scout robot rather than deadlocking
            perp = math.atan2(self.y - scout_pos[1], self.x - scout_pos[0])
            detour_found = False
            for sign in [1.0, -1.0]:
                alt_ang = perp + sign * (math.pi / 2.0)
                alt_x = self.x + move_dist * math.cos(alt_ang)
                alt_y = self.y + move_dist * math.sin(alt_ang)
                if (not arena.is_point_colliding(alt_x, alt_y, margin=self.radius + 2.0)
                        and not arena.is_inside_hazard(alt_x, alt_y, margin=self.radius + 2.0)):
                    if math.hypot(alt_x - scout_pos[0], alt_y - scout_pos[1]) >= (self.radius + 18.0 - 2.0):
                        cand_x, cand_y = alt_x, alt_y
                        detour_found = True
                        break
            if not detour_found:
                self.stuck_timer += dt
                return self._handle_stuck(dt, arena, scout_pos, other_rescuers, is_returning, message_bus, current_time, planner, dispatch_manager)

        # Other rescuers dynamic separation (keep >= 26px)
        for other in other_rescuers:
            if other.id != self.id:
                d_other = math.hypot(cand_x - other.x, cand_y - other.y)
                if d_other < 26.0 and other.id < self.id:
                    # Yield to lower ID rescuer
                    self.stuck_timer += dt
                    return self._handle_stuck(dt, arena, scout_pos, other_rescuers, is_returning, message_bus, current_time, planner, dispatch_manager)

        # Check net displacement to detect oscillation deadlocks
        if not hasattr(self, "_pos_history"):
            self._pos_history = []
        self._pos_history.append((self.x, self.y))
        if len(self._pos_history) > 30:
            self._pos_history.pop(0)

        is_oscillating = False
        if len(self._pos_history) >= 25:
            disp = math.hypot(self.x - self._pos_history[0][0], self.y - self._pos_history[0][1])
            if disp < 8.0:
                is_oscillating = True

        if math.hypot(cand_x - self.x, cand_y - self.y) < 0.1 or is_oscillating:
            self.stuck_timer += dt
            stuck_event = self._handle_stuck(dt, arena, scout_pos, other_rescuers, is_returning, message_bus, current_time, planner, dispatch_manager)
            if stuck_event:
                self._pos_history.clear()
                return stuck_event
        else:
            self.stuck_timer = 0.0
            self.replan_count = 0
            self._next_replan_thresh = 2.0

        self.vx = (cand_x - self.x) / max(1e-4, dt)
        self.vy = (cand_y - self.y) / max(1e-4, dt)
        self.x = cand_x
        self.y = cand_y

        return event


TRIAGE_RANKS = {"CRITICAL": 0, "INJURED": 1, "STABLE": 2}


def triage_rank(triage: str) -> int:
    """Returns priority rank (lower value = higher priority: CRITICAL=0, INJURED=1, STABLE=2)."""
    return TRIAGE_RANKS.get(str(triage).upper(), 3)


def survivor_priority_key(item, ref_pos=(0, 0)):
    """Shared priority key: (triage_rank, distance_to_ref_pos)."""
    triage = getattr(item, "triage", None) or getattr(item, "triage_level", "STABLE")
    x = getattr(item, "x", None)
    if x is None:
        x = getattr(item, "est_x", None)
    if x is None:
        x = getattr(item, "est_world_x", 0.0)
    y = getattr(item, "y", None)
    if y is None:
        y = getattr(item, "est_y", None)
    if y is None:
        y = getattr(item, "est_world_y", 0.0)
    dist = math.hypot(x - ref_pos[0], y - ref_pos[1])
    return (triage_rank(triage), dist)


class DispatchManager:
    """
    Coordinates multi-survivor rescue requests, prioritization, and unit dispatch.
    Priority order: CRITICAL > INJURED > STABLE, then shortest A* path length.
    """

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.sos_queue: List[SOSMessage] = []
        self.active_sos: Dict[str, SOSMessage] = {}
        self.completed_sos: Dict[str, SOSMessage] = {}
        self.unreachable_sos: Dict[str, SOSMessage] = {}

    @staticmethod
    def triage_rank(triage: str) -> int:
        """Returns priority rank (lower value = higher priority)."""
        return triage_rank(triage)

    def release_sos(self, survivor_id: str):
        """Releases an active SOS back to the queue so another rescuer can take it."""
        sos = None
        if survivor_id in self.active_sos:
            sos = self.active_sos.pop(survivor_id)
        if sos is not None:
            sos.status = "PENDING"
            sos.assigned_rescuer_id = None
            sos.dispatch_time = None
            if not any(s.survivor_id == survivor_id for s in self.sos_queue):
                self.sos_queue.insert(0, sos)

    def submit_sos(
        self,
        survivor_id: str,
        est_x: float,
        est_y: float,
        triage: str,
        current_time: float,
        detection_range: float = 0.0,
    ) -> bool:
        """Submits an SOS to the rescue dispatch queue with spatial deduplication."""
        if survivor_id in self.active_sos or survivor_id in self.completed_sos:
            return False
        for s in self.sos_queue:
            if s.survivor_id == survivor_id:
                return False

        # Range-adaptive spatial deduplication: skip if SOS for same physical survivor is already queued/active
        # FIX 1: dedupe gate < 40 px so survivors ~50 px apart each get their own SOS
        gate = min(38.0, 25.0 + 0.05 * detection_range)
        for s in self.active_sos.values():
            if math.hypot(est_x - s.est_x, est_y - s.est_y) < gate:
                return False
        for s in self.sos_queue:
            if math.hypot(est_x - s.est_x, est_y - s.est_y) < gate:
                return False
        for s in self.completed_sos.values():
            if (s.est_x > 0.0 or s.est_y > 0.0) and math.hypot(est_x - s.est_x, est_y - s.est_y) < gate:
                return False

        sos = SOSMessage(
            survivor_id=survivor_id,
            est_x=est_x,
            est_y=est_y,
            triage=triage,
            timestamp=current_time,
        )
        self.sos_queue.append(sos)

        # Broadcast SOS over bus
        self.bus.send(
            msg_type="SOS",
            sender="SCOUT",
            recipient="DISPATCHER",
            payload={
                "survivor_id": survivor_id,
                "coords": (est_x, est_y),
                "triage": triage,
            },
            current_time=current_time,
        )
        return True

    def dispatch_next_sos(
        self,
        rescuers: List[RescuerAgent],
        arena,
        planner,
        current_time: float,
        log_callback=None,
    ) -> bool:
        available_rescuers = [r for r in rescuers if r.state == RescuerState.STANDBY]
        if not available_rescuers or not self.sos_queue:
            return False

        # Sort pending SOS requests: severity first (CRITICAL > INJURED > STABLE), then nearest-first within same severity
        ref_x = sum(r.x for r in available_rescuers) / len(available_rescuers)
        ref_y = sum(r.y for r in available_rescuers) / len(available_rescuers)
        self.sos_queue.sort(key=lambda s: survivor_priority_key(s, (ref_x, ref_y)))

        for sos in list(self.sos_queue):
            best_rescuer = None
            best_path = None
            best_dist = float("inf")

            for rescuer in available_rescuers:
                path = planner.plan_path((rescuer.x, rescuer.y), (sos.est_x, sos.est_y), arena, ignore_rescuer_id=rescuer.id)
                if path:
                    path_len = 0.0
                    for k in range(len(path) - 1):
                        path_len += math.hypot(path[k+1][0] - path[k][0], path[k+1][1] - path[k][1])
                    if path_len < best_dist:
                        best_dist = path_len
                        best_rescuer = rescuer
                        best_path = path

            if best_rescuer is not None and best_path is not None:
                self.sos_queue.remove(sos)
                sos.status = "DISPATCHED"
                sos.assigned_rescuer_id = best_rescuer.name
                sos.dispatch_time = current_time
                self.active_sos[sos.survivor_id] = sos
                available_rescuers.remove(best_rescuer)

                best_rescuer.dispatch(sos.survivor_id, (sos.est_x, sos.est_y), best_path)

                self.bus.send(
                    msg_type="ACK",
                    sender="DISPATCHER",
                    recipient="SCOUT",
                    payload={"rescuer_id": best_rescuer.name, "survivor_id": sos.survivor_id},
                    current_time=current_time,
                )
                try:
                    from sound import get_sound_manager
                    get_sound_manager().play_rescuer_dispatched(sos.survivor_id, sos.triage)
                except Exception:
                    pass

                if log_callback:
                    log_callback(f"[COMM] ACK: Rescuer {best_rescuer.name} dispatched to {sos.survivor_id} ({sos.triage}).")
                return True
            else:
                sos.status = "UNREACHABLE"
                if log_callback:
                    log_callback(f"[DISPATCH] Target {sos.survivor_id} currently unreachable.")

        return False

    def update(
        self,
        dt: float,
        current_time: float,
        rescuers: List[RescuerAgent],
        arena,
        planner,
        log_callback=None,
        auto_dispatch: bool = True,
    ):
        """
        Processes delivered bus messages and dispatches available rescuers according to triage priority.
        """
        # 1. Process delivered bus messages
        messages = self.bus.update(current_time)
        for msg in messages:
            if msg.msg_type == "SOS":
                p = msg.payload
                if log_callback:
                    coords = p.get("coords", (0, 0))
                    log_callback(f"[COMM] SOS: {p['survivor_id']} ({p['triage']}) at ({coords[0]:.1f}, {coords[1]:.1f})")

            elif msg.msg_type == "ACK":
                p = msg.payload
                if log_callback:
                    log_callback(f"[COMM] ACK: Rescuer {p['rescuer_id']} dispatched to {p['survivor_id']}.")

            elif msg.msg_type == "RELEASE_SOS":
                p = msg.payload
                sid = p.get("survivor_id")
                if sid:
                    self.release_sos(sid)
                if log_callback:
                    log_callback(f"[COMM] RELEASE: Rescuer {p.get('rescuer_id')} released {sid} back to queue.")

            elif msg.msg_type == "ARRIVED":
                p = msg.payload
                sid = p["survivor_id"]
                if sid in self.active_sos:
                    self.active_sos[sid].arrival_time = current_time
                    self.active_sos[sid].status = "EXTRACTING"
                if log_callback:
                    log_callback(f"[COMM] ARRIVED: Rescuer {p['rescuer_id']} reached {sid}.")

            elif msg.msg_type == "SECURED":
                p = msg.payload
                sid = p["survivor_id"]
                track_id = p.get("track_id")
                matched_sos = None
                if track_id and track_id in self.active_sos:
                    matched_sos = self.active_sos.pop(track_id)
                elif sid in self.active_sos:
                    matched_sos = self.active_sos.pop(sid)
                if matched_sos:
                    matched_sos.secured_time = current_time
                    matched_sos.status = "SECURED"
                    self.completed_sos[sid] = matched_sos
                    if track_id:
                        self.completed_sos[track_id] = matched_sos
                else:
                    new_sos = SOSMessage(survivor_id=sid, est_x=0.0, est_y=0.0, triage="CRITICAL", timestamp=current_time)
                    new_sos.status = "SECURED"
                    new_sos.secured_time = current_time
                    self.completed_sos[sid] = new_sos
                    if track_id:
                        self.completed_sos[track_id] = new_sos
                try:
                    from sound import get_sound_manager
                    sm = get_sound_manager()
                    sm.clear_siren_flag(sid)
                    if track_id:
                        sm.clear_siren_flag(track_id)
                    triage_lvl = matched_sos.triage if matched_sos else "CRITICAL"
                    sm.play_extraction_complete(sid, triage_lvl)
                except Exception:
                    pass
                if log_callback:
                    log_callback(f"Survivor {sid} secured by Rescuer {p['rescuer_id']}")

        # Note: SOS release is handled explicitly via RELEASE_SOS message bus protocol

        # 2. Sort pending SOS requests: CRITICAL > INJURED > STABLE, then nearest-first
        ref_x = sum(r.x for r in rescuers) / len(rescuers) if rescuers else 0.0
        ref_y = sum(r.y for r in rescuers) / len(rescuers) if rescuers else 0.0
        self.sos_queue.sort(key=lambda s: survivor_priority_key(s, (ref_x, ref_y)))

        # 3. Attempt to dispatch available rescuers if auto_dispatch is True
        if auto_dispatch:
            while self.dispatch_next_sos(rescuers, arena, planner, current_time, log_callback):
                pass

    def reset(self):
        self.bus.reset()
        self.sos_queue.clear()
        self.active_sos.clear()
        self.completed_sos.clear()
        self.unreachable_sos.clear()
