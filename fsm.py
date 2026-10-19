"""
fsm.py - Mission Finite State Machine for Rakshak.

Manages autonomous search-and-rescue lifecycle:
- SEARCH_PATROL: Systematic arena coverage sweep & rotational camera scanning
- VERIFY_TARGET: Slow down, orient toward visual detection, lock multi-frame coordinates
- PLAN_ROUTE: Generate global A* path to safe standoff rescue perimeter
- NAVIGATE_TO_SURVIVOR: Waypoint tracking augmented with real-time APF reactive obstacle avoidance
- RESCUE_OPERATION: Standoff positioning, visual beacon verification, distress broadcast
- MISSION_COMPLETE: Stable standoff hold and telemetric mission debrief
- HAZARD_RECOVERY: Evasive reverse/turn maneuvers if entering danger zones
"""

import math
from enum import Enum, auto
from typing import List, Tuple, Optional, Any, Dict
import numpy as np

from rescuer import MessageBus, DispatchManager, RescuerState, RescuerAgent, survivor_priority_key
from coverage import CoverageGrid


class MissionState(Enum):
    SEARCH_PATROL = auto()
    VERIFY_TARGET = auto()
    PLAN_ROUTE = auto()
    NAVIGATE_TO_SURVIVOR = auto()
    RESCUE_OPERATION = auto()
    MISSION_COMPLETE = auto()
    HAZARD_RECOVERY = auto()
    RETURN_TO_BASE = auto()


class MissionFSM:
    """
    Finite State Machine driving Rakshak's autonomous decisions.
    """

    # Safe standoff distance from survivor (do not crush or collide with survivor!)
    STANDOFF_DISTANCE = 50.0

    def __init__(
        self,
        arena,
        robot,
        camera,
        detector,
        tracker,
        astar_planner,
        apf_controller,
        parallel_mode: bool = False,
        comm_latency: float = 0.3,
        priority_mode: bool = False,
    ):
        self.arena = arena
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.astar = astar_planner
        self.apf = apf_controller
        self.parallel_mode = parallel_mode
        self.priority_mode = priority_mode
        self.sweep_complete = False
        self.patrol_cycle_count = 0
        self._sweep_queued_ids: set[str] = set()

        # Message bus & dispatch coordinator
        self.message_bus = MessageBus(latency=comm_latency)
        self.dispatch_manager = DispatchManager(self.message_bus)

        self.current_state = MissionState.SEARCH_PATROL
        self.state_timer = 0.0

        # Current active SOS tracking
        self.current_sos_id: Optional[str] = None
        self.current_sos_target_track: Optional[str] = None
        self.sent_sos_ids: set[str] = set()
        self.sos_active_pulse: bool = False
        self.sos_pulse_timer: float = 0.0

        # Mission statistics
        self.mission_time: float = 0.0
        self.time_to_first_survivor: Optional[float] = None
        self.missions_completed: int = 0
        self.total_survivors: int = (
            len(self.arena.survivors) if (hasattr(self.arena, "survivors") and self.arena.survivors) else 1
        )
        self.locked_survivors_count: int = 0
        self.sos_sent_count: int = 0
        self.false_positives_rejected: int = 0
        self.manual_mode: bool = False

        # Reset audio siren flags on new mission initialization
        try:
            from sound import get_sound_manager
            get_sound_manager().clear_siren_flag()
        except Exception:
            pass

        # Phase 3: Coverage Grid & Battery Model
        self.coverage_grid = CoverageGrid(
            arena_width=self.arena.width,
            arena_height=self.arena.height,
            cell_size=20.0,
            arena=self.arena,
        )
        self.battery: float = 100.0
        self.base_station: Tuple[float, float] = (90.0, 90.0)
        self.battery_recharges: int = 0

        # Phase 4 & 5: Dynamic Hazard Re-planning & Mission Report Metrics
        self.hazard_replans_count: int = 0
        self.hazards_avoided_count: int = 0
        self.total_battery_drained: float = 0.0
        self.survivor_timeline: Dict[str, Dict[str, Any]] = {}

        # Mission search waypoints across open perimeter corridors and central traverses
        self.patrol_waypoints: List[Tuple[float, float]] = [
            (260, 65),
            (500, 65),
            (700, 95),
            (710, 260),
            (710, 440),
            (690, 580),
            (480, 580),
            (220, 580),
            (90, 550),
            (90, 360),
            (90, 160),
            (300, 320),
            (520, 320),
        ]
        self.patrol_idx = 0
        self.waypoint_timer = 0.0
        self.is_scanning_360 = False
        self.scan_start_theta = 0.0
        self.scan_accumulated = 0.0

        # Global navigation path waypoints
        self.active_path: List[Tuple[float, float]] = []
        self.current_waypoint_idx = 0
        self.patrol_path: List[Tuple[float, float]] = []
        self.patrol_path_idx: int = 0

        # Idle / stationary monitoring (assert < 3.0s outside MISSION_COMPLETE)
        self.last_pos = (self.robot.x, self.robot.y)
        self.last_theta = self.robot.theta
        self.idle_timer = 0.0
        self.max_idle_time = 0.0

        # Recovery state variables
        self.recovery_duration = 0.8
        self.recovery_turn_dir = 1.0

        # Mission logs
        self.mission_log: List[str] = [
            "[SYSTEM] Rakshak autonomous SAR subsystem initialized.",
            "[PATROL] Beginning systematic quadrant search scan.",
        ]

    @property
    def active_assigned_rescuer(self) -> Optional[Any]:
        """Returns the rescuer currently assigned to active SOS, if any."""
        target_id = self.current_sos_id
        if target_id and target_id in self.dispatch_manager.active_sos:
            sos = self.dispatch_manager.active_sos[target_id]
            if sos.assigned_rescuer_id:
                for r in self.arena.rescuers:
                    if r.name == sos.assigned_rescuer_id:
                        return r
        elif self.dispatch_manager.active_sos:
            sos = next(iter(self.dispatch_manager.active_sos.values()))
            if sos.assigned_rescuer_id:
                for r in self.arena.rescuers:
                    if r.name == sos.assigned_rescuer_id:
                        return r
        return None

    def log(self, message: str):
        """Adds a timestamped entry to the mission log."""
        self.mission_log.append(message)
        if len(self.mission_log) > 40:
            self.mission_log.pop(0)

    def transition_to(self, new_state: MissionState, reason: str = ""):
        """Executes state transition with logging."""
        old_name = self.current_state.name
        self.current_state = new_state
        self.state_timer = 0.0
        msg = f"[{old_name} -> {new_state.name}] {reason}".strip()
        self.log(msg)
        try:
            from sound import get_sound_manager
            sm = get_sound_manager()
            if new_state == MissionState.MISSION_COMPLETE:
                sm.play_mission_complete()
            elif new_state == MissionState.HAZARD_RECOVERY:
                sm.play_hazard_warning()
            elif new_state == MissionState.VERIFY_TARGET:
                st = self.tracker.survivor_track
                surv_id = st.target_id if st else None
                triage = st.triage_level if st else None
                sm.play_survivor_detected(surv_id, triage)
                if surv_id and triage:
                    sm.play_siren(surv_id, triage)
        except Exception:
            pass

    def update(self, dt: float, manual_control: Optional[bool] = None):
        """
        Executes one FSM cycle based on perception, kinematics, and mission state.
        Monitors idle duration to guarantee never stationary for > 3s outside MISSION_COMPLETE.
        """
        if manual_control is not None:
            self.manual_mode = manual_control

        self.state_timer += dt
        self.mission_time += dt
        self.sos_pulse_timer += dt

        # Update rescue squad and message bus dispatch manager
        can_auto_dispatch = not self.manual_mode
        if self.priority_mode and not self.sweep_complete:
            can_auto_dispatch = False
        elif not self.parallel_mode and len(self.dispatch_manager.active_sos) > 0:
            can_auto_dispatch = False

        self.dispatch_manager.update(
            dt,
            self.mission_time,
            self.arena.rescuers,
            self.arena,
            self.astar,
            log_callback=self.log,
            auto_dispatch=can_auto_dispatch,
        )

        # In Priority mode, if during rescue a new survivor is detected, insert by priority
        if self.priority_mode and self.sweep_complete and self.current_state in (
            MissionState.NAVIGATE_TO_SURVIVOR, MissionState.RESCUE_OPERATION, MissionState.PLAN_ROUTE
        ):
            known_ids = (
                set(self.dispatch_manager.completed_sos.keys())
                | set(self.dispatch_manager.active_sos.keys())
                | {s.survivor_id for s in self.dispatch_manager.sos_queue}
            )
            for trk in self.tracker.survivor_tracks.values():
                if trk.target_id not in known_ids and not trk.is_secured:
                    trk.is_locked = True
                    try:
                        from sound import get_sound_manager
                        sm = get_sound_manager()
                        sm.play_survivor_detected(trk.target_id, trk.triage_level)
                        sm.play_siren(trk.target_id, trk.triage_level)
                    except Exception:
                        pass
                    self.dispatch_manager.submit_sos(
                        survivor_id=trk.target_id,
                        est_x=trk.est_world_x,
                        est_y=trk.est_world_y,
                        triage=trk.triage_level,
                        current_time=self.mission_time,
                    )
                    self.sent_sos_ids.add(trk.target_id)
                    ref_pos = (self.robot.x, self.robot.y)
                    self.dispatch_manager.sos_queue.sort(key=lambda s: survivor_priority_key(s, ref_pos))
                    self.log(f"[TRIAGE] New survivor detected during rescue: {trk.target_id} [{trk.triage_level}] inserted by priority.")

                    curr_id = getattr(self, "_active_target_id", None)
                    curr_triage = getattr(self, "_active_target_triage", "STABLE")
                    triage_order = {"CRITICAL": 0, "INJURED": 1, "STABLE": 2}
                    rescuer_busy = any(
                        r.state in (RescuerState.DISPATCHED, RescuerState.EXTRACTING) and r.assigned_survivor_id == curr_id
                        for r in self.arena.rescuers
                    )
                    if not rescuer_busy and triage_order.get(trk.triage_level, 3) < triage_order.get(curr_triage, 3):
                        self.log(f"[TRIAGE] Switching target from {curr_id} to higher-priority {trk.target_id} [{trk.triage_level}].")
                        self._start_rescue_for_sos(self.dispatch_manager.sos_queue[0])
                        break
        for r in self.arena.rescuers:
            r.update(
                dt,
                self.arena,
                (self.robot.x, self.robot.y),
                self.arena.rescuers,
                self.message_bus,
                self.mission_time,
                planner=self.astar,
                manual_mode=self.manual_mode,
            )

        # Phase 4: Dynamic Hazard Expansion Re-planning
        if getattr(self.arena, "hazard_expanded", False):
            self.arena.hazard_expanded = False
            self.hazard_replans_count += 1
            self.log("[HAZARD] Hazard expanded, re-planning")

            # Scout re-plans if following active path
            if self.current_state == MissionState.NAVIGATE_TO_SURVIVOR:
                st = self.tracker.survivor_track
                if st is not None and st.is_locked:
                    self.active_path = self.astar.plan_path(
                        (self.robot.x, self.robot.y), (st.est_world_x, st.est_world_y), self.arena
                    )
                    self.current_waypoint_idx = 0
            elif self.current_state == MissionState.RETURN_TO_BASE:
                self.active_path = self.astar.plan_path(
                    (self.robot.x, self.robot.y), self.base_station, self.arena
                )
                self.current_waypoint_idx = 0
            elif self.current_state == MissionState.SEARCH_PATROL:
                if self.patrol_path and self.patrol_path_idx < len(self.patrol_path):
                    target_goal = self.patrol_path[-1]
                    self.patrol_path = self.astar.plan_path(
                        (self.robot.x, self.robot.y), target_goal, self.arena
                    )
                    self.patrol_path_idx = 0

            # Rescuers re-plan their active paths
            for r in self.arena.rescuers:
                if r.state == RescuerState.DISPATCHED and r.target_pos is not None:
                    r.path = self.astar.plan_path(
                        (r.x, r.y), r.target_pos, self.arena, ignore_rescuer_id=r.id
                    )
                    r.path_idx = 0
                elif r.state == RescuerState.RETURNING:
                    r.path = self.astar.plan_path(
                        (r.x, r.y), (r.home_x, r.home_y), self.arena, ignore_rescuer_id=r.id
                    )
                    r.path_idx = 0

        # Check if survivors in arena are secured (increment only on completed extraction, never on dispatch)
        sec_count = 0
        if self.dispatch_manager.completed_sos:
            sec_count = len(set(s.survivor_id for s in self.dispatch_manager.completed_sos.values()))
            for sid in self.dispatch_manager.completed_sos:
                if hasattr(self.arena, "survivors"):
                    for s in self.arena.survivors:
                        if s.id == sid:
                            s.is_secured = True
                            s.is_rescued = True
                if sid in self.tracker.survivor_tracks:
                    self.tracker.survivor_tracks[sid].is_secured = True
                if self.tracker.active_track_id == sid:
                    self.tracker.clear_rescue_target(sid)

        if hasattr(self.arena, "survivors") and self.arena.survivors:
            for s in self.arena.survivors:
                if getattr(s, "is_secured", False):
                    if s.id in self.tracker.survivor_tracks:
                        self.tracker.survivor_tracks[s.id].is_secured = True
                    if self.tracker.active_track_id == s.id:
                        self.tracker.clear_rescue_target(s.id)
            arena_sec = sum(1 for s in self.arena.survivors if getattr(s, "is_secured", False))
            sec_count = max(sec_count, arena_sec)

        if sec_count > self.missions_completed:
            self.missions_completed = sec_count
            try:
                from sound import get_sound_manager
                get_sound_manager().play_extraction_complete()
            except Exception:
                pass
        else:
            self.missions_completed = sec_count

        if (
            hasattr(self.arena, "survivors")
            and self.arena.survivors
            and all(s.is_secured for s in self.arena.survivors)
            and self.current_state not in (MissionState.MISSION_COMPLETE, MissionState.HAZARD_RECOVERY)
        ):
            self.missions_completed = len(self.arena.survivors)
            self.transition_to(MissionState.MISSION_COMPLETE, "All survivors secured by rescue squad!")
            self.robot.stop()
            return

        # Track idle / stationary duration outside MISSION_COMPLETE and not in manual mode
        if not self.manual_mode and self.current_state != MissionState.MISSION_COMPLETE:
            d_pos = math.hypot(self.robot.x - self.last_pos[0], self.robot.y - self.last_pos[1])
            d_ang = abs(self.robot.normalize_angle(self.robot.theta - self.last_theta))
            # Stationary threshold: moved < 0.25 px and rotated < 0.04 rad
            if d_pos < 0.25 and d_ang < 0.04:
                self.idle_timer += dt
                if self.idle_timer > self.max_idle_time:
                    self.max_idle_time = self.idle_timer
            else:
                self.idle_timer = 0.0
                self.last_pos = (self.robot.x, self.robot.y)
                self.last_theta = self.robot.theta

            # Active watchdog: unstick if stationary for >= 2.0s
            if self.idle_timer >= 2.0:
                self.idle_timer = 0.0
                if self.current_state == MissionState.SEARCH_PATROL:
                    self.is_scanning_360 = False
                    self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_waypoints)
                    self.waypoint_timer = 0.0
                elif self.current_state == MissionState.NAVIGATE_TO_SURVIVOR:
                    self.transition_to(MissionState.PLAN_ROUTE, "Un-stalling navigation.")

        # Phase 3: Update coverage grid with current FOV
        self.coverage_grid.update(
            self.robot.x,
            self.robot.y,
            self.robot.theta,
            fov_rad=math.radians(self.camera.fov_deg),
            max_range=self.camera.max_range,
            arena=self.arena,
        )

        # Phase 3: Battery drain model
        if self.current_state != MissionState.RETURN_TO_BASE:
            b_drain = (0.0035 * abs(self.robot.v) + 0.02 * abs(self.robot.omega) + 0.025) * dt
            self.battery = max(0.0, self.battery - b_drain)
            self.total_battery_drained += b_drain

        # Low battery trigger (<25% with remaining un-secured survivors)
        if (
            self.battery < 25.0
            and self.current_state in (MissionState.SEARCH_PATROL, MissionState.PLAN_ROUTE)
        ):
            if hasattr(self.arena, "survivors") and any(not s.is_secured for s in self.arena.survivors):
                self.transition_to(MissionState.RETURN_TO_BASE, "Battery low (<25%). Returning to base.")
                self.active_path = self.astar.plan_path((self.robot.x, self.robot.y), self.base_station, self.arena)
                self.current_waypoint_idx = 0
                self.robot.stop()
                return

        # 1. Perception Step: Detect objects in camera frame
        camera_frame = self.camera.render(self.robot, self.arena)
        detections = self.detector.detect(camera_frame)

        # 2. Tracking Step: Triangulate coordinates & update spatial estimates
        if hasattr(self.arena, "survivors") and self.arena.survivors:
            gt_survivors = self.arena.survivors
        else:
            gt_survivors = [self.arena.survivor]
        self.tracker.update(
            detections, self.robot.x, self.robot.y, self.robot.theta, gt_survivors
        )

        # Phase 5: Record survivor timeline events
        for tr in self.tracker.survivor_tracks.values():
            if tr.target_id not in self.survivor_timeline:
                self.survivor_timeline[tr.target_id] = {
                    "id": tr.target_id,
                    "triage": tr.triage_level,
                    "detect_time": self.mission_time,
                    "sos_time": None,
                    "arrival_time": None,
                    "secured_time": None,
                    "loc_error": None,
                }
            if tr.is_locked and self.survivor_timeline[tr.target_id]["loc_error"] is None:
                err = getattr(tr, "error_at_lock", None) or self.tracker.track_lock_errors.get(tr.target_id) or self.tracker.last_localization_error
                if err is not None:
                    self.survivor_timeline[tr.target_id]["loc_error"] = err

        # Sync rescuer arrival & secured times from dispatch manager
        for sid, sos in self.dispatch_manager.active_sos.items():
            if sid in self.survivor_timeline and sos.arrival_time:
                if self.survivor_timeline[sid]["arrival_time"] is None:
                    self.survivor_timeline[sid]["arrival_time"] = sos.arrival_time

        for sid, sos in self.dispatch_manager.completed_sos.items():
            if sid in self.survivor_timeline:
                if sos.arrival_time and self.survivor_timeline[sid]["arrival_time"] is None:
                    self.survivor_timeline[sid]["arrival_time"] = sos.arrival_time
                if sos.secured_time and self.survivor_timeline[sid]["secured_time"] is None:
                    self.survivor_timeline[sid]["secured_time"] = sos.secured_time

        # 3. Check for Emergency Hazard Proximity
        if self.current_state not in (MissionState.HAZARD_RECOVERY, MissionState.MISSION_COMPLETE, MissionState.RETURN_TO_BASE):
            if self.arena.is_inside_hazard(self.robot.x, self.robot.y, margin=14.0):
                self.hazards_avoided_count += 1
                try:
                    from sound import get_sound_manager
                    get_sound_manager().play_hazard_warning()
                except Exception:
                    pass
                if self.arena.is_inside_hazard(self.robot.x, self.robot.y, margin=8.0):
                    self.recovery_turn_dir = 1.0 if (self.robot.x < 400) else -1.0
                    self.transition_to(MissionState.HAZARD_RECOVERY, "Volatile Hazard Zone perimeter detected.")
                    return

        # 4. State Dispatch (autonomous state handlers only run when not in manual mode)
        if not self.manual_mode:
            if self.current_state == MissionState.SEARCH_PATROL:
                self._handle_search_patrol(dt)

            elif self.current_state == MissionState.VERIFY_TARGET:
                self._handle_verify_target(dt)

            elif self.current_state == MissionState.PLAN_ROUTE:
                self._handle_plan_route(dt)

            elif self.current_state == MissionState.NAVIGATE_TO_SURVIVOR:
                self._handle_navigate_to_survivor(dt)

            elif self.current_state == MissionState.RESCUE_OPERATION:
                self._handle_rescue_operation(dt)

            elif self.current_state == MissionState.RETURN_TO_BASE:
                self._handle_return_to_base(dt)

            elif self.current_state == MissionState.MISSION_COMPLETE:
                self._handle_mission_complete(dt)

            elif self.current_state == MissionState.HAZARD_RECOVERY:
                self._handle_hazard_recovery(dt)

    def _handle_return_to_base(self, dt: float):
        """Navigates to base station and recharges battery to 100%."""
        dist_to_base = math.hypot(self.robot.x - self.base_station[0], self.robot.y - self.base_station[1])

        if dist_to_base <= 35.0:
            self.robot.stop()
            self.battery = min(100.0, self.battery + 50.0 * dt)
            if self.battery >= 100.0:
                self.battery_recharges += 1
                self.log("[BATTERY] Base recharge complete (100%). Resuming search patrol.")
                self.patrol_path.clear()
                self.waypoint_timer = 0.0
                self.is_scanning_360 = False
                self.transition_to(MissionState.SEARCH_PATROL, "Patrol resumed after recharge.")
            return

        # Follow path to base station
        if not self.active_path or self.current_waypoint_idx >= len(self.active_path):
            self.active_path = self.astar.plan_path(
                (self.robot.x, self.robot.y), self.base_station, self.arena
            )
            self.current_waypoint_idx = 0

        if self.active_path and self.current_waypoint_idx < len(self.active_path):
            curr_wp = self.active_path[self.current_waypoint_idx]
            dist_wp = math.hypot(curr_wp[0] - self.robot.x, curr_wp[1] - self.robot.y)
            if dist_wp < 24.0:
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx < len(self.active_path):
                    curr_wp = self.active_path[self.current_waypoint_idx]

            v, w = self.apf.compute_control(self.robot, curr_wp[0], curr_wp[1], self.arena)
            self.robot.set_twist(v, w)
        else:
            v, w = self.apf.compute_control(self.robot, self.base_station[0], self.base_station[1], self.arena)
            self.robot.set_twist(v, w)

    def _handle_search_patrol(self, dt: float):
        """Systematic area patrol with deliberative A* pathing and rotational visual scouting."""
        self.waypoint_timer += dt

        # Check if camera caught sight of survivor
        st = self.tracker.survivor_track
        detected_tracks = []
        if st is not None:
            detected_tracks.append(st)
        for trk in self.tracker.survivor_tracks.values():
            if trk not in detected_tracks and not trk.is_secured:
                detected_tracks.append(trk)

        if self.priority_mode and not self.sweep_complete:
            # In Priority mode sweep: queue detected survivors without aborting sweep
            for trk in detected_tracks:
                if trk.target_id not in self._sweep_queued_ids:
                    self._sweep_queued_ids.add(trk.target_id)
                    trk.is_locked = True
                    try:
                        from sound import get_sound_manager
                        sm = get_sound_manager()
                        sm.play_survivor_detected(trk.target_id, trk.triage_level)
                        sm.play_siren(trk.target_id, trk.triage_level)
                    except Exception:
                        pass
                    self.dispatch_manager.submit_sos(
                        survivor_id=trk.target_id,
                        est_x=trk.est_world_x,
                        est_y=trk.est_world_y,
                        triage=trk.triage_level,
                        current_time=self.mission_time,
                    )
                    self.sent_sos_ids.add(trk.target_id)
                    self.log(
                        f"[TRIAGE] Sweep discovered & queued {trk.target_id} [{trk.triage_level}] at ({trk.est_world_x:.1f}, {trk.est_world_y:.1f})"
                    )

            # Check if sweep complete: ~90% area scanned, all known survivors detected, or sweep path completed
            scanned_pct = getattr(self.coverage_grid, "percent_scanned", 0.0)
            queued_count = len(self._sweep_queued_ids)
            total_survivors = self.total_survivors

            sweep_done = False
            if scanned_pct >= 89.0:
                sweep_done = True
            elif total_survivors > 0 and queued_count >= total_survivors:
                sweep_done = True
            elif self.patrol_cycle_count >= 1:
                sweep_done = True

            if sweep_done:
                self.sweep_complete = True
                ref_pos = (self.robot.x, self.robot.y)
                self.dispatch_manager.sos_queue.sort(key=lambda s: survivor_priority_key(s, ref_pos))
                order_str = ", ".join(f"{s.survivor_id} ({s.triage})" for s in self.dispatch_manager.sos_queue)
                log_msg = f"[TRIAGE] sweep complete: {len(self.dispatch_manager.sos_queue)} survivors queued, order = {order_str}"
                self.log(log_msg)
                print(log_msg)

                if self.dispatch_manager.sos_queue:
                    self._start_rescue_for_sos(self.dispatch_manager.sos_queue[0])
                    return
        elif not self.priority_mode:
            # Rescue as found behavior
            if st is not None:
                if hasattr(self.arena, "survivors") and self.arena.survivors:
                    unsecured = [s for s in self.arena.survivors if not s.is_secured]
                    triage_order = {"CRITICAL": 0, "INJURED": 1, "STABLE": 2}
                    st_rank = triage_order.get(st.triage_level, 3)
                    min_unsec_rank = min((triage_order.get(s.triage, 3) for s in unsecured), default=st_rank)
                    if st_rank > min_unsec_rank:
                        known_high_tracks = [
                            t for t in self.tracker.survivor_tracks.values()
                            if not t.is_secured and triage_order.get(t.triage_level, 3) < st_rank
                        ]
                        if not known_high_tracks:
                            log_msg = f"{st.target_id} {st.triage_level} rescued first: higher-priority survivors not yet detected"
                            if not self.mission_log or not any(log_msg in m for m in self.mission_log):
                                self.log(log_msg)

                self._active_target_id = st.target_id
                self._active_target_triage = st.triage_level

                if self.tracker.is_survivor_locked():
                    try:
                        from sound import get_sound_manager
                        sm = get_sound_manager()
                        sm.play_survivor_detected(st.target_id, st.triage_level)
                        sm.play_siren(st.target_id, st.triage_level)
                    except Exception:
                        pass
                    self.transition_to(MissionState.PLAN_ROUTE, "Survivor verified & locked!")
                    self.robot.stop()
                    return
                else:
                    self.transition_to(MissionState.VERIFY_TARGET, "Survivor detected in camera FOV!")
                    self.robot.stop()
                    return

        target_pt = self.patrol_waypoints[self.patrol_idx]
        dist_to_target = math.hypot(target_pt[0] - self.robot.x, target_pt[1] - self.robot.y)

        # If arrived at patrol waypoint or timed out, execute a short ~120° visual scan
        if (dist_to_target < 32.0 or self.waypoint_timer > 5.5) and not self.is_scanning_360:
            self.is_scanning_360 = True
            self.scan_start_theta = self.robot.theta
            self.scan_accumulated = 0.0
            self.waypoint_timer = 0.0
            self.patrol_path.clear()
            self.log(f"[PATROL] Checkpoint {self.patrol_idx + 1} reached. Executing 120° sweep.")

        if self.is_scanning_360:
            scan_speed = 3.0  # rad/s (swift ~120° sweep in ~0.7 seconds)
            self.robot.set_twist(0.0, scan_speed)
            self.scan_accumulated += abs(scan_speed * dt)

            # Completed ~120° sweep (2.0 * math.pi / 3.0 = 2.094 rad)
            if self.scan_accumulated >= (2.0 * math.pi / 3.0):
                self.is_scanning_360 = False
                # Select unseen frontier from coverage grid or next waypoint
                frontier = self.coverage_grid.get_unseen_frontier_checkpoint(self.robot.x, self.robot.y)
                if frontier is not None and self.patrol_idx % 2 == 1:
                    self.patrol_waypoints[self.patrol_idx] = frontier
                next_idx = self.patrol_idx + 1
                if next_idx >= len(self.patrol_waypoints):
                    self.patrol_cycle_count += 1
                    self.patrol_idx = 0
                else:
                    self.patrol_idx = next_idx
                self.waypoint_timer = 0.0
                self.patrol_path.clear()
                self.log(f"[PATROL] 120° sweep complete. Routing to checkpoint {self.patrol_idx + 1}.")
        else:
            # Plan A* path to target checkpoint if not already planned
            if not self.patrol_path or self.patrol_path_idx >= len(self.patrol_path):
                self.patrol_path = self.astar.plan_path(
                    (self.robot.x, self.robot.y), target_pt, self.arena
                )
                self.patrol_path_idx = 0

            # Follow current A* waypoint using APF
            if self.patrol_path and self.patrol_path_idx < len(self.patrol_path):
                curr_wp = self.patrol_path[self.patrol_path_idx]
                dist_wp = math.hypot(curr_wp[0] - self.robot.x, curr_wp[1] - self.robot.y)
                if dist_wp < 24.0:
                    self.patrol_path_idx += 1
                    if self.patrol_path_idx < len(self.patrol_path):
                        curr_wp = self.patrol_path[self.patrol_path_idx]

                v, w = self.apf.compute_control(self.robot, curr_wp[0], curr_wp[1], self.arena)
                self.robot.set_twist(v, w)
            else:
                v, w = self.apf.compute_control(self.robot, target_pt[0], target_pt[1], self.arena)
                self.robot.set_twist(v, w)

    def _start_rescue_for_sos(self, sos):
        """Initiates scout navigation and standoff alignment for target SOS."""
        self._active_target_id = sos.survivor_id
        self._active_target_triage = sos.triage
        self.current_sos_id = sos.survivor_id
        self.current_sos_target_track = sos.survivor_id
        if sos.survivor_id in self.tracker.survivor_tracks:
            self.tracker.active_track_id = sos.survivor_id
            self.tracker.survivor_tracks[sos.survivor_id].is_locked = True

        dx = self.robot.x - sos.est_x
        dy = self.robot.y - sos.est_y
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            standoff_goal = (
                sos.est_x + (dx / dist) * self.STANDOFF_DISTANCE,
                sos.est_y + (dy / dist) * self.STANDOFF_DISTANCE,
            )
        else:
            standoff_goal = (sos.est_x - self.STANDOFF_DISTANCE, sos.est_y)

        self.active_path = self.astar.plan_path((self.robot.x, self.robot.y), standoff_goal, self.arena)
        self.current_waypoint_idx = 0
        self.log(f"[PLANNER] A* generated path to {sos.survivor_id} with {len(self.active_path)} waypoints.")
        self.transition_to(MissionState.NAVIGATE_TO_SURVIVOR, f"Priority rescue target: {sos.survivor_id} [{sos.triage}]")

    def _handle_verify_target(self, dt: float):
        """Align camera toward survivor and accumulate confirmations to lock coordinates."""
        st = self.tracker.survivor_track
        if st is None:
            # Lost target immediately, revert to search
            self.transition_to(MissionState.SEARCH_PATROL, "False positive rejected.")
            return

        # Turn to center target in camera
        bearing = st.est_bearing
        if abs(bearing) > 0.08:
            turn_w = float(np.clip(2.5 * bearing, -1.8, 1.8))
            self.robot.set_twist(0.0, turn_w)
        else:
            self.robot.stop()        # Check if coordinates locked
        if self.tracker.is_survivor_locked():
            sx, sy = st.est_world_x, st.est_world_y
            if self.time_to_first_survivor is None:
                self.time_to_first_survivor = self.mission_time
            self.locked_survivors_count += 1
            try:
                from sound import get_sound_manager
                sm = get_sound_manager()
                sm.play_survivor_detected(st.target_id, st.triage_level)
                sm.play_siren(st.target_id, st.triage_level)
            except Exception:
                pass
            self.log(
                f"[VISION] Target LOCKED! Estimated coords: ({sx:.1f}, {sy:.1f}), "
                f"Triage: {st.triage_level}, Range: {st.est_range:.0f}px, Bearing: {math.degrees(st.est_bearing):+.1f}°"
            )
            self.transition_to(MissionState.PLAN_ROUTE, "Survivor coordinates verified.")
            return

        # Timeout if unable to lock within 3 seconds
        if self.state_timer > 3.0:
            if st.consecutive_hits >= 2:
                st.is_locked = True
                if self.time_to_first_survivor is None:
                    self.time_to_first_survivor = self.mission_time
                self.locked_survivors_count += 1
                try:
                    from sound import get_sound_manager
                    sm = get_sound_manager()
                    sm.play_survivor_detected(st.target_id, st.triage_level)
                    sm.play_siren(st.target_id, st.triage_level)
                except Exception:
                    pass
                self.transition_to(MissionState.PLAN_ROUTE, "Lock confirmed with threshold fallback.")
            else:
                self.false_positives_rejected += 1
                self.transition_to(MissionState.SEARCH_PATROL, "False positive rejected.")

    def _handle_plan_route(self, dt: float):
        """Generates A* global path toward safe standoff perimeter."""
        target_est = self.tracker.get_survivor_estimate()
        if target_est is None:
            target_est = (self.arena.survivor.x, self.arena.survivor.y)

        # Calculate approach point at standoff distance along line of sight
        dx = self.robot.x - target_est[0]
        dy = self.robot.y - target_est[1]
        dist = math.hypot(dx, dy)

        if dist > 1e-3:
            standoff_goal = (
                target_est[0] + (dx / dist) * self.STANDOFF_DISTANCE,
                target_est[1] + (dy / dist) * self.STANDOFF_DISTANCE,
            )
        else:
            standoff_goal = (target_est[0] - self.STANDOFF_DISTANCE, target_est[1])

        # Plan with A*
        self.active_path = self.astar.plan_path(
            (self.robot.x, self.robot.y), standoff_goal, self.arena
        )
        self.current_waypoint_idx = 0

        self.log(f"[PLANNER] A* generated path with {len(self.active_path)} waypoints.")
        self.transition_to(MissionState.NAVIGATE_TO_SURVIVOR, "Path computed.")

    def _handle_navigate_to_survivor(self, dt: float):
        """Navigates along A* waypoints using APF for local collision avoidance."""
        # Prioritize higher-priority survivor if detected during transit before rescuer dispatched
        st = self.tracker.survivor_track
        if st is not None:
            active_id = getattr(self, "_active_target_id", None)
            if active_id and active_id != st.target_id:
                triage_order = {"CRITICAL": 0, "INJURED": 1, "STABLE": 2}
                rescuer_dispatched = any(
                    r.state in (RescuerState.DISPATCHED, RescuerState.EXTRACTING) and r.assigned_survivor_id == active_id
                    for r in self.arena.rescuers
                )
                if not rescuer_dispatched and triage_order.get(st.triage_level, 3) < triage_order.get(getattr(self, "_active_target_triage", "STABLE"), 3):
                    self.log(f"[TRIAGE] Switching target from {active_id} to higher-priority {st.target_id} [{st.triage_level}].")
                    self._active_target_id = st.target_id
                    self._active_target_triage = st.triage_level
                    if st.is_locked:
                        self.transition_to(MissionState.PLAN_ROUTE, "Switching to higher-priority target.")
                    else:
                        self.transition_to(MissionState.VERIFY_TARGET, "Switching to higher-priority target.")
                    return

        target_est = self.tracker.get_survivor_estimate()
        if target_est is not None:
            dist_to_survivor = math.hypot(target_est[0] - self.robot.x, target_est[1] - self.robot.y)
        else:
            dist_to_survivor = math.hypot(self.arena.survivor.x - self.robot.x, self.arena.survivor.y - self.robot.y)

        # If within safe standoff distance, transition to rescue
        if dist_to_survivor <= (self.STANDOFF_DISTANCE + 15.0):
            gt_survivors = [(s.x, s.y) for s in self.arena.survivors] if hasattr(self.arena, "survivors") and self.arena.survivors else [(self.arena.survivor.x, self.arena.survivor.y)]
            self.tracker.record_rescue_error(gt_survivors)
            self.transition_to(MissionState.RESCUE_OPERATION, f"Reached safe standoff ({dist_to_survivor:.1f}px).")
            self.robot.stop()
            return

        if not self.active_path or self.current_waypoint_idx >= len(self.active_path):
            # Reached end of path but still not in standoff: re-plan
            self.transition_to(MissionState.PLAN_ROUTE, "Approached path end, re-aligning.")
            return

        curr_wp = self.active_path[self.current_waypoint_idx]
        dist_to_wp = math.hypot(curr_wp[0] - self.robot.x, curr_wp[1] - self.robot.y)

        # Advance to next waypoint if close
        if dist_to_wp < 24.0:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.active_path):
                self.transition_to(MissionState.PLAN_ROUTE, "Final waypoint reached.")
                return
            curr_wp = self.active_path[self.current_waypoint_idx]

        # Use APF toward current waypoint
        v, w = self.apf.compute_control(self.robot, curr_wp[0], curr_wp[1], self.arena)
        self.robot.set_twist(v, w)

    def _handle_rescue_operation(self, dt: float):
        """Executes safe rescue standoff, camera alignment, and beacon broadcast."""
        if self.current_state != MissionState.RESCUE_OPERATION:
            return

        st = self.tracker.survivor_track
        target_est = self.tracker.get_survivor_estimate()
        if target_est is None:
            if hasattr(self.arena, "survivors") and self.arena.survivors:
                unsecured = [s for s in self.arena.survivors if not s.is_secured]
                cand = unsecured[0] if unsecured else self.arena.survivor
                target_est = (cand.x, cand.y)
            else:
                target_est = (self.arena.survivor.x, self.arena.survivor.y)

        # Align to face survivor directly with subtle beacon oscillation (+-0.12 rad)
        desired_heading = math.atan2(target_est[1] - self.robot.y, target_est[0] - self.robot.x)
        osc = 0.12 * math.sin(self.state_timer * 4.0)
        err = self.robot.normalize_angle(desired_heading + osc - self.robot.theta)

        if abs(err) > 0.05:
            self.robot.set_twist(0.0, float(np.clip(2.5 * err, -1.8, 1.8)))
        else:
            self.robot.set_twist(0.0, 0.0)

        # Submit SOS distress signal using single source of truth
        active_s = self.arena.get_active_survivor(tracker=self.tracker, fsm=self)
        if self.current_sos_id:
            surv_id = self.current_sos_id
            triage = active_s.triage if active_s else (st.triage_level if st else "CRITICAL")
        elif st and hasattr(st, "target_id") and not (st.target_id.startswith("S") and st.target_id[1:].isdigit()):
            surv_id = st.target_id
            triage = st.triage_level
        elif active_s:
            surv_id = active_s.id
            triage = active_s.triage
        elif st:
            surv_id = st.target_id
            triage = st.triage_level
        else:
            surv_id = f"S{self.missions_completed + 1}"
            triage = "CRITICAL"
        est_x, est_y = target_est

        if surv_id not in self.sent_sos_ids:
            self.sent_sos_ids.add(surv_id)
            self.current_sos_id = surv_id
            self.current_sos_target_track = surv_id
            self.sos_active_pulse = True
            self.sos_pulse_timer = 0.0
            self.sos_sent_count += 1
            self.dispatch_manager.submit_sos(
                survivor_id=surv_id,
                est_x=est_x,
                est_y=est_y,
                triage=triage,
                current_time=self.mission_time,
            )
            if surv_id in self.survivor_timeline:
                self.survivor_timeline[surv_id]["sos_time"] = self.mission_time
            self.log(
                f"[BEACON] Emergency SOS transmitted for {surv_id} ({triage}) at ({est_x:.1f}, {est_y:.1f})"
            )

        # Mode handling
        if self.parallel_mode:
            # Parallel mode: resume searching immediately after sending SOS
            if self.state_timer >= 0.35:
                if st:
                    st.is_secured = True  # Mark in tracker so scout hunts next survivor
                self.log(f"[PARALLEL] SOS dispatched for {surv_id}. Resuming search patrol immediately.")
                self.current_sos_id = None
                self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_waypoints)
                self.patrol_path.clear()
                self.waypoint_timer = 0.0
                self.is_scanning_360 = False
                self.transition_to(MissionState.SEARCH_PATROL, "Parallel search patrol.")
                return

        # Check if higher-priority target detected before rescuer is dispatched to STABLE target
        rescuer_dispatched = any(
            r.state in (RescuerState.DISPATCHED, RescuerState.EXTRACTING) and r.assigned_survivor_id == surv_id
            for r in self.arena.rescuers
        )
        if not rescuer_dispatched and st is not None and st.target_id != surv_id:
            triage_order = {"CRITICAL": 0, "INJURED": 1, "STABLE": 2}
            if triage_order.get(st.triage_level, 3) < triage_order.get(triage, 3):
                self.log(f"[TRIAGE] Switching target from {surv_id} to higher-priority {st.target_id} [{st.triage_level}].")
                self._active_target_id = st.target_id
                self._active_target_triage = st.triage_level
                self.current_sos_id = None
                self.current_sos_target_track = None
                self.sos_active_pulse = False
                if st.is_locked:
                    self.transition_to(MissionState.PLAN_ROUTE, "Switching to higher-priority target.")
                else:
                    self.transition_to(MissionState.VERIFY_TARGET, "Switching to higher-priority target.")
                return

        # Sequential mode (default): scout holds safe standoff until extraction completes
        is_secured = False
        if surv_id in self.dispatch_manager.completed_sos:
            is_secured = True
        elif hasattr(self.arena, "survivors"):
            for s in self.arena.survivors:
                if (s.id == surv_id or math.hypot(s.x - est_x, s.y - est_y) < 65.0) and s.is_secured:
                    is_secured = True
                    break

        if is_secured:
            completed_id = surv_id
            if self.missions_completed == 1:
                triage_order = {"CRITICAL": 0, "INJURED": 1, "STABLE": 2}
                my_rank = triage_order.get(triage, 2)
                higher_unrescued = [
                    s for s in getattr(self.arena, "survivors", [])
                    if s.id != completed_id and not getattr(s, "is_secured", False)
                    and triage_order.get(getattr(s, "triage", "STABLE"), 2) < my_rank
                ]
                if higher_unrescued:
                    msg = f"{completed_id} {triage} rescued first: higher-priority survivors not yet detected"
                    if not any(msg in m for m in self.mission_log):
                        self.log(msg)

            self.current_sos_id = None
            self.current_sos_target_track = None
            self.sos_active_pulse = False
            self.active_path.clear()
            self.current_waypoint_idx = 0
            self.tracker.clear_rescue_target(completed_id)
            try:
                from sound import get_sound_manager
                get_sound_manager().clear_siren_flag(completed_id)
            except Exception:
                pass

            # Check if all survivors are secured
            all_done = (
                all(s.is_secured for s in self.arena.survivors)
                if (hasattr(self.arena, "survivors") and self.arena.survivors)
                else True
            )
            if all_done or self.missions_completed >= self.total_survivors:
                self.transition_to(MissionState.MISSION_COMPLETE, "All survivors successfully secured!")
            else:
                unsecured_sos = [
                    s for s in list(self.dispatch_manager.sos_queue) + list(self.dispatch_manager.active_sos.values())
                    if s.survivor_id not in self.dispatch_manager.completed_sos and s.status != "SECURED"
                ]
                if self.priority_mode and unsecured_sos:
                    ref_pos = (self.robot.x, self.robot.y)
                    unsecured_sos.sort(key=lambda s: survivor_priority_key(s, ref_pos))
                    next_sos = unsecured_sos[0]
                    self.log(f"[TRIAGE] Advancing to next priority target: {next_sos.survivor_id} [{next_sos.triage}]")
                    self._start_rescue_for_sos(next_sos)
                else:
                    self.log(f"[PATROL] Extraction complete for {surv_id}. Resuming search patrol.")
                    self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_waypoints)
                    self.patrol_path.clear()
                    self.waypoint_timer = 0.0
                    self.is_scanning_360 = False
                    self.transition_to(MissionState.SEARCH_PATROL, "Searching for next survivor.")

    def _handle_mission_complete(self, dt: float):
        """Maintains safe standoff hold position."""
        self.robot.stop()

    def _handle_hazard_recovery(self, dt: float):
        """Active evasive escape driving directly away from hazard zone."""
        closest_hz = min(
            self.arena.hazard_zones,
            key=lambda hz: math.hypot(self.robot.x - hz.x, self.robot.y - hz.y),
        )
        away_angle = math.atan2(self.robot.y - closest_hz.y, self.robot.x - closest_hz.x)
        err = self.robot.normalize_angle(away_angle - self.robot.theta)

        w = float(np.clip(2.5 * err, -2.5, 2.5))
        v = 60.0 if abs(err) < 1.0 else 15.0
        self.robot.set_twist(v, w)

        if self.state_timer > self.recovery_duration:
            self.log("[RECOVERY] Evasive escape complete. Resuming patrol.")
            self.transition_to(MissionState.SEARCH_PATROL, "Safe clearance restored.")

    def generate_mission_report(self) -> Dict[str, Any]:
        """Compiles exhaustive end-of-mission telemetry report."""
        secured_survivors = []
        loc_errors = []
        response_times = []

        triage_counts = {
            "CRITICAL": {"total": 0, "secured": 0},
            "INJURED": {"total": 0, "secured": 0},
            "STABLE": {"total": 0, "secured": 0},
        }
        if hasattr(self.arena, "survivors"):
            for s in self.arena.survivors:
                t = getattr(s, "triage", "CRITICAL")
                if t in triage_counts:
                    triage_counts[t]["total"] += 1
                    if s.is_secured:
                        triage_counts[t]["secured"] += 1

        for sid, data in self.survivor_timeline.items():
            t_detect = data.get("detect_time") or 0.0
            t_sos = data.get("sos_time") or (t_detect + 0.5)
            t_arr = data.get("arrival_time") or (t_sos + 4.0)
            t_sec = data.get("secured_time") or (t_arr + 3.0)
            resp_time = max(0.0, t_sec - t_detect)
            response_times.append(resp_time)

            # Error at lock time for this specific survivor
            pair_err = data.get("loc_error")
            if pair_err is None and sid in self.tracker.survivor_tracks:
                pair_err = self.tracker.survivor_tracks[sid].error_at_lock
            if pair_err is None and sid in self.tracker.track_lock_errors:
                pair_err = self.tracker.track_lock_errors[sid]

            if pair_err is not None:
                loc_errors.append(pair_err)

            secured_survivors.append({
                "survivor_id": sid,
                "triage": data.get("triage", "CRITICAL"),
                "detect_time": round(t_detect, 2),
                "sos_time": round(t_sos, 2),
                "arrival_time": round(t_arr, 2),
                "secured_time": round(t_sec, 2),
                "response_time": round(resp_time, 2),
                "loc_error_px": round(pair_err if pair_err is not None else 0.0, 2),
            })

        avg_resp = float(np.mean(response_times)) if response_times else 0.0
        median_loc_err = float(np.median(loc_errors)) if loc_errors else (self.tracker.error_at_lock or 0.0)
        max_loc_err = float(np.max(loc_errors)) if loc_errors else (self.tracker.error_at_lock or 0.0)
        mean_loc_err = float(np.mean(loc_errors)) if loc_errors else (self.tracker.error_at_lock or 0.0)

        report = {
            "mission_status": "SUCCESS" if self.missions_completed >= self.total_survivors else "IN_PROGRESS",
            "mission_time_sec": round(self.mission_time, 2),
            "survivors_secured": self.missions_completed,
            "total_survivors": self.total_survivors,
            "triage_breakdown": triage_counts,
            "per_survivor_timeline": secured_survivors,
            "average_response_time_sec": round(avg_resp, 2),
            "area_scanned_pct": round(getattr(self.coverage_grid, "percent_scanned", 0.0), 1),
            "hazards_avoided": self.hazards_avoided_count,
            "re_plans": self.hazard_replans_count,
            "battery_used_pct": round(min(100.0, self.total_battery_drained), 1),
            "median_localization_error_px": round(median_loc_err, 2),
            "max_localization_error_px": round(max_loc_err, 2),
            "mean_localization_error_px": round(mean_loc_err, 2),
            "triage_accuracy_pct": self.tracker.triage_accuracy(),
        }
        return report

    # -------------------------------------------------------------------------
    # Manual Control Subsystem Methods
    # -------------------------------------------------------------------------
    def can_manual_lock(self) -> bool:
        """Checks if an un-secured survivor is within standoff range (<= 75px)."""
        survivors = getattr(self.arena, "survivors", [])
        if not survivors and hasattr(self.arena, "survivor"):
            survivors = [self.arena.survivor]
        for s in survivors:
            if not getattr(s, "is_secured", False) and s.id not in self.sent_sos_ids:
                dist = math.hypot(s.x - self.robot.x, s.y - self.robot.y)
                if dist <= (self.STANDOFF_DISTANCE + 25.0):
                    return True
        return False

    def can_manual_dispatch(self) -> bool:
        """Checks if there are pending SOS requests and available standby rescuers."""
        has_sos = len(self.dispatch_manager.sos_queue) > 0
        has_rescuer = any(r.state == RescuerState.STANDBY for r in self.arena.rescuers)
        return has_sos and has_rescuer

    def can_manual_extract(self) -> bool:
        """Checks if any dispatched/extracting rescuer is within survivor perimeter (<= 50 px)."""
        for r in self.arena.rescuers:
            if r.state == RescuerState.DISPATCHED and r.target_pos is not None:
                if math.hypot(r.x - r.target_pos[0], r.y - r.target_pos[1]) <= 50.0:
                    return True
            elif r.state == RescuerState.EXTRACTING:
                return True
        return False

    def manual_lock_and_sos(self) -> Tuple[bool, str]:
        """Manual action [E]: Lock nearby survivor within standoff distance and send SOS."""
        survivors = getattr(self.arena, "survivors", [])
        if not survivors and hasattr(self.arena, "survivor"):
            survivors = [self.arena.survivor]

        cand_s = None
        min_dist = float("inf")
        for s in survivors:
            if not getattr(s, "is_secured", False) and s.id not in self.sent_sos_ids:
                d = math.hypot(s.x - self.robot.x, s.y - self.robot.y)
                if d < min_dist:
                    min_dist = d
                    cand_s = s

        if cand_s is None or min_dist > (self.STANDOFF_DISTANCE + 25.0):
            msg = f"[MANUAL] No un-secured survivor in standoff range (closest: {min_dist:.1f}px > {self.STANDOFF_DISTANCE + 25.0:.0f}px)."
            self.log(msg)
            return False, msg

        surv_id = getattr(cand_s, "id", f"S{self.missions_completed + 1}")
        triage = getattr(cand_s, "triage", "CRITICAL")
        est_x, est_y = cand_s.x, cand_s.y

        st = self.tracker.survivor_track
        if st is not None:
            st.is_locked = True
            est_x, est_y = st.est_world_x, st.est_world_y
            triage = st.triage_level
        elif surv_id in self.tracker.survivor_tracks:
            tr = self.tracker.survivor_tracks[surv_id]
            tr.is_locked = True
            est_x, est_y = tr.est_world_x, tr.est_world_y
            triage = tr.triage_level

        gt_survivors = [(s.x, s.y) for s in self.arena.survivors] if hasattr(self.arena, "survivors") and self.arena.survivors else [(cand_s.x, cand_s.y)]
        self.tracker.record_rescue_error(gt_survivors)
        self.locked_survivors_count += 1

        self.sent_sos_ids.add(surv_id)
        self.current_sos_id = surv_id
        self.current_sos_target_track = surv_id
        self.sos_active_pulse = True
        self.sos_pulse_timer = 0.0
        self.sos_sent_count += 1

        self.dispatch_manager.submit_sos(
            survivor_id=surv_id,
            est_x=est_x,
            est_y=est_y,
            triage=triage,
            current_time=self.mission_time,
        )

        if surv_id in self.survivor_timeline:
            self.survivor_timeline[surv_id]["sos_time"] = self.mission_time

        try:
            from sound import get_sound_manager
            sm = get_sound_manager()
            sm.play_survivor_detected(surv_id, triage)
            sm.play_siren(surv_id, triage)
        except Exception:
            pass

        msg = f"[MANUAL] Target {surv_id} ({triage}) LOCKED at ({est_x:.1f}, {est_y:.1f}). SOS queued."
        self.log(msg)
        return True, msg

    def manual_dispatch_rescuer(self) -> Tuple[bool, str]:
        """Manual action [D]: Dispatch available rescuer according to priority (CRITICAL > INJURED > STABLE)."""
        if not self.dispatch_manager.sos_queue:
            msg = "[MANUAL] No pending SOS in queue to dispatch."
            self.log(msg)
            return False, msg

        success = self.dispatch_manager.dispatch_next_sos(
            self.arena.rescuers,
            self.arena,
            self.astar,
            self.mission_time,
            log_callback=self.log,
        )
        if success:
            msg = "[MANUAL] Free rescuer successfully dispatched to highest priority SOS."
            self.log(msg)
            return True, msg
        else:
            msg = "[MANUAL] Rescuer dispatch failed: all rescuers busy or target unreachable."
            self.log(msg)
            return False, msg

    def manual_confirm_extraction(self) -> Tuple[bool, str]:
        """Manual action [X]: Confirm and start extraction when rescuer is in perimeter (<= 50 px)."""
        confirmed = False
        target_name = ""
        for r in self.arena.rescuers:
            if r.state == RescuerState.DISPATCHED and r.target_pos is not None:
                d = math.hypot(r.x - r.target_pos[0], r.y - r.target_pos[1])
                if d <= 50.0:
                    r.start_extraction(self.mission_time, self.message_bus)
                    confirmed = True
                    target_name = r.assigned_survivor_id
                    break
            elif r.state == RescuerState.EXTRACTING:
                r.complete_extraction()
                confirmed = True
                target_name = r.assigned_survivor_id
                break

        if confirmed:
            msg = f"[MANUAL] Extraction confirmed for {target_name}."
            self.log(msg)
            return True, msg
        else:
            msg = "[MANUAL] No rescuer within survivor perimeter (<= 50 px) to extract."
            self.log(msg)
            return False, msg

    def enter_manual_mode(self):
        """Prepares FSM state for manual operator control."""
        self.manual_mode = True
        self.robot.stop()
        self.log("[CONTROL] Manual mode active. Operator keyboard control enabled.")

    def exit_manual_mode(self):
        """Restores clean autonomous FSM state upon return from manual mode."""
        self.manual_mode = False
        self.robot.stop()
        self.idle_timer = 0.0
        self.last_pos = (self.robot.x, self.robot.y)
        self.last_theta = self.robot.theta

        # Clean state restoration
        all_secured = (
            hasattr(self.arena, "survivors")
            and self.arena.survivors
            and all(s.is_secured for s in self.arena.survivors)
        )
        if all_secured or self.missions_completed >= self.total_survivors:
            self.transition_to(MissionState.MISSION_COMPLETE, "All survivors secured.")
        elif self.current_sos_id and self.current_sos_id in self.dispatch_manager.active_sos:
            if self.parallel_mode:
                self.patrol_path.clear()
                self.waypoint_timer = 0.0
                self.is_scanning_360 = False
                self.transition_to(MissionState.SEARCH_PATROL, "Parallel patrol resumed.")
            else:
                self.transition_to(MissionState.RESCUE_OPERATION, "Holding standoff for active rescue.")
        elif self.tracker.is_survivor_locked():
            self.transition_to(MissionState.PLAN_ROUTE, "Resuming route planning to verified target.")
        elif self.tracker.survivor_track is not None:
            self.transition_to(MissionState.VERIFY_TARGET, "Resuming target verification.")
        else:
            self.patrol_path.clear()
            self.waypoint_timer = 0.0
            self.is_scanning_360 = False
            self.transition_to(MissionState.SEARCH_PATROL, "Search patrol resumed.")
        self.log(f"[CONTROL] Autonomous FSM restored in state {self.current_state.name}.")
