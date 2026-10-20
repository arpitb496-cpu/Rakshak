"""
tracker.py - Spatial Target Tracking and Coordinate Triangulation for Rakshak.

Combines vision detection outputs (estimated range Z, bearing theta_bearing)
with the robot's odometric pose (x_r, y_r, theta_r) to compute and track
the estimated world coordinates of the survivor and rescuers:

  X_target_est = X_robot + Z * cos(theta_robot + theta_bearing)
  Y_target_est = Y_robot + Z * sin(theta_robot + theta_bearing)

Applies exponential moving average filtering across frames to mitigate
pixel discretization noise and provide rock-solid coordinate estimates.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
from detector import Detection


@dataclass
class TrackedTarget:
    """State of a tracked entity in world coordinates."""
    class_label: str
    est_world_x: float
    est_world_y: float
    est_range: float
    est_bearing: float
    mask_area_ratio: float
    target_id: str = "S1"
    triage_level: str = "CRITICAL"  # "CRITICAL", "INJURED", "STABLE"
    consecutive_hits: int = 1
    frames_since_seen: int = 0
    is_locked: bool = False
    is_secured: bool = False
    recent_measurements: List[Tuple[float, float]] = None
    triage_history: List[str] = None
    triage_frozen: bool = False       # True once label is majority-confirmed and track is locked
    true_triage: str = ""             # ground-truth label (metrics only, never used for dispatch)
    error_at_lock: Optional[float] = None  # localization error at moment this specific track locked

    def __post_init__(self):
        if self.recent_measurements is None:
            self.recent_measurements = []
        if self.triage_history is None:
            self.triage_history = []


class VisionTracker:
    """
    Multi-frame spatial tracker with nearest-estimate data association and variance-based locking.
    """

    def __init__(
        self,
        lock_threshold: int = 4,
        max_lost_frames: int = 30,
        smoothing_alpha: float = 0.35,
        max_lock_variance: float = 20.0,
        max_range: float = 340.0,
    ):
        self.lock_threshold = lock_threshold
        self.max_lost_frames = max_lost_frames
        self.alpha = smoothing_alpha
        self.max_lock_variance = max_lock_variance
        self.max_range = max_range

        # Multi-target survivor track dictionary: target_id -> TrackedTarget
        self.survivor_tracks: dict[str, TrackedTarget] = {}
        self.frozen_triage_records: dict[str, tuple] = {}
        self.active_track_id: Optional[str] = None
        self._manual_survivor_track: Optional[TrackedTarget] = None

        # Lock and rescue accuracy benchmarks
        self.has_locked: bool = False
        self.track_lock_errors: dict[str, float] = {}
        self._last_error_at_lock: Optional[float] = None
        self.error_at_rescue: Optional[float] = None
        self._next_track_num: int = 1
        self.all_created_track_ids: set[str] = set()
        self.unmatched_lock_tracks: set[str] = set()

        # Friendly rescuer tracks
        self.rescuer_tracks: List[TrackedTarget] = []

        # Ground truth tracking stats for telemetry and benchmarking
        self.last_localization_error: Optional[float] = None
        self.cumulative_readings: int = 0

    @property
    def error_at_lock(self) -> Optional[float]:
        st = self.survivor_track
        if st is not None:
            if getattr(st, "error_at_lock", None) is not None:
                return st.error_at_lock
            if st.target_id in self.track_lock_errors:
                return self.track_lock_errors[st.target_id]
        if self._last_error_at_lock is not None:
            return self._last_error_at_lock
        return None

    @error_at_lock.setter
    def error_at_lock(self, val: Optional[float]):
        self._last_error_at_lock = val
        st = self.survivor_track
        if st is not None:
            st.error_at_lock = val
            if val is not None:
                self.track_lock_errors[st.target_id] = val

    @property
    def survivor_track(self) -> Optional[TrackedTarget]:
        """Returns primary active survivor track prioritized by triage (CRITICAL > INJURED > STABLE)."""
        if self._manual_survivor_track is not None:
            return self._manual_survivor_track

        triage_order = {"CRITICAL": 0, "INJURED": 1, "STABLE": 2}
        unsecured = [trk for trk in self.survivor_tracks.values() if not trk.is_secured]
        if not unsecured:
            return None

        # Check if active_track_id is still valid and has highest available priority
        if self.active_track_id and self.active_track_id in self.survivor_tracks:
            act = self.survivor_tracks[self.active_track_id]
            if not act.is_secured:
                act_rank = triage_order.get(act.triage_level, 3)
                best_rank = min(triage_order.get(t.triage_level, 3) for t in unsecured)
                if act_rank <= best_rank:
                    return act

        # Sort by: 1) triage rank, 2) is_locked, 3) est_range
        unsecured.sort(key=lambda t: (triage_order.get(t.triage_level, 3), 0 if t.is_locked else 1, t.est_range))
        return unsecured[0]

    @survivor_track.setter
    def survivor_track(self, st: Optional[TrackedTarget]):
        self._manual_survivor_track = st
        if st is not None:
            self.survivor_tracks[st.target_id] = st
            self.active_track_id = st.target_id

    def clear_rescue_target(self, track_id: Optional[str] = None):
        """Atomically clears active per-rescue tracking state upon rescue completion."""
        if track_id:
            if track_id in self.survivor_tracks:
                self.survivor_tracks[track_id].is_secured = True
            for trk in self.survivor_tracks.values():
                if trk.target_id == track_id:
                    trk.is_secured = True
        elif self.active_track_id and self.active_track_id in self.survivor_tracks:
            self.survivor_tracks[self.active_track_id].is_secured = True

        self.active_track_id = None
        self._manual_survivor_track = None
        self.has_locked = False
        self._last_error_at_lock = None
        self.error_at_rescue = None
        self.last_localization_error = None
        try:
            from sound import get_sound_manager
            get_sound_manager().clear_siren_flag(track_id)
        except Exception:
            pass

    def update(
        self,
        detections: List[Detection],
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        ground_truth_survivor: Optional[Any] = None,
    ):
        """
        Updates tracking estimates with nearest-estimate data association across detections.
        Rejects detections beyond camera max range and verifies multi-frame consistency before lock.
        """
        self._manual_survivor_track = None

        # Filter survivor detections within valid camera range
        survivor_dets = [
            d for d in detections
            if d.class_label == "SURVIVOR" and 10.0 <= d.estimated_range <= self.max_range
        ]

        matched_track_ids = set()

        for det in survivor_dets:
            z = det.estimated_range
            bearing = det.estimated_bearing
            world_angle = robot_theta + bearing
            cos_b = max(0.2, math.cos(bearing))
            r_euclid = z / cos_b

            meas_x = robot_x + r_euclid * math.cos(world_angle)
            meas_y = robot_y + r_euclid * math.sin(world_angle)

            # Nearest-estimate data association
            best_track_id = None
            min_dist = float("inf")
            for tid, trk in self.survivor_tracks.items():
                if tid in matched_track_ids:
                    continue
                d = math.hypot(meas_x - trk.est_world_x, meas_y - trk.est_world_y)
                if d < min_dist:
                    min_dist = d
                    best_track_id = tid

            # Gating threshold: 85.0 pixels
            if best_track_id is not None and min_dist < 85.0:
                track = self.survivor_tracks[best_track_id]
                matched_track_ids.add(best_track_id)

                # Exponential smoothing filter (frozen once secured)
                if not track.is_secured:
                    track.est_world_x = self.alpha * meas_x + (1.0 - self.alpha) * track.est_world_x
                    track.est_world_y = self.alpha * meas_y + (1.0 - self.alpha) * track.est_world_y
                    track.est_range = self.alpha * z + (1.0 - self.alpha) * track.est_range
                    track.est_bearing = self.alpha * bearing + (1.0 - self.alpha) * track.est_bearing
                    track.mask_area_ratio = det.mask_area_ratio
                # Accumulate perceived triage votes (last 8 frames only)
                if not track.triage_frozen:
                    track.triage_history.append(det.triage_level)
                    if len(track.triage_history) > 8:
                        track.triage_history.pop(0)

                # Ground-truth: used ONLY for (a) ID matching and (b) securing flag
                if ground_truth_survivor is not None:
                    cands = ground_truth_survivor if isinstance(ground_truth_survivor, list) else [ground_truth_survivor]
                    best_cand = None
                    best_cd = float("inf")
                    for c in cands:
                        cx = getattr(c, "x", None)
                        cy = getattr(c, "y", None)
                        if cx is None and isinstance(c, (tuple, list)) and len(c) >= 2:
                            cx, cy = c[0], c[1]
                        if cx is not None and cy is not None:
                            cd = math.hypot(meas_x - cx, meas_y - cy)
                            if cd < best_cd:
                                best_cd = cd
                                best_cand = c
                    if best_cand is not None and best_cd < 85.0:
                        if getattr(best_cand, "is_secured", False):
                            track.is_secured = True
                        # Store true triage for metrics (never used for dispatch)
                        gt_label = getattr(best_cand, "triage", "")
                        if gt_label:
                            track.true_triage = gt_label
                            if track.triage_frozen:
                                self.frozen_triage_records[track.target_id] = (track.true_triage, track.triage_level)

                # Majority-vote triage from perceived labels (>=60% of last 8 frames)
                if not track.triage_frozen:
                    from collections import Counter
                    if track.triage_history:
                        top_label, top_count = Counter(track.triage_history).most_common(1)[0]
                        vote_pct = top_count / len(track.triage_history)
                        if vote_pct >= 0.60:
                            track.triage_level = top_label

                track.consecutive_hits += 1
                track.frames_since_seen = 0

                track.recent_measurements.append((meas_x, meas_y))
                if len(track.recent_measurements) > self.lock_threshold * 3:
                    track.recent_measurements.pop(0)

                # Variance check for lock confirmation
                if len(track.recent_measurements) >= self.lock_threshold:
                    pts = track.recent_measurements[-self.lock_threshold:]
                    mx = sum(p[0] for p in pts) / len(pts)
                    my = sum(p[1] for p in pts) / len(pts)
                    var_pos = sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in pts) / len(pts)
                    if math.sqrt(var_pos) <= self.max_lock_variance:
                        track.is_locked = True
                        if not track.triage_frozen and track.triage_history:
                            from collections import Counter
                            top_label, top_count = Counter(track.triage_history).most_common(1)[0]
                            vote_pct = top_count / len(track.triage_history)
                            if vote_pct >= 0.70 and len(track.triage_history) >= 4:
                                track.triage_level = top_label
                                track.triage_frozen = True
                                if not track.true_triage and ground_truth_survivor is not None:
                                    cands = ground_truth_survivor if isinstance(ground_truth_survivor, list) else [ground_truth_survivor]
                                    for c in cands:
                                        cx = getattr(c, "x", None)
                                        cy = getattr(c, "y", None)
                                        if cx is not None and cy is not None and math.hypot(track.est_world_x - cx, track.est_world_y - cy) < 85.0:
                                            track.true_triage = getattr(c, "triage", "")
                                            break
                                if track.true_triage:
                                    self.frozen_triage_records[track.target_id] = (track.true_triage, track.triage_level)
                                print(f"[TRIAGE] {track.target_id} label frozen: {track.triage_level} "
                                      f"({vote_pct*100:.0f}% of {len(track.triage_history)} frames)")
            else:
                # Correlate new track with real survivor identity if ground truth available
                matched_id = None
                _gt_triage = ""
                is_cand_secured = False
                if ground_truth_survivor is not None:
                    cands = ground_truth_survivor if isinstance(ground_truth_survivor, list) else [ground_truth_survivor]
                    best_cand = None
                    best_cd = float("inf")
                    for c in cands:
                        cx = getattr(c, "x", None)
                        cy = getattr(c, "y", None)
                        if cx is None and isinstance(c, (tuple, list)) and len(c) >= 2:
                            cx, cy = c[0], c[1]
                        if cx is not None and cy is not None:
                            cd = math.hypot(meas_x - cx, meas_y - cy)
                            if cd < best_cd:
                                best_cd = cd
                                best_cand = c
                    if best_cand is not None and best_cd < 85.0:
                        matched_id = getattr(best_cand, "id", None)
                        is_cand_secured = getattr(best_cand, "is_secured", False)
                        # GT triage stored for metrics; dispatch uses perceived label only
                        _gt_triage = getattr(best_cand, "triage", "")

                if matched_id and matched_id not in self.all_created_track_ids and matched_id not in self.survivor_tracks:
                    new_id = matched_id
                    if matched_id.startswith("S") and matched_id[1:].isdigit():
                        self._next_track_num = max(self._next_track_num, int(matched_id[1:]) + 1)
                else:
                    while True:
                        cand_id = f"S{self._next_track_num}"
                        self._next_track_num += 1
                        if cand_id not in self.all_created_track_ids and cand_id not in self.survivor_tracks:
                            new_id = cand_id
                            break
                self.all_created_track_ids.add(new_id)
                new_track = TrackedTarget(
                    class_label="SURVIVOR",
                    target_id=new_id,
                    est_world_x=meas_x,
                    est_world_y=meas_y,
                    est_range=z,
                    est_bearing=bearing,
                    mask_area_ratio=det.mask_area_ratio,
                    triage_level=det.triage_level,  # perceived label only
                    true_triage=_gt_triage,          # GT for metrics
                    consecutive_hits=1,
                    frames_since_seen=0,
                    is_locked=True if is_cand_secured else (self.lock_threshold <= 1),
                    is_secured=is_cand_secured,
                    recent_measurements=[(meas_x, meas_y)],
                    triage_history=[det.triage_level],
                )
                self.survivor_tracks[new_id] = new_track
                matched_track_ids.add(new_id)

            self.cumulative_readings += 1

        # Age unmatched tracks
        for tid, trk in list(self.survivor_tracks.items()):
            if tid not in matched_track_ids:
                trk.frames_since_seen += 1
                if trk.frames_since_seen > 3:
                    trk.recent_measurements.clear()
                if trk.frames_since_seen > self.max_lost_frames:
                    if not trk.is_secured:
                        trk.is_locked = False
                if trk.frames_since_seen > self.max_lost_frames * 3:
                    if not trk.is_secured:
                        del self.survivor_tracks[tid]

        # Update active track ID
        if not self.active_track_id or self.active_track_id not in self.survivor_tracks:
            active = self.survivor_track
            self.active_track_id = active.target_id if active else None

        if ground_truth_survivor is not None:
            self._last_ground_truth = ground_truth_survivor

        # Record error_at_lock for each track when it locks using shared function
        for trk in self.survivor_tracks.values():
            if trk.is_locked and trk.error_at_lock is None:
                self.record_lock_error(trk, ground_truth_survivor)

        st = self.survivor_track
        if ground_truth_survivor is not None and st is not None:
            err = self.compute_lock_error(st, ground_truth_survivor)
            if err is not None:
                self.last_localization_error = err
                if st.is_locked:
                    self.record_lock_error(st, ground_truth_survivor)
                    self.has_locked = True

        # Update Rescuer tracks
        rescuer_dets = [d for d in detections if d.class_label == "RESCUER"]
        self.rescuer_tracks = []
        for rd in rescuer_dets:
            rz = rd.estimated_range
            rb = rd.estimated_bearing
            r_angle = robot_theta + rb
            rx = robot_x + rz * math.cos(r_angle)
            ry = robot_y + rz * math.sin(r_angle)
            self.rescuer_tracks.append(
                TrackedTarget(
                    class_label="RESCUER",
                    est_world_x=rx,
                    est_world_y=ry,
                    est_range=rz,
                    est_bearing=rb,
                    mask_area_ratio=rd.mask_area_ratio,
                    is_locked=True,
                )
            )

    def compute_lock_error(self, track: TrackedTarget, ground_truth=None, gate_threshold: float = 85.0) -> Optional[float]:
        """
        Computes localization error for a track measured against that survivor's own ground truth.
        Checks spatial gate (gate_threshold). If no ground-truth survivor is within the gate, returns None.
        Does NOT silently re-associate cross-arena tracks.
        """
        gt_source = ground_truth if ground_truth is not None else getattr(self, "_last_ground_truth", None)
        if gt_source is None:
            return None
        cands = gt_source if isinstance(gt_source, list) else [gt_source]
        if not cands:
            return None
        def _pt(p):
            return (p.x, p.y) if hasattr(p, "x") else (p[0], p[1])
        # 1. Exact survivor ID match
        for c in cands:
            if getattr(c, "id", None) == track.target_id:
                gt_x, gt_y = _pt(c)
                err = float(math.hypot(track.est_world_x - gt_x, track.est_world_y - gt_y))
                if err <= gate_threshold:
                    return err
                # ID matches but is beyond gate: do NOT silently re-associate, treat as unmatched
                return None
        # 2. If track ID is not among candidates, check if nearest candidate is within gate
        best_cand = None
        best_dist = float("inf")
        for c in cands:
            gt_x, gt_y = _pt(c)
            d = float(math.hypot(track.est_world_x - gt_x, track.est_world_y - gt_y))
            if d < best_dist:
                best_dist = d
                best_cand = c
        if best_dist <= gate_threshold:
            return best_dist
        return None

    def record_lock_error(self, track: TrackedTarget, ground_truth=None, gate_threshold: float = 85.0) -> Optional[float]:
        """
        Shared function: measures and records error_at_lock at the moment a track first becomes locked,
        against that survivor's own ground truth. Never overwrites existing lock error for the same track.
        If outside gate, marks track as unmatched.
        """
        if track.target_id in self.track_lock_errors:
            return self.track_lock_errors[track.target_id]
        if track.target_id in self.unmatched_lock_tracks:
            return None
        if track.error_at_lock is not None:
            self.track_lock_errors[track.target_id] = track.error_at_lock
            return track.error_at_lock
        err = self.compute_lock_error(track, ground_truth, gate_threshold=gate_threshold)
        if err is not None:
            track.error_at_lock = err
            self.track_lock_errors[track.target_id] = err
            self._last_error_at_lock = err
        else:
            self.unmatched_lock_tracks.add(track.target_id)
        return err

    def is_survivor_locked(self) -> bool:
        """Returns True if survivor coordinates are verified and locked."""
        return self.survivor_track is not None and self.survivor_track.is_locked

    def get_survivor_estimate(self) -> Optional[Tuple[float, float]]:
        """Returns current estimated (x, y) coordinates of active survivor."""
        st = self.survivor_track
        if st is not None:
            return st.est_world_x, st.est_world_y
        return None

    def record_rescue_error(self, ground_truth_survivor) -> Optional[float]:
        """Records and returns the localization error at standoff distance."""
        st = self.survivor_track
        if st is not None:
            if isinstance(ground_truth_survivor, list) and ground_truth_survivor:
                best_gt = min(
                    ground_truth_survivor,
                    key=lambda p: math.hypot(st.est_world_x - p[0], st.est_world_y - p[1]),
                )
            else:
                if (
                    hasattr(self, "last_matched_gt")
                    and self.last_matched_gt is not None
                    and math.hypot(st.est_world_x - ground_truth_survivor[0], st.est_world_y - ground_truth_survivor[1]) > 80.0
                ):
                    best_gt = self.last_matched_gt
                else:
                    best_gt = ground_truth_survivor

            self.error_at_rescue = math.hypot(
                st.est_world_x - best_gt[0],
                st.est_world_y - best_gt[1],
            )
            return self.error_at_rescue
        return None

    def triage_accuracy(self) -> float:
        """Percent of locked tracks whose frozen perceived label matches ground-truth.
        Only tracks that have a known true_triage are counted."""
        total, correct = 0, 0
        for trk in self.survivor_tracks.values():
            if trk.true_triage:
                total += 1
                if trk.triage_level == trk.true_triage:
                    correct += 1
        return round(100.0 * correct / total, 1) if total > 0 else 100.0

    def reset(self):
        """Resets all active tracks and benchmark counters."""
        self.survivor_tracks.clear()
        self.frozen_triage_records.clear()
        self.active_track_id = None
        self._manual_survivor_track = None
        self.rescuer_tracks.clear()
        self.has_locked = False
        self.error_at_lock = None
        self.error_at_rescue = None
        self._next_track_num = 1
        self.all_created_track_ids.clear()
        self.unmatched_lock_tracks.clear()
        self.last_localization_error = None
        self.cumulative_readings = 0
