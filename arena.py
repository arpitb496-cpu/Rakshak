"""
arena.py - Disaster Site Simulation Environment for Rakshak.

Defines the 2D disaster site containing:
- Perimeter boundary walls
- Rubble and debris obstacles (solid irregular polygonal & rectangular blocks)
- Hazard zones (volatile danger areas treated as no-go zones)
- Rescuers (blue friendly personnel with required safe clearance zones)
- Survivor (target person icon, orange with pulsating beacon ring)
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Obstacle:
    """Rectangular or polygonal obstacle (rubble / debris / wall)."""
    name: str
    x: float
    y: float
    width: float
    height: float
    color: Tuple[int, int, int] = (105, 100, 95)
    rubble_type: str = "rubble"  # 'wall', 'rubble', 'debris'
    vertices: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.vertices:
            # Default to axis-aligned rectangle vertices
            self.vertices = [
                (self.x, self.y),
                (self.x + self.width, self.y),
                (self.x + self.width, self.y + self.height),
                (self.x, self.y + self.height),
            ]

    def get_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Returns boundary line segments for raycasting."""
        segments = []
        n = len(self.vertices)
        for i in range(n):
            segments.append((self.vertices[i], self.vertices[(i + 1) % n]))
        return segments

    def contains_point(self, px: float, py: float, margin: float = 0.0) -> bool:
        """Point-in-polygon test with safety margin."""
        min_x = self.x - margin
        max_x = self.x + self.width + margin
        min_y = self.y - margin
        max_y = self.y + self.height + margin
        return min_x <= px <= max_x and min_y <= py <= max_y


@dataclass
class HazardZone:
    """Volatile hazard zone (chemical spill / structural collapse risk)."""
    name: str
    x: float
    y: float
    radius: float
    severity: float = 1.0  # cost multiplier for path planning
    color: Tuple[int, int, int, int] = (230, 45, 45, 60)


from rescuer import RescuerAgent, RescuerState

Rescuer = RescuerAgent


@dataclass
class Survivor:
    """Target survivor needing rescue."""
    x: float
    y: float
    id: str = "S1"
    triage: str = "CRITICAL"  # "CRITICAL", "INJURED", "STABLE"
    radius: float = 13.0
    physical_width: float = 24.0  # Real width used for pinhole range estimation
    color: Tuple[int, int, int] = (255, 115, 20)  # High-visibility rescue orange
    beacon_phase: float = 0.0
    is_located: bool = False
    is_rescued: bool = False
    is_secured: bool = False

    @property
    def tag_color(self) -> Tuple[int, int, int]:
        """RGB color of the triage identification tag."""
        if self.triage == "CRITICAL":
            return (235, 35, 35)    # Red
        elif self.triage == "INJURED":
            return (245, 225, 35)   # Yellow
        else:
            return (35, 220, 50)    # Green


def get_active_survivor(
    arena,
    tracker=None,
    fsm=None,
    max_distance: float = 95.0,
) -> Optional[Survivor]:
    """
    Single authoritative helper that resolves the active survivor across all subsystems.
    Matches the tracker's estimated position to the nearest real survivor (within gating distance)
    and validates against the assigned ID / FSM current_sos_id / rescuer assignment.
    """
    survivors = getattr(arena, "survivors", None) or [arena.survivor]
    if not survivors:
        return None

    # Priority 1: FSM active SOS ID (if an SOS / rescue is currently underway)
    current_sos_id = getattr(fsm, "current_sos_id", None) if fsm is not None else None
    if current_sos_id:
        for s in survivors:
            if s.id == current_sos_id:
                return s

    # Priority 2: Rescuer assigned survivor ID
    rescuers = getattr(arena, "rescuers", [])
    for r in rescuers:
        if getattr(r, "state", None) in (RescuerState.DISPATCHED, RescuerState.EXTRACTING):
            assigned_id = getattr(r, "assigned_survivor_id", None)
            if assigned_id:
                for s in survivors:
                    if s.id == assigned_id:
                        return s

    # Priority 3: Match tracker's active spatial estimate and validate against assigned ID
    st = getattr(tracker, "survivor_track", None) if tracker is not None else None
    if st is not None and not getattr(st, "is_secured", False):
        target_id = getattr(st, "target_id", None)

        # 3a. Validate assigned ID: if assigned ID matches a survivor within gating distance
        if target_id:
            for s in survivors:
                if s.id == target_id:
                    d = math.hypot(s.x - st.est_world_x, s.y - st.est_world_y)
                    if d <= max_distance:
                        return s

        # 3b. Assigned ID invalid or mismatched (> max_distance): resolve to nearest real survivor
        unsecured = [s for s in survivors if not s.is_secured]
        pool = unsecured if unsecured else survivors
        best_s = min(pool, key=lambda s: math.hypot(s.x - st.est_world_x, s.y - st.est_world_y))
        dist = math.hypot(best_s.x - st.est_world_x, best_s.y - st.est_world_y)
        if dist <= max_distance:
            # Synchronize synthetic ID (S1, S2, etc.) with ground-truth survivor
            if target_id is not None and (target_id.startswith("S") and target_id[1:].isdigit()):
                st.target_id = best_s.id
                st.triage_level = best_s.triage
            return best_s
        elif target_id:
            for s in survivors:
                if s.id == target_id:
                    return s

    # Priority 4: Tracker target_id string match if tracker has an active track
    if st is not None and getattr(st, "target_id", None):
        for s in survivors:
            if s.id == st.target_id and not s.is_secured:
                return s
        for s in survivors:
            if s.id == st.target_id:
                return s

    # Priority 5: Fallback to first un-secured survivor or primary survivor
    for s in survivors:
        if not s.is_secured:
            return s
    return survivors[0]


class DisasterArena:
    """
    Complete disaster site environment.
    Coordinates: x in [0, width], y in [0, height].
    """

    def get_active_survivor(self, tracker=None, fsm=None, max_distance: float = 95.0) -> Optional[Survivor]:
        """Resolves active survivor using the single source-of-truth helper."""
        return get_active_survivor(self, tracker=tracker, fsm=fsm, max_distance=max_distance)

    def __init__(self, width: int = 800, height: int = 650):
        self.width = width
        self.height = height
        self.obstacles: List[Obstacle] = []
        self.hazard_zones: List[HazardZone] = []
        self.rescuers: List[Rescuer] = []
        self.survivors: List[Survivor] = []
        self._primary_survivor: Optional[Survivor] = None
        self.beacon_time: float = 0.0

        # Phase 4: Dynamic Hazard Zone Growth
        self.hazard_growth_enabled: bool = True
        self.dynamic_hazard_idx: int = 0
        self.hazard_growth_rate: float = 0.5  # px / second
        self.hazard_growth_accum: float = 0.0
        self.hazard_max_radius: float = 78.0
        self.hazard_expanded: bool = False

        self._build_environment()

    @property
    def survivor(self) -> Survivor:
        """Backward-compatible property returning primary active survivor."""
        if self.survivors:
            return self.survivors[0]
        if self._primary_survivor is None:
            self._primary_survivor = Survivor(x=700, y=130, id="S1", triage="CRITICAL")
        return self._primary_survivor

    @survivor.setter
    def survivor(self, s: Survivor):
        if not self.survivors:
            self.survivors.append(s)
        else:
            self.survivors[0] = s
        self._primary_survivor = s

    def _build_environment(self):
        """Constructs realistic disaster site layout."""
        self.obstacles.clear()
        self.hazard_zones.clear()
        self.rescuers.clear()

        # 1. Boundary walls
        wall_thick = 16.0
        w = float(self.width)
        h = float(self.height)
        wall_color = (65, 65, 70)

        # Top, Bottom, Left, Right perimeter
        self.obstacles.append(Obstacle("Wall_Top", 0, 0, w, wall_thick, wall_color, "wall"))
        self.obstacles.append(Obstacle("Wall_Bottom", 0, h - wall_thick, w, wall_thick, wall_color, "wall"))
        self.obstacles.append(Obstacle("Wall_Left", 0, 0, wall_thick, h, wall_color, "wall"))
        self.obstacles.append(Obstacle("Wall_Right", w - wall_thick, 0, wall_thick, h, wall_color, "wall"))

        # 2. Rubble & Debris clusters (collapsed concrete pillars, stone mounds)
        rubble_defs = [
            # (name, x, y, width, height, color, type)
            ("Rubble_Pile_A", 180, 100, 110, 80, (115, 105, 95), "rubble"),
            ("Rubble_Pile_B", 350, 160, 90, 130, (95, 90, 85), "rubble"),
            ("Debris_Pillar_C", 540, 90, 70, 140, (130, 120, 110), "debris"),
            ("Collapsed_Beam_D", 160, 310, 130, 60, (110, 100, 90), "rubble"),
            ("Center_Rubble_E", 380, 370, 120, 90, (100, 95, 90), "rubble"),
            ("Debris_Mound_F", 580, 360, 100, 110, (125, 115, 105), "debris"),
            ("South_Wall_Fragment", 210, 480, 80, 110, (105, 100, 95), "rubble"),
            ("East_Rubble_G", 500, 520, 120, 70, (115, 110, 100), "rubble"),
        ]

        for name, rx, ry, rw, rh, col, rtype in rubble_defs:
            self.obstacles.append(Obstacle(name, rx, ry, rw, rh, col, rtype))

        # 3. Hazard Zones (red translucent areas: fire risk, chemical leak)
        self.hazard_zones.append(HazardZone("Hazard_Alpha", 270, 240, radius=52.0, severity=1.8))
        self.hazard_zones.append(HazardZone("Hazard_Beta", 480, 270, radius=48.0, severity=1.5))

        # 4. Rescuer personnel (Blue uniforms, wandering or guarding safe perimeter)
        self.rescuers.append(Rescuer(id=1, x=130, y=220, radius=12.0, safe_buffer=38.0))
        self.rescuers.append(Rescuer(id=2, x=340, y=530, radius=12.0, safe_buffer=38.0))
        self.rescuers.append(Rescuer(id=3, x=670, y=490, radius=12.0, safe_buffer=38.0))

        # 5. Survivors (initial default positions with distinct triage tags)
        self.survivors = [
            Survivor(x=700, y=130, id="S1", triage="CRITICAL"),
            Survivor(x=470, y=70, id="S2", triage="INJURED"),
            Survivor(x=690, y=260, id="S3", triage="STABLE"),
            Survivor(x=680, y=560, id="S4", triage="CRITICAL"),
        ]
        self._primary_survivor = self.survivors[0]

    def update(self, dt: float):
        """Updates dynamic entities (pulsing beacon) and slow dynamic hazard growth."""
        self.beacon_time += dt
        for s in self.survivors:
            s.beacon_phase = (s.beacon_phase + dt * 3.5) % (2.0 * math.pi)

        # Phase 4: Dynamic Hazard Zone Growth (spreading hazard)
        if self.hazard_growth_enabled and self.hazard_zones:
            hz = self.hazard_zones[self.dynamic_hazard_idx]
            if hz.radius < self.hazard_max_radius:
                growth = self.hazard_growth_rate * dt
                hz.radius += growth
                self.hazard_growth_accum += growth
                if self.hazard_growth_accum >= 4.0:
                    self.hazard_expanded = True
                    self.hazard_growth_accum = 0.0

    def reset_survivor(self, x: Optional[float] = None, y: Optional[float] = None, seed: Optional[int] = None):
        """Places survivors in valid non-colliding locations."""
        if x is not None and y is not None:
            self.survivor.x = x
            self.survivor.y = y
            self.survivor.is_located = False
            self.survivor.is_rescued = False
            self.survivor.is_secured = False
        else:
            self.randomize_layout(seed=seed)
        self.survivor.is_rescued = False

    def is_point_colliding(self, px: float, py: float, margin: float = 0.0) -> bool:
        """Checks if a point collides with any solid obstacle."""
        for obs in self.obstacles:
            if obs.contains_point(px, py, margin):
                return True
        return False

    def is_inside_hazard(self, px: float, py: float, margin: float = 0.0) -> bool:
        """Checks if a point is within a hazard zone."""
        for hz in self.hazard_zones:
            dist = math.hypot(px - hz.x, py - hz.y)
            if dist <= hz.radius + margin:
                return True
        return False

    def get_all_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Returns all line segments in the arena for raycasting."""
        segments = []
        for obs in self.obstacles:
            segments.extend(obs.get_segments())
        return segments

    def check_circle_collision(self, cx: float, cy: float, radius: float) -> bool:
        """Checks if circle collides with any solid obstacle."""
        return self.is_point_colliding(cx, cy, margin=radius)

    def is_pose_safe(self, px: float, py: float, radius: float = 0.0) -> bool:
        """
        Comprehensive safety check: returns True iff the circular footprint (px, py, radius)
        has zero overlap with boundary walls, rubble, hazard zones, or rescuers.
        """
        wall_margin = 16.0 + radius
        if px <= wall_margin or px >= (self.width - wall_margin):
            return False
        if py <= wall_margin or py >= (self.height - wall_margin):
            return False

        if self.is_point_colliding(px, py, margin=radius):
            return False

        if self.is_inside_hazard(px, py, margin=radius):
            return False

        for r in self.rescuers:
            if math.hypot(px - r.x, py - r.y) <= (r.radius + radius):
                return False

        return True

    def randomize_layout(self, seed: Optional[int] = None):
        """
        Generates a randomized, solvable disaster site layout with shifted rubble piles,
        repositioned rescuers/hazards, and a randomly situated survivor across quadrants.
        """
        rng = random.Random(seed)

        self._build_environment()

        # Rubble layout perturbations that preserve open corridors
        shift_x = rng.uniform(-15.0, 15.0)
        shift_y = rng.uniform(-15.0, 15.0)
        for obs in self.obstacles:
            if obs.rubble_type != "wall":
                obs.x += shift_x * 0.5
                obs.y += shift_y * 0.5
                obs.vertices = [
                    (obs.x, obs.y),
                    (obs.x + obs.width, obs.y),
                    (obs.x + obs.width, obs.y + obs.height),
                    (obs.x, obs.y + obs.height),
                ]

        # Candidate survivor locations across all quadrants and corridors
        candidate_survivors = [
            (700, 130),  # Northeast nook
            (470, 70),   # North-central corridor
            (690, 260),  # East-central corridor
            (680, 560),  # Southeast corner
            (460, 570),  # South-central corridor
            (210, 570),  # South-west corridor
            (85, 480),   # West-south corridor
            (85, 230),   # West-central corridor
            (240, 70),   # North-west corridor
            (700, 390),  # East mid-corridor
            (630, 210),  # East inner opening
            (100, 370),  # West mid corridor
            (330, 70),   # North corridor mid
            (560, 70),   # Northeast upper corridor
            (690, 480),  # Southeast mid
            (120, 550),  # Southwest pocket
            (260, 380),  # Center west opening
            (520, 320),  # Center east opening
            (700, 180),  # Northeast lower nook
            (400, 570),  # South center corridor
        ]

        if seed is not None:
            chosen = candidate_survivors[seed % len(candidate_survivors)]
        else:
            chosen = rng.choice(candidate_survivors)

        # 3 to 5 survivors per layout
        n_surv = rng.randint(3, 5)
        # S1 always at chosen to guarantee 100% invariant compatibility with existing tests
        s1 = Survivor(x=float(chosen[0]), y=float(chosen[1]), id="S1", triage="CRITICAL")
        self.survivors = [s1]

        other_candidates = [c for c in candidate_survivors if c != chosen]
        rng.shuffle(other_candidates)
        triage_cycle = ["INJURED", "STABLE", "CRITICAL", "INJURED"]
        for idx in range(1, n_surv):
            c_pos = other_candidates[idx - 1]
            triage_lvl = triage_cycle[(idx - 1) % len(triage_cycle)]
            self.survivors.append(
                Survivor(x=float(c_pos[0]), y=float(c_pos[1]), id=f"S{idx + 1}", triage=triage_lvl)
            )

        self._primary_survivor = self.survivors[0]
