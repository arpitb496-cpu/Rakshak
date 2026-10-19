"""
app.py - Main Interactive Visual Application for Rakshak.

Features:
- Dual-panel interface: Tactical 2D Arena (800x740) + Cybernetic HUD (480x740)
- Anti-aliased rendering of differential robot, sensor rays, camera frustum, and obstacles
- First-Person Camera feed (160x90 scaled to 320x180) with real-time OpenCV detection overlay
- 8-sensor Polar Radar widget
- Live Ground-Truth vs Estimated Coordinate telemetry and error tracking
- Mission event terminal and interactive control hotkeys
"""

import sys
import math
import time
from typing import Optional, Tuple
import numpy as np
import cv2
import pygame

from arena import DisasterArena
from robot import DifferentialDriveRobot
from camera import RobotCamera
from detector import VisionDetector
from tracker import VisionTracker
from apf import APFController
from astar import AStarPlanner
from fsm import MissionFSM, MissionState
from rescuer import RescuerState


class RakshakApp:
    """
    Main Pygame Application for Rakshak Search-and-Rescue Robot.
    """

    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    ARENA_WIDTH = 800
    ARENA_HEIGHT = 640

    def __init__(
        self,
        fps: int = 60,
        lite_mode: bool = False,
        demo_mode: bool = False,
        seed: int = None,
        demo_output: str = "demo.mp4",
        demo_layouts: int = 3,
        parallel_mode: bool = False,
        fullscreen: bool = False,
        priority_mode: bool = True,
    ):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Rakshak - Autonomous Search & Rescue Robot | Hack in Hills '26")

        self.seed = seed
        self.fps = 30 if (lite_mode or demo_mode) else fps
        self.lite_mode = lite_mode
        self.demo_mode = demo_mode
        self.demo_output = demo_output
        self.demo_layouts = demo_layouts
        self.parallel_mode = parallel_mode
        self.fullscreen = fullscreen
        self.priority_mode = priority_mode
        self.show_fps = False

        # Fixed time step for rock-solid deterministic simulation
        self.fixed_dt = 1.0 / float(self.fps)

        # Off-screen fixed logical canvas (1280x720)
        self.screen = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

        # Real display window (windowed by default or fullscreen if requested)
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        try:
            if self.fullscreen:
                self.display_surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            else:
                self.display_surface = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        except Exception:
            import os
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.display.init()
            if self.fullscreen:
                self.display_surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            else:
                self.display_surface = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

        dw, dh = self.display_surface.get_size()
        self._update_scaling(dw, dh)

        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.manual_mode = False

        # Demo mode video recording state
        self.video_writer = None
        if self.demo_mode:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.demo_output, fourcc, 30.0, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
            )
            self.demo_current_layout = 1
            self.demo_seeds = [0, 1, 2]
            self.demo_hold_frames = 0
            self.demo_frame_count = 0
            self.demo_max_hold = 30

        # Visual toggles (Phase 5: sensing visible by default, APF/A*/HSV off by default)
        self.show_sensor_rays = True
        self.show_camera_frustum = True
        self.show_apf_vectors = False
        self.show_astar_path = False
        self.show_coverage_overlay = False
        self.show_hsv_mask = False
        self.moving_standby = False
        self.mission_report_saved = False
        self.final_map_saved = False

        # Fonts
        self._init_fonts()

        # Dynamic simulation & trail history
        self.trail_history = []
        self.sim_time = 0.0
        self.mission_complete_timer = 0.0
        self._loc_err_timer = 0.0
        self._loc_err_warned = False

        # Sound Subsystem
        from sound import get_sound_manager
        self.sound_manager = get_sound_manager()

        # Initialize Simulation Architecture
        self._init_system()

        # If in demo mode, set layout to seed 0 initially
        if self.demo_mode:
            self.arena.randomize_layout(seed=self.demo_seeds[0])
            self.reset_mission(randomize_survivor=False)

    def _init_fonts(self):
        """Initializes system fonts with generous sizes for high readability in 720p video."""
        try:
            self.font_title = pygame.font.SysFont("trebuchetms", 24, bold=True)
            self.font_subtitle = pygame.font.SysFont("trebuchetms", 13)
            self.font_badge = pygame.font.SysFont("consolas", 18, bold=True)
            self.font_label = pygame.font.SysFont("consolas", 20, bold=True)
            self.font_mono = pygame.font.SysFont("consolas", 16, bold=True)
            self.font_log = pygame.font.SysFont("consolas", 16)
            self.font_small = pygame.font.SysFont("consolas", 13, bold=True)
            self.font_banner_title = pygame.font.SysFont("trebuchetms", 34, bold=True)
            self.font_banner_sub = pygame.font.SysFont("consolas", 18, bold=True)
            self.font_legend = pygame.font.SysFont("consolas", 12, bold=True)
        except Exception:
            self.font_title = pygame.font.Font(None, 26)
            self.font_subtitle = pygame.font.Font(None, 14)
            self.font_badge = pygame.font.Font(None, 20)
            self.font_label = pygame.font.Font(None, 22)
            self.font_mono = pygame.font.Font(None, 18)
            self.font_log = pygame.font.Font(None, 18)
            self.font_small = pygame.font.Font(None, 14)
            self.font_banner_title = pygame.font.Font(None, 38)
            self.font_banner_sub = pygame.font.Font(None, 20)
            self.font_legend = pygame.font.Font(None, 13)

    def _update_scaling(self, dw: int, dh: int):
        """Calculates uniform scale factor and letterbox offsets for current display size."""
        scale_x = dw / self.WINDOW_WIDTH
        scale_y = dh / self.WINDOW_HEIGHT
        self.scale = min(scale_x, scale_y)
        scaled_w = int(self.WINDOW_WIDTH * self.scale)
        scaled_h = int(self.WINDOW_HEIGHT * self.scale)
        self.offset_x = (dw - scaled_w) // 2
        self.offset_y = (dh - scaled_h) // 2

    def screen_to_logical(self, screen_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Converts physical window/display coordinates to fixed logical canvas (1280x720) coordinates."""
        if self.scale <= 0:
            return screen_pos
        sx, sy = screen_pos
        lx = int((sx - self.offset_x) / self.scale)
        ly = int((sy - self.offset_y) / self.scale)
        return lx, ly

    def toggle_fullscreen(self):
        """Toggles between windowed mode and fullscreen display without resetting simulation state."""
        self.fullscreen = not self.fullscreen
        target_state = "ON" if self.fullscreen else "OFF"
        driver = pygame.display.get_driver() if pygame.display.get_init() else "unknown"
        try:
            toggled = False
            try:
                toggled = bool(pygame.display.toggle_fullscreen())
            except Exception:
                toggled = False

            if not toggled:
                if self.fullscreen:
                    try:
                        self.display_surface = pygame.display.set_mode(
                            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT),
                            pygame.SCALED | pygame.FULLSCREEN,
                        )
                    except Exception:
                        self.display_surface = pygame.display.set_mode(
                            (0, 0), pygame.FULLSCREEN
                        )
                else:
                    self.display_surface = pygame.display.set_mode(
                        (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
                    )
        except Exception as ex:
            print(f"[DISPLAY ERROR] Fullscreen toggle failed: {ex}")

        dw, dh = self.display_surface.get_size()
        self._update_scaling(dw, dh)
        print(f"[DISPLAY] fullscreen toggle -> {target_state}, size=({dw}, {dh}), driver={driver}")

    def _init_system(self):
        """Initializes all modular robotics subsystems."""
        self.arena = DisasterArena(width=self.ARENA_WIDTH, height=self.ARENA_HEIGHT)
        if self.seed is not None:
            import random as _random
            import numpy as _np
            _random.seed(self.seed)
            _np.random.seed(self.seed)
            self.arena.randomize_layout(seed=self.seed)
        self.robot = DifferentialDriveRobot(x=90.0, y=90.0, theta=0.4)
        self.camera = RobotCamera(width=160, height=90, fov_deg=65.0, max_range=340.0)
        self.detector = VisionDetector(focal_length=self.camera.focal_length, cx=self.camera.cx, cy=self.camera.cy)
        self.tracker = VisionTracker(lock_threshold=3, max_lost_frames=25)
        self.apf = APFController()
        self.astar = AStarPlanner(arena_width=self.ARENA_WIDTH, arena_height=self.ARENA_HEIGHT)
        self.fsm = MissionFSM(
            self.arena,
            self.robot,
            self.camera,
            self.detector,
            self.tracker,
            self.astar,
            self.apf,
            parallel_mode=self.parallel_mode,
            priority_mode=self.priority_mode,
        )

        # Pre-scan sensors once
        self.robot.scan_proximity_sensors(self.arena)

    def reset_mission(self, randomize_survivor: bool = True):
        """Resets the mission and relocates entities."""
        self.robot.x = 90.0
        self.robot.y = 90.0
        self.robot.theta = 0.4
        self.robot.stop()
        self.robot.trail.clear()
        self.trail_history.clear()
        self.sim_time = 0.0
        self.mission_complete_timer = 0.0
        self._loc_err_timer = 0.0
        self._loc_err_warned = False

        if randomize_survivor:
            self.arena.reset_survivor()

        self.tracker.reset()
        if hasattr(self, "sound_manager") and self.sound_manager is not None:
            self.sound_manager.clear_siren_flag()
        self.mission_report_saved = False
        self.final_map_saved = False
        self.fsm = MissionFSM(
            self.arena,
            self.robot,
            self.camera,
            self.detector,
            self.tracker,
            self.astar,
            self.apf,
            parallel_mode=self.parallel_mode,
            priority_mode=self.priority_mode,
        )
        if self.manual_mode:
            self.fsm.enter_manual_mode()
        self.robot.scan_proximity_sensors(self.arena)
        self.fsm.log("[RESET] Mission environment reset. New survivor beacon active.")

    def handle_events(self):
        """Processes keyboard and mouse inputs."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                is_cmd = bool((getattr(event, "mod", 0) & (pygame.KMOD_GUI | pygame.KMOD_META)) or (pygame.key.get_mods() & (pygame.KMOD_GUI | pygame.KMOD_META)))
                is_alt = bool((getattr(event, "mod", 0) & pygame.KMOD_ALT) or (pygame.key.get_mods() & pygame.KMOD_ALT))
                is_enter = event.key in (pygame.K_RETURN, pygame.K_KP_ENTER)
                if event.key in (pygame.K_f, pygame.K_F11) or (is_enter and (is_cmd or is_alt)):
                    self.toggle_fullscreen()
                elif event.key == pygame.K_ESCAPE:
                    if self.fullscreen:
                        self.toggle_fullscreen()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_mission(randomize_survivor=True)
                elif event.key == pygame.K_m:
                    if pygame.key.get_mods() & pygame.KMOD_CTRL:
                        muted = self.sound_manager.toggle_mute()
                        self.fsm.log(f"[AUDIO] Sound {'MUTED' if muted else 'ACTIVE'}.")
                    else:
                        self.manual_mode = not self.manual_mode
                        if self.manual_mode:
                            self.fsm.enter_manual_mode()
                        else:
                            self.fsm.exit_manual_mode()
                        self.fsm.log(f"[CONTROL] Manual mode {'ENABLED' if self.manual_mode else 'DISABLED'}.")
                elif event.key == pygame.K_u:
                    muted = self.sound_manager.toggle_mute()
                    self.fsm.log(f"[AUDIO] Sound {'MUTED' if muted else 'ACTIVE'}.")
                elif event.key == pygame.K_e:
                    if self.manual_mode:
                        self.fsm.manual_lock_and_sos()
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if self.manual_mode:
                        self.fsm.manual_dispatch_rescuer()
                elif event.key == pygame.K_x:
                    if self.manual_mode:
                        self.fsm.manual_confirm_extraction()
                elif event.key in (pygame.K_p, pygame.K_t):
                    self.priority_mode = not self.priority_mode
                    self.fsm.priority_mode = self.priority_mode
                    status = "ON" if self.priority_mode else "OFF"
                    self.fsm.log(f"[TRIAGE] Priority mode toggled {status}.")
                    print(f"[TRIAGE] Priority mode toggled {status}.")
                elif event.key == pygame.K_c:
                    self.show_camera_frustum = not self.show_camera_frustum
                elif event.key == pygame.K_y:
                    self.show_astar_path = not self.show_astar_path
                    self.fsm.log(f"[PATH] A* path overlay {'ENABLED' if self.show_astar_path else 'DISABLED'}.")
                elif event.key == pygame.K_v:
                    self.show_apf_vectors = not self.show_apf_vectors
                    self.fsm.log(f"[APF] Potential field vectors {'ENABLED' if self.show_apf_vectors else 'DISABLED'}.")
                elif event.key == pygame.K_h:
                    self.show_hsv_mask = not self.show_hsv_mask
                    self.fsm.log(f"[VISION] Camera HSV mask feed {'ENABLED' if self.show_hsv_mask else 'DISABLED'}.")
                elif event.key == pygame.K_n:
                    self.camera.noise_enabled = not self.camera.noise_enabled
                    self.fsm.log(f"[SENSOR] Camera noise + smoke haze {'ENABLED' if self.camera.noise_enabled else 'DISABLED'}.")
                elif event.key == pygame.K_o:
                    self.show_coverage_overlay = not self.show_coverage_overlay
                    self.fsm.log(f"[OVERLAY] Coverage grid overlay {'ENABLED' if self.show_coverage_overlay else 'DISABLED'}.")
                elif event.key == pygame.K_F1:
                    self.show_fps = not self.show_fps

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = self.screen_to_logical(event.pos)
                if 0 <= mx < self.ARENA_WIDTH and 0 <= my < self.ARENA_HEIGHT:
                    if event.button == 1:  # Left Click: relocate survivor
                        if not self.arena.is_point_colliding(mx, my, margin=15.0):
                            self.arena.reset_survivor(mx, my)
                            self.tracker.reset()
                            self.fsm.transition_to(MissionState.SEARCH_PATROL, "Survivor relocated by operator.")
                    elif event.button == 3:  # Right Click: add rubble block
                        from arena import Obstacle
                        new_rubble = Obstacle(
                            f"Rubble_User_{len(self.arena.obstacles)}",
                            mx - 25, my - 25, 50, 50, (110, 105, 95), "rubble"
                        )
                        self.arena.obstacles.append(new_rubble)
                        self.fsm.log(f"[OPERATOR] Deployed rubble obstacle at ({mx}, {my}).")

    def handle_manual_drive(self, dt: float):
        """Allows direct arrow/WASD manual driving for demonstration and testing."""
        keys = pygame.key.get_pressed()
        v = 0.0
        w = 0.0
        speed = 85.0
        turn_rate = 2.4

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            v += speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            v -= speed
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            w -= turn_rate
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            w += turn_rate

        self.robot.set_twist(v, w)

    def update(self, dt: float):
        """Simulation physics and perception step."""
        if self.paused:
            return

        # 1. Update dynamic arena entities
        self.arena.update(dt)

        # 2. Update robot kinematics
        if self.manual_mode:
            self.handle_manual_drive(dt)
            self.fsm.update(dt, manual_control=True)
        else:
            self.fsm.update(dt, manual_control=False)

        self.robot.update_kinematics(dt, self.arena)

        # 3. Update 8 proximity sensors
        self.robot.scan_proximity_sensors(self.arena)

        # 4. Robot trail history: maintain last ~4.0 seconds with timestamps
        self.sim_time += dt
        if not self.trail_history or math.hypot(self.robot.x - self.trail_history[-1][0], self.robot.y - self.trail_history[-1][1]) > 4.0:
            self.trail_history.append((self.robot.x, self.robot.y, self.sim_time))
        self.trail_history = [p for p in self.trail_history if (self.sim_time - p[2]) <= 4.0]

        # Runtime divergence check: assert / warn if GT differs from tracker estimate by >40px for >1.0s
        st = self.tracker.survivor_track
        if st is not None and not getattr(st, "is_secured", False):
            active_s = self.arena.get_active_survivor(tracker=self.tracker, fsm=self.fsm)
            if active_s is not None:
                err_px = math.hypot(st.est_world_x - active_s.x, st.est_world_y - active_s.y)
                if err_px > 40.0:
                    self._loc_err_timer += dt
                    if self._loc_err_timer > 1.0 and not getattr(self, "_loc_err_warned", False):
                        self._loc_err_warned = True
                        self.fsm.log(
                            f"[WARNING] Localization divergence: GT ({active_s.x:.1f}, {active_s.y:.1f}) differs from "
                            f"Est ({st.est_world_x:.1f}, {st.est_world_y:.1f}) by {err_px:.1f}px (>40px) for >1s (Target: {active_s.id})"
                        )
                else:
                    self._loc_err_timer = 0.0
                    self._loc_err_warned = False

        # 5. In interactive mode, auto-save mission_report.json on first MISSION_COMPLETE,
        #    then auto-reset after 3.0s
        if self.fsm.current_state == MissionState.MISSION_COMPLETE and not self.demo_mode:
            if not self.mission_report_saved:
                self.mission_report_saved = True
                try:
                    import json
                    report = self.fsm.generate_mission_report()
                    with open("mission_report.json", "w") as f:
                        json.dump(report, f, indent=2)
                    print(f"[REPORT] Mission report saved to mission_report.json "
                          f"({report['survivors_secured']}/{report['total_survivors']} secured, "
                          f"{report['area_scanned_pct']}% scanned)")
                except Exception as ex:
                    print(f"[REPORT] Failed to save mission report: {ex}")
            self.mission_complete_timer += dt
            if self.mission_complete_timer >= 3.0:
                self.mission_complete_timer = 0.0
                self.reset_mission(randomize_survivor=True)

        # Performance optimization in lite mode: prune trail
        if self.lite_mode and len(self.robot.trail) > 60:
            self.robot.trail = self.robot.trail[-60:]

        # Demo mode: track progression across 3 layouts
        if self.demo_mode:
            if self.fsm.current_state in (MissionState.RESCUE_OPERATION, MissionState.MISSION_COMPLETE):
                self.demo_hold_frames += 1
                if self.demo_hold_frames >= self.demo_max_hold:
                    self.demo_hold_frames = 0
                    if self.demo_current_layout < self.demo_layouts:
                        self.demo_current_layout += 1
                        seed = self.demo_seeds[self.demo_current_layout - 1]
                        self.arena.randomize_layout(seed=seed)
                        self.reset_mission(randomize_survivor=False)
                        self.fsm.log(
                            f"[DEMO] Transitioned to Layout {self.demo_current_layout}/{self.demo_layouts} (Seed {seed})."
                        )
                    else:
                        # Demo complete across all 3 layouts
                        if self.video_writer is not None:
                            self.video_writer.release()
                            self.video_writer = None
                        print(
                            f"[DEMO] Complete! Successfully recorded {self.demo_layouts} layouts to {self.demo_output} "
                            f"({self.demo_frame_count} frames, 1280x720, 30 fps, real-time speed)."
                        )
                        self.running = False

    def render(self):
        """Renders tactical arena, robot, overlays, and cybernetic HUD."""
        self.screen.fill((16, 18, 22))

        # 1. Render Arena on left surface
        self._render_arena()

        # 2. Render HUD on right panel
        self._render_hud()

        # 3. Optional Mission Complete Overlay (drawn over arena, only in non-demo mode)
        if self.fsm.current_state == MissionState.MISSION_COMPLETE and not self.demo_mode:
            self._render_mission_complete_overlay()
            # Auto-save final map screenshot once
            if not self.final_map_saved:
                self.final_map_saved = True
                try:
                    pygame.image.save(self.screen, "final_map.png")
                    print("[SNAPSHOT] final_map.png saved.")
                except Exception as ex:
                    print(f"[SNAPSHOT] Failed to save final_map.png: {ex}")

        # 4. Optional on-screen FPS counter badge (toggled via F-key)
        if self.show_fps:
            fps_val = self.clock.get_fps()
            fps_text = f"FPS: {fps_val:4.1f} | dt: {self.fixed_dt * 1000:.1f}ms (fixed)"
            badge_surf = self.font_mono.render(fps_text, True, (40, 240, 140))
            bw, bh = badge_surf.get_width() + 16, badge_surf.get_height() + 8
            bx = min(self.ARENA_WIDTH - bw - 15, 580)
            by = 12
            fps_rect = pygame.Rect(bx, by, bw, bh)
            pygame.draw.rect(self.screen, (16, 22, 28), fps_rect, border_radius=4)
            pygame.draw.rect(self.screen, (40, 240, 140), fps_rect, 1, border_radius=4)
            self.screen.blit(badge_surf, (bx + 8, by + 4))

        # 5. Optional on-screen Demo Recording status banner
        if self.demo_mode:
            rec_text = (
                f"DEMO REC // LAYOUT {self.demo_current_layout}/{self.demo_layouts} "
                f"[Seed {self.demo_seeds[self.demo_current_layout - 1]}]"
            )
            rec_surf = self.font_mono.render(rec_text, True, (255, 70, 70))
            rx, ry = 15, 12
            rw, rh = rec_surf.get_width() + 26, rec_surf.get_height() + 8
            r_box = pygame.Rect(rx, ry, rw, rh)
            pygame.draw.rect(self.screen, (22, 16, 18), r_box, border_radius=4)
            pygame.draw.rect(self.screen, (255, 60, 60), r_box, 1, border_radius=4)
            # Pulsing recording indicator dot
            rec_col = (255, 40, 40) if int(self.sim_time * 3) % 2 == 0 else (120, 20, 20)
            pygame.draw.circle(self.screen, rec_col, (rx + 10, ry + int(rh / 2)), 4)
            self.screen.blit(rec_surf, (rx + 20, ry + 4))

        # Scale logical canvas to real display surface with letterboxing
        dw, dh = self.display_surface.get_size()
        if (dw, dh) == (self.WINDOW_WIDTH, self.WINDOW_HEIGHT):
            self.display_surface.blit(self.screen, (0, 0))
            self.scale = 1.0
            self.offset_x = 0
            self.offset_y = 0
        else:
            self._update_scaling(dw, dh)
            scaled_w = int(self.WINDOW_WIDTH * self.scale)
            scaled_h = int(self.WINDOW_HEIGHT * self.scale)
            self.display_surface.fill((0, 0, 0))  # Letterbox black bars
            scaled_surf = pygame.transform.smoothscale(self.screen, (scaled_w, scaled_h))
            self.display_surface.blit(scaled_surf, (self.offset_x, self.offset_y))

        pygame.display.flip()

        # Capture frame for demo video recording
        if self.demo_mode and self.video_writer is not None:
            frame_rgb = np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)
            self.demo_frame_count += 1


    @staticmethod
    def _get_obstacle_visuals(obs):
        """Generates deterministic irregular polygon points, cracks, and label for rubble/debris."""
        x, y, w, h = obs.x, obs.y, obs.width, obs.height
        h_val = abs(hash(obs.name)) % 10000

        d1 = ((h_val % 7) - 3) * 0.022 * min(w, h)
        d2 = (((h_val // 7) % 7) - 3) * 0.022 * min(w, h)
        d3 = (((h_val // 49) % 7) - 3) * 0.022 * min(w, h)
        d4 = (((h_val // 343) % 7) - 3) * 0.022 * min(w, h)

        poly = [
            (x + w * 0.12 + d1, y + d2),
            (x + w * 0.50 + d3, y + d1 * 0.5),
            (x + w * 0.88 + d2, y + d3),
            (x + w + d4, y + h * 0.45 + d1),
            (x + w * 0.90 + d1, y + h * 0.88 + d2),
            (x + w * 0.52 + d2, y + h + d3),
            (x + w * 0.10 + d3, y + h * 0.90 + d4),
            (x + d2, y + h * 0.48 + d1),
        ]

        cx_mid = x + w * 0.5 + d1
        cy_mid = y + h * 0.5 + d2
        cracks = [
            (poly[0], (cx_mid, cy_mid)),
            ((cx_mid, cy_mid), (cx_mid + d3 * 2, cy_mid + h * 0.25)),
            ((cx_mid, cy_mid), (poly[4][0] - w * 0.1, poly[4][1] - h * 0.1)),
        ]

        label = "Debris" if ("debris" in obs.name.lower() or obs.rubble_type == "debris") else "Rubble"
        return poly, cracks, label

    def _render_arena(self):
        """Renders 2D tactical disaster site view."""
        self.screen.set_clip(pygame.Rect(0, 0, self.ARENA_WIDTH, self.WINDOW_HEIGHT))

        arena_rect = pygame.Rect(0, 0, self.ARENA_WIDTH, self.ARENA_HEIGHT)
        pygame.draw.rect(self.screen, (22, 25, 30), arena_rect)

        # Tactical background grid lines
        grid_step = 40
        for x in range(0, self.ARENA_WIDTH, grid_step):
            pygame.draw.line(self.screen, (28, 32, 40), (x, 0), (x, self.ARENA_HEIGHT), 1)
        for y in range(0, self.ARENA_HEIGHT, grid_step):
            pygame.draw.line(self.screen, (28, 32, 40), (0, y), (self.ARENA_WIDTH, y), 1)

        # Hazard Zones (translucent red)
        hazard_surf = pygame.Surface((self.ARENA_WIDTH, self.ARENA_HEIGHT), pygame.SRCALPHA)
        for hz in self.arena.hazard_zones:
            pulse_rad = hz.radius + 3.0 * math.sin(self.arena.beacon_time * 3.0)
            pygame.draw.circle(hazard_surf, (220, 40, 40, 55), (int(hz.x), int(hz.y)), int(pulse_rad))
            pygame.draw.circle(hazard_surf, (240, 60, 60, 110), (int(hz.x), int(hz.y)), int(hz.radius), 2)
            lbl = self.font_small.render("! HAZARD ZONE !", True, (255, 120, 120))
            self.screen.blit(lbl, (hz.x - 48, hz.y - 7))
        self.screen.blit(hazard_surf, (0, 0))

        # Base Station / Recharge pad at (90, 90)
        bx, by = 60, 60
        bw, bh = 60, 60
        base_rect = pygame.Rect(bx, by, bw, bh)
        pygame.draw.rect(self.screen, (24, 28, 36), base_rect, border_radius=4)
        pygame.draw.rect(self.screen, (255, 210, 40), base_rect, 2, border_radius=4)
        # Charging bolt / icon
        pygame.draw.line(self.screen, (255, 220, 50), (bx + 33, by + 14), (bx + 25, by + 30), 2)
        pygame.draw.line(self.screen, (255, 220, 50), (bx + 25, by + 30), (bx + 35, by + 30), 2)
        pygame.draw.line(self.screen, (255, 220, 50), (bx + 35, by + 30), (bx + 27, by + 46), 2)
        base_lbl = self.font_legend.render("BASE", True, (255, 215, 60))
        self.screen.blit(base_lbl, (bx + bw // 2 - base_lbl.get_width() // 2, by + bh - 13))

        # Coverage grid overlay (toggle with O)
        if self.show_coverage_overlay and hasattr(self.fsm, "coverage_grid"):
            self.fsm.coverage_grid.render_overlay(self.screen)

        # Obstacles (Irregular polygon rubble/debris with cracks, rock details, and labels)
        for obs in self.arena.obstacles:
            r = pygame.Rect(int(obs.x), int(obs.y), int(obs.width), int(obs.height))
            if obs.rubble_type == "wall":
                pygame.draw.rect(self.screen, obs.color, r)
                pygame.draw.rect(self.screen, (85, 85, 90), r, 1)
            else:
                poly, cracks, label = self._get_obstacle_visuals(obs)
                poly_int = [(int(p[0]), int(p[1])) for p in poly]

                # Base concrete polygon fill
                pygame.draw.polygon(self.screen, (58, 56, 54), poly_int)

                # 3D beveled light and shadow edges
                for i in range(len(poly_int)):
                    p_start = poly_int[i]
                    p_end = poly_int[(i + 1) % len(poly_int)]
                    mid_y = (p_start[1] + p_end[1]) / 2.0
                    edge_col = (115, 110, 105) if mid_y < (obs.y + obs.height * 0.5) else (34, 32, 30)
                    pygame.draw.line(self.screen, edge_col, p_start, p_end, 2)

                # Internal crack lines
                for c_start, c_end in cracks:
                    pygame.draw.line(self.screen, (36, 34, 32), (int(c_start[0]), int(c_start[1])), (int(c_end[0]), int(c_end[1])), 1)

                # Small rock details
                f_x1 = int(obs.x + obs.width * 0.25)
                f_y1 = int(obs.y + obs.height * 0.3)
                pygame.draw.circle(self.screen, (82, 79, 75), (f_x1, f_y1), 2)
                f_x2 = int(obs.x + obs.width * 0.72)
                f_y2 = int(obs.y + obs.height * 0.68)
                pygame.draw.circle(self.screen, (82, 79, 75), (f_x2, f_y2), 2)

                # Small cybernetic label pill
                lbl_surf = self.font_legend.render(label, True, (215, 210, 200))
                lw, lh = lbl_surf.get_width() + 10, lbl_surf.get_height() + 4
                lcx = int(obs.x + obs.width / 2.0 - lw / 2.0)
                lcy = int(obs.y + obs.height / 2.0 - lh / 2.0)
                pygame.draw.rect(self.screen, (22, 20, 18), (lcx, lcy, lw, lh), border_radius=3)
                pygame.draw.rect(self.screen, (75, 72, 68), (lcx, lcy, lw, lh), 1, border_radius=3)
                self.screen.blit(lbl_surf, (lcx + 5, lcy + 2))

        # Planned A* Path (thin dashed line)
        if self.show_astar_path and self.fsm.active_path:
            pts = self.fsm.active_path
            dash_len = 8.0
            gap_len = 6.0
            total_step = dash_len + gap_len

            for i in range(len(pts) - 1):
                p1 = pts[i]
                p2 = pts[i + 1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                seg_dist = math.hypot(dx, dy)
                if seg_dist < 1e-3:
                    continue
                ux = dx / seg_dist
                uy = dy / seg_dist

                d = 0.0
                while d < seg_dist:
                    d_end = min(seg_dist, d + dash_len)
                    start_pt = (int(p1[0] + ux * d), int(p1[1] + uy * d))
                    end_pt = (int(p1[0] + ux * d_end), int(p1[1] + uy * d_end))
                    pygame.draw.line(self.screen, (0, 235, 255), start_pt, end_pt, 1)
                    d += total_step

            # Waypoint node dots
            for i, pt in enumerate(pts):
                pt_int = (int(pt[0]), int(pt[1]))
                pygame.draw.circle(self.screen, (0, 255, 255), pt_int, 3)
                if i == self.fsm.current_waypoint_idx:
                    pygame.draw.circle(self.screen, (255, 255, 0), pt_int, 5, 1)

        # Friendly Rescuers (Active agents with clearance halo and status)
        for rescuer in self.arena.rescuers:
            rx, ry = int(rescuer.x), int(rescuer.y)
            pygame.draw.circle(self.screen, (30, 100, 200), (rx, ry), int(rescuer.safe_buffer), 1)
            pygame.draw.circle(self.screen, (35, 125, 245), (rx, ry), int(rescuer.radius))
            pygame.draw.circle(self.screen, (240, 245, 255), (rx, ry), 4)  # Helmet
            r_name = getattr(rescuer, "name", f"R{rescuer.id}")
            r_st = rescuer.state.value if hasattr(rescuer, "state") else "STANDBY"
            st_col = (
                (130, 180, 255)
                if r_st == "STANDBY"
                else (
                    (255, 205, 50)
                    if r_st == "DISPATCHED"
                    else ((50, 240, 120) if r_st == "EXTRACTING" else (180, 150, 255))
                )
            )
            lbl = self.font_legend.render(f"{r_name} [{r_st}]", True, st_col)
            self.screen.blit(lbl, (rx - lbl.get_width() // 2, ry + 15))

            # Draw rescuer planned path if navigating
            if rescuer.path and len(rescuer.path) > 1 and rescuer.state in (RescuerState.DISPATCHED, RescuerState.RETURNING):
                r_pts = [(int(p[0]), int(p[1])) for p in rescuer.path[rescuer.path_idx:]]
                if len(r_pts) >= 2:
                    pygame.draw.lines(self.screen, (45, 145, 255), False, [(rx, ry)] + r_pts, 1)

        # SOS Signal Pulse & Communication Link to Rescuer
        if self.fsm.sos_active_pulse or self.fsm.current_state == MissionState.RESCUE_OPERATION:
            pulse_surf = pygame.Surface((self.ARENA_WIDTH, self.ARENA_HEIGHT), pygame.SRCALPHA)
            for p_i in range(3):
                phase = (self.sim_time * 2.2 + p_i * 0.33) % 1.0
                p_rad = int(14 + 65 * phase)
                p_alpha = int(220 * (1.0 - phase))
                pygame.draw.circle(pulse_surf, (255, 160, 40, p_alpha), (int(self.robot.x), int(self.robot.y)), p_rad, 2)
            self.screen.blit(pulse_surf, (0, 0))

            # Dashed line from scout to chosen rescuer
            assigned_rescuer = self.fsm.active_assigned_rescuer
            if assigned_rescuer is not None:
                p1 = (self.robot.x, self.robot.y)
                p2 = (assigned_rescuer.x, assigned_rescuer.y)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                link_dist = math.hypot(dx, dy)
                if link_dist > 5.0:
                    ux = dx / link_dist
                    uy = dy / link_dist
                    dash_len = 8.0
                    gap_len = 6.0
                    step = dash_len + gap_len
                    offset = (self.sim_time * 25.0) % step
                    d = offset
                    while d < link_dist:
                        d_end = min(link_dist, d + dash_len)
                        s_pt = (int(p1[0] + ux * d), int(p1[1] + uy * d))
                        e_pt = (int(p1[0] + ux * d_end), int(p1[1] + uy * d_end))
                        pygame.draw.line(self.screen, (255, 200, 50), s_pt, e_pt, 2)
                        d += step

        # Extraction progress bar over survivor
        for r in self.arena.rescuers:
            if r.state == RescuerState.EXTRACTING and r.target_pos is not None:
                prog = r.extraction_progress
                bx = int(r.target_pos[0]) - 28
                by = int(r.target_pos[1]) - 48
                bw = 56
                bh = 7
                pygame.draw.rect(self.screen, (20, 24, 32), (bx, by, bw, bh), border_radius=3)
                pygame.draw.rect(self.screen, (40, 245, 120), (bx, by, int(bw * prog), bh), border_radius=3)
                pygame.draw.rect(self.screen, (200, 220, 240), (bx, by, bw, bh), 1, border_radius=3)
                p_txt = self.font_legend.render(f"EXTRACTING {int(prog * 100)}%", True, (60, 255, 140))
                self.screen.blit(p_txt, (bx + bw // 2 - p_txt.get_width() // 2, by - 14))

        # Target Survivors: person icon with triage tags and pulsing beacon ring
        survivors = self.arena.survivors if hasattr(self.arena, "survivors") and self.arena.survivors else [self.arena.survivor]
        for surv in survivors:
            sx, sy = int(surv.x), int(surv.y)

            # Pulsing beacon rings
            beacon_surf = pygame.Surface((self.ARENA_WIDTH, self.ARENA_HEIGHT), pygame.SRCALPHA)
            ring_rad1 = 15 + int(22 * (math.sin(surv.beacon_phase) * 0.5 + 0.5))
            ring_rad2 = 25 + int(28 * (math.sin(surv.beacon_phase + 1.2) * 0.5 + 0.5))
            pygame.draw.circle(beacon_surf, (255, 120, 20, 75), (sx, sy), ring_rad1, 2)
            pygame.draw.circle(beacon_surf, (255, 160, 40, 45), (sx, sy), ring_rad2, 1)

            # Standoff perimeter circle (safe rescue distance)
            pygame.draw.circle(beacon_surf, (50, 220, 120, 80), (sx, sy), int(MissionFSM.STANDOFF_DISTANCE), 1)
            self.screen.blit(beacon_surf, (0, 0))

            # Person Icon (head + body) in high-visibility orange (or green if rescued/secured)
            is_done = surv.is_rescued or getattr(surv, "is_secured", False)
            surv_color = (40, 225, 110) if is_done else (255, 120, 20)
            # Head
            pygame.draw.circle(self.screen, surv_color, (sx, sy - 9), 7)
            pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy - 9), 3)  # Helmet reflection
            # Torso
            pygame.draw.rect(self.screen, surv_color, (sx - 8, sy - 2, 16, 16), border_radius=4)
            # Amber high-vis reflective chest stripe
            pygame.draw.rect(self.screen, (255, 215, 45), (sx - 7, sy + 2, 14, 3))

            # Triage Tag Badge on torso (Red: CRITICAL, Yellow: INJURED, Green: STABLE)
            triage_lvl = getattr(surv, "triage", "CRITICAL")
            if triage_lvl == "CRITICAL":
                tag_col = (235, 35, 35)
            elif triage_lvl == "INJURED":
                tag_col = (245, 225, 35)
            else:
                tag_col = (35, 220, 50)
            pygame.draw.rect(self.screen, tag_col, (sx - 4, sy + 6, 8, 6), border_radius=1)

            # Shoulder pads / arms
            pygame.draw.circle(self.screen, surv_color, (sx - 7, sy + 3), 3)
            pygame.draw.circle(self.screen, surv_color, (sx + 7, sy + 3), 3)

            # Survivor text label with dark pill background: "S1 CRITICAL", etc.
            if is_done:
                surv_text = f"{surv.id} SECURED ✓"
                lbl_col = (60, 245, 140)
            else:
                surv_text = f"{surv.id} {triage_lvl}"
                lbl_col = (255, 80, 80) if triage_lvl == "CRITICAL" else (
                    (255, 220, 50) if triage_lvl == "INJURED" else (70, 235, 110)
                )

            lbl_s = self.font_legend.render(surv_text, True, lbl_col)
            lw, lh = lbl_s.get_width() + 12, lbl_s.get_height() + 4
            lx = sx - lw // 2
            ly = sy - 34
            pygame.draw.rect(self.screen, (20, 22, 28), (lx, ly, lw, lh), border_radius=3)
            pygame.draw.rect(self.screen, lbl_col, (lx, ly, lw, lh), 1, border_radius=3)
            self.screen.blit(lbl_s, (lx + 6, ly + 2))

        # Robot Comet Trail: last ~4s with fading alpha
        if len(self.trail_history) >= 2:
            trail_surf = pygame.Surface((self.ARENA_WIDTH, self.ARENA_HEIGHT), pygame.SRCALPHA)
            n_pts = len(self.trail_history)
            for i in range(n_pts - 1):
                p1 = self.trail_history[i]
                p2 = self.trail_history[i + 1]
                age = (self.sim_time - p1[2]) / 4.0
                alpha = max(15, min(240, int(240 * (1.0 - age))))
                pygame.draw.line(trail_surf, (0, 215, 255, alpha), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 2)
            self.screen.blit(trail_surf, (0, 0))

        # Camera Frustum Cone
        if self.show_camera_frustum:
            frustum_poly = self.camera.get_frustum_polygon(self.robot)
            frustum_surf = pygame.Surface((self.ARENA_WIDTH, self.ARENA_HEIGHT), pygame.SRCALPHA)
            poly_int = [(int(p[0]), int(p[1])) for p in frustum_poly]
            pygame.draw.polygon(frustum_surf, (0, 200, 255, 28), poly_int)
            pygame.draw.lines(frustum_surf, (0, 220, 255, 80), True, poly_int, 1)
            self.screen.blit(frustum_surf, (0, 0))

        # 8 Proximity Sensor Rays
        if self.show_sensor_rays:
            for i in range(self.robot.num_sensors):
                dist = self.robot.sensor_distances[i]
                hit_pt = self.robot.sensor_hit_points[i]
                if hit_pt is not None:
                    if dist < 35.0:
                        ray_color = (255, 50, 50)
                    elif dist < 65.0:
                        ray_color = (255, 210, 40)
                    else:
                        ray_color = (80, 220, 80)

                    pygame.draw.line(
                        self.screen,
                        ray_color,
                        (int(self.robot.x), int(self.robot.y)),
                        (int(hit_pt[0]), int(hit_pt[1])),
                        1,
                    )
                    pygame.draw.circle(self.screen, ray_color, (int(hit_pt[0]), int(hit_pt[1])), 3)

        # APF Force Vectors
        if self.show_apf_vectors and not self.manual_mode:
            rx, ry = int(self.robot.x), int(self.robot.y)
            fa = self.apf.f_att
            if np.linalg.norm(fa) > 1e-2:
                pygame.draw.line(self.screen, (40, 240, 60), (rx, ry), (rx + int(fa[0]), ry + int(fa[1])), 2)
            fr = self.apf.f_rep
            if np.linalg.norm(fr) > 1e-2:
                pygame.draw.line(self.screen, (255, 60, 60), (rx, ry), (rx + int(fr[0]), ry + int(fr[1])), 2)
            fn = self.apf.f_net
            if np.linalg.norm(fn) > 1e-2:
                pygame.draw.line(self.screen, (255, 240, 60), (rx, ry), (rx + int(fn[0]), ry + int(fn[1])), 3)

        # Render Robot
        self._render_robot()

        # Compact Legend at the bottom of the arena
        self._render_compact_legend()

        # Centered Rescue Banner ("SURVIVOR LOCATED") - only drawn during RESCUE_OPERATION
        # (Draw only one overlay at a time to prevent ghost banners behind the MISSION COMPLETE card)
        if self.fsm.current_state == MissionState.RESCUE_OPERATION:
            bw = 540
            bh = 145
            bx = (self.ARENA_WIDTH - bw) // 2
            by = (self.ARENA_HEIGHT - bh) // 2

            # Solid opaque backdrop so nothing behind bleeds through
            pygame.draw.rect(self.screen, (14, 18, 26), (bx, by, bw, bh), border_radius=8)
            pygame.draw.rect(self.screen, (40, 240, 140), (bx, by, bw, bh), 2, border_radius=8)
            pygame.draw.rect(self.screen, (24, 46, 36), (bx + 3, by + 3, bw - 6, bh - 6), 1, border_radius=6)

            # Determine active target survivor for consistent telemetry and banner display
            st = self.tracker.survivor_track
            active_surv = self.arena.get_active_survivor(tracker=self.tracker, fsm=self.fsm)

            # Banner Title
            if self.fsm.current_state == MissionState.MISSION_COMPLETE:
                b_title = "ALL SURVIVORS SECURED"
            else:
                surv_tag = f" ({active_surv.id})" if hasattr(active_surv, "id") else ""
                b_title = f"SURVIVOR LOCATED{surv_tag}"
            t_surf = self.font_banner_title.render(b_title, True, (40, 240, 140))
            self.screen.blit(t_surf, (bx + (bw - t_surf.get_width()) // 2, by + 16))

            # Coordinates of active target survivor
            coords_str = f"Target Position: X = {active_surv.x:.1f} px,  Y = {active_surv.y:.1f} px"
            c_surf = self.font_banner_sub.render(coords_str, True, (240, 248, 255))
            self.screen.blit(c_surf, (bx + (bw - c_surf.get_width()) // 2, by + 58))

            # Standoff and Localization Error
            dist_standoff = math.hypot(active_surv.x - self.robot.x, active_surv.y - self.robot.y)
            if st is not None:
                err_val = math.hypot(st.est_world_x - active_surv.x, st.est_world_y - active_surv.y)
            else:
                err_val = self.tracker.error_at_rescue or self.tracker.last_localization_error or 0.0
            stat_str = f"Standoff: {dist_standoff:.1f} px  |  Error: {err_val:.1f} px"
            s_surf = self.font_mono.render(stat_str, True, (160, 225, 200))
            self.screen.blit(s_surf, (bx + (bw - s_surf.get_width()) // 2, by + 86))

            # Status subtitle or countdown
            if self.demo_mode:
                reset_str = f"Advancing layout in {max(0.0, 1.0 - self.demo_hold_frames / 30.0):.1f}s..."
            elif self.fsm.current_state == MissionState.MISSION_COMPLETE:
                reset_str = f"Mission Complete! Auto-resetting in {max(0.0, 3.0 - self.mission_complete_timer):.1f}s..."
            else:
                extracting_r = next((r for r in self.arena.rescuers if r.state == RescuerState.EXTRACTING), None)
                dispatched_r = next((r for r in self.arena.rescuers if r.state == RescuerState.DISPATCHED), None)
                if extracting_r:
                    pct = int(min(100.0, (extracting_r.extraction_timer / extracting_r.extraction_duration) * 100))
                    reset_str = f"Rescuer {extracting_r.name} extracting survivor... {pct}%"
                elif dispatched_r:
                    reset_str = f"Rescuer {dispatched_r.name} dispatched -> Inbound to target..."
                else:
                    reset_str = "Beacon locked -> Holding safe standoff..."
            r_surf = self.font_small.render(reset_str, True, (140, 160, 180))
            self.screen.blit(r_surf, (bx + (bw - r_surf.get_width()) // 2, by + 114))

        # Arena Bottom Status Strip
        self._render_arena_status_bar()
        self.screen.set_clip(None)

    def _render_compact_legend(self):
        """Renders compact horizontal legend bar at the bottom of the arena."""
        leg_w = 660
        leg_h = 24
        leg_x = (self.ARENA_WIDTH - leg_w) // 2
        leg_y = self.ARENA_HEIGHT - 32

        leg_surf = pygame.Surface((leg_w, leg_h), pygame.SRCALPHA)
        pygame.draw.rect(leg_surf, (18, 22, 28, 215), (0, 0, leg_w, leg_h), border_radius=4)
        pygame.draw.rect(leg_surf, (45, 55, 70, 180), (0, 0, leg_w, leg_h), 1, border_radius=4)
        self.screen.blit(leg_surf, (leg_x, leg_y))

        # 4 items: Survivor, Rescuer, Hazard Zone, Debris
        items = [
            ("Survivor", (255, 130, 20), "circle_person"),
            ("Rescuer", (40, 140, 255), "circle_rescuer"),
            ("Hazard Zone", (255, 70, 70), "hazard"),
            ("Debris", (140, 135, 130), "debris"),
        ]

        slot_w = leg_w // 4
        for idx, (label, col, glyph_type) in enumerate(items):
            slot_x = leg_x + idx * slot_w
            slot_cy = leg_y + leg_h // 2

            gx = slot_x + 18
            if glyph_type == "circle_person":
                pygame.draw.circle(self.screen, col, (gx, slot_cy - 3), 4)
                pygame.draw.rect(self.screen, col, (gx - 4, slot_cy + 1, 8, 5), border_radius=1)
            elif glyph_type == "circle_rescuer":
                pygame.draw.circle(self.screen, col, (gx, slot_cy), 5)
                pygame.draw.circle(self.screen, (240, 245, 255), (gx, slot_cy), 2)
            elif glyph_type == "hazard":
                pygame.draw.circle(self.screen, (255, 60, 60), (gx, slot_cy), 5, 1)
                pygame.draw.circle(self.screen, (255, 90, 90), (gx, slot_cy), 3)
            elif glyph_type == "debris":
                pygame.draw.rect(self.screen, (75, 72, 68), (gx - 5, slot_cy - 4, 10, 8))
                pygame.draw.rect(self.screen, (130, 125, 120), (gx - 5, slot_cy - 4, 10, 8), 1)

            txt_surf = self.font_legend.render(label, True, (215, 222, 230))
            self.screen.blit(txt_surf, (gx + 12, slot_cy - txt_surf.get_height() // 2))

    def _render_robot(self):
        """Renders detailed differential-drive robot body, treads, and heading indicator."""
        rx, ry = self.robot.x, self.robot.y
        r_rad = self.robot.radius
        theta = self.robot.theta

        # Robot chassis circle
        pygame.draw.circle(self.screen, (235, 240, 245), (int(rx), int(ry)), int(r_rad))
        pygame.draw.circle(self.screen, (45, 52, 60), (int(rx), int(ry)), int(r_rad), 2)

        # Differential wheels (left & right tread boxes)
        wheel_w = 12.0
        wheel_h = 5.0
        left_angle = theta + math.pi / 2.0
        right_angle = theta - math.pi / 2.0
        d_wheel = self.robot.wheel_base / 2.0

        for ang in [left_angle, right_angle]:
            wx = rx + d_wheel * math.cos(ang)
            wy = ry + d_wheel * math.sin(ang)
            pygame.draw.circle(self.screen, (35, 40, 45), (int(wx), int(wy)), 4)

        # Forward Direction Pointer (Heading)
        front_x = rx + (r_rad + 6.0) * math.cos(theta)
        front_y = ry + (r_rad + 6.0) * math.sin(theta)
        pygame.draw.line(self.screen, (255, 70, 70), (int(rx), int(ry)), (int(front_x), int(front_y)), 3)
        pygame.draw.circle(self.screen, (255, 100, 100), (int(front_x), int(front_y)), 3)

    def _render_arena_status_bar(self):
        """Bottom strip under arena showing mode and keybind hints."""
        bar_rect = pygame.Rect(0, self.ARENA_HEIGHT, self.ARENA_WIDTH, self.WINDOW_HEIGHT - self.ARENA_HEIGHT)
        pygame.draw.rect(self.screen, (20, 22, 28), bar_rect)
        pygame.draw.line(self.screen, (40, 45, 55), (0, self.ARENA_HEIGHT), (self.ARENA_WIDTH, self.ARENA_HEIGHT), 1)

        mode_str = "MANUAL [WASD/ARROWS]" if self.manual_mode else "AUTONOMOUS FSM"
        mode_col = (255, 180, 50) if self.manual_mode else (60, 220, 160)
        mode_surf = self.font_label.render(f"CONTROL: {mode_str}", True, mode_col)
        self.screen.blit(mode_surf, (15, self.ARENA_HEIGHT + 12))

        if self.lite_mode:
            lite_surf = self.font_label.render("[LITE MODE]", True, (255, 205, 50))
            self.screen.blit(lite_surf, (270, self.ARENA_HEIGHT + 12))

        # Audio status badge [U]
        if not self.sound_manager.is_enabled():
            audio_col = (180, 70, 70)
            audio_text = "[U] AUDIO: ERR"
        elif self.sound_manager.is_muted():
            audio_col = (255, 80, 80)
            audio_text = "[U] AUDIO: MUTED"
        else:
            audio_col = (60, 230, 160)
            audio_text = "[U] AUDIO: ON"
        audio_surf = self.font_small.render(audio_text, True, audio_col)
        self.screen.blit(audio_surf, (self.ARENA_WIDTH - 290, self.ARENA_HEIGHT + 12))

        # Noise & standby indicator badges
        noise_col = (255, 75, 75) if self.camera.noise_enabled else (70, 85, 100)
        noise_surf = self.font_small.render("[N] NOISE", True, noise_col)
        self.screen.blit(noise_surf, (self.ARENA_WIDTH - 165, self.ARENA_HEIGHT + 12))

        standby_col = (80, 200, 255) if self.moving_standby else (70, 85, 100)
        standby_surf = self.font_small.render("[P] PATROL", True, standby_col)
        self.screen.blit(standby_surf, (self.ARENA_WIDTH - 80, self.ARENA_HEIGHT + 12))

        paused_str = " [PAUSED]" if self.paused else ""
        if paused_str:
            p_surf = self.font_label.render(paused_str, True, (255, 70, 70))
            self.screen.blit(p_surf, (400 if self.lite_mode else 270, self.ARENA_HEIGHT + 12))

        if self.manual_mode:
            can_lock = self.fsm.can_manual_lock()
            can_dispatch = self.fsm.can_manual_dispatch()
            can_extract = self.fsm.can_manual_extract()

            x_pos = 15
            y_pos = self.ARENA_HEIGHT + 42

            drive_surf = self.font_small.render("[WASD/Arrows] Move", True, (180, 190, 205))
            self.screen.blit(drive_surf, (x_pos, y_pos))
            x_pos += drive_surf.get_width() + 10

            sep0 = self.font_small.render("|", True, (65, 75, 90))
            self.screen.blit(sep0, (x_pos, y_pos))
            x_pos += sep0.get_width() + 10

            # [E] Lock & SOS
            lock_col = (50, 235, 130) if can_lock else (95, 105, 120)
            lock_label = "[E] LOCK & SOS" + (" (READY)" if can_lock else " (OUT OF RANGE)")
            lock_surf = self.font_small.render(lock_label, True, lock_col)
            self.screen.blit(lock_surf, (x_pos, y_pos))
            x_pos += lock_surf.get_width() + 10

            sep1 = self.font_small.render("|", True, (65, 75, 90))
            self.screen.blit(sep1, (x_pos, y_pos))
            x_pos += sep1.get_width() + 10

            # [Enter] Dispatch
            disp_col = (255, 205, 50) if can_dispatch else (95, 105, 120)
            disp_label = "[Enter] DISPATCH" + (" (READY)" if can_dispatch else " (NO SOS)")
            disp_surf = self.font_small.render(disp_label, True, disp_col)
            self.screen.blit(disp_surf, (x_pos, y_pos))
            x_pos += disp_surf.get_width() + 10

            sep2 = self.font_small.render("|", True, (65, 75, 90))
            self.screen.blit(sep2, (x_pos, y_pos))
            x_pos += sep2.get_width() + 10

            # [X] Extract
            ext_col = (60, 220, 255) if can_extract else (95, 105, 120)
            ext_label = "[X] EXTRACT" + (" (READY)" if can_extract else " (STANDBY)")
            ext_surf = self.font_small.render(ext_label, True, ext_col)
            self.screen.blit(ext_surf, (x_pos, y_pos))
            x_pos += ext_surf.get_width() + 10

            sep3 = self.font_small.render("|", True, (65, 75, 90))
            self.screen.blit(sep3, (x_pos, y_pos))
            x_pos += sep3.get_width() + 10

            p_status = "ON" if getattr(self, "priority_mode", True) else "OFF"
            ret_surf = self.font_small.render(f"[M] Auto | [F] Full | [P] Priority: {p_status}", True, (140, 150, 165))
            # Clip to arena width so it never overflows
            clip_w = max(0, self.ARENA_WIDTH - x_pos - 5)
            clipped = ret_surf.subsurface((0, 0, min(ret_surf.get_width(), clip_w), ret_surf.get_height()))
            self.screen.blit(clipped, (x_pos, y_pos))
        else:
            p_status = "ON" if getattr(self, "priority_mode", True) else "OFF"
            # Row A: navigation & mode controls
            row_a = f"[SPACE] Pause  [R] Reset  [M] Manual  [F] Fullscreen  [P] Priority: {p_status}"
            # Row B: overlay toggles
            row_b = "[Y] A* path  [V] APF field  [O] Grid  [N] Noise  [U] Mute"
            surf_a = self.font_small.render(row_a, True, (155, 165, 180))
            surf_b = self.font_small.render(row_b, True, (100, 112, 130))
            max_w = self.ARENA_WIDTH - 20
            # Clip each row if still too wide (Retina safety)
            if surf_a.get_width() > max_w:
                surf_a = surf_a.subsurface((0, 0, max_w, surf_a.get_height()))
            if surf_b.get_width() > max_w:
                surf_b = surf_b.subsurface((0, 0, max_w, surf_b.get_height()))
            self.screen.blit(surf_a, (15, self.ARENA_HEIGHT + 36))
            self.screen.blit(surf_b, (15, self.ARENA_HEIGHT + 55))

    def _render_hud(self):
        """Renders the cybernetic HUD on the right 480px panel."""
        hud_x = self.ARENA_WIDTH
        hud_w = self.WINDOW_WIDTH - self.ARENA_WIDTH
        hud_h = self.WINDOW_HEIGHT

        # HUD background
        hud_rect = pygame.Rect(hud_x, 0, hud_w, hud_h)
        pygame.draw.rect(self.screen, (20, 23, 29), hud_rect)
        pygame.draw.line(self.screen, (45, 52, 65), (hud_x, 0), (hud_x, hud_h), 2)

        curr_y = 12

        # 1. Header Banner
        t_title = self.font_title.render("RAKSHAK // SAR", True, (0, 220, 255))
        t_sub = self.font_subtitle.render("See. Locate. Rescue. | Hack in Hills '26", True, (140, 155, 175))
        self.screen.blit(t_title, (hud_x + 22, curr_y))
        self.screen.blit(t_sub, (hud_x + 22, curr_y + 26))
        curr_y += 50

        # 2. Mission State Illuminated Badge
        state = self.fsm.current_state
        state_colors = {
            MissionState.SEARCH_PATROL: (0, 190, 240),
            MissionState.VERIFY_TARGET: (255, 205, 50),
            MissionState.PLAN_ROUTE: (120, 160, 255),
            MissionState.NAVIGATE_TO_SURVIVOR: (70, 150, 255),
            MissionState.RESCUE_OPERATION: (255, 130, 30),
            MissionState.MISSION_COMPLETE: (40, 235, 120),
            MissionState.HAZARD_RECOVERY: (255, 60, 60),
            MissionState.RETURN_TO_BASE: (255, 210, 40),
        }
        if self.manual_mode:
            badge_col = (255, 180, 50)
            badge_text = "MODE: MANUAL OVERRIDE"
        else:
            badge_col = state_colors.get(state, (200, 200, 200))
            badge_text = f"STATE: {state.name}"
        b_surf = self.font_badge.render(badge_text, True, badge_col)

        badge_box = pygame.Rect(hud_x + 16, curr_y, hud_w - 32, 26)
        pygame.draw.rect(self.screen, (28, 33, 42), badge_box, border_radius=4)
        pygame.draw.rect(self.screen, badge_col, badge_box, 1, border_radius=4)
        self.screen.blit(b_surf, (hud_x + 24, curr_y + 5))
        curr_y += 30

        # Phase 3: Battery & Area Scanned Bar
        bat = getattr(self.fsm, "battery", 100.0)
        bat_col = (50, 240, 110) if bat > 50 else ((255, 215, 40) if bat > 25 else (255, 60, 60))
        scanned_pct = int(self.fsm.coverage_grid.percent_scanned) if hasattr(self.fsm, "coverage_grid") else 0

        gauge_box = pygame.Rect(hud_x + 16, curr_y, hud_w - 32, 26)
        pygame.draw.rect(self.screen, (26, 30, 38), gauge_box, border_radius=4)
        pygame.draw.rect(self.screen, (45, 52, 65), gauge_box, 1, border_radius=4)

        b_label = self.font_small.render(f"BAT: {int(bat)}%", True, bat_col)
        self.screen.blit(b_label, (hud_x + 24, curr_y + 6))
        bar_w = 75
        bar_h = 9
        bar_x = hud_x + 24 + b_label.get_width() + 8
        bar_y = curr_y + 8
        pygame.draw.rect(self.screen, (40, 45, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=2)
        fill_w = int(bar_w * (bat / 100.0))
        pygame.draw.rect(self.screen, bat_col, (bar_x, bar_y, fill_w, bar_h), border_radius=2)

        cov_text = self.font_small.render(f"Area scanned: {scanned_pct}%", True, (40, 220, 200))
        self.screen.blit(cov_text, (hud_x + hud_w - 32 - cov_text.get_width() - 8, curr_y + 6))
        curr_y += 32

        # 3. First-Person Camera Feed (160x90 scaled up to 320x180)
        cam_box = pygame.Rect(hud_x + 16, curr_y, 320, 180)
        annotated_bgr = self.detector.get_annotated_feed(self.camera.last_frame, scale=2)

        # Convert BGR (OpenCV) to RGB for Pygame surface
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        cam_surface = pygame.surfarray.make_surface(np.transpose(annotated_rgb, (1, 0, 2)))

        self.screen.blit(cam_surface, (hud_x + 16, curr_y))
        pygame.draw.rect(self.screen, (55, 65, 80), cam_box, 2)

        # REC indicator
        rec_dot_color = (255, 40, 40) if int(self.sim_time * 2) % 2 == 0 else (120, 20, 20)
        pygame.draw.circle(self.screen, rec_dot_color, (hud_x + 28, curr_y + 12), 4)
        rec_text = self.font_small.render("CAM-1 [160x90]", True, (200, 210, 220))
        self.screen.blit(rec_text, (hud_x + 36, curr_y + 6))

        # 4. Proximity Sensor Polar Radar (placed to the right of camera feed)
        radar_cx = hud_x + 320 + 72
        radar_cy = curr_y + 90
        self._render_radar(radar_cx, radar_cy, radius=52)

        curr_y += 190

        # 5. Survivor Telemetry Card (Essentials Only: GT vs Est, Range, Error, Status)
        card_rect = pygame.Rect(hud_x + 16, curr_y, hud_w - 32, 142)
        pygame.draw.rect(self.screen, (26, 30, 38), card_rect, border_radius=4)
        pygame.draw.rect(self.screen, (42, 48, 60), card_rect, 1, border_radius=4)

        c_title = self.font_label.render("TARGET TELEMETRY", True, (255, 140, 40))
        self.screen.blit(c_title, (hud_x + 24, curr_y + 10))

        # Missions Completed count
        m_str = f"Missions: {self.fsm.missions_completed}/{self.fsm.total_survivors}"
        m_surf = self.font_mono.render(m_str, True, (60, 245, 130))
        self.screen.blit(m_surf, (hud_x + hud_w - 32 - m_surf.get_width() - 10, curr_y + 12))

        st = self.tracker.survivor_track
        gt_x, gt_y = None, None
        if st is not None:
            matched_surv = self.arena.get_active_survivor(tracker=self.tracker, fsm=self.fsm)
            gt_x, gt_y = (matched_surv.x, matched_surv.y) if matched_surv else (0.0, 0.0)
            err_px = math.hypot(st.est_world_x - gt_x, st.est_world_y - gt_y)

            status_str = "LOCKED" if st.is_locked else "ACQUIRING..."
            status_col = (50, 240, 100) if st.is_locked else (240, 205, 40)

            s1 = self.font_mono.render(f"GT: ({gt_x:.1f}, {gt_y:.1f})  |  Est: ({st.est_world_x:.1f}, {st.est_world_y:.1f})", True, (220, 230, 240))
            s2 = self.font_mono.render(f"Range: {st.est_range:.0f} px  |  Bearing: {math.degrees(st.est_bearing):+.1f}°", True, (220, 230, 240))
            s3 = self.font_mono.render(f"Position Error:  {err_px:.1f} px", True, (220, 230, 240))
            s4 = self.font_mono.render("Tracking Status: ", True, (220, 230, 240))
            s_stat = self.font_mono.render(f"{st.target_id} {st.triage_level} [{status_str}]", True, status_col)

            self.screen.blit(s1, (hud_x + 24, curr_y + 38))
            self.screen.blit(s2, (hud_x + 24, curr_y + 62))
            self.screen.blit(s3, (hud_x + 24, curr_y + 86))
            self.screen.blit(s4, (hud_x + 24, curr_y + 110))
            self.screen.blit(s_stat, (hud_x + 24 + s4.get_width(), curr_y + 110))
        else:
            s1 = self.font_mono.render("GT: --  |  Est: SCANNING...", True, (175, 190, 205))
            s2 = self.font_mono.render("Range: SCANNING ARENA", True, (145, 160, 175))
            s3 = self.font_mono.render("Position Error:  ACQUIRING BEACON", True, (145, 160, 175))
            s4 = self.font_mono.render("Tracking Status: ", True, (175, 190, 205))
            s_stat = self.font_mono.render("SEARCH_PATROL", True, (0, 190, 240))

            self.screen.blit(s1, (hud_x + 24, curr_y + 38))
            self.screen.blit(s2, (hud_x + 24, curr_y + 62))
            self.screen.blit(s3, (hud_x + 24, curr_y + 86))
            self.screen.blit(s4, (hud_x + 24, curr_y + 110))
            self.screen.blit(s_stat, (hud_x + 24 + s4.get_width(), curr_y + 110))
        curr_y += 152

        # 6a. Rescue Team Status Panel
        arena_rescuers = getattr(self.arena, "rescuers", [])
        num_rescuers = len(arena_rescuers)
        team_h = 32 + max(1, num_rescuers) * 20 + 6
        team_h = max(team_h, 52)
        t_rect = pygame.Rect(hud_x + 16, curr_y, hud_w - 32, team_h)
        pygame.draw.rect(self.screen, (22, 27, 36), t_rect, border_radius=4)
        pygame.draw.rect(self.screen, (60, 100, 150), t_rect, 1, border_radius=4)
        t_title = self.font_label.render("RESCUE TEAM", True, (80, 180, 255))
        self.screen.blit(t_title, (hud_x + 24, curr_y + 8))
        row_y = curr_y + 30
        if num_rescuers == 0:
            no_r = self.font_small.render("No rescuers dispatched", True, (110, 120, 140))
            self.screen.blit(no_r, (hud_x + 24, row_y))
        else:
            for resc in arena_rescuers:
                st_str = resc.state.name if hasattr(resc, "state") else "STANDBY"
                tid = getattr(resc, "assigned_survivor_id", None)
                target_str = f"\u2192 {tid}" if tid else "STANDBY"
                eta_str = ""
                r_path = getattr(resc, "path", [])
                if st_str == "EN_ROUTE" and r_path:
                    dist = len(r_path) * 40.0 / max(resc.speed, 1.0)
                    eta_str = f"  ETA~{dist:.0f}s"
                row_col = (60, 235, 120) if st_str in ("EXTRACTING", "SECURED") else (
                    (255, 165, 40) if st_str == "EN_ROUTE" else (140, 155, 175))
                r_surf = self.font_small.render(
                    f"  {resc.name}: {st_str} {target_str}{eta_str}", True, row_col)
                self.screen.blit(r_surf, (hud_x + 16, row_y))
                row_y += 20
        curr_y += team_h + 6

        # 6b. SOS Queue Panel (dispatch_manager.sos_queue, sorted by triage priority)
        dm = getattr(self.fsm, "dispatch_manager", None)
        sos_queue_raw = dm.sos_queue if dm else []
        sos_count = max(1, min(3, len(sos_queue_raw)))
        sos_h = 32 + sos_count * 20 + 6
        s_rect = pygame.Rect(hud_x + 16, curr_y, hud_w - 32, sos_h)
        pygame.draw.rect(self.screen, (30, 18, 18), s_rect, border_radius=4)
        pygame.draw.rect(self.screen, (180, 60, 60), s_rect, 1, border_radius=4)
        s_title = self.font_label.render("SOS QUEUE", True, (255, 100, 100))
        self.screen.blit(s_title, (hud_x + 24, curr_y + 8))
        row_y = curr_y + 30
        if not sos_queue_raw:
            empty_s = self.font_small.render("Queue empty \u2014 all handled", True, (60, 235, 120))
            self.screen.blit(empty_s, (hud_x + 24, row_y))
        else:
            sorted_sos = sorted(sos_queue_raw, key=lambda m: dm.triage_rank(m.triage))
            for i, msg in enumerate(sorted_sos):
                if i >= 3:
                    break
                rank = dm.triage_rank(msg.triage)
                triage_col = (255, 60, 60) if rank == 0 else (
                    (255, 190, 40) if rank == 1 else (140, 215, 140))
                q_surf = self.font_small.render(
                    f"  #{i+1} {msg.survivor_id} {msg.triage} [{msg.status}]", True, triage_col)
                self.screen.blit(q_surf, (hud_x + 16, row_y))
                row_y += 20
        curr_y += sos_h + 6

        # 6c. Mission Event Log (last 4 lines)
        log_h = hud_h - curr_y - 12
        log_h = max(log_h, 70)
        l_rect = pygame.Rect(hud_x + 16, curr_y, hud_w - 32, log_h)
        pygame.draw.rect(self.screen, (16, 18, 24), l_rect, border_radius=4)
        pygame.draw.rect(self.screen, (40, 45, 55), l_rect, 1, border_radius=4)

        l_title = self.font_label.render("MISSION EVENT LOG", True, (130, 230, 160))
        self.screen.blit(l_title, (hud_x + 24, curr_y + 10))

        # Show last 4 log entries (space-constrained)
        max_log_lines = max(1, (log_h - 38) // 22)
        recent_logs = self.fsm.mission_log[-max_log_lines:]
        log_y = curr_y + 38
        max_log_w = hud_w - 48
        for log_line in recent_logs:
            if "secured by Rescuer" in log_line or "SUCCESS" in log_line or "LOCKED" in log_line:
                col = (60, 245, 130)
            elif "[COMM]" in log_line or "[BEACON]" in log_line:
                col = (100, 220, 255)
            elif "Hazard" in log_line or "RECOVERY" in log_line:
                col = (255, 95, 95)
            elif "False positive" in log_line or "rejected" in log_line:
                col = (255, 225, 40)
            elif "detected" in log_line or "VERIFY" in log_line:
                col = (255, 215, 55)
            elif "A*" in log_line or "PLANNER" in log_line:
                col = (85, 210, 255)
            else:
                col = (175, 190, 205)

            display_line = log_line
            if self.font_log.size(display_line)[0] > max_log_w:
                while len(display_line) > 6 and self.font_log.size(display_line + "...")[0] > max_log_w:
                    display_line = display_line[:-1]
                display_line = display_line + "..."

            surf = self.font_log.render(display_line, True, col)
            self.screen.blit(surf, (hud_x + 24, log_y))
            log_y += 22

    def _render_mission_complete_overlay(self):
        """Renders a semi-transparent mission summary overlay on the arena."""
        ow, oh = 460, 330
        ox = (self.ARENA_WIDTH - ow) // 2
        oy = (self.ARENA_HEIGHT - oh) // 2

        overlay = pygame.Surface((ow, oh))
        overlay.fill((10, 14, 20))
        self.screen.blit(overlay, (ox, oy))

        # Border
        pygame.draw.rect(self.screen, (40, 240, 140), (ox, oy, ow, oh), 2, border_radius=6)

        # Title
        t1 = self.font_banner_title.render("MISSION COMPLETE", True, (40, 240, 140))
        self.screen.blit(t1, (ox + (ow - t1.get_width()) // 2, oy + 18))
        t2 = self.font_subtitle.render("Rakshak SAR System  //  Hack in Hills '26", True, (120, 145, 175))
        self.screen.blit(t2, (ox + (ow - t2.get_width()) // 2, oy + 55))

        # Separator
        pygame.draw.line(self.screen, (40, 60, 80), (ox + 24, oy + 75), (ox + ow - 24, oy + 75), 1)

        # Report stats
        rep = self.fsm.generate_mission_report() if hasattr(self.fsm, "generate_mission_report") else {}
        secured = rep.get("survivors_secured", self.fsm.missions_completed)
        total = rep.get("total_survivors", self.fsm.total_survivors)
        time_elapsed = rep.get("mission_time_sec", getattr(self.fsm, "mission_time", 0.0))
        scanned = rep.get("area_scanned_pct", 0)
        bat_used = rep.get("battery_used_pct", 0)
        battery_rem = max(0, int(100 - bat_used))
        hazards = rep.get("hazards_avoided", getattr(self.fsm, "hazards_avoided_count", 0))
        mean_err = rep.get("mean_localization_error_px", 0.0)

        lines = [
            (f"Survivors Secured:   {secured} / {total}", (60, 240, 130) if secured == total else (255, 190, 40)),
            (f"Mission Time:        {time_elapsed:.1f} s", (200, 215, 235)),
            (f"Area Scanned:        {scanned:.1f}%", (80, 210, 255)),
            (f"Battery Remaining:   {battery_rem}%", (200, 215, 235)),
            (f"Loc Error (mean):    {mean_err:.1f} px", (200, 215, 235)),
            (f"Triage Accuracy:     {rep.get('triage_accuracy_pct', 100.0):.0f}%", (80, 230, 180)),
            (f"Hazards Avoided:     {hazards}", (200, 215, 235)),
            ("Report:              mission_report.json", (100, 160, 200)),
        ]

        row_y = oy + 90
        for line_text, line_col in lines:
            s = self.font_mono.render(line_text, True, line_col)
            self.screen.blit(s, (ox + 30, row_y))
            row_y += 28

        # Bottom: dismiss hint
        hint = self.font_small.render("[R] Reset  |  Resetting in 3s...", True, (80, 95, 115))
        self.screen.blit(hint, (ox + (ow - hint.get_width()) // 2, oy + oh - 26))

    def _render_radar(self, cx: int, cy: int, radius: int = 50):
        """Renders polar 8-sensor proximity radar widget."""
        # Radar circles
        pygame.draw.circle(self.screen, (28, 35, 45), (cx, cy), radius)
        for r_step in [radius * 0.33, radius * 0.66, radius]:
            pygame.draw.circle(self.screen, (45, 55, 70), (cx, cy), int(r_step), 1)

        # Cross axes
        pygame.draw.line(self.screen, (45, 55, 70), (cx - radius, cy), (cx + radius, cy), 1)
        pygame.draw.line(self.screen, (45, 55, 70), (cx, cy - radius), (cx, cy + radius), 1)

        # Miniature robot at center
        pygame.draw.circle(self.screen, (220, 230, 240), (cx, cy), 5)
        pygame.draw.line(self.screen, (255, 80, 80), (cx, cy), (cx, cy - 8), 2)

        # 8 Sensor distance spokes
        for i, angle_offset in enumerate(self.robot.SENSOR_ANGLES):
            # In radar widget, robot heading is UP (-pi/2)
            radar_angle = -math.pi / 2.0 + angle_offset
            dist = self.robot.sensor_distances[i]
            frac = min(1.0, dist / self.robot.sensor_range)
            blip_r = frac * radius

            bx = cx + blip_r * math.cos(radar_angle)
            by = cy + blip_r * math.sin(radar_angle)

            # Color code
            if dist < 35.0:
                col = (255, 50, 50)
            elif dist < 65.0:
                col = (255, 210, 40)
            else:
                col = (70, 230, 70)

            pygame.draw.line(self.screen, (col[0], col[1], col[2], 120), (cx, cy), (int(bx), int(by)), 1)
            pygame.draw.circle(self.screen, col, (int(bx), int(by)), 3)

        radar_lbl = self.font_small.render("8-RADAR", True, (130, 145, 165))
        self.screen.blit(radar_lbl, (cx - 18, cy + radius + 4))

    def run(self):
        """Main application game loop with fixed time step."""
        dt = self.fixed_dt

        while self.running:
            self.handle_events()
            self.update(dt)
            self.render()
            if not self.demo_mode:
                self.clock.tick(self.fps)

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        pygame.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rakshak - Autonomous Search & Rescue Robot")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    parser.add_argument("--lite", action="store_true", help="Run in low-overhead lite mode")
    parser.add_argument("--demo", action="store_true", help="Run multi-layout video demo recording")
    parser.add_argument("--parallel", action="store_true", help="Run multi-agent parallel search mode")
    parser.add_argument("--priority", dest="priority", action="store_true", default=True, help="Enable Priority mode (default: ON)")
    parser.add_argument("--no-priority", dest="priority", action="store_false", help="Disable Priority mode (rescue as found)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible arena layout")
    args, unknown = parser.parse_known_args()

    app = RakshakApp(
        lite_mode=args.lite,
        demo_mode=args.demo,
        parallel_mode=args.parallel,
        fullscreen=args.fullscreen,
        priority_mode=args.priority,
        seed=args.seed,
    )
    app.run()


if __name__ == "__main__":
    main()
