"""
sound.py - Audio and Sound Effects Subsystem for Rakshak SAR Simulation.

Handles pygame.mixer initialization, procedural synthesis of audio assets (zero external downloads),
sound loading, playback triggers for tactical mission events, and mute/unmute state management.

Events supported:
1. survivor_detected: Dual-frequency alert blip (880 Hz -> 1320 Hz)
2. rescuer_dispatched: Ascending three-tone radio dispatch chime (C5 -> E5 -> G5)
3. extraction_complete: Warm harmonic resolution chime (E5+B5 -> G5+E6)
4. hazard_warning: Pulsed dual-tone alert buzz (180 Hz)
5. mission_complete: Grand victory fanfare arpeggio with sustained major chord
"""

import os
import wave
import time
from typing import Optional, Any
import numpy as np
import pygame


class SoundManager:
    """Manages audio playback, asset synthesis, dedicated channels, and mute state for Rakshak."""

    def __init__(self, sounds_dir: str = "assets/sounds"):
        self.sounds_dir = sounds_dir
        self.enabled: bool = False
        self.muted: bool = False
        self.sounds = {}
        self.last_hazard_time: float = 0.0
        self.chan_sfx = None
        self.chan_siren = None
        self.siren_played_survivors = set()
        self.suppress_logs: bool = False

        # 1. Initialize mixer safely
        self._init_mixer()

        # 2. Ensure sound assets exist (procedural synthesis if missing)
        try:
            self._ensure_sound_files()
        except Exception as e:
            print(f"[AUDIO ERROR] Failed to generate sound assets: {e}")

        # 3. Load sounds if mixer is enabled
        if self.enabled:
            self._load_sounds()

    def _init_mixer(self):
        """Attempts to initialize pygame.mixer with >=16 channels and dedicated channels."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(16)
            self.chan_sfx = pygame.mixer.Channel(0)
            self.chan_siren = pygame.mixer.Channel(1)
            self.enabled = True
        except Exception as e:
            self.enabled = False
            self.chan_sfx = None
            self.chan_siren = None
            print(f"[AUDIO ERROR] pygame.mixer initialization failed: {e}")

    def _write_wav(self, filename: str, samples: np.ndarray, sample_rate: int = 44100):
        """Writes 16-bit PCM mono wav file."""
        samples = np.clip(samples, -1.0, 1.0)
        int_samples = (samples * 32767).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int_samples.tobytes())

    def _ensure_sound_files(self):
        """Generates procedural beep/chime/siren audio assets if not present."""
        os.makedirs(self.sounds_dir, exist_ok=True)
        sr = 44100

        # Helper note generator
        def note(freq: float, dur: float, amp: float = 0.45):
            t = np.linspace(0, dur, int(sr * dur), False)
            return amp * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))

        # 1. survivor_detected.wav
        p1 = os.path.join(self.sounds_dir, "survivor_detected.wav")
        if not os.path.exists(p1):
            t1 = np.linspace(0, 0.08, int(sr * 0.08), False)
            t2 = np.linspace(0, 0.10, int(sr * 0.10), False)
            s1 = 0.5 * np.sin(2 * np.pi * 880 * t1) * np.linspace(0.1, 1.0, len(t1))
            s2 = 0.5 * np.sin(2 * np.pi * 1320 * t2) * np.linspace(1.0, 0.0, len(t2))
            self._write_wav(p1, np.concatenate([s1, s2]))

        # 2. rescuer_dispatched.wav
        p2 = os.path.join(self.sounds_dir, "rescuer_dispatched.wav")
        if not os.path.exists(p2):
            c_note = lambda freq, dur: 0.5 * np.sin(2 * np.pi * freq * np.linspace(0, dur, int(sr * dur), False)) * np.hanning(int(sr * dur))
            self._write_wav(p2, np.concatenate([c_note(523.25, 0.08), c_note(659.25, 0.08), c_note(783.99, 0.10)]))

        # 3. extraction_complete.wav
        p3 = os.path.join(self.sounds_dir, "extraction_complete.wav")
        if not os.path.exists(p3):
            t_a = np.linspace(0, 0.16, int(sr * 0.16), False)
            t_b = np.linspace(0, 0.24, int(sr * 0.24), False)
            ch1 = 0.3 * (np.sin(2 * np.pi * 659.25 * t_a) + np.sin(2 * np.pi * 987.77 * t_a)) * np.linspace(0.1, 1.0, len(t_a))
            ch2 = 0.3 * (np.sin(2 * np.pi * 783.99 * t_b) + np.sin(2 * np.pi * 1318.51 * t_b)) * np.linspace(1.0, 0.0, len(t_b))
            self._write_wav(p3, np.concatenate([ch1, ch2]))

        # 4. hazard_warning.wav
        p4 = os.path.join(self.sounds_dir, "hazard_warning.wav")
        if not os.path.exists(p4):
            def buzz(dur):
                t = np.linspace(0, dur, int(sr * dur), False)
                b = 0.4 * np.sign(np.sin(2 * np.pi * 180 * t))
                return b * np.hanning(len(t))
            gap = np.zeros(int(sr * 0.04))
            self._write_wav(p4, np.concatenate([buzz(0.11), gap, buzz(0.13)]))

        # 5. mission_complete.wav
        p5 = os.path.join(self.sounds_dir, "mission_complete.wav")
        if not os.path.exists(p5):
            chord_t = np.linspace(0, 0.35, int(sr * 0.35), False)
            chord = 0.3 * (np.sin(2 * np.pi * 1046.5 * chord_t) + np.sin(2 * np.pi * 1318.5 * chord_t)) * np.linspace(1.0, 0.0, len(chord_t))
            self._write_wav(p5, np.concatenate([note(523.25, 0.09), note(659.25, 0.09), note(783.99, 0.11), chord]))

        # 6. siren_critical.wav: fast urgent two-tone hi-lo siren, 2 repeats (>= 1.5s)
        p6 = os.path.join(self.sounds_dir, "siren_critical.wav")
        if not os.path.exists(p6):
            t_half = np.linspace(0, 0.4, int(sr * 0.4), False)
            hi_tone = 0.5 * np.sin(2 * np.pi * 920 * t_half)
            lo_tone = 0.5 * np.sin(2 * np.pi * 620 * t_half)
            one_cycle = np.concatenate([hi_tone, lo_tone])
            crit_samples = np.concatenate([one_cycle, one_cycle])  # 1.6s
            self._write_wav(p6, crit_samples)

        # 7. siren_injured.wav: slower single siren, played once (>= 1.5s)
        p7 = os.path.join(self.sounds_dir, "siren_injured.wav")
        if not os.path.exists(p7):
            dur_inj = 1.6
            t_inj = np.linspace(0, dur_inj, int(sr * dur_inj), False)
            freq_mod = 550 + 150 * np.sin(2 * np.pi * (1.0 / (2 * dur_inj)) * t_inj)
            phase = 2 * np.pi * np.cumsum(freq_mod) / sr
            inj_samples = 0.5 * np.sin(phase) * np.sin(np.pi * t_inj / dur_inj)
            self._write_wav(p7, inj_samples)

        # 8. beep_stable.wav / siren_stable.wav: short single alert beep (0.15s) for STABLE survivors
        p8 = os.path.join(self.sounds_dir, "beep_stable.wav")
        t_beep = np.linspace(0, 0.15, int(sr * 0.15), False)
        beep_samples = 0.4 * np.sin(2 * np.pi * 880.0 * t_beep) * np.hanning(len(t_beep))
        if not os.path.exists(p8):
            self._write_wav(p8, beep_samples)
        p8_alias = os.path.join(self.sounds_dir, "siren_stable.wav")
        if not os.path.exists(p8_alias):
            self._write_wav(p8_alias, beep_samples)

    def _load_sounds(self):
        """Loads WAV sound files into Pygame Sound objects."""
        sound_keys = [
            "survivor_detected",
            "rescuer_dispatched",
            "extraction_complete",
            "hazard_warning",
            "mission_complete",
            "siren_critical",
            "siren_injured",
            "siren_stable",
            "beep_stable",
        ]
        for key in sound_keys:
            filepath = os.path.join(self.sounds_dir, f"{key}.wav")
            if os.path.exists(filepath):
                try:
                    self.sounds[key] = pygame.mixer.Sound(filepath)
                except Exception as e:
                    print(f"[AUDIO ERROR] Could not load sound file {filepath}: {e}")
            else:
                print(f"[AUDIO WARNING] Sound file not found: {filepath}")

    def toggle_mute(self) -> bool:
        """Toggles audio mute state. Immediately stops running sounds if muted. Returns current muted status."""
        self.muted = not self.muted
        if self.muted and self.enabled:
            try:
                pygame.mixer.stop()
            except Exception:
                pass
        return self.muted

    def set_muted(self, muted: bool):
        self.muted = muted
        if self.muted and self.enabled:
            try:
                pygame.mixer.stop()
            except Exception:
                pass

    def is_muted(self) -> bool:
        return self.muted

    def _audio_log(self, msg: str):
        if not getattr(self, "suppress_logs", False):
            print(msg)

    def is_enabled(self) -> bool:
        return self.enabled and pygame.mixer.get_init() is not None

    def play(
        self,
        sound_name: str,
        survivor_id: Optional[Any] = None,
        severity: Optional[str] = None,
    ):
        """Plays a registered sound on dedicated SFX channel if audio is enabled and not muted."""
        if self.muted:
            self._audio_log("[AUDIO] skipped: muted")
            return
        if not self.enabled or pygame.mixer.get_init() is None:
            self._audio_log("[AUDIO] skipped: no channel")
            return
        snd = self.sounds.get(sound_name)
        if snd is None:
            self._audio_log("[AUDIO] skipped: file missing")
            return

        chan = self.chan_sfx
        if chan is None:
            try:
                chan = pygame.mixer.find_channel()
            except Exception:
                chan = None
        if chan is None:
            self._audio_log("[AUDIO] skipped: no channel")
            return

        try:
            chan.play(snd)
            tag = ""
            if survivor_id is not None:
                sid = str(survivor_id)
                sid_str = sid if sid.startswith("S") else f"S{sid}"
                sev_str = f" {severity}" if severity else ""
                tag = f" {sid_str}{sev_str}"
            elif severity is not None:
                tag = f" {severity}"
            self._audio_log(f"[AUDIO] played: {sound_name}{tag}")
        except Exception:
            self._audio_log("[AUDIO] skipped: no channel")

    def play_siren(self, survivor_id: Any, severity: str):
        """
        Triggers emergency siren on dedicated channel (Channel 1).
        Per-survivor 'once' flag ensures siren only sounds once per survivor until secured or reset.
        - CRITICAL: fast urgent two-tone hi-lo siren, 2 repeats (>= 1.5s).
        - INJURED: slower single siren, played once (>= 1.5s).
        - STABLE: short beep only, no siren.
        """
        if self.muted:
            self._audio_log("[AUDIO] skipped: muted")
            return
        if not self.enabled or pygame.mixer.get_init() is None:
            self._audio_log("[AUDIO] skipped: no channel")
            return
        if self.chan_siren is None:
            self._audio_log("[AUDIO] skipped: no channel")
            return

        sev_upper = str(severity).upper() if severity else "INJURED"
        sid = str(survivor_id)
        sid_str = sid if sid.startswith("S") else f"S{sid}"

        # Per-survivor once flag check
        if sid_str in self.siren_played_survivors or sid in self.siren_played_survivors:
            self._audio_log("[AUDIO] skipped: flag set")
            return

        if sev_upper == "CRITICAL":
            sound_key = "siren_critical"
        elif sev_upper == "INJURED":
            sound_key = "siren_injured"
        elif sev_upper == "STABLE":
            sound_key = "beep_stable"
        else:
            sound_key = "siren_injured"

        snd = self.sounds.get(sound_key)
        if snd is None:
            self._audio_log("[AUDIO] skipped: file missing")
            return

        try:
            self.chan_siren.play(snd)
            self.siren_played_survivors.add(sid_str)
            self.siren_played_survivors.add(sid)
            if sev_upper != "STABLE":
                self._audio_log(f"[AUDIO] siren played: {sid_str} {sev_upper}")
            self._audio_log(f"[AUDIO] played: {sound_key} {sid_str} {sev_upper}")
        except Exception:
            self._audio_log("[AUDIO] skipped: no channel")

    def clear_siren_flag(self, survivor_id: Optional[Any] = None):
        """Clears per-survivor once flag for the specified survivor or all survivors if None."""
        if survivor_id is None:
            self.siren_played_survivors.clear()
        else:
            sid = str(survivor_id)
            sid_str = sid if sid.startswith("S") else f"S{sid}"
            self.siren_played_survivors.discard(sid)
            self.siren_played_survivors.discard(sid_str)

    def play_survivor_detected(self, survivor_id: Optional[Any] = None, severity: Optional[str] = None):
        self.play("survivor_detected", survivor_id, severity)

    def play_rescuer_dispatched(self, survivor_id: Optional[Any] = None, severity: Optional[str] = None):
        self.play("rescuer_dispatched", survivor_id, severity)

    def play_extraction_complete(self, survivor_id: Optional[Any] = None, severity: Optional[str] = None):
        self.play("extraction_complete", survivor_id, severity)

    def play_hazard_warning(self, cooldown: float = 2.0):
        now = time.time()
        if now - self.last_hazard_time >= cooldown:
            self.last_hazard_time = now
            self.play("hazard_warning")

    def play_mission_complete(self):
        self.play("mission_complete")


# Global singleton instance
_GLOBAL_SOUND_MANAGER = None


def get_sound_manager() -> SoundManager:
    """Returns singleton SoundManager instance."""
    global _GLOBAL_SOUND_MANAGER
    if _GLOBAL_SOUND_MANAGER is None:
        _GLOBAL_SOUND_MANAGER = SoundManager()
    return _GLOBAL_SOUND_MANAGER
