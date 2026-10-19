"""
tests/test_seed_reproducibility.py
Verifies Task 3: same --seed produces identical survivor positions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest
from app import RakshakApp


def _survivor_positions(seed):
    app = RakshakApp(fps=30, seed=seed)
    return [(s.x, s.y, s.triage) for s in app.arena.survivors]


def test_same_seed_same_positions():
    """Two apps with the same seed must have identical survivor layouts."""
    pos_a = _survivor_positions(42)
    pos_b = _survivor_positions(42)
    assert pos_a == pos_b, f"Seed 42 produced different layouts:\n{pos_a}\n{pos_b}"


def test_different_seeds_different_positions():
    """Different seeds should (almost always) produce different layouts."""
    pos_a = _survivor_positions(1)
    pos_b = _survivor_positions(2)
    assert pos_a != pos_b, "Seeds 1 and 2 produced identical layouts (very unlikely)"


def test_no_seed_does_not_crash():
    """Omitting seed should work and not raise."""
    app = RakshakApp(fps=30, seed=None)
    assert len(app.arena.survivors) > 0
