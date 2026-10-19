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


def test_benchmark_same_seed_identical_results():
    """Verifies that running execute_benchmark with the same seed yields identical results."""
    from tests.test_random_layouts import execute_benchmark
    res_a = execute_benchmark(num_layouts=2, max_frames=500, seed=42)
    res_b = execute_benchmark(num_layouts=2, max_frames=500, seed=42)
    assert res_a["reach_rate"] == res_b["reach_rate"]
    assert res_a["median_lock_error"] == res_b["median_lock_error"]
    assert res_a["triage_accuracy"] == res_b["triage_accuracy"]


def test_benchmark_different_seeds_different_layouts():
    """Verifies that different base seeds produce different layouts."""
    from tests.test_random_layouts import run_single_layout
    layout_a = run_single_layout(seed=10, max_frames=50)
    layout_b = run_single_layout(seed=20, max_frames=50)
    assert layout_a["survivor_pos"] != layout_b["survivor_pos"]


def test_twenty_benchmark_layouts_are_all_distinct():
    """Verifies that all 20 benchmark layouts (seed = base_seed + i) have distinct survivor locations."""
    from arena import DisasterArena
    from tests.test_random_layouts import DEFAULT_BASE_SEED
    positions = set()
    for i in range(20):
        arena = DisasterArena(width=800, height=650)
        arena.randomize_layout(seed=DEFAULT_BASE_SEED + i)
        positions.add((arena.survivor.x, arena.survivor.y))
    assert len(positions) == 20, f"Expected 20 distinct survivor positions, got {len(positions)}"
