"""
run.py - Single command entry point for Rakshak Search-and-Rescue Robot.

Usage:
  python3 run.py               # Launches interactive GUI application
  python3 run.py --benchmark   # Runs headless autonomous simulation benchmark and reports metrics
"""

import sys
import os
from typing import Optional
import subprocess
import argparse


def ensure_venv_execution():
    """
    Ensures Rakshak runs with its configured virtual environment dependencies.
    If executed with system Python, transparently delegates to .venv/Scripts/python.exe.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python_win = os.path.join(script_dir, ".venv", "Scripts", "python.exe")
    venv_python_nix = os.path.join(script_dir, ".venv", "bin", "python")

    venv_python = venv_python_win if os.path.exists(venv_python_win) else venv_python_nix

    # Check if we are already inside the venv
    is_in_venv = (sys.prefix != sys.base_prefix) or (os.path.normcase(sys.executable) == os.path.normcase(venv_python))

    if not is_in_venv and os.path.exists(venv_python):
        # Re-execute seamlessly inside virtual environment
        result = subprocess.run([venv_python] + sys.argv, cwd=script_dir)
        sys.exit(result.returncode)


def run_benchmark(num_layouts: int = 20, max_frames: int = 3000, seed: Optional[int] = None):
    """
    Headless benchmark mode across 20 randomized disaster layouts.
    Validates perception, planning, collision avoidance, and localization accuracy.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from tests.test_random_layouts import execute_benchmark
    return execute_benchmark(num_layouts=num_layouts, max_frames=max_frames, seed=seed)


def main():
    ensure_venv_execution()

    parser = argparse.ArgumentParser(description="Rakshak: Autonomous Search-and-Rescue Robot")
    parser.add_argument("--benchmark", action="store_true", help="Run headless autonomous benchmark across 20 layouts")
    parser.add_argument("--demo", action="store_true", help="Run 3 layouts back to back and record demo.mp4 (1280x720, 30 fps)")
    parser.add_argument("--lite", action="store_true", help="Enable lightweight low-resource rendering mode with fixed time step")
    parser.add_argument("--parallel", action="store_true", help="Scout continues searching immediately after sending SOS (concurrent rescue)")
    parser.add_argument("--layouts", type=int, default=20, help="Number of random layouts for benchmark (default: 20)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible arena layout")
    parser.add_argument("--max-frames", type=int, default=3000, help="Max frames per benchmark layout (default: 3000)")
    parser.add_argument("--fps", type=int, default=60, help="Target simulation frames per second (default: 60)")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(num_layouts=args.layouts, max_frames=args.max_frames, seed=args.seed)
    elif args.demo:
        from app import RakshakApp
        app = RakshakApp(fps=30, demo_mode=True, demo_output="demo.mp4", demo_layouts=3, parallel_mode=args.parallel)
        app.run()
    else:
        from app import RakshakApp
        app = RakshakApp(fps=args.fps, lite_mode=args.lite, parallel_mode=args.parallel, seed=getattr(args, 'seed', None))
        app.run()


if __name__ == "__main__":
    main()
