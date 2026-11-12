"""Shared launcher: run one of Isaac Lab's rsl_rl scripts with the xarm_twin tasks registered."""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import xarm_twin  # noqa: E402,F401  (registers the gym ids; imports gymnasium only)

# Load tensordict's C extension before Kit starts. Imported after Kit (as rsl_rl does), _C.pyd
# crashes with an access violation on this Windows setup (Isaac Sim 5.1 pip, tensordict 0.13).
import torch  # noqa: E402,F401
import tensordict  # noqa: E402,F401


def isaaclab_path() -> str:
    path = os.environ.get("ISAACLAB_PATH")
    if path:
        return path
    default = r"C:\IsaacLab" if os.name == "nt" else os.path.expanduser("~/IsaacLab")
    if not os.path.isdir(default):
        sys.exit(f"Isaac Lab not found at {default}; set ISAACLAB_PATH to your Isaac Lab checkout.")
    return default


def run_rsl_rl(script_name: str) -> None:
    """Run scripts/reinforcement_learning/rsl_rl/<script_name> from the Isaac Lab checkout."""
    script = os.path.join(isaaclab_path(), "scripts", "reinforcement_learning", "rsl_rl", script_name)
    if not os.path.isfile(script):
        sys.exit(f"Isaac Lab script not found: {script}")
    sys.path.insert(0, os.path.dirname(script))
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
