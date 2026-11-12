"""Shared launcher: run Isaac Lab's unified train/play commands with the xarm_twin tasks registered."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import xarm_twin  # noqa: E402,F401  (registers the gym ids; imports gymnasium only)

# Load tensordict's C extension before Kit starts. Imported after Kit (as rsl_rl does), _C.pyd
# crashed with an access violation on Isaac Sim 5.1 (pip, tensordict 0.13); kept as a cheap safeguard.
import torch  # noqa: E402,F401
import tensordict  # noqa: E402,F401

# Same for h5py: with the GUI experience, Kit loads its own HDF5 DLL first and h5py's extension then
# fails with "DLL load failed while importing _errors". Headless runs are not affected.
import h5py  # noqa: E402,F401


def run_rl(command: str) -> None:
    """Run Isaac Lab's unified `isaaclab train` or `isaaclab play` with rsl_rl as the default library."""
    from isaaclab_rl.entrypoints import run_play_cli, run_train_cli

    argv = sys.argv[1:]
    if not any(a == "--rl_library" or a.startswith("--rl_library=") for a in argv):
        argv = ["--rl_library", "rsl_rl", *argv]
    run = {"train": run_train_cli, "play": run_play_cli}[command]
    sys.exit(run(argv))
