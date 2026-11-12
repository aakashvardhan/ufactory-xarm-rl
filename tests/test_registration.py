"""Simulator-free checks: the package imports without Isaac Lab loaded and registers its gym ids."""
import os
import sys

import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import xarm_twin  # noqa: E402,F401

TASKS = ("XArm6-PickPlace-v0", "XArm6-PickPlace-IK-v0")


def test_tasks_registered():
    for task in TASKS:
        assert task in gym.registry, task


def test_entry_points_name_this_package():
    for task in TASKS:
        assert gym.registry[task].kwargs["env_cfg_entry_point"].startswith("xarm_twin.tasks.lift.env_cfg:")


def test_import_does_not_load_isaaclab():
    # Kit must be launched before any isaaclab module is imported, so registration has to stay lazy.
    assert not any(m == "isaaclab" or m.startswith("isaaclab.") for m in sys.modules)
