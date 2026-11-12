r"""Run Isaac Lab's `isaaclab train` (rsl_rl by default) with the xarm_twin tasks registered.

Usage: C:\IsaacLab3\.venv\Scripts\python.exe scripts\train.py --task XArm7-PickPlace-v0
"""
from _launch import run_rl

run_rl("train")
