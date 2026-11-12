r"""Run Isaac Lab's `isaaclab play` (rsl_rl by default) with the xarm_twin tasks registered.

Usage: C:\IsaacLab3\.venv\Scripts\python.exe scripts\play.py --task XArm6-PickPlace-v0 --checkpoint latest --viz kit
"""
from _launch import run_rl

run_rl("play")
