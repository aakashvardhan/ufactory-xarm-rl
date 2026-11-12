r"""Run Isaac Lab's rsl_rl play.py with the xarm_twin tasks registered.

Usage: C:\IsaacLab\.venv\Scripts\python.exe scripts\play.py --task XArm6-PickPlace-Play-v0 --headless
"""
from _launch import run_rsl_rl

run_rsl_rl("play.py")
