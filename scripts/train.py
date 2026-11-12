r"""Run Isaac Lab's rsl_rl train.py with the xarm_twin tasks registered.

Usage: C:\IsaacLab\.venv\Scripts\python.exe scripts\train.py --task XArm6-PickPlace-v0 --headless
"""
from _launch import run_rsl_rl

run_rsl_rl("train.py")
