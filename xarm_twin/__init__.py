"""xArm Twin: UFactory xArm7 + xArm gripper tasks for Isaac Lab.

Only gym registrations live here (no isaaclab imports), so this package can be imported
before the simulator app is launched.
"""
from . import tasks  # noqa: F401
