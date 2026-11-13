"""Depth cameras of the lab-bench scene: one source for scripts/build_scene.py and the Isaac Lab configs.

Pure Python, no Isaac imports. Overhead height is measured; horizontal placement and wrist pose are estimates.

Quaternions are (w, x, y, z), the order USD's Gf.Quatd takes, in the USD camera convention: the camera
looks along its -Z axis and +Y is image up. Isaac Lab 3 takes (x, y, z, w): pass xyzw(q) to
CameraCfg.OffsetCfg with convention="opengl".
"""
import math

# Intel RealSense D435 colour lens, 69 x 42 degrees. Depth is rendered through the same lens, like the
# real camera's depth stream aligned to colour.
FOCAL_LENGTH = 1.93
HORIZONTAL_APERTURE = 2.65
VERTICAL_APERTURE = 1.49
WIDTH, HEIGHT = 640, 360
CLIPPING_RANGE = (0.02, 10.0)
DATA_TYPES = ["rgb", "distance_to_image_plane"]
BODY_SIZE = (0.090, 0.025, 0.025)  # D435 housing; long side along the image's horizontal axis
BODY_BEHIND = 0.015  # the housing centre sits this far behind the lens, so the camera never sees it


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def xyzw(q):
    """(w, x, y, z) -> (x, y, z, w), the order Isaac Lab 3 uses."""
    return (q[1], q[2], q[3], q[0])


def look_quat(direction, up):
    """(w, x, y, z) of a camera looking along `direction` with image-up as close to `up` as possible."""
    z = tuple(-c for c in _unit(direction))
    x = _unit(_cross(up, z))
    y = _cross(z, x)
    m = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = (s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2 * math.sqrt(1 + m[0][0] - m[1][1] - m[2][2])
        q = ((m[2][1] - m[1][2]) / s, s / 4, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] > m[2][2]:
        s = 2 * math.sqrt(1 + m[1][1] - m[0][0] - m[2][2])
        q = ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, s / 4, (m[1][2] + m[2][1]) / s)
    else:
        s = 2 * math.sqrt(1 + m[2][2] - m[0][0] - m[1][1])
        q = ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, s / 4)
    return q


# Wrist camera, in the frame of the gripper base link (z runs along the fingers, the fingers open along y).
# It sits beside the gripper body and looks along the fingers, tilted 15 degrees toward them. Image-up
# points back toward the gripper body (-x).
WRIST_PARENT = "gripper/xarm_gripper_base_link"  # below the robot's root prim
WRIST_POS = (0.070, 0.0, 0.030)
WRIST_TILT = math.radians(15.0)
WRIST_QUAT = look_quat((-math.sin(WRIST_TILT), 0.0, math.cos(WRIST_TILT)), (-1.0, 0.0, 0.0))

# Overhead camera in the bench frame, on the Tarion TKA01-L stand and tilted toward the workspace.
# Image-up approximately follows -X, toward the opposite long edge.
# Photo refinement: clamp and robot are on the same (+X) long edge. Clamp is 56 cm from the photo's
# reference J1 at X=0.583 m; subsequent robot-base translations keep the camera stand fixed.
# The boom reaches inward (-X) and left (-Y) at 45 degrees; lens height remains the measured 81.2 cm.
# Its 42 cm horizontal reach and optical aim are inferred from the side photos, not calibrated extrinsics.
BASE_TO_CAMERA_MOUNT = 0.560
STAND_BASE_XY = (0.774, math.sqrt(BASE_TO_CAMERA_MOUNT**2 - (0.774 - 0.583)**2))
STAND_YAW_LEFT_DEG = 45.0
_boom_direction = (-math.cos(math.radians(STAND_YAW_LEFT_DEG)),
                   -math.sin(math.radians(STAND_YAW_LEFT_DEG)), 0.0)
OVERHEAD_POS = (STAND_BASE_XY[0] + 0.42 * _boom_direction[0],
                STAND_BASE_XY[1] + 0.42 * _boom_direction[1], 0.812)
OVERHEAD_QUAT = look_quat((-0.35, 0.20, -0.812), (-1.0, 0.0, 0.0))
