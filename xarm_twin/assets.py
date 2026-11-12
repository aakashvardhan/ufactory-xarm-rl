"""xArm6 + UFactory xArm gripper articulation config.

Drive gains: Isaac Lab's implicit PD (Nm/rad). The stock USD authors per-degree gains with almost no
damping (e.g. joint2 k=114 Nm/deg, d=0.046 Nm·s/deg, about 1% of critical), which rings on any step.
These values are the starting point that M3 calibration replaces with fitted ones.
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

USD_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets", "usd", "xarm6_gripper.usd"))

# effort limits from the asset (Nm): j1 50, j2 50, j3 32, j4 32, j5 32, j6 20
XARM6_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=False,
        # Gravity off on the robot links models the real xArm controller, which compensates gravity in its
        # position loop. With gravity on, implicit PD at k=800 sags 1.3–1.5 deg at joints 2–3 (measured, M0b).
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -0.26,  # top-down home from IK (M0b): TCP at (0.39, 0, 0.23), gripper z down
            "joint3": -0.93,
            "joint4": 0.0,
            "joint5": 1.24,
            "joint6": 0.0,
            "drive_joint": 0.0,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"], effort_limit_sim=50.0, velocity_limit_sim=3.14, stiffness=800.0, damping=80.0
        ),
        "elbow_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-5]"], effort_limit_sim=32.0, velocity_limit_sim=3.14, stiffness=600.0, damping=60.0
        ),
        "wrist6": ImplicitActuatorCfg(
            joint_names_expr=["joint6"], effort_limit_sim=20.0, velocity_limit_sim=3.14, stiffness=400.0, damping=40.0
        ),
        # The xArm gripper squeezes with tens of newtons, not hundreds. At 20 Nm on the drive joint the
        # mimic-joint linkage, blocked by the cube, pushed back hard enough to saturate the arm and stall the
        # TCP 18 cm from its target while carrying (M1 trace). 2 Nm is the right order of magnitude.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["drive_joint"], effort_limit_sim=2.0, velocity_limit_sim=2.0, stiffness=40.0, damping=4.0
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Tool centre point: midpoint between the closed finger pads, measured along the gripper base z-axis.
# Finger body origins sit 0.101 m (open) / 0.114 m (closed) out; the pads extend about 4 cm further.
TCP_OFFSET = 0.15
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.85  # drive_joint limit is 48.7 deg
