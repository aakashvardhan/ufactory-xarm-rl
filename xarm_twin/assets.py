"""xArm7 + UFactory xArm gripper articulation config.

The lab's two arms are xArm 7s (controller type 7, XS1305, firmware v2.7.1, read from 192.168.1.241).

Drive gains: Isaac Lab's implicit PD (Nm/rad). The stock USD authors per-degree gains with almost no
damping, which rings on any step. These values are the starting point that M3 calibration replaces with
fitted ones.
"""
import os

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

USD_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets", "usd", "xarm7_gripper.usd"))

ARM_JOINTS = "joint[1-7]"

# effort limits from the asset (Nm): j1 50, j2 50, j3 30, j4 30, j5 30, j6 20, j7 20
XARM7_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=False,
        # Gravity off on the robot links models the real xArm controller, which compensates gravity in its
        # position loop. With gravity on, implicit PD at k=800 sagged 1.3–1.5 deg at the shoulder on the xArm6 (M0b).
        # PhysX only: the gripper linkage relies on PhysX mimic joints.
        rigid_props=PhysxRigidBodyCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # top-down home from IK (scripts/diagnostics/check_tracking_ik.py): TCP at (0.40, 0, 0.25),
            # gripper z down, joints 1, 3, 5 and 7 at zero
            "joint1": 0.0,
            "joint2": -0.259,
            "joint3": 0.0,
            "joint4": 0.962,
            "joint5": 0.0,
            "joint6": 1.221,
            "joint7": 0.0,
            "drive_joint": 0.0,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"], joint_effort_limit=50.0, joint_velocity_limit=3.14, stiffness=800.0, damping=80.0
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-5]"], joint_effort_limit=30.0, joint_velocity_limit=3.14, stiffness=600.0, damping=60.0
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[6-7]"], joint_effort_limit=20.0, joint_velocity_limit=3.14, stiffness=400.0, damping=40.0
        ),
        # The xArm gripper squeezes with tens of newtons, not hundreds. At 20 Nm on the drive joint the
        # mimic-joint linkage, blocked by the cube, pushed back hard enough to saturate the arm and stall the
        # TCP 18 cm from its target while carrying (M1 trace). 2 Nm is the right order of magnitude.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["drive_joint"], joint_effort_limit=2.0, joint_velocity_limit=2.0, stiffness=40.0, damping=4.0
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Tool centre point along the gripper base z-axis, from the real controller's TCP offset (0, 0, 172 mm),
# read from 192.168.1.241. The gripper base link sits on the flange, so the two frames share an origin.
TCP_OFFSET = 0.172
# Tool payload from the same controller: 0.82 kg, centre of mass 48 mm out along z.
TCP_LOAD_KG, TCP_LOAD_COM = 0.82, (0.0, 0.0, 0.048)
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.85  # drive_joint limit is 48.7 deg
