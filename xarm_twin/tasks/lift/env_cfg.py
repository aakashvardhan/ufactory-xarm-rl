"""XArm6-PickPlace-v0: pick a cube off the table and carry it to a commanded goal position.

Built on Isaac Lab's manager-based lift task (same rewards and observations as Isaac-Lift-Cube-Franka-v0),
with the UFactory xArm6 + xArm gripper and a workspace matched to the xArm's reach.
"""
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from xarm_twin.assets import GRIPPER_CLOSED, GRIPPER_OPEN, TCP_OFFSET, XARM6_CFG


@configclass
class XArm6PickPlaceEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2048  # 16 GB laptop GPU
        self.scene.robot = XARM6_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint[1-6]"], scale=0.5, use_default_offset=True
        )
        # Only the drive joint is commanded; the five linkage joints follow through PhysX mimic joints.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["drive_joint"],
            open_command_expr={"drive_joint": GRIPPER_OPEN},
            close_command_expr={"drive_joint": GRIPPER_CLOSED},
        )
        self.commands.object_pose.body_name = "xarm_gripper_base_link"
        self.commands.object_pose.ranges.pos_x = (0.35, 0.55)
        self.commands.object_pose.ranges.pos_y = (-0.2, 0.2)
        self.commands.object_pose.ranges.pos_z = (0.15, 0.35)

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),  # 4.8 cm cube; the xArm gripper opens to about 8.6 cm
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        self.events.reset_object_position.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)}

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper/xarm_gripper_base_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, TCP_OFFSET]),
                ),
            ],
        )


@configclass
class XArm6PickPlaceEnvCfg_PLAY(XArm6PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------------------------
# IK variant used by the scripted expert and for recording demonstrations (absolute TCP pose).
# ---------------------------------------------------------------------------------------------
@configclass
class XArm6PickPlaceIKEnvCfg(XArm6PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            body_name="xarm_gripper_base_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, TCP_OFFSET]),
        )
