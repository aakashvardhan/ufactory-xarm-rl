"""M0: spawn the xArm7 + gripper in Isaac Lab headless and report joints, limits, bodies and TCP geometry."""
import argparse
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args()
args.require_kit = True  # Kit PhysX: the gripper linkage uses PhysX mimic joints

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import math  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from xarm_twin.assets import GRIPPER_CLOSED, XARM7_CFG  # noqa: E402


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
    with launch_simulation(sim_cfg, args):
        run(sim_cfg)


def run(sim_cfg):
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(XARM7_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    print("joints:", robot.joint_names)
    print("bodies:", robot.body_names)
    lim = robot.data.joint_pos_limits.torch[0]
    for i, n in enumerate(robot.joint_names):
        print(f"  {n:28s} [{math.degrees(lim[i,0]):8.1f}, {math.degrees(lim[i,1]):8.1f}] deg  default {math.degrees(robot.data.default_joint_pos.torch[0,i]):7.1f}")
    def bodies():
        p = robot.data.body_pos_w.torch[0]
        return {n: [round(v, 4) for v in p[i].tolist()] for i, n in enumerate(robot.body_names)}
    # settle at default pose, gripper open
    for _ in range(100):
        robot.actuators.target_command.set_position_index(value=robot.data.default_joint_pos.torch)
        robot.write_data_to_sim(); sim.step(); robot.update(0.01)
    b = bodies()
    for k in ("link6", "xarm_gripper_base_link", "left_finger", "right_finger"):
        print(f"open   {k:24s} {b.get(k)}")
    err = (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[0].abs().max().item()
    print(f"settle error at default pose: {math.degrees(err):.3f} deg")
    # close gripper
    tgt = robot.data.default_joint_pos.torch.clone()
    gi = robot.joint_names.index("drive_joint")
    tgt[:, gi] = GRIPPER_CLOSED
    for _ in range(150):
        robot.actuators.target_command.set_position_index(value=tgt)
        robot.write_data_to_sim(); sim.step(); robot.update(0.01)
    b = bodies()
    for k in ("left_finger", "right_finger"):
        print(f"closed {k:24s} {b.get(k)}")
    q = robot.data.joint_pos.torch[0]
    print("gripper joints closed (deg):", {n: round(math.degrees(q[i].item()), 1) for i, n in enumerate(robot.joint_names) if i >= 7})
    print("CHECK_ASSET_OK")


if __name__ == "__main__":
    main()
