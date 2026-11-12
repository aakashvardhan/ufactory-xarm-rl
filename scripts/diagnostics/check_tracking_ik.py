"""M0b: per-joint tracking at the home pose, TCP offset in the gripper frame, and an IK solve for a top-down home pose."""
import argparse, math, os, sys
from isaaclab.app import add_launcher_args, launch_simulation
parser = argparse.ArgumentParser(); add_launcher_args(parser)
args = parser.parse_args(); args.require_kit = True  # Kit PhysX for the gripper's mimic joints
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_inv, matrix_from_quat  # noqa: E402
from xarm_twin.assets import GRIPPER_CLOSED, GRIPPER_OPEN, TCP_OFFSET, XARM7_CFG  # noqa: E402

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
    with launch_simulation(sim_cfg, args):
        run(sim_cfg)

def run(sim_cfg):
    sim = sim_utils.SimulationContext(sim_cfg)
    robot = Articulation(XARM7_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    names = robot.joint_names
    def step(tgt, n):
        for _ in range(n):
            robot.actuators.target_command.set_position_index(value=tgt); robot.write_data_to_sim(); sim.step(); robot.update(0.01)
    tgt = robot.data.default_joint_pos.torch.clone()
    step(tgt, 300)
    e = (robot.data.joint_pos.torch - tgt)[0]
    print("TRACK", {names[i]: round(math.degrees(e[i].item()), 2) for i in range(8)})
    print("TORQUE", {names[i]: round(robot.actuators.applied_effort.torch[0, i].item(), 2) for i in range(8)})
    bi = robot.body_names.index("xarm_gripper_base_link")
    gi = names.index("drive_joint")
    for label, g in (("open", GRIPPER_OPEN), ("closed", GRIPPER_CLOSED)):
        t2 = tgt.clone(); t2[:, gi] = g; step(t2, 200)
        bp = robot.data.body_pos_w.torch[0, bi]; bq = robot.data.body_quat_w.torch[0, bi]
        lf = robot.data.body_pos_w.torch[0, robot.body_names.index("left_finger")]
        rf = robot.data.body_pos_w.torch[0, robot.body_names.index("right_finger")]
        mid = 0.5 * (lf + rf)
        local = quat_apply(quat_inv(bq), mid - bp)
        print(f"TCP_{label} finger-mid in gripper frame:", [round(v, 4) for v in local.tolist()], "finger gap", round((lf - rf).norm().item(), 4))
    R = matrix_from_quat(bq)
    print("GRIPPER_AXES_WORLD x,y,z cols:", [[round(v, 3) for v in R[:, k].tolist()] for k in range(3)])
    # IK to a top-down pose using the gripper base body; target TCP offset applied below
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=args.device)
    arm = list(range(7))
    jac_idx = bi - 1  # fixed base: jacobian body index offset by one
    tcp_off = TCP_OFFSET  # same offset the env and the expert use
    goal_tcp = torch.tensor([[0.40, 0.0, 0.25]], device=args.device)
    goal_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=args.device)  # x,y,z,w: gripper z-axis pointing down (180 deg about x)
    goal_base = goal_tcp - quat_apply(goal_q, torch.tensor([[0.0, 0.0, tcp_off]], device=args.device))
    ik.set_command(torch.cat([goal_base, goal_q], dim=-1))
    t3 = tgt.clone(); t3[:, gi] = GRIPPER_OPEN
    for it in range(400):
        jac = robot.data.body_link_jacobian_w.torch[:, jac_idx, :, arm]
        bpw = robot.data.body_pos_w.torch[:, bi]; bqw = robot.data.body_quat_w.torch[:, bi]
        rp, rq = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
        p_b, q_b = subtract_frame_transforms(rp, rq, bpw, bqw)
        qdes = ik.compute(p_b, q_b, jac, robot.data.joint_pos.torch[:, arm])
        t3[:, arm] = qdes
        step(t3, 1)
    bp = robot.data.body_pos_w.torch[0, bi]; bq = robot.data.body_quat_w.torch[0, bi]
    tcp = bp + quat_apply(bq, torch.tensor([0.0, 0.0, tcp_off], device=args.device))
    print("IK_HOME joints (rad):", [round(v, 4) for v in robot.data.joint_pos.torch[0, :7].tolist()])
    print("IK_HOME tcp:", [round(v, 4) for v in tcp.tolist()], "quat:", [round(v, 3) for v in bq.tolist()])
    print("CHECK2_OK")

if __name__ == "__main__":
    main()
