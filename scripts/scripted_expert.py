"""M1: scripted pick-and-place expert on XArm6-PickPlace-IK-v0 (absolute TCP pose + binary gripper).

A per-env state machine: hover over the cube, descend, close, lift, carry to the commanded goal.
Every finished episode is scored: success = the episode ran to its time-out with the cube within --tol
of the goal. An episode that terminates early (cube dropped off the table) counts as a failure. This is
the physics check for the grasp (does the mimic-joint gripper actually hold a 4.8 cm cube?) and the demo
source for SmolVLA.

  python scripts/scripted_expert.py --num_envs 64 --episodes 256 --headless
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch  # noqa: E402,F401
import tensordict  # noqa: E402,F401
import h5py  # noqa: E402,F401  (must load before Kit's own HDF5 DLL when running with the GUI)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--tol", type=float, default=0.03)
parser.add_argument("--hover", type=float, default=0.10)
parser.add_argument("--debug", action="store_true")
parser.add_argument(
    "--ik_test", action="store_true", help="ignore the cube; send the TCP straight to the goal and score the TCP"
)
parser.add_argument("--grasp_dz", type=float, default=0.0, help="TCP height relative to cube centre at grasp")
add_launcher_args(parser)
args = parser.parse_args()

import gymnasium as gym  # noqa: E402

import xarm_twin  # noqa: E402,F401
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "XArm6-PickPlace-IK-v0"
HOVER, DESCEND, GRASP, LIFT, CARRY = range(5)
DOWN = torch.tensor([1.0, 0.0, 0.0, 0.0])  # x,y,z,w: 180 deg about x, gripper z-axis pointing at the table

# State machine: a state ends when the TCP is within the distance tolerance or its time budget runs out.
HOVER_TOL, HOVER_TIMEOUT = 0.015, 1.2  # m, s
DESCEND_TOL, DESCEND_TIMEOUT = 0.008, 1.0  # m, s
GRASP_TIME = 0.5  # s, fingers closing
LIFT_TIME = 0.6  # s, straight up before carrying
DEBUG_EVERY = 5  # steps


def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    with launch_simulation(cfg, args):
        run(cfg)


def run(cfg):
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    dev = u.device
    n = u.num_envs
    robot = u.scene["robot"]
    down = DOWN.to(dev).repeat(n, 1)
    state = torch.zeros(n, dtype=torch.long, device=dev)
    timer = torch.zeros(n, device=dev)
    lift_xy = torch.zeros(n, 2, device=dev)
    dt = u.step_dt

    def in_root_frame(pos_w):
        # The IK action and the goal command are both expressed in the robot root frame.
        pos, _ = subtract_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, pos_w)
        return pos

    env.reset()
    done_eps, successes, dropped, final_err = 0, 0, 0, []
    while done_eps < args.episodes and u.sim.is_running():
        cube = in_root_frame(u.scene["object"].data.root_pos_w.torch)
        goal = u.command_manager.get_command("object_pose")[:, :3]
        tcp = in_root_frame(u.scene["ee_frame"].data.target_pos_w.torch[:, 0, :])
        target = cube.clone()
        target[:, 2] += args.hover
        grip = torch.ones(n, device=dev)  # +1 open, -1 close
        # state transitions
        d_hover = (tcp - target).norm(dim=-1)
        grasp_pt = cube.clone()
        grasp_pt[:, 2] += args.grasp_dz
        d_grasp = (tcp - grasp_pt).norm(dim=-1)
        timer += dt
        nxt = state.clone()
        nxt[(state == HOVER) & ((d_hover < HOVER_TOL) | (timer > HOVER_TIMEOUT))] = DESCEND
        nxt[(state == DESCEND) & ((d_grasp < DESCEND_TOL) | (timer > DESCEND_TIMEOUT))] = GRASP
        nxt[(state == GRASP) & (timer > GRASP_TIME)] = LIFT
        nxt[(state == LIFT) & (timer > LIFT_TIME)] = CARRY
        entering_lift = (nxt == LIFT) & (state != LIFT)
        lift_xy[entering_lift] = cube[entering_lift, :2]
        timer[nxt != state] = 0.0
        state = nxt
        # targets per state
        tgt = target.clone()
        m = state == DESCEND
        tgt[m] = grasp_pt[m]
        m = state == GRASP
        tgt[m] = grasp_pt[m]
        grip[m] = -1
        m = state == LIFT
        tgt[m] = torch.cat([lift_xy[m], goal[m, 2:3]], -1)
        grip[m] = -1
        m = state == CARRY
        tgt[m] = goal[m]
        grip[m] = -1
        if args.ik_test:
            tgt = goal.clone()
            grip = torch.ones(n, device=dev)
        action = torch.cat([tgt, down, grip.unsqueeze(-1)], dim=-1)
        if args.debug and int(u.episode_length_buf[0]) % DEBUG_EVERY == 0:
            gi = robot.joint_names.index("drive_joint")
            print(f"DBG t={int(u.episode_length_buf[0]):3d} st={int(state[0])} tcp={[round(v,3) for v in tcp[0].tolist()]} "
                  f"cube={[round(v,3) for v in cube[0].tolist()]} goal={[round(v,3) for v in goal[0].tolist()]} "
                  f"drive={robot.data.joint_pos.torch[0, gi].item():.3f} tau={[round(v,1) for v in robot.actuators.applied_effort.torch[0,:7].tolist()]} q={[round(v,2) for v in robot.data.joint_pos.torch[0,:6].tolist()]} qt={[round(v,2) for v in robot.actuators.target_command.position.torch[0,:6].tolist()]}", flush=True)
        # Envs that finish on this step are reset inside env.step, so the error is taken from the state
        # just before it (one control step old), against the goal that episode was given.
        err = ((tcp if args.ik_test else cube) - goal).norm(dim=-1)
        _, _, terminated, truncated, _ = env.step(action)
        finished = (terminated | truncated).nonzero(as_tuple=False).flatten()
        finished = finished[: args.episodes - done_eps]  # never score more than --episodes
        if len(finished) > 0:
            early = terminated[finished]
            final_err += err[finished].tolist()
            successes += int((~early & (err[finished] < args.tol)).sum())
            dropped += int(early.sum())
            done_eps += len(finished)
        reset = u.episode_length_buf == 0
        state[reset] = HOVER
        timer[reset] = 0.0
    label = "IK_TEST (TCP vs goal)" if args.ik_test else "EXPERT"
    if final_err:
        errs = torch.tensor(final_err)
        print(f"{label} episodes={done_eps} success={successes / done_eps:.3f} early_terminations={dropped} "
              f"median_err={errs.median().item():.4f} p90_err={errs.quantile(0.9).item():.4f} tol={args.tol}")
    else:
        print(f"{label} episodes=0: no episode finished before the app closed")
    env.close()


if __name__ == "__main__":
    main()
