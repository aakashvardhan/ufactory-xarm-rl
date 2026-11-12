# xarm_twin

UFactory xArm6 + xArm gripper tasks for Isaac Lab: a cube pick-and-place task trained with rsl_rl PPO,
plus a scripted IK expert used as a grasp check and demonstration source.

## Requirements

- Isaac Sim 5.1 and an Isaac Lab checkout with its Python environment. The scripts look for Isaac Lab at
  `ISAACLAB_PATH`, falling back to `C:\IsaacLab` (Windows) or `~/IsaacLab`.
- Network access on first use. Two assets are loaded by URL from NVIDIA's asset server and cached by
  Omniverse afterwards:
  - `assets/usd/xarm6_gripper.usd` is a thin override layer; it references the stock xArm6 asset under
    `Assets/Isaac/5.1/Isaac/Robots/Ufactory/xarm6/xarm6.usd`.
  - The cube is `Props/Blocks/DexCube/dex_cube_instanceable.usd` from `ISAAC_NUCLEUS_DIR`.

Run everything with Isaac Lab's Python, for example `C:\IsaacLab\.venv\Scripts\python.exe`. The scripts
add the repository to `sys.path` themselves; `pip install -e .` is optional.

## Tasks

| Gym id | Actions | Use |
| --- | --- | --- |
| `XArm6-PickPlace-v0` | joint position targets + binary gripper | PPO training |
| `XArm6-PickPlace-Play-v0` | same, 16 envs, no observation noise | playing a checkpoint |
| `XArm6-PickPlace-IK-v0` | absolute TCP pose + binary gripper | scripted expert, demonstrations |

Rewards, observations and terminations are inherited from Isaac Lab's `Isaac-Lift-Cube-Franka-v0`.

## Usage

```
python scripts/train.py --task XArm6-PickPlace-v0 --headless
python scripts/play.py --task XArm6-PickPlace-Play-v0
python scripts/scripted_expert.py --num_envs 64 --episodes 256 --headless
```

`scripted_expert.py` scores every finished episode. An episode succeeds if it runs to its time-out with
the cube within `--tol` of the goal; an episode that ends early because the cube fell counts as a failure.

Other scripts:

- `scripts/make_asset.py` rebuilds `assets/usd/xarm6_gripper.usd` from the stock asset.
- `scripts/diagnostics/check_asset.py` prints joints, limits, bodies and finger positions.
- `scripts/diagnostics/check_tracking_ik.py` reports joint tracking error, finger geometry in the gripper
  frame, and an IK solve for the top-down home pose.

## Tests

```
python -m pytest
```

These tests do not start the simulator. They check that the package imports without Isaac Lab loaded and
that the gym ids are registered.

## Modelling notes

- Gravity is disabled on the robot links to stand in for the real controller's gravity compensation.
  Joint torques in simulation therefore carry no gravity load.
- Drive gains in `xarm_twin/assets.py` are starting values, not yet fitted to the real arm.
- `TCP_OFFSET`, `GRIPPER_OPEN` and `GRIPPER_CLOSED` in `xarm_twin/assets.py` are the single source for the
  tool geometry; the env, the expert and the diagnostics all import them.
