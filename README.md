# xarm_twin

UFactory xArm6 + xArm gripper tasks for Isaac Lab: a cube pick-and-place task trained with rsl_rl PPO,
plus a scripted IK expert used as a grasp check and demonstration source.

## Requirements

- Isaac Lab 3.0.0 with Isaac Sim 6.2, Python 3.12, PyTorch 2.11 (CUDA 13.0) and RSL-RL 5.5.1. NVIDIA
  driver 580.88 or newer on Windows. Isaac Lab 2.x / Isaac Sim 5.1 are no longer supported: 3.0 changed
  the quaternion order to (x, y, z, w) and made asset data `ProxyArray`s, and the code here follows that.
- Network access on first use. Two assets are loaded by URL from NVIDIA's asset server and cached by
  Omniverse afterwards:
  - `assets/usd/xarm6_gripper.usd` is a thin override layer; it references the stock xArm6 asset under
    `Assets/Isaac/5.1/Isaac/Robots/Ufactory/xarm6/xarm6.usd`.
  - The cube is `Props/Blocks/DexCube/dex_cube_instanceable.usd` from `ISAAC_NUCLEUS_DIR`.

### Setup (Windows)

```
git clone --branch v3.0.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git C:\IsaacLab3
cd C:\IsaacLab3
uv sync --extra isaacsim --extra video
uv pip install --python .venv\Scripts\python.exe -e <path to this repo> pytest
```

`uv sync` removes packages that are not in Isaac Lab's lock file, so rerun the `uv pip install` line after
every sync. Installing this package registers the tasks with `isaaclab train/play/list_envs` through the
`isaaclab.tasks` entry point. Isaac Sim asks to accept the NVIDIA Omniverse EULA on first start; set
`OMNI_KIT_ACCEPT_EULA=YES` to accept it non-interactively.

Run everything with that environment's Python, `C:\IsaacLab3\.venv\Scripts\python.exe`. When redirecting
output to a file on Windows, also set `PYTHONUTF8=1`: Isaac Lab's start-up summary prints characters the
default code page cannot encode.

## Tasks

| Gym id | Actions | Use |
| --- | --- | --- |
| `XArm6-PickPlace-v0` | joint position targets + binary gripper | PPO training and playback |
| `XArm6-PickPlace-IK-v0` | absolute TCP pose + binary gripper | scripted expert, demonstrations |

Rewards, observations and terminations are inherited from Isaac Lab's `IsaacContrib-Lift-Cube-Franka`
(`isaaclab_tasks.contrib.lift`). Playback uses the training task id: `isaaclab play` applies the env
config's `play_mode()` (16 envs, no observation noise) unless `--train_env_cfg` is passed. A checkpoint
selector such as `--checkpoint latest` only finds runs trained under the same task id.

## Usage

```
python scripts/train.py --task XArm6-PickPlace-v0
python scripts/play.py --task XArm6-PickPlace-v0 --checkpoint latest --viz kit
python scripts/scripted_expert.py --num_envs 64 --episodes 256
```

`train.py` and `play.py` run Isaac Lab's `isaaclab train` / `isaaclab play` with `--rl_library rsl_rl`, so
they take the same flags (`--num_envs`, `--max_iterations`, `--checkpoint`, `--video`, ...). There is no
`--headless` flag any more: a run without `--viz` is headless, and `--viz kit` opens the Isaac Sim window.

`scripted_expert.py` scores every finished episode. An episode succeeds if it runs to its time-out with
the cube within `--tol` of the goal; an episode that ends early because the cube fell counts as a failure.

Other scripts:

- `scripts/make_asset.py` rebuilds `assets/usd/xarm6_gripper.usd` from the stock asset.
- `scripts/diagnostics/check_asset.py` prints joints, limits, bodies and finger positions.
- `scripts/diagnostics/check_tracking_ik.py` reports joint tracking error, finger geometry in the gripper
  frame, and an IK solve for the top-down home pose.
- `scripts/build_scene.py` writes the lab-bench scene into `assets/usd/`: `lab_bench.usd` (bench, mount
  plate, wall, floor, and the overhead depth camera on its Tarion TKA01-L desk-mount stand), `lab_lights.usd`, `paper_cube.usd` (5.2 cm
  cube), `food_bowl.usd` (square container with five box colliders) and `lab_scene.usd`, which assembles them
  with the xArm6 and its wrist depth camera for viewing in Isaac Sim. Only the cube edge is measured; the
  other sizes are estimates set by the constants at the top of the script. The pick-and-place tasks do not
  use this scene yet.
- `xarm_twin/cameras.py` holds the lens (RealSense D435 colour, 640 x 360) and the poses of the wrist and
  overhead depth cameras. `build_scene.py` and the Isaac Lab camera configs both read it.
- `scripts/diagnostics/check_scene.py` loads that scene, checks resting heights, a cube drop into the bowl
  and what both depth cameras see, and writes four images to `logs/`.

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
