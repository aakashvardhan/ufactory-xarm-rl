# xarm_twin

UFactory xArm7 + xArm gripper tasks for Isaac Lab: a cube pick-and-place task trained with rsl_rl PPO,
plus a scripted IK expert used as a grasp check and demonstration source.

## Requirements

- Isaac Lab 3.0.0 with Isaac Sim 6.2, Python 3.12, PyTorch 2.11 (CUDA 13.0) and RSL-RL 5.5.1. NVIDIA
  driver 580.88 or newer on Windows. Isaac Lab 2.x / Isaac Sim 5.1 are no longer supported: 3.0 changed
  the quaternion order to (x, y, z, w) and made asset data `ProxyArray`s, and the code here follows that.
- Network access on first use. Two assets are loaded by URL from NVIDIA's asset server and cached by
  Omniverse afterwards:
  - `assets/usd/xarm7_gripper.usd` is a thin override layer; it references the stock xArm7 asset under
    `Assets/Isaac/6.2/Isaac/Robots/Ufactory/xarm7/xarm7.usd`.
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
| `XArm7-PickPlace-v0` | joint position targets + binary gripper | PPO training and playback |
| `XArm7-PickPlace-IK-v0` | absolute TCP pose + binary gripper | scripted expert, demonstrations |

Rewards, observations and terminations are inherited from Isaac Lab's `IsaacContrib-Lift-Cube-Franka`
(`isaaclab_tasks.contrib.lift`). Playback uses the training task id: `isaaclab play` applies the env
config's `play_mode()` (16 envs, no observation noise) unless `--train_env_cfg` is passed. A checkpoint
selector such as `--checkpoint latest` only finds runs trained under the same task id.

## Usage

```
python scripts/train.py --task XArm7-PickPlace-v0
python scripts/play.py --task XArm7-PickPlace-v0 --checkpoint latest --viz kit
python scripts/scripted_expert.py --num_envs 64 --episodes 256
```

`train.py` and `play.py` run Isaac Lab's `isaaclab train` / `isaaclab play` with `--rl_library rsl_rl`, so
they take the same flags (`--num_envs`, `--max_iterations`, `--checkpoint`, `--video`, ...). There is no
`--headless` flag any more: a run without `--viz` is headless, and `--viz kit` opens the Isaac Sim window.

`scripted_expert.py` scores every finished episode. An episode succeeds if it runs to its time-out with
the cube within `--tol` of the goal; an episode that ends early because the cube fell counts as a failure.

Other scripts:

- `scripts/make_asset.py` rebuilds `assets/usd/xarm7_gripper.usd` from the stock asset.
- `scripts/diagnostics/check_asset.py` prints joints, limits, bodies and finger positions.
- `scripts/diagnostics/check_tracking_ik.py` reports joint tracking error, finger geometry in the gripper
  frame, and an IK solve for the top-down home pose.
- `scripts/build_scene.py` writes the lab-bench scene into `assets/usd/`: `lab_bench.usd` (bench, mount
  plate, four walls, floor, and the overhead depth camera on its Tarion TKA01-L desk-mount stand), `lab_lights.usd`, `paper_cube.usd` (5.2 cm
  cube), `food_bowl.usd` (square container with five box colliders) and `lab_scene.usd`, which assembles them
  with the xArm7 and its wrist depth camera for viewing in Isaac Sim. The assembly opens with the gripper
  pointing sideways as in the reference photo, its lower finger 64.3 cm above the tabletop, plus a
  guide-purpose measurement marker. The pick-and-place tasks do not
  use this scene yet.
- `xarm_twin/cameras.py` holds the lens (RealSense D435 colour, 640 x 360) and the poses of the wrist and
  overhead depth cameras. `build_scene.py` and the Isaac Lab camera configs both read it.
- `scripts/diagnostics/check_scene.py` loads that scene, checks resting heights, a cube drop into the bowl
  and what both depth cameras see, and writes four images to `logs/`.
- `scripts/diagnostics/check_scene_measurements.py` checks the assembled scene's measured dimensions,
  lower-finger clearance, four wall gaps, joint limits and matching joint anchors/drive targets.

## Tests

```
python -m pytest
```

These tests do not start the simulator. They check that the package imports without Isaac Lab loaded and
that the gym ids are registered.

## Modelling notes

- Active scene arm: **192.168.1.220**. The table is **92.4 x 183.1 cm**. J1 uses the requested
  scene position **X = 70.1 cm, Y = 0**, **10.3 cm** inside the +X long edge and **22.1 cm** from
  the left short edge. The scene origin remains the previous base location. The assembly is raised
  76 cm so its floor is at world Z = 0; scene metadata records measurements in the tabletop frame.
- The photo review of **IMG_2472 through IMG_2478** places the robot and camera clamp on the **same
  +X edge**, with A/B at the **opposite -X edge**. This corrects the earlier interpretation of the photo's
  top edge. B's outer rim is **14.3 cm** from that edge and its bottom-right rim envelope corner is
  **53 cm from the photo-reference J1 at X = 58.3 cm**; A's nearest side is **18 cm** from the same edge. B is left of A (+Y points right
  in the reference top view). Their **38.4 cm** sideways separation and B's **15 cm** rim size are estimates.
- A/B and the camera retain their positions after the base translation. The mounting plate and robot
  cabling follow J1. The current J1-to-B corner distance is approximately **62.6 cm**.
- The clamp is **56 cm horizontally from the photo-reference J1**, approximately **53.1 cm** from
  the translated J1. Its pole is **55 cm** from clamp base to
  shoulder; shoulder-to-elbow and elbow-to-wrist links are each **26.4 cm**. The boom reaches inward
  and **45 degrees left** (scene yaw -135 degrees), with an estimated **42 cm** horizontal projection.
  The camera lens is **81.2 cm** above the tabletop and aimed into the workspace. Optical aim, ball-head
  dimensions and camera extrinsics are photo estimates, not a calibrated camera-to-robot transform.
- The arm uses the **recorded .220 joint posture as the IK seed**, with flange XY approximately
  **(48 cm, -14 cm)** in its own base frame. The target preserves **64.3 cm clearance at the lowest
  finger mesh point**, a horizontal finger direction and sideways opening. It closely retains the
  photographed shoulder/elbow configuration instead of the earlier arbitrary folded pose. The wrist
  camera has its metal bracket, adjustment slots, lens face and cable. The gripper drive remains at an
  estimated **21.1 degrees**. This is a saved scene pose, not a command to the physical arm.
- `scripts/scene_details.py` adds deterministic butcher-block and wrinkled-paper textures, blue A/B
  cards, a clear bowl with rolled rim, camera lens panels and rear fins, plate slots and bolts, cable
  routing, a black vented equipment tower, pump bottle, mini tripod, nearby grey desk, storage cabinet,
  wall monitor/whiteboard, blinds, suspended cable trays and strip lights. These details are estimated from the photos;
  unrelated personal text in the background is not reproduced. Generated textures live in
  `assets/usd/textures/` and use relative references. The equipment tower has a static box collider;
  trim, labels, cable geometry and other decorative props do not add rigid bodies or colliders.
- Unmeasured room geometry remains approximate: **76 cm** table height, **4.4 cm** thickness and four
  **2.6 m** walls around a **2.924 x 3.831 m** interior. Walls remain 1 m from the tabletop edges; nearby
  equipment occupies some of that space. These room dimensions are not a surveyed reconstruction.
  `scripts/diagnostics/render_scene_views.py` saves front, opposite, detail, plan and room images to
  `logs/photo_scene_*.png` without advancing physics. Use these alongside the original photographs.

- The lab's arms are xArm 7s (controller type 7, firmware v2.7.1) at 192.168.1.220 and 192.168.1.241. The
  TCP offset (172 mm along the flange z-axis) and tool payload (0.82 kg, centre of mass 48 mm out) in
  `xarm_twin/assets.py` were read from the controller at 192.168.1.241 and verified by a read-only report
  from **192.168.1.220** for the current scene. Both report zero TCP orientation and base/world offsets.

- Gravity is disabled on the robot links to stand in for the real controller's gravity compensation.
  Joint torques in simulation therefore carry no gravity load.
- Drive gains in `xarm_twin/assets.py` are starting values, not yet fitted to the real arm.
- `TCP_OFFSET`, `GRIPPER_OPEN` and `GRIPPER_CLOSED` in `xarm_twin/assets.py` are the single source for the
  tool geometry; the env, the expert and the diagnostics all import them.
