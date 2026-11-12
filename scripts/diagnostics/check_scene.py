"""Load the lab-bench scene in Isaac Lab and test it: resting heights, the cube dropped into the bowl, two renders.

Also renders colour and depth from the two depth cameras (wrist and overhead).

Writes logs/scene_bench.png, scene_in_bowl.png, scene_cam_wrist.png and scene_cam_overhead.png (the last two
are colour on the left, depth on the right; nearer is brighter).

Usage: C:\IsaacLab3\.venv\Scripts\python.exe scripts\diagnostics\check_scene.py
"""
import argparse
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args()
args.require_kit = True  # Kit PhysX for the gripper's mimic joints, Kit RTX for the cameras
args.enable_cameras = True
# Start Kit before the imports below: imported first, isaaclab.sensors loads a USD build that clashes with
# Kit's (usd_tf.dll "procedure could not be found"). The script ends with os._exit, so Kit is never closed.
_kit = launch_simulation(None, args)  # keep a reference: collecting the context manager closes Kit
_kit.__enter__()

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

from xarm_twin import cameras as cams  # noqa: E402
from xarm_twin.assets import XARM7_CFG  # noqa: E402

USD = os.path.join(ROOT, "assets", "usd")
LOGS = os.path.join(ROOT, "logs")
DT = 0.01
CUBE_EDGE = 0.052
PLATE = 0.010
BOWL_FLOOR = 0.004  # collision floor thickness
CUBE_START = (0.40, 0.0)
BOWL_START = (0.42, 0.30)


def main():
    os.makedirs(LOGS, exist_ok=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    for name, file in (("Bench", "lab_bench.usd"), ("Lights", "lab_lights.usd")):
        cfg = sim_utils.UsdFileCfg(usd_path=os.path.join(USD, file))
        cfg.func(f"/World/{name}", cfg)
    robot = Articulation(
        XARM7_CFG.replace(prim_path="/World/Robot", init_state=XARM7_CFG.init_state.replace(pos=(0.0, 0.0, PLATE)))
    )
    cube = RigidObject(RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(USD, "paper_cube.usd")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*CUBE_START, 0.10)),
    ))
    bowl = RigidObject(RigidObjectCfg(
        prim_path="/World/Bowl",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(USD, "food_bowl.usd")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*BOWL_START, 0.02)),
    ))
    camera = Camera(CameraCfg(
        prim_path="/World/ViewCam", update_period=0.0, width=1280, height=720, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=1.93, horizontal_aperture=2.65, clipping_range=(0.05, 20.0)),
    ))
    # The wrist camera is created on the gripper link from the shared constants. The overhead camera is the
    # prim that build_scene.py wrote into lab_bench.usd.
    lens = sim_utils.PinholeCameraCfg(
        focal_length=cams.FOCAL_LENGTH, horizontal_aperture=cams.HORIZONTAL_APERTURE, clipping_range=cams.CLIPPING_RANGE
    )
    wrist = Camera(CameraCfg(
        prim_path=f"/World/Robot/{cams.WRIST_PARENT}/wrist_cam", update_period=0.0, width=cams.WIDTH,
        height=cams.HEIGHT, data_types=cams.DATA_TYPES, spawn=lens,
        offset=CameraCfg.OffsetCfg(pos=cams.WRIST_POS, rot=cams.xyzw(cams.WRIST_QUAT), convention="opengl"),
    ))
    overhead = Camera(CameraCfg(
        prim_path="/World/Bench/OverheadCam/camera", update_period=0.0, width=cams.WIDTH, height=cams.HEIGHT,
        data_types=cams.DATA_TYPES, spawn=None,
    ))
    sim.reset()
    dev = sim.device

    def step(n, render=False):
        for _ in range(n):
            robot.actuators.target_command.set_position_index(value=robot.data.default_joint_pos.torch)
            robot.write_data_to_sim()
            sim.step(render=render)
            for asset in (robot, cube, bowl):
                asset.update(DT)

    def shoot(name, eye, target):
        camera.set_world_poses_from_view(torch.tensor([eye], device=dev), torch.tensor([target], device=dev))
        step(60, render=True)
        camera.update(DT)
        rgb = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
        path = os.path.join(LOGS, f"scene_{name}.png")
        Image.fromarray(rgb).save(path)
        print(f"SCENE render {name}: {path} mean brightness {rgb.mean():.1f}/255", flush=True)

    def depth_shot(sensor, name):
        """Save colour | depth side by side and return both arrays (depth in metres along the optical axis)."""
        # Render before reading. Read straight after another camera, with no frame rendered in between, the
        # colour buffer came back all zeros while depth was fine.
        step(15, render=True)
        sensor.update(DT)
        rgb = sensor.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
        depth = sensor.data.output["distance_to_image_plane"].torch[0].squeeze(-1).cpu().numpy()
        seen = np.isfinite(depth)
        near, far = float(depth[seen].min()), min(float(depth[seen].max()), 1.2)
        grey = 255 * (1 - (np.clip(np.where(seen, depth, far), near, far) - near) / max(far - near, 1e-6))
        panel = np.concatenate([rgb, np.repeat(grey[..., None], 3, axis=-1).astype(np.uint8)], axis=1)
        path = os.path.join(LOGS, f"scene_cam_{name}.png")
        Image.fromarray(panel).save(path)
        print(f"SCENE camera {name}: {path} {rgb.shape[1]}x{rgb.shape[0]} colour+depth, depth {near:.3f} to "
              f"{float(depth[seen].max()):.3f} m, {100 * seen.mean():.1f}% of pixels hit something", flush=True)
        return rgb, depth

    results = []

    def check(label, value, low, high):
        ok = low <= value <= high
        results.append(ok)
        print(f"SCENE {'PASS' if ok else 'FAIL'} {label}: {value:.4f} (allowed {low:.4f} to {high:.4f})", flush=True)

    # 1. Everything dropped from a few centimetres comes to rest where the geometry says it should.
    step(200)
    c, b = cube.data.root_pos_w.torch[0], bowl.data.root_pos_w.torch[0]
    check("cube rests on bench, centre height", c[2].item(), CUBE_EDGE / 2 - 0.0015, CUBE_EDGE / 2 + 0.0015)
    check("cube sideways drift while settling", (c[:2] - torch.tensor(CUBE_START, device=dev)).norm().item(), 0.0, 0.003)
    check("bowl rests on bench, underside height", b[2].item(), -0.0015, 0.0015)
    check("cube speed at rest", cube.data.root_lin_vel_w.torch[0].norm().item(), 0.0, 0.01)
    check("robot base height on the plate", robot.data.root_pos_w.torch[0, 2].item(), PLATE - 0.0005, PLATE + 0.0005)
    check("robot holds home pose, worst joint error (rad)",
          (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[0, :7].abs().max().item(), 0.0, 0.01)
    tcp_z = robot.data.body_pos_w.torch[0, robot.body_names.index("xarm_gripper_base_link"), 2].item()
    check("gripper base clear of the bench", tcp_z, 0.15, 1.0)
    shoot("bench", (1.45, -0.80, 0.95), (0.25, 0.25, 0.30))

    # 2. The two depth cameras see what their poses say they should.
    height = cams.OVERHEAD_POS[2]
    rgb, depth = depth_shot(overhead, "overhead")
    check("overhead camera, depth to the bench top (median)", float(np.median(depth)), height - 0.005, height + 0.005)
    raised = int(((depth > height - 0.07) & (depth < height - 0.01)).sum())
    check("overhead camera, pixels 1 to 7 cm above the bench (bowl walls)", raised, 300, 20000)
    check("overhead camera, colour contrast (standard deviation)", float(rgb.std()), 5, 128)
    rgb, depth = depth_shot(wrist, "wrist")
    check("wrist camera, nearest thing in view is its own fingers (1st percentile depth)",
          float(np.percentile(depth, 1)), 0.03, 0.20)
    check("wrist camera, depth to the bench (median)", float(np.median(depth)), 0.30, 0.45)
    check("wrist camera, colour contrast (standard deviation)", float(rgb.std()), 5, 128)

    # 3. The cube dropped over the bowl ends up inside it, on the bowl floor, and the bowl stays put.
    bowl_before = bowl.data.root_pos_w.torch[0].clone()
    pose = torch.tensor([[bowl_before[0], bowl_before[1], 0.14, 0.0, 0.0, 0.0, 1.0]], device=dev)  # x,y,z + quat x,y,z,w
    cube.write_root_pose_to_sim_index(root_pose=pose)
    cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev))
    step(200)
    c, b = cube.data.root_pos_w.torch[0], bowl.data.root_pos_w.torch[0]
    check("cube in bowl, distance from bowl centre", (c[:2] - b[:2]).norm().item(), 0.0, 0.045)
    check("cube sits on the bowl floor, centre height above bowl underside", (c[2] - b[2]).item(),
          BOWL_FLOOR + CUBE_EDGE / 2 - 0.003, BOWL_FLOOR + CUBE_EDGE / 2 + 0.003)
    check("bowl moved when the cube landed", (b - bowl_before).norm().item(), 0.0, 0.005)
    check("cube speed at rest in bowl", cube.data.root_lin_vel_w.torch[0].norm().item(), 0.0, 0.01)
    shoot("in_bowl", (0.85, -0.05, 0.38), (0.40, 0.28, 0.03))

    print(f"CHECK_SCENE {'OK' if all(results) else 'FAILED'} {sum(results)}/{len(results)}", flush=True)
    # Closing Kit can hang on Windows once a SimulationContext exists; the work is done, so exit here
    # instead of letting launch_simulation close the app.
    sys.stdout.flush()
    os._exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
