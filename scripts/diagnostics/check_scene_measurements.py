"""Check measured dimensions and joint consistency in the generated lab_scene.usd.

Run with the Isaac Lab Python environment. This never connects to the physical robot.
"""
import math
import json
import os

from isaaclab.app import launch_simulation


def check():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stage = Usd.Stage.Open(os.path.join(root, "assets", "usd", "lab_scene.usd"))
    cache = UsdGeom.XformCache()

    def matrix(path):
        prim = stage.GetPrimAtPath(path)
        assert prim, path
        return cache.GetLocalToWorldTransform(prim)

    def close(label, actual, expected, tolerance=1e-5):
        assert abs(actual - expected) <= tolerance, (label, actual, expected)
        print(f"PASS {label}: {actual:.6f}", flush=True)

    table = "/Scene/Bench/Bench/Top"
    top = matrix(table).ExtractTranslation()[2] + 0.044 / 2
    scale = stage.GetPrimAtPath(table).GetAttribute("xformOp:scale").Get()
    close("table width", scale[0], 0.924)
    close("table length", scale[1], 1.831)
    centre = matrix(table).ExtractTranslation()
    close("J1 centre to left short edge", -(centre[1] - scale[1] / 2), 0.221)
    base_position = matrix("/Scene/RobotA/xarm7/world").ExtractTranslation()
    close("J1 requested absolute X", base_position[0], 0.701)
    close("J1 to opposite long edge", centre[0] + scale[0] / 2 - base_position[0], 0.103)
    close("mount plate follows J1", matrix("/Scene/Bench/MountA").ExtractTranslation()[0], base_position[0])
    notes = stage.GetDefaultPrim().GetCustomDataByKey("measurement_notes")
    assert notes["active_robot_ip"] == "192.168.1.220"
    top_edge = centre[0] - scale[0] / 2
    bowl = matrix("/Scene/Bowl").ExtractTranslation()
    cube = matrix("/Scene/Cube").ExtractTranslation()
    scan_path = os.path.join(root, "assets", "calibration", "tabletop_scan.json")
    if os.path.isfile(scan_path):
        with open(scan_path, encoding="utf-8") as scan_file:
            scan = json.load(scan_file)
        for name, position in (("cube", cube), ("bowl", bowl)):
            for axis in (0, 1):
                close(f"physical scan {name} {'XY'[axis]}", position[axis], scan["object_xy_m"][name][axis])
        close("cube rests on tabletop", cube[2] - top, .026)
        close("bowl rests on tabletop", bowl[2] - top, 0)
    else:
        close("B rim to opposite top long edge", bowl[0] - 0.150 / 2 - top_edge, 0.143)
        close("A top side to opposite top long edge", cube[0] - 0.052 / 2 - top_edge, 0.180)
        close("photo-reference J1 to B bottom-right rim corner", math.hypot(bowl[0] + 0.150 / 2 - 0.583,
                                                         bowl[1] + 0.150 / 2 - base_position[1]), 0.530)
    close("B edge-gap metadata", notes["bowl_top_edge_gap_m"], bowl[0] - .075 - top_edge)
    close("A edge-gap metadata", notes["cube_top_edge_gap_m"], cube[0] - .026 - top_edge)
    close("current J1 to B corner metadata", notes["current_base_to_bowl_bottom_right_corner_m"],
          math.hypot(bowl[0] + 0.075 - base_position[0], bowl[1] + 0.075 - base_position[1]))
    assert cube[1] > bowl[1], "A should be to the right of B in IMG_2474"
    for name, position, half_size in (("B", bowl, 0.150 / 2), ("A", cube, 0.052 / 2)):
        assert any(abs(position[i] - base_position[i]) > 0.20 / 2 + half_size for i in (0, 1)), \
            f"{name} overlaps the robot mount plate"

    stand = "/Scene/Bench/CameraStand"
    base = matrix(stand + "/Pole").ExtractTranslation()
    close("photo-reference J1 to camera clamp", math.hypot(base[0] - 0.583, base[1]), 0.560)
    close("current J1 to camera clamp metadata", notes["current_base_to_camera_clamp_m"],
          math.hypot(base[0] - base_position[0], base[1] - base_position[1]))
    assert base[0] > centre[0] and base_position[0] > centre[0], "Robot and pole must share the mounting edge"
    camera_pos = matrix("/Scene/Bench/OverheadCam/camera").ExtractTranslation()
    yaw = math.degrees(math.atan2(camera_pos[1] - base[1], camera_pos[0] - base[0]))
    close("camera boom inward and left", yaw, -135.0)
    for name, expected in (("Pole", 0.550), ("Arm1", 0.264), ("Arm2", 0.264)):
        close("stand " + name, stage.GetPrimAtPath(stand + "/" + name).GetAttribute("height").Get(), expected)
    close("camera lens above tabletop", matrix("/Scene/Bench/OverheadCam/camera").ExtractTranslation()[2] - top, 0.812)

    robot = "/Scene/RobotA/xarm7"
    grip = matrix(robot + "/gripper/xarm_gripper_base_link")
    lowest = math.inf
    for name in ("left_finger", "right_finger"):
        body = stage.GetPrimAtPath(robot + "/gripper/" + name)
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh) and "collision" not in str(prim.GetPath()).lower():
                transform = cache.GetLocalToWorldTransform(prim)
                lowest = min(lowest, *(transform.Transform(Gf.Vec3d(p))[2]
                                       for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()))
    close("lower finger above tabletop", lowest - top, 0.643)
    height = stage.GetPrimAtPath("/Scene/Measurements/GripperClearance/Height")
    close("measured gripper clearance reference", height.GetAttribute("height").Get(), 0.643)
    close("gripper fingers point inward across table", grip.TransformDir(Gf.Vec3d(0, 0, 1))[0], -math.cos(math.radians(6.1)))
    close("finger separation is horizontal", grip.TransformDir(Gf.Vec3d(0, 1, 0))[2], 0)
    close("wrist camera side points up", grip.TransformDir(Gf.Vec3d(1, 0, 0))[2], 1)
    for name, letter in (("Cube", "A"), ("Bowl", "B")):
        label = stage.GetPrimAtPath(f"/Scene/{name}/Label")
        assert label.GetCustomDataByKey("text") == letter
        assert stage.GetPrimAtPath(f"/Scene/{name}/Label/{letter}").IsA(UsdGeom.Mesh)
    for path in ("/Scene/Bench/Bench/WoodSurface", "/Scene/Bench/Equipment/Tower/Case",
                 "/Scene/RobotA/xarm7/gripper/xarm_gripper_base_link/CameraBracket/Upright",
                 "/Scene/RobotA/xarm7/gripper/xarm_gripper_base_link/wrist_cam/housing/Lens_0",
                 "/Scene/Bowl/RolledLip/Segment_0", "/Scene/Cube/Paper/Face_0"):
        assert stage.GetPrimAtPath(path), f"Missing photo detail: {path}"
    tower = stage.GetPrimAtPath("/Scene/Bench/Equipment/Tower/Case")
    tower_position = matrix(str(tower.GetPath())).ExtractTranslation()
    tower_size = tower.GetAttribute("xformOp:scale").Get()
    for name, position, half_size in (("A", cube, 0.052 / 2), ("B", bowl, 0.150 / 2)):
        assert any(abs(position[i] - tower_position[i]) > tower_size[i] / 2 + half_size for i in (0, 1)), \
            f"{name} overlaps the equipment tower"
    walls = list(stage.GetPrimAtPath("/Scene/Bench/Backdrop").GetChildren())
    assert len(walls) == 4
    for wall in walls:
        assert wall.HasAPI(UsdPhysics.CollisionAPI)
        pos = matrix(str(wall.GetPath())).ExtractTranslation()
        axis = 0 if wall.GetName() in ("West", "East") else 1
        gap = abs(pos[axis] - centre[axis]) - scale[axis] / 2 - 0.10 / 2
        close(wall.GetName() + " wall clear space", gap, 1.0)

    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot)):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not parents or not children:
            continue
        p0 = matrix(str(parents[0])).Transform(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
        p1 = matrix(str(children[0])).Transform(Gf.Vec3d(joint.GetLocalPos1Attr().Get()))
        close(prim.GetName() + " joint anchors coincide", (p0 - p1).GetLength(), 0)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            angle = prim.GetAttribute("state:angular:physics:position").Get()
            assert prim.GetAttribute("physics:lowerLimit").Get() - 1e-4 <= angle
            assert angle <= prim.GetAttribute("physics:upperLimit").Get() + 1e-4
            if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
                close(prim.GetName() + " drive matches state", UsdPhysics.DriveAPI(prim, "angular").GetTargetPositionAttr().Get(), angle)
    print("SCENE_MEASUREMENTS_OK", flush=True)


if __name__ == "__main__":
    with launch_simulation(None, {"require_kit": True}):
        try:
            check()
        except Exception:
            import traceback
            traceback.print_exc()
            raise
