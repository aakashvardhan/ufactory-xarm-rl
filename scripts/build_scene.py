"""Build the lab-bench scene USD files into assets/usd/.

  lab_bench.usd   butcher-block bench, mount plate, wall, floor, and the overhead depth camera on its
                  Tarion TKA01-L desk-mount stand. Static colliders, no rigid bodies.
                  Variant set "arms": "single" (default) or "dual" (adds the second mount plate and a
                  visual-only copy of the second arm).
  lab_lights.usd  dome + ceiling light. Spawn once per stage, not once per environment.
  paper_cube.usd  5.2 cm paper-wrapped cube, rigid body.
  food_bowl.usd   square frosted food container, rigid body, five box colliders (the cavity stays open).
  lab_scene.usd   assembly of the files above plus a ground plane and the xArm7 with its wrist depth camera, for opening in
                  Isaac Sim. Pick /Scene/Bench/OverheadCam/camera or .../wrist_cam/camera in the viewport.

Frame: origin at the centre of arm A's base on the bench top (z = 0). +X across the bench away from the
mounting edge, +Y along the mounting edge toward arm B, +Z up. Metres, kilograms. lab_scene.usd alone lifts
/Scene by BENCH_HEIGHT so its floor sits on z = 0, where Isaac Sim draws its viewport grid; the Isaac Lab
configs use the files in the bench-top frame.

Only the cube edge (0.052 m) is measured. Every other size is an estimate from photos; edit the constants
below and rerun.

Usage: C:\IsaacLab3\.venv\Scripts\python.exe scripts\build_scene.py
"""
import math
import os
import sys

from isaaclab.app import launch_simulation

_kit = launch_simulation(None, {"require_kit": True})  # Kit provides pxr; closed in __main__
_kit.__enter__()

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from xarm_twin import cameras as cams  # noqa: E402

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "usd"))
STOCK_XARM7 = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac"
    "/Robots/Ufactory/xarm7/xarm7.usd"
)

# ---- dimensions (m, kg) ---------------------------------------------------------------------------
BENCH_SIZE = (0.76, 1.83, 0.044)  # estimate
BENCH_X0, BENCH_Y0 = -0.12, -0.25  # bench corner nearest arm A, estimate
BENCH_HEIGHT = 0.76  # floor to bench top, estimate
PLATE = (0.20, 0.20, 0.010)  # estimate
ARM_B_Y = 1.20  # estimate
CUBE_EDGE = 0.052  # measured
CUBE_MASS = 0.050  # estimate
BOWL_RIM, BOWL_BASE, BOWL_H = 0.150, 0.125, 0.065  # outer widths and height, estimate
BOWL_WALL, BOWL_RIM_RADIUS = 0.002, 0.020
BOWL_MASS = 0.045  # estimate
ROBOT_Z = PLATE[2]
WALL_GAP = 0.90  # wall to the bench's mounting edge, estimate
# Tarion TKA01-L desk-mount camera stand: C-clamp, upright pole, two arms on three toothed joints, ball head.
# Published: 30 mm steel tubes, 38 in (0.965 m) fully extended, clamp opens to 60 mm. The split into pole and
# arm lengths is an estimate from the drawing on the box.
STAND_Y = 0.32  # where the clamp sits along the mounting edge, estimate
STAND_TUBE_RADIUS = 0.015
STAND_POLE = 0.40
STAND_ARM = 0.28  # each of the two arms
GROUND_SIZE = 20.0  # visual square; the collision plane is infinite
GROUND_GAP = 0.001  # ground plane this far below the lab floor's top face
CUBE_START = (0.40, 0.0, CUBE_EDGE / 2)
BOWL_START = (0.42, 0.30, 0.0)


def new_stage(name, root):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, root).GetPrim())
    return stage


def look(stage, path, color, roughness, metallic=0.0, opacity=1.0, ior=1.5):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def physics_material(stage, path, static, dynamic, restitution):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(restitution)
    mat.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("average")
    return mat


def box(stage, path, size, centre, visual=None, physics=None, collide=False, guide=False):
    """A unit cube scaled to `size` at `centre`. Visual only unless `collide` is set."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    prim = cube.GetPrim()
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    if visual:
        binding.Bind(visual)
    if collide:
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.002)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
        if physics:
            binding.Bind(physics, materialPurpose="physics")
    if guide:
        cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    return prim


def rigid_body(prim, mass):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(mass)
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
    prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(1)


def tube(stage, path, start, end, radius, visual):
    """A visual-only cylinder from `start` to `end`."""
    a, b = Gf.Vec3d(*start), Gf.Vec3d(*end)
    axis = b - a
    length = axis.GetLength()
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    cyl.CreateHeightAttr(length)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateExtentAttr([(-radius, -radius, -length / 2), (radius, radius, length / 2)])
    cyl.AddTranslateOp().Set((a + b) / 2)
    cyl.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), axis.GetNormalized()).GetQuat()))
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(visual)
    return cyl


def camera_stand(stage, path, visual):
    """Tarion TKA01-L holding the overhead camera: clamp on the mounting edge, upright pole, two arms.

    The pole stands upright in the clamp. The two arms bend at the elbow (elbow up) to put the ball head
    right above the camera housing. Raises if the camera pose is out of the arms' reach. Simplified: the real
    joints set neighbouring tubes side by side; here they share one line.
    """
    r = STAND_TUBE_RADIUS
    ox, oy, oz = cams.OVERHEAD_POS
    UsdGeom.Xform.Define(stage, path)
    UsdGeom.Xform.Define(stage, path + "/Clamp")
    # C-clamp gripping the bench top at the mounting edge, screw underneath.
    edge, top, under = BENCH_X0, 0.0, -BENCH_SIZE[2]
    jaw, depth, width, drop = 0.012, 0.065, 0.07, 0.035
    x_mid = edge - jaw + (depth + jaw) / 2
    box(stage, path + "/Clamp/TopJaw", (depth + jaw, width, jaw), (x_mid, STAND_Y, top + jaw / 2), visual)
    box(stage, path + "/Clamp/Spine", (jaw, width, jaw - under + drop + jaw), (edge - jaw / 2, STAND_Y, (top + jaw + under - drop - jaw) / 2), visual)
    box(stage, path + "/Clamp/BottomJaw", (depth + jaw, width, jaw), (x_mid, STAND_Y, under - drop - jaw / 2), visual)
    pole_x = edge + 0.03
    tube(stage, path + "/Clamp/Screw", (pole_x, STAND_Y, under - drop - jaw - 0.03), (pole_x, STAND_Y, under - 0.004), 0.005, visual)
    tube(stage, path + "/Clamp/ScrewPad", (pole_x, STAND_Y, under - 0.004), (pole_x, STAND_Y, under), 0.014, visual)
    tube(stage, path + "/Clamp/ScrewKnob", (pole_x, STAND_Y, under - drop - jaw - 0.045), (pole_x, STAND_Y, under - drop - jaw - 0.03), 0.016, visual)

    # Pole, then the two arms to a point above the camera.
    base = (pole_x, STAND_Y, top + jaw)
    shoulder = (pole_x, STAND_Y, top + jaw + STAND_POLE)
    housing_top = oz + cams.BODY_BEHIND + cams.BODY_SIZE[2] / 2
    ball = (ox, oy, housing_top + 0.022)
    wrist = (ox, oy, housing_top + 0.06)
    gap = [w - s for w, s in zip(wrist, shoulder)]
    reach = math.sqrt(sum(c * c for c in gap))
    if reach > 2 * STAND_ARM - 0.005:
        raise ValueError(
            f"camera stand cannot reach the overhead camera: {reach:.3f} m from the top of the pole, "
            f"arms give {2 * STAND_ARM:.3f} m. Move OVERHEAD_POS or STAND_Y."
        )
    flat = math.hypot(gap[0], gap[1])
    along = (gap[0] / flat, gap[1] / flat, 0.0)  # horizontal direction from the pole to the camera
    rise = math.sqrt(STAND_ARM**2 - (reach / 2) ** 2)  # how far the elbow sits off the straight line
    normal = (-along[0] * gap[2] / reach, -along[1] * gap[2] / reach, flat / reach)  # in the arm plane, upward
    elbow = tuple(s + g / 2 + rise * n for s, g, n in zip(shoulder, gap, normal))
    side = (-along[1], along[0], 0.0)  # joint axes: horizontal, across the arm plane

    tube(stage, path + "/Pole", base, shoulder, r, visual)
    tube(stage, path + "/Arm1", shoulder, elbow, r, visual)
    tube(stage, path + "/Arm2", elbow, wrist, r, visual)
    for name, point in (("Joint1", shoulder), ("Joint2", elbow), ("Joint3", wrist)):
        ends = [tuple(p + k * c for p, c in zip(point, side)) for k in (-0.03, 0.03, 0.05)]
        tube(stage, f"{path}/{name}", ends[0], ends[1], 0.026, visual)  # toothed joint
        tube(stage, f"{path}/{name}Knob", ends[1], ends[2], 0.012, visual)  # locking knob
    UsdGeom.Xform.Define(stage, path + "/BallHead")
    tube(stage, path + "/BallHead/Stem", ball, wrist, 0.008, visual)
    sphere = UsdGeom.Sphere.Define(stage, path + "/BallHead/Ball")
    sphere.CreateRadiusAttr(0.016)
    sphere.CreateExtentAttr([(-0.016,) * 3, (0.016,) * 3])
    sphere.AddTranslateOp().Set(Gf.Vec3d(*ball))
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(visual)
    tube(stage, path + "/BallHead/Plate", (ox, oy, housing_top), (ox, oy, housing_top + 0.008), 0.02, visual)
    bend = math.degrees(2 * math.acos(min(1.0, reach / (2 * STAND_ARM))))
    print(f"camera stand: pole {STAND_POLE:.2f} m, arms 2 x {STAND_ARM:.2f} m, reach used {reach:.3f} of "
          f"{2 * STAND_ARM:.3f} m, elbow bent {bend:.0f} deg, elbow at {tuple(round(c, 3) for c in elbow)}")


def depth_camera(stage, path, pos, quat, body_look):
    """A Camera prim at `path`/camera and the D435 housing behind it, both at the given pose.

    The pose sits on the Camera prim itself as translate, orient, scale: Isaac Lab refuses a camera prim
    without exactly those three ops.
    """
    UsdGeom.Xform.Define(stage, path)
    double = UsdGeom.XformOp.PrecisionDouble
    orientation = Gf.Quatd(quat[0], Gf.Vec3d(*quat[1:]))
    cam = UsdGeom.Camera.Define(stage, path + "/camera")
    cam.AddTranslateOp(double).Set(Gf.Vec3d(*pos))
    cam.AddOrientOp(double).Set(orientation)
    cam.AddScaleOp(double).Set(Gf.Vec3d(1.0, 1.0, 1.0))
    cam.CreateFocalLengthAttr(cams.FOCAL_LENGTH)
    cam.CreateHorizontalApertureAttr(cams.HORIZONTAL_APERTURE)
    cam.CreateVerticalApertureAttr(cams.VERTICAL_APERTURE)
    cam.CreateClippingRangeAttr(Gf.Vec2f(*cams.CLIPPING_RANGE))
    housing = UsdGeom.Xform.Define(stage, path + "/housing")
    housing.AddTranslateOp(double).Set(Gf.Vec3d(*pos))
    housing.AddOrientOp(double).Set(orientation)
    box(stage, path + "/housing/body", cams.BODY_SIZE, (0.0, 0.0, cams.BODY_BEHIND), body_look)
    return cam


def build_bench():
    s = new_stage("lab_bench.usd", "/World")
    maple = look(s, "/World/Looks/Maple", (0.82, 0.66, 0.42), 0.45)
    steel = look(s, "/World/Looks/Frame", (0.23, 0.24, 0.26), 0.6)
    # metallic stays low: a fully metallic plate rendered black under the plain dome light
    plate = look(s, "/World/Looks/Plate", (0.75, 0.76, 0.78), 0.4, metallic=0.2)
    wall = look(s, "/World/Looks/Wall", (0.92, 0.92, 0.91), 0.9)
    floor = look(s, "/World/Looks/Floor", (0.62, 0.63, 0.65), 0.8)
    wood = physics_material(s, "/World/PhysicsMaterials/Wood", 0.50, 0.40, 0.05)
    alu = physics_material(s, "/World/PhysicsMaterials/Aluminium", 0.40, 0.35, 0.05)

    sx, sy, sz = BENCH_SIZE
    cx, cy = BENCH_X0 + sx / 2, BENCH_Y0 + sy / 2
    UsdGeom.Xform.Define(s, "/World/Bench")
    UsdGeom.Xform.Define(s, "/World/Bench/Frame")
    box(s, "/World/Bench/Top", BENCH_SIZE, (cx, cy, -sz / 2), maple, wood, collide=True)
    leg_h = BENCH_HEIGHT - sz
    inset = 0.08
    xs = (BENCH_X0 + inset, BENCH_X0 + sx - inset)
    ys = (BENCH_Y0 + inset, BENCH_Y0 + sy - inset)
    for i, (x, y) in enumerate((x, y) for x in xs for y in ys):
        box(s, f"/World/Bench/Frame/Leg_{i}", (0.06, 0.06, leg_h), (x, y, -sz - leg_h / 2), steel)
    for i, x in enumerate(xs):
        box(s, f"/World/Bench/Frame/Rail_{i}", (0.04, sy - 2 * inset, 0.08), (x, cy, -sz - 0.06), steel)
    box(s, "/World/MountA", PLATE, (0.0, 0.0, PLATE[2] / 2), plate, alu, collide=True)
    UsdGeom.Xform.Define(s, "/World/Backdrop")
    wall_x = BENCH_X0 - WALL_GAP
    box(s, "/World/Backdrop/Wall", (0.02, 6.0, 2.6), (wall_x - 0.01, cy, -BENCH_HEIGHT + 1.3), wall)
    box(s, "/World/Floor", (4.0, 6.0, 0.02), (wall_x + 2.0, cy, -BENCH_HEIGHT - 0.01), floor, collide=True)

    # Overhead depth camera on its desk-mount stand. The stand is scenery (no colliders).
    black = look(s, "/World/Looks/Stand", (0.05, 0.05, 0.05), 0.5)
    silver = look(s, "/World/Looks/CameraBody", (0.72, 0.73, 0.75), 0.4, metallic=0.2)
    camera_stand(s, "/World/CameraStand", black)
    depth_camera(s, "/World/OverheadCam", cams.OVERHEAD_POS, cams.OVERHEAD_QUAT, silver)

    # Second arm: off by default. It is scenery (no physics), so it never joins the simulation.
    arms = s.GetDefaultPrim().GetVariantSets().AddVariantSet("arms")
    arms.AddVariant("single")
    arms.AddVariant("dual")
    arms.SetVariantSelection("dual")
    with arms.GetVariantEditContext():
        box(s, "/World/MountB", PLATE, (0.0, ARM_B_Y, PLATE[2] / 2), plate, alu, collide=True)
        holder = UsdGeom.Xform.Define(s, "/World/RobotB")
        holder.AddTranslateOp().Set(Gf.Vec3d(0.0, ARM_B_Y, ROBOT_Z))
        arm = s.DefinePrim("/World/RobotB/xarm7")
        arm.GetReferences().AddReference(STOCK_XARM7)
        for name, choice in (("Physics", "None"), ("Variant_Set", "xarm_gripper"), ("Sensor", "None")):
            arm.GetVariantSets().SetSelection(name, choice)
        live = [
            str(p.GetPath())
            for p in Usd.PrimRange(s.GetPrimAtPath("/World/RobotB"))
            if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.ArticulationRootAPI) or p.IsA(UsdPhysics.Joint)
        ]
        meshes = sum(1 for p in Usd.PrimRange(s.GetPrimAtPath("/World/RobotB")) if p.IsA(UsdGeom.Mesh))
        print(f"dual variant: arm B has {meshes} meshes and {len(live)} physics prims")
        if live or not meshes:
            s.RemovePrim("/World/RobotB")
            print("dual variant: arm B left out (it would be simulated, or did not load); plate only")
    arms.SetVariantSelection("single")
    s.GetRootLayer().Save()
    return s


def build_lights():
    s = new_stage("lab_lights.usd", "/Lights")
    dome = UsdLux.DomeLight.Define(s, "/Lights/Dome")
    dome.CreateIntensityAttr(450.0)
    dome.CreateColorAttr(Gf.Vec3f(0.95, 0.97, 1.0))
    rect = UsdLux.RectLight.Define(s, "/Lights/Ceiling")  # emits along -Z
    rect.CreateWidthAttr(1.2)
    rect.CreateHeightAttr(2.0)
    rect.CreateIntensityAttr(1800.0)
    rect.CreateEnableColorTemperatureAttr(True)
    rect.CreateColorTemperatureAttr(5000.0)
    rect.AddTranslateOp().Set(Gf.Vec3d(0.3, 0.6, 2.0))
    s.GetRootLayer().Save()
    return s


def build_cube():
    s = new_stage("paper_cube.usd", "/Cube")
    paper = look(s, "/Cube/Looks/Paper", (0.93, 0.93, 0.91), 0.9)
    phys = physics_material(s, "/Cube/PhysicsMaterial", 0.60, 0.50, 0.05)
    rigid_body(s.GetDefaultPrim(), CUBE_MASS)
    box(s, "/Cube/geom", (CUBE_EDGE,) * 3, (0.0, 0.0, 0.0), paper, phys, collide=True)
    s.GetRootLayer().Save()
    return s


def rounded_square(half, radius, z, n=6):
    """Outline of a rounded square, counter-clockwise seen from +Z."""
    c = half - radius
    points = []
    for cx, cy, start in ((c, c, 0), (-c, c, 90), (-c, -c, 180), (c, -c, 270)):
        for k in range(n + 1):
            a = math.radians(start + 90.0 * k / n)
            points.append(Gf.Vec3f(cx + radius * math.cos(a), cy + radius * math.sin(a), z))
    return points


def build_bowl():
    s = new_stage("food_bowl.usd", "/Bowl")
    # Opaque milky white. At opacity 0.35 the walls vanished in the RTX render (only a dark floor patch
    # showed), which would hide the bowl from a camera policy.
    frosted = look(s, "/Bowl/Looks/Frosted", (0.86, 0.89, 0.91), 0.5, ior=1.49)
    phys = physics_material(s, "/Bowl/PhysicsMaterial", 0.35, 0.30, 0.10)
    rigid_body(s.GetDefaultPrim(), BOWL_MASS)

    # Visual shell: tapered walls, rounded corners. Origin at the centre of the underside.
    rim, base, t = BOWL_RIM / 2, BOWL_BASE / 2, BOWL_WALL
    r_base = BOWL_RIM_RADIUS * base / rim
    rings = [
        rounded_square(base, r_base, 0.0),  # outer base
        rounded_square(rim, BOWL_RIM_RADIUS, BOWL_H),  # outer rim
        rounded_square(rim - t, BOWL_RIM_RADIUS - t, BOWL_H),  # inner rim
        rounded_square(base - t, r_base - t, t),  # inner floor
    ]
    n = len(rings[0])
    counts, indices = [], []
    for a, b in ((0, 1), (1, 2), (2, 3)):  # outer wall, rim, inner wall
        for i in range(n):
            j = (i + 1) % n
            counts.append(4)
            indices += [a * n + i, a * n + j, b * n + j, b * n + i]
    counts.append(n)
    indices += [3 * n + i for i in range(n)]  # inner floor, facing up
    counts.append(n)
    indices += [i for i in reversed(range(n))]  # underside, facing down
    mesh = UsdGeom.Mesh.Define(s, "/Bowl/visual")
    mesh.CreatePointsAttr([p for ring in rings for p in ring])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([(-rim, -rim, 0.0), (rim, rim, BOWL_H)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(frosted)

    # Collision: a floor and four walls. Walls are vertical at the mean width and 4 mm thick (twice the
    # real wall) so a fast cube cannot tunnel through. A convex hull would fill the cavity.
    UsdGeom.Xform.Define(s, "/Bowl/collision")
    ft, wt = 0.004, 0.004
    mean = (BOWL_RIM + BOWL_BASE) / 4 - wt / 2 + 0.001  # wall centre offset
    span = 2 * mean + wt
    wall_h = BOWL_H - ft
    wz = ft + wall_h / 2
    box(s, "/Bowl/collision/Floor", (BOWL_BASE, BOWL_BASE, ft), (0, 0, ft / 2), physics=phys, collide=True, guide=True)
    for name, size, centre in (
        ("WallPX", (wt, span, wall_h), (mean, 0, wz)),
        ("WallNX", (wt, span, wall_h), (-mean, 0, wz)),
        ("WallPY", (span, wt, wall_h), (0, mean, wz)),
        ("WallNY", (span, wt, wall_h), (0, -mean, wz)),
    ):
        box(s, f"/Bowl/collision/{name}", size, centre, physics=phys, collide=True, guide=True)
    s.GetRootLayer().Save()
    return s


def ground_plane(stage, path, z, size=GROUND_SIZE):
    """Ground plane at height `z`: an infinite collision plane and a grey square to see it by."""
    UsdGeom.Xform.Define(stage, path).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
    plane = UsdGeom.Plane.Define(stage, path + "/CollisionPlane")  # PhysX treats a Plane as infinite
    plane.CreateAxisAttr(UsdGeom.Tokens.z)
    plane.CreateWidthAttr(size)
    plane.CreateLengthAttr(size)
    plane.CreateExtentAttr([(-size / 2, -size / 2, 0.0), (size / 2, size / 2, 0.0)])
    plane.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    h = size / 2
    mesh = UsdGeom.Mesh.Define(stage, path + "/Visual")
    mesh.CreatePointsAttr([(-h, -h, 0.0), (h, -h, 0.0), (h, h, 0.0), (-h, h, 0.0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([(-h, -h, 0.0), (h, h, 0.0)])
    grey = look(stage, path + "/Looks/Ground", (0.45, 0.46, 0.48), 0.9)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(grey)


def build_assembly():
    s = new_stage("lab_scene.usd", "/Scene")
    # Floor on z = 0 for viewing: without this the viewport grid cuts through the bench top.
    UsdGeom.Xformable(s.GetDefaultPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, BENCH_HEIGHT))
    # World z = -1 mm: just under the lab floor's top face, so the two surfaces do not flicker.
    ground_plane(s, "/Scene/GroundPlane", -BENCH_HEIGHT - GROUND_GAP)
    scene = UsdPhysics.Scene.Define(s, "/Scene/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    for name, ref, pos in (
        ("Bench", "./lab_bench.usd", None),
        ("Lights", "./lab_lights.usd", None),
        ("Cube", "./paper_cube.usd", CUBE_START),
        ("Bowl", "./food_bowl.usd", BOWL_START),
    ):
        prim = s.DefinePrim(f"/Scene/{name}", "Xform")
        prim.GetReferences().AddReference(ref)
        if pos:
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*pos))
    holder = UsdGeom.Xform.Define(s, "/Scene/RobotA")
    holder.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, ROBOT_Z))
    s.DefinePrim("/Scene/RobotA/xarm7").GetReferences().AddReference("./xarm7_gripper.usd")
    link = s.GetPrimAtPath("/Scene/RobotA/xarm7/" + cams.WRIST_PARENT)
    if not link or link.IsInstanceProxy():
        raise RuntimeError(f"gripper link {cams.WRIST_PARENT} not found in the robot asset")
    silver = look(s, "/Scene/Looks/CameraBody", (0.72, 0.73, 0.75), 0.4, metallic=0.2)
    depth_camera(s, str(link.GetPath()) + "/wrist_cam", cams.WRIST_POS, cams.WRIST_QUAT, silver)
    s.GetRootLayer().Save()
    return s


def report(name):
    stage = Usd.Stage.Open(os.path.join(OUT, name))
    prims = list(stage.Traverse())
    colliders = sum(1 for p in prims if p.HasAPI(UsdPhysics.CollisionAPI))
    bodies = sum(1 for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI))
    roots = sum(1 for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    cameras = [str(p.GetPath()) for p in prims if p.IsA(UsdGeom.Camera)]
    size = os.path.getsize(os.path.join(OUT, name))
    print(f"BUILT {name:16s} {size:7d} bytes  prims={len(prims):4d} colliders={colliders:3d} rigid_bodies={bodies:3d} "
          f"articulation_roots={roots} default={stage.GetDefaultPrim().GetPath()} up={UsdGeom.GetStageUpAxis(stage)} "
          f"m/unit={UsdGeom.GetStageMetersPerUnit(stage)}")
    for path in cameras:
        print(f"BUILT   camera {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    for build in (build_bench, build_lights, build_cube, build_bowl, build_assembly):
        build()
    for name in ("lab_bench.usd", "lab_lights.usd", "paper_cube.usd", "food_bowl.usd", "lab_scene.usd"):
        report(name)
    print("BUILD_SCENE_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        _kit.__exit__(None, None, None)
