"""Photo-derived visual details for build_scene.py (dimensions in metres).

The measured scene remains in the builder. These are deliberately separate, estimated visual details:
butcher-block boards, paper wrapping, clear container lip, camera fittings, equipment and room context.
Textures are deterministic procedural assets, not crops of the supplied photographs.
"""
import math
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


def make_textures(folder):
    import numpy as np
    from PIL import Image, ImageFilter

    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(2475)
    # Image U follows the long table axis. Narrow, staggered boards match the photographed block top.
    w, h = 2048, 1024
    wood = np.zeros((h, w, 3), dtype=np.float32)
    for row, start in enumerate(range(0, h, 48)):
        stop = min(start + 48, h)
        cursor = -int(rng.integers(40, 450))
        while cursor < w:
            end = cursor + int(rng.integers(240, 650))
            lo, hi = max(cursor, 0), min(end, w)
            if hi <= lo:
                cursor = end
                continue
            base = np.array([212, 185, 143]) + rng.normal(0, 8) * np.array([1, 0.90, 0.65])
            yy, xx = np.mgrid[start:stop, lo:hi]
            grain = (2.8 * np.sin(yy * 1.9 + 1.8 * np.sin(xx * 0.012 + row))
                     + 1.8 * np.sin(yy * 5.3 + xx * 0.004)
                     + rng.normal(0, 1.5, (stop - start, hi - lo)))
            wood[start:stop, lo:hi] = base + grain[..., None]
            if lo > 0:
                wood[start:stop, lo:lo + 2] *= 0.90
            cursor = end
        wood[start:start + 1] *= 0.91
    Image.fromarray(np.uint8(np.clip(wood, 0, 255))).save(os.path.join(folder, "maple_blocks.png"))
    noise = rng.normal(0, 1, (512, 512))
    paper = np.stack([236 + noise * 2, 235 + noise * 2, 228 + noise * 2], axis=-1)
    # Soft wrinkles in the paper-wrapped cube rather than a polished white block.
    yy, xx = np.mgrid[:512, :512]
    for _ in range(30):
        slope, intercept = rng.uniform(-2, 2), rng.uniform(-500, 900)
        crease = np.exp(-((yy - slope * xx - intercept) / rng.uniform(1, 3))**2)
        paper -= crease[..., None] * rng.uniform(4, 14)
    Image.fromarray(np.uint8(np.clip(paper, 0, 255))).filter(ImageFilter.GaussianBlur(0.4)).save(
        os.path.join(folder, "wrinkled_paper.png"))


class PhotoDetails:
    def __init__(self, stage, box, tube, look):
        self.s, self.box, self.tube, self.look = stage, box, tube, look

    def material(self, path, color, rough=0.5, metal=0):
        return self.look(self.s, path, color, rough, metallic=metal)

    def textured(self, path, texture, rough=0.5):
        material = self.material(path, (0.8, 0.7, 0.5), rough)
        surface = UsdShade.Shader(self.s.GetPrimAtPath(path + "/Shader"))
        reader = UsdShade.Shader.Define(self.s, path + "/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(self.s, path + "/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surface.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
        return material

    def glass(self, path, color, rough):
        """Thin clear polymer in Kit's bundled MDL renderer, with a USD Preview fallback."""
        material = UsdShade.Material.Define(self.s, path)
        shader = UsdShade.Shader.Define(self.s, path + "/GlassShader")
        shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
        shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.49)
        shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(rough)
        shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        return material

    def quad(self, path, points, visual):
        mesh = UsdGeom.Mesh.Define(self.s, path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
        st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(visual)
        return mesh

    def line(self, path, points, radius, visual):
        # Catmull-Rom samples make a smooth cable from short tube meshes. Remove repeated end points
        # so no segment has a degenerate tangent.
        clean = []
        for point in points:
            point = Gf.Vec3d(*point)
            if not clean or (point - clean[-1]).GetLength() > 1e-7:
                clean.append(point)
        padded = [clean[0]] + clean + [clean[-1]]
        samples = []
        for i in range(1, len(padded) - 2):
            p0, p1, p2, p3 = padded[i - 1:i + 3]
            for k in range(6):
                t = k / 6
                samples.append(0.5 * (2 * p1 + (-p0 + p2) * t
                               + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                               + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
        samples.append(clean[-1])
        root = UsdGeom.Xform.Define(self.s, path)
        for i, (a, b) in enumerate(zip(samples, samples[1:])):
            if (b - a).GetLength() > 1e-7:
                self.tube(self.s, path + f"/Segment_{i}", a, b, radius, visual)
        return root

    def ball(self, path, centre, radius, visual):
        ball = UsdGeom.Sphere.Define(self.s, path)
        ball.CreateRadiusAttr(radius)
        ball.AddTranslateOp().Set(Gf.Vec3d(*centre))
        UsdShade.MaterialBindingAPI.Apply(ball.GetPrim()).Bind(visual)

    def bench(self, x0, y0, size, height, robot_x, pole_xy):
        sx, sy, thickness = size
        mat = self.textured("/World/Looks/BlockWood", "./textures/maple_blocks.png", 0.48)
        self.quad("/World/Bench/WoodSurface", [(x0, y0, 0.0001), (x0, y0 + sy, 0.0001),
                  (x0 + sx, y0 + sy, 0.0001), (x0 + sx, y0, 0.0001)], mat)
        black = self.material("/World/Looks/BlackHardware", (0.025, 0.028, 0.032), 0.6)
        silver = self.material("/World/Looks/BrushedHardware", (0.5, 0.53, 0.56), 0.35, 0.65)
        for i in range(4):
            leg = self.s.GetPrimAtPath(f"/World/Bench/Frame/Leg_{i}")
            UsdShade.MaterialBindingAPI.Apply(leg).Bind(black)
            position = leg.GetAttribute("xformOp:translate").Get()
            self.tube(self.s, f"/World/Bench/Frame/Foot_{i}", (position[0], position[1], -height),
                      (position[0], position[1], -height + 0.025), 0.038, black)
        # The original plate collider remains; these dark inset slots and screw heads are visual.
        for i, (dx, dy) in enumerate(((-0.083, -0.067), (-0.083, 0.067), (0.083, -0.067), (0.083, 0.067))):
            self.box(self.s, f"/World/MountA_Details/Slot_{i}", (0.008, 0.035, 0.0004),
                     (robot_x + dx, dy, 0.0103), black)
        for i in range(6):
            a = 2 * math.pi * i / 6
            x, y = robot_x + 0.074 * math.cos(a), 0.074 * math.sin(a)
            self.tube(self.s, f"/World/MountA_Details/Washer_{i}", (x, y, 0.010), (x, y, 0.011), 0.006, silver)
            self.tube(self.s, f"/World/MountA_Details/Bolt_{i}", (x, y, 0.011), (x, y, 0.014), 0.004, black)
        # Upper collar, locking screw and the externally routed USB lead are prominent in IMG_2475/6.
        px, py = pole_xy
        self.tube(self.s, "/World/CameraStand/TopCollar", (px, py, 0.525), (px, py, 0.557), 0.019, silver)
        self.box(self.s, "/World/CameraStand/Clamp/Lock", (0.04, 0.04, 0.024), (px, py, 0.025), black)
        self.line("/World/CameraStand/USBLead", [(px, py + 0.014, 0.55), (px, py + 0.014, 0.52),
                  (px + 0.012, py + 0.014, 0.30), (px + 0.015, py + 0.014, 0.05),
                  (px + 0.05, py + 0.04, -0.10), (px + 0.12, py + 0.12, -0.18)], 0.0024, black)

    def camera(self, path, size, behind):
        black = self.material(path + "/Looks/Bezel", (0.015, 0.018, 0.022), 0.5)
        glass = self.material(path + "/Looks/Lens", (0.018, 0.036, 0.047), 0.12, 0.3)
        metal = self.material(path + "/Looks/Rear", (0.45, 0.48, 0.5), 0.3, 0.65)
        w, h, d = size
        front, back = behind - d / 2, behind + d / 2
        self.box(self.s, path + "/FrontPanel", (w - 0.004, h - 0.004, 0.001), (0, 0, front - 0.0005), black)
        for i, x in enumerate((-0.034, -0.016, 0.017, 0.034)):
            self.tube(self.s, path + f"/Lens_{i}", (x, 0, front - 0.0018), (x, 0, front - 0.001),
                      0.0044 if i != 1 else 0.0033, glass)
        for i in range(11):
            self.box(self.s, path + f"/HeatFin_{i}", (0.0017, h - 0.006, 0.0015),
                     (-0.035 + i * 0.007, 0, back + 0.00075), metal)
        for i, x in enumerate((-0.040, 0.040)):
            self.tube(self.s, path + f"/Screw_{i}", (x, 0.008, front - 0.001), (x, 0.008, front), 0.0011, metal)
        self.box(self.s, path + "/USBPlug", (0.010, 0.009, 0.009), (w / 2 + 0.004, 0, behind), black)

    def cube(self, edge):
        paper = self.textured("/Cube/Looks/WrappedPaper", "./textures/wrinkled_paper.png", 0.85)
        h = edge / 2
        # Separate faces give every side real UVs while retaining the original simple cube collider.
        faces = [((-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)),
                 ((-h, -h, -h), (-h, h, -h), (h, h, -h), (h, -h, -h)),
                 ((-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)),
                 ((h, h, -h), (-h, h, -h), (-h, h, h), (h, h, h)),
                 ((-h, h, -h), (-h, -h, -h), (-h, -h, h), (-h, h, h)),
                 ((h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h))]
        # Hide only the original visual, retaining its collision API.
        self.s.GetPrimAtPath("/Cube/geom").GetAttribute("purpose").Set("guide")
        for i, points in enumerate(faces):
            self.quad(f"/Cube/Paper/Face_{i}", points, paper)
        seam = self.material("/Cube/Looks/PaperFold", (0.73, 0.72, 0.68), 0.9)
        self.quad("/Cube/Paper/Fold", [(-h, -h - 0.0001, -h), (h, -h - 0.0001, -h),
                  (h * 0.3, -h - 0.0001, h * 0.2), (-h * 0.3, -h - 0.0001, h * 0.2)], seam)

    def bowl(self, rim, radius, height):
        # Match the clear food container, not the previous opaque grey bin.
        surface = UsdShade.Shader(self.s.GetPrimAtPath("/Bowl/Looks/Frosted/Shader"))
        if surface:
            surface.GetInput("diffuseColor").Set(Gf.Vec3f(0.88, 0.91, 0.91))
            surface.GetInput("opacity").Set(0.10)
            surface.GetInput("roughness").Set(0.12)
        self.glass("/Bowl/Looks/Frosted", (0.98, 0.99, 0.99), 0.12)
        # The thick rolled lip reads as milky white in the references, even when the sides are clear.
        lip = self.look(self.s, "/Bowl/Looks/Rim", (0.89, 0.91, 0.91), 0.28)
        points = []
        half = rim / 2
        for x, y, start in ((half - radius, half - radius, 0), (-half + radius, half - radius, 90),
                            (-half + radius, -half + radius, 180), (half - radius, -half + radius, 270)):
            for k in range(9):
                a = math.radians(start + k * 90 / 8)
                points.append((x + radius * math.cos(a), y + radius * math.sin(a), height))
        for i, (a, b) in enumerate(zip(points, points[1:] + points[:1])):
            self.tube(self.s, f"/Bowl/RolledLip/Segment_{i}", a, b, 0.0018, lip)

    def equipment(self, robot_x, pole_xy, table_height, bench_x0, bench_y0, bench_size):
        root = "/World/Equipment"
        black = self.material(root + "/Looks/Black", (0.018, 0.021, 0.025), 0.45)
        vent = self.material(root + "/Looks/Vents", (0.006, 0.007, 0.009), 0.75)
        steel = self.material(root + "/Looks/Silver", (0.58, 0.6, 0.62), 0.4, 0.55)
        paper = self.material(root + "/Looks/Paper", (0.92, 0.92, 0.9), 0.85)
        # Tall black workstation/controller housing beside the mount. All dimensions inferred from photos.
        x, y, w, d, h = 0.55, 0.95, 0.31, 0.22, 0.43
        self.box(self.s, root + "/Tower/Case", (w, d, h), (x, y, h / 2), black, collide=True)
        for i in range(17):
            self.box(self.s, root + f"/Tower/FrontVent_{i}", (w * 0.7, 0.001, 0.0025),
                     (x, y - d / 2 - 0.0008, h - 0.07 - i * 0.009), vent)
        for i in range(8):
            self.box(self.s, root + f"/Tower/TopVent_{i}", (w * 0.7, 0.005, 0.001),
                     (x, y - 0.07 + i * 0.018, h + 0.0008), vent)
        self.box(self.s, root + "/Tower/SidePanel", (0.001, d - 0.022, h - 0.03),
                 (x - w / 2 - 0.0006, y, h / 2), black)
        self.box(self.s, root + "/Tower/RearPort", (0.035, 0.002, 0.020), (x + 0.06, y + d / 2 + 0.001, 0.06), vent)
        for i, z in enumerate((0.025, 0.405)):
            self.ball(root + f"/Tower/PanelScrew_{i}", (x - w / 2 - 0.001, y - d / 2 + 0.015, z), 0.002, steel)
        self.line(root + "/Tower/PowerCable", [(x + 0.06, y + d / 2, 0.06), (x + 0.06, y + d / 2, 0.04),
                  (0.74, 1.16, 0.03), (0.79, 1.20, 0.01), (0.85, 1.22, -0.05),
                  (0.87, 1.24, -0.40), (0.87, 1.24, -0.5)], 0.004, black)
        # White sheet, small silver tabletop tripod and clear pump bottle visible beside the stand.
        self.box(self.s, root + "/Paper", (0.21, 0.17, 0.0003), (0.35, 1.29, 0.0002), paper)
        tx, ty = 0.58, 1.30
        for i in range(3):
            a = 2 * math.pi * i / 3
            foot = (tx + 0.065 * math.cos(a), ty + 0.065 * math.sin(a), 0.005)
            self.tube(self.s, root + f"/MiniTripod/Leg_{i}", foot, (tx, ty, 0.12), 0.004, steel)
            self.ball(root + f"/MiniTripod/Foot_{i}", foot, 0.006, black)
        self.tube(self.s, root + "/MiniTripod/Head", (tx, ty, 0.12), (tx, ty, 0.15), 0.014, black)
        bottle = self.look(self.s, root + "/Looks/Bottle", (0.64, 0.81, 0.84), 0.2, opacity=0.32)
        self.glass(root + "/Looks/Bottle", (0.94, 0.98, 0.99), 0.09)
        bx, by = 0.74, 1.39
        self.box(self.s, root + "/PumpBottle/Body", (0.055, 0.043, 0.12), (bx, by, 0.06), bottle)
        self.box(self.s, root + "/PumpBottle/Label", (0.04, 0.0005, 0.065), (bx, by - 0.022, 0.065), paper)
        self.tube(self.s, root + "/PumpBottle/Neck", (bx, by, 0.12), (bx, by, 0.16), 0.010, paper)
        self.box(self.s, root + "/PumpBottle/Pump", (0.035, 0.018, 0.009), (bx - 0.010, by, 0.16), paper)
        # Near-side lab context, inside the estimated four-wall room.
        grey = self.material(root + "/Looks/Desk", (0.49, 0.49, 0.46), 0.75)
        self.box(self.s, root + "/BackDesk/Top", (0.52, bench_size[1], 0.025),
                 (bench_x0 - 0.39, bench_y0 + bench_size[1] / 2, -0.015), grey)
        for i, yy in enumerate((bench_y0 + 0.12, bench_y0 + bench_size[1] - 0.12)):
            self.box(self.s, root + f"/BackDesk/Leg_{i}", (0.04, 0.05, table_height - 0.04),
                     (bench_x0 - 0.39, yy, -(table_height - 0.04) / 2 - 0.04), black)
        screen = self.material(root + "/Looks/Screen", (0.007, 0.011, 0.016), 0.28)
        self.box(self.s, root + "/DeskMonitor/Base", (0.12, 0.19, 0.007), (bench_x0 - 0.48, 0.04, 0.008), black)
        self.tube(self.s, root + "/DeskMonitor/Stem", (bench_x0 - 0.48, 0.04, 0.012),
                  (bench_x0 - 0.48, 0.04, 0.16), 0.012, black)
        self.box(self.s, root + "/DeskMonitor/Body", (0.020, 0.33, 0.22), (bench_x0 - 0.48, 0.04, 0.23), black)
        self.box(self.s, root + "/DeskMonitor/Screen", (0.001, 0.307, 0.196), (bench_x0 - 0.469, 0.04, 0.23), screen)
        self.box(self.s, root + "/Ruler/Body", (0.028, 0.30, 0.001), (bench_x0 - 0.20, 0.46, 0.002), steel)
        for i in range(31):
            self.box(self.s, root + f"/Ruler/Tick_{i}", (0.014 if i % 5 == 0 else 0.007, 0.0005, 0.0002),
                     (bench_x0 - 0.207, 0.31 + i * 0.01, 0.0027), black)
        blue = self.material(root + "/Looks/BluePen", (0.018, 0.07, 0.42), 0.35)
        self.tube(self.s, root + "/Pen/Body", (bench_x0 - 0.30, 0.70, 0.004), (bench_x0 - 0.30, 0.80, 0.004), 0.004, steel)
        self.tube(self.s, root + "/Pen/Cap", (bench_x0 - 0.30, 0.80, 0.004), (bench_x0 - 0.30, 0.84, 0.004), 0.0045, blue)
        self.box(self.s, root + "/StorageCabinet", (0.38, 0.28, 1.85),
                 (0.30, bench_y0 - 0.58, -table_height + 0.925), black)
        self.box(self.s, root + "/StorageCabinet/DoorSeam", (0.002, 0.002, 1.79),
                 (0.30, bench_y0 - 0.438, -table_height + 0.925), vent)
        self.tube(self.s, root + "/StorageCabinet/Handle", (0.34, bench_y0 - 0.435, 0.12),
                  (0.34, bench_y0 - 0.435, 0.25), 0.006, steel)

    def robot(self, root, table_height):
        grip = root + "/gripper/xarm_gripper_base_link"
        metal = self.material("/Scene/Looks/Bracket", (0.62, 0.65, 0.67), 0.32, 0.6)
        black = self.material("/Scene/Looks/CableLoom", (0.012, 0.015, 0.018), 0.8)
        # Upright aluminium L bracket above the flange, including the adjustment slots.
        self.box(self.s, grip + "/CameraBracket/Upright", (0.055, 0.034, 0.003), (0.044, 0, -0.007), metal)
        self.box(self.s, grip + "/CameraBracket/Seat", (0.003, 0.034, 0.046), (0.070, 0, 0.015), metal)
        for i, y in enumerate((-0.009, 0, 0.009)):
            self.box(self.s, grip + f"/CameraBracket/Slot_{i}", (0.025, 0.0025, 0.0004),
                     (0.046, y, -0.0087), black)
        for i, y in enumerate((-0.012, 0.012)):
            self.tube(self.s, grip + f"/CameraBracket/Bolt_{i}", (0.02, y, -0.009),
                      (0.02, y, -0.011), 0.0025, black)
        cache = UsdGeom.XformCache()
        path = "/Scene/RobotCabling"
        # Points are authored in /Scene's tabletop frame, not the floor/world frame.
        positions = [cache.GetLocalToWorldTransform(self.s.GetPrimAtPath(root + "/" + link)).ExtractTranslation()
                     for link in ("world", "link1", "link3", "link4", "link5", "link6", "link7")]
        points = [(positions[0][0], positions[0][1] + 0.06, 0.035)]
        for i, point in enumerate(positions[1:]):
            points.append((point[0], point[1] + 0.068, point[2] - table_height + (0.04 if i > 2 else 0)))
        points.extend([points[-1], points[-1]])
        self.line(path + "/ArmLoom", [points[0], points[0]] + points, 0.004, black)
        # Slack at the base is visible in all low-angle references.
        x, y, _ = points[0]
        self.line(path + "/BaseSlack", [(x, y, 0.07), (x, y, 0.07), (x + 0.08, y - 0.03, 0.04),
                  (x + 0.12, y - 0.08, 0.008), (x + 0.08, y - 0.16, 0.006),
                  (x + 0.01, y - 0.19, 0.008), (x, y - 0.18, 0.035), (x, y - 0.18, 0.035)], 0.006, black)
        self.line(grip + "/WristCameraUSB", [(0.07, -0.05, 0.04), (0.07, -0.05, 0.04),
                  (0.04, -0.06, 0), (0, -0.065, -0.02), (-0.04, -0.055, -0.03),
                  (-0.04, -0.055, -0.03)], 0.002, black)

    def room(self, x0, y0, size, table_height, gap, wall_height):
        root = "/World/RoomDetails"
        black = self.material(root + "/Looks/Black", (0.035, 0.038, 0.045), 0.7)
        white = self.material(root + "/Looks/White", (0.89, 0.9, 0.9), 0.45)
        glass = self.material(root + "/Looks/Display", (0.009, 0.015, 0.021), 0.16)
        # Wall-mounted whiteboard and switched-off monitor; intentionally no copied personal text.
        wx = x0 - gap + 0.012
        self.box(self.s, root + "/Whiteboard/Frame", (0.025, 0.73, 0.55), (wx, y0 + 0.45, 0.75), black)
        self.box(self.s, root + "/Whiteboard/Face", (0.002, 0.70, 0.52), (wx + 0.014, y0 + 0.45, 0.75), white)
        self.box(self.s, root + "/WallMonitor/Body", (0.035, 0.65, 0.39), (wx + 0.005, y0 + 1.30, 0.76), black)
        self.box(self.s, root + "/WallMonitor/Screen", (0.001, 0.62, 0.355), (wx + 0.024, y0 + 1.30, 0.76), glass)
        # Closed dark roller blinds on the end wall and paired suspended cable trays overhead.
        far_y = y0 + size[1] + gap - 0.02
        self.box(self.s, root + "/Blind", (size[0] + 0.95, 0.018, 1.15),
                 (x0 + size[0] / 2, far_y, 0.75), black)
        for i, x in enumerate((x0 + 0.02, x0 + size[0] - 0.02)):
            for j, z in enumerate((wall_height - table_height - 0.16, wall_height - table_height - 0.26)):
                self.box(self.s, root + f"/CeilingTray_{i}_{j}", (0.045, size[1] + 1.4, 0.045),
                         (x, y0 + size[1] / 2, z), black)
            for j, y in enumerate((y0 + 0.15, y0 + size[1] - 0.15)):
                self.tube(self.s, root + f"/CeilingHanger_{i}_{j}",
                          (x, y, wall_height - table_height), (x, y, wall_height - table_height - 0.26), 0.005, black)
        fixture = self.material(root + "/Looks/LightHousing", (0.82, 0.84, 0.85), 0.38, 0.25)
        diffuser = self.material(root + "/Looks/LightDiffuser", (1, 1, 1), 0.6)
        UsdShade.Shader(self.s.GetPrimAtPath(str(diffuser.GetPath()) + "/Shader")).CreateInput(
            "emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(4, 4, 4))
        cx = x0 + size[0] / 2
        light_z = wall_height - table_height - 0.30
        for i, y in enumerate((y0 + 0.46, y0 + 1.40)):
            self.box(self.s, root + f"/StripLight_{i}/Housing", (0.10, 0.76, 0.08),
                     (cx, y, light_z), fixture)
            self.box(self.s, root + f"/StripLight_{i}/Diffuser", (0.095, 0.75, 0.002),
                     (cx, y, light_z - 0.041), diffuser)
            for j, dy in enumerate((-0.27, 0.27)):
                self.tube(self.s, root + f"/StripLight_{i}/Suspension_{j}",
                          (cx, y + dy, wall_height - table_height),
                          (cx, y + dy, light_z + 0.04), 0.0012, black)
