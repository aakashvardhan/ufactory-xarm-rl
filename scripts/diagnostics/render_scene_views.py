"""Render the saved lab scene from photo-comparison viewpoints without advancing physics.

Usage: C:\\IsaacLab3\\.venv\\Scripts\\python.exe scripts/diagnostics/render_scene_views.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render():
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        import omni.usd
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom

        context = omni.usd.get_context()
        assert context.open_stage(os.path.join(ROOT, "assets", "usd", "lab_scene.usd"))
        for _ in range(15):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        views = (
            ("front", (1.65, 0.45, 1.75), (0.30, 0.62, 1.12), (0, 0, 1)),
            ("opposite", (-0.75, -0.55, 1.75), (0.30, 0.60, 1.10), (0, 0, 1)),
            ("detail", (1.10, -0.50, 1.92), (0.46, 0.35, 1.29), (0, 0, 1)),
            ("plan", (0.342, 0.69, 3.0), (0.342, 0.69, 0.76), (-1, 0, 0)),
            ("room", (3.3, -3.5, 6.0), (0.34, 0.55, 0.5), (0, 0, 1)),
        )
        for name, eye, target, up in views:
            # A plan is a tabletop layout check; lift the overhead fixtures out of this view only.
            for prim in stage.GetPrimAtPath("/Scene/Bench/RoomDetails").GetChildren():
                if prim.GetName().startswith(("CeilingTray_", "CeilingHanger_", "StripLight_")):
                    for child in Usd.PrimRange(prim):
                        imageable = UsdGeom.Imageable(child)
                        if imageable:
                            imageable.CreateVisibilityAttr().Set("invisible" if name == "plan" else "inherited")
            camera = UsdGeom.Camera.Define(stage, "/PhotoReview/" + name)
            transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse()
            camera.MakeMatrixXform().Set(transform)
            camera.CreateFocalLengthAttr(22.0)
            camera.CreateHorizontalApertureAttr(36.0)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
            if name == "plan":
                camera.CreateProjectionAttr("orthographic")
                camera.CreateHorizontalApertureAttr(22.0)
                camera.CreateVerticalApertureAttr(16.5)
            product = rep.create.render_product(str(camera.GetPath()), (1280, 960))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach([product])
            rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=True)
            pixels = rgb.get_data()
            assert pixels.size and pixels[..., :3].mean() > 1, name
            output = os.path.join(ROOT, "logs", "photo_scene_" + name + ".png")
            Image.fromarray(pixels[..., :3]).save(output)
            print("PHOTO_RENDER_OK", name, output, flush=True)
            rgb.detach([product])
            product.destroy()
        print("PHOTO_REVIEW_OK", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    code = render()
    sys.stdout.flush()
    sys.stderr.flush()
    # Isolated render worker: avoid Kit teardown re-entering or saving the user's interactive stage.
    os._exit(code)
