"""Build assets/usd/xarm6_gripper.usd: a thin override layer over Isaac Sim's UFactory xArm6 asset.

The stock asset (Robots/Ufactory/xarm6/xarm6.usd, variant xarm_gripper) carries TWO articulation roots:
/UF_ROBOT/root_joint (arm) and /UF_ROBOT/gripper/root_joint (the gripper's own root, kept from its
standalone asset). Isaac Lab needs exactly one, so this layer removes the gripper's root API and keeps
everything else (meshes, mimic joints, drives) by reference.
"""
import os

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Usd, UsdPhysics  # noqa: E402

ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"
SRC = f"{ASSET_ROOT}/Robots/Ufactory/xarm6/xarm6.usd"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "assets", "usd", "xarm6_gripper.usd"))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    stage = Usd.Stage.CreateNew(OUT)
    root = stage.DefinePrim("/UF_ROBOT")
    root.GetReferences().AddReference(SRC)
    stage.SetDefaultPrim(root)
    vs = root.GetVariantSets()
    for name, choice in (("Physics", "PhysX"), ("Variant_Set", "xarm_gripper"), ("Sensor", "None")):
        if vs.HasVariantSet(name):
            vs.GetVariantSet(name).SetVariantSelection(choice)
    roots = [p.GetPath() for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print("articulation roots before:", [str(r) for r in roots])
    g = stage.OverridePrim("/UF_ROBOT/gripper/root_joint")
    g.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    for schema in list(g.GetAppliedSchemas()):
        if schema.startswith("PhysxArticulation"):
            g.RemoveAppliedSchema(schema)
    stage.GetRootLayer().Save()
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print("articulation roots after:", roots)
    assert roots == ["/UF_ROBOT/root_joint"], roots
    print("wrote", OUT)


if __name__ == "__main__":
    main()
    app.close()
