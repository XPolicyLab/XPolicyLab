"""Select the visible SAPIEN GPU, then run the unmodified native evaluator."""

from pathlib import Path
import os
import runpy
import sys


def main():
    script = Path(sys.argv[1]).absolute()
    root = script.parent.parent
    os.chdir(root)
    for path in (root, script.parent, root / "envs/curobo/src"):
        if path.is_dir():
            sys.path.insert(0, str(path))
    import sapien

    if hasattr(sapien, "Device"):
        import sapien.core as legacy

        device = sapien.Device("cuda:0")
        renderer_base = sapien.pysapien.render.SapienRenderer

        class BoundRenderer(legacy.SapienRenderer):
            def __init__(self, **unused):
                renderer_base.__init__(self, device)

        def create_scene(engine, config=None):
            if config is not None:
                sapien.physx.set_scene_config(config)
            return sapien.Scene(
                [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device)]
            )

        legacy.SapienRenderer = BoundRenderer
        legacy.Engine.create_scene = create_scene
        print(f"[WorldLA] Renderer visible cuda:0, PCI={device.pci_string}", flush=True)
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
