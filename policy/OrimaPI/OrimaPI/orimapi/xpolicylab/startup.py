"""Prepare and check the same Kit experience used by RoboDojo, without a task."""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from .kit_cache import kit_cache_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    # Kit must receive only its own arguments, not this probe's CLI options.
    sys.argv = [sys.argv[0]]

    from isaaclab.app import AppLauncher
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    options = argparse.Namespace(
        headless=args.headless,
        enable_cameras=True,
        device=f"cuda:{args.gpu}",
        kit_args=(
            f"--/renderer/activeGpu={args.gpu} --/physics/cudaDevice={args.gpu} "
            "--/renderer/multiGpu/enabled=false "
            + kit_cache_arguments(os.environ)
            + (" --/app/extensions/registryEnabled=false" if args.offline else "")
        ),
    )
    add_zero_delay_kit_args(options)
    launcher = AppLauncher(options)
    try:
        # A zero exit code alone is insufficient: Kit can replace an exception's
        # exit status during shutdown. Require imports, GPU work and a ready file.
        import omni.kit.usd  # noqa: F401
        from isaaclab.assets import Articulation  # noqa: F401
        import torch

        launcher.app.update()
        value = torch.arange(4, device=f"cuda:{args.gpu}")
        assert (value * value).cpu().tolist() == [0, 1, 4, 9]
        report = {
            "gpu": args.gpu,
            "offline": args.offline,
            "experience": launcher._sim_experience_file,
        }
        # Kit's fast shutdown can terminate Python inside app.close().
        args.ready_file.write_text(json.dumps(report, indent=2) + "\n")
        print("ORIMAPI_ISAACLAB_STARTUP_OK " + json.dumps(report), flush=True)
    except BaseException:
        # Preserve the original failure before Kit's shutdown replaces its exit.
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
