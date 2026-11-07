"""Replay two lossless real episodes in A/B/A order in one loaded model."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from validation import attach, decode, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["packaged", "reference"], required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xpl-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    args = parser.parse_args()
    os.environ.pop("BAGEL_VALIDATION_TRACE", None)
    from backend import configure_backend
    configure_backend(1)
    if args.implementation == "reference":
        if args.reference_root is None:
            parser.error("--reference-root is required")
        root = args.reference_root.resolve()
        sys.path[:0] = [str(root / "source/vpp2"), str(root / "source/vpp2/src"),
                       str(root / "source/simulator"), str(root / "source/simulator/XPolicyLab")]
        module = importlib.import_module("XPolicyLab.policy.VPP2.model")
        config = json.loads((root / "inputs/vpp2-policy.json").read_text())
    else:
        if args.xpl_root is None:
            parser.error("--xpl-root is required")
        root = args.xpl_root.resolve()
        sys.path[:0] = [str(root.parent), str(root)]
        module = importlib.import_module("XPolicyLab.policy.bagel_robotics_1.model")
        import yaml
        config = yaml.safe_load((root / "policy/bagel_robotics_1/deploy.yml").read_text())
        config.update(seed=1, bench_name="RoboDojo", env_cfg_type="arx_x5", action_type="ee",
                      ckpt_name="cp1000", task_name=None)
    model = module.Model(config)
    recorder = attach(model, args.output)
    import torch
    environment = {
        "implementation": args.implementation, "python": sys.version,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "model_source": module.__file__,
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    fixtures = json.loads(args.fixtures.read_text())["fixtures"]
    if len(fixtures) != 2:
        raise ValueError("Two independent fixtures are required for A/B/A reset validation")
    for source in [fixtures[0], fixtures[1], fixtures[0]]:
        model.reset()
        client = {model.CLIENT_ID_FIELD: "validation-replay"}
        model.reset_client(obs=client)
        for row in records(source):
            if row["kind"] == "observation":
                obs = decode(row["data"]["observation"], Path(source) / "arrays")
                obs.update(client)
                model.update_obs(obs)
            elif row["kind"] == "commands":
                model.get_action(obs=client)
        recorder.stream.flush()
    recorder.stream.close()
    print("Completed A/B/A replay:", args.implementation, flush=True)


if __name__ == "__main__":
    main()
