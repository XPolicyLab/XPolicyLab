#!/usr/bin/env python3
"""Serve OpenWAM-alpha as an XPolicyLab policy (websocket; end-effector actions) for run_policy.

    CUDA_VISIBLE_DEVICES=4 HARNESS_ROBODOJO_ROOT=/data/RoboDojo python services/servers/serve_openwam_policy.py \\
        --ckpt-dir /data/weights/openwam --port 19010

XPolicyLab's own server for the OpenWAM adapter (setup_policy_server.py) loads the whole
checkpoint on the GPU with the pinned numpy, which neither unpickles the checkpoint's
normalization stats (pickled under numpy 2) nor fits one 24 GB GPU. This server loads it as
the world-model service does (serve_openwam.load_adapter: the stats re-saved under numpy 1,
the text encoder on the CPU) and serves the adapter through XPolicyLab's websocket
PolicyServer, so run_policy talks to it like any XPolicyLab policy (reset, update_obs,
get_action: link6 poses and grippers of both arms).
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from serve_openwam import load_adapter, xpolicylab_root  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19010)
    parser.add_argument("--env-cfg-type", default="arx_x5")
    parser.add_argument("--replan-steps", type=int, help="actions of each predicted chunk to execute "
                                                         "(default: the checkpoint's horizon)")
    args = parser.parse_args()

    model, seconds = load_adapter(args.ckpt_dir, decode_video=False, model_cfg={
        "env_cfg_type": args.env_cfg_type, "eval_batch": False, "replan_steps": args.replan_steps})
    sys.path.insert(0, str(xpolicylab_root()))  # client_server ships in the XPolicyLab checkout
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    import torch

    print(json.dumps({"serving": "openwam-policy", "port": args.port, "load_seconds": round(seconds),
                      "gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 1),
                      "replan_steps": model.replan_steps}), flush=True)
    server = PolicyServer(model, PolicyServerConfig(host=args.host, port=args.port, ws_ping_interval_s=20.0,
                                                    ws_ping_timeout_s=20.0))
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
