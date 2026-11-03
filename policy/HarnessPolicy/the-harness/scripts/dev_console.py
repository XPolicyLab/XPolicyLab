#!/usr/bin/env python3
"""Drive a runtime worker by hand with the kernel API, one snippet at a time.

    python scripts/dev_console.py --endpoint http://127.0.0.1:8771 --reset stack_bowls 1
    python scripts/dev_console.py --endpoint http://127.0.0.1:8771 --exec snippet.py
    python scripts/dev_console.py --endpoint http://127.0.0.1:8771 --code "print(get_state()['step'])"

For checking skills with a known-good program, independent of any model.
The episode id is kept in $TMPDIR/dev_console_<port>.json between calls; the
robot state lives in the worker, Python variables do not persist. Images passed
to show() are written to --images. Runs in the harness-policy environment.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.core.robot import RemoteEnvironment  # noqa: E402
from harness_policy.runtime.session import create_session  # noqa: E402
from harness_policy.services.registry import load_services  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8771")
    parser.add_argument("--reset", nargs=2, metavar=("TASK", "SEED"))
    parser.add_argument("--exec", type=Path, help="Python file to run in the kernel namespace")
    parser.add_argument("--code", help="Python code to run in the kernel namespace")
    parser.add_argument("--vlm", default="http://127.0.0.1:8800/v1")
    parser.add_argument("--vlm-model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--vlm-backend", choices=["openai_chat", "openrouter"], default="openai_chat",
                        help="openrouter reads OPENROUTER_API_KEY; --vlm is then ignored")
    parser.add_argument("--segmenter", default="http://127.0.0.1:8810")
    parser.add_argument("--policy", default="ws://127.0.0.1:19008", help="learned policy server (default: Simate's port)")
    parser.add_argument("--policy-name", default="simate", help="the name run_policy takes for it")
    parser.add_argument("--reactive-budget-ms", type=float, default=20.0)
    parser.add_argument("--world-model", default="http://127.0.0.1:8820", help="OpenWAM server (serve_openwam.py)")
    parser.add_argument("--images", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "dev_console_images")
    args = parser.parse_args()
    port = args.endpoint.rsplit(":", 1)[-1]
    state_file = Path(os.environ.get("TMPDIR", "/tmp")) / f"dev_console_{port}.json"
    if args.reset:
        env = RemoteEnvironment(args.endpoint, timeout=600)
        env.tools()
        observation = env.reset(args.reset[0], int(args.reset[1]))
        state_file.write_text(json.dumps({"episode_id": observation.episode_id,
                                          "instruction": observation.instruction}))
        print(f"reset {args.reset[0]} seed {args.reset[1]}: {observation.instruction}")
    saved = json.loads(state_file.read_text())
    episode = saved["episode_id"]
    # The policy's client settings from its service file (scoped, description), as the agent
    # gets them: a policy with memory then keeps it per episode, as in the agent's runs.
    spec = load_services().get(args.policy_name)
    policy = {**(spec.client if spec else {}), "url": args.policy}
    session = create_session({
        "endpoint": args.endpoint, "episode_id": episode,
        "vlm": ({"backend": "openai_chat", "base_url": args.vlm, "model": args.vlm_model}
                if args.vlm_backend == "openai_chat" else
                {"backend": "openrouter", "model": args.vlm_model, "reasoning_effort": "none", "retries": 3}),
        "segmenter": {"url": args.segmenter},
        "policies": {args.policy_name: policy},
        "world_models": {"openwam": {"url": args.world_model}},
        "reactive_budget_ms": args.reactive_budget_ms,
        "layers": {"learned": True, "reactive": True, "handlers": True},
    })
    session.namespace["INSTRUCTION"] = saved.get("instruction", "")
    code = args.exec.read_text() if args.exec else (args.code or "")
    if code:
        try:
            exec(compile(code, "<console>", "exec"), session.namespace)
        finally:
            drained = session.drain()
            args.images.mkdir(parents=True, exist_ok=True)
            for index, item in enumerate(drained["attachments"]):
                path = args.images / f"{index:02d}_{item['caption'].replace(' ', '_') or 'image'}.jpg"
                path.write_bytes(base64.b64decode(item["image_url"].split(",", 1)[1]))
                print("image:", path)
            for event in drained["events"]:
                print("execution:", event["skill"], event["event"]["kind"], json.dumps(event["event"]["detail"])[:200])


if __name__ == "__main__":
    main()
