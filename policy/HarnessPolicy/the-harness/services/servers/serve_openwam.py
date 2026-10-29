#!/usr/bin/env python3
"""Serve OpenWAM world-action predictions over HTTP for predict_future.

    CUDA_VISIBLE_DEVICES=4 python services/servers/serve_openwam.py \
        --ckpt-dir /mnt/.../OpenWAM-Alpha-Sim-RoboDojo --port 8820

POST /predict {"obs": <XPolicyLab observation; vision colors as image data URLs>,
               "seed": 42, "denoise_steps": null}
  -> {"frames": [JPEG data URL, ...], "actions": [{"left_ee_pose", "left_ee_joint_state",
      "right_ee_pose", "right_ee_joint_state"}, ...], "seconds": s}

frames is the predicted multiview video (the layout the checkpoint was trained on),
actions the predicted end-effector chunk in the env frame, as the XPolicyLab
OpenWAM adapter (policy/OpenWAM/model.py) converts it. The checkpoint does not fit
on one 24 GB GPU: the UMT5-XXL text encoder runs on the CPU (float32, one pass per
new instruction, cached) and everything else on the GPU. One request at a time.
"""

import argparse
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


_CONVERT = """
import sys
import numpy as np
import numpy.core.multiarray, numpy.core.numeric
# Pickled under numpy 2 (module numpy._core); only this process sees the alias.
sys.modules["numpy._core"] = np.core
sys.modules["numpy._core.multiarray"] = np.core.multiarray
sys.modules["numpy._core.numeric"] = np.core.numeric
data = np.load(sys.argv[1], allow_pickle=True)
np.save(sys.argv[2], data, allow_pickle=True)
"""


def numpy1_overlay(ckpt_dir):
    """The checkpoint's normalization_stats.npy was pickled under numpy 2, which the
    pinned numpy 1.26 cannot unpickle. Return a sibling directory of links to the
    checkpoint files with that one file re-saved under numpy 1 (the download itself
    stays untouched); ckpt_dir when it loads as is."""
    import subprocess
    import sys

    import numpy as np

    stats = ckpt_dir / "normalization_stats.npy"
    try:
        np.load(stats, allow_pickle=True)
        return ckpt_dir
    except ModuleNotFoundError:
        pass
    overlay = ckpt_dir.with_name(ckpt_dir.name + "-numpy1")
    overlay.mkdir(exist_ok=True)
    for item in ckpt_dir.iterdir():
        link = overlay / item.name
        if item.name != stats.name and not link.exists():
            link.symlink_to(item)
    subprocess.run([sys.executable, "-c", _CONVERT, str(stats), str(overlay / stats.name)], check=True)
    return overlay


def xpolicylab_root():
    """The XPolicyLab checkout of the RoboDojo checkout ($HARNESS_ROBODOJO_ROOT)."""
    return Path(os.environ.get("HARNESS_ROBODOJO_ROOT", "third_party/RoboDojo")).resolve() / "XPolicyLab"


def load_adapter(ckpt_dir, *, decode_video, model_cfg=None):
    """XPolicyLab's OpenWAM adapter (policy/OpenWAM/model.py) with the checkpoint loaded so
    that it fits one 24 GB GPU: normalization_stats through numpy1_overlay, the UMT5-XXL text
    encoder on the CPU, the rest on the GPU. `model_cfg` adds to the adapter's config
    (action_type ee, env_cfg_type arx_x5); `decode_video` also decodes the predicted views
    (predict_future needs them, a policy does not). Returns (model, load seconds)."""
    import torch
    from omegaconf import OmegaConf

    ckpt_dir = numpy1_overlay(Path(ckpt_dir))
    spec = importlib.util.spec_from_file_location("openwam_adapter", xpolicylab_root() / "policy" / "OpenWAM" /
                                                  "model.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    # The adapter's observation encoding and action decoding, without its own loading.
    model = adapter.Model({"action_type": "ee", "env_cfg_type": "arx_x5", **(model_cfg or {}),
                           "allow_dummy_policy": True})

    from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

    deploy_cfg = _load_deploy_yaml(str(model.openwam_root / "configs" / "deploy.yaml"))
    OmegaConf.update(deploy_cfg, "optimization.dit_cache.enabled", False, merge=False)
    OmegaConf.update(deploy_cfg, "optimization.compile.enabled", False, merge=False)
    OmegaConf.update(deploy_cfg, "optimization.decode_video", bool(decode_video), merge=False)
    OmegaConf.update(deploy_cfg, "inference.inference_mode", "sync", merge=False)
    started = time.perf_counter()
    server = build_server_from_config(deploy_cfg, str(ckpt_dir), device="cpu")
    server._init_policy()
    engine = server.engine
    device = torch.device("cuda")
    backbone = engine.architecture.video_backbone
    text_encoder = backbone._modules.pop("text_encoder").float()
    engine.architecture.to(device)
    for module in engine.architecture.modules():
        if hasattr(module, "_device"):
            module._device = device

    class CpuTextEncoder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, ids, mask):
            with torch.autocast("cpu", enabled=False):
                out = self.inner(ids.to("cpu"), mask.to("cpu"))
            return out.to(device=device, dtype=backbone.dtype)

    backbone.text_encoder = CpuTextEncoder(text_encoder)
    model._engine, model._wam_policy, model._preprocessor = engine, server._policy, server._obs_preprocessor
    if model.replan_steps is None:  # as the adapter's own loading: the checkpoint's horizon
        horizon = OmegaConf.select(server.cfg, "inference.inference_horizon", default=None)
        model.replan_steps = None if horizon is None else int(horizon)
    model.allow_dummy_policy = False
    return model, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8820)
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    model, seconds = load_adapter(args.ckpt_dir, decode_video=True)
    engine = model._engine
    lock = threading.Lock()
    print(json.dumps({"serving": "openwam", "port": args.port, "load_seconds": round(seconds),
                      "gpu_gb": round(torch.cuda.memory_allocated() / 1e9, 1)}), flush=True)

    def decode(url):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))

    def encode(image):
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    def predict(request):
        obs = dict(request["obs"])
        obs["vision"] = {camera: {"color": decode(item["color"])} for camera, item in obs["vision"].items()}
        conditions = model._conditions(model._encode_obs(obs))
        conditions["seed"] = int(request.get("seed", 42))
        if request.get("denoise_steps"):
            conditions["denoise_steps"] = int(request["denoise_steps"])
        with lock, torch.inference_mode():
            result = engine.generate(conditions)
        actions = result["actions"]
        actions = np.asarray(actions.detach().float().cpu() if hasattr(actions, "cpu") else actions)
        if actions.ndim == 2:
            actions = actions[None]
        actions = model._wam_policy._project_binary_dims(actions)
        native = model._eef20_chunk_to_native(actions[0])
        return {"frames": [encode(frame) for frame in (result.get("video") or [])],
                "actions": [{key: np.asarray(value).astype(float).tolist() for key, value in step.items()}
                            for step in native]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *items):
            pass

        def _send(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path.rstrip("/") != "/predict":
                self._send(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                began = time.perf_counter()
                body = predict(json.loads(self.rfile.read(size)))
                body["seconds"] = round(time.perf_counter() - began, 2)
                print(json.dumps({"frames": len(body["frames"]), "actions": len(body["actions"]),
                                  "seconds": body["seconds"]}), flush=True)
                self._send(200, body)
            except Exception as error:
                import traceback
                traceback.print_exc()
                self._send(400, {"error": f"{type(error).__name__}: {error}"})

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
