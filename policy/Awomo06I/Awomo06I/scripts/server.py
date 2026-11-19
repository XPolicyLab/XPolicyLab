# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve the frozen five-domain recipe with the RoboTwin joint contract."""
import io
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from transformers import AutoProcessor

if __package__:
    from . import backend
else:
    import backend
from nemo_automodel.components.datasets.diffusion.five_domain_prompt import (
    JointDomainPromptCodec,
    joint_domain_collate,
)
from nemo_automodel.components.datasets.diffusion.robotwin_frozen.dataset import ACTIVE, JOINTS, normalize, scatter
from nemo_automodel.components.datasets.diffusion.robotwin_frozen.stats_contract import load_approved_stats


class RobotTwinBackend(backend.AP394HistoryVLBackend):
    """Reuse strict weight loading and denoising, with the frozen RoboTwin adapter."""

    def __init__(self, root: Path, config: dict) -> None:
        self.runtime_root = Path(config["runtime_root"])
        self.model_config = Path(config["model_config"])
        self.checkpoint = Path(config["checkpoint"])
        self.output_root = root
        self.options = backend.AP394BackendOptions(weights="ema", subtask_decode_every=0)
        self.weights = "ema"
        self._validate_checkpoint_files()
        contract = json.loads((self.checkpoint / "dataset_statistics/five_domain/adapter_contract.json").read_text())
        assert contract["schema_version"] == 21 and contract["history_slots"]["robotwin"] == 0
        assert contract["joint_reference"]["robotwin"] == "relative"
        self.stats, _ = load_approved_stats(self.checkpoint / "dataset_statistics/five_domain/robotwin/0_normalization.json")
        self.active_action_mask = np.zeros(80, np.bool_)
        self.active_action_mask[list(ACTIVE)] = True
        self.lora_meta = json.loads((self.checkpoint / "vl_lora/meta.json").read_text())
        self.recipe, self.cfg = self._build_recipe(self.runtime_root / "Automodel")
        self._validate_model_contract()
        self.weight_info = self._load_weights()
        assert float(self.cfg.action_patch.scheduler.shift) == 5
        self.codec = JointDomainPromptCodec(
            AutoProcessor.from_pretrained(str(self.cfg.action_patch.encoders.text_encoder_path)), "robotwin", False
        )
        (root / "weight_loading.json").write_text(json.dumps(self.weight_info, indent=2))

    @torch.inference_mode()
    def predict(self, data: dict) -> np.ndarray:
        """Match training resize, prompt, state normalization and joint decoding."""
        raw_state = np.asarray(data["state"], np.float32)
        assert raw_state.shape == (14,) and np.isfinite(raw_state).all()
        views = []
        for name in ("head", "left", "right"):
            rgb = np.asarray(data[name])
            assert rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8
            views.append(F.resize(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255,
                                  [240, 320], antialias=True))
        top = F.resize(views[0], [192, 256], antialias=True)
        bottom = torch.cat([F.resize(v, [96, 128], antialias=True) for v in views[1:]], dim=-1)
        frame = torch.cat([top, bottom], dim=-2).permute(1, 2, 0).contiguous().numpy()
        task = "A video recorded from a robot's point of view executing the following instruction: " + str(data["instruction"].item())
        sample = dict(
            video=np.stack([frame, frame]),
            state=normalize(scatter(raw_state), self.stats["state"]["default"])[None],
            action=np.zeros((16, 80), np.float32),
            action_mask=np.broadcast_to(self.active_action_mask, (16, 80)).copy(),
            history=np.empty((0, 288, 256, 3), np.float32),
            history_valid=np.empty(0, np.bool_),
            has_real_action=True, has_subtask=False, domain="robotwin",
            vl=self.codec.encode(task, frame, []),
        )
        batch = joint_domain_collate([sample])
        denoised, _, _, timings = self._predict(batch, 42, False)
        assert denoised.shape == (16, 80) and np.isfinite(denoised).all()
        # 训练的常量维仍使用单位尺度，解码与原生RoboTwin政策一致。
        low, high = [np.asarray(self.stats["action"]["default"]["global_" + k], np.float32)
                     for k in ("q01", "q99")]
        span = high - low
        constant = span < 1e-4
        span[constant] = 2
        scale = 2 / span
        offset = -1 - scale * low
        offset[constant] = -low[constant]
        action = (denoised - offset) / scale
        action[:, list(JOINTS)] += scatter(raw_state)[list(JOINTS)]
        output = action[:, list(ACTIVE)].astype(np.float32)
        # 训练侧 convert() 对 raw action 夹爪做了 1-raw 反转（closed_is_one_flipped），
        # 解码回 raw 14 维后反转回来，恢复模拟器期望的 0=closed/1=open 语义。
        output[:, [6, 13]] = 1.0 - output[:, [6, 13]]
        assert output.shape == (16, 14) and np.isfinite(output).all()
        witness = dict(domain="robotwin", prediction_shape=list(output.shape), history_slots=0,
                       video_shape=list(sample["video"].shape), weights=self.weight_info,
                       checkpoint=str(self.checkpoint), timings=timings)
        with (self.output_root / "inference.jsonl").open("a") as f:
            f.write(json.dumps(witness) + "\n")
        return output


def main() -> None:
    """Load one model and serve one serialized worker stream."""
    root = Path(sys.argv[1])
    worker = int(sys.argv[2])
    out = root / "servers" / str(worker)
    out.mkdir(parents=True, exist_ok=True)
    model = RobotTwinBackend(out, json.loads((root / "config.json").read_text()))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ready")

        def do_POST(self) -> None:
            try:
                payload = self.rfile.read(int(self.headers["Content-Length"]))
                with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                    output = model.predict(data)
                buffer = io.BytesIO()
                np.save(buffer, output, allow_pickle=False)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(buffer.getvalue())
            except Exception:
                traceback.print_exc()
                self.send_response(500)
                self.end_headers()
                self.wfile.write(traceback.format_exc().encode())

    print("ROBOTWIN_FIVE_RAW_SERVER_READY", worker, flush=True)
    HTTPServer((os.environ.get("AWOMO_BIND_HOST", "127.0.0.1"), 18700 + worker), Handler).serve_forever()


if __name__ == "__main__":
    main()
