import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import draccus
import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from PIL import Image
from transformers import AutoTokenizer

from go1.internvl.model.go1 import GO1Model, GO1ModelConfig
from go1.internvl.train.constants import IMG_END_TOKEN
from go1.internvl.train.dataset import build_transform, dynamic_preprocess, preprocess_internvl2_5

json_numpy.patch()


def normalize(data, stats):
    return (data - stats["mean"]) / (stats["std"] + 1e-6)


def unnormalize(data, stats):
    return data * stats["std"] + stats["mean"]


def get_stats_tensor(stats_json):
    stats_tensor = {}
    for name in ["state", "action"]:
        stats_tensor[name] = {}
        for key in ["mean", "std"]:
            stats_tensor[name][key] = torch.from_numpy(np.array(stats_json[name][key]))
    return stats_tensor


def multi_image_get_item(
    raw_target: Dict[str, Any],
    img_transform,
    text_tokenizer,
    num_image_token,
    cam_keys: List[str] = [
        "cam_head_color",
        "cam_hand_right_color",
        "cam_hand_left_color",
    ],
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=6,
    image_size=448,
):
    images, num_tiles = [], []
    num_image = 0
    for cam_key in cam_keys:
        if cam_key in raw_target:
            num_image += 1
            if dynamic_image_size:
                image = dynamic_preprocess(
                    raw_target[cam_key],
                    min_num=min_dynamic_patch,
                    max_num=max_dynamic_patch,
                    image_size=image_size,
                    use_thumbnail=use_thumbnail,
                )
                images += image
                num_tiles.append(len(image))
            else:
                images.append(raw_target[cam_key])
                num_tiles.append(1)

    pixel_values = [img_transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    num_patches = pixel_values.size(0)

    # Preprocess the conversations and generate the return dictionary
    num_image_tokens = [num_image_token * num_tile for num_tile in num_tiles]
    ntp_target = raw_target.get("ntp_target", "")
    conversation = [
        {"from": "human", "value": f"{'<image>'*num_image}{raw_target['final_prompt']}"},
        {"from": "gpt", "value": ntp_target},
    ]
    ret = preprocess_internvl2_5(
        "internvl2_5",
        [conversation],
        text_tokenizer,
        num_image_tokens,
        num_image=num_image,
        group_by_length=True,
    )

    # Calculate position_ids for packed dataset
    position_ids = ret["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
    image_end_token_id = text_tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    assert (ret["input_ids"][0] == image_end_token_id).sum() == num_image, "image tokens are truncated"

    # Create the final return dictionary
    final_ret = dict(
        input_ids=ret["input_ids"][0],
        labels=ret["labels"][0],
        attention_mask=ret["attention_mask"][0],
        position_ids=position_ids[0],
        pixel_values=pixel_values,
        image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
    )
    return final_ret


class GO1Infer:
    def __init__(
        self,
        model_path: Union[str, Path],
        data_stats_path: Union[str, Path] = None,
    ) -> Path:
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.config = GO1ModelConfig.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=False,
        )
        self.image_size = self.config.force_image_size
        self.num_image_token: int = int(
            (self.image_size // self.config.vision_config.patch_size) ** 2 * (self.config.downsample_ratio**2)
        )
        self.dynamic_image_size = self.config.dynamic_image_size

        self.go1 = GO1Model.from_pretrained(model_path, config=self.config)
        self.go1.to(torch.bfloat16).to(self.device)

        self.img_transform = build_transform(
            is_train=False, input_size=self.image_size, pad2square=self.config.pad2square
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            model_path, add_eos_token=False, trust_remote_code=True, use_fast=False
        )

        self.norm = getattr(self.config, "norm", False)
        if self.norm:
            assert data_stats_path is not None, "data_stats_path must be provided when norm is True"
            with open(data_stats_path, "rb") as f:
                self.data_stats = get_stats_tensor(json.load(f))

    def predict_action(self, inputs: Dict[str, Any]) -> str:
        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = inputs["position_ids"]
        image_flags = inputs["image_flags"]
        ctrl_freqs = inputs["ctrl_freqs"]

        state = inputs["state"]
        # Normalize state if needed
        if self.norm:
            state = normalize(state, self.data_stats["state"])

        start_time = time.time()
        device = self.device
        with torch.no_grad():
            action = self.go1(
                pixel_values=pixel_values.to(dtype=torch.bfloat16, device=device),
                input_ids=input_ids.to(device).unsqueeze(0),
                attention_mask=attention_mask.to(device).unsqueeze(0),
                position_ids=position_ids.to(device).unsqueeze(0),
                image_flags=image_flags.to(device),
                state=state.to(dtype=torch.bfloat16, device=device).unsqueeze(0),
                ctrl_freqs=ctrl_freqs.to(dtype=torch.bfloat16, device=device).unsqueeze(0),
            )
        print(f"Model inference time: {(time.time() - start_time)*1000:.3f} ms")
        outputs = action[1][0].float().cpu()

        # Unnormalize action if needed
        if self.norm:
            outputs = unnormalize(outputs, self.data_stats["action"])

        outputs = outputs.numpy()

        return outputs

    def inference(self, payload: Dict[str, Any]):
        if "top" in payload:
            payload["cam_head_color"] = Image.fromarray(payload["top"])
        if "right" in payload:
            payload["cam_hand_right_color"] = Image.fromarray(payload["right"])
        if "left" in payload:
            payload["cam_hand_left_color"] = Image.fromarray(payload["left"])

        prompt = payload["instruction"]
        payload["final_prompt"] = f"What action should the robot take to {prompt}?"

        inputs = multi_image_get_item(
            raw_target=payload,
            img_transform=self.img_transform,
            text_tokenizer=self.text_tokenizer,
            num_image_token=self.num_image_token,
            dynamic_image_size=self.dynamic_image_size,
            use_thumbnail=self.config.use_thumbnail,
            min_dynamic_patch=self.config.min_dynamic_patch,
            max_dynamic_patch=self.config.max_dynamic_patch,
            image_size=self.image_size,
        )

        inputs["state"] = torch.from_numpy(payload["state"].copy())
        inputs["ctrl_freqs"] = torch.from_numpy(payload["ctrl_freqs"].copy())

        return self.predict_action(inputs)


class GO1Server:
    def __init__(
        self,
        model_path: Union[str, Path],
        data_stats_path: Union[str, Path] = None,
    ) -> Path:
        self.model = GO1Infer(
            model_path=model_path,
            data_stats_path=data_stats_path,
        )

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()

        @self.app.post("/act")
        async def act_endpoint(request: Request):
            payload = await request.json()
            actions = self.model.inference(payload)

            return actions.tolist() if hasattr(actions, "tolist") else actions

        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # Model Configuration
    model_path: Union[str, Path]
    data_stats_path: Union[str, Path] = None

    # Server Configuration
    host: str = "0.0.0.0"  # Host IP Address
    port: int = 9000  # Host Port


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = GO1Server(cfg.model_path, cfg.data_stats_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
