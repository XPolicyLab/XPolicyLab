# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import copy
import glob
import json
import os
import random
import re
from functools import lru_cache

import numpy as np
import torch
import torchvision.transforms.functional as vision_f
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation
from transformers.utils import logging
from tqdm import tqdm

from mibot.utils.io import (
    build_action_mask,
    compose_action,
    compose_state,
    get_value,
    normalize_action,
    resize_image,
    rotm2aa_batch,
    validate_stats,
)

logger = logging.get_logger(__name__)
PROMPT_RE = re.compile(r"(<image>|<video>)")


class JsonDataset(Dataset):
    def __init__(self, params):
        data = params["train_datasets"]
        self.action_length = int(data.get("action_length", params.get("action_length", 30)))
        self.batch_size = int(data.get("batch_size", 16))
        self.max_samples = int(params.get("max_steps", 1000)) * self.batch_size * int(os.environ.get("WORLD_SIZE", 1))
        self.mean, self.std = validate_stats(data["mean"], data["std"], self.action_length)
        self.files = self._json_files(data.get("train_path", []))
        self.samples = self._samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        traj = self._read_json(sample["file"])
        frame = sample["frame_index"]
        steps = min(self.action_length, int(traj["num_frames"]) - frame)
        if steps <= 0:
            raise IndexError(f"invalid sample at index={index}: frame_index={frame}")

        prompt = self._prompt(traj)
        left = self._arm_action(traj, "left", frame, steps)
        right = self._arm_action(traj, "right", frame, steps)
        action = compose_action(*left, *right, action_length=self.action_length)

        return {
            "messages": self._messages(prompt["conversations"], self._augment(self._images(traj, prompt["images"], frame))),
            "action": torch.from_numpy(normalize_action(action, self.mean, self.std)),
            "action_mask": torch.from_numpy(build_action_mask(self.action_length, self._mask(traj, steps))),
            "state": torch.from_numpy(self._state(traj, frame)),
        }

    @staticmethod
    @lru_cache(maxsize=32)
    def _read_json(path):
        with open(path, "r") as file:
            return json.load(file)

    @staticmethod
    def _json_files(paths):
        files = []
        for path in paths:
            if os.path.isfile(path) and path.endswith(".json"):
                files.append(path)
            elif os.path.isdir(path):
                files.extend(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
        return sorted(files)

    def _samples(self):
        samples = []
        logger.info(f"train set has {len(self.files)} files.")
        for path in tqdm(self.files):
            count = int(self._read_json(path)["num_frames"]) - self.action_length + 1
            if count <= 0:
                logger.warning(f"File {path} has fewer than {self.action_length} frames, skipping.")
                continue
            samples.extend({"file": path, "frame_index": i} for i in range(count))

        logger.info(f"Raw train samples: {len(samples)}")
        if samples and len(samples) < self.max_samples:
            q, r = divmod(self.max_samples, len(samples))
            samples = samples * q + samples[:r]
            logger.info(f"Repeated to {self.max_samples} samples for training")
        logger.info(f"Total train samples: {len(samples)}")
        return samples

    @staticmethod
    def _prompt(traj):
        prompts = traj.get("instruction", {}).get("general") or []
        if not prompts:
            raise ValueError(f"trajectory {traj.get('time', '<unknown>')} missing instruction.general")
        prompt = copy.deepcopy(random.choice(prompts))
        conversations = prompt.get("conversations", [])
        if len(conversations) >= 2:
            conversations[0]["value"] += " /no_cot"
            conversations[1]["value"] = "<cot></cot>"
        return prompt

    @staticmethod
    def _images(traj, keys, frame):
        images = []
        frame = [max(frame, 0)]
        for key in keys:
            infos = get_value(traj, key)
            if infos is None:
                raise ValueError(f"{key} is missing in trajectory {traj.get('time', '<unknown>')}")
            for info in infos:
                video = VideoReader(info["path"], num_threads=2)
                video.seek(0)
                images.extend(Image.fromarray(image) for image in video.get_batch(frame).asnumpy())
        return images

    def _augment(self, images):
        ops = (
            (adjust_brightness, 1.0 + random.uniform(-32.0 / 255.0, 32.0 / 255.0)),
            (adjust_contrast, random.uniform(0.5, 1.5)),
            (adjust_saturation, random.uniform(0.5, 1.5)),
            (adjust_hue, 0.0),
        )
        flags = [random.randint(0, 1) == 0 for _ in ops]

        out = []
        for image in images:
            image = self._crop(resize_image(image, factor=32, max_pixels=90000))
            for use, (op, value) in zip(flags, ops):
                if use:
                    image = op(image, value)
            out.append(image)
        return out

    @staticmethod
    def _crop(image, ratio=0.95):
        height, width = image.size[1], image.size[0]
        crop_h, crop_w = int(height * ratio), int(width * ratio)
        top = int(np.random.randint(0, height - crop_h + 1))
        left = int(np.random.randint(0, width - crop_w + 1))
        return vision_f.resize(vision_f.crop(image, top, left, crop_h, crop_w), size=(height, width))

    @staticmethod
    def _messages(conversations, images):
        pool = [{"type": "image", "image": image} for image in images]
        messages = []

        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            if role == "assistant":
                messages.append({"role": role, "content": [{"type": "text", "text": turn["value"]}]})
                continue

            content = []
            for part in PROMPT_RE.split(turn["value"]):
                if part == "<image>":
                    if not pool:
                        raise ValueError("number of <image> placeholders exceeds the number of provided images")
                    content.append(pool.pop(0))
                elif part == "<video>":
                    raise ValueError("video placeholders are not supported in the JSON dataset")
                elif part:
                    content.append({"type": "text", "text": part})
            messages.append({"role": role, "content": content})

        if pool:
            raise ValueError(f"{len(pool)} image(s) remain unused")
        return messages

    def _mask(self, traj, steps):
        kind = traj.get("trajectory_type", "success")
        if kind == "invalid":
            return np.ones(self.action_length, dtype=np.int32)
        if kind not in {"success", "ongoing"}:
            raise ValueError(f"Unsupported trajectory_type: {kind}")
        mask = np.zeros(self.action_length, dtype=np.int32)
        mask[:steps] = 1
        return mask

    def _arm_action(self, traj, arm, frame, steps):
        rotm = self._frame(traj, f"proprios.{arm}_ee_rotm", frame).reshape(3, 3)
        pos = self._frame(traj, f"proprios.{arm}_ee_pos", frame)
        target_pos = self._future(traj, f"actions.{arm}_ee_pos", frame, steps)
        target_rotm = self._future(traj, f"actions.{arm}_ee_rotm", frame, steps).reshape(-1, 3, 3)
        return (
            self._pad((rotm.T @ (target_pos - pos).T).T, steps),
            self._pad(rotm2aa_batch(rotm.T @ target_rotm), steps),
            self._delta(traj, f"proprios.{arm}_gripper_pos", f"actions.{arm}_gripper_pos", frame, steps),
            self._delta(traj, f"proprios.{arm}_arm_joint", f"actions.{arm}_arm_joint", frame, steps),
        )

    def _delta(self, traj, current_key, target_key, frame, steps):
        return self._pad(self._future(traj, target_key, frame, steps) - self._frame(traj, current_key, frame), steps)

    @staticmethod
    def _frame(traj, key, frame):
        return np.asarray(get_value(traj, key)[frame], dtype=np.float32)

    @staticmethod
    def _future(traj, key, frame, steps):
        return np.asarray(get_value(traj, key)[frame : frame + steps], dtype=np.float32)

    def _pad(self, value, steps):
        value = np.asarray(value, dtype=np.float32)
        if steps == self.action_length:
            return value
        return np.concatenate([value, np.repeat(value[-1:], self.action_length - steps, axis=0)], axis=0)

    @staticmethod
    def _state(traj, frame):
        return compose_state(
            left_gripper=np.asarray(get_value(traj, "proprios.left_gripper_pos")[frame], dtype=np.float32),
            left_joint=np.asarray(get_value(traj, "proprios.left_arm_joint")[frame], dtype=np.float32),
            right_gripper=np.asarray(get_value(traj, "proprios.right_gripper_pos")[frame], dtype=np.float32),
            right_joint=np.asarray(get_value(traj, "proprios.right_arm_joint")[frame], dtype=np.float32),
        )
