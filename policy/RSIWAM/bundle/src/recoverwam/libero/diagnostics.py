from pathlib import Path
from typing import Sequence

import torch

from recoverwam.utils.video_io import save_mp4


class VideoComparisonEvaluator:
    """Render fixed examples with identical seeds at multiple denoising depths."""

    def __init__(
        self,
        dataset,
        output_dir,
        fixed_indices: Sequence[int],
        inference_steps: Sequence[int] = (1, 10),
        seed: int = 42,
        fps: int = 8,
    ):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.fixed_indices = tuple((int(index) for index in fixed_indices))
        self.inference_steps = tuple((int(steps) for steps in inference_steps))
        self.seed = int(seed)
        self.fps = int(fps)
        if not self.fixed_indices:
            raise ValueError("`fixed_indices` must contain at least one sample index.")
        dataset_size = len(self.dataset)
        invalid_indices = [
            index for index in self.fixed_indices if index < 0 or index >= dataset_size
        ]
        if invalid_indices:
            raise ValueError(
                f"Fixed evaluation indices out of range for dataset size {dataset_size}: {invalid_indices}"
            )
        if not self.inference_steps or any((steps <= 0 for steps in self.inference_steps)):
            raise ValueError(
                f"`inference_steps` must contain positive integers, got {self.inference_steps}"
            )
        if self.fps <= 0:
            raise ValueError(f"`fps` must be positive, got {self.fps}")

    @staticmethod
    def _prepare_sample(sample):
        video = sample["video"]
        if not isinstance(video, torch.Tensor) or video.ndim != 4:
            raise ValueError(
                f"Fixed video evaluation expects sample['video'] with shape [C,T,H,W], got {type(video)} {getattr(video, 'shape', None)}"
            )
        if video.shape[0] != 3 or video.shape[1] <= 1:
            raise ValueError(
                f"Fixed video evaluation requires [3,T>1,H,W], got {tuple(video.shape)}"
            )
        infer_kwargs = {
            "input_image": video[:, 0].unsqueeze(0),
            "num_frames": int(video.shape[1]),
            "action": sample.get("action"),
            "action_horizon": int(sample["action"].shape[0])
            if isinstance(sample.get("action"), torch.Tensor)
            else None,
            "proprio": sample["proprio"][0]
            if isinstance(sample.get("proprio"), torch.Tensor) and sample["proprio"].ndim == 2
            else sample.get("proprio"),
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "tiled": False,
        }
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("context and context_mask must be provided together.")
            infer_kwargs.update({"prompt": None, "context": context, "context_mask": context_mask})
        else:
            infer_kwargs["prompt"] = sample["prompt"]
        if "condition_video" in sample:
            infer_kwargs["condition_video"] = sample["condition_video"]
        if "condition_observation_video" in sample:
            infer_kwargs["condition_observation_video"] = sample["condition_observation_video"]
        return infer_kwargs

    def evaluate(self, model, global_step: int):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, str] = {}
        for sample_index in self.fixed_indices:
            sample = self.dataset[sample_index]
            base_kwargs = self._prepare_sample(sample)
            for inference_steps in self.inference_steps:
                output = model.infer(
                    **base_kwargs, num_inference_steps=inference_steps, seed=self.seed
                )
                frames = output["video"]
                output_path = (
                    self.output_dir
                    / f"step_{int(global_step):06d}_sample_{sample_index:06d}_{inference_steps}step.mp4"
                )
                save_mp4(frames, str(output_path), fps=self.fps)
                key = f"video/sample_{sample_index:06d}/{inference_steps}_step_path"
                metrics[key] = str(output_path)
                if len(self.fixed_indices) == 1 and inference_steps == 1:
                    metrics["video/one_step_path"] = str(output_path)
                if len(self.fixed_indices) == 1 and inference_steps == 10:
                    metrics["video/ten_step_path"] = str(output_path)
        return metrics


class ActionDiagnosticEvaluator:
    """Measure fixed offline action loss and padded-token-masked action L1."""

    def __init__(
        self, dataset, fixed_indices: Sequence[int], num_inference_steps: int = 10, seed: int = 42
    ):
        self.dataset = dataset
        self.fixed_indices = tuple((int(index) for index in fixed_indices))
        self.num_inference_steps = int(num_inference_steps)
        self.seed = int(seed)
        if not self.fixed_indices:
            raise ValueError("`fixed_indices` must contain at least one index.")
        invalid = [index for index in self.fixed_indices if index < 0 or index >= len(self.dataset)]
        if invalid:
            raise ValueError(f"Action diagnostic indices out of range: {invalid}")
        if self.num_inference_steps <= 0:
            raise ValueError(
                f"`num_inference_steps` must be positive, got {self.num_inference_steps}"
            )

    @staticmethod
    def _batch_sample(sample):
        batched = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                batched[key] = value.unsqueeze(0)
            elif isinstance(value, str):
                batched[key] = [value]
            else:
                batched[key] = value
        return batched

    def _denormalize_action(self, action, proprio):
        processor_owner = getattr(self.dataset, "lerobot_dataset", self.dataset)
        processor = getattr(processor_owner, "processor", None)
        if processor is None:
            return action.detach().to(device="cpu", dtype=torch.float32)
        action_btd = action.detach().to(device="cpu", dtype=torch.float32).unsqueeze(0)
        proprio_btd = proprio.detach().to(device="cpu", dtype=torch.float32)
        if proprio_btd.ndim == 2:
            proprio_btd = proprio_btd.unsqueeze(0)
        batch = {"action": action_btd, "state": proprio_btd}
        batch = processor.action_state_merger.backward(batch)
        batch = processor.normalizer.backward(batch)
        action_meta = processor.shape_meta["action"]
        state_meta = processor.shape_meta["state"]
        merged_batch = {
            "action": {
                meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta
            },
            "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
        }
        merged_batch = processor.action_state_merger.forward(merged_batch)
        return merged_batch["action"].detach().to(device="cpu", dtype=torch.float32)

    def evaluate(self, model, global_step: int):
        del global_step
        losses = []
        l1_values = []
        for sample_index in self.fixed_indices:
            sample = self.dataset[sample_index]
            model_device = torch.device(getattr(model, "device", "cpu"))
            rng_devices = [model_device] if model_device.type == "cuda" else []
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(self.seed)
                if model_device.type == "cuda":
                    torch.cuda.manual_seed(self.seed)
                loss, _ = model.training_loss(self._batch_sample(sample))
            losses.append(float(loss.detach().float().item()))
            video = sample["video"]
            gt_action = sample["action"]
            proprio = sample["proprio"]
            context = sample.get("context")
            context_mask = sample.get("context_mask")
            infer_kwargs = {
                "prompt": None if context is not None else sample["prompt"],
                "input_image": video[:, 0].unsqueeze(0),
                "action_horizon": int(gt_action.shape[0]),
                "proprio": proprio[0],
                "context": context,
                "context_mask": context_mask,
                "num_video_frames": int(video.shape[1]),
                "num_inference_steps": self.num_inference_steps,
                "seed": self.seed,
                "tiled": False,
            }
            if "condition_video" in sample:
                infer_kwargs["condition_video"] = sample["condition_video"]
            if "condition_observation_video" in sample:
                infer_kwargs["condition_observation_video"] = sample["condition_observation_video"]
            pred_action = model.infer_action(**infer_kwargs)["action"]
            pred_action = self._denormalize_action(pred_action, proprio)
            gt_action = self._denormalize_action(gt_action, proprio)
            action_is_pad = sample.get("action_is_pad")
            if action_is_pad is None:
                valid = torch.ones(gt_action.shape[0], dtype=torch.bool)
            else:
                valid = ~action_is_pad.detach().cpu().bool()
            if not bool(valid.any().item()):
                raise ValueError(f"All action tokens are padded for fixed sample {sample_index}.")
            l1_values.append(float((pred_action[valid] - gt_action[valid]).abs().mean().item()))
        return {
            "action/val_loss": sum(losses) / len(losses),
            "action/l1": sum(l1_values) / len(l1_values),
        }
