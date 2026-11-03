import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .utils.video_io import save_mp4


class _ActionMetrics:
    """Measure fixed offline action loss and padded-token-masked action L1."""

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
        processor_owner = getattr(
            self.dataset,
            "lerobot_dataset",
            self.dataset,
        )
        processor = getattr(processor_owner, "processor", None)
        if processor is None:
            return action.detach().to(device="cpu", dtype=torch.float32)

        action_btd = (
            action.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .unsqueeze(0)
        )
        proprio_btd = proprio.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        if proprio_btd.ndim == 2:
            proprio_btd = proprio_btd.unsqueeze(0)
        batch = {
            "action": action_btd,
            "state": proprio_btd,
        }
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
        return (
            merged_batch["action"]
            .detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
        )


class HeldoutActionEvaluator:
    """Open-loop action error on held-out episodes, sharded across all ranks.

    Every held-out trajectory contributes ``windows_per_episode`` fixed windows at
    evenly spaced steps. For each window it records the one-step training losses
    (fixed seed) and the action chunk produced by ``infer_action`` with
    ``num_inference_steps`` denoising steps, compared with the ground truth on the
    non-padded tokens in normalized space (MSE, L1) and after denormalization (L1).

    ``num_inference_steps`` may be a list and ``samplers`` a list of joint-denoising
    samplers; every (sampler, steps) variant then runs on the same windows with the
    same seeds.  The first variant keeps the legacy metric names (``heldout/mse``,
    ...); every variant is also reported as ``heldout/<tag>/<metric>``.

    Unlike the other evaluators it runs on every rank: ``evaluate_local`` returns
    per-task sums for this rank's share of the windows, the trainer sum-reduces them,
    and ``summarize`` turns the global sums into metrics.
    """

    distributed = True
    ACTION_FIELDS = ("mse", "l1", "l1_denorm")
    JOINT_DENOISING_SAMPLERS = ("euler", "unipc")

    def __init__(
        self,
        dataset,
        windows_per_episode: int = 16,
        num_inference_steps=10,
        seed: int = 42,
        task_column: str = "task_name",
        output_path=None,
        samplers=None,
    ):
        self.dataset = dataset
        steps_list = (
            [num_inference_steps]
            if isinstance(num_inference_steps, (int, np.integer))
            else list(num_inference_steps)
        )
        steps_list = [int(steps) for steps in steps_list]
        if not steps_list or any(steps <= 0 for steps in steps_list):
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        sampler_list = [None] if not samplers else [str(x).strip().lower() for x in samplers]
        unknown = [
            x for x in sampler_list if x is not None and x not in self.JOINT_DENOISING_SAMPLERS
        ]
        if unknown:
            raise ValueError(f"unknown joint-denoising samplers {unknown}")
        # (tag, sampler or None, steps); variant 0 feeds the legacy metric names.
        self.variants = []
        for sampler in sampler_list:
            for steps in steps_list:
                tag = f"steps{steps}" if sampler is None else f"{sampler}_steps{steps}"
                self.variants.append((tag, sampler, steps))
        if len({tag for tag, _, _ in self.variants}) != len(self.variants):
            raise ValueError(f"duplicate held-out variants: {[tag for tag, _, _ in self.variants]}")
        self.num_inference_steps = self.variants[0][2]
        fields = ["count", "mse", "l1", "l1_denorm", "loss_action", "loss_video", "errors"]
        for tag, _, _ in self.variants[1:]:
            fields.extend(f"{name}@{tag}" for name in self.ACTION_FIELDS)
        self.FIELDS = tuple(fields)
        self.MEAN_FIELDS = tuple(name for name in self.FIELDS if name not in ("count", "errors"))
        self.seed = int(seed)
        self.output_path = None if output_path is None else Path(output_path)
        windows = int(windows_per_episode)
        if windows <= 0:
            raise ValueError(f"`windows_per_episode` must be positive, got {windows_per_episode}")
        if (
            getattr(dataset, "is_training_set", False)
            or getattr(dataset, "balance_values", None)
            or getattr(dataset, "is_multi_source", False)
            or getattr(dataset, "segment_sampling_mode", "step_uniform") != "step_uniform"
        ):
            raise ValueError(
                "held-out evaluation needs a non-training, unbalanced, single-source, "
                "step_uniform dataset"
            )
        metadata = dataset.metadata
        self.tasks = sorted(str(t) for t in metadata[task_column].unique())
        task_ids = {task: i for i, task in enumerate(self.tasks)}
        # (dataset index, task id, trajectory, local step). Without balancing, a
        # single-source dataset index is the global step over trajectories in metadata
        # order, so these windows are fixed forever. (trajectory_start_steps is the
        # in-episode trim offset, not this prefix.) evaluate_local re-checks the
        # resolved trajectory/step of every window.
        # Windows start no later than length - action_horizon, so every chunk is fully
        # inside the episode instead of the tail windows being mostly padding.
        lengths = np.asarray(dataset.trajectory_lengths, dtype=np.int64)
        prefix = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        horizon = int(getattr(dataset, "action_horizon", 1))
        self.items = []
        for traj in range(int(dataset.num_trajectories)):
            length = int(lengths[traj])
            start = int(prefix[traj])
            last = max(0, length - horizon)
            locals_ = sorted({int(round(x)) for x in np.linspace(0, last, windows)})
            task = task_ids[str(metadata.iloc[traj][task_column])]
            self.items.extend((start + s, task, traj, s) for s in locals_)
        if not self.items:
            raise ValueError("held-out dataset has no trajectories")

    def evaluate_local(self, model, rank: int, world_size: int) -> torch.Tensor:
        if any(sampler is not None for _, sampler, _ in self.variants) and not bool(
            getattr(model, "joint_denoising", False)
        ):
            # Identical on every rank, so raising here cannot desynchronize the reduce.
            raise ValueError("held-out `samplers` require a joint-denoising model")
        sums = torch.zeros(len(self.tasks), len(self.FIELDS), dtype=torch.float64)
        model_device = torch.device(getattr(model, "device", "cpu"))
        rng_devices = [model_device] if model_device.type == "cuda" else []
        # A failing window is counted as an error instead of raising, so every rank
        # still reaches the trainer's reduce (a one-rank exception would hang the rest).
        for item_index in range(int(rank), len(self.items), int(world_size)):
            task = self.items[item_index][1]
            try:
                row = self._evaluate_window(model, item_index, model_device, rng_devices)
            except Exception as err:  # noqa: BLE001 - reported after the reduce
                print(f"[heldout] window {self.items[item_index]} failed: {err!r}", flush=True)
                sums[task, self.FIELDS.index("errors")] += 1.0
                continue
            if row is not None:
                for name, value in row.items():
                    sums[task, self.FIELDS.index(name)] += float(value)
        return sums

    def _evaluate_window(self, model, item_index, model_device, rng_devices):
        index, task, traj, step = self.items[item_index]
        # _get skips the dataset's "return a random sample on error" fallback.
        getter = getattr(self.dataset, "_get", None) or self.dataset.__getitem__
        sample = getter(index)
        if int(sample["trajectory_idx"]) != traj or int(sample["step_idx"]) != step:
            raise RuntimeError(
                f"held-out window {index} resolved to trajectory/step "
                f"{int(sample['trajectory_idx'])}/{int(sample['step_idx'])}, expected {traj}/{step}"
            )
        item_seed = self.seed + item_index
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(item_seed)
            if model_device.type == "cuda":
                torch.cuda.manual_seed(item_seed)
            _, loss_dict = model.training_loss(_ActionMetrics._batch_sample(sample))

        video = sample["video"]
        gt_action = sample["action"]
        proprio = sample["proprio"]
        context = sample.get("context")
        num_video_frames = sample.get("video_num_frames", video.shape[1])
        infer_kwargs = {
            "prompt": None if context is not None else sample["prompt"],
            "input_image": video[:, 0].unsqueeze(0),
            "action_horizon": int(gt_action.shape[0]),
            "proprio": proprio[0],
            "context": context,
            "context_mask": sample.get("context_mask"),
            "num_video_frames": int(num_video_frames),
            "num_inference_steps": self.num_inference_steps,
            "seed": item_seed,
            "tiled": False,
        }
        if "condition_video" in sample:
            infer_kwargs["condition_video"] = sample["condition_video"]
        if "condition_observation_video" in sample:
            infer_kwargs["condition_observation_video"] = sample["condition_observation_video"]
        gt = gt_action.detach().to(device="cpu", dtype=torch.float32)
        action_is_pad = sample.get("action_is_pad")
        valid = (
            torch.ones(gt.shape[0], dtype=torch.bool)
            if action_is_pad is None
            else ~action_is_pad.detach().cpu().bool()
        )
        if not bool(valid.any()):
            return None
        denorm_gt = _ActionMetrics._denormalize_action(self, gt, proprio)
        row = {
            "count": 1.0,
            "loss_action": float(loss_dict.get("loss_action", float("nan"))),
            "loss_video": float(loss_dict.get("loss_video", float("nan"))),
        }
        for variant_index, (tag, sampler, steps) in enumerate(self.variants):
            previous_sampler = getattr(model, "joint_denoising_sampler", None)
            if sampler is not None:
                model.joint_denoising_sampler = sampler
            try:
                pred = model.infer_action(**{**infer_kwargs, "num_inference_steps": steps})[
                    "action"
                ]
            finally:
                if sampler is not None:
                    model.joint_denoising_sampler = previous_sampler
            pred = pred.detach().to(device="cpu", dtype=torch.float32).reshape(gt_action.shape)
            diff = pred[valid] - gt[valid]
            denorm = _ActionMetrics._denormalize_action(self, pred, proprio)
            suffix = "" if variant_index == 0 else f"@{tag}"
            row[f"mse{suffix}"] = float(diff.pow(2).mean())
            row[f"l1{suffix}"] = float(diff.abs().mean())
            row[f"l1_denorm{suffix}"] = float((denorm[valid] - denorm_gt[valid]).abs().mean())
        return row

    def summarize(self, sums: torch.Tensor, global_step: int) -> dict:
        sums = sums.detach().to(device="cpu", dtype=torch.float64)
        col = {name: j for j, name in enumerate(self.FIELDS)}
        count = sums[:, col["count"]]
        errors = sums[:, col["errors"]]
        if float(count.sum()) <= 0:
            raise RuntimeError(
                f"held-out evaluation produced no valid windows ({int(errors.sum())} errors)"
            )
        per_task = {}
        for i, task in enumerate(self.tasks):
            if count[i] > 0:
                per_task[task] = {
                    name: float(sums[i, col[name]] / count[i]) for name in self.MEAN_FIELDS
                }
                per_task[task]["count"] = int(count[i])
            if errors[i] > 0:
                per_task.setdefault(task, {})["errors"] = int(errors[i])
        # Window-weighted means; also the task-balanced open-loop MSE, since episodes
        # (and therefore windows) differ little but tasks weigh equally in evaluation.
        metrics = {
            f"heldout/{name}": float(sums[:, col[name]].sum() / count.sum())
            for name in self.MEAN_FIELDS
        }
        metrics["heldout/mse_task_mean"] = float(
            np.mean([v["mse"] for v in per_task.values() if "mse" in v])
        )
        if len(self.variants) > 1:
            for variant_index, (tag, _, _) in enumerate(self.variants):
                suffix = "" if variant_index == 0 else f"@{tag}"
                for name in self.ACTION_FIELDS:
                    metrics[f"heldout/{tag}/{name}"] = float(
                        sums[:, col[name + suffix]].sum() / count.sum()
                    )
                metrics[f"heldout/{tag}/mse_task_mean"] = float(
                    np.mean([v["mse" + suffix] for v in per_task.values() if "mse" + suffix in v])
                )
        metrics["heldout/windows"] = float(count.sum())
        metrics["heldout/errors"] = float(errors.sum())
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": int(global_step),
                            "num_inference_steps": self.num_inference_steps,
                            "variants": [
                                {"tag": tag, "sampler": sampler, "num_inference_steps": steps}
                                for tag, sampler, steps in self.variants
                            ],
                            **metrics,
                            "per_task": per_task,
                        }
                    )
                    + "\n"
                )
        return metrics
