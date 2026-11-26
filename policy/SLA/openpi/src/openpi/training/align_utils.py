from __future__ import annotations

from collections.abc import Iterable
import concurrent.futures
import dataclasses
import math
import os
import time
from typing import Any, Literal

import jax
import numpy as np
from safetensors.torch import load_file
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models import model as _model
from vggt.heads.utils import custom_pooling
from vggt.utils.load_fn import preprocess_images_from_openpi

AlignTargetModel = Literal["auto", "vggt", "omnivggt"]


@dataclasses.dataclass
class AlignFeatures:
    targets: dict[int, torch.Tensor]
    mask: dict[int, torch.Tensor]
    compute_s: float = 0.0
    compute_started_s: float | None = None


@dataclasses.dataclass(frozen=True)
class AlignDeviceLayout:
    pi_device_ids: list[int]
    vggt_device_ids: list[int]


def infer_target_model(config: Any) -> Literal["vggt", "omnivggt"]:
    target = config.align_target_model
    weight_path = config.vggt_weight_path
    if target in ("vggt", "omnivggt"):
        return target
    elif "omnivggt" in weight_path.lower() and target == "auto":
        return "omnivggt"
    elif "vggt" in weight_path.lower() and target == "auto":
        return "vggt"
    else:
        raise ValueError(f"Invalid align_target_model {target} or vggt_weight_path {weight_path}")


def resolve_align_devices(config: Any) -> AlignDeviceLayout:
    count = torch.cuda.device_count()
    vggt_devices = config.align_vggt_devices if count > 1 else 1
    vggt_devices = max(1, min(vggt_devices, count))
    mode = config.align_device_mode
    if mode == "shared" or count == 1:
        return AlignDeviceLayout(
            pi_device_ids=list(range(count)),
            vggt_device_ids=list(range(count - vggt_devices, count)),
        )
    elif mode == "reserved":
        return AlignDeviceLayout(
            pi_device_ids=list(range(count - vggt_devices)),
            vggt_device_ids=list(range(count - vggt_devices, count)),
        )
    else:
        raise ValueError(f"Invalid align_device_mode {mode}. Must be one of ['shared', 'reserved'].")


def _split_batch_ranges(batch_size: int, shard_count: int) -> list[tuple[int, int]]:
    boundaries = np.linspace(0, batch_size, shard_count + 1, dtype=int)
    return [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(shard_count)]


def _balanced_chunk_ranges(batch_size: int, max_chunk_size: int | None) -> list[tuple[int, int]]:
    if batch_size <= 0:
        return []
    if max_chunk_size is None or max_chunk_size >= batch_size:
        return [(0, batch_size)]
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive when set.")
    return _split_batch_ranges(batch_size, math.ceil(batch_size / max_chunk_size))


def _assemble_feature_chunks(
    batch_size: int,
    chunks: Iterable[tuple[int, int, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    iterator = iter(chunks)
    try:
        first_start, first_stop, first_target, first_mask = next(iterator)
    except StopIteration as error:
        raise ValueError("At least one feature chunk is required.") from error

    targets = first_target.new_empty((batch_size, *first_target.shape[1:]))
    masks = first_mask.new_empty((batch_size, *first_mask.shape[1:]))
    targets[first_start:first_stop].copy_(first_target)
    masks[first_start:first_stop].copy_(first_mask)
    del first_target, first_mask

    expected_start = first_stop
    for start, stop, target, mask in iterator:
        if start != expected_start:
            raise ValueError(f"Feature chunks must be contiguous; expected start {expected_start}, got {start}.")
        targets[start:stop].copy_(target)
        masks[start:stop].copy_(mask)
        expected_start = stop
        del target, mask
    if first_start != 0 or expected_start != batch_size:
        raise ValueError(f"Feature chunks cover [{first_start}, {expected_start}), expected [0, {batch_size}).")
    return targets, masks


class _TorchAlignFeatureShard:
    def __init__(self, config: Any, *, device_id: int, state_dict: dict[str, torch.Tensor]):
        self.config = config
        self.target_model = infer_target_model(config)
        self.device = torch.device(f"cuda:{device_id}")

        self.model = self._load_model(state_dict).to(self.device)
        self.model.eval()
        self.stream = torch.cuda.Stream(device=self.device)

    def _load_model(self, state_dict: dict[str, torch.Tensor]):
        if self.target_model == "omnivggt":
            from omnivggt.models.omnivggt import OmniVGGT

            model = OmniVGGT(
                enable_camera=False,
                enable_point=False,
                enable_depth=False,
            )
        else:
            from vggt.models.vggt import VGGT

            model = VGGT(
                enable_camera=False,
                enable_point=False,
                enable_depth=False,
                enable_track=False,
                feature_only=True,
            )
        model.load_state_dict(state_dict, strict=False)
        return model

    def close(self) -> None:
        self.stream.synchronize()
        del self.model
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def __call__(self, observation: _model.Observation, *, reference_tokens: int) -> AlignFeatures:
        start_s = time.perf_counter()
        chunk_size = self.config.align_feature_batch_size
        batch_size = next(iter(observation.images.values())).shape[0]
        ranges = _balanced_chunk_ranges(batch_size, chunk_size)
        with torch.cuda.stream(self.stream):
            if len(ranges) > 1:
                chunks = (
                    (
                        start,
                        stop,
                        *self._extract_features(
                            _slice_observation_batch(observation, start, stop),
                            reference_tokens,
                        ),
                    )
                    for start, stop in ranges
                )
                targets, mask = _assemble_feature_chunks(batch_size, chunks)
            else:
                targets, mask = self._extract_features(observation, reference_tokens)
        return AlignFeatures(
            targets={self.device.index: targets},
            mask={self.device.index: mask},
            compute_started_s=start_s,
        )

    def _extract_features(
        self, observation: _model.Observation, reference_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.device(self.device):
            images, img_padding_mask, img_masks = self._prepare_images(observation)
            feature_layer = None if self.config.vggt_adapt_layers_align else self.config.vggt_layers_align

            with torch.autocast("cuda", dtype=torch.bfloat16):
                if self.target_model == "omnivggt" and self.config.use_camera_params:
                    extrinsics, intrinsics, camera_param_mask = self._prepare_camera_params(
                        observation, list(observation.images)
                    )
                    vggt_images, intrinsics = preprocess_images_from_openpi(images, intrinsics)
                    features, patch_start_idx = self.model.aggregator(
                        images=vggt_images,
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                        camera_gt_index=camera_param_mask,
                        feature_layer=feature_layer,
                    )
                else:
                    vggt_images = preprocess_images_from_openpi(images)
                    features, patch_start_idx = self.model.aggregator(
                        vggt_images, feature_layer=feature_layer
                    )

            vggt_hidden = [feature[:, :, patch_start_idx:, :] for feature in features]

            h, w = vggt_images.shape[-2:]
            patch_h, patch_w = h // self.model.patch_size, w // self.model.patch_size
            reference = torch.zeros(
                (vggt_images.shape[0], reference_tokens, 1),
                dtype=vggt_hidden[0].dtype,
                device=self.device,
            )
            pooled_hidden = [
                custom_pooling(
                    hidden,
                    (patch_h, patch_w),
                    (h, w),
                    reference,
                    self.config.pooling_func,
                    self.config.use_vggt_pe,
                )
                for hidden in vggt_hidden
            ]
            target = (
                torch.stack(pooled_hidden, dim=1) if self.config.vggt_adapt_layers_align else pooled_hidden[0].float()
            )
            mask = self._make_align_mask(img_masks, img_padding_mask, reference_tokens)
            return (
                target.to(torch.bfloat16).contiguous().detach(),
                mask.contiguous().detach(),
            )

    def _prepare_images(
        self, observation: _model.Observation
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], list[torch.Tensor]]:
        images = []
        padding_masks = {}
        image_masks = []
        for key, image in observation.images.items():
            empty_image = not np.asarray(image).any()
            image_t = torch.as_tensor(image, device=self.device)
            if image_t.ndim != 4:
                raise ValueError(f"Expected image {key} to have shape [B,H,W,C], got {image_t.shape}")
            image_t = (
                torch.ones_like(image_t, dtype=torch.float32)
                if empty_image
                else torch.clamp(image_t.to(torch.float32) / 2.0 + 0.5, 0, 1)
            )

            padding_mask = torch.as_tensor(observation.image_padding_mask[key], device=self.device).bool()
            image_t[~padding_mask] = 1.0
            images.append(image_t.permute(0, 3, 1, 2).contiguous())
            padding_masks[key] = padding_mask
            image_masks.append(torch.as_tensor(observation.image_masks[key], device=self.device).bool())
        return images, padding_masks, image_masks

    def _prepare_camera_params(self, observation: _model.Observation, image_keys: list[str]):
        if not observation.camera_param:
            return None, None, None
        extrinsics = torch.stack(
            [
                torch.as_tensor(observation.camera_param[f"{key}_extrinsics"], device=self.device).float()
                for key in image_keys
            ],
            dim=1,
        )
        intrinsics = torch.stack(
            [
                torch.as_tensor(observation.camera_param[f"{key}_intrinsics"], device=self.device).float()
                for key in image_keys
            ],
            dim=1,
        )
        camera_param_mask = [
            idx
            for idx, key in enumerate(image_keys)
            if observation.camera_param_mask
            and key in observation.camera_param_mask
            and bool(observation.camera_param_mask[key].all())
        ]
        return extrinsics, intrinsics, camera_param_mask

    def _make_align_mask(
        self,
        img_masks: list[torch.Tensor],
        img_padding_mask: dict[str, torch.Tensor],
        reference_tokens: int,
    ) -> torch.Tensor:
        # empty image feature masks for alignment loss
        tokens_per_img = reference_tokens // len(img_masks)
        mask = torch.repeat_interleave(torch.stack(img_masks, dim=1), repeats=tokens_per_img, dim=1)

        # useless image padding feature masks for alignment loss
        if self.config.ignore_img_padding_area:
            padding = torch.stack(list(img_padding_mask.values()), dim=1)
            target_size = padding.shape[-1] // 14
            mask_downsampled = (
                F.interpolate(
                    padding.float(),
                    size=(target_size, target_size),
                    mode="nearest",
                ).bool().flatten(start_dim=1)
            )
            mask = mask & mask_downsampled
        return mask


class TorchAlignFeatureExtractor:
    def __init__(self, config: Any, *, devices: list[int]):
        self.devices = devices
        target_model = infer_target_model(config)
        filename = "OmniVGGT.safetensors" if target_model == "omnivggt" else "model.pt"
        checkpoint_path = os.path.join(config.vggt_weight_path, filename)
        state_dict = (
            load_file(checkpoint_path)
            if target_model == "omnivggt"
            else torch.load(checkpoint_path, map_location="cpu")
        )
        self.shards = [
            _TorchAlignFeatureShard(
                config,
                device_id=device_id,
                state_dict=state_dict,
            )
            for device_id in devices
        ]
        self.executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=len(self.shards)) if len(self.shards) > 1 else None
        )

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        for shard in self.shards:
            shard.close()

    def finalize(self, features: AlignFeatures, *, release_cache: bool) -> None:
        for shard in self.shards:
            shard.stream.synchronize()
        if features.compute_started_s is not None:
            features.compute_s = time.perf_counter() - features.compute_started_s
        if release_cache:
            for device_id in self.devices:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()

    def __call__(self, observation: _model.Observation, *, reference_tokens: int) -> AlignFeatures:
        if len(self.shards) == 1:
            return self.shards[0](observation, reference_tokens=reference_tokens)

        start_s = time.perf_counter()
        batch_size = next(iter(observation.images.values())).shape[0]
        ranges = _split_batch_ranges(batch_size, len(self.shards))
        futures = []
        for shard, (start, stop) in zip(self.shards, ranges, strict=True):
            if start == stop:
                continue
            futures.append(
                self.executor.submit(
                    shard,
                    _slice_observation_batch(observation, start, stop),
                    reference_tokens=reference_tokens,
                )
            )
        features = [future.result() for future in futures]
        return AlignFeatures(
            targets={device: value for feature in features for device, value in feature.targets.items()},
            mask={device: value for feature in features for device, value in feature.mask.items()},
            compute_started_s=start_s,
        )


def _make_global_array(
    values: dict[int, torch.Tensor],
    *,
    global_batch_size: int,
    sharding: jax.sharding.Sharding,
) -> jax.Array:
    global_shape = (global_batch_size, *next(iter(values.values())).shape[1:])
    local_devices = list(jax.local_devices())
    source_positions = [i for i, device in enumerate(local_devices) if device.local_hardware_id in values]
    devices_by_process = np.asarray(jax.devices(), dtype=object).reshape(jax.process_count(), -1)
    devices = devices_by_process[:, source_positions].reshape(-1)
    source_sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.asarray(devices), ("align",)),
        jax.sharding.PartitionSpec("align"),
    )
    arrays = [
        jax.dlpack.from_dlpack(values[local_devices[position].local_hardware_id]) for position in source_positions
    ]
    array = jax.make_array_from_single_device_arrays(global_shape, source_sharding, arrays)
    return jax.device_put(array, sharding)


def shard_align_features(
    features: AlignFeatures,
    *,
    sharding: jax.sharding.Sharding,
    global_batch_size: int,
) -> tuple[jax.Array, jax.Array, float]:
    start_s = time.perf_counter()
    targets = _make_global_array(
        features.targets,
        global_batch_size=global_batch_size,
        sharding=sharding,
    )
    mask = _make_global_array(
        features.mask,
        global_batch_size=global_batch_size,
        sharding=sharding,
    )
    return targets, mask, time.perf_counter() - start_s


class AsyncAlignFeatureWorker:
    def __init__(self, extractor: TorchAlignFeatureExtractor):
        self.extractor = extractor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future: concurrent.futures.Future[AlignFeatures] | None = None

    def submit(self, observation: _model.Observation, *, reference_tokens: int) -> None:
        if self.future is not None:
            raise RuntimeError("An align batch is already pending.")
        self.future = self.executor.submit(
            self.extractor,
            observation,
            reference_tokens=reference_tokens,
        )

    def result(self, *, release_cache: bool = True) -> AlignFeatures:
        if self.future is None:
            raise RuntimeError("No align batch is pending.")
        future, self.future = self.future, None
        features = future.result()
        self.extractor.finalize(features, release_cache=release_cache)
        return features

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        self.extractor.close()


def _slice_observation_batch(observation: _model.Observation, start: int, stop: int) -> _model.Observation:
    return _model.Observation(
        images={key: value[start:stop] for key, value in observation.images.items()},
        image_masks={key: value[start:stop] for key, value in observation.image_masks.items()},
        state=observation.state[start:stop],
        image_padding_mask=None
        if observation.image_padding_mask is None
        else {key: value[start:stop] for key, value in observation.image_padding_mask.items()},
        tokenized_prompt=None if observation.tokenized_prompt is None else observation.tokenized_prompt[start:stop],
        tokenized_prompt_mask=None
        if observation.tokenized_prompt_mask is None
        else observation.tokenized_prompt_mask[start:stop],
        token_ar_mask=None if observation.token_ar_mask is None else observation.token_ar_mask[start:stop],
        token_loss_mask=None if observation.token_loss_mask is None else observation.token_loss_mask[start:stop],
        camera_param=None
        if observation.camera_param is None
        else {key: value[start:stop] for key, value in observation.camera_param.items()},
        camera_param_mask=None
        if observation.camera_param_mask is None
        else {key: value[start:stop] for key, value in observation.camera_param_mask.items()},
        index=None if observation.index is None else observation.index[start:stop],
    )
