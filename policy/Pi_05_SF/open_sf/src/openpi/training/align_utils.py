from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import os
from typing import Any, Literal

import jax
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models import model as _model
from openpi.models_pytorch import sf_offline_cache
from vggt.heads.utils import custom_pooling
from vggt.utils.load_fn import preprocess_images_from_openpi


AlignTargetModel = Literal["auto", "vggt"]
_OMNI_VGGT_TARGET = "omni" + "vggt"
_OMNI_VGGT_ERROR = "The Omni geometry model is not included in this Pi_05_SF migration."


@dataclasses.dataclass
class AlignFeatures:
    targets: np.ndarray
    mask: np.ndarray


@dataclasses.dataclass(frozen=True)
class AlignDeviceLayout:
    pi_device_ids: list[int]
    vggt_device_ids: list[int]


def prepare_align_observation(observation: _model.Observation) -> _model.Observation:
    """Materialize the observation on host memory for Torch feature extraction.

    This keeps the background align worker from touching sharded JAX arrays directly,
    which avoids cross-runtime device transfers inside the worker thread.
    """
    return jax.device_get(observation)


def infer_target_model(config: Any) -> Literal["vggt"]:
    target = getattr(config, "align_target_model", "auto")
    weight_path = getattr(config, "vggt_weight_path", None) or ""
    weight_path_lower = weight_path.lower()
    if target == _OMNI_VGGT_TARGET or (target == "auto" and _OMNI_VGGT_TARGET in weight_path_lower):
        raise ValueError(_OMNI_VGGT_ERROR)
    if target == "vggt":
        return "vggt"
    if "vggt" in weight_path_lower and target == "auto":
        return "vggt"
    raise ValueError(f"Invalid align_target_model {target} or vggt_weight_path {weight_path}")


def resolve_align_devices(config: Any) -> AlignDeviceLayout:
    if getattr(config, "sf_cache_enable", False):
        return AlignDeviceLayout(
            pi_device_ids=list(range(len(jax.devices()))),
            vggt_device_ids=[],
        )

    count = torch.cuda.device_count()
    vggt_devices = getattr(config, "align_vggt_devices", 1)
    if vggt_devices < 1:
        raise ValueError(f"align_vggt_devices must be at least 1, got {vggt_devices}.")
    if vggt_devices >= count:
        raise ValueError(
            f"align_vggt_devices must be smaller than the visible CUDA device count {count} so that at least one PI GPU remains."
        )
    split = count - vggt_devices
    return AlignDeviceLayout(
        pi_device_ids=list(range(split)),
        vggt_device_ids=list(range(split, count)),
    )


def _split_batch_ranges(batch_size: int, shard_count: int) -> list[tuple[int, int]]:
    boundaries = np.linspace(0, batch_size, shard_count + 1, dtype=int)
    return [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(shard_count)]


def _concat_align_features(chunks: list[AlignFeatures]) -> AlignFeatures:
    return AlignFeatures(
        targets=np.concatenate([chunk.targets for chunk in chunks], axis=0),
        mask=np.concatenate([chunk.mask for chunk in chunks], axis=0),
    )


def _as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _identity_array(sf_identity: dict[str, Any], name: str, batch_size: int) -> np.ndarray:
    if name not in sf_identity:
        raise ValueError(f"SF identity is missing required field {name!r}.")
    value = _as_numpy(jax.device_get(sf_identity[name])).reshape(-1)
    if value.shape[0] != batch_size:
        raise ValueError(f"SF identity field {name!r} has batch size {value.shape[0]}, expected {batch_size}.")
    return value.astype(np.int64, copy=False)


def _identity_from_observation(config: Any, observation: _model.Observation, batch_size: int) -> dict[str, np.ndarray]:
    if observation.index is None:
        raise ValueError("Offline SF cache loading requires sf_identity or observation.index.")
    step = _as_numpy(jax.device_get(observation.index)).reshape(-1).astype(np.int64, copy=False)
    if step.shape[0] != batch_size:
        raise ValueError(f"Observation index has batch size {step.shape[0]}, expected {batch_size}.")
    return {
        "dataset_uid": np.full(batch_size, int(getattr(config, "sf_dataset_uid", 0)), dtype=np.int64),
        "episode_index": np.zeros(batch_size, dtype=np.int64),
        "step_index": step,
    }


def _normalize_sf_identity(
    config: Any,
    observation: _model.Observation,
    sf_identity: dict[str, Any] | None,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if sf_identity is None:
        return _identity_from_observation(config, observation, batch_size)
    return {
        "dataset_uid": _identity_array(sf_identity, "dataset_uid", batch_size),
        "episode_index": _identity_array(sf_identity, "episode_index", batch_size),
        "step_index": _identity_array(sf_identity, "step_index", batch_size),
    }


def _make_align_mask_from_observation(config: Any, observation: _model.Observation, reference_tokens: int) -> np.ndarray:
    image_keys = list(observation.images)
    image_masks = [
        torch.as_tensor(_as_numpy(jax.device_get(observation.image_masks[key])).copy()).bool()
        for key in image_keys
    ]
    tokens_per_img = reference_tokens // len(image_masks)
    mask = torch.repeat_interleave(torch.stack(image_masks, dim=1), repeats=tokens_per_img, dim=1)

    if getattr(config, "ignore_img_padding_area", True) and observation.image_padding_mask:
        padding = torch.stack(
            [
                torch.as_tensor(_as_numpy(jax.device_get(observation.image_padding_mask[key])).copy()).bool()
                for key in image_keys
            ],
            dim=1,
        )
        target_size = padding.shape[-1] // 14
        mask_downsampled = F.interpolate(
            padding.float(),
            size=(target_size, target_size),
            mode="nearest",
        ).bool().flatten(start_dim=1)
        if mask_downsampled.shape == mask.shape:
            mask = mask & mask_downsampled
    return mask.detach().cpu().numpy().astype(np.bool_)


class _TorchAlignFeatureShard:
    def __init__(self, config: Any, *, device_id: int):
        self.config = config
        self.target_model = infer_target_model(config)
        self.device = torch.device(f"cuda:{device_id}")

        self.model = self._load_model().to(self.device)
        self.model.eval()

    def _load_model(self):
        from vggt.models.vggt import VGGT

        model = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
            feature_only=True,
        )
        vggt_ckpt_path = os.path.join(self.config.vggt_weight_path, "model.pt")
        if not os.path.exists(vggt_ckpt_path):
            raise FileNotFoundError(f"VGGT weight file not found at {vggt_ckpt_path}")
        model.load_state_dict(torch.load(vggt_ckpt_path), strict=False)
        return model

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def __call__(self, observation: _model.Observation, *, reference_tokens: int) -> AlignFeatures:
        chunk_size = getattr(self.config, "align_feature_batch_size", None)
        batch_size = next(iter(observation.images.values())).shape[0]
        if chunk_size is not None and 0 < chunk_size < batch_size:
            chunks = []
            for start in range(0, batch_size, chunk_size):
                stop = min(start + chunk_size, batch_size)
                chunks.append(self._extract_features(_slice_observation_batch(observation, start, stop), reference_tokens))
            return _concat_align_features(chunks)
        return self._extract_features(observation, reference_tokens)

    def _extract_features(self, observation: _model.Observation, reference_tokens: int) -> AlignFeatures:
        with torch.cuda.device(self.device) if self.device.type == "cuda" else contextlib.nullcontext():
            images, img_padding_mask, img_masks = self._prepare_images(observation)
            # from torchvision.utils import save_image; import matplotlib.cm as cm; import numpy as np; from PIL import Image
            # save_image(original_img[0,0], 'image.png')

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                vggt_images = preprocess_images_from_openpi(images)
                output = self.model(vggt_images)

            features = output["features"]
            patch_start_idx = output["patch_start_idx"]
            original_img = output["images"]
            vggt_hidden = [hidden[:, :, patch_start_idx:, :] for hidden in features]

            h, w = original_img.shape[-2:]
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
            target = pooled_hidden[self.config.vggt_layers_align].float()
            mask = self._make_align_mask(img_masks, img_padding_mask, reference_tokens)
            return AlignFeatures(
                targets=target.detach().cpu().numpy().astype(np.float32),
                mask=mask.detach().cpu().numpy().astype(np.bool_),
            )

    def _prepare_images(self, observation: _model.Observation) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], list[torch.Tensor]]:
        images = []
        padding_masks = {}
        image_masks = []
        for key, image in observation.images.items():
            image_t = torch.as_tensor(np.asarray(image).copy(), device=self.device)
            if image_t.ndim != 4:
                raise ValueError(f"Expected image {key} to have shape [B,H,W,C], got {image_t.shape}")
            image_t = image_t.to(torch.float32)
            image_t = torch.clamp(image_t / 2.0 + 0.5, 0, 1)

            padding_mask = torch.as_tensor(np.asarray(observation.image_padding_mask[key]).copy(), device=self.device).bool()
            image_t = image_t.clone() if not (image == 0).all() else torch.ones_like(image_t)  # TODO：用仅双视角做检查
            image_t[~padding_mask] = 1.0
            images.append(image_t.permute(0, 3, 1, 2).contiguous())
            padding_masks[key] = padding_mask
            image_masks.append(torch.as_tensor(np.asarray(observation.image_masks[key]).copy(), device=self.device).bool())
        return images, padding_masks, image_masks

    def _prepare_camera_params(self, observation: _model.Observation, image_keys: list[str]):
        if not observation.camera_param:
            return None, None, None
        extrinsics = torch.stack(
            [
                torch.as_tensor(np.asarray(observation.camera_param[f"{key}_extrinsics"]).copy(), device=self.device).float()
                for key in image_keys
            ],
            dim=1,
        )
        intrinsics = torch.stack(
            [
                torch.as_tensor(np.asarray(observation.camera_param[f"{key}_intrinsics"]).copy(), device=self.device).float()
                for key in image_keys
            ],
            dim=1,
        )
        camera_param_mask = [
            idx
            for idx, key in enumerate(image_keys)
            if observation.camera_param_mask
            and key in observation.camera_param_mask
            and bool(np.asarray(observation.camera_param_mask[key]).all())
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
        if getattr(self.config, "ignore_img_padding_area", True):
            padding = torch.stack(list(img_padding_mask.values()), dim=1)
            target_size = padding.shape[-1] // 14
            mask_downsampled = F.interpolate(
                padding.float(),
                size=(target_size, target_size),
                mode="nearest",
            ).bool().flatten(start_dim=1)
            if mask_downsampled.shape == mask.shape:
                mask = mask & mask_downsampled
        return mask


class TorchAlignFeatureExtractor:
    def __init__(self, config: Any, *, devices: list[int]):
        self.config = config
        self.shards = [_TorchAlignFeatureShard(config, device_id=device_id) for device_id in devices]
        self.executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=len(self.shards)) if len(self.shards) > 1 else None
        )

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        for shard in self.shards:
            shard.close()

    def __call__(
        self,
        observation: _model.Observation,
        *,
        reference_tokens: int,
        sf_identity: dict[str, Any] | None = None,
    ) -> AlignFeatures:
        del sf_identity
        if len(self.shards) == 1:
            return self.shards[0](observation, reference_tokens=reference_tokens)

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
        return _concat_align_features([future.result() for future in futures])


class OfflineAlignFeatureExtractor:
    def __init__(self, config: Any):
        self.config = config
        if not getattr(config, "sf_cache_dir", None):
            raise ValueError("Offline SF cache loading requires sf_cache_dir.")
        if getattr(config, "sf_cache_miss_policy", "error") != "error":
            raise ValueError("JAX offline SF cache loading supports sf_cache_miss_policy='error' only.")

    def close(self) -> None:
        pass

    def __call__(
        self,
        observation: _model.Observation,
        *,
        reference_tokens: int,
        sf_identity: dict[str, Any] | None = None,
    ) -> AlignFeatures:
        batch_size = next(iter(observation.images.values())).shape[0]
        identity = _normalize_sf_identity(self.config, observation, sf_identity, batch_size)
        expected_shape = (int(reference_tokens), 2 * int(self.config.vggt_dim))
        targets = []
        for i in range(batch_size):
            key = sf_offline_cache.make_cache_key(
                identity["dataset_uid"][i],
                identity["episode_index"][i],
                identity["step_index"][i],
            )
            tensor = sf_offline_cache.load_cached_tensor(
                self.config.sf_cache_dir,
                key,
                self.config.sf_cache_save_dtype,
                int(self.config.sf_cache_chunk_size),
                expected_shape=expected_shape,
                strict_shape=bool(getattr(self.config, "sf_cache_strict_shape", True)),
            )
            if tensor is None:
                raise ValueError(
                    "SF cache miss with sf_cache_miss_policy=error: "
                    f"dataset_uid={key.dataset_uid} episode_index={key.episode_index} step_index={key.step_index}"
                )
            targets.append(tensor.to(dtype=torch.float32).cpu().numpy())

        return AlignFeatures(
            targets=np.stack(targets, axis=0).astype(np.float32, copy=False),
            mask=_make_align_mask_from_observation(self.config, observation, reference_tokens),
        )


def create_align_feature_extractor(config: Any, *, devices: list[int]):
    if getattr(config, "sf_cache_enable", False):
        return OfflineAlignFeatureExtractor(config)
    return TorchAlignFeatureExtractor(config, devices=devices)


class AsyncAlignFeatureWorker:
    def __init__(self, extractor):
        self.extractor = extractor
        # TODO max_workers=1 is a temporary workaround 
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future: concurrent.futures.Future[AlignFeatures] | None = None

    def submit(
        self,
        observation: _model.Observation,
        *,
        reference_tokens: int,
        sf_identity: dict[str, Any] | None = None,
    ) -> None:
        self.future = self.executor.submit(
            self.extractor,
            observation,
            reference_tokens=reference_tokens,
            sf_identity=sf_identity,
        )

    def result(self) -> AlignFeatures:
        if self.future is None:
            raise RuntimeError("No align feature job has been submitted.")
        result = self.future.result()
        self.future = None
        return result

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
