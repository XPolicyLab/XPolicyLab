"""Stateful causal encoding for Wan video VAE observation streams."""

from contextlib import nullcontext

import torch


class IncrementalWanVAEEncoder:
    """Encode consecutive observation chunks with one causal VAE cache."""

    def __init__(self, vae):
        self._vae = vae
        self.reset()

    def reset(self) -> None:
        """Drop all causal features accumulated for this observation stream."""
        model = self._vae.model
        model.clear_cache()
        self._feat_map = [None] * len(model._enc_feat_map)
        self._spatial_shape: tuple[int, int] | None = None

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode one new raw-frame chunk without resetting causal features.

        Args:
            video: Normalized video with shape ``[1, 3, T, H, W]``. The first
                request normally contains one frame. Later requests normally
                contain a multiple of four frames, matching Wan VAE's temporal
                stride.

        Returns:
            Newly produced latents with shape ``[1, C, T_latent, H', W']``.

        Raises:
            ValueError: If the input shape, temporal chunking, or stream
                resolution is incompatible with causal encoding.
        """
        if video.ndim != 5 or video.shape[0] != 1:
            raise ValueError("incremental VAE encode requires shape [1,C,T,H,W], got " f"{tuple(video.shape)}")
        if video.shape[2] <= 0:
            raise ValueError("incremental VAE encode requires at least one frame")

        model = self._vae.model
        video = video.to(model.conv1.weight.device, dtype=self._vae.dtype)
        video = self._patchify(video)
        spatial_shape = tuple(video.shape[-2:])
        if self._spatial_shape is None:
            self._spatial_shape = spatial_shape
        elif spatial_shape != self._spatial_shape:
            raise ValueError(
                "incremental VAE stream resolution changed from " f"{self._spatial_shape} to {spatial_shape}"
            )

        chunks = self._temporal_chunks(video)
        autocast = (
            torch.amp.autocast("cuda", dtype=self._vae.dtype)
            if video.device.type == "cuda" and self._vae.dtype != torch.float32
            else nullcontext()
        )
        outputs = []
        with autocast:
            for chunk in chunks:
                feat_idx = [0]
                output = model.encoder(
                    chunk,
                    feat_cache=self._feat_map,
                    feat_idx=feat_idx,
                )
                outputs.append(output)
            encoded = torch.cat(outputs, dim=2)
            mean, _ = model.conv1(encoded).chunk(2, dim=1)

        latent_mean = self._vae.mean.to(mean.device).view(1, model.z_dim, 1, 1, 1)
        latent_inv_std = (1.0 / self._vae.std).to(mean.device).view(1, model.z_dim, 1, 1, 1)
        return ((mean - latent_mean) * latent_inv_std).float()

    @staticmethod
    def _temporal_chunks(video: torch.Tensor) -> list[torch.Tensor]:
        """Split new frames using the same 1-then-4 cadence as Wan VAE."""
        frame_count = video.shape[2]
        if frame_count == 1:
            return [video]
        if frame_count % 4 == 0:
            return [video[:, :, offset : offset + 4] for offset in range(0, frame_count, 4)]
        if (frame_count - 1) % 4 == 0:
            chunks = [video[:, :, :1]]
            chunks.extend(video[:, :, offset : offset + 4] for offset in range(1, frame_count, 4))
            return chunks
        raise ValueError(f"incremental VAE frame count {frame_count} must be 1, 4N, or 1+4N")

    @staticmethod
    def _patchify(video: torch.Tensor) -> torch.Tensor:
        """Apply Wan VAE's 2x2 channel-folding layout."""
        batch, channels, frames, height, width = video.shape
        if height % 2 or width % 2:
            raise ValueError("Wan VAE input height and width must be divisible by 2, got " f"{height}x{width}")
        return (
            video.reshape(
                batch,
                channels,
                frames,
                height // 2,
                2,
                width // 2,
                2,
            )
            .permute(0, 1, 6, 4, 2, 3, 5)
            .reshape(
                batch,
                channels * 4,
                frames,
                height // 2,
                width // 2,
            )
        )


class IncrementalWanVAEDecoder:
    """Decode predictions from a cache committed only with ground-truth latents."""

    def __init__(self, vae):
        self._vae = vae
        self.reset()

    def reset(self) -> None:
        """Drop decoder features accumulated from the observation stream."""
        model = self._vae.model
        model.clear_cache()
        self._feat_map = [None] * len(model._feat_map)
        self._initialized = False
        self._spatial_shape: tuple[int, int] | None = None

    @torch.no_grad()
    def decode(
        self,
        ground_truth_latents: torch.Tensor,
        prediction_latents: torch.Tensor,
        *,
        include_condition: bool = True,
    ) -> torch.Tensor:
        """Advance the persistent cache with GT and decode predictions on a fork.

        Args:
            ground_truth_latents: Newly encoded observation latents with shape
                ``[1, C, T_gt, H, W]``.
            prediction_latents: Future latents with shape
                ``[1, C, T_pred, H, W]``.
            include_condition: Include the latest decoded GT frame before the
                predicted frames.

        Returns:
            A decoded video with the latest GT frame followed by predicted
            frames, or only predicted frames when ``include_condition=False``.
        """
        self._validate_latents(ground_truth_latents, "ground_truth_latents")
        self._validate_latents(prediction_latents, "prediction_latents")
        if ground_truth_latents.shape[1] != prediction_latents.shape[1]:
            raise ValueError("ground-truth and prediction latent channels must match")
        if ground_truth_latents.shape[-2:] != prediction_latents.shape[-2:]:
            raise ValueError("ground-truth and prediction latent resolutions must match")

        model = self._vae.model
        device = model.conv2.weight.device
        ground_truth_latents = ground_truth_latents.to(
            device,
            dtype=self._vae.dtype,
        )
        prediction_latents = prediction_latents.to(
            device,
            dtype=self._vae.dtype,
        )

        ground_truth_video = self._decode_latents(
            ground_truth_latents,
            self._feat_map,
            commit=True,
        )
        prediction_cache = [
            item.clone() if isinstance(item, torch.Tensor) else item
            for item in self._feat_map
        ]
        prediction_video = self._decode_latents(
            prediction_latents,
            prediction_cache,
            commit=False,
        )

        videos = []
        if include_condition:
            videos.append(ground_truth_video[:, :, -1:])
        if prediction_video.shape[2] > 0:
            videos.append(prediction_video)
        if not videos:
            return prediction_video
        return torch.cat(videos, dim=2).clamp_(-1, 1).float()

    def _validate_latents(self, latents: torch.Tensor, name: str) -> None:
        if latents.ndim != 5 or latents.shape[0] != 1:
            raise ValueError(
                f"{name} must have shape [1,C,T,H,W], got "
                f"{tuple(latents.shape)}"
            )
        if name == "ground_truth_latents" and latents.shape[2] <= 0:
            raise ValueError("ground_truth_latents must contain at least one latent")
        spatial_shape = tuple(latents.shape[-2:])
        if self._spatial_shape is None:
            self._spatial_shape = spatial_shape
        elif spatial_shape != self._spatial_shape:
            raise ValueError(
                "incremental VAE decode resolution changed from "
                f"{self._spatial_shape} to {spatial_shape}"
            )

    def _decode_latents(
        self,
        latents: torch.Tensor,
        feat_map: list,
        *,
        commit: bool,
    ) -> torch.Tensor:
        if latents.shape[2] == 0:
            batch, _, _, height, width = latents.shape
            return latents.new_empty((batch, 3, 0, height * 16, width * 16))

        model = self._vae.model
        mean = self._vae.mean.to(latents.device).view(
            1,
            model.z_dim,
            1,
            1,
            1,
        )
        std = self._vae.std.to(latents.device).view(
            1,
            model.z_dim,
            1,
            1,
            1,
        )
        decoded_latents = model.conv2(latents * std + mean)
        outputs = []
        for index in range(decoded_latents.shape[2]):
            first_chunk = not self._initialized and commit and index == 0
            outputs.append(
                model.decoder(
                    decoded_latents[:, :, index : index + 1],
                    feat_cache=feat_map,
                    feat_idx=[0],
                    first_chunk=first_chunk,
                )
            )
            if commit:
                self._initialized = True
        return self._unpatchify(torch.cat(outputs, dim=2))

    @staticmethod
    def _unpatchify(video: torch.Tensor) -> torch.Tensor:
        """Undo Wan VAE's 2x2 channel-folding layout."""
        batch, channels, frames, height, width = video.shape
        if channels % 4:
            raise ValueError(
                "Wan VAE decoded channels must be divisible by four, got "
                f"{channels}"
            )
        return (
            video.reshape(
                batch,
                channels // 4,
                2,
                2,
                frames,
                height,
                width,
            )
            .permute(0, 1, 4, 5, 3, 6, 2)
            .reshape(
                batch,
                channels // 4,
                frames,
                height * 2,
                width * 2,
            )
        )
