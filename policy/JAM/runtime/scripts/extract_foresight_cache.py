"""Extract frozen foresight targets to a bf16+zstd cache (bit-exact).

Reuses the EXACT training code path (same hydra config, same dataset, same
FastWAM.build_inputs VAE encode and the same target helpers) so cached targets
equal what online training would compute. Historical ``encoder_hidden`` jobs
cache full teacher states. A ``vae_direct`` task stores compact raw 192-D VAE
patches and training applies the same fixed isometric 192->3072 lift.

Run under the same hydra config as training, with extraction knobs added via `+`:

  python scripts/extract_foresight_cache.py \
      task=robotwin_foresight_only_3cam_384_1e-4 \
      '+extract.out_roots=[/original-workspace/fs_cache_rt,/original-workspace/FastWAM/fs_cache_rt]' \
      '+extract.layers=[16]' \
      +extract.batch_size=32 +extract.num_workers=40

Multi-rank: launch one process per GPU (mpirun); rank r takes shards r::world.
Resumable: completed shards (final .bin+.idx present) are skipped.

Extraction bypasses the base dataset's random decode-failure substitution. If
an exact source index cannot be decoded, the current shard is aborted and the
job fails; a later resume retries that shard atomically. This guarantees that a
cache record can never contain the target of a different sample index.
"""

import logging
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastwam.datasets.foresight_cache import (  # noqa: E402
    ShardWriter, pick_root, shard_name, write_meta,
)
from fastwam.models.wan22.foresight import (  # noqa: E402
    encode_foresight_hidden,
    normalize_foresight_target_type,
    patch_foresight_vae_latents,
)
from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402
from fastwam.utils import misc  # noqa: E402

register_default_resolvers()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("extract_foresight_cache")


class _ExactIndexSubset(Dataset):
    """Subset that never permits RobotVideoDataset's random index fallback."""

    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        exact_get = getattr(self.dataset, "_get", None)
        if exact_get is None:
            raise TypeError(
                "Exact cache extraction requires a dataset._get(index) method "
                "that bypasses random decode-failure substitution."
            )
        return exact_get(index)


def _dist_rank_world() -> tuple[int, int]:
    for r_key, w_key in (("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
                         ("RANK", "WORLD_SIZE"), ("PMI_RANK", "PMI_SIZE")):
        if r_key in os.environ:
            return int(os.environ[r_key]), int(os.environ[w_key])
    return 0, 1


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    ex = cfg.get("extract")
    if ex is None:
        raise SystemExit("pass extraction knobs via +extract.out_roots=[...] etc. (see module docstring)")
    out_roots = [Path(p) for p in ex.out_roots]
    layers = sorted(int(x) for x in ex.layers)
    batch_size = int(ex.get("batch_size", 16))
    num_workers = int(ex.get("num_workers", 24))
    shard_size = int(ex.get("shard_size", 1024))
    limit_shards = ex.get("limit_shards")  # smoke-test knob
    input_latent_only = bool(ex.get("input_latent_only", False))
    source_split = str(ex.get("source_split", "train"))
    if source_split not in {"train", "val"}:
        raise SystemExit(
            f"extract.source_split must be train or val, got {source_split!r}."
        )
    target_type = normalize_foresight_target_type(
        cfg.model.foresight.get("target_type", "encoder_hidden")
    )
    stored_offsets_cfg = ex.get("offsets", None)
    if stored_offsets_cfg is None:
        stored_offsets_cfg = cfg.model.foresight.get("offsets", None)
    stored_offsets = (
        None
        if stored_offsets_cfg is None
        else [int(value) for value in stored_offsets_cfg]
    )
    cache_target_type = "input_vae_latent" if input_latent_only else target_type
    if input_latent_only:
        if layers != [0]:
            raise SystemExit(
                f"Input-latent cache uses metadata layer [0], got layers={layers}."
            )
        stored_offsets = None
    elif target_type == "vae_direct":
        if layers != [0]:
            raise SystemExit(
                f"VAE-direct cache uses metadata layer [0], got layers={layers}."
            )
        if not stored_offsets:
            raise SystemExit("VAE-direct compact cache requires non-empty extract.offsets.")
    elif target_type != "encoder_hidden":
        raise SystemExit(
            f"Cache extraction supports encoder_hidden or vae_direct, got {target_type}."
        )

    rank, world = _dist_rank_world()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for r in out_roots:
        r.mkdir(parents=True, exist_ok=True)

    dataset_cfg = OmegaConf.to_container(
        cfg.data[source_split], resolve=True
    )
    if input_latent_only:
        # Only the current RGB observation is needed while producing the cache.
        # Training later reads the cached latent and opens no video at all.
        dataset_cfg["first_frame_only"] = True
        dataset_cfg["use_input_latent_cache"] = False
    # Dataset construction saves a copy of normalization stats on rank zero.
    # Fresh source snapshots have no default ./runs directory, so register the
    # configured output directory before constructing the dataset.
    misc.register_work_dir(cfg.output_dir)
    dataset = instantiate(dataset_cfg)
    n = len(dataset)
    logger.info(
        "rank %d/%d split=%s dataset=%d target_type=%s layers=%s "
        "stored_offsets=%s roots=%s",
        rank,
        world,
        source_split,
        n,
        cache_target_type,
        layers,
        stored_offsets,
        out_roots,
    )

    # Full FastWAM exactly as in training: provides the VAE (build_inputs) and the
    # frozen encoder copied from the PRETRAINED video expert. Ensure the model cfg
    # has foresight enabled and no cache path (we are producing the cache).
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    fs = model_cfg.setdefault("foresight", {})
    if not fs.get("enabled", False):
        raise SystemExit("model config must have foresight.enabled=true for extraction")
    fs.pop("target_cache_roots", None)
    fs["encoder_layers"] = layers
    # Historical hidden-state extraction uses matching student/teacher depths.
    # VAE-direct uses metadata encoder layer0 but must retain a valid student
    # Wan layer (the task's layer18); FastWAM rejects student layer0.
    if target_type == "encoder_hidden":
        fs["layers"] = layers
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=device)
    model.eval()
    encoder = model.foresight_encoder
    if target_type == "encoder_hidden" and encoder is None:
        raise RuntimeError("model built without a foresight encoder; cannot extract")
    if target_type == "vae_direct" and encoder is not None:
        raise RuntimeError("VAE-direct extraction must not instantiate a foresight encoder.")

    seq_len = dim = None
    shard_starts = list(range(0, n, shard_size))
    if limit_shards is not None:
        shard_starts = shard_starts[: int(limit_shards)]
    my_shards = shard_starts[rank::world]

    # Clean stale tmp files from a previous killed run (rank 0 only; no job races us).
    if rank == 0:
        for r in out_roots:
            for p in r.glob("*.tmp"):
                p.unlink(missing_ok=True)

    def _complete(start):
        return any((r / (shard_name(start) + ".bin")).exists() and
                   (r / (shard_name(start) + ".idx.npy")).exists() for r in out_roots)

    todo = [st for st in my_shards if not _complete(st)]
    skipped = len(my_shards) - len(todo)
    indices = [i for st in todo for i in range(st, min(st + shard_size, n))]
    logger.info("rank %d: %d shards todo (%d skipped), %d samples", rank, len(todo), skipped, len(indices))

    done = 0
    t0 = time.time()
    if indices:
        # ONE persistent loader per rank (no per-shard respawn: that doubled RSS and
        # caused the cgroup OOM with 40 workers); shard writers roll at boundaries.
        loader = DataLoader(
            _ExactIndexSubset(dataset, indices),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=(device == "cuda"),
            multiprocessing_context="spawn" if num_workers > 0 else None,
            persistent_workers=(num_workers > 0),
            prefetch_factor=1 if num_workers > 0 else None,  # decode-bound; 1 halves in-flight batch RAM
        )
        writer = None
        cur_start = -1
        ptr = 0
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda" else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        )
        try:
            def _mem_gb():
                try:
                    with open("/proc/meminfo") as fh:
                        for line in fh:
                            if line.startswith("MemAvailable"):
                                return int(line.split()[1]) / 1e6
                except OSError:
                    return -1.0
                return -1.0

            nb = 0
            with torch.no_grad(), autocast_ctx:
                for sample in loader:
                    nb += 1
                    if nb % 50 == 1:
                        logger.info("rank %d: batch %d MemAvailable=%.1fGB", rank, nb, _mem_gb())
                    num_target_frames = None
                    latent_shape = None
                    if input_latent_only:
                        video = sample.get("video")
                        if video is None or video.ndim != 5 or video.shape[2] != 1:
                            raise RuntimeError(
                                "Input-latent extraction requires a one-frame RGB batch, "
                                f"got {None if video is None else tuple(video.shape)}."
                            )
                        input_latents = model._encode_video_latents(
                            video.to(
                                device=model.device,
                                dtype=model.torch_dtype,
                                non_blocking=True,
                            )
                        )
                        latent_shape = [int(value) for value in input_latents.shape[1:]]
                        rec = input_latents.to(torch.bfloat16).cpu().reshape(
                            input_latents.shape[0],
                            1,
                            -1,
                            input_latents.shape[-1],
                        )
                    else:
                        inputs = model.build_inputs(sample)
                    if not input_latent_only and target_type == "encoder_hidden":
                        per_layer = []
                        for L in layers:
                            enc_hidden, tpf = encode_foresight_hidden(
                                encoder=encoder,
                                clean_latents=inputs["input_latents"],
                                context=inputs["context"],
                                context_mask=inputs["context_mask"],
                                capture_layer=L,
                                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                            )
                            per_layer.append(enc_hidden.to(torch.bfloat16).cpu())
                        rec = torch.stack(per_layer, dim=1)  # [B, L, S, D]
                    elif not input_latent_only:
                        raw, tpf = patch_foresight_vae_latents(
                            inputs["input_latents"],
                            patch_size=model.video_expert.patch_size,
                        )
                        if raw.shape[1] % int(tpf) != 0:
                            raise RuntimeError(
                                f"VAE-direct token shape {tuple(raw.shape)} "
                                f"is not divisible by tpf={tpf}."
                            )
                        num_target_frames = int(raw.shape[1]) // int(tpf)
                        if any(
                            offset < 0 or offset >= num_target_frames
                            for offset in stored_offsets
                        ):
                            raise RuntimeError(
                                f"stored offsets {stored_offsets} outside "
                                f"[0, {num_target_frames}) for VAE-direct cache."
                            )
                        raw = raw.view(
                            raw.shape[0],
                            num_target_frames,
                            int(tpf),
                            raw.shape[-1],
                        )[:, stored_offsets]
                        raw = raw.reshape(
                            raw.shape[0],
                            len(stored_offsets) * int(tpf),
                            raw.shape[-1],
                        )
                        rec = raw.to(torch.bfloat16).cpu().unsqueeze(1)
                    if seq_len is None:
                        seq_len, dim = rec.shape[2], rec.shape[3]
                        meta_extra = {
                            "target_type": cache_target_type,
                        }
                        if input_latent_only:
                            meta_extra.update(
                                latent_shape=latent_shape,
                                representation="frozen_policy_vae_current_latent",
                                source_split=source_split,
                            )
                        else:
                            meta_extra["tokens_per_frame"] = int(tpf)
                        if not input_latent_only and target_type == "vae_direct":
                            meta_extra.update(
                                stored_offsets=stored_offsets,
                                num_target_frames=num_target_frames,
                                representation="raw_vae_patch_before_isometric_lift",
                            )
                        for r in out_roots:
                            write_meta(r, layers=layers, seq_len=seq_len, dim=dim,
                                       num_samples=n, shard_size=shard_size,
                                       extra=meta_extra)
                    for b in range(rec.shape[0]):
                        idx = indices[ptr]; ptr += 1
                        st = (idx // shard_size) * shard_size
                        if st != cur_start:
                            if writer is not None:
                                writer.finalize(); done += 1
                                rate = ptr / max(time.time() - t0, 1e-9)
                                logger.info("rank %d: shard %d done (%d/%d, %.1f samples/s, eta %.1fh)",
                                            rank, cur_start, done, len(todo), rate,
                                            (len(indices) - ptr) / max(rate, 1e-9) / 3600)
                            raw_shard_bytes = (
                                min(shard_size, n - st)
                                * int(rec.shape[1])
                                * int(rec.shape[2])
                                * int(rec.shape[3])
                                * 2
                            )
                            root = pick_root(
                                out_roots,
                                need_bytes=max(raw_shard_bytes, 1),
                            )
                            writer = ShardWriter(root, st, min(shard_size, n - st))
                            cur_start = st
                        writer.add(idx, rec[b])
            if writer is not None:
                writer.finalize(); done += 1
        except Exception:
            if writer is not None:
                writer.abort()
            raise
    logger.info("rank %d finished: %d shards written, %d skipped (already complete)", rank, done, skipped)


if __name__ == "__main__":
    main()
