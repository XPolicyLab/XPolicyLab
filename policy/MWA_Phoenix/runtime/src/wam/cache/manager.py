"""Public CacheManager facade.

Business code calls this module; repository/backend/domain details stay inside
``wam.cache``.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Any, Mapping, Sequence

import torch

from .binding import CacheBinding, CacheDataProjection, CacheRuntimeBindings
from .config import CacheConfig, CacheMode
from .contracts import (
    ArtifactContract,
    PreflightReport,
    ValidatedCacheSnapshot,
)
from .errors import CacheError
from .fingerprint import vae_fingerprint_from_model, vlm_fingerprint_from_path
from .fields import CacheField, DEFAULT_VAE_CACHE_FIELDS, vlm_artifact_id
from .repository import CacheRepository
from .vae_latent import VaeLatentSource, output_key_mapping
from .vlm_latent import (
    VlmLatentSource,
    output_key_mapping as vlm_output_key_mapping,
)


logger = logging.getLogger(__name__)


def _distributed_active() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _is_main_process() -> bool:
    return not _distributed_active() or torch.distributed.get_rank() == 0


def _broadcast_payload(payload: list[Any]) -> None:
    if _distributed_active():
        torch.distributed.broadcast_object_list(payload, src=0)


def _select_training_data_projection(
    *,
    fields: Sequence[CacheField],
    model: Any,
    dataset: Any,
    vlm_enabled: bool = False,
    vlm_partial: bool = False,
) -> CacheDataProjection:
    """Select an optimization only when cached/model data semantics permit it."""

    if set(fields) != set(DEFAULT_VAE_CACHE_FIELDS):
        return CacheDataProjection.DEFAULT
    if bool(getattr(model, "memory_enabled", False)):
        return CacheDataProjection.DEFAULT
    if bool(getattr(model, "understanding_enabled", False)):
        if vlm_enabled and not vlm_partial and callable(
            getattr(dataset, "get_item_without_images", None)
        ):
            return CacheDataProjection.LATENT_ONLY
        loader = getattr(dataset, "get_item_with_vlm_current_images", None)
        supports_projection = bool(
            getattr(dataset, "return_vlm_current_images", True)
        )
        if callable(loader) and supports_projection:
            return CacheDataProjection.VLM_CURRENT
        return CacheDataProjection.DEFAULT
    if callable(getattr(dataset, "get_item_without_images", None)):
        return CacheDataProjection.LATENT_ONLY
    return CacheDataProjection.DEFAULT


class CacheManager:
    """Single public facade for every artifact cache configuration and lifecycle."""

    def __init__(self, config: CacheConfig, *, raw_config: Any = None) -> None:
        self.config = config
        self.raw_config = raw_config
        self.repository = (
            None
            if config.root is None
            else CacheRepository(
                config.root,
                max_cached_shards=config.max_cached_shards,
                validation_workers=config.validation_workers,
                validation_max_inflight_bytes=(
                    config.validation_max_inflight_bytes
                ),
                validation_progress_interval_seconds=(
                    config.validation_progress_interval_seconds
                ),
            )
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "CacheManager":
        return cls(CacheConfig.from_config(cfg), raw_config=cfg)

    @property
    def mode(self) -> CacheMode:
        return self.config.mode

    def prepare_training(
        self,
        *,
        model: Any,
        datasets: Mapping[str, Any],
        distributed: Any,
        profiler: Any = None,
    ) -> CacheRuntimeBindings:
        """Validate and pin artifacts, then return external runtime bindings."""

        del distributed  # torch.distributed is the process-authoritative state.
        if self.mode is CacheMode.GENERATE:
            raise ValueError(
                "CacheManager.prepare_training() cannot run in generate mode."
            )
        if self.mode is CacheMode.OFF:
            return CacheRuntimeBindings.disabled()
        if self.repository is None:
            raise CacheError("Cache read mode has no configured repository root.")
        train_dataset = datasets.get("train")
        if train_dataset is None:
            raise ValueError("Cache training preparation requires a train dataset.")
        dtype = getattr(model, "torch_dtype", None)
        if not isinstance(dtype, torch.dtype):
            raise CacheError(f"Loaded model exposes invalid cache dtype: {dtype!r}.")

        vae_source = (
            VaeLatentSource.create(train_dataset, self.config.fields)
            if self.config.fields
            else None
        )
        vlm_source = (
            VlmLatentSource.create(train_dataset) if self.config.vlm_enabled else None
        )
        expected_keys = (
            vae_source.expected_keys()
            if vae_source is not None
            else vlm_source.expected_keys()
        )

        # Producer identity and every dependency contract are computed once on
        # rank 0, then broadcast.  Other ranks never rescan dataset inventories.
        contract_payload: list[Any] = [None]
        local_error: Exception | None = None
        if _is_main_process():
            started = time.perf_counter()
            try:
                fingerprint_started = time.perf_counter()
                contracts: dict[str, ArtifactContract] = {}
                fingerprints: dict[str, str] = {}
                if vae_source is not None:
                    vae = getattr(model, "vae", None)
                    if vae is None:
                        raise CacheError(
                            "Loaded model does not expose a VAE for cache validation."
                        )
                    vae_fingerprint = vae_fingerprint_from_model(model)
                    if not vae_fingerprint:
                        raise CacheError(
                            "Unable to resolve a content fingerprint for the loaded VAE."
                        )
                    fingerprints["vae"] = vae_fingerprint
                    contracts.update(
                        vae_source.contracts(
                            vae=vae,
                            vae_fingerprint=vae_fingerprint,
                            dtype=dtype,
                            value_codec=self.config.vae_value_codec,
                        )
                    )
                if vlm_source is not None:
                    understanding = getattr(model, "understanding", None)
                    understanding_cfg = getattr(model, "understanding_cfg", {})
                    if understanding is None:
                        raise CacheError(
                            "Selected VLM cache requires model understanding to be enabled."
                        )
                    vlm_path = understanding_cfg.get("vlm_model_path")
                    if not vlm_path:
                        raise CacheError(
                            "VLM cache requires a configured vlm_model_path."
                        )
                    vlm_fingerprint = vlm_fingerprint_from_path(vlm_path)
                    fingerprints["vlm"] = vlm_fingerprint
                    vlm_contract = vlm_source.contract(
                        producer_fingerprint=vlm_fingerprint,
                        context_dim=int(understanding.context_dim),
                        dtype=dtype,
                        understanding_config=understanding_cfg,
                        value_codec=self.config.vlm_value_codec,
                    )
                    contracts[vlm_contract.artifact_id] = vlm_contract
                fingerprint_seconds = time.perf_counter() - fingerprint_started
                contract_payload[0] = {
                    "ok": True,
                    "fingerprints": fingerprints,
                    "fingerprint_seconds": fingerprint_seconds,
                    "total_seconds": time.perf_counter() - started,
                    "contracts": {
                        name: contract.to_dict() for name, contract in contracts.items()
                    },
                }
            except Exception as exc:
                local_error = exc
                contract_payload[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "seconds": time.perf_counter() - started,
                }
        _broadcast_payload(contract_payload)
        contract_result = contract_payload[0]
        if not isinstance(contract_result, dict) or not contract_result.get("ok"):
            if local_error is not None:
                raise local_error
            raise CacheError(
                "Cache contract construction failed on rank 0: "
                f"{(contract_result or {}).get('error_type', 'Error')}: "
                f"{(contract_result or {}).get('error', 'unknown error')}"
            )
        raw_contracts = contract_result.get("contracts")
        if not isinstance(raw_contracts, dict):
            raise CacheError("Rank 0 did not publish artifact contracts.")
        contracts = {
            str(name): ArtifactContract.from_dict(payload)
            for name, payload in raw_contracts.items()
            if isinstance(payload, dict)
        }
        if set(contracts) != set(raw_contracts):
            raise CacheError("Rank 0 published malformed VAE artifact contracts.")
        logger.info(
            "Cache contracts ready: producers=%s fingerprint=%.3fs "
            "contracts_total=%.3fs artifacts=%s",
            contract_result["fingerprints"],
            float(contract_result["fingerprint_seconds"]),
            float(contract_result["total_seconds"]),
            list(contracts),
        )

        preflight_payload: list[Any] = [None]
        local_error = None
        section = (
            profiler.section("setup/video_latent_cache_preflight")
            if profiler is not None
            else nullcontext()
        )
        with section:
            if _is_main_process():
                try:
                    logger.info(
                        "Cache preflight start: root=%s artifacts=%s validation=%s",
                        self.repository.root,
                        list(contracts),
                        self.config.validation_mode.value,
                    )
                    snapshot, report = self.repository.preflight(
                        contracts,
                        expected_keys=expected_keys,
                        validation_mode=self.config.validation_mode,
                        allow_partial_artifacts=(
                            (vlm_artifact_id(self.config.vlm_value_codec),)
                            if self.config.vlm_enabled
                            else ()
                        ),
                    )
                    preflight_payload[0] = {
                        "ok": True,
                        "snapshot": snapshot.to_dict(),
                        "report": report.to_dict(),
                    }
                except Exception as exc:
                    local_error = exc
                    preflight_payload[0] = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
            _broadcast_payload(preflight_payload)
        result = preflight_payload[0]
        if not isinstance(result, dict) or not result.get("ok"):
            if local_error is not None:
                raise local_error
            raise CacheError(
                "Cache preflight failed on rank 0: "
                f"{(result or {}).get('error_type', 'Error')}: "
                f"{(result or {}).get('error', 'unknown error')}"
            )
        snapshot = ValidatedCacheSnapshot.from_dict(result["snapshot"])
        report = PreflightReport.from_dict(result["report"])
        vlm_id = (
            vlm_artifact_id(self.config.vlm_value_codec)
            if self.config.vlm_enabled
            else None
        )
        vlm_partial = bool(
            vlm_id is not None
            and report.artifacts[vlm_id].missing_sample_count > 0
        )
        for artifact_id, artifact_report in report.artifacts.items():
            logger.info(
                "Cache artifact passed: artifact=%s build=%s samples=%d "
                "validation=%s shards=%d bytes=%.2f GiB manifest=%.3fs index=%.3fs "
                "structure=%.3fs shard_data=%.3fs workers=%d "
                "stat_sum=%.3fs read_sum=%.3fs hash_sum=%.3fs "
                "load_sum=%.3fs check_sum=%.3fs total=%.3fs",
                artifact_id,
                artifact_report.build_id,
                artifact_report.sample_count,
                artifact_report.validation_mode,
                artifact_report.shard_count,
                artifact_report.total_bytes / (1024**3),
                artifact_report.manifest_seconds,
                artifact_report.index_seconds,
                artifact_report.structure_seconds,
                artifact_report.shard_seconds,
                artifact_report.validation_workers,
                artifact_report.shard_stat_seconds,
                artifact_report.shard_read_seconds,
                artifact_report.shard_hash_seconds,
                artifact_report.shard_load_seconds,
                artifact_report.shard_check_seconds,
                artifact_report.total_seconds,
            )
            if artifact_report.missing_sample_count and _is_main_process():
                logger.warning(
                    "Cache artifact partially covered; online fallback enabled: "
                    "artifact=%s cached=%d expected=%d missing=%d coverage=%.2f%% "
                    "examples=%s",
                    artifact_id,
                    artifact_report.sample_count,
                    artifact_report.expected_sample_count,
                    artifact_report.missing_sample_count,
                    artifact_report.coverage_ratio * 100.0,
                    list(artifact_report.missing_examples),
                )
        logger.info(
            "Cache preflight passed: validation=%s artifacts=%s samples=%d shards=%d "
            "bytes=%.2f GiB wall=%.3fs snapshot_catalog=%s",
            report.validation_mode,
            list(report.artifacts),
            report.sample_count,
            report.shard_count,
            report.total_bytes / (1024**3),
            report.wall_seconds,
            snapshot.catalog_sha256,
        )
        mappings = output_key_mapping(
            self.config.fields,
            value_codec=self.config.vae_value_codec,
        )
        if self.config.vlm_enabled:
            mappings.update(vlm_output_key_mapping(self.config.vlm_value_codec))
        data_projection = _select_training_data_projection(
            fields=self.config.fields,
            model=model,
            dataset=train_dataset,
            vlm_enabled=self.config.vlm_enabled,
            vlm_partial=vlm_partial,
        )
        train_binding = CacheBinding(
            snapshot=snapshot,
            contracts=contracts,
            output_keys=mappings,
            max_cached_shards=self.config.max_cached_shards,
            report=report,
            data_projection=data_projection,
            partial_artifact_hit_keys=(
                {vlm_id: "vlm_cache_hit_mask"}
                if vlm_partial and vlm_id is not None
                else {}
            ),
        )
        if data_projection is CacheDataProjection.LATENT_ONLY:
            logger.info(
                "Cache training projection enabled: RGB decode disabled; all "
                "selected visual producer outputs come from the validated snapshot."
            )
        elif data_projection is CacheDataProjection.VLM_CURRENT:
            logger.info(
                "Artifact cache VLM projection enabled: decoding only the current "
                "multi-camera observation; current/anchor/recent model inputs "
                "are supplied by the validated latent snapshot."
            )
        else:
            logger.info(
                "Artifact cache training projection disabled: retaining source RGB "
                "decode for partial-cache or image-consuming model semantics."
            )
        # Validation remains online by default.  Its sample mapping is
        # independent and must not accidentally inherit the train snapshot.
        return CacheRuntimeBindings(
            {"train": train_binding, "validation": CacheBinding()}
        )

    def generate(self, *, distributed: Any = None, progress: Any = None) -> Any:
        """Generate all selected artifacts through one strict build session."""

        if self.mode is not CacheMode.GENERATE:
            raise ValueError(
                "CacheManager.generate() requires video_latent_cache=generate."
            )
        if self.raw_config is None:
            raise ValueError("CacheManager.generate() requires the composed config.")
        from .vae_generation import generate_vae_cache

        return generate_vae_cache(
            manager=self,
            cfg=self.raw_config,
            distributed=distributed,
            progress=progress,
        )
