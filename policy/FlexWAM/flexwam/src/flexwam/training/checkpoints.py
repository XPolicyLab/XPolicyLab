from typing import Any
import torch
import torch.nn as nn
from flexwam.utils.logging_config import get_logger
from flexwam.models.architectures.state_conditioning import state_config
logger = get_logger(__name__)

from flexwam.models.architectures.view_scoring_head import (
    CANONICAL_HEAD_ARCH,
    SpatialQueryHead,
    arches_compatible,
    normalize_head_arch,
)

HEAD_WEIGHT_KEYS = (
    "feature_dim",
    "pool_grid",
    "num_queries",
    "num_heads",
    "spatial_hidden_dim",
    "branch_output_dim",
    "head_hidden_dim",
    "state_conditioning",
    "proprio_dim",
)


class CheckpointMixin:
    def state_conditioning_signature(self) -> dict[str, Any]:
        """Architecture and dropout contract for full training-state resumes."""
        return {
            "mode": self.state_conditioning_mode,
            "video": dict(getattr(self.video_expert, "state_conditioning", state_config(None, video=True))),
            "router": dict(self.action_camera_router.state_conditioning),
            "critic": dict(self.action_camera_critic.state_conditioning),
            "proprio_dim": self.proprio_dim,
            "drop_prob": self.state_drop_prob,
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        payload["state_proj"] = self.state_proj.state_dict()
        if self.context_adapter is not None:
            payload["context_adapter"] = self.context_adapter.state_dict()
        payload["action_camera_router"] = self.action_camera_router.state_dict()
        payload["action_router_arch"] = str(getattr(self, "action_router_arch", CANONICAL_HEAD_ARCH))
        payload["action_router_feature_layer_index"] = int(self.action_router_feature_layer_index)
        payload["action_router_config"] = dict(getattr(self, "action_router_config", {}))
        payload["action_camera_critic"] = self.action_camera_critic.state_dict()
        payload["action_critic_arch"] = str(getattr(self, "action_critic_arch", CANONICAL_HEAD_ARCH))
        payload["action_critic_feature_layer_index"] = int(self.action_critic_feature_layer_index)
        payload["action_critic_config"] = dict(getattr(self, "action_critic_config", {}))
        video_state = getattr(self.video_expert, "state_conditioning", state_config(None, video=True))
        if self.state_conditioning_mode == "shared" or video_state["enabled"] or self.state_drop_prob or any(
            head.state_encoder is not None for head in (self.action_camera_router, self.action_camera_critic)
        ):
            payload["state_conditioning"] = {
                "version": 2, "mode": self.state_conditioning_mode,
                "video": dict(video_state), "proprio_dim": self.proprio_dim,
                "drop_prob": self.state_drop_prob,
            }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    @staticmethod
    def _payload_head_config(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
        stored = payload.get(f"{prefix}_config")
        if isinstance(stored, dict) and stored:
            config = dict(stored)
        else:
            config = {}
            arch = payload.get(f"{prefix}_arch")
            if prefix == "action_critic" and arch is None:
                arch = payload.get("action_router_arch")
            if arch is not None:
                config["arch"] = arch
            layer = payload.get(f"{prefix}_feature_layer_index")
            if prefix == "action_critic" and layer is None:
                layer = payload.get("action_router_feature_layer_index")
            if layer is not None:
                config["feature_layer_index"] = int(layer)
        if "arch" in config:
            try:
                config["arch"] = normalize_head_arch(config["arch"])
            except ValueError:
                pass
        pool_grid = config.get("pool_grid")
        if pool_grid is not None:
            config["pool_grid"] = [int(pool_grid[0]), int(pool_grid[1])]
        return config

    def _head_mismatch(self, module: SpatialQueryHead, layer_index: int, payload_config: dict[str, Any]) -> str | None:
        current = module.network_config()
        current["feature_layer_index"] = int(layer_index)
        if "arch" in payload_config and not arches_compatible(payload_config.get("arch"), current["arch"]):
            return f"arch checkpoint={payload_config.get('arch')!r} current={current['arch']!r}"
        payload_layer = payload_config.get("feature_layer_index")
        if payload_layer is not None and int(payload_layer) != int(layer_index):
            return f"feature_layer_index checkpoint={payload_layer} current={layer_index}"
        for key in HEAD_WEIGHT_KEYS:
            if key not in payload_config:
                continue
            left, right = payload_config[key], current[key]
            if key == "pool_grid":
                left = [int(left[0]), int(left[1])]
                right = [int(right[0]), int(right[1])]
            if left != right:
                return f"{key} checkpoint={left!r} current={right!r}"
        return None

    def _load_camera_head_state(
        self,
        module: nn.Module,
        payload: dict[str, Any],
        key: str,
        *,
        prefix: str,
        layer_index: int,
        reinit_incompatible_heads: bool,
    ) -> bool:
        if key not in payload:
            logger.warning("Checkpoint has no `%s` weights; keeping current params.", key)
            return False
        payload_config = self._payload_head_config(payload, prefix)
        mismatch = self._head_mismatch(module, layer_index, payload_config)
        if mismatch is None:
            # Preflight before load_state_dict: a failing strict load can already
            # have overwritten matching parameters in an otherwise incompatible head.
            expected = module.state_dict()
            incoming = payload[key]
            missing = sorted(expected.keys() - incoming.keys())
            unexpected = sorted(incoming.keys() - expected.keys())
            if missing or unexpected:
                mismatch = f"state keys missing={missing} unexpected={unexpected}"
            else:
                for name, tensor in expected.items():
                    if incoming[name].shape != tensor.shape:
                        mismatch = (f"{name} shape checkpoint={tuple(incoming[name].shape)} "
                                    f"current={tuple(tensor.shape)}")
                        break
        if mismatch is not None:
            message = f"Incompatible `{key}` checkpoint ({mismatch})."
            if reinit_incompatible_heads:
                logger.warning("%s Reinitializing this head.", message)
                return True
            raise ValueError(message + " Pass reinit_incompatible_heads=True to reinitialize this head.")
        module.load_state_dict(payload[key], strict=True)
        return False

    def _migrate_state_heads(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Only widen the final MLP input; preserve every existing visual weight."""
        payload = dict(payload)
        for role in ("router", "critic"):
            key, prefix = f"action_camera_{role}", f"action_{role}"
            module = getattr(self, key)
            incoming = payload.get(key)
            if module.state_encoder is None or incoming is None or any(
                name.startswith("state_encoder.") for name in incoming
            ):
                continue
            config = self._payload_head_config(payload, prefix)
            config.update(state_conditioning=module.state_conditioning, proprio_dim=module.proprio_dim)
            mismatch = self._head_mismatch(module, getattr(self, f"action_{role}_feature_layer_index"), config)
            expected = module.state_dict()
            old_keys = {name for name in expected if not name.startswith("state_encoder.")}
            if mismatch or set(incoming) != old_keys:
                raise ValueError(f"Cannot migrate {key}: {mismatch or 'unrelated parameter differences'}")
            migrated = dict(expected)
            visual_dim = module.num_views * 2 * module.branch_output_dim
            for name, value in incoming.items():
                shape = expected[name].shape
                if name == "head.0.weight":
                    if value.shape != (shape[0], visual_dim):
                        raise ValueError(f"Cannot migrate {key}: incompatible visual MLP shape.")
                    expanded = value.new_zeros(shape)
                    expanded[:, :visual_dim] = value
                    migrated[name] = expanded
                else:
                    if value.shape != shape:
                        raise ValueError(f"Cannot migrate {key}.{name}: shape mismatch.")
                    migrated[name] = value
            payload[key] = migrated
            payload[f"{prefix}_config"] = config
            logger.info("Migrated %s with zero-initialized state columns; preserved visual weights.", key)
        return payload

    def load_checkpoint(self, path, optimizer=None, reinit_incompatible_heads: bool = False,
                        migrate_state_conditioning: bool = False):
        payload = torch.load(path, map_location="cpu")
        adapter_state = payload.get("context_adapter")
        if (adapter_state is not None) != (self.context_adapter is not None):
            raise ValueError("Checkpoint context adapter does not match context_input_dim/text_dim configuration.")
        if self.context_adapter is not None:
            self.context_adapter.load_state_dict(adapter_state, strict=True)
        saved_state = payload.get("state_conditioning", {})
        if saved_state and saved_state.get("version") not in (1, 2):
            raise ValueError("Unsupported state conditioning checkpoint version.")
        saved_mode = saved_state.get("mode", "separate")
        if saved_mode != self.state_conditioning_mode:
            raise ValueError(
                f"State conditioning mode mismatch: checkpoint={saved_mode}, "
                f"model={self.state_conditioning_mode}. Load with the checkpoint's mode."
            )
        incoming_video = payload.get("mot", payload.get("dit", {}))
        video_prefix = "mixtures.video.state_proj." if "mot" in payload else "state_proj."
        saved_video_enabled = any(key.startswith(video_prefix) for key in incoming_video)
        if saved_mode == "separate" and saved_state and bool(saved_state.get("video", {}).get("enabled", False)) != saved_video_enabled:
            raise ValueError("Video state checkpoint metadata does not match saved projection weights.")
        if saved_mode == "shared":
            if "state_proj" not in payload:
                raise ValueError("Shared state checkpoint is missing its state projection.")
            if optimizer is not None and saved_state["video"] != self.video_expert.state_conditioning:
                raise ValueError("Video state switch differs from checkpoint during optimizer resume.")
        video_enabled = getattr(self.video_expert, "state_proj", None) is not None
        needs_migration = video_enabled and not saved_video_enabled
        if saved_video_enabled and not video_enabled:
            raise ValueError("Checkpoint uses video state conditioning; enable it in the model config.")
        if saved_video_enabled and saved_state:
            if (saved_state.get("video") != self.video_expert.state_conditioning
                    or saved_state.get("proprio_dim") != self.proprio_dim):
                raise ValueError("Video state conditioning checkpoint config mismatch.")
        for role in ("router", "critic"):
            module = getattr(self, f"action_camera_{role}")
            incoming = payload.get(f"action_camera_{role}", {})
            head_config = self._payload_head_config(payload, f"action_{role}")
            if head_config.get("state_conditioning", {}).get("enabled", False) and not any(
                k.startswith("state_encoder.") for k in incoming
            ):
                raise ValueError(f"{role} checkpoint is missing its declared state encoder.")
            if module.state_encoder is not None and not any(k.startswith("state_encoder.") for k in incoming):
                needs_migration = True
        if needs_migration:
            if not migrate_state_conditioning:
                raise ValueError("Checkpoint lacks enabled state paths; pass migrate_state_conditioning=True to initialize them.")
            if optimizer is not None:
                raise ValueError("Cannot restore optimizer during state conditioning migration; use a fresh optimizer.")
            payload = self._migrate_state_heads(payload)
            if video_enabled and not saved_video_enabled:
                logger.info("Initialized video state projection; the added prefix requires fine-tuning.")
        if optimizer is not None and float(saved_state.get("drop_prob", 0.0)) != self.state_drop_prob:
            raise ValueError("State dropout differs from checkpoint during optimizer resume.")
        if "mot" in payload:
            conditioning_prefixes = (
                "mixtures.video.action_embedder.",
                "mixtures.action.context_embedder.",
            )
            fusion_prefix = "mixtures.video.cross_view_fusion."
            mot_state = {}
            ignored_conditioning = 0
            ignored_fusion = 0
            for key, value in payload["mot"].items():
                if key.startswith(conditioning_prefixes):
                    ignored_conditioning += 1
                elif key.startswith(fusion_prefix):
                    ignored_fusion += 1
                else:
                    mot_state[key] = value
            if ignored_conditioning:
                logger.info(
                    "Ignored %d obsolete LingBot conditioning weights from checkpoint.",
                    ignored_conditioning,
                )
            if ignored_fusion:
                logger.info(
                    "Ignored %d cross-view fusion weights; video layers now use cross-view attention.",
                    ignored_fusion,
                )
            self.mot.load_state_dict(mot_state, strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            dit_state = {
                key: value
                for key, value in payload["dit"].items()
                if not key.startswith("cross_view_fusion.")
            }
            ignored_fusion = len(payload["dit"]) - len(dit_state)
            if ignored_fusion:
                logger.info(
                    "Ignored %d cross-view fusion weights; video layers now use cross-view attention.",
                    ignored_fusion,
                )
            self.video_expert.load_state_dict(dit_state, strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if "proprio_encoder" in payload:
            logger.warning("Checkpoint contains unused `proprio_encoder` weights; ignoring.")

        if "state_proj" in payload:
            self.state_proj.load_state_dict(payload["state_proj"], strict=True)
        else:
            logger.warning("Checkpoint has no `state_proj` weights; keeping current `state_proj` params.")

        reinitialized = self._load_camera_head_state(
            self.action_camera_router,
            payload,
            "action_camera_router",
            prefix="action_router",
            layer_index=int(self.action_router_feature_layer_index),
            reinit_incompatible_heads=reinit_incompatible_heads,
        )
        reinitialized = self._load_camera_head_state(
            self.action_camera_critic,
            payload,
            "action_camera_critic",
            prefix="action_critic",
            layer_index=int(self.action_critic_feature_layer_index),
            reinit_incompatible_heads=reinit_incompatible_heads,
        ) or reinitialized

        if optimizer is not None and "optimizer" in payload:
            if reinitialized:
                raise ValueError(
                    "Refusing to restore optimizer state after reinitializing incompatible camera heads."
                )
            optimizer.load_state_dict(payload["optimizer"])
        return payload
