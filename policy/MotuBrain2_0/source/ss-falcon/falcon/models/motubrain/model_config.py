"""Configuration for the Motubrain embodied-action model."""

from falcon.config import LazyConfig
from falcon.models.motu_3_2_gva.model_config import GVAConfig


class MotubrainConfig(GVAConfig):
    """GVA configuration specialized for deterministic actions."""

    action_dim: int
    action_mask_concat: bool
    bridge_layer_range: tuple[int, int] | None
    v2a: bool
    action_loss_weight: float
    multiview_rope_mode: str
    view_embed_names: list[str] | None
    view_embed_mode: str
    action_conditions: dict[str, dict]

    def __init__(self, cfg: LazyConfig):
        super().__init__(cfg)
        self.action_dim = cfg.model.get("action_dim", 55)
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        self.action_mask_concat = cfg.model.get(
            "action_mask_concat",
            False,
        )
        if not isinstance(self.action_mask_concat, bool):
            raise ValueError(
                "action_mask_concat must be a boolean, got "
                f"{self.action_mask_concat!r}"
            )
        self.a_in_channels = self.action_dim
        self.a_embed_in_channels = self.action_dim * (
            2 if self.action_mask_concat else 1
        )

        bridge_layer_range = cfg.model.get("bridge_layer_range", None)
        self.bridge_layer_range = None if bridge_layer_range is None else tuple(bridge_layer_range)
        self.v2a = cfg.model.get("v2a", False)
        if not isinstance(self.v2a, bool):
            raise ValueError(f"v2a must be a boolean, got {self.v2a!r}")
        self.action_conditions = dict(sorted(cfg.model.get("action_conditions", {}).items()))
        if self.action_conditions and not self.v2a:
            raise ValueError("action conditioning requires v2a=True")
        for name, condition in self.action_conditions.items():
            if not name or "." in name:
                raise ValueError("action condition names must be nonempty and contain no dots")
            layers = tuple(condition["layers"])
            if (
                not layers
                or len(set(layers)) != len(layers)
                or any(not isinstance(layer, int) or not 1 <= layer <= self.num_layers for layer in layers)
            ):
                raise ValueError("action condition layers must be unique one-based model layer numbers")
            if condition["feature_dim"] <= 0:
                raise ValueError("action condition feature_dim must be positive")
        multiview_rope_mode = cfg.model.get("multiview_rope_mode", "hw_offset")
        if multiview_rope_mode not in ("hw_offset", "per_view"):
            raise ValueError("multiview_rope_mode must be 'hw_offset' or 'per_view', " f"got {multiview_rope_mode!r}")
        self.multiview_rope_mode = multiview_rope_mode
        names = cfg.model.get("view_embed_names", None)
        self.view_embed_names = None if names is None else list(names)
        if names is not None:
            if isinstance(names, str) or not names or any(not isinstance(name, str) or not name for name in names):
                raise ValueError("view_embed_names must be a non-empty list of camera names")
            if len(set(names)) != len(names):
                raise ValueError("view_embed_names must not contain duplicates")
        default_mode = "none" if names is None else "additive"
        self.view_embed_mode = cfg.model.get("view_embed_mode", default_mode)
        if self.view_embed_mode not in ("none", "additive"):
            raise ValueError("view_embed_mode must be 'none' or 'additive'")
        if self.view_embed_mode == "additive" and names is None:
            raise ValueError("view_embed_mode='additive' requires view_embed_names")
        self.action_loss_weight = float(cfg.model.get("action_loss_weight", 1.0))
        if self.action_loss_weight < 0:
            raise ValueError("action_loss_weight must be non-negative, got " f"{self.action_loss_weight}")
