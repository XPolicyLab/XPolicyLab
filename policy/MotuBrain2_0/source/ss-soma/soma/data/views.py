"""Map packed camera names to the model's learned view identities."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def get_view_embed_ids(
    view_names: Sequence[str],
    model_config: Mapping[str, Any],
    aliases: Mapping[str, str] | None = None,
) -> torch.Tensor | None:
    """Return one ID per packed view, or None when embedding is disabled.

    Aliases translate observation keys to training camera names without
    changing the order of the video tokens or the checkpoint's embedding rows.
    """
    names = model_config.get("view_embed_names")
    default_mode = "none" if names is None else "additive"
    if model_config.get("view_embed_mode", default_mode) == "none":
        return None
    if names is None:
        raise ValueError("additive view embedding requires view_embed_names")
    name_to_id = {name: index for index, name in enumerate(names)}
    ids = []
    for name in view_names:
        camera = aliases.get(name, name) if aliases is not None else name
        if camera not in name_to_id:
            raise ValueError(f"View {camera!r} is missing from view_embed_names {list(names)!r}")
        ids.append(name_to_id[camera])
    return torch.tensor(ids, dtype=torch.int32)
