"""ME_U0 model package."""

def _install_tied_weights_compat():
    """Monkey-patch PreTrainedModel.get_expanded_tied_weights_keys to accept
    old-style list _tied_weights_keys (pre-transformers 5.0 format).

    transformers >= 5.0 changed _tied_weights_keys from list to dict.  Custom
    models loaded via trust_remote_code may still use the old format, which
    crashes in post_init().  This patch converts lists on the fly.
    """
    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "get_expanded_tied_weights_keys"):
        return  # Only needed for transformers >= 5.0

    _original = PreTrainedModel.get_expanded_tied_weights_keys

    def _patched(self, all_submodels=False):
        keys = getattr(self, "_tied_weights_keys", None)
        if isinstance(keys, list):
            new_keys = {}
            for k in keys:
                if "lm_head" in k:
                    new_keys[k] = k.replace("lm_head", "model.embed_tokens")
                else:
                    new_keys[k] = k
            # Update both instance and class so the fix sticks
            type(self)._tied_weights_keys = new_keys
        return _original(self, all_submodels=all_submodels)

    PreTrainedModel.get_expanded_tied_weights_keys = _patched


# Apply the compat patch at import time
_install_tied_weights_compat()
