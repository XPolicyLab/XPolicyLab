"""RLINF_EXT_MODULE entry point, executed in the driver and each Ray worker."""

from .core.extensions import policy_extensions, simulator_extensions


def register():
    from rlinf.models import register_model
    from .core.factory import get_model

    register_model("xpolicylab", get_model, category="embodied", force=True)
    for extension in (*policy_extensions().values(), *simulator_extensions().values()):
        if hasattr(extension, "register"):
            extension.register()
