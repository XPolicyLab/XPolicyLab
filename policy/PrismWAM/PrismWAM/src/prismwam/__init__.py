"""PrismWAM: decoupling semantics and physics in world action models."""

__version__ = "0.1.0"
__all__ = ["load_pretrained"]


def __getattr__(name):
    # Keep `import prismwam` light; torch is only imported when a model is loaded.
    if name == "load_pretrained":
        from prismwam.hub import load_pretrained

        return load_pretrained
    raise AttributeError(f"module 'prismwam' has no attribute {name!r}")
