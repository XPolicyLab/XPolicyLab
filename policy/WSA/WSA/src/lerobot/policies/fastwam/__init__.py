from .configuration_fastwam import FastWAMConfig, FastWAMDatasetConfig

__all__ = ["FastWAMConfig", "FastWAMDatasetConfig", "FastWAMPolicy"]


def __getattr__(name: str):
    if name == "FastWAMPolicy":
        from .modeling_fastwam import FastWAMPolicy

        return FastWAMPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
