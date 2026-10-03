from .modeling_giga_brain_0 import GigaBrain0Policy


_LAZY_EXPORTS = {
    'Gemma3VLMModel': ('.gemma3_vlm', 'Gemma3VLMModel'),
    'Gemma3WithExpertModel': ('.gemma3_with_expert', 'Gemma3WithExpertModel'),
    'Qwen2_5VLVLMModel': ('.qwen2_5_vl_vlm', 'Qwen2_5VLVLMModel'),
    'Qwen2_5VLWithExpertModel': ('.qwen2_5_vl_with_expert', 'Qwen2_5VLWithExpertModel'),
    'Qwen3_5VLMModel': ('.qwen3_5_vlm', 'Qwen3_5VLMModel'),
    'Qwen3_5WithExpertModel': ('.qwen3_5_with_expert', 'Qwen3_5WithExpertModel'),
    'Qwen3VLVLMModel': ('.qwen3_vl_vlm', 'Qwen3VLVLMModel'),
    'Qwen3VLWithExpertModel': ('.qwen3_vl_with_expert', 'Qwen3VLWithExpertModel'),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    import importlib

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = ['GigaBrain0Policy']
