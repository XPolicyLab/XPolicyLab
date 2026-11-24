"""Expose flat checkpoint metadata under the filenames expected by Transformers."""
from functools import lru_cache
from pathlib import Path
import tempfile

PROCESSOR_FILES = {
    't5': ('tokenizer.json', 'tokenizer_config.json'),
    'vlm': ('config.json', 'tokenizer.json', 'tokenizer_config.json',
            'preprocessor_config.json', 'video_preprocessor_config.json', 'chat_template.jinja'),
}


@lru_cache(maxsize=8)
def _processor_directory(root, component):
    temporary = tempfile.TemporaryDirectory(prefix=f'wsp_{component}_')
    for name in PROCESSOR_FILES[component]:
        (Path(temporary.name) / name).write_bytes((Path(root) / f'{component}_{name}').read_bytes())
    return temporary


def processor_directory(root, component):
    return _processor_directory(str(Path(root).resolve()), component).name
