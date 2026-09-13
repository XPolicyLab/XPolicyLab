"""Apply explicit LeRobot v3 compatibility fixes to pinned KinRT source.

The delivery describes a local DataFrame-to-dict fix but does not include its
original patch. This adapter supplies its own reconstructed task-table fix and
a v3 episode reader for custom router-label generation. Neither changes model
or clustering logic. Only known original or exactly patched files are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "e5ed766edaa99b125fb4ae06d2fff2893641007d23ef8a71e6501beb84a9eddc"
SOURCE_PATH = "src/openpi/training/data_loader.py"
ORIGINAL_BLOCK = '''    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
'''
PATCHED_BLOCK = '''    if data_config.prompt_from_task:
        # Reconstructed XPolicyLab fix for the delivered LeRobot v3 task table.
        tasks = dataset_meta.tasks
        if not isinstance(tasks, dict):
            if not hasattr(tasks, "columns") or "task_index" not in tasks.columns:
                raise TypeError("Expected a task mapping or LeRobot v3 task_index DataFrame.")
            task_mapping = {}
            for prompt, row in tasks.iterrows():
                task_index = int(row["task_index"])
                if task_index != row["task_index"] or task_index in task_mapping:
                    raise ValueError("LeRobot task indices must be unique integers.")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("LeRobot task prompts must be non-empty strings in the DataFrame index.")
                task_mapping[task_index] = prompt
            tasks = task_mapping
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks)])
'''
GENERATOR_PATH = "scripts/generate_router_labels.py"
GENERATOR_SHA256 = "392798c12d8d65ccdfe6008fda11c5619f6eba6e6449e5ee88786f77db35678a"
EPISODE_READER = '''# XPolicyLab compatibility: v3 may pack or split episodes across Parquet files.
@lru_cache(maxsize=2)
def _v3_data_dataset(repo_root: Path):
    return pads.dataset(repo_root / "data", format="parquet")


def _read_episode_table(repo_root, data_path_pattern, episode_index, chunks_size, columns):
    read_columns = list(dict.fromkeys([*columns, "index", "episode_index"]))
    if "{chunk_index" in data_path_pattern and "{file_index" in data_path_pattern:
        table = _v3_data_dataset(repo_root).to_table(
            columns=read_columns,
            filter=pads.field("episode_index") == episode_index,
        )
    else:
        path = _episode_path(repo_root, data_path_pattern, episode_index, chunks_size)
        table = pq.read_table(path, columns=read_columns)
    if table.num_rows == 0:
        raise ValueError(f"No frames found for episode {episode_index}.")
    if np.any(_int_column_to_numpy(table, "episode_index") != episode_index):
        raise ValueError(f"Unexpected episode indices while reading episode {episode_index}.")
    table = table.sort_by([("index", "ascending")])
    indices = _int_column_to_numpy(table, "index")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"Duplicate global frame indices in episode {episode_index}.")
    return table.select(columns)


'''
GENERATOR_REPLACEMENTS = (
    ("from pathlib import Path\n", "from functools import lru_cache\nfrom pathlib import Path\n"),
    ("import pyarrow.parquet as pq\n", "import pyarrow.dataset as pads\nimport pyarrow.parquet as pq\n"),
    ("def _fixed_list_column_to_numpy(", EPISODE_READER + "def _fixed_list_column_to_numpy("),
    (
        '''        path = _episode_path(repo_root, data_path_pattern, episode_index, chunks_size)
        columns = [action_column, "index", "episode_index", "frame_index", "task_index"]
        if relative_to_state:
            columns.append(state_column)
        table = pq.read_table(
            path,
            columns=columns,
        )
''',
        '''        columns = [action_column, "index", "episode_index", "frame_index", "task_index"]
        if relative_to_state:
            columns.append(state_column)
        table = _read_episode_table(repo_root, data_path_pattern, episode_index, chunks_size, columns)
''',
    ),
    ('    chunks_size = int(info["chunks_size"])\n', '    chunks_size = int(info.get("chunks_size", 1000))\n'),
    (
        '''    first_path = _episode_path(repo_root, data_path_pattern, episodes[0], chunks_size)
    first_columns = [args.action_column, "index"]
    if args.relative_to_state:
        first_columns.append(args.state_column)
    first_table = pq.read_table(first_path, columns=first_columns)
''',
        '''    first_columns = [args.action_column, "index"]
    if args.relative_to_state:
        first_columns.append(args.state_column)
    first_table = _read_episode_table(repo_root, data_path_pattern, episodes[0], chunks_size, first_columns)
''',
    ),
)
SOURCE_PATCHES = (
    (SOURCE_PATH, SOURCE_SHA256, ((ORIGINAL_BLOCK, PATCHED_BLOCK),)),
    (GENERATOR_PATH, GENERATOR_SHA256, GENERATOR_REPLACEMENTS),
)


def prepare_source(openpi_root: Path, *, check: bool = False, revert: bool = False) -> str:
    updates = []
    all_patched = True
    for relative_path, original_sha256, replacements in SOURCE_PATCHES:
        path = openpi_root / relative_path
        source = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest == original_sha256:
            original = source
            patched = source
            for old, new in replacements:
                if patched.count(old) != 1:
                    raise RuntimeError(f"Missing expected compatibility target in {path}.")
                patched = patched.replace(old, new)
        else:
            original = source
            for old, new in reversed(replacements):
                if original.count(new) != 1:
                    raise RuntimeError(f"Unknown source changes in {path}; no file was changed.")
                original = original.replace(new, old)
            if hashlib.sha256(original.encode("utf-8")).hexdigest() != original_sha256:
                raise RuntimeError(f"Unknown source changes in {path}; no file was changed.")
            patched = source
        if source != patched:
            all_patched = False
        target = original if revert else patched
        if source != target:
            updates.append((path, target))

    # Validate every file before modifying either, preserving unknown user edits.
    if check:
        return "already applied" if all_patched else "ready to apply"
    if not updates:
        return "already restored" if revert else "already applied"
    for path, target in updates:
        path.write_text(target, encoding="utf-8", newline="\n")
    return "restored pinned source" if revert else "applied explicit LeRobot v3 compatibility fixes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("openpi_root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate source without changing it.")
    mode.add_argument("--revert", action="store_true", help="Restore only this helper's exact source change.")
    args = parser.parse_args()
    try:
        status = prepare_source(args.openpi_root.resolve(), check=args.check, revert=args.revert)
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"[KinRT][ERROR] {exc}\n")
    print(f"[KinRT] {status}")


if __name__ == "__main__":
    main()
