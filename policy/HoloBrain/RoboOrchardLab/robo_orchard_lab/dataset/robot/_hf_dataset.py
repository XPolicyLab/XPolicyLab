# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
import warnings
from typing import Type

import fsspec
import pyarrow as pa
from datasets import (
    Features as HFFeatures,
    IterableDataset as HFIterableDataset,
)
from datasets.arrow_dataset import (
    Dataset,
    DatasetInfo,
    InMemoryTable,
    MemoryMappedTable,
    Optional,
    Path,
    PathLike,
    Split,
    concat_tables,
    estimate_dataset_size,
    hf_tqdm,
    is_remote_filesystem,
    is_small_dataset,
    json,
    thread_map,
)
from fsspec import url_to_fs

from robo_orchard_lab.dataset.datatypes.hg_features import RODictDataFeature

__all__ = ["load_from_disk", "add_hf_iterable_cls"]


def load_from_disk(
    dataset_path: PathLike,
    keep_in_memory: Optional[bool] = None,
    storage_options: Optional[dict] = None,
) -> Dataset:
    """A wrapper around `datasets.load_from_disk`.

    Unlike the original `datasets.load_from_disk` which cast the arrow table
    to match the features defined in the dataset info, this wrapper will adapt
    the features to match the arrow table schema. This provides faster loading
    speed.
    """
    import posixpath

    fs: fsspec.AbstractFileSystem
    fs, dataset_path = url_to_fs(dataset_path, **(storage_options or {}))
    import datasets.config as config

    dest_dataset_path = dataset_path
    dataset_dict_json_path = posixpath.join(
        dest_dataset_path,  # type: ignore
        config.DATASETDICT_JSON_FILENAME,  # type: ignore
    )
    dataset_state_json_path = posixpath.join(
        dest_dataset_path,  # type: ignore
        config.DATASET_STATE_JSON_FILENAME,  # type: ignore
    )
    dataset_info_path = posixpath.join(
        dest_dataset_path,  # type: ignore
        config.DATASET_INFO_FILENAME,  # type: ignore
    )

    dataset_dict_is_file = fs.isfile(dataset_dict_json_path)
    dataset_info_is_file = fs.isfile(dataset_info_path)
    dataset_state_is_file = fs.isfile(dataset_state_json_path)
    if not dataset_info_is_file and not dataset_state_is_file:
        if dataset_dict_is_file:
            raise FileNotFoundError(
                f"No such files: '{dataset_info_path}', nor '{dataset_state_json_path}' found. Expected to load a `Dataset` object, but got a `DatasetDict`. Please use either `datasets.load_from_disk` or `DatasetDict.load_from_disk` instead."  # noqa: E501
            )
        raise FileNotFoundError(
            f"No such files: '{dataset_info_path}', nor '{dataset_state_json_path}' found. Expected to load a `Dataset` object but provided path is not a `Dataset`."  # noqa: E501
        )
    if not dataset_info_is_file:
        if dataset_dict_is_file:
            raise FileNotFoundError(
                f"No such file: '{dataset_info_path}' found. Expected to load a `Dataset` object, but got a `DatasetDict`. Please use either `datasets.load_from_disk` or `DatasetDict.load_from_disk` instead."  # noqa: E501
            )
        raise FileNotFoundError(
            f"No such file: '{dataset_info_path}'. Expected to load a `Dataset` object but provided path is not a `Dataset`."  # noqa: E501
        )
    if not dataset_state_is_file:
        if dataset_dict_is_file:
            raise FileNotFoundError(
                f"No such file: '{dataset_state_json_path}' found. Expected to load a `Dataset` object, but got a `DatasetDict`. Please use either `datasets.load_from_disk` or `DatasetDict.load_from_disk` instead."  # noqa: E501
            )
        raise FileNotFoundError(
            f"No such file: '{dataset_state_json_path}'. Expected to load a `Dataset` object but provided path is not a `Dataset`."  # noqa: E501
        )

    # copies file from filesystem if it is remote filesystem to local
    # filesystem and modifies dataset_path to temp directory
    # containing local copies
    if is_remote_filesystem(fs):
        src_dataset_path = dest_dataset_path
        dest_dataset_path = Dataset._build_local_temp_path(src_dataset_path)  # type: ignore
        fs.download(
            src_dataset_path, dest_dataset_path.as_posix(), recursive=True
        )
        dataset_state_json_path = posixpath.join(
            dest_dataset_path, config.DATASET_STATE_JSON_FILENAME
        )
        dataset_info_path = posixpath.join(
            dest_dataset_path, config.DATASET_INFO_FILENAME
        )

    with open(dataset_state_json_path, encoding="utf-8") as state_file:
        state = json.load(state_file)
    with open(dataset_info_path, encoding="utf-8") as dataset_info_file:
        dataset_info = DatasetInfo.from_dict(json.load(dataset_info_file))

    dataset_size = estimate_dataset_size(
        Path(dest_dataset_path, data_file["filename"])  # type: ignore
        for data_file in state["_data_files"]
    )
    keep_in_memory = (
        keep_in_memory
        if keep_in_memory is not None
        else is_small_dataset(dataset_size)
    )
    table_cls = InMemoryTable if keep_in_memory else MemoryMappedTable

    arrow_table = concat_tables(
        thread_map(
            table_cls.from_file,
            [
                posixpath.join(dest_dataset_path, data_file["filename"])
                for data_file in state["_data_files"]
            ],
            tqdm_class=hf_tqdm,
            desc="Loading dataset from disk",
            # set `disable=None` rather than `disable=False` by default
            # to disable progress bar when no TTY attached
            disable=len(state["_data_files"]) <= 16 or None,
        )
    )

    split = state["_split"]
    split = Split(split) if split is not None else split
    assert dataset_info.features is not None, "Dataset info must have features"

    if arrow_table.schema != dataset_info.features.arrow_schema:
        adapted_features = _adapt_features_to_table_schema(
            arrow_table.schema, dataset_info.features
        )
        dataset_info.features = adapted_features

    dataset = Dataset(
        arrow_table=arrow_table,
        info=dataset_info,
        split=split,
        fingerprint=state["_fingerprint"],
    )

    format = {
        "type": state["_format_type"],
        "format_kwargs": state["_format_kwargs"],
        "columns": state["_format_columns"],
        "output_all_columns": state["_output_all_columns"],
    }
    dataset = dataset.with_format(**format)

    return dataset


def _adapt_features_to_table_schema(
    schema: pa.Schema, features: HFFeatures
) -> HFFeatures:
    """Adapt the features to match the table schema.

    This is a workaround to fix the issue when loading `Dataset` object when
    info.feature does not match the arrow table schema exactly. It is a bug in
    `Dataset` implementation, and we provide this wrapper to fix the issue.

    Note:
        This method only handles missing fields. If the field type is changed,
        it will not be handled here!

    """
    # find all field
    existing_fields = set(schema.names)

    # reconstruct features with missing fields filled with null values
    adapted_features = {}
    for field, feature in features.items():
        if field in existing_fields:
            adapted_features[field] = features[field]
            # check if the field type is compatible with the schema
            if features[field].pa_type != schema.field(field).type:
                err_msg = (
                    f"Failed to adapt feature '{field}' to match the "
                    "table schema. Please check the feature definition and "
                    "the table schema to make sure they are compatible, "
                    "or consider repackaging the dataset with the current "
                    "version of the code. "
                )
                if isinstance(feature, RODictDataFeature):
                    if not feature.adapt_for_pa_type(schema.field(field).type):
                        raise TypeError(err_msg)

        else:
            warnings.warn(
                f"Field '{field}' is missing in the table schema. Filling "
                "it with None values. This should be an error instead of "
                "warning, but we use warning here to be more robust to "
                "schema changes. "
            )
            adapted_features[field] = None

    return HFFeatures(adapted_features)


def _safe_add_base(cls: Type, new_base: Type) -> bool:
    if new_base in cls.__mro__:
        return True
    if issubclass(new_base, cls):
        raise TypeError("new_base is a subclass of cls; would create cycle")

    bases = list(cls.__bases__)
    bases = [b for b in bases if b is not new_base]

    # try to insert new_base into the bases in all possible positions
    # to find a valid MRO
    for i in range(len(bases) + 1):
        trial = tuple(bases[:i] + [new_base] + bases[i:])
        try:
            cls.__bases__ = trial
            return True
        except TypeError:
            continue

    try:
        cls.__bases__ = tuple(bases + [new_base])
        return True
    except TypeError:
        return False


def add_hf_iterable_cls(cls, instance: object | None = None):
    """Add HFIterableDataset to the base classes of the given class.

    This is a workaround to make the class compatible with Hugging Face's
    Accelerate library, which checks for the presence of HFIterableDataset
    in the `prepare` method to determine if the dataset is an iterable dataset.
    """

    def _create_combined_class(
        base_cls: Type, new_base: Type
    ) -> Optional[Type]:
        name = f"{base_cls.__name__}With{getattr(new_base, '__name__', 'HF')}"
        # Try (base_cls, new_base) then (new_base, base_cls)
        for order in ((base_cls, new_base), (new_base, base_cls)):
            try:
                combined_cls = type(name, order, {})
                # Ensure the dynamic class is importable/picklable by
                # placing it in the same module as the base class and
                # setting its __module__ accordingly.
                try:
                    import sys

                    combined_cls.__module__ = getattr(
                        base_cls, "__module__", "builtins"
                    )
                    mod = sys.modules.get(combined_cls.__module__)
                    if mod is not None:
                        setattr(mod, combined_cls.__name__, combined_cls)
                except Exception:
                    # Best-effort only; if registration fails, fallback
                    # to returning the class (may not be picklable).
                    pass
                return combined_cls
            except TypeError:
                continue
        return None

    if HFIterableDataset in cls.__mro__:
        return

    # First try to modify class bases in-place (may fail due to MRO)
    if _safe_add_base(cls, HFIterableDataset):
        return

    # If we have an instance available, try to create a per-instance combined
    # subclass that includes both the original class and HFIterableDataset.
    if instance is not None:
        combined_cls = _create_combined_class(cls, HFIterableDataset)
        if combined_cls is not None:
            try:
                instance.__class__ = combined_cls
                return
            except TypeError:
                # Fall through to try the other order if not already tried
                pass

    # If we reach here, adding HFIterableDataset failed. Raise an informative
    # error so callers may fallback to an adapter.
    raise TypeError(
        f"Failed to make {cls} compatible with {HFIterableDataset}: "
        "MRO conflict. Callers should fallback to using an adapter "
        "wrapper instance."
    )
