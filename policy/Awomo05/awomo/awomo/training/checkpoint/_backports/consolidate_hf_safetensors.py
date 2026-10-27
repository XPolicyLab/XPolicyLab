# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# taken and edited from
# https://github.com/pytorch/pytorch/blob/2fde10d9148d2dae9dc168ccd647de7f3cafea6b/torch/distributed/checkpoint/_consolidate_hf_safetensors.py

import concurrent.futures
import glob
import json
import logging
import math
import mmap
import os
import shutil
import struct
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import distributed as dist

from awomo.awomo.training.checkpoint._backports.hf_utils import (
    DATA_OFFSETS_KEY,
    DEFAULT_EXTRA_METADATA_KEY,
    DTYPE_KEY,
    SAVED_OFFSETS_KEY,
    SHAPE_KEY,
    SUFFIX,
    _gen_file_name,
    _get_dcp_custom_metadata,
    _get_safetensors_file_metadata,
    _metadata_fn,
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class _FqnData:
    """Dataclass to store information about a tensor (identified by its fully qualified name)."""

    offset_in_file: int = 0
    shape_in_file: list[int] = field(default_factory=list)
    dtype_size: int = 0
    dtype_str: str = ""


@dataclass
class _OutputFileData:
    """Dataclass to store information about an output safetensors file."""

    metadata_size: int = 0
    fqn_data: dict[str, _FqnData] = field(default_factory=dict)


@dataclass
class _InputFileData:
    """Dataclass to store information about an input safetensors file."""

    metadata_size: int = 0
    metadata: Any = None


def _parse_input_metadata(
    input_files_data: dict[str, _InputFileData],
    output_files_data: dict[str, _OutputFileData],
) -> None:

    from safetensors.torch import _getdtype  # type: ignore[import]

    fqn_to_size_mapping: dict[str, tuple[list[int], str]] = {}

    for file_data in input_files_data.values():
        safetensors_metadata = file_data.metadata
        dcp_sharding_info = _get_dcp_custom_metadata(safetensors_metadata)
        if not dcp_sharding_info:
            raise ValueError(
                "No DCP custom metadata found in safetensors file. The file must be saved with DCP to be consolidated."
            )

        for key, val in safetensors_metadata.items():
            if key == DEFAULT_EXTRA_METADATA_KEY:
                continue

            sizes = val[SHAPE_KEY]
            offsets = dcp_sharding_info[key][SAVED_OFFSETS_KEY]

            if key not in fqn_to_size_mapping:
                cur_size = [size + offset for size, offset in zip(sizes, offsets)]
                fqn_to_size_mapping[key] = (cur_size, val[DTYPE_KEY])
            else:
                cur_size = fqn_to_size_mapping[key][0]
                for i in range(len(sizes)):
                    cur_size[i] = max(cur_size[i], sizes[i] + offsets[i])

    for fqn, tensor_info in fqn_to_size_mapping.items():
        tensor_size = tensor_info[0]
        dtype_str = tensor_info[1]
        for output_data in output_files_data.values():
            if fqn in output_data.fqn_data:
                dtype = _getdtype(dtype_str)
                try:
                    dtype_size = torch.finfo(dtype).bits // 8
                except TypeError:
                    dtype_size = torch.tensor([], dtype=dtype).element_size()
                output_data.fqn_data[fqn] = _FqnData(
                    shape_in_file=tensor_size,
                    dtype_size=dtype_size,
                    dtype_str=dtype_str,
                )


def _write_metadata(
    output_files_data: dict[str, _OutputFileData],
) -> None:
    for file_path, output_data in output_files_data.items():
        with open(file_path, "wb") as f:
            metadata = {}
            curr_offset = 0

            for fqn, fqn_data in output_data.fqn_data.items():
                end_offset = curr_offset + math.prod(fqn_data.shape_in_file) * fqn_data.dtype_size

                metadata[fqn] = {
                    SHAPE_KEY: fqn_data.shape_in_file,
                    DTYPE_KEY: fqn_data.dtype_str,
                    DATA_OFFSETS_KEY: [
                        curr_offset,
                        end_offset,
                    ],
                }
                fqn_data.offset_in_file = curr_offset

                curr_offset = end_offset

            json_metadata = json.dumps(metadata)
            json_bytes = json_metadata.encode("utf-8")

            size_in_bytes = len(json_bytes)
            header_len = struct.pack("<Q", size_in_bytes)

            f.write(header_len)
            f.write(json_bytes)

            output_data.metadata_size = f.tell()


def _read_tensor_data_mmap(
    file_path: str,
    start_offset: int,
    end_offset: int,
    metadata_size: int,
) -> bytes:
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            absolute_start = metadata_size + start_offset
            absolute_end = metadata_size + end_offset
            return bytes(mm[absolute_start:absolute_end])


def _process_output_file(
    output_file: str,
    output_data: _OutputFileData,
    input_files_data: dict[str, _InputFileData],
) -> None:

    sorted_tensors = sorted(output_data.fqn_data.items(), key=lambda x: x[1].offset_in_file)

    if not os.path.exists(output_file):
        raise FileNotFoundError(
            f"Expected output file {output_file} to exist (header written in _write_metadata()), but it does not."
        )

    with open(output_file, "ab") as output_stream:
        for tensor_fqn, tensor_fqn_data in sorted_tensors:
            full_tensor_mv = memoryview(
                bytearray(math.prod(tensor_fqn_data.shape_in_file) * tensor_fqn_data.dtype_size)
            )

            for safetensors_file in input_files_data.keys():
                file_metadata = input_files_data[safetensors_file].metadata
                input_metadata_size = input_files_data[safetensors_file].metadata_size

                if tensor_fqn not in file_metadata.keys():
                    continue

                metadata = file_metadata[tensor_fqn]

                data_offsets = metadata[DATA_OFFSETS_KEY]

                data_to_write = _read_tensor_data_mmap(
                    safetensors_file,
                    data_offsets[0],
                    data_offsets[1],
                    input_metadata_size,
                )

                fqn_custom_metadata = _get_dcp_custom_metadata(file_metadata)[tensor_fqn]  # type: ignore[index]
                offsets_of_tensor_being_read = fqn_custom_metadata[SAVED_OFFSETS_KEY]  # type: ignore[index]

                _write_sub_tensor_to_file_optimized(
                    full_tensor_mv,
                    data_to_write,
                    tensor_fqn_data.dtype_size,
                    tensor_fqn_data.shape_in_file,
                    offsets_of_tensor_being_read,
                    metadata[SHAPE_KEY],
                )

            output_stream.write(full_tensor_mv)


def _write_data(
    input_files_data: dict[str, _InputFileData],
    output_files_data: dict[str, _OutputFileData],
    num_threads: int = 1,
) -> None:
    """
    Write tensor data from input files to the output files using memory mapping.

    This function reads tensor data from each input file and writes it to the appropriate
    position in the output files based on the tensor's offsets. When num_threads > 1,
    the work is split across threads with each thread handling a different output file.

    Args:
        input_files_data: Dictionary mapping input file paths to their metadata
        output_files_data: Dictionary mapping output file paths to their metadata
        num_threads: Number of threads to use for parallel processing
    """
    if num_threads <= 1 or len(output_files_data) <= 1:
        for output_file, output_data in output_files_data.items():
            _process_output_file(output_file, output_data, input_files_data)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_threads, len(output_files_data))) as executor:
            futures = []
            for output_file, output_data in output_files_data.items():
                futures.append(
                    executor.submit(
                        _process_output_file,
                        output_file,
                        output_data,
                        input_files_data,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing output file: {e}")
                    raise


def _write_sub_tensor_to_file_optimized(
    full_tensor_mv: memoryview,
    sub_tensor_bytes: bytes,
    element_size: int,
    tensor_shape: list[int],
    sub_tensor_offsets: list[int],
    sub_tensor_shape: list[int],
) -> None:
    if len(tensor_shape) == 0 and len(sub_tensor_shape) == 0:
        bytes_to_write = min(len(sub_tensor_bytes), element_size)
        if bytes_to_write:
            full_tensor_mv[:bytes_to_write] = sub_tensor_bytes[:bytes_to_write]
        return

    if not tensor_shape or not sub_tensor_shape:
        return

    tensor_strides = [1]
    for i in range(len(tensor_shape) - 1, 0, -1):
        tensor_strides.insert(0, tensor_strides[0] * tensor_shape[i])

    sub_tensor_strides = [1]
    for i in range(len(sub_tensor_shape) - 1, 0, -1):
        sub_tensor_strides.insert(0, sub_tensor_strides[0] * sub_tensor_shape[i])

    total_elements = math.prod(sub_tensor_shape)

    elements_written = 0
    while elements_written < total_elements:
        temp_idx = elements_written
        indices = []
        for dim_size in reversed(sub_tensor_shape):
            indices.append(temp_idx % dim_size)
            temp_idx //= dim_size
        indices.reverse()

        max_contiguous = _calculate_max_contiguous_elements(indices, sub_tensor_shape, tensor_shape)

        src_pos = sum(idx * stride for idx, stride in zip(indices, sub_tensor_strides))
        src_byte_offset = src_pos * element_size

        dest_indices = [idx + offset for idx, offset in zip(indices, sub_tensor_offsets)]
        dest_pos = sum(idx * stride for idx, stride in zip(dest_indices, tensor_strides))
        dest_byte_offset = dest_pos * element_size

        bytes_to_write = max_contiguous * element_size
        chunk_data = sub_tensor_bytes[src_byte_offset : src_byte_offset + bytes_to_write]
        full_tensor_mv[dest_byte_offset : dest_byte_offset + bytes_to_write] = chunk_data

        elements_written += max_contiguous


def _calculate_max_contiguous_elements(
    indices: list[int],
    sub_tensor_shape: list[int],
    tensor_shape: list[int],
) -> int:
    if not indices or not sub_tensor_shape or not tensor_shape:
        raise ValueError("Input lists cannot be empty")

    if not (len(indices) == len(sub_tensor_shape) == len(tensor_shape)):
        raise ValueError(
            f"All input lists must have the same length. Got indices: {len(indices)}, "
            f"sub_tensor_shape: {len(sub_tensor_shape)}, tensor_shape: {len(tensor_shape)}"
        )

    for i, (idx, sub_dim) in enumerate(zip(indices, sub_tensor_shape)):
        if idx >= sub_dim:
            raise ValueError(f"Index {idx} at dimension {i} is out of bounds for sub-tensor shape {sub_tensor_shape}")

    for i, (sub_dim, tensor_dim) in enumerate(zip(sub_tensor_shape, tensor_shape)):
        if sub_dim > tensor_dim:
            raise ValueError(f"Sub-tensor dimension {sub_dim} at position {i} exceeds tensor dimension {tensor_dim}")

    max_contiguous = sub_tensor_shape[-1] - indices[-1]


    if len(sub_tensor_shape) >= 2:
        if indices[-1] == 0:
            rows_remaining = sub_tensor_shape[-2] - indices[-2]

            if sub_tensor_shape[-1] == tensor_shape[-1]:
                max_contiguous = rows_remaining * sub_tensor_shape[-1]

            if len(sub_tensor_shape) >= 3 and indices[-2] == 0:
                remaining_in_dim = sub_tensor_shape[-3] - indices[-3]
                if sub_tensor_shape[-1] == tensor_shape[-1] and sub_tensor_shape[-2] == tensor_shape[-2]:
                    max_contiguous = remaining_in_dim * sub_tensor_shape[-2] * sub_tensor_shape[-1]

    return max_contiguous


def _write_overall_metadata_file(
    output_dir: str,
    output_files_data: dict[str, _OutputFileData],
) -> None:
    total_size = 0
    weight_map = {}
    for output_path, value in output_files_data.items():
        for fqn, fqn_data in value.fqn_data.items():
            total_size += math.prod(fqn_data.shape_in_file) * fqn_data.dtype_size
            weight_map[fqn] = os.path.basename(output_path)

    metadata_to_write: dict[str, Any] = {}
    metadata_to_write["metadata"] = {"total_size": total_size}
    metadata_to_write["weight_map"] = weight_map

    metadata_path = os.path.join(output_dir, f"{_metadata_fn}")
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata_to_write, metadata_file, indent=2)


def _write_overall_metadata_file_from_shards(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
) -> None:
    from safetensors.torch import _getdtype  # type: ignore[import]

    safetensors_files = glob.glob(os.path.join(input_dir, f"*{SUFFIX}"))

    input_files_data: dict[str, _InputFileData] = {}
    for input_file in safetensors_files:
        with open(input_file, "rb") as f:
            metadata, metadata_size = _get_safetensors_file_metadata(f)
            input_files_data[input_file] = _InputFileData(
                metadata_size=metadata_size,
                metadata=metadata,
            )

    fqn_to_size_mapping: dict[str, tuple[list[int], str]] = {}
    for file_data in input_files_data.values():
        safetensors_metadata = file_data.metadata
        dcp_sharding_info = _get_dcp_custom_metadata(safetensors_metadata)
        if not dcp_sharding_info:
            raise ValueError(
                "No DCP custom metadata found in safetensors file. The file must be saved with DCP to be consolidated."
            )

        for key, val in safetensors_metadata.items():
            if key == DEFAULT_EXTRA_METADATA_KEY:
                continue

            sizes = val[SHAPE_KEY]
            offsets = dcp_sharding_info[key][SAVED_OFFSETS_KEY]

            if key not in fqn_to_size_mapping:
                cur_size = [size + offset for size, offset in zip(sizes, offsets)]
                fqn_to_size_mapping[key] = (cur_size, val[DTYPE_KEY])
            else:
                cur_size = fqn_to_size_mapping[key][0]
                for i in range(len(sizes)):
                    cur_size[i] = max(cur_size[i], sizes[i] + offsets[i])

    max_index = max(fqn_to_index_mapping.values())
    total_size = 0
    weight_map = {}

    for fqn, (tensor_shape, dtype_str) in fqn_to_size_mapping.items():
        dtype = _getdtype(dtype_str)
        try:
            dtype_size = torch.finfo(dtype).bits // 8
        except TypeError:
            dtype_size = torch.tensor([], dtype=dtype).element_size()

        total_size += math.prod(tensor_shape) * dtype_size

        idx = fqn_to_index_mapping[fqn]
        weight_map[fqn] = _gen_file_name(idx, max_index)

    metadata_to_write: dict[str, Any] = {}
    metadata_to_write["metadata"] = {"total_size": total_size}
    metadata_to_write["weight_map"] = weight_map

    metadata_path = os.path.join(output_dir, _metadata_fn)
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata_to_write, metadata_file, indent=2)


def _consolidate_safetensors_files(
    input_dir: str,
    output_dir: str,
    fqn_to_file_mapping: dict[str, str],
    num_threads: int,
    use_staging: bool = False,
    staging_dir: Optional[str] = None,
) -> dict[str, _OutputFileData]:
    output_files_data: dict[str, _OutputFileData] = {}
    for fqn, filename in fqn_to_file_mapping.items():
        output_path = os.path.join(output_dir, filename)

        if output_path not in output_files_data:
            output_files_data[output_path] = _OutputFileData(fqn_data={fqn: _FqnData()})
        else:
            output_files_data[output_path].fqn_data[fqn] = _FqnData()

    safetensors_files = glob.glob(os.path.join(input_dir, f"*{SUFFIX}"))

    input_files_data: dict[str, _InputFileData] = {}
    for safetensor_file in safetensors_files:
        with open(safetensor_file, "rb") as f:
            metadata, size = _get_safetensors_file_metadata(f)
            input_files_data[safetensor_file] = _InputFileData(metadata_size=size, metadata=metadata)

    if use_staging:
        if staging_dir is not None:
            os.makedirs(staging_dir, exist_ok=True)
            temp_dir = tempfile.mkdtemp(prefix="safetensors_consolidate_", dir=staging_dir)
        else:
            temp_dir = tempfile.mkdtemp(prefix="safetensors_consolidate_")

        try:
            temp_output_files_data: dict[str, _OutputFileData] = {}
            temp_to_final_mapping: dict[str, str] = {}

            for final_path, file_data in output_files_data.items():
                filename = os.path.basename(final_path)
                temp_path = os.path.join(temp_dir, filename)
                temp_output_files_data[temp_path] = file_data
                temp_to_final_mapping[temp_path] = final_path

            _parse_input_metadata(input_files_data, temp_output_files_data)

            _write_metadata(temp_output_files_data)

            _write_data(input_files_data, temp_output_files_data, num_threads)

            for temp_path, final_path in temp_to_final_mapping.items():
                shutil.copy2(temp_path, final_path)
                logger.info("Copied %s to %s", temp_path, final_path)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        _parse_input_metadata(input_files_data, output_files_data)

        _write_metadata(output_files_data)

        _write_data(input_files_data, output_files_data, num_threads)

    return output_files_data


def consolidate_safetensors_files(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 1,
    use_staging: bool = False,
    staging_dir: Optional[str] = None,
) -> None:
    """
    Main function to consolidate sharded safetensors files into one or more output files.

    This function orchestrates the entire consolidation process:
    1. Sets up the output file structure based on the fqn_to_index_mapping
    2. Finds all safetensors files in the input directory
    3. Parses metadata from all input files
    4. Writes metadata to the output files
    5. Writes tensor data from input files to output files
    6. Writes overall model.index.safetensors.json file with weight map

    Args:
        input_dir: Directory containing sharded safetensors files
        output_dir: Directory where consolidated files will be written
        fqn_to_index_mapping: Optional mapping of tensor names to output file indices.
                             If None, all tensors will be consolidated into a single file.
        num_threads: Number of threads to use for parallel processing of saving data to output files.
        use_staging: If True, write to a staging directory first then copy to output_dir.
                    Required for remote storage systems that don't support
                    direct-append or non-sequential writes. Default is False.
        staging_dir: Optional directory for staging files during consolidation. If provided,
                    temporary files will be created in this directory instead of the system temp.
                    Only used when use_staging=True. Useful when system temp has limited space.
    """
    start_time = time.time()
    logger.info(
        "Consolidating safetensors files from %s to %s. Beginning at time %f",
        input_dir,
        output_dir,
        start_time,
    )

    max_index = max(fqn_to_index_mapping.values())
    fqn_to_file_mapping = {fqn: _gen_file_name(idx, max_index) for fqn, idx in fqn_to_index_mapping.items()}

    output_files_data = _consolidate_safetensors_files(
        input_dir, output_dir, fqn_to_file_mapping, num_threads, use_staging, staging_dir
    )

    _write_overall_metadata_file(output_dir, output_files_data)

    logger.info("Done consolidating. Took %.2f secs.", time.time() - start_time)


def consolidate_safetensors_files_on_every_rank(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
    use_staging: bool = False,
    staging_dir: Optional[str] = None,
) -> None:

    start_time = time.time()
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
    else:
        rank = 0
        world_size = 1
        logger.warning("Distributed environment not initialized. Running in single process mode.")
    logger.info(
        "Rank %d/%d: Consolidating safetensors files from %s to %s",
        rank,
        world_size,
        input_dir,
        output_dir,
    )

    unique_indices = set(fqn_to_index_mapping.values())

    indices_for_this_rank = []
    for idx in unique_indices:
        if idx % world_size == rank:
            indices_for_this_rank.append(idx)

    logger.info(
        "Rank %d: Assigned %d output files out of %d total files",
        rank,
        len(indices_for_this_rank),
        len(unique_indices),
    )

    filtered_mapping = {fqn: idx for fqn, idx in fqn_to_index_mapping.items() if idx in indices_for_this_rank}

    if filtered_mapping:
        max_index = max(unique_indices)
        filtered_filename_mapping = {}
        for fqn, idx in filtered_mapping.items():
            filename = _gen_file_name(idx, max_index)
            filtered_filename_mapping[fqn] = filename

        _consolidate_safetensors_files(
            input_dir=input_dir,
            output_dir=output_dir,
            fqn_to_file_mapping=filtered_filename_mapping,
            num_threads=num_threads,
            use_staging=use_staging,
            staging_dir=staging_dir,
        )

    if rank == 0:
        _write_overall_metadata_file_from_shards(input_dir, output_dir, fqn_to_index_mapping)

    logger.info(
        "Rank %d: Done consolidating. Processed %d unique indices in %.2f secs.",
        rank,
        len(indices_for_this_rank),
        time.time() - start_time,
    )

    if dist.is_available() and dist.is_initialized():
        logger.info("Rank %d: Waiting for all ranks to complete...", rank)
        dist.barrier()
        logger.info("Rank %d: All ranks have completed.", rank)
        if rank == 0:
            logger.info("Total time taken: %.2f secs.", time.time() - start_time)
