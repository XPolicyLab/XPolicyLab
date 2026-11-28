"""
ModelConfig - Model configuration class
"""

import os
import glob
from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class ModelConfig:
    """Model configuration class for managing model paths and downloads"""

    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_resource: str = "ModelScope"
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None
    local_model_path: str = None
    skip_download: bool = True

    def download_if_necessary(self, use_usp=False):
        """Download model (if necessary)"""
        if self.path is None:
            if self.model_id is None:
                raise ValueError(
                    'No valid model files. Please use `ModelConfig(path="xxx")` '
                    'or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`.'
                )

            # Skip if not in rank 0
            if use_usp:
                import torch.distributed as dist

                _skip_download = self.skip_download or dist.get_rank() != 0
            else:
                _skip_download = self.skip_download

            # Check whether the origin path is a folder
            if self.origin_file_pattern is None or self.origin_file_pattern == "":
                self.origin_file_pattern = ""
                is_folder = True
            elif isinstance(
                self.origin_file_pattern, str
            ) and self.origin_file_pattern.endswith("/"):
                is_folder = True
            else:
                is_folder = False

            # Wan-initialized training requires an explicit local resource root.
            # Complete-checkpoint serving does not call this loader.
            self.local_model_path = os.environ.get("WALLX_MODEL_ROOT")
            if self.local_model_path is None:
                raise ValueError(
                    "Set WALLX_MODEL_ROOT to the local Wan resource directory "
                    "or provide ModelConfig(path=...)."
                )
            _skip_download = True  # Always skip download during training

            # Let rank 1, 2, ... wait for rank 0
            if use_usp:
                import torch.distributed as dist

                dist.barrier(device_ids=[dist.get_rank()])

            # Return downloaded files
            if is_folder:
                self.path = os.path.join(
                    self.local_model_path, self.model_id, self.origin_file_pattern
                )
            else:
                self.path = glob.glob(
                    os.path.join(
                        self.local_model_path, self.model_id, self.origin_file_pattern
                    )
                )
            if isinstance(self.path, list) and len(self.path) == 1:
                self.path = self.path[0]
