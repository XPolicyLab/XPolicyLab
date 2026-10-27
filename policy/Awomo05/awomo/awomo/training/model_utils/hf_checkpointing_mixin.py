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

"""HuggingFace-compatible checkpointing mixin for NeMo Automodel."""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass

    pass

logger = logging.getLogger(__name__)


class HFCheckpointingMixin:
    """Mixin providing HF-compatible API using NeMo's checkpointing infrastructure."""

    def save_pretrained(
        self,
        save_directory: str,
        checkpointer: Optional["Checkpointer"] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ) -> None:
        if checkpointer is None:
            raise ValueError("No checkpointer provided. Please pass the `checkpointer` argument.")

        checkpointer.save_model(
            model=self,
            weights_path=save_directory,
            peft_config=kwargs.get("peft_config", None),
            tokenizer=tokenizer,
        )
