from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from transformers import AutoTokenizer

import torch
from copy import deepcopy
import numpy as np
from lerobot.utils.constants import (
    OBS_STATE, 
    ACTION, 
    OBS_IMAGES, 
    OBS_LANGUAGE_TOKENS, 
    OBS_LANGUAGE_ATTENTION_MASK, 
)
from lerobot.transforms.core import DataTransformFn, DataDict
import torch.nn.functional as F

def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (sequence_length x features_dimension)
    or (features_dimension)
    For 1D tensor: (features_dimension) -> pad on the right
    For 2D tensor: (batch, features_dimension) -> pad on the right for each batch
    """
    if vector.shape[-1] >= new_dim:
        return vector
    # Calculate padding needed
    pad_size = new_dim - vector.shape[-1]
    return F.pad(vector, (0, pad_size))

@DataTransformFn.register_subclass("pi05_gemma_tokenizer")
@dataclass
class PI05GemmaTokenizerTransformFn(DataTransformFn):
    pretrained_model_name_or_path: str = 'google/paligemma-3b-pt-224'
    max_length: int = 200
    max_state_dim: int = 32
    task_key: str = "task"  
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    # tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)

    def __call__(self, data: DataDict) -> DataDict: 

        state = data[OBS_STATE]
        state = deepcopy(state)
        # Prepare state (pad to max_state_dim)
        state = pad_vector(state, self.max_state_dim)
        # State should already be normalized to [-1, 1] by the NormalizerTransformFn that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        # TODO: Since we use mean-std normalization for the state, so we assume the normalized states
        # follow a standard normal distribution N(0, 1).
        # We thus divide the state by 3 so that ~99.74% of the values (±3σ) fall
        # within the range [-1, 1], which matches the discretization range used
        # by the tokenizer.
        state_np = state.cpu().numpy() / 3
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        task = data[self.task_key]
  
        cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, discretized_states))
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "

        lang_inputs = self.tokenizer(
            full_prompt, 
            max_length=self.max_length, 
            padding_side=self.padding_side, 
            padding=self.padding, 
            truncation=self.truncation, 
        )

        data[OBS_LANGUAGE_TOKENS] = torch.tensor(lang_inputs.input_ids)
        data[OBS_LANGUAGE_ATTENTION_MASK] = torch.tensor(lang_inputs.attention_mask)

        return data
    

@DataTransformFn.register_subclass("unify_pi05_inputs")
@dataclass
class UnifyPI05InputsTransformFn(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict: 
        data = {
            OBS_STATE: data[OBS_STATE], 
            ACTION: data[ACTION], 
            f"{OBS_IMAGES}.image0": data[f"{OBS_IMAGES}.image0"], 
            f"{OBS_IMAGES}.image1": data[f"{OBS_IMAGES}.image1"], 
            f"{OBS_IMAGES}.image2": data[f"{OBS_IMAGES}.image2"], 
            f"{OBS_IMAGES}.image0_mask": data[f"{OBS_IMAGES}.image0_mask"], 
            f"{OBS_IMAGES}.image1_mask": data[f"{OBS_IMAGES}.image1_mask"], 
            f"{OBS_IMAGES}.image2_mask": data[f"{OBS_IMAGES}.image2_mask"], 
            # "task": data["task"], 
            OBS_LANGUAGE_TOKENS: data[OBS_LANGUAGE_TOKENS], 
            OBS_LANGUAGE_ATTENTION_MASK: data[OBS_LANGUAGE_ATTENTION_MASK], 
        }
        return data


if __name__ == "__main__":
    sample = {
        "task": "This is a test sample.", 
        "observation.state": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    }
    tokenizer = PI05GemmaTokenizerTransformFn()
    output = tokenizer(sample)
    print(output)
