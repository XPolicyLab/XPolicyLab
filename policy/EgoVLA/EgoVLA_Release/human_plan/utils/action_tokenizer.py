"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase, PretrainedConfig
from human_plan.utils.quantile import get_class_label, get_quantile
import pickle

class ActionTokenizer:
  def __init__(
    self,
    tokenizer: PreTrainedTokenizerBase,
    # config: PretrainedConfig
    model_args
  ) -> None:
    self.tokenizer = tokenizer
    # self.num_bins = config.num_action_bins
    self.num_bins = model_args.num_action_bins

    # just add a random additional padding in case something stupid happens
    random_additional_padding = 10
    self.invalid_token_idx = int(
        self.tokenizer.vocab_size - self.num_bins - 1 - random_additional_padding
    )
    random_additional_padding_input = 11
    self.input_placeholder_token_idx = int(
        self.tokenizer.vocab_size - self.num_bins - 1 - random_additional_padding_input
    )

    self.input_placeholder_start_token_idx = self.input_placeholder_token_idx

    self.multiplier = 1
    if model_args.sep_query_token:
      self.multiplier = 3
    self.input_placeholder_end_token_idx = self.input_placeholder_start_token_idx - \
      model_args.predict_future_step * 2 * self.multiplier

  def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
    raise NotImplementedError

  def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
    raise NotImplementedError
  
  @property
  def vocab_size(self) -> int:
    return self.num_bins


# Basically copied from OpenVLA
class UniformActionTokenizer(ActionTokenizer):
  def __init__(
      self,
      tokenizer: PreTrainedTokenizerBase,
      # config: PretrainedConfig,
      model_args
  ) -> None:
    """
    Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

    NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
             appear at the end of the vocabulary!

    :param tokenizer: Base LLM/VLM tokenizer to extend.
    :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
    :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
    :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
    """
    # super().__init__(self, tokenizer, config)
    super().__init__(tokenizer, model_args)

    self.min_action = model_args.min_action
    self.max_action = model_args.max_action

    # Create Uniform Bins + Compute Bin Centers
    self.bins = np.linspace(self.min_action, self.max_action, self.num_bins)
    self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
    #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
    self.action_token_begin_idx: int = int(
        self.tokenizer.vocab_size - (self.num_bins + 1))

  def __call__(self, action: np.ndarray, mask: np.ndarray) -> Union[str, List[str]]:
    """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
    action = np.clip(action, a_min=float(self.min_action),
                     a_max=float(self.max_action))
    discretized_action = np.digitize(action, self.bins)
    # Handle single element vs. batch
    discretized_action = self.tokenizer.vocab_size - discretized_action - 1
    discretized_action[~mask] = self.invalid_token_idx
    if len(discretized_action.shape) == 1:
     return self.tokenizer.decode(list(discretized_action))
    else:
     return self.tokenizer.batch_decode(discretized_action.tolist())
    # return self.tokenizer.vocab_size - discretized_action - 1

  def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
    """
    Returns continuous actions for discrete action token IDs.

    NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
             digitization returns bin indices between [1, # bins], inclusive, when there are actually only
             (# bins - 1) bin intervals.

             Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

    EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                is still one index (i==255) that would cause an out-of-bounds error if used to index into
                self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                the last bin center. We implement this simply via clipping between [0, 255 - 1].
    """

    # print(self.tokenizer.vocab_size)
    discretized_actions = self.tokenizer.vocab_size - action_token_ids
    discretized_actions = np.clip(
        discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

    return self.bin_centers[discretized_actions]


# Basically copied from OpenVLA
class NormalizedUniformActionTokenizer(UniformActionTokenizer):
  def __init__(
      self,
      tokenizer: PreTrainedTokenizerBase,
      # config: PretrainedConfig,
      model_args
  ) -> None:
    super().__init__(tokenizer, model_args)
    self.load_normalization(
      model_args.normalization_file_path
    )
    
  def load_normalization(
    self, normalization_file_path
  ):
    with open(normalization_file_path, ) as f:
      data_dict = pickle.load(f)
    self.normalization_mean = data_dict["mean"]
    self.normalization_std = data_dict["std"]


  def normalize(self, action):
    return (action - self.normalization_mean) / (self.normalization_std + 1e-6)
  
  def denormalize(self, action):
    return action * self.normalization_std + self.normalization_mean

  def __call__(self, action: np.ndarray, mask: np.ndarray) -> Union[str, List[str]]:
    """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
    action = self.normalize(action)
    return super().__call__(action, mask)

  def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
    denormalized_action = self.denormalize(super().decode_token_ids_to_actions(action_token_ids))
    return denormalized_action


# Basically copied from OpenVLA
class SepDimUniformActionTokenizer(ActionTokenizer):
  def __init__(
      self,
      tokenizer: PreTrainedTokenizerBase,
      # config: PretrainedConfig,
      model_args
  ) -> None:
    """
    Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

    NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
             appear at the end of the vocabulary!

    :param tokenizer: Base LLM/VLM tokenizer to extend.
    :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
    :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
    :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
    """
    # super().__init__(self, tokenizer, config)
    super().__init__(tokenizer, model_args)

    # self.min_action = config.min_action
    # self.max_action = config.max_action
    self.min_action = model_args.min_action
    self.max_action = model_args.max_action

    # Create Uniform Bins + Compute Bin Centers
    self.bins = np.linspace(self.min_action, self.max_action, self.num_bins)
    self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
    
    self.num_action_dims = model_args.num_action_dims
    self.num_action_sep_dims = model_args.num_action_sep_dims
    # Use Customized 
    assert self.num_action_dims % self.num_action_sep_dims == 0
    self.bin_bias = np.array(
      list(range(self.num_action_sep_dims)) * (self.num_action_dims // self.num_action_sep_dims)
    ) * self.num_bins

    # just add a random additional padding in case something stupid happens
    random_additional_padding = 10
    self.invalid_token_idx = int(
        self.tokenizer.vocab_size - self.num_bins * self.num_action_sep_dims - 1 - random_additional_padding
    )
    random_additional_padding_input = 9
    self.input_placeholder_token_idx = int(
        self.tokenizer.vocab_size - self.num_bins * self.num_action_sep_dims - 1 - random_additional_padding_input
    )
    # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
    #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
    self.action_token_begin_idx: int = int(
        self.tokenizer.vocab_size - (self.num_bins + 1))

  def __call__(self, action: np.ndarray, mask: np.ndarray) -> Union[str, List[str]]:
    """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
    action = np.clip(action, a_min=float(self.min_action),
                     a_max=float(self.max_action))
    discretized_action = np.digitize(action, self.bins)
    # print(self.tokenizer.vocab_size)
    # Handle single element vs. batch
    # discretized_action = self.tokenizer.vocab_size - discretized_action - 1

    # Different action dims maps to different bins
    discretized_action = self.tokenizer.vocab_size - discretized_action - 1 - self.bin_bias

    # print(mask)
    discretized_action[~mask] = self.invalid_token_idx
    # print(discretized_action)
    if len(discretized_action.shape) == 1:
     return self.tokenizer.decode(list(discretized_action))
    else:
     return self.tokenizer.batch_decode(discretized_action.tolist())
    # return self.tokenizer.vocab_size - discretized_action - 1

  def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
    """
    Returns continuous actions for discrete action token IDs.

    NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
             digitization returns bin indices between [1, # bins], inclusive, when there are actually only
             (# bins - 1) bin intervals.

             Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

    EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                is still one index (i==255) that would cause an out-of-bounds error if used to index into
                self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                the last bin center. We implement this simply via clipping between [0, 255 - 1].
    """

    # print(self.tokenizer.vocab_size)
    discretized_actions = self.tokenizer.vocab_size - action_token_ids - self.bin_bias
    discretized_actions = np.clip(
        discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

    return self.bin_centers[discretized_actions]

class SepDimNormUniformActionTokenizer(SepDimUniformActionTokenizer):
  def __init__(
      self,
      tokenizer: PreTrainedTokenizerBase,
      model_args
  ) -> None:
    super().__init__(tokenizer, model_args)
    self.load_normalization(
      model_args.normalization_file_path
    )
    
  def load_normalization(
    self, normalization_file_path
  ):
    with open(normalization_file_path, ) as f:
      data_dict = pickle.load(f)
    self.normalization_mean = data_dict["mean"]
    self.normalization_std = data_dict["std"]

  def normalize(self, action):
    return (action - self.normalization_mean) / (self.normalization_std + 1e-6)
  
  def denormalize(self, action):
    return action * self.normalization_std + self.normalization_mean

  def __call__(self, action: np.ndarray, mask: np.ndarray) -> Union[str, List[str]]:
    """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
    action = self.normalize(action)
    return super().__call__(action, mask)

  def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
    denormalized_action = self.denormalize(super().decode_token_ids_to_actions(action_token_ids))
    return denormalized_action


# Basically copied from OpenVLA
class PerDimUniformActionTokenizer(ActionTokenizer):
  def __init__(
      self,
      tokenizer: PreTrainedTokenizerBase,
      # config: PretrainedConfig,
      model_args
  ) -> None:
    """
    Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

    NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
             appear at the end of the vocabulary!

    :param tokenizer: Base LLM/VLM tokenizer to extend.
    :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
    :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
    :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
    """
    # super().__init__(self, tokenizer, config)
    super().__init__(tokenizer, model_args)

    # self.min_action = config.min_action
    # self.max_action = config.max_action
    self.min_action = model_args.min_action
    self.max_action = model_args.max_action

    # Create Uniform Bins + Compute Bin Centers
    self.bins = np.linspace(self.min_action, self.max_action, self.num_bins)
    self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
    
    self.num_action_dims = model_args.num_action_dims
    self.bin_bias = np.arange(self.num_action_dims) * self.num_bins

    # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
    #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
    self.action_token_begin_idx: int = int(
        self.tokenizer.vocab_size - (self.num_bins + 1))


    # just add a random additional padding in case something stupid happens
    random_additional_padding = 10
    self.invalid_token_idx = int(
        self.tokenizer.vocab_size - self.num_bins * self.num_action_dims - 1 - random_additional_padding
    )
    random_additional_padding_input = 9
    self.input_placeholder_token_idx = int(
        self.tokenizer.vocab_size - self.num_bins * self.num_action_dims - 1 - random_additional_padding_input
    )

  def __call__(self, action: np.ndarray, mask: np.ndarray) -> Union[str, List[str]]:
    """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
    action = np.clip(action, a_min=float(self.min_action),
                     a_max=float(self.max_action))
    discretized_action = np.digitize(action, self.bins)
    # print(self.tokenizer.vocab_size)
    # Handle single element vs. batch
    # discretized_action = self.tokenizer.vocab_size - discretized_action - 1

    # Different action dims maps to different bins
    discretized_action = self.tokenizer.vocab_size - discretized_action - 1 - self.bin_bias

    # print(mask)
    discretized_action[~mask] = self.invalid_token_idx
    # print(discretized_action)
    if len(discretized_action.shape) == 1:
     return self.tokenizer.decode(list(discretized_action))
    else:
     return self.tokenizer.batch_decode(discretized_action.tolist())
    # return self.tokenizer.vocab_size - discretized_action - 1

  def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
    """
    Returns continuous actions for discrete action token IDs.

    NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
             digitization returns bin indices between [1, # bins], inclusive, when there are actually only
             (# bins - 1) bin intervals.

             Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

    EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                is still one index (i==255) that would cause an out-of-bounds error if used to index into
                self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                the last bin center. We implement this simply via clipping between [0, 255 - 1].
    """

    # print(self.tokenizer.vocab_size)
    discretized_actions = self.tokenizer.vocab_size - action_token_ids - self.bin_bias
    discretized_actions = np.clip(
        discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

    return self.bin_centers[discretized_actions]


# Basically copied from OpenVLA
class QuantileActionTokenizer(ActionTokenizer):
  def __init__(
      self,
      tokenizer: PreTrainedTokenizerBase,
      model_args,
      # quantile: np.ndarray,
  ) -> None:
    """
    Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

    NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
             appear at the end of the vocabulary!

    :param tokenizer: Base LLM/VLM tokenizer to extend.
    :param quantiles: 
    """
    super().__init__(tokenizer, model_args)

    # self.quantile = self.load()
    self.load_quantile(model_args)
    # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
    #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
    self.action_token_begin_idx: int = int(
      self.tokenizer.vocab_size - (self.num_bins + 1)
    )

  def load_quantile(self, model_args):
    self.quantile =np.load(model_args.quantile_path).reshape(-1)
    assert np.prod(self.quantile.shape) == model_args.num_action_bins - 1

  def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
    """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
    # action_shape = action.shape
    # discretized_action = get_class_label(
    #   action.reshape(-1), self.quantile
    # ).reshape(action_shape)
    discretized_action = np.digitize(action, self.quantile)
    return self.tokenizer.vocab_size - discretized_action - 1

    # Handle single element vs. batch
    if len(discretized_action.shape) == 1:
      return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
    else:
      return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

  def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
    """
    Returns continuous actions for discrete action token IDs.

    NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
             digitization returns bin indices between [1, # bins], inclusive, when there are actually only
             (# bins - 1) bin intervals.

             Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

    EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                is still one index (i==255) that would cause an out-of-bounds error if used to index into
                self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                the last bin center. We implement this simply via clipping between [0, 255 - 1].
    """
    discretized_actions = self.tokenizer.vocab_size - action_token_ids
    discretized_actions = np.clip(
        discretized_actions - 1, a_min=0, a_max=self.quantile.shape[0] - 1)

    return self.quantile[discretized_actions]


def build_action_tokenizer(
    # action_tokenizer, tokenizer, config,
    action_tokenizer, tokenizer, model_args,
) -> ActionTokenizer:
  if action_tokenizer == "uniform":
    return UniformActionTokenizer(
      # tokenizer, config
      tokenizer, model_args
    )
  if action_tokenizer == "norm_uniform":
    return NormalizedUniformActionTokenizer(
      # tokenizer, config
      tokenizer, model_args
    )
  if action_tokenizer == "perdim_uniform":
    return PerDimUniformActionTokenizer(
      # tokenizer, config
      tokenizer, model_args
    )
  if action_tokenizer == "sepdim_uniform":
    return SepDimUniformActionTokenizer(
      # tokenizer, config
      tokenizer, model_args
    )
  if action_tokenizer == "norm_sepdim_uniform":
    return SepDimNormUniformActionTokenizer(
      # tokenizer, config
      tokenizer, model_args
    )
  elif action_tokenizer == "quantile":
    return QuantileActionTokenizer(
      tokenizer, model_args
    )
  raise NotImplementedError