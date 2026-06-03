"""
FAST tokenizer for Pi0-FAST model.

This implementation is based on the OpenPI project:
https://github.com/Physical-Intelligence/openpi

The FAST tokenizer maps continuous actions to discrete tokens in the PaliGemma vocabulary,
enabling autoregressive action generation using a standard language model.

Adapted to PyTorch by Galaxea AI.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import sentencepiece

from accelerate.logging import get_logger

logger = get_logger(__name__)


class FASTTokenizer:
    """
    FAST tokenizer that maps action tokens to PaliGemma vocabulary.

    This tokenizer:
    1. Tokenizes text prompts and discretized state using PaliGemma tokenizer
    2. Tokenizes continuous actions using HuggingFace FAST tokenizer
    3. Maps FAST action tokens to the last tokens in PaliGemma vocabulary
    4. Creates attention masks for prefix-LM training (bidirectional on prefix, causal on suffix)
    """

    def __init__(
        self,
        max_len: int = 256,
        fast_tokenizer_path: str = "physical-intelligence/fast",
        paligemma_tokenizer_path: str = "/TO/Your/Path/openpi/big_vision/paligemma_tokenizer.model",
        pad_token_id: int = 0,
        image_token_index: int = 257151,
        num_tokens_per_image: int = 256,
        num_input_images: int = 1,
    ):
        self._max_len = max_len
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab (special tokens)
        self.pad_token_id = pad_token_id
        self.image_token_index = image_token_index
        self.num_tokens_per_image = num_tokens_per_image
        self.num_input_images = num_input_images
        self.total_image_tokens = num_tokens_per_image * num_input_images

        # Load PaliGemma tokenizer
        with open(paligemma_tokenizer_path, "rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Get special token IDs from tokenizer
        self.bos_token_id = self._paligemma_tokenizer.bos_id()
        self.eos_token_id = self._paligemma_tokenizer.eos_id()

        # Load FAST tokenizer from HuggingFace
        try:
            from transformers import AutoProcessor
            self._fast_tokenizer = AutoProcessor.from_pretrained(
                fast_tokenizer_path, trust_remote_code=True
            )
            logger.info(f"Loaded FAST tokenizer from {fast_tokenizer_path}")
        except Exception as e:
            logger.warning(f"Failed to load FAST tokenizer: {e}. Using fallback.")
            self._fast_tokenizer = None

        # Training mode flag (default to training)
        self._is_train = True

    def train(self):
        """Set tokenizer to training mode."""
        self._is_train = True
        return self

    def eval(self):
        """Set tokenizer to evaluation/inference mode."""
        self._is_train = False
        return self

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize a single sample."""
        instruction = sample["instruction"]

        # Handle proprio dimension: could be [time, dim] or [dim]
        if "proprio" in sample:
            proprio = sample["proprio"]
            if proprio.ndim == 2:
                state = proprio[-1].cpu().numpy()  # Take last timestep
            else:
                state = proprio.cpu().numpy()
        else:
            state = None

        actions = sample["action"].cpu().numpy() if "action" in sample and sample["action"] is not None else None

        tokens, token_mask, ar_mask, loss_mask = self.tokenize(instruction, state, actions)

        sample["input_ids"] = torch.as_tensor(tokens, dtype=torch.long)
        sample["attention_mask"] = torch.as_tensor(token_mask, dtype=torch.bool)
        sample["ar_mask"] = torch.as_tensor(ar_mask, dtype=torch.long)
        sample["loss_mask"] = torch.as_tensor(loss_mask, dtype=torch.bool)

        return sample

    def tokenize(
        self,
        prompt: str,
        state: Optional[np.ndarray],
        actions: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Tokenize prompt, state, and actions into a single sequence.

        Args:
            prompt: Text instruction
            state: Proprioceptive state array (will be discretized)
            actions: Continuous action array [horizon, action_dim]

        Returns:
            tokens: Token IDs including image tokens, text tokens, and action tokens
            token_mask: Boolean mask for valid tokens
            ar_mask: 0 for bidirectional attention (prefix), 1 for causal attention (suffix)
            loss_mask: Boolean mask for tokens to compute loss on (action tokens only)
        """
        cleaned_text = prompt.lower().strip().replace("_", " ").replace("\n", " ")

        # Build prefix: image tokens + text tokens (NO BOS in text, matching lerobot)
        # Image tokens come first
        image_tokens = [self.image_token_index] * self.total_image_tokens

        # Discretize state into 256 bins if provided
        if state is not None:
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            discretized_state = np.clip(discretized_state, 0, 255)
            state_str = " ".join(map(str, discretized_state.astype(int)))
            text_prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        else:
            text_prefix = f"Task: {cleaned_text};\n"

        # Encode text WITHOUT BOS (BOS will be at start of action sequence, like lerobot)
        text_tokens = self._paligemma_tokenizer.encode(text_prefix, add_bos=False)
        prefix_tokens = image_tokens + text_tokens

        # Build postfix: BOS + "Action: " + action tokens + "|"
        # Note: No EOS - we use "|" as the action end marker (following lerobot)
        if self._is_train and actions is not None and self._fast_tokenizer is not None:
            # Tokenize actions with FAST tokenizer
            action_tokens = self._fast_tokenizer(actions[None])[0]
            if isinstance(action_tokens, list):
                action_tokens = np.array(action_tokens)
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # BOS at start of action sequence (like lerobot)
            bos_token = [self.bos_token_id]
            action_prefix = self._paligemma_tokenizer.encode("Action: ")
            action_suffix = self._paligemma_tokenizer.encode("|")

            postfix_tokens = bos_token + action_prefix + action_tokens_in_pg.tolist() + action_suffix

            # Loss mask: include "Action: ", action tokens, and "|" in loss
            # Only BOS is excluded (it's the start token for generation)
            num_bos_overhead = len(bos_token)
            loss_mask_postfix = (
                [False] * num_bos_overhead +  # BOS - excluded from loss
                [True] * len(action_prefix) +  # "Action: " - included in loss
                [True] * len(action_tokens_in_pg) +  # actual action tokens
                [True] * len(action_suffix)  # "|" - included in loss (end marker)
            )
        else:
            # Inference mode: only add BOS token as the starting point for autoregressive generation
            # The model will generate "Action: ", action tokens, and "|" autoregressively
            bos_token = [self.bos_token_id]
            postfix_tokens = bos_token
            loss_mask_postfix = [False]  # BOS not included in loss (inference mode anyway)

        # Combine prefix and postfix
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)  # 0=bidirectional, 1=causal
        loss_mask = [False] * len(prefix_tokens) + loss_mask_postfix

        # Pad or truncate to max_len
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding_len = self._max_len - tokens_len
            tokens = tokens + [self.pad_token_id] * padding_len
            token_mask = token_mask + [False] * padding_len
            ar_mask = ar_mask + [0] * padding_len
            loss_mask = loss_mask + [False] * padding_len
        else:
            if tokens_len > self._max_len:
                logger.warning(
                    f"Token length ({tokens_len}) exceeds max length ({self._max_len}), truncating."
                )
            tokens = tokens[:self._max_len]
            token_mask = token_mask[:self._max_len]
            ar_mask = ar_mask[:self._max_len]
            loss_mask = loss_mask[:self._max_len]

        return (
            np.asarray(tokens, dtype=np.int64),
            np.asarray(token_mask, dtype=bool),
            np.asarray(ar_mask, dtype=np.int64),
            np.asarray(loss_mask, dtype=bool),
        )

    def extract_actions(
        self,
        tokens: np.ndarray,
        action_horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        """
        Extract and decode action tokens from model output.

        Uses string-based extraction (following OpenPI) for robustness.
        The model generates: "Action: " + action_tokens + "|"
        We decode to string, extract between "Action: " and "|", then re-encode.

        Args:
            tokens: Generated token IDs (the raw tokens generated by the model)
            action_horizon: Number of action steps
            action_dim: Dimension of each action

        Returns:
            actions: Decoded continuous actions [horizon, action_dim]
        """
        if self._fast_tokenizer is None:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Convert tokens to list
        tokens_list = tokens.tolist() if isinstance(tokens, np.ndarray) else list(tokens)

        # Step 1: Decode tokens to string (following OpenPI string-based approach)
        decoded_str = self._paligemma_tokenizer.decode(tokens_list)

        # Step 2: Check if "Action: " exists anywhere in the decoded string
        if "Action: " not in decoded_str:
            preview = decoded_str[:50].replace('\n', '\\n') if decoded_str else "<empty>"
            logger.warning(f"No 'Action: ' found in decoded tokens. Preview: {preview}...")
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        try:
            # Step 3: Extract content between "Action: " and "|" using string split
            # This is more robust than token-level matching
            action_str = decoded_str.split("Action: ")[1].split("|")[0].strip()

            if not action_str:
                logger.warning("Empty action content after split")
                return np.zeros((action_horizon, action_dim), dtype=np.float32)

            # Step 4: Re-encode the action string to get action tokens
            raw_action_tokens = np.array(
                self._paligemma_tokenizer.encode(action_str, add_bos=False)
            )

            if len(raw_action_tokens) == 0:
                logger.warning("No tokens after re-encoding action string")
                return np.zeros((action_horizon, action_dim), dtype=np.float32)

            # Step 5: Convert from PaliGemma token space to FAST token space
            action_tokens = self._paligemma_tokens_to_act_tokens(raw_action_tokens)

            # Step 6: Decode with FAST tokenizer
            actions = self._fast_tokenizer.decode(
                [action_tokens.tolist()],
                time_horizon=action_horizon,
                action_dim=action_dim,
            )[0]

            return actions

        except Exception as e:
            logger.warning(
                f"Error decoding tokens: {e}. "
                f"Decoded preview: {decoded_str[:100]}... "
                f"Action horizon={action_horizon}, dim={action_dim}"
            )
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """Map FAST tokens to PaliGemma vocabulary (last tokens before special tokens)."""
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        vocab_size = self._paligemma_tokenizer.vocab_size()
        return vocab_size - 1 - self._fast_skip_tokens - tokens

    def _paligemma_tokens_to_act_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """Map PaliGemma tokens back to FAST token space."""
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        vocab_size = self._paligemma_tokenizer.vocab_size()
        return vocab_size - 1 - self._fast_skip_tokens - tokens
