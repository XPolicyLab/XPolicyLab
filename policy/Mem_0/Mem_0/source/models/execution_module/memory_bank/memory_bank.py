from typing import Optional, Union, List, Tuple
import torch
from termcolor import cprint
import os
from source.utils import BOLD, RESET_BOLD
import torch.nn as nn
import numpy as np
import math


class PositionEmbedder(nn.Module):
    """
    Embeds relative positions into vector representations using sinusoidal embeddings.
    Position 1 = most recent, Position N = oldest.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, dtype: Optional[torch.dtype] = None):
        """
        Initialize PositionEmbedder.
        
        Args:
            hidden_size: Output dimension of position embeddings
            frequency_embedding_size: Dimension of frequency embeddings before MLP
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    @staticmethod
    def position_embedding(pos: torch.Tensor, dim: int, max_period: int = 30) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.
        Optimized for window_size=30: max_period set to 30 to better encode positions in range [1, 30].
        
        Args:
            pos: 1-D Tensor of position indices, shape (N,)
            dim: Dimension of the output embeddings
            max_period: Controls the minimum frequency of the embeddings (default: 30 for window_size=30)
            
        Returns:
            Tensor of shape (N, dim) containing positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(pos.device)
        args = pos[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through position embedder.
        
        Args:
            pos: Tensor of relative positions, shape (N,)
                 1 = most recent, N = oldest
            
        Returns:
            Tensor of shape (N, hidden_size)
        """
        pos = pos.to(next(self.mlp.parameters()).device)
        pos_freq = self.position_embedding(pos, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        pos_emb = self.mlp(pos_freq)
        return pos_emb


class MemoryBank(nn.Module):
    """
    MemoryBank manages a sliding window of hidden states for each episode.
    Uses relative positional encoding optimized for window_size=30.
    
    Architecture:
    - Pre-LN: Layer normalization applied before attention (not after)
    - Two-stage cross-attention:
      1. First: Cross-attention with sliding window (recent operation history)
      2. Second: Cross-attention with anchor (task initial state)
    
    For each new input vector (B, 1, H) with episode_ids:
    1. In stream mode: maintain episode continuity across batches
    2. Group samples by episode_id (same episode_id share the same memory bank)
    3. If memory exists, perform two sequential cross-attention operations:
       a. First cross-attention: Q = current_vector (Pre-LN), K/V = sliding window memory
          - Focuses on recent operation history (window_size up to 30)
          - Position encoding: 1 = most recent, 30 = oldest
       b. Second cross-attention: Q = first_attention_output (Pre-LN), K/V = anchor memory
          - Focuses on task initial state (anchor_size = 1)
          - Aligns current state with original task intent
    4. If no stored vectors, output the input vector directly
    5. Add the original input vector to the sliding window (FIFO, max size=30)
    6. Anchor: stores the first input vector (anchor_size=1, never popped)
    
    Memory clearing policy:
    - Memory is cleared ONLY when end_signal_count reaches memory_accumulation threshold
    - Episode changes do NOT trigger memory clearing
    - Each episode maintains its own memory independently
    
    Each episode is processed independently without cross-episode interference.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        window_size: int,
        initial_anchor_size: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        memory_accumulation: int = 8,
        frequency_embedding_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize MemoryBank.
        
        Args:
            hidden_dim: Dimension of hidden states (H)
            window_size: Maximum number of vectors to store in sliding window (recommended: 30)
            initial_anchor_size: Number of anchor vectors to store (must be 1)
            
            num_heads: Number of attention heads for cross-attention
            dropout: Dropout rate for attention layers
            frequency_embedding_size: Dimension of frequency embeddings (default: hidden_dim // 4)
            device: Device to store tensors on
            dtype: Data type for tensors
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.initial_anchor_size = initial_anchor_size
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.memory_accumulation = memory_accumulation
        
        # Relative position encoder for sliding window only
        # Position encoding optimized for window_size=30 (max_period=30 in position_embedding)
        # Note: anchor does not use positional encoding (single frame, no temporal ordering)
        freq_dim = frequency_embedding_size if frequency_embedding_size is not None else hidden_dim // 4
        self.window_position_encoder = PositionEmbedder(hidden_dim, freq_dim)  # For sliding window (positions 1-30)
        
        # Two cross-attention layers
        self.cross_attn1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,  # Use (seq_len, batch, embed_dim) format
        )
        
        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        
        # Layer normalization for Pre-LN architecture (before attention)
        self.norm1 = nn.LayerNorm(hidden_dim)  # For window attention
        self.norm2 = nn.LayerNorm(hidden_dim)  # For anchor attention
        
        # Ensure all modules have the same dtype as input (important for BF16 inputs)
        def _cast(module: nn.Module):
            # Safely cast module to configured dtype/device if provided
            params = list(module.parameters())
            if len(params) == 0:
                # No parameters to cast
                return module
            target_dtype = self.dtype
            target_device = self.device
            if target_dtype is None and target_device is None:
                return module
            if target_device is not None and params[0].device != target_device:
                module = module.to(device=target_device)
            if target_dtype is not None and params[0].dtype != target_dtype:
                module = module.to(dtype=target_dtype)
            return module
        
        self.window_position_encoder = _cast(self.window_position_encoder)
        self.cross_attn1 = _cast(self.cross_attn1)
        self.cross_attn2 = _cast(self.cross_attn2)
        self.norm1 = _cast(self.norm1)
        self.norm2 = _cast(self.norm2)
        
        # Ensure modules can be aligned to runtime input dtype/device if not preset
        def _ensure_module_dtype_device(input_tensor: torch.Tensor):
            # If dtype/device not preset, adopt from incoming tensor once
            if self.dtype is None:
                self.dtype = input_tensor.dtype
            if self.device is None:
                self.device = input_tensor.device
            # Re-cast modules if required
            self.window_position_encoder = _cast(self.window_position_encoder)
            self.cross_attn1 = _cast(self.cross_attn1)
            self.cross_attn2 = _cast(self.cross_attn2)
            self.norm1 = _cast(self.norm1)
            self.norm2 = _cast(self.norm2)

        # Store helper for later use
        self._ensure_module_dtype_device = _ensure_module_dtype_device

        
        # Storage for sliding windows and anchors:
        # bank[episode_id] = {"anchors": [vector, ...], "window": [vector, ...]}
        # Each stored vector is shape (1, H)
        self.bank = {}
        
        # Stream mode: track current episode across batches
        self.eid_stream = None
        
        # When end signal is received by {self.memory_accumulation} consecutive steps, we clear the memory bank
        self.end_signal_count = {}
        
        # Optional debug flag controlled externally
        self.debug = False
        
    def forward(
        self, 
        new_vector: torch.Tensor,
        episode_ids: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
        subtask_end_flags: Optional[List] = None,
    ) -> torch.Tensor:
        """
        Process a new vector through the memory bank (stream mode).
        
        In training mode: processes samples sequentially to handle episode transitions.
        Maintains episode continuity across batches using eid_stream.
        
        Memory clearing policy:
        - Memory is cleared ONLY when end_signal_count reaches memory_accumulation threshold
        - Episode changes do NOT trigger memory clearing
        - Each episode maintains its own memory independently
        
        Two-stage Pre-LN cross-attention:
        1. First: Cross-attention with sliding window (recent operation history, max 30 steps)
        2. Second: Cross-attention with anchor (task initial state, 1 frame)
        
        Args:
            new_vector: Tensor of shape (B, 1, H) - new input vectors
            episode_ids: Episode identifiers for each sample in the batch.
                        Can be List, np.ndarray, or torch.Tensor.
                        If None, each sample is treated as a separate episode.
            subtask_end_flags: Optional list of subtask end flags
            
        Returns:
            Tensor of shape (B, 1, H) - fused output vectors
        """
        if self.debug:
            cprint ("[mem] memory bank forward called!", "green")
            
        batch_size, seq_len, hidden_dim = new_vector.shape
        assert seq_len == 1, f"Expected sequence length 1, got {seq_len}"
        assert hidden_dim == self.hidden_dim, f"Expected hidden dim {self.hidden_dim}, got {hidden_dim}"
        # Note: initial_anchor_size is fixed to 1 in __init__
        
        # Convert episode_ids to list if provided
        if episode_ids is None:
            episode_ids = list(range(batch_size))
        elif isinstance(episode_ids, torch.Tensor):
            episode_ids = episode_ids.cpu().tolist()
        elif isinstance(episode_ids, np.ndarray):
            episode_ids = episode_ids.tolist()
        
        assert len(episode_ids) == batch_size, \
            f"episode_ids length ({len(episode_ids)}) must match batch_size ({batch_size})"
        # Normalize each episode id to a Python scalar (int/str) so dict keys are stable.
        normalized_ids = []
        for eid in episode_ids:
            # handle 0-d torch tensors, numpy scalars, and plain python ints/strings
            try:
                if isinstance(eid, torch.Tensor):
                    # prefer int when possible
                    if eid.numel() == 1:
                        normalized_ids.append(int(eid.item()))
                    else:
                        normalized_ids.append(eid)
                elif isinstance(eid, np.generic):
                    normalized_ids.append(int(eid))
                else:
                    normalized_ids.append(eid)
            except Exception:
                # fallback: keep original
                normalized_ids.append(eid)
        episode_ids = normalized_ids
        
        # Align module dtype/device to incoming tensor
        self._ensure_module_dtype_device(new_vector)

        # Stream mode: update eid_stream for tracking (no memory clearing on episode change)
        if self.training:
            first_eid = episode_ids[0]
            # Only clear memory if end_signal_count reaches threshold (not on episode change)
            if self.eid_stream is not None and \
                self.end_signal_count.get(self.eid_stream, 0) >= self.memory_accumulation:
                # Clear memory only when end signal count reaches threshold
                self.clear_episode(self.eid_stream)
            self.eid_stream = first_eid
        
        # Process batch sequentially to handle episode transitions within batch
        fused_tensor, anchor_tensor = self._forward_sequential(new_vector, episode_ids, subtask_end_flags=subtask_end_flags)
        return fused_tensor, anchor_tensor
        
    def reset(self, episode_ids: Optional[list] = None):
        """
        Reset memory banks for specified episodes or all episodes.
        
        Args:
            episode_ids: List of episode IDs to reset. If None, reset all.
        """
        if episode_ids is None:
            # Reset all
            self.bank = {}
            self.end_signal_count = {}
            self.eid_stream = None
            if self.debug:
                cprint("    [mem] reset: all episodes cleared", "yellow")
        else:
            # Reset specified episodes
            for eid in episode_ids:
                # normalize incoming eid
                try:
                    if isinstance(eid, torch.Tensor) and eid.numel() == 1:
                        key = int(eid.item())
                    elif isinstance(eid, np.generic):
                        key = int(eid)
                    else:
                        key = eid
                except Exception:
                    key = eid

                if key in self.bank:
                    del self.bank[key]
                    if self.debug:
                        cprint(f"    [mem] reset: cleared episode {BOLD}{key}{RESET_BOLD}", "yellow")
                
                if key in self.end_signal_count:
                    del self.end_signal_count[key]
                    if self.debug:
                        cprint(f"    [mem] reset: cleared end_signal_count for episode {BOLD}{key}{RESET_BOLD}", "yellow")
    
    def clear_episode(self, episode_id):
        """
        Clear memory for a specific episode.
        
        Args:
            episode_id: Episode identifier to clear
        """
        # normalize key to Python scalar to match storage keys
        try:
            if isinstance(episode_id, torch.Tensor) and episode_id.numel() == 1:
                episode_id = int(episode_id.item())
            elif isinstance(episode_id, np.generic):
                episode_id = int(episode_id)
        except Exception:
            pass

        if episode_id in self.bank:
            del self.bank[episode_id]
            if self.debug:
                cprint(f"    [mem] clear: eid = {BOLD}{episode_id}{RESET_BOLD}", "yellow")
                
        if episode_id in self.end_signal_count:
            del self.end_signal_count[episode_id]
            if self.debug:
                cprint(f"    [mem] clear: cleared end_signal_count for episode {BOLD}{episode_id}{RESET_BOLD}", "yellow")
    
    def _get_memory_tensor(self, episode_id) -> Optional[tuple]:
        """
        Get stored vectors for a specific episode.
        
        Args:
            episode_id: Episode identifier
            
        Returns:
            Tuple (memory_tensor, anchor_len, window_len) where memory_tensor has shape
            (num_stored, H). Returns None if no memory exists.
        """
        # normalize key to Python scalar to match storage keys
        try:
            if isinstance(episode_id, torch.Tensor) and episode_id.numel() == 1:
                episode_id = int(episode_id.item())
            elif isinstance(episode_id, np.generic):
                episode_id = int(episode_id)
        except Exception:
            pass

        if episode_id not in self.bank:
            return None
        
        anchors = self.bank[episode_id].get("anchors", [])
        window = self.bank[episode_id].get("window", [])
        total_len = len(anchors) + len(window)
        if total_len == 0:
            return None
        
        tensors = []
        if len(anchors) > 0:
            tensors.append(torch.stack(anchors, dim=0).squeeze(1))
        if len(window) > 0:
            tensors.append(torch.stack(window, dim=0).squeeze(1))
        
        if not tensors:
            return None
        
        memory_tensor = torch.cat(tensors, dim=0)
        
        return memory_tensor, len(anchors), len(window)
    
    def _upd_to_window(
        self,
        episode_id: int,
        vector: torch.Tensor,
    ) -> bool:
        """
        Update the memory bank's window with a new vector for a specific episode.
        Window stores original vectors (with FIFO)
        
        Args:
            episode_id: Episode identifier
            vector: Tensor of shape (1, H) to add into window
        """
        # normalize key to Python scalar to avoid using tensor objects as dict keys
        try:
            if isinstance(episode_id, torch.Tensor) and episode_id.numel() == 1:
                episode_id = int(episode_id.item())
            elif isinstance(episode_id, np.generic):
                episode_id = int(episode_id)
        except Exception:
            pass

        if episode_id not in self.bank:
            self.bank[episode_id] = {"anchors": [], "window": []}
        
        entry = self.bank[episode_id]
        entry["window"].append(vector.detach().clone())
        if len(entry["window"]) > self.window_size:
            entry["window"].pop(0)
        
        if self.debug:
            rank = int(os.environ.get("RANK", 0))
            cprint(
                f"    [mem] rank = {BOLD}{rank}{RESET_BOLD}; eid = {BOLD}{episode_id}{RESET_BOLD}, bank = {{anchors: {BOLD}{len(entry['anchors'])}{RESET_BOLD}, window: {BOLD}{len(entry['window'])}{RESET_BOLD}}}",
                "cyan",
            )
    
    def _upd_to_anchor(
        self,
        episode_id: int,
        vector: torch.Tensor,
    ) -> bool:
        """
        Update the memory bank's window with a new vector for a specific episode.
        Anchors store original vectors until full.
        
        Args:
            episode_id: Episode identifier
            vector: Tensor of shape (1, H) to add into window
            subtask_end_flag: whether this sample marks the end of a subtask
        """
        # normalize key to Python scalar to avoid using tensor objects as dict keys
        try:
            if isinstance(episode_id, torch.Tensor) and episode_id.numel() == 1:
                episode_id = int(episode_id.item())
            elif isinstance(episode_id, np.generic):
                episode_id = int(episode_id)
        except Exception:
            pass

        if episode_id not in self.bank:
            self.bank[episode_id] = {"anchors": [], "window": []}
        
        entry = self.bank[episode_id]
            
        # Anchors: replace on subtask_end_flag or first time (store fused_vector)
        if len(entry["anchors"]) == 0:
            entry["anchors"] = [vector.detach().clone()]
            if self.debug:
                rank = int(os.environ.get("RANK", 0))
                cprint(
                    f"    [mem] rank = {BOLD}{rank}{RESET_BOLD}; anchor add: eid = {BOLD}{episode_id}{RESET_BOLD}; anchors = {BOLD}{len(entry['anchors'])}{RESET_BOLD}",
                    "yellow",
                )
    
    def _forward_sequential(
        self,
        new_vector: torch.Tensor,
        episode_ids: List,
        subtask_end_flags: Optional[List] = None,
    ) -> torch.Tensor:
        """
        Process batch sequentially, handling episode transitions.
        Memory is cleared only when end_signal_count reaches memory_accumulation threshold,
        not when episode_id changes.
        
        Two-stage Pre-LN cross-attention:
        1. First: Cross-attention with sliding window (recent operation history)
        2. Second: Cross-attention with anchor (task initial state)
        
        Args:
            new_vector: Tensor of shape (B, 1, H)
            episode_ids: List of episode IDs for each sample
            subtask_end_flags: Optional list of subtask end flags
            
        Returns:
            Tensor of shape (B, 1, H) - fused output vectors
        """
        batch_size = new_vector.shape[0]
        fused_sliding_outputs = []
        fused_anchor_outputs = []
        device = new_vector.device
        
        for i in range(batch_size):
            # Episode management: update eid_stream but don't clear memory on episode change
            current_eid = episode_ids[i]
            if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                # Episode changed: update eid_stream but don't clear memory
                if self.training:
                    self.eid_stream = current_eid
            
            # Get current vector
            current_vector = new_vector[i, :, :]  # (1, H)
            
            # Get stored memory for this episode (before updating)
            memory_result = self._get_memory_tensor(current_eid)
            
            if memory_result is None:
                # No memory exists: return current vector with proper shape (1, 1, H)
                # Use unsqueeze(1) to match the shape used in the else branch
                fused_sliding_vector = current_vector.unsqueeze(1)  # (1, H) -> (1, 1, H)
                fused_anchor_vector = current_vector.unsqueeze(1)   # (1, H) -> (1, 1, H)
            else:
                memory_tensor, anchor_len, window_len = memory_result
                memory_tensor = memory_tensor.to(dtype=current_vector.dtype)
                
                q = current_vector.unsqueeze(1)  # (1, H) -> (1, 1, H)    
                
                # Cross-attention with sliding window
                if window_len > 0:
                    # get sliding window memory
                    window_mem = memory_tensor[anchor_len:]  # (window_len, H)
                    
                    q_norm = self.norm1(q.squeeze(1)).unsqueeze(1)  # (1, 1, H)
                    
                    # Generate positional encoding for window (positions: 1=most recent, window_len=oldest)
                    window_positions = torch.arange(window_len, 0, -1, dtype=torch.long, device=device)
                    window_pe = self.window_position_encoder(window_positions).to(
                        device=window_mem.device, dtype=window_mem.dtype
                    )
                    # Add positional encoding first, then normalize (ensures Q/K/V have consistent statistics)
                    window_mem_with_pe = window_mem + window_pe  # (window_len, H)
                    window_mem_norm = self.norm1(window_mem_with_pe)  # (window_len, H)
                    
                    # Cross-attention: Q from current vector, K/V from window memory (both normalized)
                    k_window = window_mem_norm.unsqueeze(1)  # (window_len, 1, H)
                    v_window = window_mem_norm.unsqueeze(1)  # (window_len, 1, H)
                    
                    attn_output1, _ = self.cross_attn1(q_norm, k_window, v_window)  # (1, 1, H)
                    fused_sliding_vector = attn_output1 + q  # Residual connection (Pre-LN: add original q)
                else:
                    fused_sliding_vector = q  # (1, 1, H)
                
                # Cross-attention with anchor
                if anchor_len > 0:
                    # get anchor memory
                    anchor_mem = memory_tensor[:anchor_len]  # (anchor_len, H)
                    
                    q_norm = self.norm2(q.squeeze(1)).unsqueeze(1)  # (1, 1, H)
                    anchor_mem_norm = self.norm2(anchor_mem)  # (anchor_len, H)
                    
                    # Cross-attention: Q from current vector, K/V from anchor memory (both normalized)
                    k_anchor = anchor_mem_norm.unsqueeze(1)  # (anchor_len, 1, H)
                    v_anchor = anchor_mem_norm.unsqueeze(1)  # (anchor_len, 1, H)
                    
                    attn_output2, _ = self.cross_attn2(q_norm, k_anchor, v_anchor)  # (1, 1, H)
                    fused_anchor_vector = attn_output2 + q  # Residual connection (Pre-LN: add original q)
                else:
                    fused_anchor_vector = q  # (1, 1, H)
            
            # Update memory bank: anchor and window together
            # Always update both anchor and window (anchor update only happens if anchor is empty)
            self._upd_to_window(current_eid, current_vector)
            self._upd_to_anchor(current_eid, current_vector)
            
            # Update end signal count and clear memory if threshold reached
            sub_end_flag = subtask_end_flags[i] if subtask_end_flags is not None else False
            if self.end_signal_count.get(current_eid, None) is None:
                self.end_signal_count[current_eid] = 0
            if sub_end_flag:
                self.end_signal_count[current_eid] += 1
                # Clear memory only when end signal count reaches threshold
                if self.end_signal_count[current_eid] >= self.memory_accumulation:
                    self.clear_episode(current_eid)
            
            # Both vectors should be (1, 1, H) at this point
            fused_sliding_outputs.append(fused_sliding_vector)  # (1, 1, H)
            fused_anchor_outputs.append(fused_anchor_vector)    # (1, 1, H)
        
        # Stack all outputs: (B, 1, H)
        fused_sliding_tensor = torch.cat(fused_sliding_outputs, dim=0)  # (B, 1, H)
        fused_anchor_tensor = torch.cat(fused_anchor_outputs, dim=0)    # (B, 1, H)
        return fused_sliding_tensor, fused_anchor_tensor
    
    @torch.inference_mode()
    def update_on_eval (
        self, 
        new_vector: torch.Tensor,
        text_vector: torch.Tensor,
        classifier, # SubtaskEndClassifier
        episode_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Process a new vector through the memory bank during evaluation.
        In evaluation mode: uses the classifier to detect subtask boundaries.
        
        Uses the same logic as _forward_sequential:
        1. Update anchor first (if empty)
        2. Get stored memory for this episode
        3. Perform cross-attention with sliding window and anchor separately
        4. Update window with original vector
        5. Use classifier to predict subtask end
        6. Return fused sliding vector, fused anchor vector and subtask end flag
        
        Args:
            new_vector: Tensor of shape (1, 1, H) - single sample
            text_vector: Tensor of shape (1, 1, H) or (1, H) - text features
            classifier: SubtaskEndClassifier instance
            episode_id: Episode identifier
            
        Returns:
            Tuple of (fused_sliding_vector, fused_anchor_vector, sub_end_flag)
            - fused_sliding_vector: Fused output after cross-attention with sliding window, shape (1, 1, H)
            - fused_anchor_vector: Fused output after cross-attention with anchor, shape (1, 1, H)
            - sub_end_flag: Boolean indicating if subtask ended
        """
            
        batch_size, seq_len, hidden_dim = new_vector.shape
        assert seq_len == 1, f"Expected sequence length 1, got {seq_len}"
        assert hidden_dim == self.hidden_dim, f"Expected hidden dim {self.hidden_dim}, got {hidden_dim}"
        assert batch_size == 1, f"batch_size must be 1, got {batch_size}"
        
        # Align module dtype/device to incoming tensor
        self._ensure_module_dtype_device(new_vector)
        
        device = new_vector.device 
        
        # Get current vector
        current_vector = new_vector[0, :, :]  # (1, H)
        
        # Get stored memory for this episode (after updating anchor)
        memory_result = self._get_memory_tensor(episode_id)
        
        if memory_result is None:
            # No memory exists: return current vector with proper shape (1, 1, H)
            fused_sliding_vector = current_vector.unsqueeze(1)  # (1, H) -> (1, 1, H)
            fused_anchor_vector = current_vector.unsqueeze(1)   # (1, H) -> (1, 1, H)
        else:
            memory_tensor, anchor_len, window_len = memory_result
            memory_tensor = memory_tensor.to(dtype=current_vector.dtype)
            
            q = current_vector.unsqueeze(1)  # (1, H) -> (1, 1, H)
            
            # Cross-attention with sliding window
            if window_len > 0:
                # Get sliding window memory
                window_mem = memory_tensor[anchor_len:]  # (window_len, H)
                
                q_norm = self.norm1(q.squeeze(1)).unsqueeze(1)  # (1, 1, H)
                
                # Generate positional encoding for window (positions: 1=most recent, window_len=oldest)
                window_positions = torch.arange(window_len, 0, -1, dtype=torch.long, device=device)
                window_pe = self.window_position_encoder(window_positions).to(
                    device=window_mem.device, dtype=window_mem.dtype
                )
                # Add positional encoding first, then normalize (ensures Q/K/V have consistent statistics)
                window_mem_with_pe = window_mem + window_pe  # (window_len, H)
                window_mem_norm = self.norm1(window_mem_with_pe)  # (window_len, H)
                
                # Cross-attention: Q from current vector, K/V from window memory (both normalized)
                k_window = window_mem_norm.unsqueeze(1)  # (window_len, 1, H)
                v_window = window_mem_norm.unsqueeze(1)  # (window_len, 1, H)
                
                attn_output1, _ = self.cross_attn1(q_norm, k_window, v_window)  # (1, 1, H)
                fused_sliding_vector = attn_output1 + q  # Residual connection (Pre-LN: add original q)
            else:
                fused_sliding_vector = q  # (1, 1, H)
            
            # Cross-attention with anchor
            if anchor_len > 0:
                # get anchor memory
                anchor_mem_raw = memory_tensor[:anchor_len]  # (anchor_len, H)
                
                q_norm = self.norm2(q.squeeze(1)).unsqueeze(1)  # (1, 1, H)
                anchor_mem_norm = self.norm2(anchor_mem_raw)  # (anchor_len, H)
                
                # Cross-attention: Q from current vector, K/V from anchor memory (both normalized)
                k_anchor = anchor_mem_norm.unsqueeze(1)  # (anchor_len, 1, H)
                v_anchor = anchor_mem_norm.unsqueeze(1)  # (anchor_len, 1, H)
                
                attn_output2, _ = self.cross_attn2(q_norm, k_anchor, v_anchor)  # (1, 1, H)
                fused_anchor_vector = attn_output2 + q  # Residual connection (Pre-LN: add original q)
            else:
                fused_anchor_vector = q  # (1, 1, H)
        
        # Update memory bank: update window with original vector
        self._upd_to_window(episode_id, current_vector)
        # Update anchor with original vector
        self._upd_to_anchor(episode_id, current_vector)
        
        summary_vector = torch.cat([fused_sliding_vector, fused_anchor_vector, text_vector], dim=2) #(1, 1, 3*H)
        
        with torch.autocast("cuda", dtype=torch.float32):
            cls_output = classifier.predict(summary_vector)  # (1, 1, 3*H) -> (1, 1)
        sub_end_flag = (cls_output["prob"] >= 0.5).item()
            
        # Update end signal count and clear memory if threshold reached
        if self.end_signal_count.get(episode_id, None) is None:
            self.end_signal_count[episode_id] = 0
        if sub_end_flag:
            self.end_signal_count[episode_id] += 1
            # Clear memory only when end signal count reaches threshold
            # if self.end_signal_count[episode_id] >= self.memory_accumulation:
            #     self.clear_episode(episode_id)
        
        return fused_sliding_vector, fused_anchor_vector, sub_end_flag
         
    
    def get_memory_size(self, episode_id: Optional[Union[int, str, list]] = None) -> Union[int, list, dict]:
        """
        Get the current size of memory banks.
        
        Args:
            episode_id: If specified, return size for this episode. 
                       If list, return sizes for these episodes.
                       If None, return dict mapping all episode_ids to sizes.
            
        Returns:
            Memory size(s) - int, list, or dict depending on input
        """
        def _size(entry):
            if not isinstance(entry, dict):
                return len(entry)
            return len(entry.get("anchors", [])) + len(entry.get("window", []))

        if episode_id is None:
            # Return dict mapping episode_id to size
            return {eid: _size(bank) for eid, bank in self.bank.items()}
        elif isinstance(episode_id, list):
            # Return list of sizes for specified episodes
            return [_size(self.bank.get(eid, {})) for eid in episode_id]
        else:
            # Return size for single episode
            return _size(self.bank.get(episode_id, {}))
    
    def save_state(self) -> dict:
        """
        Save the current state of MemoryBank (bank, end_signal_count, eid_stream).
        Used for evaluation to avoid affecting training state.
        
        Returns:
            dict: Saved state containing bank, end_signal_count, and eid_stream
        """
        saved_state = {
            "bank": {},
            "end_signal_count": self.end_signal_count.copy(),
            "eid_stream": self.eid_stream,
        }
        
        # Deep copy bank entries (each vector is already detached)
        for episode_id, entry in self.bank.items():
            saved_state["bank"][episode_id] = {
                "anchors": [vec.clone() for vec in entry.get("anchors", [])],
                "window": [vec.clone() for vec in entry.get("window", [])],
            }
        
        return saved_state
    
    def restore_state(self, saved_state: dict):
        """
        Restore MemoryBank state from saved state.
        
        Args:
            saved_state: State dict returned by save_state()
        """
        # Restore bank
        self.bank = {}
        for episode_id, entry in saved_state["bank"].items():
            self.bank[episode_id] = {
                "anchors": [vec.clone() for vec in entry.get("anchors", [])],
                "window": [vec.clone() for vec in entry.get("window", [])],
            }
        
        # Restore end_signal_count
        self.end_signal_count = saved_state["end_signal_count"].copy()
        
        # Restore eid_stream
        self.eid_stream = saved_state["eid_stream"]
