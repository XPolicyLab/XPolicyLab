import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional

class ModelSelector(nn.Module):
    def __init__(self, image_feature_dim: int, text_feature_dim: int, history_embedding_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        # history code: 0 = 未调用, 1 = 使用 AffordVLA, 2 = 使用 DiT Action
        self.history_embedding = nn.Embedding(num_embeddings=3, embedding_dim=history_embedding_dim)
        self.history_gru = nn.GRU(input_size=history_embedding_dim, hidden_size=hidden_dim, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(image_feature_dim + text_feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_embeds: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        history: torch.Tensor,
        history_lengths: torch.Tensor,
    ):
        # image_features: (batch_size, image_feature_dim) or (batch_size, num_patches, image_feature_dim)
        # text_embeds: (batch_size, seq_len, text_feature_dim) or already pooled (batch_size, text_feature_dim)
        # text_attention_mask: Optional[(batch_size, seq_len)]
        # history: (batch_size, max_history_len) - integer tensor in {0,1,2}
        # history_lengths: (batch_size,) - actual lengths of history sequences

        # 1. Process history
        history_embedded = self.history_embedding(history) # (batch_size, max_history_len, history_embedding_dim)

        # Pack padded sequence for GRU
        packed_history = pack_padded_sequence(history_embedded, history_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, history_hidden = self.history_gru(packed_history) # history_hidden: (1, batch_size, hidden_dim)
        
        # Squeeze the first dimension (num_layers)
        history_hidden = history_hidden.squeeze(0) # (batch_size, hidden_dim)

        # 2. Aggregate image features
        if len(image_features.shape) == 3:
            # If image_features is (batch_size, num_patches, feature_dim), aggregate it
            image_features = image_features.mean(dim=1)  # (batch_size, feature_dim)

        # 3. Aggregate text features (masked mean if mask provided)
        if len(text_embeds.shape) == 3:
            if text_attention_mask is not None:
                mask = text_attention_mask.to(dtype=text_embeds.dtype, device=text_embeds.device).unsqueeze(-1)  # (B, L, 1)
                denom = mask.sum(dim=1).clamp_min(1e-6)  # (B, 1)
                text_features = (text_embeds * mask).sum(dim=1) / denom  # (B, D)
            else:
                text_features = text_embeds.mean(dim=1)  # (B, D)
        else:
            # Already pooled
            text_features = text_embeds

        # 4. Concatenate features and predict selection score
        combined_features = torch.cat([image_features, text_features, history_hidden], dim=-1) # (batch_size, image_feature_dim + text_feature_dim + hidden_dim)

        # 5. Predict selection score
        selection_score = self.fc(combined_features) # (batch_size, 1)

        # Apply sigmoid to get a probability for selection (e.g., probability of choosing DiffusionTransformerAction)
        # return torch.sigmoid(selection_score)
        return selection_score # return logits, sigmoid will be applied in loss function or training process
