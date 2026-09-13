from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import ModelOutput

from methods.base.attention import TransformerBlock
from methods.ft_transformer.config import FTTransformerConfig


class FTTransformer(nn.Module):
    def __init__(self, config: FTTransformerConfig):
        """
        Args:
            config (FTTransformerConfig): Model configuration object containing all hyperparameters.
        """
        super().__init__()
        
        self.config = config
        self.num_features = config.num_features
        self.hidden_dim = config.hidden_dim
        
        # --- 1. Feature Tokenizer (Per-feature scalar projection) ---
        # Conv1d with groups=num_features maps each feature independently: (B, N, 1) -> (B, N * hidden_dim, 1)
        # Includes per-feature weights and biases inherently
        self.scalar_projection = nn.Conv1d(
            in_channels=config.num_features,
            out_channels=config.num_features * config.hidden_dim,
            kernel_size=1,
            groups=config.num_features,
            dtype=torch.bfloat16,
        )
        
        # --- 2. [CLS] Token ---
        # Learnable token prepended to the sequence of feature embeddings
        self.cls_token = nn.Parameter(
            torch.empty(1, 1, config.hidden_dim, dtype=torch.bfloat16)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # --- 3. Transformer Backbone ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                head_dim=config.head_dim,
                ffn_ratio=config.ffn_ratio,
                ffn_dropout=config.ffn_dropout,
                attention_bias=config.attention_bias,
                attention_dropout=config.attention_dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        # --- 4. Classification Head ---
        # Final projection using the [CLS] token representation for binary classification
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 2, dtype=torch.bfloat16),
        )
    
    def forward(
        self,
        numeric_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
        """
        Args:
            numeric_features (torch.Tensor): Tabular input of shape (batch_size, num_features).
            labels (torch.Tensor, optional): Binary ground-truth targets.
            
        Returns:
            ModelOutput: HuggingFace generic output containing `loss` and `logits`.
        """
        batch_size = numeric_features.shape[0]
        
        # Cast input to matching precision and shape: (B, N) -> (B, N, 1)
        numeric_features = numeric_features.to(dtype=torch.bfloat16).unsqueeze(-1)
        
        # Step 1: Feature Tokenization via grouped Conv1d
        # (B, N, 1) -> (B, N * hidden_dim, 1)
        tokens = self.scalar_projection(numeric_features)
        # (B, N * hidden_dim, 1) -> (B, N, hidden_dim)
        tokens = tokens.view(batch_size, self.num_features, self.hidden_dim)
        
        # Step 2: Prepend the learnable [CLS] token
        # Expand [CLS] across batch: (1, 1, D) -> (B, 1, D)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Sequence: (B, 1 + N, hidden_dim)
        hidden_states = torch.cat([cls_tokens, tokens], dim=1)
        
        # Step 3: Pass through Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)
        
        # Step 4: Extract [CLS] representation and project to scalar logit
        # (B, 1 + N, hidden_dim) -> (B, hidden_dim)
        cls_output = hidden_states[:, 0]
        # (B, hidden_dim) -> (B, 2)
        logits = self.head(cls_output)
        
        # Step 5: Compute BCE loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return ModelOutput(
            loss=loss,
            logits=logits,
        )
