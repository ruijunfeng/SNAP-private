import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base.attention import MultiHeadSelfAttention, TransformerBlock


class NumericalEmbedding(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        half_dim = hidden_dim // 2
        
        # Use Conv1d with groups=num_features to achieve per-feature linear projection
        # Equivalent to having separate Linear layers for each feature, but more efficient
        # Like input with shape [batch, num_features, 1], it got groups * [1, hidden_dim] of non-shared weight matrix
        self.scalar_projection = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features * half_dim, # Changed to half_dim to accommodate sin/cos concat
            kernel_size=1,
            groups=num_features,
            dtype=torch.bfloat16,
        )
        nn.init.normal_(self.scalar_projection.weight, std=1.2)
        if self.scalar_projection.bias is not None:
            nn.init.uniform_(self.scalar_projection.bias, 0, 2*math.pi)
        
        # Normalization
        # The key is to do Norm over the embedding dimension
        # Ensuring that the embedding vectors for each feature have controlled magnitude, independent of batch distribution
        self.normalization = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
        
        # Use a linear layer for feature projection
        self.feature_projection = nn.Linear(hidden_dim, hidden_dim, dtype=torch.bfloat16)
    
    def forward(self, x):
        # x: [batch_size, num_features]
        
        # --- Step 1: Signed Log ---
        # Reference: Mastering Diverse Domains through World Models (DreamerV3)
        # This to replace Standardization (Mean/Std)
        # Ensuring the model can handle wide-ranging numerical values without instability
        # Even the input is 100 million, after log it's about 20, which neural networks can handle
        x = torch.sign(x) * torch.log1p(torch.abs(x))
        
        # --- Step 2: Scalar Projection ---
        x = x.unsqueeze(-1) # [batch, num_features] -> [batch, num_features, 1]
        freq = self.scalar_projection(x) # -> [batch, num_features * half_dim, 1]
        freq = freq.view(-1, self.num_features, self.hidden_dim // 2) # -> [batch, num_features, half_dim]
        # Concatenate sine and cosine to form the full hidden_dim
        x = torch.stack([torch.sin(freq), torch.cos(freq)], dim=-1).flatten(start_dim=-2) # -> [batch, num_features, hidden_dim]
        
        # --- Step 3: Normalization ---
        # Treat is as batch * num_features of hidden_dim vectors to normalize individually
        # This ensures the normalization is per-feature embedding, not across features
        # Even batch size is 1, it still works correctly
        x = self.normalization(x)
        
        # --- Step 4: Feature Projection ---
        x = self.feature_projection(x)
        
        return x


class PromptEmbedding(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.prompt_embeddings = nn.Embedding(num_features, hidden_dim, dtype=torch.bfloat16)
    
    def forward(self, x):
        # Expand based on batch size
        batch_size, num_features = x.shape
        feature_indices = torch.arange(num_features, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        return self.prompt_embeddings(feature_indices)


class ScalarEmbedding(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Conv1d with groups=num_features maps each feature independently: (B, N, 1) -> (B, N * hidden_dim, 1)
        # Includes per-feature weights and biases inherently
        self.scalar_projection = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features * hidden_dim,
            kernel_size=1,
            groups=num_features,
            dtype=torch.bfloat16,
        )
        
        # Normalization
        # Normalizes over the embedding dimension for each feature vector independently
        self.normalization = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
    
    def forward(self, x):
        # x: [batch_size, num_features]
        
        # --- Step 1: Scalar Projection ---
        x = x.unsqueeze(-1)  # [batch, num_features] -> [batch, num_features, 1]
        x = self.scalar_projection(x)  # -> [batch, num_features * hidden_dim, 1]
        x = x.view(-1, self.num_features, self.hidden_dim)  # -> [batch, num_features, hidden_dim]
        
        # --- Step 2: Normalization ---
        x = self.normalization(x)
        
        return x


class NumericalProfiling(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        head_dim,
        ffn_ratio,
        ffn_dropout,
        attention_bias,
        attention_dropout,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Stack multiple Transformer blocks as individual layers to track multi-order interactions
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                head_dim=head_dim,
                ffn_ratio=ffn_ratio,
                ffn_dropout=ffn_dropout,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        # x: [batch_size, num_features, hidden_dim]
        for block in self.blocks:
            x = block(x)
        return x


class NumericalPromptEncoder(nn.Module):
    def __init__(
        self, 
        use_numerical_embedding, 
        use_numerical_profiling, 
        use_projector,
        num_features, 
        embed_dim, 
        hidden_dim, 
        head_dim, 
        ffn_ratio, 
        ffn_dropout, 
        attention_bias, 
        attention_dropout, 
        num_layers, 
        projector_ratio, 
    ):
        super().__init__()
        # whether to use numerical embeddings
        if use_numerical_embedding:
            self.numerical_embedding = NumericalEmbedding(
                num_features=num_features, 
                hidden_dim=hidden_dim, # use a smaller embedding dimension for numerical features to save parameters and Postvent overfitting
            )
        else:
            self.numerical_embedding = ScalarEmbedding(
                num_features=num_features,
                hidden_dim=hidden_dim,
            )
        # whether to use multi-head self-attention for numerical profiling
        if use_numerical_profiling:
            self.numerical_profiling = NumericalProfiling(
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                head_dim=head_dim,
                ffn_ratio=ffn_ratio,
                ffn_dropout=ffn_dropout,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
            )
        else:
            self.numerical_profiling = nn.Identity()
        # whether to use projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, embed_dim*projector_ratio, dtype=torch.bfloat16), 
                nn.GELU(),
                nn.Linear(embed_dim*projector_ratio, embed_dim, dtype=torch.bfloat16),
            ) # project to the same dimension as word embeddings for seamless integration
        else:
            self.projector = nn.Identity()
    
    def forward(self, x):
        # x: [batch_size, num_features]
        
        # 1. Numerical Embedding
        x = self.numerical_embedding(x) # [batch_size, num_features, hidden_dim]
        
        # 2. Numerical Profiling
        x = self.numerical_profiling(x) # [batch_size, num_features, hidden_dim]
        
        # 3. Final Projection
        x = self.projector(x) # [batch_size, num_features, embed_dim]
        
        return x


if __name__ == "__main__":
    # Numerical Embedding Test
    model = NumericalEmbedding(num_features=5, embed_dim=16)
    
    # One sample with numerical features in different magnitudes
    input_one = torch.tensor([[100000.0, 0.05, -500.0, 3.0, 0.0]])
    
    output = model(input_one)
    
    print("Output shape:", output.shape) # [1, 5, 16]
    print("Contain NaN?", torch.isnan(output).any().item())
    print("Mean (approx 0):", output.mean().item()) # LayerNorm ensure the mean is around 0
    
    # Multi-Head Self-Attention Test
    batch_size = 2
    num_features = 23
    hidden_dim = 4096
    head_dim = 128
    input_tensor = torch.randn(batch_size, num_features, hidden_dim)
    
    # Initialize Multi-Head Self-Attention
    attention = MultiHeadSelfAttention(hidden_dim, head_dim)
    
    # Forward pass
    output = attention(input_tensor)
    
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
    assert input_tensor.shape == output.shape
