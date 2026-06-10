import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericalEmbedding(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Use Conv1d with groups=num_features to achieve per-feature linear projection
        # Equivalent to having separate Linear layers for each feature, but more efficient
        # Like input with shape [batch, num_features, 1], it got groups * [1, hidden_dim] of non-shared weight matrix
        self.scalar_projection = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features * hidden_dim,
            kernel_size=1,
            groups=num_features,
            dtype=torch.bfloat16,
        )
        
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
        x = self.scalar_projection(x) # -> [batch, num_features * hidden_dim, 1]
        x = x.view(-1, self.num_features, self.hidden_dim) # -> [batch, num_features, hidden_dim]
        
        # --- Step 3: Normalization ---
        # Treat is as batch * num_features of hidden_dim vectors to normalize individually
        # This ensures the normalization is per-feature embedding, not across features
        # Even batch size is 1, it still works correctly
        x = self.normalization(x)
        
        # --- Step 4: Feature Projection ---
        x = self.feature_projection(x)
        
        return x


class PromptEmbeddings(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.prompt_embeddings = nn.Embedding(num_features, hidden_dim, dtype=torch.bfloat16)
    
    def forward(self, x):
        # Expand based on batch size
        batch_size, num_features = x.shape
        feature_indices = torch.arange(num_features, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        return self.prompt_embeddings(feature_indices)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, head_dim, attention_bias=False, attention_dropout=0.0):
        super().__init__()
        
        # Check that hidden_dim is divisible by head_dim
        if hidden_dim % head_dim != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by head_dim ({head_dim})"
            )
        
        # Hyperparameters used in forward pass
        self.num_attention_heads = hidden_dim // head_dim
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        
        # Learnable projections for Q, K, V
        self.q_proj = nn.Linear(
            hidden_dim, self.num_attention_heads * self.head_dim, bias=attention_bias, dtype=torch.bfloat16,
        )
        self.k_proj = nn.Linear(
            hidden_dim, self.num_attention_heads * self.head_dim, bias=attention_bias, dtype=torch.bfloat16,
        )
        self.v_proj = nn.Linear(
            hidden_dim, self.num_attention_heads * self.head_dim, bias=attention_bias, dtype=torch.bfloat16,
        )
        self.o_proj = nn.Linear(
            hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16,
        )
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, hidden_states):
        batch_size, num_features, _ = hidden_states.shape
        
        # --- Step 1: Linear Projections ---
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # --- Step 2: Reshape & Transpose (Multi-Head) ---
        # Decompose hidden_dim into num_heads * head_dim
        # (B, N, H*D) -> (B, N, H, D) -> (B, H, N, D)
        query_states = query_states.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # --- Step 3: Scaled Dot-Product Attention ---
        # Q * K^T / sqrt(d)
        # Q: (B, H, N, D) K^T: (B, H, D, N)
        # attn_weights: (B, H, N, N) describing attention scores between all feature pairs
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        # Softmax normalization
        attn_weights = F.softmax(attn_weights, dim=-1)
        # Attention dropout
        attn_weights = self.dropout(attn_weights)
        
        # --- Step 4: Weighted Sum (Aggregate Values) ---
        # attn_weights * V: (B, H, N, N) * (B, H, N, D) -> (B, H, N, D)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # --- Step 5: Restore Shape (Concat Multi-Head) ---
        # (B, H, N, D) -> (B, N, H, D) -> (B, N, H*D)
        # Notice: after transpose, memory is not contiguous, must call .contiguous() before view
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_features, -1)
        
        # --- Step 6: Final Linear Projection ---
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, head_dim, mlp_ratio=2, mlp_dropout=0.0, attention_bias=False, attention_dropout=0.0):
        """
        Args:
            hidden_dim (int): The dimension of the input and output hidden states.
            head_dim (int): The dimension of each attention head.
            mlp_ratio (float): The expansion ratio for the Feed-Forward Network. 
                               Standard is 4.0 (e.g., 768 -> 3072 -> 768).
            mlp_dropout (float): Dropout probability for the FFN and residual connections.
            attention_bias (boolen): Whether to use bias term for projection.
            attention_dropout (float): Dropout probability within the attention mechanism.
        """
        super().__init__()
        
        # --- 1. Self-Attention Sublayer ---
        # LayerNorm applied after the attention mechanism
        self.norm1 = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
        # Using your provided MultiHeadSelfAttention class
        self.attn = MultiHeadSelfAttention(
            hidden_dim=hidden_dim, 
            head_dim=head_dim, 
            attention_bias=attention_bias, 
            attention_dropout=attention_dropout, 
        )
        
        # --- 2. Feed-Forward Network (FFN / MLP) Sublayer ---
        # LayerNorm applied after the FFN
        self.norm2 = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
        
        # The FFN typically expands the dimensionality, applies a non-linearity, 
        # and then projects it back to the original hidden_dim.
        mlp_hidden_dim = hidden_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim, dtype=torch.bfloat16),
            nn.GELU(),               # Gaussian Error Linear Unit
            nn.Dropout(mlp_dropout), # Dropout after activation
            nn.Linear(mlp_hidden_dim, hidden_dim, dtype=torch.bfloat16),
            nn.Dropout(mlp_dropout)  # Dropout before the residual connection
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_dim)
        """
        
        # --- Step 1: Attention block with Post-Norm and Residual Connection ---
        # Equation: x = LayerNorm(x + Attention(x))
        # The input 'x' bypasses the attention via the residual connection (+)
        attn_out = self.attn(x)
        x = x + attn_out
        x = self.norm1(x)
        
        # --- Step 2: FFN block with Post-Norm and Residual Connection ---
        # Equation: x = LayerNorm(x + FFN(x))
        # The output from Step 1 bypasses the MLP via the second residual connection (+)
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)
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
        mlp_ratio, 
        mlp_dropout, 
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
            self.numerical_embedding = PromptEmbeddings(
                num_features=num_features,
                hidden_dim=hidden_dim,
            )
        # whether to use multi-head self-attention for numerical profiling
        if use_numerical_profiling:
            self.numerical_profiling = nn.Sequential(
                *[TransformerBlock(
                    hidden_dim=hidden_dim, 
                    head_dim=head_dim, 
                    mlp_ratio=mlp_ratio, 
                    mlp_dropout=mlp_dropout, 
                    attention_bias=attention_bias, 
                    attention_dropout=attention_dropout, 
                ) for _ in range(num_layers)] # stack multiple blockes for deeper profiling
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
