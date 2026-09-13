
class PolynomialCrossAttention(nn.Module):
    """
    Polynomial Cross-Attention Module for Explicit Feature Interaction.
    
    Unlike standard self-attention which merely aggregates values via weighted summation,
    this module introduces an explicit element-wise (Hadamard) interaction kernel:
        Z_i = \sum_j A_ij * ( (W_v E_j) \odot (W_u E_i) )
    This explicitly computes pairwise multi-order polynomial feature products.
    """
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
        
        # Explicit polynomial cross projection (W_u) to interact with V_j
        self.u_proj = nn.Linear(
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
        cross_states = self.u_proj(hidden_states)
        
        # --- Step 2: Reshape & Transpose (Multi-Head) ---
        # Decompose hidden_dim into num_heads * head_dim
        # (B, N, H*D) -> (B, N, H, D) -> (B, H, N, D)
        query_states = query_states.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2)
        cross_states = cross_states.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # --- Step 3: Scaled Dot-Product Attention ---
        # Q * K^T / sqrt(d)
        # Q: (B, H, N, D) K^T: (B, H, D, N)
        # attn_weights: (B, H, N, N) describing attention scores between all feature pairs
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        # Softmax normalization
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Attention dropout
        attn_weights = self.dropout(attn_weights)
        
        # --- Step 4: Explicit Polynomial Interaction Aggregation ---
        # Standard: attn_output = torch.matmul(attn_weights, value_states)
        # Polynomial: Z_i = u_i \odot (\sum_j A_ij * v_j)
        # Mathematically equivalent to: \sum_j A_ij * (v_j \odot u_i)
        # Shape: (B, H, N, N) * (B, H, N, D) -> (B, H, N, D)
        aggregated_values = torch.matmul(attn_weights, value_states)
        
        # Explicit Hadamard product between target feature representation and context values
        attn_output = aggregated_values * cross_states
        
        # --- Step 5: Restore Shape (Concat Multi-Head) ---
        # (B, H, N, D) -> (B, N, H, D) -> (B, N, H*D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_features, -1)
        
        # --- Step 6: Final Linear Projection ---
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class PolynomialTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, head_dim, ffn_ratio=2, ffn_dropout=0.0, attention_bias=False, attention_dropout=0.0):
        """
        Args:
            hidden_dim (int): The dimension of the input and output hidden states.
            head_dim (int): The dimension of each attention head.
            ffn_ratio (float): The expansion ratio for the Feed-Forward Network. 
            ffn_dropout (float): Dropout probability for the FFN and residual connections.
            attention_bias (bool): Whether to use bias term for projection.
            attention_dropout (float): Dropout probability within the attention mechanism.
        """
        super().__init__()
        
        # --- 1. Polynomial Cross-Attention Sublayer ---
        # LayerNorm applied after the attention mechanism
        self.norm1 = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
        # Using the explicit polynomial interaction attention module
        self.attn = PolynomialCrossAttention(
            hidden_dim=hidden_dim, 
            head_dim=head_dim, 
            attention_bias=attention_bias, 
            attention_dropout=attention_dropout, 
        )
        
        # --- 2. Feed-Forward Network Sublayer ---
        # LayerNorm applied after the FFN
        self.norm2 = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
        
        # The FFN typically expands the dimensionality, applies a non-linearity, 
        # and then projects it back to the original hidden_dim.
        ffn_hidden_dim = hidden_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Dropout(ffn_dropout),  # Dropout after activation
            nn.Linear(ffn_hidden_dim, hidden_dim, dtype=torch.bfloat16),
            nn.Dropout(ffn_dropout)   # Dropout before the residual connection
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_dim)
        """
        # --- Step 1: Polynomial Attention block with Post-Norm and Residual Connection ---
        # Equation: x = LayerNorm(x + PolynomialAttention(x))
        attn_out = self.attn(x)
        x = x + attn_out
        x = self.norm1(x)
        
        # --- Step 2: FFN block with Post-Norm and Residual Connection ---
        # Equation: x = LayerNorm(x + FFN(x))
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        
        return x
