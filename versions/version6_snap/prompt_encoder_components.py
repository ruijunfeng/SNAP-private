import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextualFeatureGate(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        # Concatenate all features into a single long vector to perceive the user's global context state.
        # Map back to the num_features dimension, corresponding to the gating score for each feature.
        self.context_proj = nn.Linear(num_features * hidden_dim, num_features, dtype=torch.bfloat16)
    
    def forward(self, x):
        # x shape: [batch_size, num_features, hidden_dim]
        batch_size = x.size(0)
        
        # 1. Flatten to extract global context: [batch_size, num_features * hidden_dim]
        global_context = x.view(batch_size, -1)
        
        # 2. Compute gating scores and compress to (0, 1) range using Sigmoid: [batch_size, num_features]
        gate_scores = torch.sigmoid(self.context_proj(global_context))
        
        # 3. Add a dimension for broadcasting: [batch_size, num_features, 1]
        gate_scores = gate_scores.unsqueeze(-1)
        
        # 4. Element-wise multiplication to softly filter out irrelevant features
        return x * gate_scores


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, mlp_hidden_dim, dropout=0.0):
        super().__init__()
        # Gating path (Gate)
        self.w1 = nn.Linear(hidden_dim, mlp_hidden_dim, dtype=torch.bfloat16)
        # Value path (Value)
        self.w2 = nn.Linear(hidden_dim, mlp_hidden_dim, dtype=torch.bfloat16)
        # Output projection
        self.w3 = nn.Linear(mlp_hidden_dim, hidden_dim, dtype=torch.bfloat16)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Gate passes through SiLU activation, Value remains linear, followed by element-wise multiplication
        gate = F.silu(self.w1(x))
        value = self.w2(x)
        gated_hidden = gate * value
        return self.dropout(self.w3(gated_hidden))


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim, head_dim, attention_bias=False, attention_dropout=0.0):
        super().__init__()
        if hidden_dim % head_dim != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by head_dim ({head_dim})")
        
        self.num_attention_heads = hidden_dim // head_dim
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        
        # Learnable projections
        self.q_proj = nn.Linear(hidden_dim, self.num_attention_heads * self.head_dim, bias=attention_bias, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden_dim, self.num_attention_heads * self.head_dim, bias=attention_bias, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(hidden_dim, self.num_attention_heads * self.head_dim, bias=attention_bias, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, query_states, encoder_hidden_states):
        """
        query_states: [batch_size, num_queries, hidden_dim] (learnable Queries)
        encoder_hidden_states: [batch_size, num_features, hidden_dim] (numercial features x)
        """
        batch_size, num_queries, _ = query_states.shape
        _, num_features, _ = encoder_hidden_states.shape
        
        # --- Step 1: Linear Projections ---
        q = self.q_proj(query_states)
        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)
        
        # --- Step 2: Reshape & Transpose ---
        q = q.view(batch_size, num_queries, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_features, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # --- Step 3: Scaled Dot-Product Attention ---
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # --- Step 4: Weighted Sum ---
        attn_output = torch.matmul(attn_weights, v)
        
        # --- Step 5: Restore Shape ---
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_queries, -1)
        
        # --- Step 6: Final Projection ---
        attn_output = self.o_proj(attn_output)
        return attn_output


class QFormerBlock(nn.Module):
    def __init__(self, hidden_dim, head_dim, mlp_ratio=2, dropout=0.0, attention_bias=False, attention_dropout=0.0):
        super().__init__()
        
        # 1. Self-attention
        self.norm1 = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
        self.self_attn = MultiHeadSelfAttention(
            hidden_dim=hidden_dim, head_dim=head_dim, attention_bias=attention_bias, attention_dropout=attention_dropout
        )
        
        # 2. Cross attention
        self.norm2 = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
        self.cross_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim, head_dim=head_dim, attention_bias=attention_bias, attention_dropout=attention_dropout
        )
        
        # 3. FFN
        self.norm3 = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
        mlp_hidden_dim = hidden_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim, dtype=torch.bfloat16),
            nn.Dropout(dropout)
        )
    
    def forward(self, query_states, encoder_hidden_states):
        # Step 1: Self-Attention
        q = self.norm1(query_states)
        q = query_states + self.self_attn(q)
        
        # Step 2: Cross-Attention
        q_norm = self.norm2(q)
        q = q + self.cross_attn(query_states=q_norm, encoder_hidden_states=encoder_hidden_states)
        
        # Step 3: FFN
        q_norm = self.norm3(q)
        q = q + self.mlp(q_norm)
        
        return q


class QFormer(nn.Module):
    def __init__(self, hidden_dim, head_dim, num_queries=8, num_layers=1, mlp_ratio=2, dropout=0.0, attention_bias=False, attention_dropout=0.0):
        """
        Args:
            hidden_dim: Internal processing dimension
            head_dim: Attention head dimension
            num_queries: Number of tokens to compress into
            num_layers: Number of Q-Former Blocks
        """
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Initialie learnable Query
        self.query_tokens = nn.Embedding(num_queries, hidden_dim, dtype=torch.bfloat16).weight
        
        # Stack multiple Q-Former Blocks
        self.layers = nn.ModuleList([
            QFormerBlock(
                hidden_dim=hidden_dim, 
                head_dim=head_dim, 
                mlp_ratio=mlp_ratio,
                dropout=dropout, 
                attention_bias=attention_bias, 
                attention_dropout=attention_dropout,
            ) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
    
    def forward(self, x):
        """
        x: [batch_size, num_features, hidden_dim] numerical features
        Returns:
            [batch_size, num_queries, hidden_dim]
        """
        batch_size = x.shape[0]
        
        # Expand Query to match batch_size
        query_states = self.query_tokens.expand(batch_size, -1, -1)
        
        # Pass through Q-Former Blocks
        for layer in self.layers:
            query_states = layer(query_states=query_states, encoder_hidden_states=x)
        
        # Final layer normalization
        query_states = self.final_norm(query_states)
        
        return query_states

q_former = QFormer(
    hidden_dim=768, 
    head_dim=128, 
    num_queries=16, 
    num_layers=3, 
    mlp_ratio=2, 
    dropout=0.1, 
    attention_bias=False, 
    attention_dropout=0.1, 
)
