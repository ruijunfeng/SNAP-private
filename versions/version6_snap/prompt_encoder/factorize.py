
class FactorizedProfilingBlock(nn.Module):
    def __init__(self, hidden_dim, head_dim, ffn_ratio=2, ffn_dropout=0.0, attention_bias=False, attention_dropout=0.0):
        """
        Factorized Interaction Block (FIB):
        Decouples explicit 2nd-order bi-interaction and higher-order topological attention.
        """
        super().__init__()
        
        # --- 1. Explicit 2nd-Order Bi-Interaction Branch ---
        self.bi_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.bfloat16)
        
        # --- 2. Higher-Order Self-Attention Branch ---
        self.attn = MultiHeadSelfAttention(
            hidden_dim=hidden_dim, 
            head_dim=head_dim, 
            attention_bias=attention_bias, 
            attention_dropout=attention_dropout, 
        )
        
        # --- 3. Multi-Order Fusion & Normalization ---
        # Learnable gating factor to balance 2nd-order and higher-order interactions
        self.norm1 = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
        
        # --- 4. Feed-Forward Network Sublayer ---
        self.norm2 = nn.LayerNorm(hidden_dim, dtype=torch.bfloat16)
        ffn_hidden_dim = hidden_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim, dtype=torch.bfloat16),
            nn.Dropout(ffn_dropout)
        )
    
    def forward(self, x):
        # x: (batch_size, num_features, hidden_dim)
        
        # 2nd-Order Bi-Interaction (Element-wise Product)
        bi_out = x * self.bi_proj(x)
        
        # High-Order Interaction via Multi-Head Self-Attention
        attn_out = self.attn(x)
        
        # Dynamic Gated Fusion + Residual Connection
        x = self.norm1(x + bi_out + attn_out)
        
        # Standard FFN Transformation
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x