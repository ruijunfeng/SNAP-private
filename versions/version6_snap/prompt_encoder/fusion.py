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
        
        # Multi-scale aggregation via 1D Convolution over layer channels
        # Total channels = (num_layers + 1) including initial embeddings H^(0) and all intermediate block outputs
        # Treats each layer representation as a channel and fuses multi-order interactions back to hidden_dim
        num_channels = num_layers + 1
        self.channel_fusion = nn.Conv1d(
            in_channels=num_channels * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            dtype=torch.bfloat16,
        )
    
    def forward(self, x):
        # x: [batch_size, num_features, hidden_dim]
        # Preserve multi-order interaction states across layers: [H^(0), H^(1), ..., H^(L)]
        layer_outputs = [x]
        
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)
        
        # Concat along hidden dimension to form multi-order channels: [batch_size, num_features, (L+1) * hidden_dim]
        multi_scale_features = torch.cat(layer_outputs, dim=-1)
        
        # Transpose to [batch_size, (L+1) * hidden_dim, num_features] for Conv1d channel fusion
        multi_scale_features = multi_scale_features.transpose(1, 2)
        fused = self.channel_fusion(multi_scale_features) # -> [batch_size, hidden_dim, num_features]
        
        # Restore shape: [batch_size, num_features, hidden_dim]
        output = fused.transpose(1, 2)
        
        return output
