from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache

from methods.snap.prompt_encoder import NumericalPromptEncoder

class SNAP(nn.Module):
    """
    Similar to PeftModelForCausalLM, this class is a wrapper for SNAP. 
    The prompt encoder takes numerical features as input and injects them 
    as shared Key/Value prefixes (past_key_values) across all transformer layers.
    
    Args:
        config: The configuration of the prompt encoder.
        base_model: The base model to be used.
    """
    def __init__(self, config, base_model):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.word_embeddings = self.base_model.get_input_embeddings()
        
        # 1. Fetch architecture dimensions for KV cache shaping
        self.num_layers = self.base_model.config.num_hidden_layers
        self.num_heads = self.base_model.config.num_attention_heads
        
        # Support for Grouped-Query Attention (GQA) models (e.g., Llama-2, Mistral)
        self.num_kv_heads = getattr(self.base_model.config, "num_key_value_heads", self.num_heads)
        
        # Calculate head dimension
        if hasattr(self.base_model.config, "head_dim"):
            self.head_dim = self.base_model.config.head_dim
        else:
            self.head_dim = self.base_model.config.hidden_size // self.num_heads

        self.prompt_encoder = NumericalPromptEncoder(
            use_numerical_embedding=config.use_numerical_embedding,
            use_numerical_profiling=config.use_numerical_profiling,
            use_projector=config.use_projector,
            num_features=config.num_features,
            embed_dim=self.num_kv_heads * self.head_dim * 2,
            head_dim=config.head_dim,
            mlp_ratio=config.mlp_ratio,
            mlp_dropout=config.mlp_dropout,
            hidden_dim=config.hidden_dim,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            num_layers=config.num_layers,
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        numeric_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        
        # 1. Standard text embeddings (NO concatenation here)
        input_embeds = self.word_embeddings(input_ids)
        
        # 2. Generate soft prompts from numerical features
        # Note: Your encoder's final output dimension should ideally be: 
        # (num_kv_heads * head_dim * 2) so it can be split into Keys and Values.
        # Expected shape from encoder: (batch_size, prefix_len, num_kv_heads * head_dim * 2)
        soft_prompts = self.prompt_encoder(numeric_features)
        prefix_len = soft_prompts.size(1)
        
        # 3. Reshape and split into Key and Value tensors
        # View as: (batch_size, prefix_len, 2, num_kv_heads, head_dim)
        kv_prompts = soft_prompts.view(
            batch_size, 
            prefix_len, 
            2, 
            self.num_kv_heads, 
            self.head_dim
        )
        
        # Permute to Hugging Face's expected KV shape: 
        # (2, batch_size, num_kv_heads, prefix_len, head_dim)
        kv_prompts = kv_prompts.permute(2, 0, 3, 1, 4)
        
        # Extract Key and Value
        key_prefix, value_prefix = kv_prompts[0], kv_prompts[1]
        
        # 4. Construct past_key_values: Share the same KV pair across ALL layers
        past_key_values = tuple(
            (key_prefix, value_prefix) for _ in range(self.num_layers)
        )
        
        past_key_values = DynamicCache()
        
        for layer_idx in range(self.num_layers):
            # The update method will safely initialize the cache for each layer
            # using your shared Key and Value prefixes
            past_key_values.update(key_prefix, value_prefix, layer_idx)
            
        # 5. Extend attention mask for the prefix length
        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, prefix_len, device=attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            
        # [NOTE] You DO NOT need to modify the labels tensor anymore! 
        # Since the prefix is processed as historical KV cache, the current sequence length 
        # remains identical to the labels length.
        
        # 6. Forward pass with the base_model
        outputs = self.base_model(
            inputs_embeds=input_embeds, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            **kwargs
        )
        
        # Adds the soft_prompts to the outputs if needed for tracking
        outputs["soft_prompts"] = soft_prompts
        return outputs
    
    def print_trainable_parameters(self):
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())
        # Calculate trainable percentage
        trainable_percent = trainable_params / total_params * 100 if total_params > 0 else 0
        # Print the information in the desired format
        print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {trainable_percent:.4f}")
