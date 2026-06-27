from typing import Optional

import torch
from torch import nn

from methods.snap.prompt_encoder import NumericalPromptEncoder


class SNAP(nn.Module):
    """
    Similar to PeftModelForCausalLM, this class is a wrapper for SNAP.
    The difference is that the prompt encoder here takes a numerical features as input.
    
    Args:
        config: The configuration of the prompt encoder.
        base_model: The base model to be used.
    """
    def __init__(self, config, base_model):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.word_embeddings = self.base_model.get_input_embeddings()
        self.prompt_encoder = NumericalPromptEncoder(
            use_numerical_embedding=config.use_numerical_embedding,
            use_numerical_profiling=config.use_numerical_profiling,
            use_projector=config.use_projector,
            num_features=config.num_features,
            embed_dim=self.word_embeddings.weight.shape[1],
            head_dim=config.head_dim,
            mlp_ratio=config.mlp_ratio,
            mlp_dropout=config.mlp_dropout,
            hidden_dim=config.hidden_dim,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            num_layers=config.num_layers,
            projector_ratio=config.projector_ratio,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        numeric_features: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Generate input embeddings from the base model's word embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        
        # Generate soft prompts
        soft_prompts = self.prompt_encoder(numeric_features)
        total_virtual_tokens = soft_prompts.size(1)
        
        # If attention_mask is not provided, create it using the pad_token_id.
        if attention_mask is None:
            pad_token_id = getattr(self.config, "pad_token_id", None)
            if pad_token_id is not None:
                attention_mask = (input_ids != pad_token_id).long()
            else:
                # Fallback: if no pad_token_id exists, assume all tokens are valid
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        # Generate position ids
        # Position ids of text tokens are started from 1 instead of 0. As prefixs are added to the left.
        # Example: mask [0, 0, 1, 1] -> cumsum [0, 0, 1, 2]
        text_position_ids = attention_mask.cumsum(dim=-1).long()
        # Prefix position ids are all 0, as numerical features are orderless.
        prefix_position_ids = torch.zeros(
            (batch_size, total_virtual_tokens), 
            dtype=text_position_ids.dtype, 
            device=device
        )
        position_ids = torch.cat((prefix_position_ids, text_position_ids), dim=1)
        
        # Concat soft prompts
        inputs_embeds = torch.cat((soft_prompts, inputs_embeds), dim=1)
        
        # Concat attention mask with prefix: soft prompts are always attended to (mask = 1)
        prefix_attention_mask = torch.ones(
            (batch_size, total_virtual_tokens), 
            dtype=attention_mask.dtype, 
            device=device
        )
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        # Concat labels with prefix: ignore index (-100) for soft prompts so they don't contribute to loss
        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, total_virtual_tokens), 
                -100,
                dtype=labels.dtype,
                device=device,
            )
            labels = torch.cat((prefix_labels, labels), dim=1)
        
        # Forward pass with the base_model
        outputs = self.base_model(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            **kwargs
        )
        
        # Adds the soft_prompts to the outputs
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
