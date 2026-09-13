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
        special_tokens_ids: List/Tensor containing [begin_token_id, end_token_id].
    """
    def __init__(self, config, base_model, special_tokens_ids):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.word_embeddings = self.base_model.get_input_embeddings()
        # Register special token IDs as a buffer to automatically handle device placement (e.g., multi-GPU)
        self.register_buffer("special_tokens_ids", torch.tensor(special_tokens_ids, dtype=torch.long))
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
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        numeric_features: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Initialize the inputs_embeds
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        inputs_embeds = self.word_embeddings(input_ids)

        # Extract and expand special token embeddings safely on the correct device
        spec_embeds = self.word_embeddings(self.special_tokens_ids)  # Shape: [2, embed_dim]
        begin_embed = spec_embeds[0].view(1, 1, -1).expand(batch_size, -1, -1)
        end_embed = spec_embeds[1].view(1, 1, -1).expand(batch_size, -1, -1)
        
        # Generate soft prompts
        soft_prompts = self.prompt_encoder(numeric_features)
        total_virtual_tokens = soft_prompts.size(1)
        
        # Concat soft prompts
        inputs_embeds = torch.cat([begin_embed, soft_prompts, end_embed, inputs_embeds], dim=1)
        
        # Construct custom position IDs
        # Rule: Begin -> 0 | Numerical Prompts -> 1 | End -> 2 | Text Tokens -> 3, 4, 5...
        pos_begin = torch.tensor([0], device=device)
        pos_numerical = torch.full((total_virtual_tokens,), 1, device=device)
        pos_end = torch.tensor([2], device=device)
        pos_text = torch.arange(3, 3 + seq_len, device=device)
        position_ids = torch.cat([pos_begin, pos_numerical, pos_end, pos_text]).unsqueeze(0)
        
        # Concat attention mask with prefix
        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, total_virtual_tokens+2).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        # Concat labels with prefix
        if labels is not None:
            prefix_labels = torch.full((batch_size, total_virtual_tokens+2), -100).to(labels.device) # prefix the labels with -100 (ignore index)
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
    
    def generate(
        self, 
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        numeric_features: torch.Tensor = None,
        **kwargs,
    ):
        # Generate soft prompts
        batch_size, seq_len = input_ids.shape
        input_embeds = self.word_embeddings(input_ids)
        soft_prompts = self.prompt_encoder(numeric_features)
        total_virtual_tokens = soft_prompts.size(1)
        
        # Concat position ids
        soft_prompt_positions = torch.zeros(total_virtual_tokens, device=inputs_embeds.device)
        input_positions = torch.arange(1, inputs_embeds.shape[1]+1, device=inputs_embeds.device)
        position_ids = torch.cat((soft_prompt_positions, input_positions), dim=0)
        position_ids = position_ids.unsqueeze(0)
        
        # Concat soft prompts
        inputs_embeds = torch.cat((soft_prompts, input_embeds), dim=1)
        
        # Concat attention mask with prefix
        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, total_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        # Generate using the base_model
        outputs = self.base_model.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            **kwargs,
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
