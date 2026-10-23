from dataclasses import asdict

import torch
from lightning.pytorch import LightningModule
from transformers import get_scheduler


class BaseClassificationModule(LightningModule):
    """
    Base PyTorch Lightning module to train a model for binary classification..
    Handles shared operations for training, validation, and checkpointing.
    
    Args:
        config: The configuration object containing hyperparameters.
        model: The model to be trained.
        tokenizer: The tokenizer used for text processing.
        num_training_samples: The number of training samples in the dataset. Used for scheduler calculation.
    """
    def __init__(
        self,
        config,
        model,
        tokenizer=None,
        num_training_samples=None,
        **kwargs,
    ):
        super().__init__()
        # Save the passed hyperparameters
        hparams = asdict(config)
        hparams.update(kwargs) # Save additional hyperparameters passed via kwargs
        hparams["num_training_samples"] = num_training_samples
        for k, v in hparams.items(): # Convert dtypes into string
            if isinstance(v, torch.dtype):
                hparams[k] = str(v)
        self.save_hyperparameters(hparams, ignore=["model", "tokenizer"])
        
        # Model settings
        self.model = model
        self.tokenizer = tokenizer
        
        # Enable automatic optimization
        self.automatic_optimization = True
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.forward(
            numeric_features=batch.get("numeric_features"),
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )
        
        # Loss is computed inside the model when labels are provided (default is cross entropy loss)
        self.log("total_loss", outputs["loss"], prog_bar=True, on_step=True, on_epoch=False)
        
        return outputs["loss"]
    
    def validation_step(self, batch, batch_idx):        
        # Forward pass (batch size is 1 during validation for stability)
        outputs = self.forward(
            numeric_features=batch.get("numeric_features"),
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )
        
        # Logging (during evaluation, normally save by epoch rather than step)
        self.log("total_loss_val", outputs["loss"], prog_bar=True, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        # Set the log directory for saving predictions
        self.log_dir = self.trainer.log_dir
        # Optimizer settings
        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.hparams.lr, weight_decay=0.001)
        # Learning rate scheduler settings
        num_training_steps = self.hparams.num_training_samples * self.hparams.max_epochs // self.hparams.batch_size // self.hparams.accumulate_grad_batches
        num_warmup_steps = 0
        lr_scheduler = get_scheduler(
            name=self.hparams.scheduler_name, 
            optimizer=optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_save_checkpoint(self, checkpoint):
        # Dynamically retrieve the names of all trainable parameters
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        
        # Filter the state_dict to keep only trainable weights and remove the 'model.' prefix
        checkpoint["state_dict"] = {
            (k[len("model."):] if k.startswith("model.") else k): v 
            for k, v in checkpoint["state_dict"].items() 
            if k in trainable_keys
        }
    
    def on_load_checkpoint(self, checkpoint):
        # Manually load the state_dict into self.model
        state_dict = checkpoint["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)


class LMHeadConstrainModule(BaseClassificationModule):
    """
    Binary classification via constrained decoding on the LM head.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Specific model settings
        self.choice_ids = self.tokenizer(["Good", "Bad"], add_special_tokens=False).input_ids
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # Forward pass (batch size is 1 during testing for stability)
        outputs = self.forward(
            numeric_features=batch.get("numeric_features"),
            input_ids=batch.get("prompts"),
            attention_mask=batch.get("prompt_attention_mask"),
        )
        
        # Constrained decoding on choice_ids
        logits = outputs.logits[:, -1, self.choice_ids].squeeze(-1) # get next token logits, shape: (batch_size, 2)
        y_proba = torch.softmax(logits, dim=-1)[:, 1] # probability of "Bad", shape: (batch_size,)
        y_pred = (y_proba >= 0.5).int() # use threshold instead of torch.argmax(logits).item()
        
        # Return for the callback to save predictions and times
        return {
            "indices": batch["indices"],
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist(),
            "logits": logits.tolist(),
        }


class LinearClassifierModule(BaseClassificationModule):
    """
    Binary classification via a classification head.
    """
    
    def test_step(self, batch, batch_idx):
        # Forward pass (batch size is 1 during testing for stability)
        outputs = self.forward(
            numeric_features=batch.get("numeric_features"),
        )
        
        # Extract probabilities for the positive class (index 1) and apply a 0.5 threshold for binary predictions
        logits = outputs.logits # shape: (batch_size, 2)
        y_proba = torch.softmax(logits, dim=-1)[:, 1] # probability of "Bad", shape: (batch_size,)
        y_pred = (y_proba >= 0.5).int() # use threshold instead of torch.argmax(logits).item(), they are equivalent
        
        # Return for the callback to save predictions and times
        return {
            "indices": batch.get("indices"),
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist(),
            "logits": logits.tolist(),
        }
