import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from methods.base.config import CLSConfig
from methods.snap.config import SNAPConfig
from methods.snap.model import SNAP

from data_module import HelocDataModule
from module import SFTModule
from callback import ResultCheckpoint

# Set the precision for matmul
torch.set_float32_matmul_precision("high")
print("number of gpus", torch.cuda.device_count())
print("number of cpus", os.cpu_count())

CONFIG_MAPPINGS = {
    "snap": SNAPConfig,
    "calm": CLSConfig,
}

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run different sets of experiments")
    
    parser.add_argument(
        "--experiment_name", type=str,
        choices=list(CONFIG_MAPPINGS.keys()),
        default="snap",
        help="The experiment to run.",
    )
    parser.add_argument(
        "--disable_numerical_embedding", action="store_true",
        help="Whether to disable numerical embedding module in the prompt encoder (snap).",
    )
    parser.add_argument(
        "--disable_numerical_profiling", action="store_true",
        help="Whether to disable numerical profiling module in the prompt encoder (snap).",
    )
    parser.add_argument(
        "--disable_projector", action="store_true",
        help="Whether to disable projector in the prompt encoder (snap).",
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Load the configuration based on the experiment name
    config = CONFIG_MAPPINGS[args.experiment_name]()
    
    # Prepare the data module
    data_module = HelocDataModule()
    
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side="left", # left padding for next-token prediction
        local_files_only=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    # Freeze the base model
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Prepare the LoRA configuration
    lora_config = LoraConfig(
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules="all-linear",
    )
    
    # Initialize the model based on the experiment name
    torch.manual_seed(233)
    if args.experiment_name == "calm":
        config.lora_config = lora_config
        model = get_peft_model(base_model, lora_config)
        model.gradient_checkpointing_enable()
    elif args.experiment_name == "snap":
        # Add config.pad_token_id for later use in when generation position_ids
        config.pad_token_id = tokenizer.pad_token_id
        
        # Customize SNAP specific configurations for ablation study
        # Only one setting can be disabled at a time
        if args.disable_numerical_embedding:
            config.use_numerical_embedding = False
            args.experiment_name = "snap/without_numerical_embedding"
        elif args.disable_numerical_profiling:
            config.use_numerical_profiling = False
            args.experiment_name = "snap/without_numerical_profiling"
        elif args.disable_projector:
            config.use_projector = False
            num_heads = int(config.hidden_dim / config.head_dim)
            config.hidden_dim = base_model.get_input_embeddings().weight.shape[1]
            config.head_dim = int(base_model.get_input_embeddings().weight.shape[1] / num_heads)
            args.experiment_name = "snap/without_projector"
        else:
            args.experiment_name = "snap/full_model"
        
        # Set up the model
        config.lora_config = lora_config
        base_model = get_peft_model(base_model, lora_config)
        base_model.gradient_checkpointing_enable()
        model = SNAP(
            config=config, 
            base_model=base_model, 
        )
    
    # Initialize the SFT module
    module = SFTModule(
        model=model, 
        tokenizer=tokenizer,
        config=config,
        num_training_samples=len(data_module.train_indices),
    )
    
    # Initialize the trainer
    save_dir = "results"
    result_checkpoint = ResultCheckpoint()
    model_checkpoint = ModelCheckpoint(
        filename="model-{epoch:02d}-{total_loss_val:.3f}", # name of the save_top_k model
        monitor="total_loss_val", # metric to monitor
        mode="min", # minimize the metric
        save_top_k=1, # save the top k models with filename.ckpt
        save_last=True, # save the last model with last.ckpt
    )
    logger = TensorBoardLogger(
        save_dir=save_dir, # directory to save logs
        name=args.experiment_name # subdirectory to save versions
    )
    trainer = Trainer(
        precision="bf16-mixed", # no need to manually convert to half precision in training_step
        accelerator="auto",
        devices=1, # disable DDP
        max_epochs=config.max_epochs,
        logger=logger,
        callbacks=[TQDMProgressBar(refresh_rate=1), model_checkpoint, result_checkpoint],
        enable_model_summary=False,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )
    
    # Start training
    train_dataloader = data_module.get_completion_dataloader(
        indices=data_module.train_indices,
        tokenizer=tokenizer,
        question_template=config.question_template,
        answer_template=config.answer_template,
        batch_size=config.batch_size,
    )
    val_dataloader = data_module.get_completion_dataloader(
        indices=data_module.val_indices,
        tokenizer=tokenizer,
        question_template=config.question_template,
        answer_template=config.answer_template,
        batch_size=config.batch_size,
    )
    trainer.fit(
        model=module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # Load the best model for evaluation
    checkpoint = torch.load(model_checkpoint.best_model_path, weights_only=False)
    module.on_load_checkpoint(checkpoint)
    module.model.eval()
    
    # Start testing
    # Create new trainer to avoid redundant event file
    trainer = Trainer(
        precision="bf16-mixed", # no need to manually convert to half precision in training_step
        accelerator="auto",
        devices=1, # disable DDP
        logger=False, # no logger for testing
        callbacks=[result_checkpoint], # save predictions for testing
        enable_checkpointing=False, # no checkpointing for testing
    )
    val_test_dataloader = data_module.get_prompt_dataloader(
        indices=data_module.val_indices+data_module.test_indices,
        tokenizer=tokenizer,
        question_template=config.question_template,
        answer_template=config.answer_template,
        batch_size=1,
    )
    trainer.test(
        model=module,
        dataloaders=val_test_dataloader,
    )
