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

from methods.base.config import LanguageModelTrainingConfig
from methods import snap, ft_transformer

from data_module import HELOCDataModule, PAKDDDataModule, GMSCDataModule, HMEQDataModule
from module import LMHeadConstrainModule, LinearClassifierModule
from callback import ResultCheckpoint

# Set the precision for matmul
torch.set_float32_matmul_precision("high")
print("number of gpus", torch.cuda.device_count())
print("number of cpus", os.cpu_count())


def setup_language_model(config):
    """
    Load a pre-trained causal language model, freeze its parameters,
    and attach LoRA adapters for parameter-efficient fine-tuning.
    """
    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side="left",
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
    
    config.lora_config = lora_config
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    return config, tokenizer, model


def setup_tabular_model_pl_module(
    config,
    experiment_specification,
    num_training_samples,
    **kwargs,
):
    """
    Set up a Lightning module for tabular deep learning models (e.g., FT-Transformer).
    """
    model = experiment_specification["model_cls"](config=config)
    module = experiment_specification["pl_module_cls"](
        model=model,
        config=config,
        num_training_samples=num_training_samples,
    )
    return module, None


def setup_language_model_pl_module(
    config,
    experiment_specification,
    num_training_samples,
    **kwargs,
):
    """
    Set up a Lightning module for causal language model fine-tuning (e.g., CALM).
    """
    config, tokenizer, model = setup_language_model(config)
    module = experiment_specification["pl_module_cls"](
        model=model,
        tokenizer=tokenizer,
        config=config,
        num_training_samples=num_training_samples,
    )
    return module, tokenizer


def setup_snap_pl_module(
    config,
    experiment_specification,
    num_training_samples,
    args,
    **kwargs,
):
    """
    Set up a Lightning module for SNAP, handling ablation configurations
    and projector dimension alignments.
    """
    config, tokenizer, peft_base_model = setup_language_model(config)
    
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
        config.hidden_dim = peft_base_model.get_input_embeddings().weight.shape[1]
        config.head_dim = int(peft_base_model.get_input_embeddings().weight.shape[1] / num_heads)
        args.experiment_name = "snap/without_projector"
    elif args.disable_language_model:
        config.use_language_model = False
        args.experiment_name = "snap/without_language_model"
    else:
        args.experiment_name = "snap/full_model"
    
    # Set up the model
    model = experiment_specification["model_cls"](
        config=config,
        base_model=peft_base_model,
    )
    
    # Initialize the SFT module
    module = experiment_specification["pl_module_cls"](
        model=model,
        tokenizer=tokenizer,
        config=config,
        num_training_samples=num_training_samples,
    )
    return module, tokenizer


DATAMODULE_REGISTRY = {
    "heloc": HELOCDataModule,
    "pakdd": PAKDDDataModule,
    "gmsc": GMSCDataModule,
    "hmeq": HMEQDataModule,
}

EXPERIMENT_REGISTRY = {
    "snap": {
        "config_cls": snap.SNAPConfig,
        "model_cls": snap.SNAP,
        "pl_module_cls": LMHeadConstrainModule,
        "setup_func": setup_snap_pl_module,
        "data_config": {
            "train": {"task_types": "completion"},
            "val": {"task_types": "completion"},
            "test": {"task_types": "prompt"},
        },
    },
    "calm": {
        "config_cls": LanguageModelTrainingConfig,
        "model_cls": None,
        "pl_module_cls": LMHeadConstrainModule,
        "setup_func": setup_language_model_pl_module,
        "data_config": {
            "train": {"task_types": "completion"},
            "val": {"task_types": "completion"},
            "test": {"task_types": "prompt"},
        },
    },
    "ft_transformer": {
        "config_cls": ft_transformer.FTTransformerConfig,
        "model_cls": ft_transformer.FTTransformer,
        "pl_module_cls": LinearClassifierModule,
        "setup_func": setup_tabular_model_pl_module,
        "data_config": {
            "train": {"task_types": "cls"},
            "val": {"task_types": "cls"},
            "test": {"task_types": "cls"},
        },
    },
}

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run different sets of experiments")
    
    parser.add_argument(
        "--experiment_name", type=str,
        choices=list(EXPERIMENT_REGISTRY.keys()),
        default="snap",
        help="The experiment to run.",
    )
    parser.add_argument(
        "--dataset_name", type=str,
        choices=["heloc", "pakdd"],
        default="pakdd",
        help="The dataset to use.",
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
    parser.add_argument(
        "--disable_language_model", action="store_true",
        help="Whether to disable language model (snap).",
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Load the configuration based on the experiment name
    experiment_specification = EXPERIMENT_REGISTRY[args.experiment_name]
    config = experiment_specification["config_cls"]()
    
    # Prepare the data module
    data_module = DATAMODULE_REGISTRY[args.dataset_name]()
    config.num_features = data_module.num_features
    data_config = experiment_specification["data_config"]
    if args.experiment_name == "snap" and args.disable_language_model:
        data_config = {
            "train": {"task_types": "cls"},
            "val": {"task_types": "cls"},
            "test": {"task_types": "cls"},
        }
        experiment_specification["pl_module_cls"] = LinearClassifierModule
    
    # Initialize the model and module polymorphically based on setup function
    torch.manual_seed(233)
    module, tokenizer = experiment_specification["setup_func"](
        config=config,
        experiment_specification=experiment_specification,
        num_training_samples=len(data_module.train_indices),
        args=args,
    )
    
    # Initialize the trainer
    save_dir = f"results/{args.dataset_name}"
    result_checkpoint = ResultCheckpoint()
    model_checkpoint = ModelCheckpoint(
        filename="model-{epoch:02d}-{total_loss_val:.3f}",  # name of the save_top_k model
        monitor="total_loss_val",  # metric to monitor
        mode="min",  # minimize the metric
        save_top_k=1,  # save the top k models with filename.ckpt
        save_last=False,  # save the last model with last.ckpt
    )
    logger = TensorBoardLogger(
        save_dir=save_dir,  # directory to save logs
        name=args.experiment_name  # subdirectory to save versions
    )
    trainer = Trainer(
        precision="bf16-mixed",  # no need to manually convert to half precision in training_step
        accelerator="auto",
        devices=1,  # disable DDP
        max_epochs=config.max_epochs,
        logger=logger,
        callbacks=[TQDMProgressBar(refresh_rate=1), model_checkpoint, result_checkpoint],
        enable_model_summary=False,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )
    
    # Start training
    train_dataloader = data_module.get_dataloader(
        task_type=data_config["train"]["task_types"],
        indices=data_module.train_indices,
        tokenizer=tokenizer,
        question_template=getattr(config, "question_template", None),
        answer_template=getattr(config, "answer_template", None),
        batch_size=config.batch_size,
        shuffle=False,
    )
    val_dataloader = data_module.get_dataloader(
        task_type=data_config["val"]["task_types"],
        indices=data_module.val_indices,
        tokenizer=tokenizer,
        question_template=getattr(config, "question_template", None),
        answer_template=getattr(config, "answer_template", None),
        batch_size=config.batch_size,
        shuffle=False,
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
        precision="bf16-mixed",  # no need to manually convert to half precision in training_step
        accelerator="auto",
        devices=1,  # disable DDP
        logger=False,  # no logger for testing
        callbacks=[result_checkpoint],  # save predictions for testing
        enable_checkpointing=False,  # no checkpointing for testing
    )
    val_test_dataloader = data_module.get_dataloader(
        task_type=data_config["test"]["task_types"],
        indices=data_module.val_indices + data_module.test_indices,
        tokenizer=tokenizer,
        question_template=getattr(config, "question_template", None),
        answer_template=getattr(config, "answer_template", None),
        batch_size=1,
        shuffle=False,
    )
    trainer.test(
        model=module,
        dataloaders=val_test_dataloader,
    )
