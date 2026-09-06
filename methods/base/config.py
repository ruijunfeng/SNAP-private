from dataclasses import dataclass, field
from peft import LoraConfig

question_template = """You are an expert in credit scoring.
Your task is to analyze an applicant's historical financial and credit information and classify the applicant as either "Good" or "Bad".

Definitions:
- "Good": The applicant is low-risk and is likely to repay debt reliably.
- "Bad": The applicant is high-risk and is likely to default.

Given the following applicant credit history:
{profile}

Is this applicant's credit Good or Bad?
"""

answer_template = """The applicant's credit classification is:\n"""

@dataclass
class BaseConfig():
    model_name: str = field(
        default="Qwen/Qwen3-4B-Instruct-2507",
        metadata={"help": "The name of the model to be used for zero-shot evaluation."},
    )
    question_template: str = field(
        default=question_template,
        metadata={"help": "The template for the question prompt."},
    )
    answer_template: str = field(
        default=answer_template,
        metadata={"help": "The template for the answer prompt."},
    )

@dataclass
class CLSConfig(BaseConfig):
    lr: float = field(
        default=2e-4,
        metadata={"help": "The learning rate for the optimizer."},
    )
    batch_size: int = field(
        default=3,
        metadata={"help": "The batch size for training."},
    )
    accumulate_grad_batches: int = field(
        default=32,
        metadata={"help": "The number of batches to accumulate gradients over."},
    )
    max_epochs: int = field(
        default=10,
        metadata={"help": "The number of training epochs."},
    )
    scheduler_name: str = field(
        default="constant",
        metadata={"help": "The learning rate scheduler to use."},
    )
    lora_config: LoraConfig = field(
        default=None,
        metadata={"help": "The configuration for LoRA fine-tuning."},
    )
