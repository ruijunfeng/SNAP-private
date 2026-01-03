from dataclasses import dataclass, field

question_template = """You are an expert credit risk assessment model.
Your task is to determine whether this user should be classified as good credit or bad credit.
A good credit means the user is likely to repay reliably.
A bad credit means the user is high-risk and likely to default.

Given the following user features:
{profile}

Is this user's credit good or bad?
"""

answer_template = "The user credit is likely to be:\n"

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
        default=1e-5,
        metadata={"help": "The learning rate for the optimizer."},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "The batch size for training."},
    )
    max_epochs: int = field(
        default=1,
        metadata={"help": "The number of training epochs."},
    )
    scheduler_name: str = field(
        default="linear",
        metadata={"help": "The learning rate scheduler to use."},
    )
