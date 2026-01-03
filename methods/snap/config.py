from dataclasses import dataclass, field
from methods.base.config import CLSConfig

@dataclass
class SNAPConfig(CLSConfig):
    use_numerical_embedding: bool = field(
        default=True,
        metadata={"help": "Whether to use numerical embeddings in the prompt encoder."},
    )
    use_multi_head_self_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use multi-head self-attention in the prompt encoder."}
    )
    num_features: int = field(
        default=23,
        metadata={"help": "The number of features in the dataset used for numerical embeddings."},
    )