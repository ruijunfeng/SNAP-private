from dataclasses import dataclass, field
from methods.base.config import CLSConfig

@dataclass
class SNAPConfig(CLSConfig):
    use_numerical_embedding: bool = field(
        default=True,
        metadata={"help": "Whether to use numerical embedding module in the prompt encoder."},
    )
    use_numerical_profiling: bool = field(
        default=True,
        metadata={"help": "Whether to use numerical profiling module in the prompt encoder."}
    )
    use_projector: bool = field(
        default=True,
        metadata={"help": "Whether to use projector in the prompt encoder."}
    )
    num_features: int = field(
        default=23,
        metadata={"help": "The number of features in the dataset used for numerical embeddings."},
    )
    hidden_dim: int = field(
        default=768,
        metadata={"help": "The dimension of numerical embedddings."},
    )
    head_dim: int = field(
        default=96,
        metadata={"help": "The dimension of each attention head in the multi-head self-attention."},
    )
    mlp_ratio: int = field(
        default=2,
        metadata={"help": "Ratio of the MLP hidden dimension applied within the MLP (feed-forward) layers."},
    )
    mlp_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate applied within the MLP (feed-forward) layers."},
    )
    attention_bias: bool = field(
        default=False,
        metadata={"help": "Whether to include bias terms in the projection layer of the multi-head self-attention."},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate for attention layers in the multi-head self-attention."},
    )
    num_layers: int = field(
        default=6,
        metadata={"help": "The total number of layers (transformer blocks) in the model."},
    )
    projector_ratio: int = field(
        default=2,
        metadata={"help": "Ratio of the projector hidden dimension applied within the projector layers."},
    )
