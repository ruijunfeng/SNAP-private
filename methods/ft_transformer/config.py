from dataclasses import dataclass, field
from methods.base.config import TrainingConfig


@dataclass
class FTTransformerConfig(TrainingConfig):
    # Override default settings in TrainingConfig
    max_epochs: int = field(
        default=50,
        metadata={"help": "The number of training epochs."},
    )
    batch_size: int = field(
        default=96,
        metadata={"help": "The batch size for training."},
    )
    accumulate_grad_batches: int = field(
        default=1,
        metadata={"help": "The number of batches to accumulate gradients over."},
    )
    # FT-Transformer specific settings
    num_features: int = field(
        default=23,
        metadata={"help": "The number of features in the dataset used for numerical embeddings."},
    )
    hidden_dim: int = field(
        default=192,
        metadata={"help": "The dimension of numerical embeddings (token representations)."},
    )
    head_dim: int = field(
        default=24,
        metadata={"help": "The dimension of each attention head in the multi-head self-attention."},
    )
    ffn_ratio: float = field(
        default=2,
        metadata={"help": "Expansion ratio of the hidden dimension within the feed-forward networks."},
    )
    ffn_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate applied within the feed-forward networks."},
    )
    attention_bias: bool = field(
        default=False,
        metadata={"help": "Whether to include bias terms in query/key/value projections."},
    )
    attention_dropout: float = field(
        default=0.2,
        metadata={"help": "Dropout rate applied to the attention maps."},
    )
    num_layers: int = field(
        default=3,
        metadata={"help": "The total number of Transformer blocks in the backbone."},
    )
