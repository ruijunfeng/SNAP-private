import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_profile_embeddings(dataset):
    embeddings = []
    for item in tqdm(dataset):
        with torch.no_grad():
            # Tokenize the input texts
            batch_dict = tokenizer(
                item["profiles"],
                padding=False,
                truncation=False,
                return_tensors="pt",
            )
            batch_dict.to(model.device)
            outputs = model(**batch_dict)
            embeddings.append(last_token_pool(
                outputs.last_hidden_state, 
                batch_dict["attention_mask"])[0, :1024].tolist()
            )
    
    # Convert to numpy array
    return np.array(embeddings)


# Load the embedding model
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-Embedding-0.6B", 
    padding_side="left",
)
model = AutoModel.from_pretrained(
    "Qwen/Qwen3-Embedding-0.6B",
    dtype="auto",
    device_map="auto",
)

# Get the profile datasets
dataset_train = data_module.get_profile_dataset(data_module.train_indices)
dataset_val = data_module.get_profile_dataset(data_module.val_indices)
dataset_test = data_module.get_profile_dataset(data_module.test_indices)

# Get the profile embeddings
embeddings_train = get_profile_embeddings(dataset_train)
embeddings_val = get_profile_embeddings(dataset_val)
embeddings_test = get_profile_embeddings(dataset_test)

# Concat with numerical features
X_train = np.hstack((X_train, embeddings_train[:32]))
X_val = np.hstack((X_val, embeddings_val))
X_test = np.hstack((X_test, embeddings_test))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ("num_scaler", StandardScaler(), slice(0, 23))
    ],
    remainder="passthrough",
)
