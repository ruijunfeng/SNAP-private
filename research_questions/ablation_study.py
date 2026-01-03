import os
import torch
import pandas as pd

from utils.json_utils import load_json
from evaluators.metric import calculate_classification_metrics

from data_module import HelocDataModule

# prepare the results for analysis
data_module = HelocDataModule()
y_true = torch.tensor([item["labels"] for item in data_module.get_profile_dataset(data_module.val_indices)])

# save the results of different ablation settings
results = []

# w/o SNAP
snap_results = load_json("results/calm/version_0/predictions.json")
y_pred = torch.tensor([int(snap_results[str(i)]["y_proba"] >=0.5) for i in data_module.val_indices])
y_proba = torch.tensor([snap_results[str(i)]["y_proba"] for i in data_module.val_indices])
result = calculate_classification_metrics(y_true, y_pred, y_proba)
results.append(("w/o SNAP", result))

# SNAP w/o numerical embeddings
snap_results = load_json("results/snap/without_numerical_embedding/version_0/predictions.json")
y_pred = torch.tensor([int(snap_results[str(i)]["y_proba"] >=0.5) for i in data_module.val_indices])
y_proba = torch.tensor([snap_results[str(i)]["y_proba"] for i in data_module.val_indices])
result = calculate_classification_metrics(y_true, y_pred, y_proba)
results.append(("SNAP w/o numerical embeddings", result))

# SNAP w/o multi-head self-attention
snap_results = load_json("results/snap/without_multi_head_self_attn/version_0/predictions.json")
y_pred = torch.tensor([int(snap_results[str(i)]["y_proba"] >=0.5) for i in data_module.val_indices])
y_proba = torch.tensor([snap_results[str(i)]["y_proba"] for i in data_module.val_indices])
result = calculate_classification_metrics(y_true, y_pred, y_proba)
results.append(("SNAP w/o multi-head self-attention", result))

# SNAP w/o numerical projecto
snap_results = load_json("results/snap/without_numerical_projector/version_0/predictions.json")
y_pred = torch.tensor([int(snap_results[str(i)]["y_proba"] >=0.5) for i in data_module.val_indices])
y_proba = torch.tensor([snap_results[str(i)]["y_proba"] for i in data_module.val_indices])
result = calculate_classification_metrics(y_true, y_pred, y_proba)
results.append(("SNAP w/o numerical projector", result))

# SNAP
snap_results = load_json("results/snap/full_model/version_0/predictions.json")
y_pred = torch.tensor([int(snap_results[str(i)]["y_proba"] >=0.5) for i in data_module.val_indices])
y_proba = torch.tensor([snap_results[str(i)]["y_proba"] for i in data_module.val_indices])
result = calculate_classification_metrics(y_true, y_pred, y_proba)
results.append(("SNAP", result))

# convert into dataframe
df = pd.DataFrame([r for _, r in results], index=[name for name, _ in results])

# save as csv
output_dir = "results/summary"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f"{output_dir}/ablation_study.csv")
