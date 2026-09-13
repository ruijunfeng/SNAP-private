import os
import torch
import pandas as pd

from utils.json_utils import load_json
from evaluators.metric import calculate_classification_metrics

from data_module import HELOCDataModule, PAKDDDataModule


def get_probabilities(predictions: dict, indices: list) -> torch.Tensor:
    """Helper function to extract prediction probabilities for a given set of indices."""
    return torch.tensor([predictions[str(i)]["y_proba"] for i in indices])


def get_predictions(predictions: dict, indices: list) -> torch.Tensor:
    """Helper function to generate binary predictions based on a threshold."""
    return torch.tensor([int(predictions[str(i)]["y_pred"]) for i in indices])


def run_evaluation_pipeline(models_map: dict, data_module, output_path: str):
    """
    Core evaluation pipeline that loads predictions, evaluates performance
    on the validation set, and saves metrics to a CSV.
    """
    # 1. Extract ground truth labels for validation dataset
    y_true = torch.tensor([item["labels"] for item in data_module.get_profile_dataset(data_module.val_indices)])
    
    results = []
    
    # 2. Process each model defined in the map
    for model_name, file_path in models_map.items():
        preds = load_json(file_path)
        
        # Generate predictions on validation set
        y_proba = get_probabilities(preds, data_module.val_indices)
        y_pred = get_predictions(preds, data_module.val_indices)
        
        # Calculate classification metrics
        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
        results.append((model_name, metrics))
    
    # 3. Convert results to DataFrame and export to CSV
    df = pd.DataFrame(
        [metrics for _, metrics in results],
        index=[name for name, _ in results]
    )
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Successfully saved results to: {output_path}")


if __name__ == "__main__":
    # Initialize shared configuration and data module
    heloc_data_module = HELOCDataModule()
    pakdd_data_module = PAKDDDataModule()
    
    # Define HELOC ablation models
    heloc_models_map = {
        "w/o SNAP": "results/heloc/calm/version_0/predictions.json",
        "SNAP w/o numerical embedding": "results/heloc/snap/without_numerical_embedding/version_0/predictions.json",
        "SNAP w/o numerical profiling": "results/heloc/snap/without_numerical_profiling/version_0/predictions.json",
        "SNAP w/o projector": "results/heloc/snap/without_projector/version_0/predictions.json",
        "SNAP w/o language model": "results/heloc/snap/without_language_model/version_0/predictions.json",
        "SNAP": "results/heloc/snap/full_model/version_0/predictions.json",
    }
    
    # Define PAKDD ablation models
    pakdd_models_map = {
        "w/o SNAP": "results/pakdd/calm/version_0/predictions.json",
        "SNAP w/o numerical embedding": "results/pakdd/snap/without_numerical_embedding/version_0/predictions.json",
        "SNAP w/o numerical profiling": "results/pakdd/snap/without_numerical_profiling/version_0/predictions.json",
        "SNAP w/o projector": "results/pakdd/snap/without_projector/version_0/predictions.json",
        "SNAP w/o language model": "results/pakdd/snap/without_language_model/version_0/predictions.json",
        "SNAP": "results/pakdd/snap/full_model/version_0/predictions.json",
    }
    
    # Run evaluation pipeline for HELOC
    run_evaluation_pipeline(
        models_map=heloc_models_map,
        data_module=heloc_data_module,
        output_path="results/heloc/summary/ablation_study.csv"
    )
    
    # Run evaluation pipeline for PAKDD
    run_evaluation_pipeline(
        models_map=pakdd_models_map,
        data_module=pakdd_data_module,
        output_path="results/pakdd/summary/ablation_study.csv"
    )
