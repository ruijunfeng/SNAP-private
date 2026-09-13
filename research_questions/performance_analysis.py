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
    """Helper function to extract prediction labels for a given set of indices."""
    return torch.tensor([predictions[str(i)]["y_pred"] for i in indices])


def run_evaluation_pipeline(models_map: dict, data_module, output_path: str):
    """
    Core evaluation pipeline that loads predictions, evaluates performance
    on the test set, and saves metrics to a CSV.
    """
    # 1. Extract ground truth labels for test dataset
    y_true = torch.tensor([item["labels"] for item in data_module.get_profile_dataset(data_module.test_indices)])
    
    results = []
    
    # 2. Process each model defined in the map
    for model_name, file_path in models_map.items():
        preds = load_json(file_path)
        
        # Generate predictions on test set
        y_pred = get_predictions(preds, data_module.test_indices)
        y_proba = get_probabilities(preds, data_module.test_indices)
        
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
    # Initialize data modules
    heloc_data_module = HELOCDataModule()
    pakdd_data_module = PAKDDDataModule()
    
    # Define HELOC models
    heloc_models_map = {
        "KNN": "results/heloc/machine_learning/KNN/predictions.json",
        "MLP": "results/heloc/machine_learning/MLP/predictions.json",
        "SVM": "results/heloc/machine_learning/SVM/predictions.json",
        "NB": "results/heloc/machine_learning/NB/predictions.json",
        "DT": "results/heloc/machine_learning/DT/predictions.json",
        "RF": "results/heloc/machine_learning/RF/predictions.json",
        "GBDT": "results/heloc/machine_learning/GBDT/predictions.json",
        "XGBoost": "results/heloc/machine_learning/XGBoost/predictions.json",
        "LightGBM": "results/heloc/machine_learning/LightGBM/predictions.json",
        "FT-Transformer": "results/heloc/ft_transformer/version_0/predictions.json",
        "Informed GPT": "results/heloc/informed_gpt/predictions.json",
        "CALM": "results/heloc/calm/version_0/predictions.json",
        "SNAP": "results/heloc/snap/full_model/version_0/predictions.json",
    }
    
    # Define PAKDD models
    pakdd_models_map = {
        "KNN": "results/pakdd/machine_learning/KNN/predictions.json",
        "MLP": "results/pakdd/machine_learning/MLP/predictions.json",
        "SVM": "results/pakdd/machine_learning/SVM/predictions.json",
        "NB": "results/pakdd/machine_learning/NB/predictions.json",
        "DT": "results/pakdd/machine_learning/DT/predictions.json",
        "RF": "results/pakdd/machine_learning/RF/predictions.json",
        "GBDT": "results/pakdd/machine_learning/GBDT/predictions.json",
        "XGBoost": "results/pakdd/machine_learning/XGBoost/predictions.json",
        "LightGBM": "results/pakdd/machine_learning/LightGBM/predictions.json",
        "FT-Transformer": "results/pakdd/ft_transformer/version_0/predictions.json",
        "Informed GPT": "results/pakdd/informed_gpt/predictions.json",
        "CALM": "results/pakdd/calm/version_0/predictions.json",
        "SNAP": "results/pakdd/snap/full_model/version_0/predictions.json",
    }
    
    # Run evaluation pipeline for HELOC
    run_evaluation_pipeline(
        models_map=heloc_models_map,
        data_module=heloc_data_module,
        output_path="results/heloc/summary/performance_analysis.csv"
    )
    
    # Run evaluation pipeline for PAKDD
    run_evaluation_pipeline(
        models_map=pakdd_models_map,
        data_module=pakdd_data_module,
        output_path="results/pakdd/summary/performance_analysis.csv"
    )
