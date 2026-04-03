import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, average_precision_score

def calculate_classification_metrics(
    y_true: torch.Tensor, 
    y_pred: torch.Tensor, 
    y_proba: torch.Tensor,
):
    """Calculate classification metrics including Accuracy, Area Under Curve (AUC), Precision, Recall, F1-score, and Kolmogorov–Smirnov (KS).
    
    Args: 
        y_true (torch.Tensor): Ground truth labels of shape (num_samples,). 
        y_pred (torch.Tensor): Predicted labels of shape (num_samples,). 
        y_proba (torch.Tensor): Predicted probabilities of shape (num_samples,). 
    
    Returns: 
        result (dict): A dictionary containing calculated metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    ks = max(tpr - fpr)
    gini = 2 * roc_auc - 1
    result = {
        "ROC-AUC": roc_auc,
        "Gini": gini,
        "KS": ks,
        "PR-AUC": pr_auc,
        "Accuracy": acc,
        "Precision_weighted": report["weighted avg"]["precision"],
        "Recall_weighted": report["weighted avg"]["recall"],
        "F1_weighted": report["weighted avg"]["f1-score"],
    }
    
    return result
