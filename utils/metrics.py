import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def calculate_metrics(y_true, y_pred, y_probs=None):
    """
    Calculate essential evaluation metrics for binary classification.
    
    Args:
        y_true (list or np.array): True labels (0: Benign, 1: Malignant).
        y_pred (list or np.array): Predicted labels.
        y_probs (list or np.array, optional): Predicted probabilities for the positive class (Malignant).
        
    Returns:
        dict: A dictionary containing Accuracy, Sensitivity, Specificity, F1, and ROC-AUC (if probabilities are provided).
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Sensitivity (Recall for the positive class: Malignant = 1)
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Specificity (Recall for the negative class: Benign = 0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1 Score": f1
    }
    
    if y_probs is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_probs)
            metrics["ROC-AUC"] = roc_auc
        except ValueError:
            # Handles case where only one class is present in y_true
            metrics["ROC-AUC"] = float('nan')
            
    return metrics

def get_confusion_matrix(y_true, y_pred):
    """
    Returns the confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)
