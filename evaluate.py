import os
import torch
import torch.nn.functional as F
from config import *
from resnet_model import LungCancerResNet
from train import get_data_loaders
from utils.metrics import calculate_metrics, get_confusion_matrix
from utils.visualization import plot_confusion_matrix

def evaluate_model():
    """
    Evaluates the trained model on the validation dataset.
    """
    print("Initializing evaluation...")
    
    # Load validation data
    _, val_loader = get_data_loaders()
    if val_loader is None:
        print("Cannot evaluate: Data loaders not available.")
        return
        
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model weights not found at {MODEL_SAVE_PATH}")
        print("Please train the model first.")
        return
        
    # Load Model
    model = LungCancerResNet(num_classes=NUM_CLASSES, freeze_backbone=True)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print(f"Loaded model from {MODEL_SAVE_PATH}")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running evaluation on validation set...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.cpu().numpy()
            
            outputs = model(inputs)
            
            # Apply softmax to get probabilities
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            # Assuming Malignant is class 1
            all_probs.extend(probs[:, 1])
            
    # Calculate Metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Confusion Matrix
    cm = get_confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(cm, CLASS_NAMES, OUTPUTS_DIR)
    print(f"Saved confusion matrix plot to {OUTPUTS_DIR}/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
